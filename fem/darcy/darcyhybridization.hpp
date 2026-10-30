// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_DARCYHYBRIDIZATION
#define MFEM_DARCYHYBRIDIZATION

#include "../../config/config.hpp"
#include "../bilinearform.hpp"
#include "../nonlinearform.hpp"
#include "nlresidual_hdg.hpp"
#ifdef MFEM_USE_MPI
#include "../pbilinearform.hpp"
#include "../pnonlinearform.hpp"
#endif //MFEM_USE_MPI

#include <atomic>
#include <exception>
#include <functional>
#include <map>
#include <utility>

#define MFEM_DARCY_HYBRIDIZATION_ELIM_BCS

#include <vector>

namespace mfem
{

class HDGDiffusionIntegrator;

/** @brief Carries the first exception thrown inside an OpenMP parallel region
    out of it, so that it reaches the caller instead of std::terminate().

    An exception may not leave an OpenMP structured block; the runtime
    terminates the process. `#pragma omp parallel if (false)` still opens the
    block -- the clause chooses the team size, not whether the region exists --
    so this applies to every element loop in DarcyForm and DarcyHybridization
    that carries the pragma, at one thread and in AssemblyMode::Serial as much
    as when threaded. gffp measured it (2026-09-25): the finiteness guard in
    AssembleFluxMassMatrix(), a catchable ErrorException before those loops
    were threaded, became `terminate called after throwing` and a SIGABRT, and
    no configuration of the caller's could get the exception back.

    The pattern, with Rethrow() straight after the region closes:

        ThreadedExceptionTrap trap;
        #pragma omp parallel if (threaded)
        {
           #pragma omp for
           for (int i = 0; i < n; i++)
           {
              if (trap.Raised()) { continue; }
              try { ... }
              catch (...) { trap.Capture(); }
           }
        }
        trap.Rethrow();

    The FIRST exception is rethrown unchanged, type and message; later ones are
    dropped. Iterations not yet started when it is raised are skipped, while
    those already running on other threads finish, so the loop's outputs are
    as partial as a serial throw would have left them. Without OpenMP the
    pattern is an ordinary try/catch that defers the throw to the end of the
    loop. */
class ThreadedExceptionTrap
{
   std::atomic<bool> raised{false};
   std::exception_ptr first;

public:
   /// Whether an exception has been captured; a hint, read without ordering.
   bool Raised() const { return raised.load(std::memory_order_relaxed); }

   /// Record the exception being handled, if it is the first. Call only from
   /// inside a catch handler.
   void Capture()
   {
      bool expected = false;
      // Only the thread that wins the exchange writes @a first, and the
      // region's closing barrier orders that write before Rethrow() reads it.
      if (raised.compare_exchange_strong(expected, true))
      {
         first = std::current_exception();
      }
   }

   /// Rethrow the captured exception, if any. Call after the region has
   /// closed, on the thread that opened it.
   void Rethrow() const
   {
      if (first) { std::rethrow_exception(first); }
   }
};

/// Class for total flux hybridization of Darcy-like mixed systems
/** Class DarcyHybridization performs total flux hybridization of mixed systems
    with (anti)symmetric weak form common for parabolic and elliptic problems.
    They can be written as:
    \verbatim
        ┌        ┐┌   ┐   ┌    ┐
        | Mu ±Bᵀ || u | _ | bu |
        | B  Mp  || p | ̅  | bp |
        └        ┘└   ┘   └    ┘
    \endverbatim
    where @a u is the flux (continuous or discontinuous) and @a p is the
    potential (assumed always discontinuous). The bilinear forms @a Mu
    and @a Mp are the mass terms of the flux and potential respectively. The
    mixed bilinear form @a B is the divergence of flux (in a generalized sense)
    and @a bu and @a bp are the right-hand-side terms of the flux and potential
    respectively.

    The sign convention of the system is chosen in the constructor
    DarcyHybridization(). Given the set of the forms (Mu, B, Mp), either a
    symmetric system without a sign change (#bsym == false) or with a flipped
    sign (#bsym == true) is formed respectively:
    \verbatim
        ┌       ┐        ┌        ┐
        | Mu Bᵀ |        | Mu -Bᵀ |
        | B  Mp |   or   | -B -Mp |
        └       ┘        └        ┘
    \endverbatim

    The process of hybridization introduces an additional constraint equation
    mediating coupling between elements, meaning discontinuous, "broken" spaces
    can be used instead. The constraint enforces continuity of the total flux,
    which can have contributions from both, flux and potential parts. The full
    system then takes the form:
    \verbatim
        ┌           ┐┌   ┐   ┌    ┐
        | Mu ±Bᵀ Cᵀ || u |   | bu |
        | B   D  E  || p | = | bp |
        | C   G  H  || λ |   | br |
        └           ┘└   ┘   └    ┘
    \endverbatim
    where @a C is the constraint term with optional stabilization contributions
    in @a E, @a G, @a H and @a D. The new variable @a λ is the Lagrange
    multiplier approximating the trace of @a p. Note the best conditioning of
    the problem is achieved with @a λ taken from the trace space of @a u,
    but the generality of the construction allows different choices with
    sufficient stabilization. Also note the hybridized system is not
    necessarily equivalent to the original mixed formulation.

    An important advantage of the hybridized formulation is discontinuity of
    the spaces for the primary quantities, which enables to eliminate (often
    referred to as static condensation) the flux and potential equations
    by means of local inversion (i.e., the inverted matrix is block diagonal):
    \verbatim
                         ┌        ┐-1 ┌    ┐
                         | Mu ±Bᵀ |   | Cᵀ |
        H  ← H  - [ C G ]| B   D  |   | E  |
                         └        ┘   └    ┘
                         ┌        ┐-1 ┌    ┐
                         | Mu ±Bᵀ |   | bu |
        br ← br - [ C G ]| B   D  |   | bp |
                         └        ┘   └    ┘
        H λ = br
    \endverbatim
    This reduced linear system is equivalent to the full formulation and
    enables an economical solution procedure, where the original quantities
    can be recovered afterwards as follows:
    \verbatim
        ┌   ┐   ┌        ┐-1 /┌    ┐   ┌    ┐   \
        | u | _ | Mu ±Bᵀ |  | | bu |   | Cᵀ |    |
        | p | ̅  | B   D  |  | | bp | ̅  | E  | λ  |
        └   ┘   └        ┘   \└    ┘   └    ┘   /
    \endverbatim

    The first step of the hybridization process is assembly at the element/face
    level. It is initiated by a call of Init() followed by subsequent calls of
    Assemble*Matrix() methods and ComputeAndAssemblePot*FaceMatrix() for face
    integration of the potential constraint. The assembly process is finished
    by Finalize(), enabling to use Mult() or access the hybridized system
    matrix through GetMatrix() (or GetParallelMatrix() in parallel). The right
    hand side of the mixed system can be reduced through ReduceRHS(). After
    solution of the hybridized system, the original quantities of the system
    can be recovered through ComputeSolution().

    Some common configurations of finite element spaces are Raviart-Thomas
    elements for the fluxes and Lagrange elements for the potentials. This RTDG
    scheme does not require stabilization due to compatibility of the spaces
    and the constraint space can be naturally chosen as
    DG_Interface_FECollection, which coincides with the trace space of RT
    elements (up to the sign convention). The hybridized scheme is then
    equivalent to the original mixed formulation.

    However, continuity of the flux space does not allow stabilization of
    the trace for advection in the potential equation. Therefore, the flux
    space can be chosen as broken Raviart-Thomas (BrokenRT_FECollection), which
    is discontinuous and enables upwinding of the trace. For more details see:
    Egger, H., & Schoberl, J. (2009). A hybrid mixed discontinuous Galerkin
    finite-element method for convection-diffusion problems. IMA Journal of
    Numerical Analysis, 30(4), 1206–1234. https://doi.org/10.1093/imanum/drn083

    Generalizing the fluxes as discontinuous, a logical choice is using
    Lagrange elements for both quantities, which yields the well-known
    Hybridizable Discontinuous Galerkin (HDG) method. As the spaces are not
    mutually compatible (in inf-sup sense), the scheme requires stabilization
    of the trace unknown to converge to the actual trace of the potential and
    vice versa. This can be achieved through redefinition of the total flux
    with a forcing term like τ(p̂-λ), which naturally stabilizes the scheme
    ( @a τ is a coefficient and @a p̂ is trace of the potential @a p ). These
    contributions populate the terms @a E, @a G, @a H and @a D and require the
    potential constraint integrator to compute all these face matrices, which
    are collectively denoted as the HDG face matrix. Some common integrators of
    this type can be found in bilininteg_hdg.hpp. For more details about
    construction of HDG for convection-diffusion problems see:
    Nguyen, N. C., Peraire, J., & Cockburn, B. (2009). An implicit high-order
    hybridizable discontinuous Galerkin method for linear convection–diffusion
    equations. Journal of Computational Physics, 228(9), 3232–3254.
    https://doi.org/10.1016/j.jcp.2009.01.030

    A notable feature of HDG schemes is reconstruction of the total flux and
    superconvergent quantities in turn. The constraint equation, which enforces
    continuity of the total flux, is used to project the total flux on the
    face restriction of the total flux finite element (typically from
    Raviart-Thomas space). The interior DOFs are determined by integral
    projection of the flux function passed to ReconstructTotalFlux(). Apart
    from being useful on its own, the total flux can be used for reconstruction
    of the original quantites (flux and potential) with polynomial order higher
    by one, where it is used as a source term for every element in the mixed
    formulation. For more details, refer to section 4 of the cited paper.

    In so far, the trace space was considered only as DG_Interface_FECollection
    matching restriction of Raviart-Thomas to skeleton of the mesh. However,
    trace space of H1 elements can be used as well (H1_Trace_FECollection).
    This choice corresponds to the Embedded Discontinuous Galerkin (EDG) method
    known in the literature: Nguyen, N. C., Peraire, J., & Cockburn, B. (2015).
    A class of embedded discontinuous Galerkin methods for computational fluid
    dynamics. Journal of Computational Physics, 302, 674–692.
    https://doi.org/10.1016/j.jcp.2015.09.024. Such construction is more
    economical, sharing the nodal DOFs between adjacent faces, at the expense
    of local conservation properties and conditioning of the local problem.
 */
class DarcyHybridization : public Hybridization
{
public:
   enum class LSsolveType
   {
      LBFGS,
      LBB,
      Newton,
   };

   enum class LPrecType
   {
      GMRES,
      LU,
   };

   /** @brief How the loop over elements that builds the reduced system is
       executed.

       Static condensation is defined by each element's flux and potential
       being eliminable independently of every other, so the loop is parallel
       by construction rather than by accident. What is not independent is the
       result: a trace dof lives on a face and a face has two elements, so the
       trace matrix receives two contributions per entry.

       Both modes assemble the *same matrix, entry for entry and bit for bit*.
       That is a property worth having rather than a coincidence -- see
       SetAssemblyMode() for what it costs and why it was paid. */
   enum class AssemblyMode
   {
      /** @brief One thread, elements in order. The historical behaviour and
          the default: no existing caller pays anything for the other mode. */
      Serial,
      /** @brief The element-local loops run in parallel: ComputeH()'s, whose
          scatter into the trace matrix stays serial and in element order, and
          MultNL()'s -- the residual and the Jacobian assembly, and so
          NPCResidual() and NPCGradient() -- which is walked in colour order
          instead. Both agree with Serial bit for bit. Requires
          MFEM_USE_OPENMP and MFEM_THREAD_SAFE, and obliges the caller's own
          integrators to be thread-safe; see SetAssemblyMode(). */
      Threaded,
      /** @brief The interior-face potential term is assembled by one batched
          kernel that scatters straight into E, G, H and D, instead of one
          host call per face.

          A DEVICE mode: it needs the storage to be device-resident to be
          worth anything, and its D accumulation goes through AtomicAdd, which
          costs on a host where the per-face loop's plain += does not.

          **It falls back silently, and for most problems it does.** The
          kernel covers HDGDiffusionIntegrator with any of its coefficients,
          both HDGConvection*Integrators, and a SumIntegrator of them, on an
          interior face of a serial conforming mesh under NPC.
          CanBatchPotFaceAssembly() answers whether it was taken -- ask it,
          because this mode was unreachable for EVERY caller in the tree until
          DarcyForm's SumIntegrator wrapper was looked through, and no answer
          and no timing showed that. */
      Batched,
   };

   /** @brief How the element-local blocks are factored.

       The flux mass A and the potential block D are factored one element at a
       time, and each element's factorisation is independent of every other's
       -- that independence is what static condensation is. These modes differ
       only in how that loop is written, not in what it computes. */
   enum class LocalFactorMode
   {
      /** @brief One LUFactors per element, in element order. The historical
          behaviour and the default: no existing caller pays anything for the
          other mode. */
      Serial,
      /** @brief The whole array in one BatchedLinAlg::LUFactor() call, which
          makes the device path a backend selection rather than new kernels --
          see BatchedLinAlg::SetActiveBackend(). Requires every element's
          block to be the same size; CanBatchLocalFactor() answers that in
          advance, and the loop above is taken when it is false. */
      Batched,
   };

   /** @brief How the element blocks reach the global trace matrix.

       The third of ComputeH(): the element-local face-pair blocks exist, and
       they have to become a CSR matrix. This is a separate axis from
       LocalFactorMode, which decides how those blocks are *computed* -- the
       two are independent settings and were measured separately. */
   enum class TraceAssemblyMode
   {
      /** @brief ScatterElementH() per element into an unfinalized
          SparseMatrix, then Finalize(). The historical behaviour and the
          default: no existing caller pays anything for the other mode. */
      Serial,
      /** @brief The CSR is built from the mesh connectivity ONCE and refilled
          by a kernel per linearisation, instead of a linked-list matrix
          rebuilt from scratch each time.

          A DEVICE mode, and the one that closes the chain -- the sparse third
          of ComputeH() was the last stage with no device path at all.
          CanBatchTraceAssembly() answers whether it was taken.

          **It does not give the same matrix OBJECT as Serial, and that is a
          real difference rather than a caveat.** The two agree on the pattern
          and on every value to the bit (see CanBatchTraceAssembly()), but a
          row's columns come out in a different ORDER, so SparseMatrix::Mult()
          reassociates and the trace solve differs in its last bits. Serial's
          order cannot be reproduced: it is the reverse of the order in which
          entries were first inserted, and AddSubMatrix(skip_zeros) declines
          to insert an element's exact zeros -- so which element first touches
          a column decides that column's position, and the order is a
          function of the VALUES. Measured, not assumed; see
          BuildTraceHMap(). */
      Batched,
   };

   /** @brief Whether the reduced (trace) gradient is assembled as a sparse
       matrix or only applied.

       Both build the same operator and both need the local blocks factored
       once per linearisation -- that is what condensation is, and no setting
       avoids it. They differ in whether the global trace matrix is formed.

       Assembling costs one local back-substitution per trace dof of the
       element, so it is worth roughly that many matrix-free applications: six
       for k = 1 triangles, near a hundred for k = 3 hexes. Against that it
       buys a matrix a direct solver or an algebraic preconditioner can use, so
       it wins outright at low order. MatrixFree carries no global matrix at
       all, does no sparse gather or scatter, and does identical work on every
       element; the case for it is memory and device fitness, and the cost of
       it is that GS, AMG and a direct factorisation are no longer available to
       precondition the trace solve. */
   enum class GradientMode
   {
      /// Assemble the Schur complement; GetGradient() returns a SparseMatrix.
      Assembled,
      /// Apply it; GetGradient() returns an Operator with no stored matrix.
      MatrixFree,
   };

protected:
   FiniteElementSpace &fes_p;       ///< potential FE space
#ifdef MFEM_USE_MPI
   ParFiniteElementSpace *pfes;     ///< parallel flux FE space
   ParFiniteElementSpace *pfes_p;   ///< parallel potential FE space
   ParFiniteElementSpace *c_pfes;   ///< parallel constraint FE space
#endif
   std::unique_ptr<BilinearFormIntegrator> c_bfi_p;      ///< constraint integrator
   std::unique_ptr<NonlinearFormIntegrator> c_nlfi_p;
   std::unique_ptr<BlockNonlinearFormIntegrator> c_nlfi;
   NonlinearFormIntegrator *m_nlfi_u{};
   NonlinearFormIntegrator *m_nlfi_p{};
   bool own_m_nlfi_u{};
   bool own_m_nlfi_p{};
   BlockNonlinearFormIntegrator *m_nlfi{};
   bool own_m_nlfi{};

   /// The potential constraint boundary face integrators
   std::vector<BilinearFormIntegrator*> boundary_constraint_pot_integs;
   /// Boundary markers for potential constraint face integrators
   std::vector<Array<int>*> boundary_constraint_pot_integs_marker;
   std::vector<NonlinearFormIntegrator*> boundary_constraint_pot_nonlin_integs;
   std::vector<Array<int>*> boundary_constraint_pot_nonlin_integs_marker;
   std::vector<BlockNonlinearFormIntegrator*> boundary_constraint_nonlin_integs;
   std::vector<Array<int>*> boundary_constraint_nonlin_integs_marker;
   /// Indicates if the boundary_constraint_pot_integs integrators are owned externally
   bool extern_bdr_constr_pot_integs{false};

   bool bsym{};      ///< sign convention, see DarcyReduction()
   bool bfin{};      ///< indicates finalized hybridization
   bool bnpc{};      ///< NPC requested on a form that may be linear
   /** @brief A load assembled on the SKELETON, in L-dofs of @a c_fes, or null.

       DarcyForm owns it -- see DarcyForm::GetTraceRHS() -- and registers it
       here through SetTraceRHS() so that BOTH routes carry it without the
       caller wiring anything: ReduceRHS() adds @f$P^T b_\lambda@f$ to the
       reduced right-hand side, and NPCResidual() subtracts it from the trace
       block, which is the same convention read off
       @f$r = A x - b@f$. Borrowed, not owned. */
   const Vector *trace_rhs{};
   DiagonalPolicy diag_policy{DIAG_ONE};  ///< diagonal policy
   /** @brief Essential *trace* true DOFs, in the constraint space @a c_fes.

       Not flux dofs, which is what this said for long enough to mislead an
       outside user into writing a substitute for the accessor below. The flux
       ones are Init()'s @a ess_flux_tdof_list and are not retained here.
       Every setter -- SetEssentialBC(), SetEssentialVDofs(),
       SetEssentialTrueDofs() -- fills this from @a c_fes, and these are the
       rows the diagonal policy pins in the reduced operator: Mult() zeroes the
       residual on them and GetGradient() leaves a unit row. */
   Array<int> ess_tdof_list;

   /** @brief Per-face polynomial degree of the trace, or empty for uniform.

       Set by SetTraceOrders(). Empty means a uniform trace and every accessor
       then falls through to the constraint space itself, returning the same
       pointers and the same nulls, so a caller that never sets a degree pays
       nothing and gets byte-identical results. See SetTraceOrders() for what
       the entries mean. */
   Array<int> tr_order;

   /** @brief The essential trace true DOFs the *caller* asked for, in the
       CONSTRAINT SPACE's numbering.

       That is the numbering a caller can name, being the space's own;
       #ess_tdof_list is the same set in the constrained numbering, which
       depends on the degrees. Finalize() rebuilds one from the other through
       MapEssentialTraceDofs(), so a second Finalize() after Reset() with
       different degrees does not inherit the first one's map. The two
       coincide while the trace is uniform. */
   Array<int> ess_tdof_user;

   /** @brief The constrained trace: the numbering, the two face maps, and the
       prolongation composed out of them.

       Built lazily by BuildTraceConstraint() and empty while the trace is
       uniform, so a caller that never sets a per-face degree allocates
       nothing and every accessor falls through to the constraint space's
       own. Cleared by SetTraceOrders() and by Init(); nothing else can
       change what they depend on, which is #tr_order and the space. */
   mutable Array<int> ctr_offsets;
   /// ctdof -> constraint-space true DOF, block diagonal over faces.
   mutable std::unique_ptr<SparseMatrix> ctr_PE;
   /// ctdof -> constraint-space VDOF, i.e. cP composed with #ctr_PE. Serial.
   mutable std::unique_ptr<SparseMatrix> ctr_Pv;
   /// E and R, keyed by (face geometry, degree) rather than by face.
   mutable std::map<std::pair<int,int>,DenseMatrix> ctr_E, ctr_R;
   /// Constraint-space VDOF -> its true DOF, or -1 when it has none here.
   mutable Array<int> ctr_vdof2tdof;
   mutable int ctr_ntdof{0};
   mutable bool ctr_built{false};
#ifdef MFEM_USE_MPI
   /// ctdof -> constraint-space LDOF, i.e. Dof_TrueDof composed with #ctr_PE.
   mutable std::unique_ptr<HypreParMatrix> ctr_pP;
   mutable Array<HYPRE_BigInt> ctr_col_starts;
#endif

private:
   struct
   {
      LSsolveType type;
      int iters;
      real_t rtol;
      real_t atol;
      int print_lvl;
      struct
      {
         LPrecType type;
         int iters;
         real_t rtol;
         real_t atol;
      } prec;
   } lsolve;

   Array<int> Ae_offsets;
   /** @brief Still an Array, and deliberately: it is copy-assigned from and
       Swap()ped with Hybridization::Af_data, which is the UPSTREAM base
       class's. The two have to change type together, so both are left for a
       separate pass over fem/hybridization.hpp. */
   Array<real_t> Af_lin_data;
   Vector Ae_data;
   bool A_empty{true};

   Array<int> Bf_offsets, Be_offsets;
   Vector Bf_data, Be_data;

   /** @brief The solution-dependent part of the local gradient's (0,1) block,
       d(flux residual)/dp.

       For a flux law q = D(p) u the flux equation depends on the potential, so
       the (0,1) block of the local Jacobian is not simply the transpose of the
       linear divergence form: it is that plus the derivative the flux function
       supplies as J_u. @a Bf_data holds the linear part and is assembled once;
       this holds the part that changes with the solution and is rebuilt at
       every Newton step.

       Indexed by @a Bf_offsets, which gives each element the same number of
       entries, but each block is read as a_dofs by d_dofs -- the transpose
       orientation of @a Bf_data -- because that is the shape of a (0,1) block.

       Left empty whenever no integrator contributes such a term, which covers
       every linear problem and every nonlinear one whose coefficients do not
       depend on the potential; @a Bnl_empty then short-circuits the extra
       work. */
   mutable Vector Bnl_data;
   mutable bool Bnl_empty{true};

   /** @brief The JACOBIAN's (1,0) block: the linear divergence form plus
       whatever a block integrator's `grad_arr(1,0)` supplied.

       The mirror of @a Bnl_data, and the doc above states the asymmetry from
       the other side. A term evaluated at the POSTPROCESSED potential -- the
       interpolatory HDG of Chen, Cockburn, Singler & Zhang -- makes the
       potential equation depend on the flux, because `u*` is reconstructed
       from it, and the divergence form alone is then not the whole (1,0)
       block. Every other problem this branch has met leaves it empty, and
       @a Bg_empty short-circuits the extra work.

       @a Bf_data's own (d_dofs, a_dofs) orientation and @a Bf_offsets
       indexing.

       @warning **This is the (1,0) block only.** @a Bf_data is ALSO read
       transposed as the (0,1) block -- LocalNLOperator holds one view and
       uses it both ways, with TransposeOperator Bt aliasing it -- and that
       role already carries @a Bnl_data. Adding this addend to the shared view
       would apply it twice over in the wrong block, so the transposed uses
       deliberately keep reading @a Bf_data. */
   mutable Vector Bg_data;
   mutable bool Bg_empty{true};
   /* Seeded with @a Bf_data and NOT with zero when first allocated, which is
      the one thing that differs from @a Bnl_data's handling and is not
      optional. @a Bnl_data is an ADDEND, so an element whose block was never
      written contributes nothing; this array REPLACES B at its readers, and
      @a Bg_empty goes false as soon as the FIRST element writes, which
      exposes every element the loop has not reached yet. Zero-filled, those
      read B = 0, the Schur complement loses its divergence form, and the
      first Newton step comes back inf. A replacing cache and an adding cache
      have different initialisation obligations. */

   /** @brief The (1,0) block of element @a el: the Jacobian's when @a gradient
       and @a Bg_data is live, and the linear divergence form otherwise.

       One funnel, so a site that forgets the distinction cannot exist. @a B is
       reset as a non-owning view; nothing is copied. */
   void GetGradBMatrix(int el, bool gradient, DenseMatrix &B) const;

   /** @brief Load the (0,1) gradient block of element @a el into @a Bnl.

       Returns false, leaving @a Bnl untouched, when there is no such block. */
   bool GetBnlMatrix(int el, DenseMatrix &Bnl) const;

   Array<int> Df_offsets, Df_f_offsets;
   mutable Vector Df_data, Df_lin_data;
   mutable Array<int> Df_ipiv;
   /** @brief The corrections AddNextGradientCorrection() has queued for the
       next NPCGradient(): element numbers, and each element's block as a
       column-major copy. Mutable because ReducedGradient() is const and is
       where they are consumed. */
   mutable Array<int> grad_corr_el;
   mutable std::vector<Vector> grad_corr_val;
   /** @brief The factored Schur complement, for LocalOpType::FluxNL only.

       Everywhere else the Schur complement is built into @a Df_data, over the
       potential block, because nothing else needs that storage. FluxNL is the
       exception: the potential mass is LINEAR there, so @a Df_data holds its
       factorisation and LocalFluxNLOperator::SolveP() needs it on every local
       nonlinear iteration. The Schur complement used to be built into a
       function-local temporary and thrown away, which is why the matrix-free
       gradient was refused in that mode and why anything reading a Schur
       complement back out of @a Df_data got the potential mass instead.

       Allocated only when the mode calls for it. MultInv() reads these
       whenever it is applying the JACOBIAN's blocks, which is exactly its
       @a with_bnl argument. */
   mutable Vector Sf_data;
   mutable Array<int> Sf_ipiv;
   bool D_empty{true};

   Array<int> Ct_offsets;
   Vector Ct_data;

   mutable Array<int> E_offsets;
   mutable Vector E_data;

   Array<int> &G_offsets{E_offsets};
   mutable Vector G_data;

   mutable Array<int> H_offsets;
   mutable Vector H_data;

   /** @brief The frozen half of E, G and H when a LIVE face constraint sits
       beside a linear one, i.e. what @a c_bfi_p assembled once at Finalize().

       The same device Df_lin_data is for D, and for the same reason: the
       gradient's face pass REWRITES E and G (see AssembleHDGGrad()'s
       @a eg_written) and zeroes H, so without a copy the frozen half would
       survive exactly one gradient. ConstructGrad() seeds E and G from these
       and lets the existing accumulate flag do the rest; ReducedGradient()
       seeds H in place of zeroing it.

       Empty unless BOTH slots are filled, which is the only configuration
       that needs them -- with c_bfi_p alone nothing rewrites the blocks, and
       with c_nlfi_p alone there is nothing frozen to keep. */
   mutable Vector E_lin_data, G_lin_data, H_lin_data;

   mutable Array<int> darcy_offsets, darcy_toffsets;
   mutable BlockVector darcy_rhs;
   Vector darcy_u, darcy_p;
   mutable Array<int> f_2_b;
   /** @brief Boundary attributes whose TRACE row is a boundary condition, so
       the flux constraint C is left out of it. Empty when none.

       See SetTraceBCAttributes(). One marker rather than a per-face array: the
       test is two lookups through @a f_2_b and it is made once per element
       face per assembly, beside products that are orders of magnitude dearer.
       An array of bool would also be a second member on a class whose layout
       this branch has paid to rebuild eight times. */
   Array<int> bc_trace_marker;
   /** @brief An element colouring, and the elements ordered by it. Built once,
       lazily, and only when AssemblyMode::Threaded is asked for.

       Two elements of one colour share no face -- Mesh::GetElementColoring()
       colours the element-to-element graph, whose edges ARE the faces -- so
       within a colour the trace dofs written by different elements are
       disjoint. That is what makes MultNL()'s two shared writes safe without
       atomics: the scatter of the trace row into @a y, and H_f's accumulation
       in AssembleHDGGrad(), which both sides of a face add into.

       E and G need no such protection and are left alone: they are stored per
       (face, SIDE), so the two elements of a face write different halves. */
   mutable Array<int> colour_order, colour_offsets;
   /** @brief Every element's flux and potential dofs, concatenated in element
       order, so that the whole gather or scatter between an L-vector and the
       element-blocked layout is ONE Vector::GetSubVector()/SetSubVector().

       Those overloads are mfem::forall kernels (linalg/vector.cpp:676, :740),
       so with the maps and the blocked vector marked UseDevice() the gather
       runs where the local blocks already live. The per-element loop they
       replace could not: it goes through the `real_t *elem_data` overloads,
       which begin with HostRead()/HostReadWrite() and are host loops by
       construction.

       Laid out to match @a Af_f_offsets and @a Df_f_offsets exactly -- the
       flux half is GetFDofs(el) for each element in turn, the potential half
       fes_p.GetElementVDofs(el) -- so element @a el's slice is at
       Af_f_offsets[el] / Df_f_offsets[el], which is where the blocked vectors
       put it. Entries carry MFEM's signed-dof convention and the kernels
       honour it, so an H(div) flux gathers correctly.

       Built once, lazily, by BuildElementDofMaps(). Both are pure functions of
       the spaces and of @a hat_dofs_marker, which Init() fixes and nothing
       afterwards changes: ConstructC() only re-marks free dofs from 0 to -1,
       and GetFDofs() selects on != 1. */
   mutable Array<int> el_u_dofs, el_p_dofs;

   /** @brief (NE*nf*nc) the trace vdof of each (element, local face, local
       trace dof) -- the third of the element-blocked dof maps, and the one the
       batched linear residual gathers x with. Built by
       BuildElementDofMaps() when the mesh gives one face count per element
       and one trace size per face, and left EMPTY otherwise, which is what
       CanBatchLinearResidual() tests. */
   mutable Array<int> el_c_dofs;

   /** @brief The connectivity TraceAssemblyMode::Batched assembles against,
       defined in the source file.

       An opaque pointer and not a handful of Array members, deliberately: a
       new member of this class changes its layout, and every translation unit
       that includes mfem.hpp sees that layout, so it is a `make clean` in
       every tree. Behind one pointer the map's contents can grow without
       costing that again. */
   struct TraceHMap;
   mutable std::unique_ptr<TraceHMap> trace_h_map;

   /** @brief The face blocks LinearResidualBatched() gathers, kept between
       evaluations; defined in the source file.

       An opaque pointer for the reason trace_h_map is one -- a new member
       changes the class layout and every TU that includes mfem.hpp sees it --
       and because this cache is exactly the sort of thing that grows. */
   struct ResidualCache;
   mutable std::unique_ptr<ResidualCache> res_cache;

   /** @brief The connectivity NPCReduce() and NPCRecover() read when their
       face loops run as kernels; see CanBatchNPCTraversal(). Defined in the
       source file.

       Kept APART from res_cache and not folded into it, because the two have
       different lifetimes and different preconditions: res_cache holds face
       BLOCKS frozen at their first value, which is right for the linear
       residual and wrong here -- under FaceConstraintMode::Live the E and G
       this traversal reads are rewritten at every gradient -- while this
       holds only offsets and a boundary mask, which a coefficient change
       cannot move. Dropped by Reset() and by SetTraceBCAttributes(). */
   struct NPCTraversalMap;
   mutable std::unique_ptr<NPCTraversalMap> npc_map;

   /** @brief The state-INDEPENDENT half of the local condensation, kept
       across a Newton loop instead of rebuilt at every gradient.

       Under LocalOpType::PotNL only the potential mass is non-linear, so A
       and its factorisation, B and the trace constraint blocks are all fixed
       for the whole solve: `A^-1 B^T`, `B A^-1 B^T`, `A^-1 C^T`,
       `B A^-1 C^T - E`, `C A^-1 B^T + G` and `C A^-1 C^T` do not move. D is
       the only block that does, and the Schur complement moves only through
       it. meq measures ComputeH() at 68-73% of their gradient leg, so this is
       the leg inside the leg.

       Dropped by InvalidateCoefficientCache() and by Reset(); a caller whose
       coefficients move must say so, exactly as for res_cache.

       **Measured**, ComputeH() seconds per gradient, serial, medians of
       three, against the same tree with the cache refused:

       | | off | on | |
       |---|---|---|---|
       | order 1, 32x32 quads | 0.00612 | 0.00455 | 1.34x |
       | order 2, 24x24 quads | 0.01091 | 0.00591 | **1.85x** |
       | order 2, 24x24 tris  | 0.00960 | 0.00553 | 1.74x |
       | order 3, 16x16 quads | 0.01242 | 0.00557 | **2.23x** |

       It grows with order because the cached products are the ones whose cost
       is quadratic or cubic in the block sizes. Against meq's M-80, where
       ComputeH() is 68-73% of a gradient leg that is a third of a solve, 1.85x
       is about 11% of a whole solve at their k=2 -- below the 17-18% a flop
       count predicted, because the remainder is memory-bound rather than
       arithmetic-bound.

       **It is not free in memory**, and the caller should know the size:
       `na*nd + nf*nc*(na + 2*nd) + nf^2*nc^2` reals per element. That is
       5.9 kB an element for 2-D order-2 quads, so ~97 MB on a 128x128 mesh.
       A caller who is memory-bound rather than gradient-bound should not want
       this, which is why the predicate is narrow rather than "whenever it is
       correct". */
   struct CondensationCache;
   mutable std::unique_ptr<CondensationCache> cond_cache;

   GradientMode grad_mode{GradientMode::Assembled};
   AssemblyMode asm_mode{AssemblyMode::Serial};
   LocalFactorMode lfac_mode{LocalFactorMode::Serial};
   TraceAssemblyMode tasm_mode{TraceAssemblyMode::Serial};
   bool integ_thread_safe{false};

   mutable long num_local_nl_iters{0};


   std::unique_ptr<SparseMatrix> He;
   OperatorHandle pHe;
   mutable std::unique_ptr<SparseMatrix> Grad;
   mutable OperatorHandle pGrad;

   /** @brief Trace dofs whose row in the assembled reduced gradient would be
       empty, and which the diagonal policy therefore regularises.

       Rebuilt by GetGradient() in GradientMode::MatrixFree, where there is no
       matrix for SetDiagIdentity() to act on. A row is empty when every face
       carrying the dof contributes nothing through C, G or H, which is what
       happens to a boundary trace dof of a problem whose constraint has no
       boundary face term. Without this the two modes are different operators
       -- the matrix-free one singular exactly where the assembled one was
       regularised to be nonsingular. */
   mutable Array<int> mf_diag_marker;
   /// Fill @a mf_diag_marker; see it.
   void MarkEmptyTraceRows() const;

   friend class Gradient;
   /// The reduced gradient applied rather than assembled; see GradientMode.
   class Gradient : public Operator
   {
      const DarcyHybridization &dh;
   public:
      Gradient(const DarcyHybridization &dh)
         : Operator(dh.Width()), dh(dh) { }

      void Mult(const Vector &x, Vector &y) const override;
   };

#ifdef MFEM_USE_MPI
   friend class ParOperator;
   class ParOperator : public Operator
   {
      const DarcyHybridization &dh;
      mutable OperatorHandle pGrad;
   public:
      /* dh.Width(), which Finalize() has already set to
         GetTraceTrueVSize(). That is c_fes.GetTrueVSize() while the trace is
         uniform and the constrained size once it is not, and this read the
         former directly -- so under a per-face trace the operator announced
         a size the vectors it is handed do not have. */
      ParOperator(const DarcyHybridization &dh)
         : Operator(dh.Width()), dh(dh) { }

      void Mult(const Vector &x, Vector &y) const override;
      Operator& GetGradient(const Vector &x) const override;
   };
   mutable OperatorHandle pOp;

   class ParGradient : public Operator
   {
      const DarcyHybridization &dh;
   public:
      ParGradient(const DarcyHybridization &dh)
         : Operator(dh.Width()), dh(dh) { }   // see ParOperator

      void Mult(const Vector &x, Vector &y) const override;
   };
#endif //MFEM_USE_MPI

   enum class LocalOpType { FluxNL, PotNL, FullNL };
   LocalOpType lop_type{LocalOpType::FullNL};

   /// Defined below, after the members it is scratch for; see there.
   struct TransWorkspace;

   /** @brief The scratch ComputeElementsHBatched() works in, hoisted out of
       the chunk loop by the caller.

       It is a struct and an argument rather than six locals because the
       allocation is not free and was not small: at one chunk of 256 elements
       and order 2 the buffers come to about 1.5 MB, every one of them above
       glibc's mmap threshold, so making them locals asked the kernel for
       fresh pages and faulted them in on first touch once per chunk. Measured
       -- that alone was 0.36 s of a 0.99 s face-pair loop at n=128.

       Vector::SetSize() does not shrink the allocation, so sizing for the
       first chunk sizes for all of them and the last, short chunk reuses what
       is there. */
   struct ElementHWorkspace
   {
      /// The element's trace blocks side by side, (na, T).
      Vector Ct;
      /// A^-1 Ct, same shape.
      Vector AiCt;
      /// E and then G; the two are the same size and their lives do not
      /// overlap.
      Vector EG;
      /// S^-1 (B A^-1 Ct - E), (nd, T).
      Vector BAiCt;
      /// C A^-1 B^T + G, (T, nd).
      Vector CAiBt;
      /// The element matrix before it is packed into blocks, (T, T).
      Vector Hfull;
   };

   friend class LocalNLOperator;
   class LocalNLOperator : public Operator
   {
      /// See the constructor's parameter of the same name.
      bool skip_int_faces{false};

   protected:
      const DarcyHybridization &dh;
      int el;
      const BlockVector &trps;
      const Array<int> &faces;

      const int a_dofs_size, d_dofs_size;
      DenseMatrix B;
      TransposeOperator Bt;
      /** @brief The (1,0) block for GetGradient() only.

          @a B stays the linear divergence form because Mult() is a RESIDUAL:
          the reaction's own contribution arrives separately through
          AddMultBlock(), so adding it to B there would count it twice. And
          @a Bt aliases @a B, so the (0,1) block must keep reading the linear
          one too. Only the Jacobian's (1,0) entry takes the addend. */
      DenseMatrix Bg;
      const FiniteElement *fe_u, *fe_p;
      IsoparametricTransformation *Tr;
      /** The caller's per-thread scratch. Every transformation this operator
          uses lives in it, so the operator allocates nothing per element --
          it used to allocate nine objects and two pointer vectors. The face
          transformations are read straight out of @a ws.lop_faces; there is no
          indirection vector because every entry would be `&ws.lop_faces[f]`. */
      TransWorkspace &ws;
      /** @brief This element's slice of the block integrator's flux row,
          precomputed for the whole mesh, or NULL to call the integrator.

          See DarcyHybridization::CanBatchLocalResidual(). Not owned. A
          member here costs nothing outside darcyhybridization.cpp -- this is
          a nested class and no translation unit constructs or contains one --
          which is exactly why the same value is a PARAMETER on
          DarcyHybridization's own methods. */
      const Vector *elem_flux_row;
      /** Every one of these is a reference into @a ws, so the operator
          allocates NOTHING per element. They were members, and a member of
          an object built once per element per evaluation is a fresh
          allocation per element however carefully the caller hoists. `mutable`
          is gone with them: a const method may write through a
          reference-to-non-const, which is what these now are. */
      const Array<int> &offsets;
      Vector &Au, &Dp, &DpEx;
      DenseMatrix &grad_A, &grad_D;
      /** The (0,1) block and, when it is nonzero, the dense sum of it with
          the linear +/-B^T that would otherwise stand there alone. */
      DenseMatrix &grad_Aup, &grad_Bt;

      /** @brief The gradient's block structure, built on FIRST USE.

          **The residual path never calls GetGradient()**, and that is the
          whole point of this being a method rather than a member:
          LocalResidual() -- every NPC residual evaluation -- constructs a
          LocalNLOperator per element and only ever calls Mult(), so a
          BlockOperator built in the constructor was 3,072 malloc/free pairs
          on a 256-element problem for an object that path does not touch.
          Neither specialised operator touches it either; both return
          grad_A / grad_D directly as their own gradient.

          Rebuilt only when the block structure changes, which on a uniform
          mesh is once for the mesh. BlockOperator COPIES its offsets, so it
          cannot be re-pointed at a new element's -- but GetGradient() re-sets
          all four block pointers on every call, so a reused one can never be
          read with a stale block. */
      BlockOperator &Grad() const;

      void AddMultBlock(const Vector &u_l, const Vector &p_l, Vector &bu,
                        Vector &bp) const;
      void AddMultA(const Vector &u_l, Vector &bu) const;
      void AddMultDE(const Vector &p_l, Vector &bp) const;
      /// Accumulates into grad_A / grad_D / grad_Aup; see the definition for
      /// why it no longer takes the two it used to shadow.
      void AddGradBlock(const Vector &u_l, const Vector &p_l) const;

      /** @brief Add the BLOCK integrator's (1,1) into @a grad, asking for no
          other block.

          AddGradBlock()'s counterpart for LocalOpType::PotNL, where the flux
          row must not be asked for at all: A there is the FACTORED linear
          flux mass and a well-behaved integrator handed a non-NULL (0,0)
          writes a zero matrix into it. Face contributions are not collected
          because the branch is not taken when c_nlfi is set. */
      void AddGradBlockPot(const Vector &u_l, const Vector &p_l,
                           DenseMatrix &grad) const;
      void AddGradA(const Vector &u_l, DenseMatrix &gA) const;
      void AddGradDE(const Vector &p_l, DenseMatrix &gD) const;

   public:
      /** @a ws supplies every transformation this operator uses; see
          TransWorkspace::lop_elem. Nothing here is owned, so there is no
          destructor. */
      /** @param skip_interior_faces leave the INTERIOR nonlinear face
          constraint out of the residual, because
          AssembleNLFaceResidualBatched() is doing it for every face at once.
          Exactly the role ConstructGrad()'s parameter of the same name plays
          for the gradient, and named the same so the two read as a pair. */
      LocalNLOperator(const DarcyHybridization &dh, int el, const BlockVector &trps,
                      const Array<int> &faces, TransWorkspace &ws,
                      const Vector *elem_flux_row = NULL,
                      bool skip_interior_faces = false);
      virtual ~LocalNLOperator() = default;

      inline const Array<int>& GetOffsets() const { return offsets; }

      void Mult(const Vector &x, Vector &y) const override;
      Operator &GetGradient(const Vector &x) const override;
   };

   class LocalFluxNLOperator : public LocalNLOperator
   {
      const Vector &bp;
      LUFactors LU_D;

      /// The recovered potential, from @a ws; see TransWorkspace::lop_other.
      Vector &p_l;

   public:
      LocalFluxNLOperator(const DarcyHybridization &dh, int el, const Vector &bp,
                          const BlockVector &trps, const Array<int> &faces,
                          TransWorkspace &ws);

      void SolveP(const Vector &u_l, Vector &p_l) const;
      void Mult(const Vector &x, Vector &y) const override;
      Operator &GetGradient(const Vector &x) const override;
   };

   class LocalPotNLOperator : public LocalNLOperator
   {
      const Vector &bu;
      LUFactors LU_A;

      /// The recovered flux, from @a ws; see TransWorkspace::lop_other.
      Vector &u_l;

   public:
      LocalPotNLOperator(const DarcyHybridization &dh, int el, const Vector &bu,
                         const BlockVector &trps, const Array<int> &faces,
                         TransWorkspace &ws);

      void SolveU(const Vector &p_l, Vector &u_l) const;
      void Mult(const Vector &x, Vector &y) const override;
      Operator &GetGradient(const Vector &x) const override;
   };

   class DenseMatrixLUSolver : public Solver
   {
      const DenseMatrix *mat;
      DenseMatrixInverse inv;
   public:
      DenseMatrixLUSolver() { }

      void SetOperator(const Operator &op) override
      {
         mat = dynamic_cast<const DenseMatrix*>(&op);
         MFEM_VERIFY(mat, "Not a DenseMatrix operator!");
         height = mat->Height();
         width = mat->Width();
         MFEM_ASSERT(height == width, "Not a square matrix!");
         inv.Factor(*mat);
      }

      void Mult(const Vector &x, Vector &y) const override
      {
         // DenseMatrixInverse::Mult() is LUFactors::Solve() on raw host
         // pointers, and NewtonSolver hands it a device-flagged correction;
         // see LocalNLOperator::Mult() for the whole of it.
         x.UseDevice(false);
         y.UseDevice(false);
         x.HostRead();
         y.HostWrite();
         inv.Mult(x, y);
      }
   };

   bool IsNonlinear() const { return c_nlfi || c_nlfi_p || m_nlfi || m_nlfi_u || m_nlfi_p; }
   /** @brief True when the element-local blocks and the element-wise H
       must be kept, which is what NPC reads and what a nonlinear form
       needs anyway. IsNonlinear() alone was the test, and it is the
       wrong one for NPC: a linear form needs exactly the same data. */
   bool NPCEnabled() const { return bnpc || IsNonlinear(); }
#ifdef MFEM_USE_MPI
   bool ParallelU() const { return pfes != NULL; }
   bool ParallelP() const { return pfes_p != NULL; }
   bool ParallelC() const { return c_pfes != NULL; }
#else
   bool ParallelU() const { return false; }
   bool ParallelP() const { return false; }
   bool ParallelC() const { return false; }
#endif

   void GetFDofs(int el, Array<int> &fdofs) const;
   void GetEDofs(int el, Array<int> &edofs) const;
   /** @brief Caller-allocated transformation storage for one thread of an
       element loop.

       Mesh keeps exactly ONE FaceElemTr, one Transformation and one
       Transformation2, and GetFaceElementTransformations(f) hands back a
       pointer into them -- so two threads in the same element loop overwrite
       each other's geometry, silently and with no wrong answer that looks like
       a race. Holding the objects here instead is the whole fix, and it needs
       no change to Mesh: every caller-allocated overload exists and is const
       (mesh.hpp, pmesh.hpp). LocalNLOperator already owns its transformations
       for this reason, which is why the local nonlinear solve was never the
       obstacle.

       This is a TYPE, not a member, so it adds nothing to the class layout --
       the trap that has cost this branch four rebuild cycles. It is passed as
       a parameter, which is what the standing note recommends. */
   struct TransWorkspace
   {
      IsoparametricTransformation elem;    ///< the element transformation
      FaceElementTransformations face;     ///< one face at a time
      IsoparametricTransformation f1, f2;  ///< that face's two side transforms

      /** @brief LocalNLOperator's storage: ALL of an element's faces at once.

          The four above serve a loop that visits one face at a time, which is
          what ConstructGrad() does. LocalNLOperator cannot use them: its
          AddMultDE() and AddMultBlock() iterate an element's faces with every
          transformation live, so it needs one per face simultaneously.

          It used to heap-allocate them -- one IsoparametricTransformation, and
          per face a FaceElementTransformations plus another
          IsoparametricTransformation for the neighbour -- in its constructor,
          which runs ONCE PER ELEMENT PER RESIDUAL EVALUATION. Measured on
          `convdiff -p 1 -o 2 -dg -hb -nl -npc -nls 3` at 128x128 with a direct
          trace solve, that constructor and destructor were 0.098 s of
          NPCResidual's 0.744 s -- 13%, and twice what the integrators inside
          the same routine cost.

          Held here they are allocated once for the whole element loop and
          reused: the vectors grow to the largest face count seen and never
          shrink. @a lop_elem is deliberately separate from @a elem so that a
          LocalNLOperator and a ConstructGrad() pass over the same element can
          never alias each other's geometry.

          **End to end it is worth about 1.5%, not the 3% the 13% above
          predicts, and the case for it is not the 1.5%.** Interleaved A/B over
          eight pairs on the case above, two binaries built from the same tree
          so no rebuild sits between the halves: minimum 3.804 s -> 3.750 s,
          median 3.928 s -> 3.865 s, faster in seven pairs of eight. What it
          buys that the number does not show is that the residual's element
          loop now allocates NOTHING per element, which is a precondition for
          threading or offloading it -- a malloc per element per evaluation is
          exactly what stops such a loop scaling.

          **A blocked measurement said the opposite and was wrong.** Building
          one version, timing five runs, rebuilding the other and timing five
          more reported 3.33 s -> 3.63 s, i.e. a 9% REGRESSION, because the two
          halves ran minutes apart on a machine whose load was still settling.
          Best-of-five does not defend against drift between the halves; only
          interleaving does. Where two variants differ by a couple of percent,
          build both binaries first and alternate the runs. */
      IsoparametricTransformation lop_elem;
      std::vector<FaceElementTransformations> lop_faces;
      std::vector<IsoparametricTransformation> lop_nbrs;

      /** @brief AssembleHDGGrad()'s two block temporaries.

          The integrator's output, and the sub-block CopyMN() lifts out of it
          on the way into D, E, G and H. Four fresh DenseMatrix objects per
          FACE per element per gradient before they were hoisted -- 6,144
          malloc/free pairs and 1.9 MB on a 256-element problem, measured with
          DHAT. The three destinations' lives do not overlap, so one @a g_blk
          serves all of them; DenseMatrix::SetSize() does not shrink, so it
          sizes once for the mesh. */
      DenseMatrix g_elmat, g_blk;

      /** @brief ConstructGrad()'s own per-element scratch.

          Six fresh objects per element per GRADIENT evaluation before they
          were hoisted, and the largest remaining site on the element loop
          after the LocalNLOperator and HDGDiffusionIntegrator rounds:
          measured with DHAT on `convdiff -p 6 -nl -o 2 -dg -hb -npc -nls 3`
          at 256 elements, 1,545 malloc/free pairs. The profile named
          VectorMassIntegrator and HyperbolicFormIntegrator, which own none of
          them -- SetSize() on a caller-supplied reference is called by the
          callee, so DHAT attributes a caller's fresh local to the integrator
          that fills it. Third time that reading has been needed on this loop.

          @a cg_fe_arr, @a cg_x_arr and @a cg_grad_arr were built from
          initialiser lists, which is a new[] per construction; the same
          shape LocalNLOperator's @a lop_fe_arr replaced.

          **The three matrices are cleared by their users before every call,
          and that is load-bearing rather than defensive.** ConstructGrad()
          tests `grad_A.Height() != 0` to mean "the integrator wrote this
          block", which was sound only while they were fresh locals; held
          here they carry the previous element's size in. SetSize(0, 0) keeps
          the buffer -- DenseMatrix::data is an Array, which shrinks its size
          and not its allocation -- so the clear is free and the hoist still
          holds. This is the trap AddMultBlock() paid for in the round
          before. */
      DenseMatrix cg_grad_A, cg_grad_D, cg_grad_Aup;
      Array<const FiniteElement*> cg_fe_arr;
      Array<const Vector*> cg_x_arr;
      Array2D<DenseMatrix*> cg_grad_arr;
      /// ConstructGrad()'s per-face "E and G have been written" flags. Sized
      /// from the element's face count, so SetSize() grows it to the largest
      /// and never shrinks; its user assigns `false` over it every element,
      /// which is what it needs and not merely a clear.
      Array<bool> cg_eg_written;

      /** @brief LocalResidual()'s two block vectors.

          Constructed per element per residual evaluation before they were
          hoisted -- 3,072 malloc/free pairs and 467 kB on a 256-element
          problem. Update() re-points them at the element's offsets and
          Vector::SetSize() does not shrink, so they size once for the mesh.
          @a lr_rv is separate from @a lr_xv because Mult() reads one and
          writes the other. */
      BlockVector lr_xv, lr_rv;

      /** @brief LocalNLOperator's per-element scratch, block structure and
          integrator argument arrays.

          A LocalNLOperator is constructed ONCE PER ELEMENT PER EVALUATION and
          every one of these used to be a member of it or a local in one of
          its methods, so each was a fresh object starting at capacity 0 and
          allocating on its first SetSize(). Measured with DHAT on
          `convdiff -p 6 -nl -o 2 -dg -hb -npc -nls 3` at 256 elements:
          3,072 + 768 malloc/free pairs in the constructor and 1,536 in
          Mult(), the largest remaining site on the element loop after the
          earlier hoists.

          Held here they are allocated once for the whole element loop, the
          same mechanism and the same reason as @a lop_elem above: SetSize()
          does not shrink, so the first element sizes them for the mesh.

          **@a lop_grad is built on FIRST USE and not by the constructor**,
          which is a separate finding rather than a hoist -- see
          LocalNLOperator::Grad(). One @a lop_other serves both specialised
          operators' recovered field because the switch in MultNL() builds
          exactly one of them. */
      Array<int> lop_offsets;
      Vector lop_Au, lop_Dp, lop_DpEx, lop_other;
      /** @brief The flux-row output handed to AddMultBlock() from
          LocalPotNLOperator::Mult(), which has no flux row of its own.

          A block integrator reaches that branch only by promising through
          GetBlockRowMask() that it writes the potential row alone, so this
          stays zero -- it exists because AssembleElementVector() takes an
          array of outputs and cannot be told to skip one. */
      Vector lop_bu_pot;
      DenseMatrix lop_grad_A, lop_grad_D, lop_grad_Aup, lop_grad_Bt;
      /// The integrators' own output blocks, added into the four above.
      DenseMatrix lop_gA, lop_gD, lop_gAup;
      /// The block integrator's argument arrays, built per call from
      /// initialiser lists before they were hoisted.
      Array<const FiniteElement*> lop_fe_arr;
      Array<const Vector*> lop_x_arr;
      Array<Vector*> lop_y_arr;
      Array2D<DenseMatrix*> lop_grad_arr;
      /// Mult()'s and GetGradient()'s two block views on the caller's
      /// vectors. Separate for the same reason @a lr_rv is.
      BlockVector lop_xv, lop_bv;
      /// LocalNLOperator::GetGradient()'s block structure; see Grad().
      std::unique_ptr<BlockOperator> lop_grad;

      /** @brief The nonconforming MASTER-face expansion's scratch.

          A coarse element's face list holds the MASTER face while the face
          term lives on the slave sub-faces, so every element-major nonlinear
          face loop visits the slaves in its place. Kept apart from @a face /
          @a f1 / @a f2 and from @a lop_faces because both of those hold the
          MASTER's own geometry live while these are being filled, and a slave
          must not overwrite it -- the same hazard @a lop_elem is kept separate
          from @a elem to avoid, one level down. */
      FaceElementTransformations nc_face;
      IsoparametricTransformation nc_f1, nc_f2;
      DenseMatrix nc_I;      ///< the master-to-slave trace transfer
      Vector nc_x, nc_y;     ///< `I lambda_m`, and the trace row carried back
      Array<int> nc_slaves;  ///< the slaves of the master being expanded
      DenseMatrix nc_eg;     ///< `E_s I` / `I^T G_s` on the way to the master
   };

   /// The shared Mesh cache. Valid only on a single-threaded path.
   /** @brief Build @a colour_order / @a colour_offsets if they are not built,
       and warm the lazily-built tables an element loop would otherwise race to
       fill. Cheap, once, and only on the threaded path. */
   void BuildElementColouring() const;

   /** @brief Build @a el_u_dofs / @a el_p_dofs if they are not built. */
   void BuildElementDofMaps() const;

   /** @brief Make the element-local block arrays readable on the host.

       The batched routes hand these arrays to BatchedLinAlg through
       Read()/ReadWrite(), whose default is on_dev = true, so with a Device
       configured they come back valid there -- and the debug backend
       mprotects the host page while that is so. Every other consumer is host
       code reaching them through raw pointers (Array::operator[],
       Vector::GetData()), which does not sync, so it has to be told.

       **Af_ipiv is the sharp one, and it is why this exists.** LUFactors uses
       its entries as ARRAY INDICES, so a host reader indexes on whatever is
       there; the fault lands inside LUFactors::Solve() naming no array; and
       Af_ipiv.GetMemory().HostIsValid() reports **1** the whole time, because
       the flag and the page protection are not the same thing. Guarding on
       the flag would therefore pass and crash anyway. Measured on the NPC
       route under Device("debug").

       That route is also what withdraws the note on InvertA() saying nothing
       reaches this. It reasoned that the only host reader is a nonlinear
       local solve and that one does not run with a device configured. NPC has
       no local nonlinear solve at all -- it reaches the same host MultInv()
       directly -- so the reader is reached, and by the ordering that matters:
       InvertA() or FactorElementsBatched() leaves the pivots on the device,
       NPCReduce()'s MultInvBatched() puts them back there, and the
       matrix-free Gradient::Mult() then indexes them on the host.

       It covers the face blocks Ct, E, G and H as well as the element ones,
       because AssemblyMode::Batched's face kernel writes E, G, H and D on the
       device and the BOUNDARY face pass reads D on the host in the very next
       loop.

       Costs a flag check per array when nothing has moved. */
   // (declared in the public section; see SyncLocalBlocksToHost() there)

   /** @brief Whether each element's field dofs are its own.

       True for two discontinuous spaces, and then an element loop that WRITES
       field dofs writes somewhere no other element does. Two things need it,
       for the same reason and at opposite ends of the machine: a threaded loop
       needs it to run without a colouring (see CanThreadFieldLoop()), and a
       device scatter needs it because Vector::SetSubVector() is an unordered
       mfem::forall -- two entries of the map naming one dof would race, where
       the serial loop's last-writer-wins is element order.

       An H(div) flux shares dofs across faces, so both fall back to the serial
       element loop there. That is the RT pathway, which this branch leaves
       alone. */
   bool FieldDofsAreElementLocal() const
   {
      return fes.FEColl()->GetContType() ==
             FiniteElementCollection::DISCONTINUOUS
             && fes_p.FEColl()->GetContType() ==
             FiniteElementCollection::DISCONTINUOUS;
   }

   /** @brief Whether an element loop that writes FIELD dofs may be threaded.

       It may when both field spaces are discontinuous, and then each element's
       flux and potential dofs are its own -- no colouring, no atomics. An
       H(div) flux shares dofs across faces, and there these loops either
       accumulate into a shared entry or, in ComputeSolution(), overwrite it,
       where serial's last-writer-wins is element order and a colouring would
       change which element wins. That is a question about the RT pathway, and
       the standing instruction on this branch is to leave it alone: with a
       non-discontinuous space these loops keep the serial loop they have
       always had.

       ReduceRHS() does not use this. It writes only TRACE dofs, so the
       colouring covers it whatever the flux space is. */
   bool CanThreadFieldLoop() const
   {
      return ThreadHostLoops() && FieldDofsAreElementLocal();
   }

   FaceElementTransformations *GetFaceTransformation(int f) const;
   /// @brief The same face transformation, built into @a ws instead of into
   /// the Mesh's shared cache, so it is safe on a threaded element loop.
   FaceElementTransformations *GetFaceTransformation(int f,
                                                     TransWorkspace &ws) const;

   /// Allocate everything whose size comes from the trace element.
   void AllocTraceBlocks();

   /// Convert a list of constraint-space VDOFs to true DOFs.
   void TraceVDofsToTDofs(const Array<int> &vdofs, Array<int> &tdofs) const;

   /** @brief Express the caller's essential trace DOFs in the constrained
       numbering.

       #ess_tdof_user holds them as the constraint space's own true DOFs,
       which is what the caller can name; the reduced system is in the
       constrained ones. A trace face's DOFs are face-interior, so a face is
       essential in a given field or it is not, and the map is per face rather
       than per DOF. Called from Finalize(), which is where the degrees stop
       changing, and idempotent because it rebuilds #ess_tdof_list from
       #ess_tdof_user each time. */
   void MapEssentialTraceDofs();

   /** @brief Build #ctr_offsets, #ctr_PE and #ctr_Pv. Idempotent, lazy. */
   void BuildTraceConstraint() const;

   /** @brief The face maps of face @a f: E embeds its degree in the ceiling,
       R reads a ceiling function back at its nodes.

       E(j,i) = phi_i^lo(node_j^hi) and R(i,j) = phi_j^hi(node_i^lo), so
       R E = I exactly and a degree-p_f function has exactly nt(p_f) degrees
       of freedom in the ceiling's storage. Cached by (geometry, degree),
       because the rows of E are the ceiling element's nodes in the face's own
       reference ordering -- which is the ordering GetFaceVDofs() returns --
       so orientation never enters them. It enters the nonconforming transfer
       and the parallel true-DOF map, and both act at the ceiling where the
       space handles it already.

       R E = I is CHECKED here, once per distinct key, rather than assumed:
       it is what says the ceiling collection really is nodal and really does
       contain the coarse space. A modal or non-nested collection fails it
       loudly instead of quietly discretising something else. */
   const DenseMatrix &TraceEmbedding(int f) const;
   const DenseMatrix &TraceRestrictionMat(int f) const;

   /** @brief The trace element of face @a f's own DEGREE, as against
       TraceFE(), which is the ceiling's and is where its DOFs live.

       Private, and needed only to build the constraint: nothing outside this
       class has a use for a basis the stored coefficients are not in. */
   const FiniteElement *FaceDegreeFE(int f) const;
   /** @brief Refuse the combinations a component-RESTRICTED flux is not
       supported in, once, at Finalize().

       Fires only when the flux constraint integrator is one of the restricted
       pair -- a caller who has not asked for a restricted flux is not
       affected, and the SHAPE guards in AssembleDivMatrix() and
       ConstructC() are what catch a flux space that is short without having
       said so.

       What it refuses, and every one of them is a thing that was measured or
       read rather than guessed:

       * a non-NPC assembly, and only HALF of that -- the LINEAR reduced
         route, which is the one Finalize() itself takes. NPCEnabled() is
         `bnpc || IsNonlinear()`, so a nonlinear form reads as NPC here
         whether or not EnableNPC() was ever called, and requiring @a bnpc
         instead would refuse DarcyOperator's own NPC branch, which cannot
         call it (it runs FormLinearSystem() first, and EnableNPC() after
         Finalize() is refused). The nonlinear reduced route is therefore
         refused at its own entry, DarcyHybridization::Mult(), which NPC
         never reaches. Out of scope rather than known broken.
       * a VECTOR-range flux space. An H(div) element's components are
         intrinsic; there is no block to drop.
       * more than one equation. The flux of a system is laid out equation
         outermost and then component, so the slice arithmetic in the two
         restricted integrators is a different expression.
       * a BLOCK nonlinear flux law. MixedConductionNLFIntegrator sizes its
         element vector as `neq * sdim * ndof_u` from Trans.GetSpaceDim() and
         asserts it only under MFEM_DEBUG, so a restricted flux would be
         silently wrong there. The LINEAR flux mass is fine --
         VectorMassIntegrator::SetVDim() has always taken the count from the
         caller.
       * a component list that disagrees with the space it is describing.

       Called TWICE, from Init() and again from Finalize(): the first so that
       a wrong equation count or a vector-range space is named before
       ConstructC()'s shape guard reports it as a block of the wrong height,
       the second because a caller may install a block nonlinear flux law
       between the two.

       It used to take a @a require_npc flag and refuse the condensation route
       outright. That clause is gone -- measured, not argued; see the note on
       the routine. */
   void CheckRestrictedFluxConfiguration() const;

   void AssembleCtFaceMatrix(int face, const DenseMatrix &elmat);
   void AssembleCtSubMatrix(int el, const DenseMatrix &elmat,
                            DenseMatrix &Ct, int ioff=0);
   using face_getter = std::function<void(int, DenseMatrix &)>;
   void AssembleNCSlaveFaceMatrix(int f,
                                  face_getter fx_Ct = face_getter(), const DenseMatrix *Ct = NULL,
                                  face_getter fx_C = face_getter(), const DenseMatrix *C = NULL,
                                  face_getter fx_H = face_getter(), const DenseMatrix *H = NULL);
   /** @brief Master-to-slave trace transfer for one nonconforming sub-face.
       Height = slave dofs, width = master dofs, so it maps master coefficients
       to slave ones. False when @a slave_face is not a slave or its master is
       a ghost. See the definition for why it is a routine and not inline. */
   bool GetNCSlaveTransfer(int slave_face, DenseMatrix &I,
                           int &master_face) const;
   /** @brief Whether @a face's trace row is a boundary condition, so the flux
       constraint is left out of it. False unless SetTraceBCAttributes() was
       given a marker covering this face's boundary attribute; false on every
       interior face, a boundary condition being a property of a boundary. */
   bool TraceRowIsBC(int face) const;
   /// Whether any face at all answers TraceRowIsBC(); the gate on the routes
   /// that gather C wholesale and cannot express the omission.
   bool HasTraceBCFaces() const { return bc_trace_marker.Size() > 0; }

   /** @brief The slave sub-faces a nonconforming MASTER face is tiled by.

       False, with @a slaves emptied, when @a face is not a master -- which is
       every face of a conforming mesh, every interior and boundary face of a
       nonconforming one, and a master whose slaves are all ghosts. */
   bool GetNCMasterSlaves(int face, Array<int> &slaves) const;
   /** @brief One slave sub-face of a master, ready for an element-major
       nonlinear face integrator.

       Fills @a I with the master-to-slave transfer, @a x_s with the master
       trace restricted to the slave (`I lambda_m`), and @a FTr with the
       slave's own two-sided transformation -- on which the coarse element is
       the SECOND, so a caller's usual `FTr->Elem1No != el` test sets the
       side bit without knowing any of this. */
   FaceElementTransformations *SetupNCSlaveFace(
      int slave, const Vector &x_m, DenseMatrix &I, Vector &x_s,
      FaceElementTransformations &FTr, IsoparametricTransformation &T1,
      IsoparametricTransformation &T2) const;

   void AssembleNCSlaveCtFaceMatrix(int f, const DenseMatrix &Ct);
   void AssembleNCSlaveEGFaceMatrix(int f, const DenseMatrix &E,
                                    const DenseMatrix &G);
   void AssembleNCSlaveHFaceMatrix(int f, const DenseMatrix &H);
   void ConstructC();
   /** @brief ConstructC()'s interior faces through
       HDGNormalTraceJumpBatched(), thresholded at @a mtol and compacted as
       AssembleCtFaceMatrix() does. False, having done nothing, unless
       CanBatchConstraintAssembly(). */
   bool ConstructCInteriorBatched(real_t mtol);
   void AllocD() const;
   void AllocEG() const;
   void AllocH() const;
   /** @a AtFields and @a GradAtFields are the NPC modes: the flux and
       potential are Newton STATE, supplied in @a darcy_u and @a darcy_p, and
       no local solve, substitution or linearisation happens at all. Every
       other mode produces the fields from the trace one way or another, which
       is what a reduced operator on the trace alone has to do and what NPC
       does not do. See NPCResidual(). */
   enum class MultNlMode { Mult, Sol, Grad, GradMult, AtFields, GradAtFields };
   /** @a r_local is where MultNlMode::AtFields writes the local rows of the
       full residual, and is required by that mode and ignored by every other.
       It is a parameter rather than a member because a member would smuggle
       state between calls, and because adding one to this class changes its
       layout -- which, with no header dependency tracking in this build,
       silently corrupts every translation unit that was not recompiled.

       NPC's reduction and recovery have their own element loops rather than
       sharing ReduceRHS()/ComputeSolution(): those two apply the LINEAR (0,1)
       block and negate the potential block on the way in, both right for a
       linear system and wrong for a Jacobian. */
   /** @brief Does MultNL()'s element loop EVALUATE an integrator, for @a mode
       and this configuration?

       This is the whole of the thread-safety question, and it took a reply
       from meq to get it right -- see SetIntegratorsThreadSafe(). Everything
       in that loop is dense linear algebra on blocks that are already
       assembled, with exactly seven exceptions, and they are the only things
       a second thread can collide inside: @a m_nlfi, @a m_nlfi_u and
       @a m_nlfi_p per element, and @a c_nlfi, @a c_nlfi_p and the two
       boundary_constraint_*_nonlin_integs arrays per face.

       The old predicate for this was `!CopyLinearGradBlocks()`, and it was
       wrong in BOTH directions. It over-refused because that routine declines
       for reasons that have nothing to do with a live integrator -- most of
       all LocalOpType::PotNL and FluxNL, where the local block is prefactored
       and ConstructGrad() deliberately leaves it alone, copying
       Af_lin_data/Df_lin_data rather than assembling anything. And it
       under-refused twice over: the RESIDUAL and local-solve modes evaluate
       the same integrators through LocalNLOperator and were never asked
       about, and ConstructGrad()'s face branches run whatever the cache did,
       so a nonlinear face constraint raced with the cache satisfied. */
   bool ThreadedLoopEvaluatesIntegrators(MultNlMode mode, bool ad_done) const;

   /** @brief @a ad_external says the element A and D blocks are ALREADY
       written and the loop must not write them again.

       A PARAMETER and not a member, deliberately: this header's layout is a
       full `make clean` in every tree, and a defaulted argument on a method
       costs nothing. It is the second producer of the state @a ad_done
       names -- the first being CopyLinearGradBlocks(), which declines for
       LocalOpType::PotNL precisely because the blocks are not a copy there --
       and it is how NonlinearGradientBatched() hands its work to the loop.

       It reaches exactly the five `ad_done` sites in ConstructGrad(), all of
       them in the A/D section; the face branches do not read it. */
   void MultNL(MultNlMode mode, const Vector &bu, const Vector &bp,
               const Vector &x, Vector &y,
               BlockVector *r_local = nullptr,
               bool ad_external = false) const;
   void MultNL(MultNlMode mode, const BlockVector &b, const Vector &x,
               Vector &y, BlockVector *r_local = nullptr,
               bool ad_external = false) const
   { MultNL(mode, b.GetBlock(0), b.GetBlock(1), x, y, r_local, ad_external); }
   void ParMultNL(MultNlMode mode, const BlockVector &b, const Vector &x,
                  Vector &y, bool ad_external = false) const;
   /** @brief The half of a gradient that follows the fields existing: factor
       the local blocks, form the Schur complement, and hand back the reduced
       trace operator -- assembled or matrix-free according to
       SetGradientMode().

       @a mode selects how the element loop obtains the fields, and is the only
       difference between GetGradient() (MultNlMode::Grad, fields produced from
       the trace) and NPCGradient() (MultNlMode::GradAtFields, fields supplied
       as Newton state). */
   Operator &ReducedGradient(MultNlMode mode, const Vector &x_tr,
                             bool ad_external = false) const;
   /** @brief Add the queued AddNextGradientCorrection() blocks into D and
       empty the queue. Called by both reduced gradients between the
       element loop and ComputeH(), and only for MultNlMode::GradAtFields. */
   void ApplyNextGradientCorrections(MultNlMode mode) const;
#ifdef MFEM_USE_MPI
   /// The same, assembling a HypreParMatrix. @a x_tr is in TRUE dofs.
   Operator &ParReducedGradient(MultNlMode mode, const Vector &x_tr,
                                bool ad_external = false) const;
#endif
   void InvertA();
   void InvertD();
   /** @brief The size every element's block has in @a f_offsets, or -1
       when they are not all equal. */
   static int UniformBlockSize(const Array<int> &f_offsets, int NE);
   void GetElementFaces(int el, Array<int> &faces) const;
   /** @brief What ComputeH() is being asked for.

       GradientFactorOnly does the first half of Gradient -- factoring A and
       the Schur complement of every element, with the Jacobian's (0,1) block
       -- and stops before assembling the global matrix. It is what
       GradientMode::MatrixFree needs, and it is also all the initialisation
       pass in MultNL() ever needed: that pass discarded the matrix it built. */
   enum class ComputeHMode { Linear, Gradient, GradientFactorOnly };
   /// Total trace dofs on @a faces, which is the side of an element's H block.
   int GetElementTraceSize(const Array<int> &faces) const;
   /** @brief The element-local half of ComputeH() for one element: factor A,
       form and factor the Schur complement, and -- unless the mode is
       GradientFactorOnly -- evaluate this element's face-pair blocks of H
       into @a Hel.

       Everything it reads is either shared and const or indexed by @a el, and
       everything it writes is indexed by @a el, so two elements may run
       concurrently. Its scratch is local for the same reason: the scratch the
       single loop hoisted and reused across elements is exactly what threads
       cannot share.

       @a Hel receives the (f2,f1) blocks contiguously, f1 outer and f2 inner,
       in the order ScatterElementH() replays them; it may be NULL when the
       mode is GradientFactorOnly. */
   struct SerialHWorkspace;
   void ComputeElementH(int el, ComputeHMode mode, real_t *Hel,
                        SerialHWorkspace &ws,
                        const Vector *AiBt_all = NULL,
                        CondensationCache *cc = NULL) const;
   /** @brief The element-local FACTORISATION half of ComputeElementH() -- the
       LU of A, the Schur complement and its LU -- for every element in one
       batch of BatchedLinAlg calls, and A^-1 times the negated (0,1) block
       into @a AiBt_all.

       Returns false, having done nothing, unless LocalFactorMode::Batched is
       asked for and the three arrays it views as DenseTensors are present and
       one block size; the element loop then factors as it always has. When it
       returns true the caller passes @a AiBt_all to ComputeElementH(), which
       skips the arithmetic done here and reads its element's slice.

       Every operation is a BatchedLinAlg call or one mfem::forall, so with a
       device configured the whole factorisation runs there and nothing comes
       back -- and in ComputeHMode::GradientFactorOnly nothing needs to, that
       mode having no face loop after it. The assembling modes do: their face
       loop is host dense work over Ct/E/G, and it pulls A, the Schur
       complement and @a AiBt_all back. That transfer is step 2's to remove,
       not this one's.

       **Bit-for-bit the element loop in a build without LAPACK, and this
       used to say it was not.** The claim was that the Schur complement goes
       through BatchedLinAlg::AddMult() where the loop uses mfem::AddMult()
       and "the two accumulate a product in different orders". They do not:
       both run the same j-k-i loop over the same products, and
       kernels::AddMult()'s `alpha` is 1.0 here, which is an exact multiply.
       The test that says so was already in the tree and passing -- "The
       batched element factorisation assembles the same trace operator"
       compares the assembled H entrywise with RequireSame(), not with a
       tolerance. An asserted claim sat next to a measurement that refuted it
       for as long as nobody read them together.
       With LAPACK the element side is dgemm_ and only round-off agreement is
       claimed; likewise on the GPU_BLAS and MAGMA backends. Measured
       agreement is on SetLocalFactorMode(). */
   bool FactorElementsBatched(ComputeHMode mode, Vector &AiBt_all,
                              CondensationCache *cc = NULL) const;

   /** @brief Whether FactorElementsBatched() will do the work: the mode asked
       for AND the storage it needs, reported through @a na and @a nd.

       **It exists so that two predicates cannot ask different questions about
       one route**, which this class has paid for once already --
       CanBatchTraceAssembly() asked the connectivity question and not the
       destination question, and two tests then passed against a deliberately
       broken kernel by comparing the serial route with itself.

       The caller here is CanCacheCondensation(), for which the distinction is
       not academic: a mesh whose A or D blocks are not all one size refuses
       the batched factorisation on storage, so refusing the condensation
       cache on the MODE alone left that caller with neither -- the fallback
       loop and no cache, in exchange for nothing.

       The one refusal of FactorElementsBatched() this does NOT carry is its
       LocalOpType::FluxNL branch, which needs Sf_data and Df_lin_data sized;
       that mode and LocalOpType::PotNL are exclusive, so it cannot bear on
       the cache, whose first test is PotNL. */
   bool BatchedLocalFactorStorage(int &na, int &nd) const;

   /** @brief The (element, local face) index map ComputeElementsHBatched()
       reads: three offsets per entry -- into @a Ct_data, into @a E_data and
       @a G_data, and into @a H_data -- for the whole mesh, laid out
       interleaved at 3*(el*@a nf + lf).

       Returns false, and @a nf and @a nc are then meaningless, unless every
       element has the same number of faces and every face the same number of
       trace dofs. That is the batched face loop's whole precondition beyond
       the uniform A and D blocks CanBatchLocalFactor() asks for, and it is
       asked of the mesh and the trace space rather than assumed from
       "uniform mesh at uniform order" -- a per-face trace degree
       (SetTraceOrders()) breaks it while leaving the local blocks uniform.

       The H offset is -1 where this element is not the face's FIRST, which is
       how the element loop's "integrate the face contribution only on one
       side" is carried into a kernel that cannot branch on the mesh. */
   bool BuildElementHFaceMap(int na, int nd, bool with_h, int &nf, int &nc,
                             Array<int> &face_map) const;
   /** @brief The FACE-PAIR half of ComputeElementH() -- everything after the
       factorisation -- for a chunk of @a nel elements at once, writing the
       same (f2, f1) block buffer the element loop writes.

       **This is the largest offloadable item in an NPC step**: 24% of one at
       order 2 and 45% at order 3, and the only share that GROWS with order.
       See ComputeH() for the profile it comes from.

       The double loop it replaces is one matrix identity per element, which
       is what makes the whole thing five BatchedLinAlg calls:

           H_el = -C^T A^-1 C + (C^T A^-1 B^T + G) S^-1 (B A^-1 C - E)

       with C the element's trace blocks side by side, (na, T), and G and E
       the matching stacks, T = @a nf * @a nc. The element loop's (f2, f1)
       block is exactly that expression's (f2, f1) block, so nothing is
       reassociated across faces -- what changes is that the inner products
       run over all of C at once instead of one face pair at a time.

       Bit-for-bit the element loop in a build without LAPACK, for the reason
       FactorElementsBatched() gives, and everything that is a SUM of two
       matrices is additionally done in the loop's own order: E is subtracted
       after the product rather than folded into an AddMult beta, which would
       accumulate on top of -E instead.

       **What it is worth, measured, interleaved A/B/C over five or six runs
       each, one thread, `convdiff -dg -hb -npc -nls 3 -gm 0 -rtol 1e-12`.**
       A is the element loop, B is LocalFactorMode::Batched before this
       existed, C is it now. Seconds, summed over the run's ComputeH() calls:

           case                    A      B      C     pairs B->C
           o2 n=128 -p 2 -nld    0.571  0.645  0.491     1.45x
           o3 n=128 -p 2 -nld    1.705  1.842  1.530     1.39x
           o2 n=128 -p 1 -nl     0.494  0.537  0.439     1.36x
           o5  n=64 -p 2 -nld    2.840  3.018  2.313     1.59x

       So the face-pair loop itself is 1.36-1.59x faster batched, and it is
       what makes LocalFactorMode::Batched a net gain at all -- **B is 5-9%
       SLOWER than A at every size**, which SetLocalFactorMode() used to claim
       the opposite of.

       **End to end it is inside run-to-run scatter**, and that is not a
       disappointment but the plan's own gate: at o3 n=128 the whole solve is
       8 s, of which the trace solve is 54-59%, so a 0.18 s saving cannot be
       resolved against a 5% spread (median 8.27 s against 8.57 s, min 7.91
       against 7.77, answers identical to every digit). Measure this where it
       happens, not at the end of a solve.

       @a Hel must be sized @a nel * T * T; the caller's chunking is what
       bounds it, and the alias Memory views mean the chunk needs no copy of
       the local blocks. The chunk length itself is nearly irrelevant on a
       host -- swept 4, 8, 16, 32, 64, 128, 256, 1024 elements at o2 n=128 and
       the face-pair time stayed in 0.73-0.86 s with no trend, which is what
       ruled out cache streaming as the explanation of the first, slow
       version. */
   void ComputeElementsHBatched(ComputeHMode mode, int el_0, int nel,
                                int na, int nd, int nf, int nc,
                                const Vector &AiBt_all,
                                const Array<int> &face_map,
                                ElementHWorkspace &ws, Vector &Hel,
                                CondensationCache *cc = NULL) const;
   /** @brief Add the blocks ComputeElementH() left in @a Hel to @a H.
       Serial by contract -- see SetAssemblyMode(). */
   /** @brief The per-element scratch the SERIAL ComputeElementH() and
       ScatterElementH() pair works in; defined in the source file. Distinct
       from ElementHWorkspace, which is the BATCHED face-pair route's.

       An opaque type for the reason TraceHMap and ResidualCache are: a struct
       spelled out here would put its layout in every TU that includes
       mfem.hpp. It exists because the routine asked c_fes for the same face's
       vdofs nf*(nf+1) times per element -- 21,504 malloc/free pairs of FOUR
       BYTES on a 256-element problem, the largest single allocation site in
       an HDG solve -- and because ComputeElementH() beside it built four
       DenseMatrix objects per element for the same reason. See the
       definition. */
   void ScatterElementH(int el, const real_t *Hel, SparseMatrix &H,
                        SerialHWorkspace &ws) const;

   /** @brief Build, or return, the cached connectivity
       TraceAssemblyMode::Batched assembles against. NULL when this
       hybridization cannot be assembled that way.

       Everything in the map is a function of the mesh and the trace space
       alone -- no value of any block enters it -- and that is the fact the
       whole mode rests on, so it is worth saying why it holds. The host
       route's pattern is `AddSubMatrix(skip_zeros=1)` per element followed by
       Finalize(), which drops an entry only if it is exactly zero, and every
       structurally present entry of H is nonzero in at least one of its
       elements. **Measured rather than argued**, because a value-dependent
       pattern would have sunk this design: the STRUCTURAL count -- every
       (row, col) whose two faces share an element -- equals
       NumNonZeroElems() exactly on 2-D triangles and quads and 3-D hexes at
       orders 0 to 3, at several mesh sizes. Where they ever differ this mode
       carries extra entries that are exactly 0.0, which changes no product.

       Four facts hold it up, each checked here rather than assumed:

       - **Every trace dof lies on exactly one face**, so one thread owns a
         whole row of H and no reduction is needed anywhere. True of a
         DG_Interface space; an H1 trace shares dofs between faces and is
         refused.
       - **A row's entry is a sum of at most two terms.** Its column's face
         and its row's face must share an element, and a face has at most two
         -- so the two contributions are written by separate passes, in
         ascending element order, and a two-term IEEE sum does not depend on
         the order anyway. No atomics, no segmented reduction.
       - **A face's dofs need not be contiguous**, and in 2-D they are not, so
         the map carries the (face, local dof) -> dof table explicitly instead
         of computing it.
       - **Each row's columns are laid out one neighbour face at a time**,
         which is what lets a per-nonzero index be recovered from
         `I[row] + slot*nc + j` and keeps the map O(NE nf^2) instead of
         O(NE (nf nc)^2). It is also why the column ORDER differs from
         Serial's; see TraceAssemblyMode::Batched. */
   const TraceHMap *BuildTraceHMap() const;
   /** @brief Give @a H the cached pattern and a zeroed data array, finalized
       and ready for ScatterElementsHBatched() to fill.

       @a V comes back as the view of @a H's data the scatter writes through,
       and it is created ONCE for the whole element loop rather than per
       chunk: a second Memory alias over the same range starts from the
       source's flags and needs syncing back, and a chain of them per chunk is
       exactly where this branch's device defects have lived. One view, one
       Sync after the loop.

       @a pattern_on_device puts @a H's @a I and @a J on the device, copied
       from the map's device copies, and leaves them host-invalid. Only
       ComputeHMode::Gradient asks for it: every reader of a gradient's
       pattern in this class is a kernel, and with a host pattern each
       linearisation uploaded it again. The linear matrix stays on the host,
       for EliminateTraceTrueDofs(); see the definition. */
   void BuildTraceHPattern(const TraceHMap &map,
                           std::unique_ptr<SparseMatrix> &H, Vector &V,
                           bool pattern_on_device = false) const;
   /** @brief Add a chunk's blocks from @a Hel straight into @a H's CSR data,
       as two kernels: the elements that are the FIRST contributor to an entry
       assign, and those that are the second add.

       Two passes rather than one with atomics, and that is what makes it
       bit-for-bit: an entry's two elements never write in the same kernel,
       and the first is always the lower element index -- so the sum is
       assembled in ascending element order however the chunks fall. */
   void ScatterElementsHBatched(const TraceHMap &map, int el_0, int nel,
                                const Vector &Hel, Vector &V) const;

   /// Elements per chunk of the element loop; see ComputeH().
   int AssemblyChunkSize(int NE) const;
   /** @brief Build the trace-trace block H, and factor the local blocks on
       the way.

       **This is the largest single cost of an NPC step, and it was not where
       the device-offload plan expected to find it.** Measured on
       `convdiff -dg -hb -npc -nls 3 -gm 0 -rtol 1e-12`, 128x128 quads, one
       thread, with a DIRECT trace solve so the Krylov question is excluded --
       shares of the step's own work, that trace solve being a further 54-59%
       of the run:

       | | `-p 1 -nl`, k=2 | `-p 2 -nld`, k=2 | `-p 2 -nld`, k=3 |
       |---|---|---|---|
       | this routine | 39% | 54% | 61% |
       | -- ComputeElementH(), dense | 24% | 31% | 45% |
       | -- ScatterElementH(), sparse | 10% | 13% | 9% |
       | -- Finalize() + RAP | 7% | 11% | 7% |
       | the integrators in the residual | 5% | 7% | 6% |

       Two things follow. The dense half GROWS with order (dofs cubed) while
       the integrator evaluation shrinks (quadrature points times dofs), so a
       plan ranked on the integrators gets further from the truth exactly where
       the method is expensive. And the whole of that dense half is batched
       now: LocalFactorMode::Batched takes the factorisation through
       FactorElementsBatched() and the `C A^-1 C^T` face-PAIR loop through
       ComputeElementsHBatched(), which is where the timings live.

       The sparse third of it, ScatterElementH() plus Finalize(), is host-only
       work by construction; GradientMode::MatrixFree is what deletes it, at
       the cost of an unpreconditioned trace solve. **It is the next item**,
       and it is now comparable to the dense half rather than a third of it:
       with the face pairs batched, the scatter is 0.21-0.35 s against the
       pairs' 0.27-1.15 s at the four sizes on ComputeElementsHBatched(). */
   void ComputeH(ComputeHMode mode, std::unique_ptr<SparseMatrix> &H) const;
#ifdef MFEM_USE_MPI
   void ComputeParH(ComputeHMode mode, std::unique_ptr<SparseMatrix> &H,
                    OperatorHandle &pH) const;
#endif
   void GetCtFaceMatrix(int f, int side, DenseMatrix & Ct) const;
   void GetEFaceMatrix(int f, int side, DenseMatrix &E) const;
   void GetGFaceMatrix(int f, int side, DenseMatrix &G) const;
   void GetHFaceMatrix(int f, DenseMatrix &H) const;
   void GetCtSubMatrix(int el, const Array<int> &c_dofs, DenseMatrix &Ct) const;
   void MultInvNL(int el, const Vector &bu_l, const Vector &bp_l,
                  const BlockVector &x_l, Vector &u_l, Vector &p_l,
                  TransWorkspace &ws) const;
   /** @brief The flux and potential the linearisation implies for the trace
       @a x_l, by a local nonlinear solve. */
   /** @brief The trace space's prolongation from true dofs to L-dofs, or
       NULL when the two coincide and no mapping is needed.

       NPC's element loops work in L-dofs, as every loop in this class does;
       its public interface is in true dofs, as every MFEM Operator's is. This
       is the one place the two meet. The flux and potential need no such
       mapping under NPC because it refuses anything but a discontinuous flux
       space, so their L-dofs are their true dofs. */
   const Operator *TraceProlongation() const;
   /** @brief NPC's shared precondition: finalized and a discontinuous flux
       space. LocalOpType::FluxNL was refused here too until its Schur
       complement got somewhere to live (@a Sf_data); see the definition. */
   void NPCCheck() const;
   /// A correctly sized zero load for a gradient pass; see the definition.
   void ZeroLoad(BlockVector &b, bool true_dofs) const;

   /** @brief The local nonlinear residual of @a el at (@a u_l, @a p_l).

       @a elem_flux_row, when given, is this element's slice of the block
       integrator's flux row as HDGMixedConductionResidualBatched() computed
       it for the whole mesh, and the element term is read from there instead
       of from the integrator. A PARAMETER and not a member, per the standing
       note: a member on DarcyHybridization moves the class layout and every
       translation unit that includes mfem.hpp with it. */
   void LocalResidual(int el, const Array<int> &faces, const BlockVector &x_l,
                      const Vector &bu_l, const Vector &bp_l,
                      const Vector &u_l, const Vector &p_l,
                      Vector &ru_l, Vector &rp_l, TransWorkspace &ws,
                      const Vector *elem_flux_row = NULL,
                      bool skip_interior_faces = false) const;
   /** @param wk optional scratch of size Af_f_offsets[el+1]-Af_f_offsets[el],
                  hoisted above an element loop so the routine's one temporary
                  is not a fresh allocation per element. See the definition. */
   void MultInv(int el, const Vector &bu, const Vector &bp, Vector &u,
                Vector &p, bool with_bnl = false, Vector *wk = NULL) const;
   /** @brief MultInv() for several right-hand sides at once, on one element.

       @a bu and @a bp carry one column per right-hand side and @a u and @a p
       come back the same way. Every step of MultInv() is either a
       LUFactors::Solve(), whose second argument is already a column count, or
       a dense product, so this is that routine with the column count carried
       through: matrix-matrix where MultInv() is matrix-vector, and the same
       arithmetic on any one column. Nothing here is a new algorithm.

       This is a DIFFERENT axis from MultInvBatched(), which blocks the same
       solve over ELEMENTS at one column. The two do not compose in either
       direction and this one takes no notice of CanBatchLocalSolve(): a
       caller with LocalFactorMode::Batched and a single right-hand side
       should stay on MultInv()/MultInvBatched().

       @param wk optional scratch, hoisted above an element loop exactly as
                 MultInv()'s is. */
   void MultInvBlocked(int el, const DenseMatrix &bu, const DenseMatrix &bp,
                       DenseMatrix &u, DenseMatrix &p, bool with_bnl = false,
                       DenseMatrix *wk = NULL) const;
   /** @brief MultInv() for every element at once, on element-blocked vectors.

       @a bu and @a bp carry the elements' right-hand sides end to end in
       element order -- sizes Af_f_offsets.Last() and Df_f_offsets.Last() --
       and @a u and @a p come back the same way. Every step is a
       BatchedLinAlg call, so on a device nothing is read back.

       The caller's gather into @a bu / @a bp and scatter out of @a u / @a p
       are kernels too now, one Vector::GetSubVector()/SetSubVector() over
       el_u_dofs / el_p_dofs each; this doxygen used to say they "remain on
       the host", and they were the larger cost. What remains on the host is
       the FACE loop at each call site, and where one follows the answers it
       still has to be read back -- except in NPCReduce() and NPCRecover(),
       whose face loops are kernels too under CanBatchNPCTraversal().

       Requires CanBatchLocalFactor(), since a DenseTensor is one block size.
       Bit-for-bit the per-element route in a build without LAPACK, where
       LUFactors::Solve() is the kernels::LSolve/USolve pair that
       NativeBatchedLinAlg::LUSolve() calls; with LAPACK the per-element side
       is dgetrs_ and only round-off agreement is claimed. See MultInv().

       **Several right-hand sides at once**, with no change to the arithmetic:
       @a bu may carry K columns per element, laid out (na, K, NE) -- element
       outermost, then column, then row, which is the shape BatchedLinAlg's
       LUSolve() and AddMult() already take -- and K is read off its size.
       Each column is solved exactly as a one-column call would solve it.
       This is what lets the blocked NPCReduce()/NPCRecover() overloads use
       the batched solve, where MultInvBlocked() is the per-element host
       route. */
   void MultInvBatched(const Vector &bu, const Vector &bp, Vector &u,
                       Vector &p, bool with_bnl = false) const;
   /// Build, or return, the map behind CanBatchNPCTraversal(); NULL when the
   /// mesh or the spaces cannot supply one. See NPCTraversalMap.
   const NPCTraversalMap *BuildNPCTraversalMap() const;
   /** @brief The whole of NPCReduce() as kernels, for @a r.Size() columns at
       once: gather, batched local solve, the face products and the scatter
       into the LOCAL trace vectors @a y, which it zeroes. The caller applies
       the prolongation and the load, as it does on the host route. */
   void NPCReduceBatched(const Array<const BlockVector *> &r,
                         const Array<Vector *> &y) const;
   /** @brief The whole of NPCRecover() as kernels, reading the LOCAL trace
       increments @a dtr_l and writing @a dx, which it zeroes first. */
   void NPCRecoverBatched(const Array<const BlockVector *> &r,
                          const Array<const Vector *> &dtr_l,
                          const Array<BlockVector *> &dx) const;
   /** @brief The element-local Jacobian blocks at the given local state.

       @a skip_interior_faces leaves the INTERIOR-face constraint terms out of
       D, E, G and H, for a caller that is about to supply them with
       AssembleNLFaceGradBatched(). Boundary faces are unaffected: they carry
       their own marked integrator lists, which the batched kernel does not
       cover, and their `eg_written` bookkeeping is per face so it stays
       self-consistent when the interior faces are removed from the loop.

       A PARAMETER rather than a member on purpose. Adding a data member to
       this class changes its layout, and every translation unit that includes
       mfem.hpp then has to be rebuilt -- a trap this branch has paid for ten
       times, presenting as malloc corruption in unrelated code. */
   void ConstructGrad(int el, const Array<int> &faces, TransWorkspace &ws,
                      const BlockVector &x_l,
                      const Vector &u_l,
                      const Vector &p_l,
                      bool skip_interior_faces = false,
                      bool ad_done = false) const;
   /** @brief Do ConstructGrad()'s A and D work as two whole-array copies when
       that is all it is, and say whether it applied; see the definition. */
   bool CopyLinearGradBlocks() const;

   /** @brief Build res_cache if it is not built; false when it cannot be.
       Shared with CopyLinearGradBlocks() so the residual and the gradient are
       built from the same blocks -- see the definition for what went wrong
       when they were not. */
   bool EnsureResidualCache(int na, int nd) const;

   /** @brief The nonlinear interior-face constraint's residual for every face
       at once -- the sibling of AssembleNLFaceGradBatched(), one tensor rank
       lower. False when it does not apply.

       The gradient carries two neq x neq weight matrices per quadrature
       point; this carries two neq-vectors -- the numerical flux and the
       stabilization AT the state rather than differentiated -- and contracts
       them against the same shape tables, so the substitution that IS this
       routine is HDGFlux::Average() where the gradient calls AverageGrad().

       **Modest on a host and that is expected**: it removes the per-face
       frame (GetInteriorFaceTransformations(), the dof lookups,
       LocalNLOperator's per-face work) and keeps the per-point evaluation on
       the host, which the frame/integrand ablation says is where the cost
       actually is. Measured on a Burgers HDG face constraint, 48x48, 20 NPC
       residuals: 1.17x, 1.17x, 1.19x, 1.33x, 1.28x, 1.36x over orders 1-3 on
       quads and triangles, run-to-run spread about 5%. The shape is the
       point -- it is what a device pass needs -- not the host ratio.

       Those were 1.05x-1.24x until the @a checked parameter below stopped
       the predicate being evaluated twice per residual; see the note on it. */
   bool AssembleNLFaceResidualBatched(const Vector &x, BlockVector &r_local,
                                      Vector &y, bool checked = false) const;
   /** @brief The whole of MultNL(MultNlMode::AtFields)'s element loop as
       batched operations: the NPC residual's two field rows and its trace
       row, for every element at once.

       1.6x to 9.7x against the element loop on a linear problem and 3.9x to
       14.2x when the mass terms sit on the nonlinear slots; the table, and
       the two costs that had to go before either was true, are on
       ResidualCache in the source file. */
   bool LinearResidualBatched(const Vector &x, const Vector &bu,
                              const Vector &bp, BlockVector &r_local,
                              Vector &y) const;

   /** @brief Every row of the local residual except the scatter, shared by
       the two batched residual routes; see the source file. */
   bool ResidualRowsBatched(const Vector &x, const Vector &bu,
                            const Vector &bp, bool frozen_dp) const;

   /// NPCResidual()'s head and tail, shared with its templated overload.
   void NPCResidualStart(const BlockVector &x, const Vector &x_tr,
                         BlockVector &r, Vector &r_tr, Vector &x_tr_l,
                         Vector &r_tr_l);
   void NPCResidualFinish(Vector &r_tr_l, Vector &r_tr);

   /** @brief The half of NonlinearResidualBatched() that runs before the
       caller's integrand: the gathers, every linear term of all three rows,
       and the quadrature tables the kernel reads.

       Leaves the potential row holding everything BUT the integrand, so the
       kernel accumulates onto it. False when the route does not apply, and
       then nothing has been written. */
   bool NonlinearResidualPrepare(const Vector &x, const Vector &bu,
                                 const Vector &bp,
                                 HDGNLResidualWork &w) const;

   /// The scatter, after the kernel; see NonlinearResidualBatched().
   void NonlinearResidualFinish(BlockVector &r_local, Vector &y) const;
   /// Carry ResidualRowsBatched()'s view flags back to their arrays.
   void SyncResidualViews(const DenseTensor &A, const DenseTensor &B,
                          const DenseTensor &D) const;

   /** @brief The mesh-wide shape table and the per-point physical coordinate
       and weight, built once and shared by the residual and gradient routes.

       Shared rather than duplicated because the two must quadrature
       IDENTICALLY -- otherwise what they return is a residual and the
       Jacobian of something else -- and one copy is what makes that a fact.
       @a nq_out and @a dim_out come back as the rule's size and the space
       dimension. */
   void EnsureElementQuadratureTables(int &nq_out, int &dim_out) const;

   /** @brief Gather the linearisation point and point @a w at D, for the
       caller's derivative. Reads @a darcy_p, GradAtFields' own state. */
   bool NonlinearGradientPrepare(HDGNLResidualWork &w) const;

   /// @brief Add the linear part of D, which the kernel does not write.
   void NonlinearGradientFinish() const;

   /** @brief The state-carrying interior-face constraint GRADIENT for every
       (element, face) pair in one batched kernel, instead of one
       AssembleHDGFaceGrad() call and four DenseMatrix::CopyMN() per pair.

       @returns false, having done nothing, whenever it does not apply --
       CanBatchNLFaceGrad() is the same question asked in advance. The caller
       then keeps its per-pair loop, so this is an optimisation and never a
       restriction.

       @a x is the global trace vector; the element states come from
       @a darcy_u / @a darcy_p, which is why this is an NPC-only path: those
       hold the fields as Newton state under MultNlMode::GradAtFields, and
       under MultNlMode::Grad the fields are produced BY the element loop and
       are gone by the time this could run.

       Interior faces only. Boundary faces stay in ConstructGrad(): they carry
       marked integrator lists the kernel does not cover, and they are the
       reason D is accumulated in a different ORDER by the two routes -- the
       boundary contributions reach it during the element loop and the
       interior ones afterwards. Measured, that is 2.2e-16 in D, bitwise
       equality in E, G and H.

       **What it is worth, measured on the reachable case.**
       `convdiff -p 6 -o 2 -dg -hb -nl -npc -nls 3 -gm 0 -rtol 1e-12` at
       128x128, one thread, direct trace solve -- a 13.4 s solve of which the
       state-carrying constraint integrators are 1.54 s (0.94 s in the
       gradient, 0.60 s in the residual), i.e. 12% of the whole run and 23% of
       the NPC step's own work excluding the trace solve. At order 3 the same
       figures are 2.46 s of 30 s, 17% of the step -- the share FALLS with
       order, because the integrators go as quadrature points times dofs while
       ComputeH's Schur complement goes as dofs cubed.

       Of one gradient call, 58% at order 2, 64% at order 3 and 78% at order 5
       is what this can move: the outer-product accumulation plus the
       DenseMatrix scaffolding around it. The rest is the host precompute --
       transformations, CalcOrtho, shape evaluation, the flux Jacobian -- which
       stays on the host in either route because ElementTransformation and
       Coefficient carry no MFEM_HOST_DEVICE. So the ceiling here is 4.9% of
       the run at order 2 and 3, and the plan's gate says no step of this chain
       pays on its own.

       The plan's own profile put "the integrators" at 5-7% of an NPC step.
       That was measured on `-p 1 -nl` and `-p 2 -nld`, whose face constraint
       is LINEAR and therefore assembled once; on a genuinely state-carrying
       constraint it is 23%.

       **And end to end it does pay, above order 2, which the plan's gate
       says it should not.** Whole-solve wall clock on the same case, three
       interleaved per-pair/batched pairs at each size, one thread:

       | | per-pair (s) | batched (s) | ratio |
       |---|---|---|---|
       | order 2, 64x64 | 2.456 2.457 2.493 | 2.536 2.564 2.525 | 1.03 1.04 1.01 |
       | order 2, 128x128 | 11.60 11.68 11.56 | 11.85 12.18 11.85 | 1.02 1.04 1.03 |
       | order 3, 64x64 | 5.213 5.076 5.110 | 4.945 4.890 4.955 | 0.95 0.96 0.97 |
       | order 3, 128x128 | 24.63 24.22 22.34 | 24.10 23.53 21.18 | 0.98 0.97 0.95 |

       1-4% SLOWER at order 2 and 2-5% FASTER at order 3, and the crossover is
       the ablation's order trend arriving end to end: the batchable share of
       a pair goes 58% -> 64% -> 78% while what the kernel adds -- AtomicAdd
       on D where the per-pair loop does a plain +=, and one weight-array
       allocation per integrator pass -- does not. It is a small number either
       way and this is a host; the reason to have it is that the accumulation
       is now EXPRESSIBLE on a device, which it was not.

       One thing not measured: CanBatchNLFaceGrad() is evaluated THREE times
       per gradient pass -- once in MultNL(), once here, and once in the
       kernel's MFEM_VERIFY -- and each sweep walks every interior face and
       side asking each integrator for its rule. That is per-Newton-step
       overhead the assembly-time kernels never pay. The fix is one
       MFEM_ASSERT and one dropped check; what it is worth has not been
       measured, and the sign of the result flipping with ORDER says it is not
       the dominant term. */
   bool AssembleNLFaceGradBatched(const Vector &x) const;
   /** @brief One face integrator's contribution to D, E, G and H.

       @a eg_written says whether E and G for this face and side have already
       been written during this gradient pass: false overwrites them, true adds
       to them, and it is set on the way out. It exists because E and G hold
       one block per face and side and are *rewritten* rather than reset, so
       the first writer has to clear whatever the previous pass left and every
       writer after it has to accumulate. See ConstructGrad().

       @a I and @a face_eg are the nonconforming MASTER-face case and are
       given together. @a FTr is then a SLAVE sub-face of the master
       @a face_eg, @a x_f is the master trace restricted to it, and @a I is
       the transfer between the two. E and G are the COARSE element's rows
       against a trace it reaches only through the master's dofs, so they go
       to @a face_eg's slot carried by `E_s I` and `I^T G_s` -- which is what
       AssembleNCSlaveEGFaceMatrix() does for the linear blocks, and shares
       this transfer with. H needs none: it belongs to the slave's own dofs
       there, exactly as the linear route leaves it. */
   void AssembleHDGGrad(int el, FaceElementTransformations *FTr,
                        NonlinearFormIntegrator &nlfi,
                        const Vector &x_f, const Vector &p_l,
                        bool &eg_written, TransWorkspace &ws,
                        const DenseMatrix *I = NULL, int face_eg = -1) const;
   /// The BLOCK overload keeps its locals: it serves c_nlfi, which the
   /// reachability study found no miniapp fills, so it is not on a hot path.
   void AssembleHDGGrad(int el, FaceElementTransformations *FTr,
                        BlockNonlinearFormIntegrator &nlfi,
                        const Vector &x_f, const Vector &u_l, const Vector &p_l,
                        bool &eg_written,
                        const DenseMatrix *I = NULL, int face_eg = -1) const;

   /** @brief Put the frozen half of E and G for one (element, face) back,
       and declare the blocks written so the live pass accumulates.

       The offsets are AssembleHDGGrad()'s, deliberately reproduced rather
       than recomputed: the two write the same range and a second expression
       for it is a second thing to get wrong. @a c_dofs_size comes from the
       caller's trace block for the same reason. Does nothing when there is
       no frozen half. */
   void SeedLinearEG(int el, int face, int c_dofs_size,
                     bool &eg_written) const;

public:
   /// Constructor
   /** @param fes_u     flux space
       @param fes_p     potential space
       @param fes_c     constraint space
       @param bsymmetrize   sign convention of the mixed formulation, where
                            false keeps all terms without a change, while true
                            flips the sign of B and Mp to obtain a symmetric
                            system with -Bᵀ in the flux equation
    */
   DarcyHybridization(FiniteElementSpace *fes_u, FiniteElementSpace *fes_p,
                      FiniteElementSpace *fes_c, bool bsymmetrize = true);

   /// Destructor
   ~DarcyHybridization();

   /** @brief Sets Operator::DiagonalPolicy used upon construction of the
       linear system.
       Policies include:

       - DIAG_ZERO (Set the diagonal values to zero)
       - DIAG_ONE  (Set the diagonal values to one)
       - DIAG_KEEP (Keep the diagonal values)
   */
   void SetDiagonalPolicy(const DiagonalPolicy diag_policy_)
   { diag_policy = diag_policy_; }

   /** @brief Gets Operator::DiagonalPolicy used upon construction of the
       linear system. */
   DiagonalPolicy GetDiagonalPolicy() const { return diag_policy; }

   /** @brief The element-local nonlinear solve, used whenever the local
       operator is LocalOpType::FullNL.

       @note **Its rtol caps the outer solver, and the default 1e-6 is not
       tight enough for a convergence study.** The reduced residual is only as
       accurate as the local solve that produced it, so a Newton on the trace
       stalls just below this value and then oscillates. Measured on a problem
       made exactly LINEAR (a reaction integrator present but returning zero):
       the outer Newton fell from 1.04 to 9.5e-06 in one step and then sat
       there for 40 iterations without converging. At rtol 1e-13 the same
       problem converges in ONE step to 4.4e-14. Anything installed on the
       block nonlinear form forces FullNL, so this applies to callers who
       never asked for a local nonlinear solve at all. */
   void SetLocalNLSolver(LSsolveType type, int iters = 1000, real_t rtol = 1e-6,
                         real_t atol = 0., int print_lvl = -1)
   {
      lsolve.type = type;
      lsolve.iters = iters;
      lsolve.rtol = rtol;
      lsolve.atol = atol;
      lsolve.print_lvl = print_lvl;
   }

   void SetLocalNLPreconditioner(LPrecType type, int iters = 1000,
                                 real_t rtol = -1., real_t atol = -1.)
   {
      lsolve.prec.type = type;
      lsolve.prec.iters = iters;
      lsolve.prec.rtol = rtol;
      lsolve.prec.atol = atol;
   }

   /** @brief A consequence of a correct Jacobian that a caller can mistake
       for a regression, kept here because it cost one a day's debugging.

       **A better Jacobian can converge to a DIFFERENT solution.** Where a
       coarse discretisation carries more than one, an iteration driven by an
       inaccurate gradient wanders and can settle on the branch a Picard
       iteration finds; with the gradient right it converges faster and stays
       on its own. A caller had a test pinning Newton against Anderson-Picard
       on one mesh at 1e-6, and after a gradient fix it read 9.1e-05 --
       bit identical when the tolerance was tightened by four orders, so both
       iterations were fully converged and their fixed points genuinely
       differed, at 1e-13 on two other meshes and 3e-06 on a third with no
       trend. That is not a defect and not a discretisation regression: it is
       a gate that was green for the wrong reason. Pinning "two solvers agree"
       on a single coarse mesh should be a sweep. */

   /** @brief Choose how the element loop that builds the reduced system runs.
       AssemblyMode::Serial by default, so nothing existing changes.

       AssemblyMode::Threaded parallelises the element-local work and leaves
       the scatter serial and in element order. The scatter is not threaded
       because it cannot safely be: SparseMatrix::AddSubMatrix() reaches the
       matrix through SetColPtr(), and an unfinalized SparseMatrix carries one
       @a current_row, one column-pointer scratch array and one RowNode
       allocator for the whole matrix. Two threads adding to *disjoint rows*
       still collide on those, and the observed failure is a hang, not a wrong
       answer -- so element colouring, which buys disjoint rows, does not make
       this loop safe on its own.

       The two modes agree bit for bit, so a test may assert equality rather
       than a tolerance -- but note what does and does not buy that. It is not
       the ordering: a trace dof lives on a face and a face has at most two
       elements, so each entry is a sum of at most two terms and IEEE addition
       of two terms is order-independent. Scattering a chunk back-to-front was
       measured and changed nothing. What exactness rests on is that the
       element-local arithmetic is per-element and so reassociates nothing.
       Element order is kept because it is free and deterministic.

       **The loops are threaded in three different ways, and which one a loop
       takes is decided by what it WRITES.**
       ComputeH()'s is the one described above: element-local work in parallel,
       scatter serial and in element order. MultNL()'s -- which is the residual
       and the Jacobian assembly, and so NPCResidual() and NPCGradient() too --
       is walked in COLOUR order instead, Mesh::GetElementColoring() colouring
       the element-to-element graph whose edges are the faces, so no two
       elements of a colour share one. ReduceRHS() and NPCReduce() take that
       same colouring, writing the trace and only reading the fields -- which
       is why they are safe under it whatever the flux space is. And the loops
       that write FIELD dofs and nothing else -- ComputeSolution(),
       NPCRecover(), EliminateVDofsInRHS(), EliminateTrueDofsInRHS() -- need
       no colouring at all when both field spaces are discontinuous, which is
       what CanThreadFieldLoop() tests.

       **NPCReduce() and NPCRecover() are additionally the only two that are
       safe here REGARDLESS of the integrators**, because they evaluate none:
       their bodies are MultInv(), the stored face blocks and dof lookups. So
       MultNL()'s refusal to run Threaded on a problem whose gradient blocks
       cannot be cached does not reach them, and a caller whose problem is
       refused there still gets these two threaded. That is what makes its two shared writes
       safe: the accumulation of the trace row into @a y, and H_f's in
       AssembleHDGGrad(), which both sides of a face add into. E and G need
       nothing, being stored per (face, SIDE).

       **Why colouring works there and not in ComputeH().** The difference is
       the target, not the loop. MultNL() scatters into a Vector, where
       disjoint indices are genuinely independent; ComputeH() scatters into an
       unfinalized SparseMatrix, where two threads on disjoint ROWS still
       collide on @a current_row, the column-pointer scratch and the RowNode
       allocator.

       **Still bit for bit**, by the same two-term argument: colouring changes
       the ORDER in which a face's two elements accumulate, and a + b == b + a
       exactly. It would not survive a trace space whose dofs are shared
       between faces -- an H1_Trace (EDG) one -- where a dof sees more than two
       contributions and associativity would start to matter.

       **What MultNL() needed beyond the colouring**, none of it visible here:
       the integrators it calls had to become thread-safe (MFEM's
       #ifndef MFEM_THREAD_SAFE convention, applied to
       MixedConductionNLFIntegrator and the HDG face integrators), the Mesh
       transformation cache had to stop being shared (TransWorkspace, since
       Mesh keeps one FaceElemTr and one Transformation for the whole mesh),
       and num_local_nl_iters had to become an atomic update.

       **A caller obligation follows and there is no way to check it here.**
       Any integrator the caller installs -- a source term in the potential
       mass, a constraint integrator -- sits on this loop and must be
       thread-safe too. An integrator holding per-point scratch as a plain
       member will race, silently.

       **AND MFEM'S OWN STOCK INTEGRATORS BREAK THAT OBLIGATION, so the
       sentence above is not a theoretical caveat.** `VectorMassIntegrator`,
       `MassIntegrator` and `DiffusionIntegrator` hold `shape`, `te_shape`,
       `vec`, `partelmat` and `mcoeff` as plain members with NO
       `#ifndef MFEM_THREAD_SAFE` guard, where `VectorFEMassIntegrator`,
       `ConvectionIntegrator`, `HyperbolicFormIntegrator`, `RusanovFlux` and
       `FluxFunction` all guard theirs. A caller cannot tell which is which
       without reading the class.

       **`convdiff -p 6 -nl -thr` therefore returns NaN, deterministically at
       two threads and above**, and `hdgperf -n 4 -o 2 -k 0.1 -thr` reproduces
       it on SIXTEEN elements. The corruption is a wrong `A`: at one thread
       every element's flux block is the same constant-coefficient mass
       matrix, and at four threads a handful of elements each get a different
       wrong one. It surfaces as a NaN only because NPC's Newton then solves
       with it -- what the race actually produces is a Jacobian that is not
       the derivative of the residual.

       **What decides whether the shared integrators are reached at all is
       ThreadedLoopEvaluatesIntegrators(), and this paragraph used to name
       `CopyLinearGradBlocks()` instead -- which was wrong in both
       directions.** meq reported the over-refusing half against their own
       solver, and it was then found in this repository's own
       test_darcy_threaded_assembly.cpp, whose nonlinear case the refusal had
       silently stopped: that routine also declines for LocalOpType::PotNL and
       FluxNL, where the local block is PREFACTORED and ConstructGrad()
       deliberately leaves it alone, copying Af_lin_data / Df_lin_data with no
       integrator in sight. Under-refusing was the other half, twice over --
       the residual and local-solve modes reach the same integrators through
       LocalNLOperator and were never asked about, and ConstructGrad()'s face
       branches run whatever the cache did, so a nonlinear face constraint
       raced with `ad_done` satisfied.

       The question that actually separates safe from unsafe is whether the
       loop EVALUATES an integrator, and seven handles are all it can reach:
       @a m_nlfi, @a m_nlfi_u and @a m_nlfi_p per element, @a c_nlfi,
       @a c_nlfi_p and the two boundary_constraint_*_nonlin_integs arrays per
       face. Everything else in that loop is dense linear algebra on blocks
       that are already assembled. Problems 1 and 2 thread correctly and agree
       with serial to every digit because theirs are cached; problem 6's are
       live, because the Burgers face constraint makes the residual
       uncacheable.

       **No small critical section fixes it, and that was measured rather
       than assumed**: serialising `ConstructGrad` in full, or the integrator
       calls, or the dof gather, or the face gather, or any pair of them, each
       still gives the wrong answer; only serialising the WHOLE loop iteration
       restores the serial result. The fix is one integrator instance per
       thread, or thread-safe guards upstream -- not a lock. Until then, treat
       Threaded as unsafe for any configuration whose element loop evaluates
       an integrator you have not read.

       **So the "identical to every digit at every thread count" claim below
       holds for the configurations it was measured on and is NOT general.**

       **This is now REFUSED rather than left to corrupt, and the refusal is
       in MultNL() rather than here.** It belongs here by rights, being a
       property of the mode; it cannot live here because the answer depends on
       @a ad_done, which comes from CopyLinearGradBlocks(), which reads
       A_empty, D_empty, the block array sizes and the residual cache -- none
       of which exist until Assemble() has run. Asked at this point it would
       answer for a route that then fires, which is the trap this class has
       already paid for with CanBatchLinearResidual(). So a caller that sets
       Threaded on an unsafe configuration gets an abort naming this method,
       not a wrong answer. ParMultNL() delegates to MultNL(), so the one guard
       covers the parallel path too.

       **A caller who has read their integrators can say so and be believed:
       SetIntegratorsThreadSafe().** That is the whole escape, and it is a
       promise rather than a setting -- MFEM offers no way to ask an
       integrator whether it is reentrant, its `#ifndef MFEM_THREAD_SAFE`
       convention being a BUILD switch that 67 of the 99 scratch-carrying
       classes do not use at all. So the library can see exactly WHICH objects
       it will evaluate and nothing about whether they are safe. Off by
       default, because the default decides what happens to a caller who never
       read any of this.

       **Measured, on the pedestal problem at (n, k) = (32,1), (48,2), (64,2)
       and (32,3), speedup at 8 threads against the serial mode:**

           NPCResidual      5.6  5.8  5.8  6.1 x
           NPCGradient      2.6  2.8  2.7  3.3 x
           whole NPC step   1.9  1.9  1.9  2.1 x

       NPCGradient lags NPCResidual because it carries the serial scatter --
       40-47% of that call, measured separately; see the NPCResidual group.
       The step number is bounded by that scatter and by the trace solve, which
       is not element-local and does not move here. 1.9-2.1x is against a
       predicted ceiling of about 2.3x, and the answer is identical to every
       digit at every thread count.

       Aborts if the build cannot honour it: MFEM_USE_OPENMP is what makes it
       parallel, and MFEM_THREAD_SAFE is what stops GetElementFaces() keeping
       its scratch in a function-local static. Falling back quietly would
       report a speedup nobody got.

       **AssemblyMode::Batched is the one batched setting that is not
       bit-reproducible run to run on a device, and it is the only one.** Its
       face kernels accumulate D through AtomicAdd from every face of an
       element -- three or four contributions, so the order they arrive in is
       visible in the last bit. Measured on an RTX 2070 SUPER, 4802
       triangles, k = 2, the reduced gradient and both NPC legs compared
       bitwise across four runs in one process: the local factorisation
       batched, the trace assembly batched, and both together all reproduce
       EXACTLY -- including the NPC traversal's own AtomicAdd, which has at
       most two contributions per entry and so is order-free
       (CanBatchNPCTraversal()) -- while this mode differs by 3.5e-16 to
       4.7e-16 relative, with the face constraint linear or not. meq reported
       the variation and suspected the batched LU; the LU is innocent. A
       caller who wants run-to-run bits on a device keeps this mode Serial
       and loses none of the others. */
   void SetAssemblyMode(AssemblyMode mode);

   /// The mode set by SetAssemblyMode().
   AssemblyMode GetAssemblyMode() const { return asm_mode; }

   /** @brief The AssemblyMode every DarcyHybridization constructed AFTER this
       call starts in, process-wide; Serial until changed.

       It exists for the one batched leg that runs before a caller can reach
       SetAssemblyMode(): DarcyForm::EnableHybridization() constructs the
       hybridization and calls Init() in the same breath, and Init() builds
       the flux constraint C (ConstructC()). By the time the caller holds the
       object, C has been assembled face by face on the host -- 0.53 s of meq's
       first assembly on 154k interior faces, single-threaded under a Device.
       Setting the default to Batched first lets that leg take
       HDGNormalTraceJumpBatched() (see CanBatchConstraintAssembly()).

       Validated exactly as SetAssemblyMode() validates, at construction, so a
       default of Threaded aborts in a build that cannot honour it. A test that
       changes it restores it. */
   static void SetDefaultAssemblyMode(AssemblyMode mode);
   /// The mode set by SetDefaultAssemblyMode().
   static AssemblyMode GetDefaultAssemblyMode();

   /** @brief Assert that every nonlinear integrator installed on the forms
       this hybridization reads is safe to evaluate from several threads at
       once, licensing AssemblyMode::Threaded where it would otherwise be
       refused.

       **This is a promise by the caller and the library cannot check it.**
       What is being promised is specific: that concurrent calls to
       AssembleElementVector(), AssembleElementGrad(), AssembleHDGFaceVector()
       and AssembleHDGFaceGrad() on the SAME integrator object do not share
       mutable state.

       **WHICH integrators that covers depends on which loops are threaded,
       and it is wider than it was.** MultNL()'s element loop reaches seven
       handles -- the flux-mass, potential-mass and block nonlinear
       integrators per element, and the two face constraint integrators and
       their two boundary arrays per face -- and evaluates no LINEAR
       integrator at all, so for that loop alone `c_bfi_p` and the flux mass
       BilinearForm are outside the promise whatever they hold. **That is no
       longer the whole story**: DarcyForm::CanThreadAssembly() gates the
       assembly element loops on this same flag, and those loops exist
       precisely to evaluate the LINEAR domain integrators. So a caller
       setting this is promising for both sets, and the earlier version of
       this paragraph -- which reasoned only about MultNL() and concluded a
       linear VectorMassIntegrator "does not count" -- would have licensed a
       race the moment threaded assembly landed.

       The stock integrators the discontinuous path installs are guarded as of
       the commit that put eight of them behind `#ifndef MFEM_THREAD_SAFE`,
       so the promise is true for them today; it is the CALLER's own
       integrators that it is really about.

       **The promise is about every integrator installed here, NOT about the
       loops that happen to be threaded when it is made -- and that asymmetry
       is the whole hazard.** The flag is read by the loop, never by the
       audit, so threading one more loop silently widens what a caller who
       set it years earlier has undertaken. There is no version to check and
       no abort to notice: the first symptom is a wrong answer on a run that
       converges. meq raised this against exactly the right gap -- **and the
       gap then fired.** DarcyForm::ReconstructFluxAndPot() was SERIAL when
       that was written and is threaded under this flag now, so a caller who
       set it for the loops of the day has silently taken on two more
       integrator calls. That is the asymmetry working exactly as described,
       and the defence is to keep the whole reach written down here:

         * **MultNL()'s element loop** -- the seven nonlinear handles above,
           and no linear integrator at all.
         * **DarcyForm::CanThreadAssembly()'s assembly loops** -- the LINEAR
           DOMAIN integrators on the flux mass, the flux divergence and the
           potential mass.
         * **DarcyForm::ReconstructFluxAndPot()'s element loop** -- those same
           domain integrators, reached through the reentrant
           ComputeElementMatrix() overloads, plus two LINEAR FACE integrators
           that nothing else here evaluates concurrently: `c_bfi`'s
           AssembleFaceMatrix() and `c_bfi_p`'s AssembleHDGFaceMatrix(). It
           also evaluates FrozenDualFluxCoefficient from several threads when
           a solution-dependent flux law is lifted onto the enriched space.

       `NormalTraceJumpIntegrator`, which is what `c_bfi` holds in every
       hybridized configuration in this tree, carried six unguarded scratch
       members until it was put behind the same switch on the trunk. The other
       stock integrators those loops install were checked one by one rather
       than assumed, and all of them -- HDGDiffusionIntegrator,
       VectorMassIntegrator, VectorFEMassIntegrator,
       VectorDivergenceIntegrator and the two convection integrators -- wrap
       their scratch in `#ifndef MFEM_THREAD_SAFE`, which AssemblyMode::
       Threaded already requires. It is the CALLER's own integrators the
       promise is really about.

       **DarcyHybridization::ReconstructTotalFlux() is covered too now**, and
       it widened the promise once more -- the same way threading
       DarcyForm::ReconstructFluxAndPot() did, which is why the whole reach
       is kept here rather than at each loop. It reaches `c_bfi`'s
       AssembleFaceMatrix(), `c_bfi_p`/`c_nlfi_p`'s AssembleHDGFaceVector()
       and both boundary-constraint lists', all of which the paragraph above
       already covers, and one thing that paragraph cannot:

       **the CALLER'S OWN FLUX LAW.** ReconstructTotalFlux() takes
       @a ut_fx by value and calls it once per quadrature point from every
       thread. It is a std::function the caller wrote; this class can see
       nothing about it, cannot audit it, and has no way to instantiate one
       per thread. A callback closing over scratch of its own -- a Vector or
       DenseMatrix member, a cached transformation, a lazily-filled table --
       is a race, and setting this flag is the caller saying it is not one.
       Nothing else in this class asks the caller to vouch for a lambda.

       **Why this exists rather than a predicate that decides for you.**
       MFEM has no way to ask an integrator whether it is reentrant. Its
       convention is to wrap scratch in `#ifndef MFEM_THREAD_SAFE`, which is a
       BUILD switch and not a property of the class, and 67 of the 99
       integrator classes that carry scratch do not use it at all --
       VectorMassIntegrator, MassIntegrator and DiffusionIntegrator among them,
       where VectorFEMassIntegrator, ConvectionIntegrator and
       HyperbolicFormIntegrator do. So the library can see exactly WHICH
       objects it will evaluate, which is
       ThreadedLoopEvaluatesIntegrators(), and nothing at all about whether
       they are safe. The caller knows; this is where the caller says so.

       **Measured, on why the refusal is not merely cautious.**
       `convdiff -p 6 -nl -thr` returns NaN deterministically at two threads
       and above, from a flux block that is right on one thread and
       differently wrong on four, and serialising every integrator call does
       not repair it. The honest fix is one integrator instance per thread,
       which MFEM has no facility for.

       Off by default, so nothing that ran before this existed changes. */
   void SetIntegratorsThreadSafe(bool yes = true) { integ_thread_safe = yes; }

   /// The promise made by SetIntegratorsThreadSafe(), false unless made.
   bool GetIntegratorsThreadSafe() const { return integ_thread_safe; }

   /** @brief Whether the element-local HOST loops may run in parallel.

       **AssemblyMode::Batched is not a promise that everything reaches a
       kernel, and the loops it leaves behind used to fall back to SERIAL
       rather than to the loop AssemblyMode::Threaded would have run.** That
       is the whole content of this predicate. The kernels cover one
       configuration at a time -- a genuinely nonlinear element integrand has
       none, and CanBatchLinearResidual() refuses outright for it -- so a
       caller asking for the device gave up host threading on every leg the
       device did not take, silently and by construction.

       **Measured by MEQ, on 4848 triangles at k = 2 with eight threads**,
       which is the configuration that named it: of six legs exactly one
       moved, and the other five fell from 6.81, 6.54, 4.70 and 3.35 cores to
       **1.00**, for 2.14x slower end to end. That is not a throughput result
       and it is not the kernels; it is this test.

       **The two modes ask the same question of the build and a different
       question of the caller.** SetIntegratorsThreadSafe() governs only the
       loops that evaluate the caller's integrators, exactly as before, and
       this predicate is not where that is decided -- CanThreadAssembly(),
       DarcyForm::ReconstructFluxAndPot() and MultNL() each ask for the
       promise themselves, because each knows whether its own loop reaches an
       integrator. What differs is the ANSWER when the promise is missing and
       the loop does reach one: Threaded aborts, because a caller asking for
       it is asking a performance question and a silent downgrade answers it
       wrongly; Batched falls back to the serial loop, because that caller
       asked for kernels and never mentioned threads. Batched's convention is
       silent fallback throughout and this keeps it.

       The build guard is the same for both and is what SetAssemblyMode()
       already aborts Threaded for, so adding it here changes nothing for
       that mode and is the whole of the condition for this one.

       **AND A CONFIGURED DEVICE REFUSES BOTH, WHICH IS MEASURED AND IS NOT
       CAUTION.** MFEM's memory bookkeeping is live only when a non-host
       backend is configured -- with none, a host Vector is plain new/delete
       and there is no map to corrupt -- and it is not thread-safe. Under
       `Device("debug")` at four threads the batched face kernel's own
       [DebugDevice] case dies in
       `std::__throw_out_of_range` from `MmuHostMemorySpace::Dealloc`, thrown
       out of a destructor and so reaching `terminate` rather than an
       assertion. A/B, one line, same binary otherwise: **15 of 15 cases pass
       at one thread and at four with these loops serial, and at four with
       them threaded the run aborts.** MEQ report the same family from a real
       CUDA build with AssemblyMode::Threaded --
       "alias pointer is not registered" out of
       `MemoryManager::CheckHostMemoryType_` -- which is the other end of the
       same map.

       Every threaded loop here builds PER-THREAD scratch inside the parallel
       region, which is the whole reason the regions are written as they are,
       so the allocation cannot simply be hoisted; making the combination work
       is a piece of work in `general/mem_manager.cpp` and not here. Until it
       is done this returns false rather than letting either mode corrupt a
       map, and SetAssemblyMode() says so once for the caller who asked for
       Threaded and is not getting it. */
   bool ThreadHostLoops() const
   {
#if defined(MFEM_USE_OPENMP) && defined(MFEM_THREAD_SAFE)
      if (Device::Allows(Backend::DEVICE_MASK)) { return false; }
      return asm_mode == AssemblyMode::Threaded
             || asm_mode == AssemblyMode::Batched;
#else
      return false;
#endif
   }

   /** @brief How many colours the threaded element loops run as separate
       passes, or 0 if no threaded loop has built the colouring yet.

       Every loop that scatters into the trace -- ReduceRHS(), NPCReduce(),
       MultNL()'s trace row, the gradient's H_f -- runs one OpenMP region and
       barrier per colour, so the count against GetNE() says how much work each
       barrier separates. The loops that write only their own element's dofs
       (NPCRecover(), ComputeSolution()) run as one pass and are untouched by
       it. Asking does not build the colouring: it is a read of what a
       threaded run has already used. */
   int GetNumElementColours() const
   { return colour_offsets.Size() > 0 ? colour_offsets.Size() - 1 : 0; }

   /** @brief Can MultNL()'s element loop evaluate a nonlinear integrator at
       all, for any mode? Equivalently: does AssemblyMode::Threaded need
       SetIntegratorsThreadSafe() on this problem?

       Exactly seven handles can be reached from that loop -- the flux-mass,
       potential-mass and block nonlinear integrators per element, and the two
       face constraint integrators and their two boundary arrays per face --
       and this asks whether any of them is installed. Everything else in the
       loop is dense linear algebra on blocks that are already assembled, and
       a LINEAR integrator is never evaluated there at all, so `c_bfi_p` and
       the flux mass BilinearForm do not count however much scratch they hold.

       It is the worst case over modes and over the state of the linear-block
       cache, which is what a caller deciding on a mode wants: false means
       Threaded is safe here whatever the integrators are, true means the
       promise is needed. The refusal itself is finer -- it also knows the
       mode and whether CopyLinearGradBlocks() succeeded -- and cannot be
       asked before Assemble() has run, which is why this one exists. */
   bool ElementLoopEvaluatesIntegrators() const;

   /** @brief Whether AssemblyMode::Batched's face kernel would actually be
       taken, which is a much narrower question than whether it was asked for.

       It needs the mode, an NPC problem (the kernel writes H into H_data,
       where the reduced route's assembled H is what a non-NPC solve reads), a
       serial constraint space (a shared face is not Mesh::FaceIsInterior()),
       every integrator on the potential-mass constraint to be one
       HDGFaceScatterBatched() implements, one integration rule across the
       face list, and the geometry HDGDiffusionFaceMatricesCanBatch() asks for.

       Measured against the 88 hybridized regression references: taken on 15,
       against 6 when the kernel covered pure diffusion alone. Of the 73 it
       still refuses, 52 carry a NONLINEAR constraint -- c_nlfi_p rather than
       c_bfi_p -- which is not an assembly-time term at all and would need a
       different kernel, over element-face pairs and per Newton step.

       **Ask this rather than inferring it from a timing.** The mode was
       unreachable for every caller in the tree until the SumIntegrator
       DarcyForm wraps around the constraint was looked through, and an
       assembly timing did not reveal that -- the run-to-run scatter is wider
       than what the mode costs. A silent fallback is the normal case here,
       so a caller that cares has to be able to ask. */
   bool CanBatchPotFaceAssembly() const;

   /** @brief Whether ConstructC() assembles the INTERIOR faces' flux
       constraint with HDGNormalTraceJumpBatched().

       It needs AssemblyMode::Batched AT INIT() -- which in practice means
       SetDefaultAssemblyMode(), since DarcyForm::EnableHybridization() calls
       Init() before returning -- and a constraint integrator and spaces that
       HDGNormalTraceJumpCanBatch() takes. Shared faces and the boundary
       constraint integrators keep their per-face loops either way. Asked of
       the spaces and the mesh, so it builds no transformation. */
   bool CanBatchConstraintAssembly() const;

   /** @brief Whether AssembleNLFaceGradBatched() would actually be taken.

       The STATE-CARRYING counterpart of CanBatchPotFaceAssembly(), and a
       different question: that one is about c_bfi_p, assembled once, and this
       one is about c_nlfi_p / c_nlfi, evaluated once per element per Newton
       step. It needs AssemblyMode::Batched, an NPC problem, a serial
       constraint space, a nonlinear face constraint at all, and every
       integrator in it to be one HDGNLFaceGradScatterBatched() weighs.

       **Ask this rather than inferring it from the reference set.** Measured
       over the 152 serial regression references by printing which constraint
       slot EnableHybridization() fills: 17 fill c_nlfi_p, with a
       SumNLFIntegrator of one HDGDiffusionIntegrator and one
       HyperbolicFormIntegrator; 5 fill c_nlfi with NOTHING in it; and none at
       all reach a MixedConductionNLFIntegrator face constraint, because in
       every -nld -hb reference the potential mass form also carries an
       HDGDiffusionIntegrator, which makes M_p or Mnl_p non-null and sends
       EnableHybridization() down a branch that never reads
       Mnl->GetInteriorFaceIntegrators(). */
   bool CanBatchNLFaceGrad() const;

   /** @brief Whether AssembleNLFaceResidualBatched() will apply -- its WHOLE
       condition, because MultNL() skips the element loop's interior face work
       on the strength of it and a later refusal would drop the term. */
   bool CanBatchNLFaceResidual() const;

   /** @brief Whether MultNL()'s element loop can be replaced wholesale by
       LinearResidualBatched() -- see that routine. PUBLIC for the reason
       CanBatchNLFaceGrad() is: a test comparing two configurations has to be
       able to assert that the route it means to exercise was taken. */
   bool CanBatchLinearResidual() const;

   /** @brief Whether NonlinearResidualBatched() applies: the sibling of
       CanBatchLinearResidual() for the case that one REFUSES.

       **Neither of that predicate's two refusals is loosened here and neither
       should be.** It declines a genuinely nonlinear `m_nlfi_p`, because its
       whole arithmetic is a matrix product and there is no matrix; and it
       declines LocalOpType::PotNL, because freezing a mass slot into
       once-assembled blocks is only safe if the GRADIENT freezes with it and
       under PotNL it does not. This predicate asks for exactly those two
       conditions instead, and it is sound for the opposite reason: **nothing
       is frozen.** The caller's integrand is evaluated afresh at every
       residual, as the element loop evaluates the integrator, so there is no
       state to go stale against a live gradient.

       What made the sibling cheap is that PotNL keeps what it needs.
       Finalize() takes `Af_lin_data = Af_data` before InvertA() and
       `Df_lin_data = Df_data` before ConstructGrad() claims Df_data, so the
       unfactored flux mass and the linear potential content are both still
       there -- which is not obvious, since PotNL is the mode that factors A in
       place, and it is the fact the whole route rests on.

       The remaining conditions are CanBatchLinearResidual()'s, unchanged and
       for its reasons: NPC only, serial constraint space, no trace boundary
       condition, no nonlinear FACE constraint, a linear face constraint
       present, uniform block sizes, and the element dof maps built.

       **And one of its own**: the integrator must carry an explicit
       IntegrationRule. The kernel has to quadrature exactly as
       AssembleElementVector() does or the two routes differ by an amount no
       test states, and an integrator that leaves Integrator::GetIntRule()
       null is choosing its rule per element from the element itself. Refused
       rather than guessed. */
   bool CanBatchNonlinearResidual() const;

   /** @brief Whether NonlinearGradientBatched() applies.

       Public for the same reason its sibling is: it is the only way a caller
       can find out whether the route it asked for was TAKEN, and a route that
       falls back silently is one nobody measures. Ask it AFTER the call --
       D is allocated on first use, so asking before reports the question. */
   bool CanBatchNonlinearGradient() const;

   /** @brief The whole local residual with the element-local nonlinearity
       supplied by the CALLER as a device callable.

       This is the sibling CanBatchNonlinearResidual() describes, and it is
       the one route in this file whose integrand MFEM does not own. It is a
       template so that the loop calling @a f can be instantiated in the
       caller's own translation unit -- which for a device caller is the one
       compiled by nvcc, where both @a f and this are visible and neither
       header mentions the other's types.

       **The template is as thin as it could be made and that is deliberate.**
       Only HDGNLResidualQuadrature() is in a header; the gathers, the block
       algebra, the face terms and the scatters are in the source file behind
       NonlinearResidualPrepare() and NonlinearResidualFinish(), so the cost
       of exposing this is eight pointers and four sizes rather than the
       element loop. See HDGNLResidualWork.

       @a f is called as `f(element, point, x, y, u)` and returns the
       integrand at that point; the quadrature weight and the contraction with
       the basis are applied here. See HDGNLResidualQuadrature().

       @returns false, having done nothing, when
       CanBatchNonlinearResidual() declines; the caller then keeps its element
       loop, which is the convention every other batched route here follows. */
   template <typename Integrand>
   bool NonlinearResidualBatched(const Vector &x, const Vector &bu,
                                 const Vector &bp, BlockVector &r_local,
                                 Vector &y, Integrand f) const
   {
      HDGNLResidualWork w;
      if (!NonlinearResidualPrepare(x, bu, bp, w)) { return false; }
      HDGNLResidualQuadrature(w, f);
      NonlinearResidualFinish(r_local, y);
      return true;
   }

   /** @brief Write every element's potential-mass Jacobian block from the
       caller's DERIVATIVE, the counterpart of NonlinearResidualBatched().

       @a g is called exactly as @a f is there, `g(element, point, x, y, u)`,
       and returns @f$\partial f/\partial u@f$ with neither the quadrature
       weight nor the basis applied; see HDGNLGradientQuadrature(). The two
       therefore differ in the integrand alone, which is what lets a caller
       whose law can report its own derivative hold one object and two thin
       lambdas over it.

       **This does NOT produce the reduced operator** -- it writes D, and the
       element loop that follows still factors it, forms the Schur complement
       and assembles or wraps H. What it removes is the per-element
       AssembleElementGrad() call, which for a caller-supplied law is the one
       thing the library cannot do on the device.

       @returns false, having written nothing, when CanBatchNonlinearGradient()
       declines; the caller then keeps the element loop, which is the
       convention every batched route here follows. A caller who gets false
       and goes on to ReducedGradient() with @a ad_external false is correct
       and merely slower. */
   template <typename Integrand>
   bool NonlinearGradientBatched(Integrand g) const
   {
      HDGNLResidualWork w;
      if (!NonlinearGradientPrepare(w)) { return false; }
      HDGNLGradientQuadrature(w, g);
      NonlinearGradientFinish();
      return true;
   }


   /** @brief Tell the hybridization that a coefficient it was entitled to
       treat as fixed has moved.

       A BilinearFormIntegrator on a NONLINEAR mass form is assembled once and
       reused, which is what makes `convdiff -nl` fast and what freezes a
       swept parameter. Call this after changing such a coefficient and before
       the next residual or gradient. See the definition. */
   void InvalidateCoefficientCache();

   /** @brief Whether the state-independent condensation products can be kept
       across a Newton loop; see CondensationCache. */
   bool CanCacheCondensation() const;
   /// Allocate the cache if it is wanted and not built; NULL if it is not.
   CondensationCache *EnsureCondensationCache() const;

   /** @brief Read-only views on one face's assembled constraint blocks, for a
       caller checking one assembly route against another.

       These are the SAME calls MultNL()'s element loop makes, promoted from
       private because comparing two routes' blocks directly is sharper than
       comparing the trace operator they end up in -- and, on a problem whose
       local Schur complement is near singular, it is the only comparison that
       means anything. Measured: the assembled trace operator came back at
       6e+15 to 6e+226 on a pure-hyperbolic constraint at neq = 2, in BOTH
       routes, while the blocks agreed bitwise.

       @a side is 0 for the face's first element and 1 for its second. The
       matrices ALIAS the hybridization's storage rather than copying it, so a
       caller must not assemble into them -- doing exactly that once left the
       hybridization corrupt behind a reconstruction. */
   void GetFaceE(int f, int side, DenseMatrix &E) const
   { GetEFaceMatrix(f, side, E); }
   void GetFaceG(int f, int side, DenseMatrix &G) const
   { GetGFaceMatrix(f, side, G); }
   void GetFaceH(int f, DenseMatrix &H) const { GetHFaceMatrix(f, H); }

   /** @brief The nonlinear face constraint's integrators, sums unwrapped.

       Reported rather than inferred for the same reason
       PotFaceConstraintIntegrators() is: DarcyForm wraps a form's face
       integrators in a SumNLFIntegrator / SumBlockNLFIntegrator
       unconditionally, so a caller cannot see what is inside. At most one of
       the two lists is ever non-empty -- SetConstraintIntegrators() clears
       the other slot. */
   void NLFaceConstraintIntegrators(
      Array<NonlinearFormIntegrator*> &integs,
      Array<BlockNonlinearFormIntegrator*> &bintegs) const;

   /** @brief How many integrators the potential-mass face constraint carries,
       with DarcyForm's SumIntegrator wrapper unwrapped.

       Reported rather than inferred for the same reason
       CanBatchPotFaceAssembly() is: a caller cannot see through the wrapper,
       and one of these numbers being 1 rather than 2 is the difference
       between exercising the batched accumulation and not. */
   int NumPotFaceConstraintIntegrators() const;

   /** @brief Whether the ELEMENT integrator of the NPC local residual is
       evaluated by one batched kernel for the whole mesh instead of one
       integrator call per element.

       False, having done nothing, unless AssemblyMode::Batched is asked for
       and the block nonlinear integrator is one
       HDGMixedConductionResidualBatched() implements -- see there for what
       that means and why a FunctionDiffusionFlux cannot be one.

       **Ask this rather than inferring it**, for the reason
       CanBatchPotFaceAssembly() gives at length: the fallback is silent, it
       is the normal case, and a timing does not separate the two.

       What it does NOT cover, each measured against the 88 hybridized
       references in miniapps/hdg/regress_test/ rather than guessed at:

       * @a m_nlfi_u / @a m_nlfi_p, which reach the residual as
         SumNLFIntegrators of pure BilinearFormIntegrators on 50 of those 88
         -- VectorMassIntegrator, VectorFEMassIntegrator and
         ConservativeConvectionIntegrator. Those element matrices are
         CONSTANT (a BilinearFormIntegrator's AssembleElementMatrix() does not
         see the state), so the residual re-assembles a fixed matrix once per
         element per evaluation in order to multiply by it. Batching that
         wants a kernel per integrator -- the mass one exists
         (HDGElementMassBatched()), the convection one does not -- and is the
         largest remaining piece.
       * @a c_nlfi / @a c_nlfi_p, a state-carrying FACE constraint, on 17 of
         the 88 (HyperbolicFormIntegrator, the -nlc cases). Those are per
         face, not per element, and the flux hierarchy is a separate piece of
         work.
       * A MixedConductionNLFIntegrator over a FunctionDiffusionFlux, on 5 of
         the 88 (`-p 8`). Its conductivity is a host std::function of the
         current potential; there is no device form of it and no per-point
         weight that can be computed ahead of the kernel.

       **What it costs. It is the first kernel of the offload plan measured
       to pay on the HOST -- and only inside NPCResidual; end to end it is
       inside run-to-run scatter, the trace solve being 54-59% of a run.**
       `convdiff -p 2 -nld -dg -hb -npc -nls 3 -gm 0
       -rtol 1e-12 -bam` at 128x128, one thread, direct trace solve, ON and
       OFF interleaved back to back in one binary through an environment gate
       so no rebuild sits between the halves -- time in NPCResidual over its
       three calls:

       | | order 2 | order 3 |
       |---|---|---|
       | per-element integrator | 0.321-0.355 s | 0.529-0.659 s |
       | batched kernel | 0.258-0.297 s | 0.409-0.491 s |
       | ratio, median of 4 pairs | 0.86 | 0.77 |

       Faster in 8 pairs of 8, and the reason is not the batching: the kernel
       is matrix free per point against ONE reference shape table for the
       mesh, where AssembleElementVector() calls CalcShape() per element per
       point and forms each contracted flux through Vector::operator*(). The
       kernel's own share is 0.019 s and 0.038 s per call; the integrator it
       replaces was 0.037 s and 0.075 s per call in a separately instrumented
       build, which is indicative only -- the ratios above are the
       interleaved measurement and the only one that survives this machine's
       drift.

       The HostRead() that follows it is FREE here -- 0.0000 s to the timer's
       resolution -- because with no Device configured it is a no-op. On a
       device it is the transfer the offload plan's gate is about, and it is
       the reason this is a link in a chain rather than a speedup: the
       consumer is still the host element loop. */
   bool CanBatchLocalResidual() const;

   /** @brief Assemble the flux mass block for every element in one batched
       kernel, from @a M_u's DOMAIN integrators, instead of one
       ComputeElementMatrix() and one AssembleFluxMassMatrix() per element.

       False, having done nothing, unless AssemblyMode::Batched is asked for
       and every domain integrator is one HDGElementMassBatched() implements;
       the caller then keeps its element loop.

       It does NOT cover the form's face or boundary integrators. That is
       deliberate and it is the difference from MFEM's AssemblyLevel::ELEMENT,
       which folds them in for a DG space -- DarcyForm routes those itself,
       into the constraint blocks rather than into the element matrix, so
       folding them here would double-count them. */
   bool AssembleFluxMassMatricesBatched(BilinearForm *M_u);

   /// The same for the potential mass block; see the flux one.
   bool AssemblePotMassMatricesBatched(BilinearForm *M_p);

   /** @brief The same for the DIVERGENCE block, from @a B's domain
       integrators.

       Its scatter is the flux mass's mask with the rows unmasked: an
       element's block is (potential dofs) x (hat dofs), a free COLUMN goes to
       Bf and an essential one to Be, and every potential row goes with it. */
   bool AssembleDivMatricesBatched(MixedBilinearForm *B);

   /** @brief Assemble every BOUNDARY-face contribution of @a M_u into the
       flux mass blocks with one kernel, instead of one
       AssembleFluxMassMatrix() per (boundary face, integrator) pair.

       @returns false, having done nothing, unless AssemblyMode::Batched is
       asked for and at least one boundary element is admitted; the caller
       then keeps the loop in DarcyForm::AssembleFluxMassBdrFaces().

       **This is NOT the shape of AssemblePotBdrFaceMatricesBatched(), and the
       difference is forced rather than chosen.** That routine dispatches on
       three known HDG integrator families and evaluates their quadrature on
       the device, because a potential-mass constraint is one of those three.
       A flux-mass boundary face integrator belongs to no family at all:
       there is no BilinearFormIntegrator ON THIS BRANCH whose
       AssembleFaceMatrix() returns the one-sided block of a vector L2 or an
       H(div) flux space, so nothing can be dispatched on. What is batched
       here is therefore the SCATTER -- the mask that splits an element's
       block between Af and Ae -- and the integrator is still evaluated on
       the host, one face at a time, exactly as the loop this replaces does
       it.

       **That qualifier was absent and the sentence was false as a claim
       about the library.** mfem::HDGExtensionIntegrator on
       gf-hdg-subdomains-dev is precisely such a class, MEQ install it on
       every curved and free-boundary problem, and it meets this routine for
       the first time in their integration tree. Being dispatch-free is what
       makes that safe: this pass asks the integrator for an element block
       and nothing else, so an integrator it has never seen needs only to
       satisfy the size MFEM_VERIFY below, which that one does.

       And the host integrand is not the residency problem it looks like.
       This is a BILINEAR form, assembled once, and the blocks it writes are
       written by the kernel and never brought back for it -- which is what
       the paragraph below is about. What a device would save is a setup
       cost, not a per-step one.

       That still buys the thing the loop cost: the block arrays are written
       by a kernel and are never brought to the host in the middle of the flux
       mass group. AssembleFluxMassMatricesBatched() used to end with
       SyncLocalBlocksToHost() for no other reason than that this loop
       followed it on the host.

       The work is grouped BY ELEMENT and each element is one thread, so a
       corner element's two boundary faces are summed by the same thread in
       the loop's own order. That is what makes this bit-for-bit rather than
       round-off, and it is why there are no atomics here where the boundary
       potential kernel needs them.

       It accumulates, and it has to: the element pass has already put the
       domain integrators' block where this adds to it. Assigning does not
       drop the boundary term, it drops everything else -- see
       AssembleFluxMassMatrix(), whose history this is.

       **Measured.** Against the loop, with only this pass switched (an
       environment gate in CanBatchFluxMassBdrFaces(), used as a probe and not
       committed), the assembled NPC trace gradient differs by **0.0 exactly**
       on eight configurations spanning order 0 to 2, quadrilaterals and
       triangles, one and two integrators and split attribute markers; and the
       reduced right-hand side, which is the only reader of Ae, by 0.0 exactly
       on four more. On a HOST there is no time in it and none was expected:
       interleaved A/B, four alternating pairs, `Assemble()` comes out at a
       host/batched ratio of 0.96-1.04 at order 3 on 64x64 quads and 0.90-0.99
       at order 1 on 96x96, with |H| bit-identical to 17 digits in every pair.
       The transfer it removes is a device transfer and there is no device
       configured in a plain host build, so it cannot be timed here at all --
       which is the plan's own gate rather than a disappointment.

       @note Who reaches it: **nothing in this tree but the unit tests.** See
       DarcyForm::AssembleFluxMassBdrFaces(), where that is written out. */
   bool AssembleFluxMassBdrMatricesBatched(BilinearForm *M_u,
                                           int skip_zeros = 0);

   /** @brief Whether AssembleFluxMassBdrMatricesBatched() would actually be
       taken. Ask rather than infer, as for the others.

       Note which refusals this one does NOT need, since every other face
       kernel in this class carries them, and why -- by inspection of what the
       routine names, which is a check anyone can repeat:

       - it names no trace-space object at all, so
         HDGDiffusionFaceMatricesCanBatch()'s conditions (a DG_Interface trace
         collection, one geometry, one order, vdim == 1) are not conditions on
         this pass. Blocks of different sizes are carried by an offset array
         rather than one stride, so a mixed-order or mixed-geometry mesh is
         admitted;
       - it never writes H, so the NPC refusal the two potential face kernels
         carry does not apply. **The reduced route is covered too**, and that
         is where Ae matters, Ae having no reader under NPC at all;
       - it walks BOUNDARY ELEMENTS, which are genuine domain boundary and
         never shared, so ParallelC() is not a reason to refuse -- though
         ParDarcyForm::Assemble() never calls the pass at all, for which see
         DarcyForm::AssembleFluxMassBdrFaces().

       What it does inherit is the periodic-mesh guard, from the same
       Mesh::GetBdrFaceTransformations() null test the loop uses. */
   bool CanBatchFluxMassBdrFaces(BilinearForm *M_u) const;

   /** @brief Whether the batched flux / potential mass assembly would
       actually be taken. Ask rather than infer: both fall back silently, on
       an integrator the kernel does not implement or on a mesh whose elements
       do not all want one integration rule. */
   bool CanBatchFluxMass(BilinearForm *M_u) const
   { return CanBatchElementMass(M_u, fes); }
   bool CanBatchPotMass(BilinearForm *M_p) const
   { return CanBatchElementMass(M_p, fes_p); }
   /// Whether the batched divergence assembly would actually be taken.
   bool CanBatchDiv(MixedBilinearForm *B) const;

   /** @brief Whether AssemblyMode::Batched's BOUNDARY face kernel would
       actually be taken. The same conditions CanBatchPotFaceAssembly() asks,
       of the boundary integrators and the faces their markers admit. */
   bool CanBatchPotBdrFaceAssembly() const;

   /** @brief Make the element-local and face block arrays readable on the
       host; see the private note.

       Public because DarcyForm::AssemblePotHDGFaces() owns the sequencing.
       Its interior and boundary passes may each be a device kernel, and one
       sync after both is right where one after each would push D back to the
       device only to pull it down again.

       **It takes OWNERSHIP for the host, and that is the point** -- it goes
       through HostReadWrite(), not HostRead(). The distinction is a defect
       that cost two device-only wrong answers. A read leaves the host AND
       the device copy marked valid; every element loop downstream of this
       call then writes these blocks through RAW POINTERS, which does not
       invalidate the device side. So from the SECOND pass onward a kernel's
       ReadWrite() found a device copy that was valid and stale, skipped its
       upload, and consumed the previous pass's data -- while every
       first-pass comparison, which is what the unit tests take, was exact.

       Measured under Device("cuda"), two consecutive NPCGradient() calls at
       the same state, against tolerances of 7.5e-12:

       | configuration | the kernel that consumed it | error |
       |---|---|---|
       | AssemblyMode::Batched | AssembleNLFaceGradBatched() | 0.17 to 0.76 |
       | LocalFactorMode::Batched | ComputeElementsHBatched() | up to 3.8 |

       The second is the sharper lesson: it needs neither AssemblyMode nor
       the face-gradient kernel, so fixing that kernel alone left a live
       instance behind, and the two share nothing but this function. D was
       the array instrumented -- 2 on the host and 4 on the device at the
       kernel's own entry, both sides marked valid, coming out at 6 where the
       first pass gave 4, i.e. drifting by its own element contribution per
       evaluation. Invisible on a host build, where there is no second copy.

       Pinned under the debug backend, which sees this where a plain host
       build cannot: "Two gradients in a row are the same operator under a
       device" in tests/unit/miniapps/test_debug_device.cpp, one section per
       lever, each asserting its own is live. Put these calls back to
       HostRead() and both fail at every order.

       The cost is one upload per kernel pass, which is the upload that was
       missing. Ct_data, Ae_data, Bf_data and Be_data are const members and
       stay on HostRead(); see the note at the end of the body for what that
       does and does not claim.

       **@a own says which arrays the host code AFTER the call writes**, and
       only those are taken over; the rest are made readable and their device
       copies left valid. Ownership is what a raw-pointer WRITER needs, and
       the default takes everything, which is always correct. It was also
       always paid: a sync followed only by READERS still invalidated every
       device copy, and the next kernel uploaded arrays nobody had touched.
       meq measured it on the fully batched NPC route (LocalFactorMode::Batched
       with a caller integrand): A's factors, the Schur block and both pivot
       arrays went down and came back up about twice per Newton step,
       ~170 MB up per run. The sites that now pass less than all:

       | site | what follows it on the host | @a own |
       |---|---|---|
       | MultNL(), MultNlMode::GradAtFields | ConstructGrad() per element | what it can write; see there |
       | ComputeH(), after the batched factorisation | readers only | #LB_NONE |
       | ComputeH(), before the HOST factorisation | ComputeElementH() factors | #LB_ALL (new) |

       The last row is new and is what makes the first two safe. Before, the
       host factorisation found its arrays owned because MultNL() had just
       taken everything; now it takes them itself. **The rule this leans on is
       that every raw-pointer writer is preceded by an owning sync of what it
       writes** -- the MultNL() loop by the first row, ComputeElementH() by
       the third, the assembly routines by Reset()'s Vector zeroing and by
       DarcyForm's sync at the end of each assembly group (which takes
       everything, and invalidates a device copy the passes left stale before
       any kernel reads it), InvertA()/InvertD() by running at Finalize().
       A new raw writer has to join that list. Pinned by the debug-device
       cases "Two gradients in a row are the same operator under a device"
       (which fails with the first row's ownership dropped) and "The
       caller-integrand NPC gradient under a device" (which aborts with the
       third row's sync removed), and by "The caller-integrand NPC route
       solves what the element loop solves", which runs the whole route over
       several steps under Device("debug") and, tagged [GPU], on hardware.
       Measured on an RTX 2070 at meq's size (4802 triangles, k = 2, 14
       columns, 12 steps), the three changes of that commit together took
       host-to-device traffic from 669 MB to 271 MB with every output bitwise
       unchanged.

       **@a leave names arrays the call does not touch at all**: neither made
       readable nor taken over, so a device-only copy stays device-only. It
       is for a site where no host code reads them before the next kernel,
       and it moves the duty to sync onto whichever host reader does come.
       One site uses it: ComputeH() after the fully batched gradient leaves
       #LB_FLUX, #LB_POT and #LB_FACTORS, i.e. A's and D's factors and their
       pivots. Their only host readers are the per-element MultInv() and
       MultInvBlocked(), and every loop that calls one now syncs first
       (MultNL() already did; ReduceRHS(), ComputeSolution(), NPCReduce() and
       NPCRecover(), both overloads each, on their per-element branches). On
       the batched branches nothing on the host reads them: the next readers
       are MultInvBatched() and the next factorisation. meq measured the
       readback this removes at 24 MB of 38 MB down over a run (4848
       triangles, k = 2, 12 gradients): D's factors at every gradient and A's
       once. The face blocks, Ct and Bf are still read, because
       ReduceRHS()'s and ReconstructTotalFlux()'s face loops index them raw
       on every route; dropping the whole call faults there under
       Device("debug"). */
   enum LocalBlockSet : unsigned
   {
      LB_NONE    = 0,
      LB_FLUX    = 1u << 0,  ///< Af_data
      LB_POT     = 1u << 1,  ///< Df_data
      LB_FACE    = 1u << 2,  ///< E_data, G_data, H_data
      LB_BNL     = 1u << 3,  ///< Bnl_data
      LB_FACTORS = 1u << 4,  ///< Af_ipiv, Df_ipiv, Sf_data, Sf_ipiv
      LB_ALL     = 0x1fu
   };
   void SyncLocalBlocksToHost(unsigned own = LB_ALL,
                              unsigned leave = LB_NONE) const;

   /** @brief Choose how the element-local blocks A and D are factored.
       LocalFactorMode::Serial by default, so nothing existing changes.

       LocalFactorMode::Batched sends the whole array through
       BatchedLinAlg::LUFactor(). Note what that does and does not buy on a
       host: the native backend is an mfem::forall, so it threads only in a
       build with MFEM_USE_OPENMP *and* a device configured to use it
       (Device("omp")), and is otherwise the same serial traversal reached by
       a different route. Its real payoff is the gpu_blas and magma backends.

       The two modes agree **bit for bit** in a build without LAPACK, and that
       is a fact about the code rather than a hope: BatchedLinAlg's native
       backend calls kernels::LUFactor(), and LUFactors::Factor() *is* that
       same routine when MFEM_USE_LAPACK is undefined -- the same partial
       pivoting, the same 1-based ipiv. With LAPACK, LUFactors::Factor() calls
       getrf_ instead, whose blocked update reassociates the arithmetic, and
       the two then agree only to round-off. A test asserting equality must
       know which build it is in.

       One behaviour differs deliberately. A block whose factorisation meets
       an exact zero pivot aborts here, where the serial loop discards
       LUFactors::Factor()'s return value and carries on into whatever the
       division by it produces.

       **What it is worth, measured, so that nobody has to guess.** In an
       MFEM_USE_OPENMP build with Device("omp"), the factorisation alone
       (NE blocks of n*n, best of five) speeds up like this against the
       serial loop:

           threads      n=8    n=16    n=32    n=64
                 1     0.85    1.08    0.96    1.00
                 2     1.43    1.70    1.83    1.56
                 4     2.45    3.10    3.17    2.76
                 8     3.82    5.56    4.52    4.94

       and the factors agree to the last bit at every thread count, pivots
       included. But the *in situ* difference -- the wall time of
       DarcyForm::Assemble(), whose only difference between the two modes is
       this call -- stayed inside run-to-run scatter at every size tried, from
       nx=24 at order 5 to nx=128 at order 2, with deltas of both signs.

       **The hot path is batched too now**, and the paragraph that used to sit
       here said it was not. InvertA() and InvertD() run once, from
       Finalize(), and only for LocalOpType::PotNL and FluxNL -- so on their
       own they batch a cold path. What runs once per *linearisation* is
       ComputeH(), and both of its element-local halves are batched:
       FactorElementsBatched() factors every A, forms every Schur complement
       and factors those too, and ComputeElementsHBatched() then does every
       element's `C A^-1 C^T` face-PAIR loop as five BatchedLinAlg calls.

       **Only the two together are worth taking, and the factorisation alone
       is a LOSS.** This doxygen used to quote "0.553 s to 0.416 s at order 2,
       n=128" for the factorisation half, and that number is real but it is
       only what leaves ComputeElementH(); it does not count what
       FactorElementsBatched() spends. Counting both, and interleaved rather
       than in blocked halves, the factorisation on its own is 5-9% slower
       than the element loop at every size tried. The face-pair loop is
       1.36-1.59x faster and pays for it. The table is on
       ComputeElementsHBatched(); the lesson is the branch's own -- a
       measurement of one half of a change is not a measurement of the
       change.

       **The setting does more than its name says**, and the name is kept for
       compatibility. Batched also sends the local SOLVES through
       MultInvBatched() -- every element's `M^-1 (bu, bp)` in one batch of
       BatchedLinAlg calls rather than one LUFactors triple per element -- in
       ReduceRHS(), ComputeSolution(), NPCReduce() and NPCRecover(); and it
       gathers the element-blocked right-hand side, and scatters the recovered
       fields, with one Vector::GetSubVector()/SetSubVector() kernel each over
       el_u_dofs / el_p_dofs instead of a per-element host loop.

       **What that last part was worth, measured, because the claim it
       replaces was the opposite.** This doxygen used to record the batched
       route as 5 to 15% *slower* in situ at every size tried, and blamed the
       transfer around it. The transfer was not the cost. Timing the four
       phases of ReduceRHS() separately, order 2 on 64x64 quads under
       `-d cuda`, per call:

           host gather loop   1.11 ms
           MultInvBatched     6.25 ms
           HostRead() back    0.26 ms
           face loop          5.90 ms

       -- the gather was **four times** the copy-back it was supposed to be
       hiding behind. With both loops replaced by kernels the host gather
       falls from 0.75 ms to 0.10 ms and the scatter from 0.80 ms to 0.08 ms,
       and the end-to-end sign flips. Steady state, `-d cpu`, batched against
       serial:

           order 2,  64x64    FormLinearSystem  9.54 -> 8.35 ms
                              RecoverFEMSolution 9.28 -> 8.33 ms
           order 2, 160x160   FormLinearSystem 61.6 -> 54.2 ms
                              RecoverFEMSolution 60.3 -> 54.3 ms
           order 6,  48x48    FormLinearSystem 58.3 -> 71.8 ms
                              RecoverFEMSolution 58.0 -> 72.5 ms

       So: 10-12% faster at order 2 and 24% slower at order 6. The remaining
       loss is the batched dense kernels themselves on large blocks, not the
       plumbing, and it is a different problem from the one that was fixed.

       **A contract that changes with a Device configured.** In this mode the
       recovered fields come back DEVICE-valid: ComputeSolution() and
       NPCRecover() scatter with a kernel and deliberately do not read the
       answer back, which is the point. Reading them on the host is then the
       caller's business, through HostRead() or any Vector operation that
       syncs -- Vector::operator() and GetData() do not. The library propagates
       the write through the BlockVector's aliases so the flags are right; see
       ComputeSolution(). An H(div) flux keeps the host loop, since two
       elements share a dof and an unordered scatter would race.

       CanBatchLocalSolve() answers whether any of it is taken. */
   void SetLocalFactorMode(LocalFactorMode mode);

   /** @brief Whether LocalFactorMode::Batched would actually be taken, which
       needs every element's A block, and every element's D block, to be the
       size of every other's.

       A uniform mesh at a uniform order is **not** enough, and assuming it is
       is the trap here. Af_f_offsets sizes each element's block by counting
       that element's *free* hat dofs, and a hat dof is essential when it
       depends only on ess_flux_tdof_list -- so any problem with essential
       flux dofs gives its boundary elements a smaller block than its interior
       ones on a perfectly uniform mesh. The question is therefore asked of
       the offsets themselves, never of the mesh and the order.

       Valid once Init() has built the offsets. */
   bool CanBatchLocalFactor() const;

   /** @brief True when the flux mass block is factored ONCE, at Finalize(),
       and reused by every later local solve and gradient.

       This is the LocalOpType::PotNL regime, named by what a caller can
       observe rather than by the enum, which stays private. It holds when
       nothing makes the flux row non-linear: no flux-mass nonlinear
       integrator, no non-linear face constraint, and no block nonlinear
       integrator EXCEPT one whose
       BlockNonlinearFormIntegrator::GetBlockRowMask() promises the potential
       row alone.

       It changes no answer -- the general path computes the same Jacobian
       from the same blocks -- so it is a performance property and a test may
       assert it only as such. Measured: the general path costs 5-7% of a
       whole Newton step at order 2 and 20-26% at order 3.

       Valid once Finalize() has run; false before, that being the default. */
   bool FluxMassIsPrefactored() const
   { return lop_type == LocalOpType::PotNL; }

   /** @brief Choose how the element blocks reach the global trace matrix.
       TraceAssemblyMode::Serial is the default and the historical behaviour.

       **This is the stage the offload plan had listed as impossible**, on the
       grounds that a sparse insert cannot be a kernel. What it costs the host
       route is not the insert: it is that `Grad.reset()` throws the matrix
       away every linearisation, so each Newton step rebuilds a linked list of
       one RowNode per nonzero, walks it to count, and walks it again to
       compact. None of that depends on anything but the mesh, and Batched
       does it once.

       **What it changes for the caller.** The assembled matrix is the same
       operator with the same pattern and the same values to the bit, but its
       rows list their columns in a different order, so it is not the same
       matrix object -- see TraceAssemblyMode::Batched.

       **Where it comes back.** The values stay where the kernel wrote them,
       which under a Device is the device. A gradient's pattern is there
       too, copied from the map's device copy (see BuildTraceHPattern()), so
       a solver that takes the matrix on the device (cuDSS) moves nothing
       per linearisation. A consumer on the host syncs for itself, as
       UMFPackSolver, GS and RAP do through HostRead*(). One that indexes
       `I[i]`, `J[k]` or `A[k]` raw through Memory::operator[] reads a stale
       page.

       This note used to say the opposite: that the values were pulled down
       and the pattern never left the host. That stopped being true when
       HDGTraceDiagIdentity() and HDGTraceEliminateRows() replaced the raw
       host passes. The one raw reader left, EliminateTraceTrueDofs() on the
       linear route, faulted under Device("debug") until it synced first.

       CanBatchTraceAssembly() answers whether it is taken. */
   void SetTraceAssemblyMode(TraceAssemblyMode mode);
   /// The current trace assembly mode; see SetTraceAssemblyMode().
   TraceAssemblyMode GetTraceAssemblyMode() const { return tasm_mode; }

   /** @brief Seconds accumulated inside ComputeH() since ResetComputeHTime().

       The gradient leg of an NPC Newton step is the whole of NPCGradient(),
       and ComputeH() is only part of it -- the local factorisation, the
       element Schur complements and the face-pair assembly. A caller timing
       GetGradient() from outside therefore gets a number that BOUNDS
       ComputeH() from above and does not locate it, which is not enough to
       tell a change in the condensation apart from a change in the element
       integrators.

       **Always on rather than behind a build flag, deliberately.** The cost
       is two StopWatch reads per ComputeH() call against a leg measured in
       milliseconds -- meq's own StepProfile makes the same trade and says so
       -- and a flag that has to be turned on is a flag that can be off when
       the measurement is taken, which is how a mode comes to be unreached
       while every test passes. An accumulator that is always running cannot
       be in that position.

       Static, so it needs no object to read and adds no data member to this
       class, which would be a layout change for every translation unit that
       includes mfem.hpp. It is therefore process-wide and NOT thread safe: it counts
       every DarcyHybridization in the process, and a caller timing one solve
       resets it first. Wall clock, not CPU, so it is a share of a wall-clock
       leg and not comparable with a user-time figure. */
   static real_t GetComputeHTime();
   /// Zero the ComputeH() accumulator; see GetComputeHTime().
   static void ResetComputeHTime();
   /// How many times ComputeH() has run since ResetComputeHTime().
   static long GetComputeHCalls();

   /** @brief Seconds accumulated inside NPCReduce() and NPCRecover() since
       ResetNPCTraversalTime().

       **The elimination traversal is the leg a bordered Newton spends most of
       its time in, and it is the one a caller cannot locate from outside.**
       NPCResidual() and NPCGradient() are O(elements) and run once per step;
       these two are O(elements x columns) and run once per right-hand side,
       so their share grows with every border column while the others' does
       not. A caller timing its own Newton step gets a total that bounds them
       and does not separate them -- the same argument GetComputeHTime() makes
       for ComputeH() inside the gradient leg.

       This is the sum of GetNPCReduceTime() and GetNPCRecoverTime(). It was
       once argued here that one accumulator was enough, because nothing would
       need to compare a reduce with a recover. **That was wrong, and it is
       withdrawn.** The two are threaded differently. NPCReduce() scatters
       into the trace, so it runs the colouring's passes, one region and
       barrier per colour (see GetNumElementColours()). NPCRecover() writes
       only its element's own dofs and runs as one static loop. A bordered
       Newton measuring the leg at 3.3 to 4.2 cores of 8 could not tell which
       half was flat without the split.

       Always on rather than behind a build flag, for GetComputeHTime()'s
       reason -- a flag that has to be turned on is a flag that can be off
       when the measurement is taken. Two clock reads against a leg measured
       in milliseconds.

       Static, so it adds no data member and no layout change; process-wide,
       NOT thread safe, and wall clock rather than CPU. A caller timing one
       solve resets it first. */
   static real_t GetNPCTraversalTime();
   /// The NPCReduce() half of GetNPCTraversalTime(), both overloads.
   static real_t GetNPCReduceTime();
   /// The NPCRecover() half of GetNPCTraversalTime(), both overloads.
   static real_t GetNPCRecoverTime();
   /// Zero the traversal accumulators, both halves; see GetNPCTraversalTime().
   static void ResetNPCTraversalTime();
   /// How many NPCReduce()/NPCRecover() calls since ResetNPCTraversalTime().
   static long GetNPCTraversalCalls();

   /** @brief How many times the batched factorisation has SOLVED for
       A^-1(-/+B^T) rather than replaying it from the condensation cache.

       **A cache cannot be told from a recomputation by its answer** -- that
       is what a cache is -- so counting is the only way a caller, or a test,
       can establish that the replay fires at all. Over N gradients of one
       Newton loop under LocalOpType::PotNL this reads 1 and not N.

       Static, over a function-local accumulator in the .cpp, for the reason
       GetComputeHTime()'s is: a data member here would be a class-layout
       change every translation unit including mfem.hpp would see. Always on;
       it is one increment against a batched triangular solve. */
   static long GetBatchedAiBtSolves();

   /** @brief How many CHUNKS the batched face-pair route has filled the cache
       for, rather than replaying. Over N gradients of one Newton loop this
       reads the chunk count once, not N times.

       The face half is the larger one: at na=12, nd=6, nc=3, nf=3 it is 3888
       of the ~4770 multiply-adds an element's condensation costs. */
   static long GetBatchedFaceFills();

   /// Zero both counts; see GetBatchedAiBtSolves() and GetBatchedFaceFills().
   static void ResetBatchedCacheCounts();

   /** @brief Whether TraceAssemblyMode::Batched would actually be taken.

       Two questions, and the second is easy to miss. The CONNECTIVITY has to
       suit: one face count per element, one dof count per face, every trace
       dof on exactly one face, and no shared faces -- asked of the trace
       space and the mesh, never of the order, and BuildTraceHMap() says what
       each condition is for. And the DESTINATION has to be free, which on a
       problem that is not under NPC it is not: the face constraint assembles
       its per-face diagonal blocks straight into the sparse H during
       Assemble(), so ComputeH() inherits a matrix rather than building one.
       **So this mode reaches NPC problems and not the reduced route** -- the
       offload plan's item 9, arriving at a second kernel.

       A first version of this predicate asked only the first question and
       returned true for a linear problem whose assembly then took the host
       route. Two of this mode's own tests passed against a DELIBERATELY
       BROKEN kernel because of it, and that is how it was found -- not by
       reading the code. The branch's note is exact: a flag is not coverage,
       and a lever has to be asserted live by something that would fail.

       Valid once Init() has built the constraint space. */
   bool CanBatchTraceAssembly() const;

   /** @brief Whether the local SOLVES are batched too, not only the
       factorisation.

       LocalFactorMode::Batched asked for, uniform blocks, and a stored Schur
       complement -- which the FullNL local operator has not got, its local
       problem going through MultInvNL() and a local Newton instead. When this
       is true, ReduceRHS() and ComputeSolution() do every element's local
       solve in one batch of BatchedLinAlg calls rather than one LUFactors
       triple per element.

       **What it is worth, and the honest answer is "nothing yet".** Two
       measurements, and the second is the one that matters.

       The local solve ON ITS OWN, batched against the LUFactors loop, on
       synthetic blocks of a hybridization's shape (n_a = dim*ND flux dofs,
       n_d = ND potential):

           n_d   NE      host (g++)   CUDA
             9   2304        1.04x    0.67x
            25   2304        0.71x    1.35x
            49   2304        0.72x    2.64x
            49   9216        0.69x    3.53x

       So on a host it is level at small blocks and about 30% slower at large
       ones -- same scalar work through the same kernels, but streaming the
       whole blocked arrays six times where the per-element route keeps one
       element's vectors in cache. On a device it crosses over around a
       thousand elements and reaches 3.5x.

       IN SITU it **used to be** 5 to 15% slower at every size tried, and that
       was blamed on the transfer around it. It was not the transfer: the
       gather and the scatter were per-element host loops through
       Vector::GetSubVector(real_t*), which begins with HostRead(), and they
       cost four times what the one copy-back did. With both replaced by
       kernels the in-situ figure is 10-12% FASTER at order 2 and 24% slower
       at order 6; the tables and the phase split are on SetLocalFactorMode().

       **The standing gate still stands and still bites**: nothing on the
       device path pays until the whole chain is device-resident, and the
       face loops around these solves are host dense work -- so one
       HostRead() per call remains wherever a face loop follows. Removing
       that is the face kernels' business (HDGFaceScatterBatched and
       HDGFaceScatterCanBatch in bilininteg_hdg.hpp), not this setting's --
       and for the NPC traversal it is done: see CanBatchNPCTraversal(),
       which this setting is a precondition of.
       What this setting buys is that the local factorisation and solve
       are EXPRESSIBLE on a device at all, which the whole-chain target
       requires and which they were not before -- see the two upstream defects
       the attempt turned up, recorded on GPUBlasBatchedLinAlg::AddMult and
       NativeBatchedLinAlg::LUSolve. Hence Serial by default. */
   bool CanBatchLocalSolve() const;

   /** @brief Whether NPCReduce() and NPCRecover() -- both overloads -- run
       their FACE loops as kernels as well as their local solves, so that on
       a device nothing in either routine comes back to the host.

       **Before this, CanBatchLocalSolve() batched the element solves and the
       face loop around them stayed host dense work**, so the single-column
       NPCReduce() read its batched answer straight back with HostRead() and
       NPCRecover() built its batched right-hand side on the host and pushed
       it up. And the BLOCKED overloads -- the ones DarcyNPCSolver::ArrayMult()
       and so a bordered Newton spend their time in -- did not batch at all:
       they called MultInvBlocked() per element on the host whatever the
       factor mode said. A caller who asked CanBatchLocalSolve() and read yes
       was told about a route its own bordered step never took.

       The kernels are element-major, and neither loop needs a colouring:

       - NPCRecover() only GATHERS the trace, and every write is at the
         calling element's own field dofs.
       - NPCReduce() SCATTERS into the trace, through the same AtomicAdd as
         HDGScatterTraceRows(), and for the same reason it costs no
         determinism: a trace dof belongs to one face, a face has at most two
         elements, the accumulator is zeroed first, and IEEE addition of two
         terms onto zero is commutative to the bit. A face-centric traversal
         would remove the atomic; it is not needed to remove the race.

       Every product is written in the loop order of the host DenseMatrix
       routine it replaces (kernels::MultTranspose, DenseMatrix::AddMult), so
       on a host backend the single-column route is BIT-FOR-BIT the host face
       loop around MultInvBatched(), which is what the tests assert. Under a
       Device only round-off agreement is claimed, for the reason
       MultInvBatched() gives.

       Requires, beyond CanBatchLocalSolve():

       - both field spaces discontinuous, so the gathers and the Recover
         scatter are over element-owned dofs (an H(div) flux keeps the host
         loops, and is not owed -- see the branch's scope);
       - one face count per element and one trace size per face
         (BuildElementDofMaps()'s trace map), and no nonconforming master
         face, which the other element-major kernels decline for the same
         reason;
       - Ct, and E and G where present, stored at their offsets' full size;
       - and NOT ThreadHostLoops(). On a host with OpenMP threading asked for,
         the OpenMP loops are kept: mfem::forall is a serial loop on a host
         backend, so taking the kernels there would give the threading up.
         On a device ThreadHostLoops() is false by construction. */
   bool CanBatchNPCTraversal() const;

   /** @brief Choose whether GetGradient() assembles the reduced system or only
       applies it. See GradientMode; the default is Assembled, which is what
       every caller written before this existed gets.

       GradientMode::MatrixFree returns an Operator with no stored matrix, so a
       caller must solve with something that needs only the action -- an
       unpreconditioned Krylov method, or one preconditioned by something not
       built from the matrix. GSSmoother, UMFPackSolver and the algebraic
       preconditioners all require a SparseMatrix and will abort.

       Supported in every LocalOpType. It was not: with only the flux mass
       nonlinear (LocalOpType::FluxNL) the Schur complement had nowhere to
       live, @a Df_data being occupied by the factored linear potential mass,
       and GetGradient() aborted rather than return something wrong. It now
       goes to @a Sf_data.

       **"Matrix free" is about the GLOBAL trace matrix, not the Jacobian.**
       The local blocks are still assembled and factored in every mode -- the
       action of S is defined in terms of them -- so no mode here is
       Jacobian-free in the stronger sense of forming no Jacobian at all.

       **What the mode costs, measured rather than asserted.** convdiff
       -p 2 -o 2 -dg -hb -up -nlu -nls 3, wall seconds, all three modes
       agreeing to six digits in both error norms:

       | nx | Assembled + UMFPack | Assembled + GS | MatrixFree |
       | -- | ------------------- | -------------- | ---------- |
       | 10 | 0.10 | 0.10 | 0.10 |
       | 20 | 0.20 | 0.10 | 0.40 |
       | 40 | 0.50 | 0.50 | 4.00 |

       So it runs and it is 8x at nx = 40, growing with the problem, because
       nothing preconditions it: with no assembled S there is no Gauss-Seidel,
       no AMG and no factorisation, and the obvious block-Jacobi replacement
       costs the same local solves that assembly costs. MatrixFree is for a
       caller who cannot afford the memory, or who brings a preconditioner not
       built from S. See doc/HDG-JACOBIAN-FREE-TRACE.md, which is where that
       open question lives. */
   void SetGradientMode(GradientMode mode);

   /** @brief Keep what NPC reads, on a form that carries no nonlinear
       integrator at all.

       Without this a linear DarcyForm cannot use the NPC pathway, and fails
       by segfault rather than by refusal. Finalize() takes a route for the
       linear case that factors each element's A and D in place and keeps no
       copy, and routes the face H into the global sparse H instead of the
       element-wise H_data -- both correct when the only thing that will ever
       be asked is one reduced solve, and both fatal to NPC, which evaluates
       the residual and the gradient at ARBITRARY states and so needs the
       blocks rather than their factorisations.

       A linear form wants NPC for the reason any DAE integrator does: the
       condensation route iterates on the trace alone, so the vector it
       iterates on is not the vector the integrator integrates and there is no
       residual over the full (q, u, lambda) state to hand it. That is as true
       of a linear problem as of a nonlinear one, and an integrator evaluating
       at predictor states or finite-difference probes needs it just as much.

       Opt-in rather than unconditional because the retained blocks are the
       largest thing the hybridization owns and no reduced-route caller should
       pay for them.

       Call it any time before Assemble(); it allocates H itself if Init() has
       already run, which it normally has -- the hybridization does not exist
       until DarcyForm::EnableHybridization() has made it, and that is what
       calls Init().

       @note This forecloses the reduced route on the same assembly:
       DarcyForm::FormLinearSystem() has no reduced H to hand back and aborts.
       They are different methods; comparing them needs two assemblies. */
   /** @brief Register a load assembled on the skeleton, in L-dofs of the
       constraint space, or null to clear it.

       Callers do not normally reach for this: DarcyForm::Assemble() registers
       DarcyForm::GetTraceRHS() here. It is public because a caller driving
       this class without a DarcyForm has no other way in.

       The vector is BORROWED and must outlive the next ReduceRHS() or
       NPCResidual(). */
   void SetTraceRHS(const Vector *b_tr_load) { trace_rhs = b_tr_load; }

   /// The registered skeleton load, or null.
   const Vector *GetTraceRHS() const { return trace_rhs; }

   /** @brief Add @f$P^T b_\lambda@f$ of the registered skeleton load into a
       reduced (true-dof) trace vector. Does nothing if none is registered. */
   void AddTraceRHS(Vector &b_tr, real_t a = 1.0) const;

   void EnableNPC();

   /** @name The NPC method: Newton on the full (q, u, lambda) system

       Nguyen, Peraire & Cockburn, JCP 228 (2009) 8841-8855, eqs (14)-(18).
       These four calls are one Newton step of it, exposed raw so that the
       shape of the method is visible. **For ordinary use take
       DarcyNPCOperator and DarcyNPCSolver instead**, which wrap them as an
       MFEM Operator over the full (q, u, lambda) vector that NewtonSolver
       drives with no special support.

       What cannot be wrapped is an Operator on the TRACE ALONE: the flux and
       the potential are Newton state here, and a trace-only operator has
       nowhere to put them. A mode that tried -- an NLOrdering enum whose
       LineariseThenCondense claimed to be this method -- is deleted; it was a
       condensation in disguise, measurably slower than the condensation it
       was meant to beat and unable to solve problems that one solves.

       One step, given a state (@a x, @a x_tr) and the load @a b:

           NPCResidual (b, x, x_tr, r, r_tr)   F(q, u, lambda)
           S = NPCGradient (x, x_tr)           factor J; S is the reduced
                                               H - C' M^-1 [C; E], assembled
                                               or matrix-free
           NPCReduce   (r, r_tr, b_tr)         -(F_lambda - C' M^-1 F_local)
           solve S dtr = b_tr                  the caller's linear solver
           NPCRecover  (r, dtr, dx)            -M^-1 (F_local + [C; E] dtr)
           x += dx;  x_tr += dtr

       That is **one local factorisation and one local linear solve per outer
       step**, no local nonlinear iteration anywhere, and therefore nothing to
       globalise locally. The convergence test belongs on the full residual
       (@a r together with @a r_tr), which is the other half of what makes it
       NPC: a test on the trace residual alone is judging half the system.

       Call NPCGradient() before NPCReduce() and NPCRecover(): both need the
       factored local blocks and the Schur complement it leaves behind, and
       both apply the Jacobian's (0,1) block rather than the linear one.

       **Where the step's time goes, and what that means for threading.** The
       four calls split into loops that reach integrators and the Mesh
       transformation cache (NPCResidual, through LocalResidual(); NPCGradient's
       first pass, through ConstructGrad()) and loops that are dense linear
       algebra alone (NPCReduce and NPCRecover, both only MultInv()). That
       separation is real -- the deleted trace-only mode fused the two inside a
       local Newton iteration and could not offer it -- but it is worth less
       than it looks. Measured over six steps on the pedestal problem at
       (n, k) = (32,1), (48,2), (64,2) and (32,3), one thread:

           integrator-bound loops     60.8  59.7  59.4  63.2 %
           integrator-free loops       6.2   5.4   5.5   5.7 %
           trace solve                33.0  34.8  35.1  31.1 %

       So on THESE cases the two integrator-free loops are **under 6% of the
       step**, flat in mesh size and order.

       **That 6% was read as a reason not to thread them, and it was the wrong
       reading -- they are threaded now.** The figure is a property of the
       configuration it was taken on, not of the method: all four cases are
       fixed-boundary with ONE right-hand side. NPCResidual() and
       NPCGradient() are O(elements) and run once per Newton step, while
       NPCReduce() and NPCRecover() are O(elements x columns) and run once per
       right-hand side -- so the moment a caller has a border, the ratio
       inverts, and it inverts further with every border column. meq's
       free-boundary Grad-Shafranov applies one factorisation to `N + 4`
       columns (14 in their shipped DIII-D case) through
       DarcyNPCSolver::ArrayMult(), and measured the two loops at **30.6% of
       their Newton step and 1.00x across eight threads** -- the largest
       single leg, ahead of the residual. Amdahl did not cap the gain; the
       measurement had been taken where the gain was not.

       **Both are threaded, and they needed different things.** NPCRecover()
       writes only the calling element's L2 flux and potential dofs, so it
       takes CanThreadFieldLoop() and needs neither colouring nor atomics.
       NPCReduce() scatters into the TRACE, which the two elements of a face
       share, so it takes the element colouring exactly as ReduceRHS() does --
       and having it, is safe whatever the flux space is, since it only READS
       the field dofs. Neither evaluates an integrator or builds an
       ElementTransformation, which is why they are gated on asm_mode alone
       and are unaffected by MultNL()'s refusal of Threaded on a problem whose
       gradient blocks cannot be cached.

       Measured here, `hdgperf -n 128 -o 3 -lin -gm 0`, one binary with the
       two loops gated serial against threaded, MKL_NUM_THREADS=1, seconds in
       GetNPCTraversalTime():

           threads        1      2      4      8
           before      0.199  0.201  0.178  0.200     flat, i.e. meq's 1.00x
           after       0.218  0.117  0.074  0.046     4.35x at eight

       The before row reproduces meq's observation independently. At one
       thread the after row is ~10% slower -- the parallel region is entered
       and left either way -- which is the price of the `if (threaded)` clause
       being a runtime test rather than a compile-time one, and is paid only
       by a caller who asked for Threaded and then ran on one core.

       **That one-thread tax is a property of the region and not of this
       fixture, and it is worth reading before the 4.35x is.** meq re-took the
       whole pair on their own free-boundary DIII-D problem, in their own
       tree, against a binary preserved from before the upgrade: the leg goes
       1.375 -> 1.446 s at OMP = MKL = 1, **0.95x**, and the whole run 6.983
       -> 7.121 s, **0.98x** -- the same sign and the same size, on a
       different problem. So a SERIAL deployment of a caller that asks for
       AssemblyMode::Threaded is very slightly worse off than before, and a
       reader who takes only the eight-thread row away will not infer it.

       Their threaded arm is the other half of the same check and landed where
       the arithmetic said: **2.61x on the leg and 1.25x on the run against a
       1.24x prediction**, with the solution identical in all twelve runs. The
       leg factor is 2.61x there against 4.35x here because their call is
       twelve calls of fourteen columns and this one is not -- which is the
       thing to carry: **a leg speedup quoted from another fixture does not
       transfer, and the run figure landed only because the prediction rested
       on meq's own measured share.**

       **NPCGradient's column is three parts, not the two this used to name.**
       It said the column was ConstructGrad (integrators, serial) plus
       ComputeElementH (dense, already threaded by AssemblyMode::Threaded) and
       that separating them was the next measurement. Taken, in a build with
       MFEM_USE_OPENMP and MFEM_THREAD_SAFE, on the same four cases:
       GradientMode::MatrixFree runs the same threaded ComputeElementH and
       skips the scatter, so the mode difference *is* the scatter, and a thread
       fit on MatrixFree separates the other two. Shares of NPCGradient:

           ConstructGrad (integrators, serial)   58  50  50  45 %
           ComputeElementH (already threaded)     2   6   4  12 %
           scatter into the SparseMatrix         40  43  47  43 %

       **The part nobody had counted is the scatter, and it is serial by
       design** -- see SetAssemblyMode(), which explains why a SparseMatrix
       target cannot be threaded even with element colouring. The
       already-threaded half is 2-12% of the column, so the answer to the
       question this used to pose is that ComputeElementH does *not* dominate
       and the integrator-bound share is essentially as the table above says.

       **End to end, AssemblyMode::Threaded buys 1.4 to 7.7% of an NPC step**
       (1/2/4/8 threads, best of; the answer is bit-identical at every thread
       count and in both assembly modes, to every digit of the solution norm
       and the final residual). Recomposing a step with the scatter taken out
       of NPCGradient's column: integrator-bound 46-53%, threadable with no
       new infrastructure 7-10%, serial assembly 12-17%, trace solve 26-31%.
       So the ceiling on threading an NPC step -- with perfect integrator
       thread-safety and a perfectly threaded ComputeElementH -- is about 2.3x,
       and the two items outside it are the scatter and the trace solve.

       One consequence for GradientMode::MatrixFree, which is not only a memory
       play: it deletes 40-47% of NPCGradient. What it pays for that is an
       unpreconditioned trace solve, so it still loses overall -- but the prize
       behind "what preconditions a never-assembled trace system" is larger
       than a memory argument makes it look. doc/HDG-JACOBIAN-FREE-TRACE.md and
       doc/HDG-ELEMENT-LOCAL-PARALLELISM.md carry those two open questions.

       **What it delivers, measured.** On a problem whose full system is linear
       one step is exact -- the residual goes 6.96e-01 to 6.22e-15, from any
       starting point -- which is the check that falsifies the elimination
       algebra if anything in it is wrong. On the pedestal source it converges
       quadratically in the full residual: 6.7e-01, 1.5e-02, 2.8e-04, 1.2e-07,
       2.3e-14, with GetNumLocalNLIterations() identically zero. The two
       gradient modes agree at every iterate above round-off.

       And it solves stiff problems the deleted trace-only mode could not. Of
       the four configurations where the reduced trace operator converges and
       that mode did not, three fall to NPC with a backtracking line search on
       the full residual -- 13, 10 and 17 steps -- and the fourth stalls at
       2.9e-03, which is ordinary Newton stagnation. Undamped, NPC converges
       one of the four (k = 3, n = 12, in 12 steps) and wanders on the other
       three as any cold Newton does -- an earlier version of this paragraph
       said all four, which a sweep of the four configurations disproved: **the globalisation this method wants
       is on the OUTER step and there is none to do locally**, which is the
       whole point of the ordering.

       **Two of those three counts are properties of the method and the third
       was a draw, and nothing knew which until they were perturbed.** Over
       MKL_NUM_THREADS in {1,2,4,8} crossed with a perturbation of the initial
       trace in {0, +-1e-13, +1e-12} -- fifteen orders below the state -- the
       k = 2, n = 8 configuration takes 13 steps on every arm and k = 3,
       n = 12 takes 10 on every arm. The third, k = 1, n = 32 at a pedestal
       width of 0.003, takes 18, 22, 25, 28 or 40: the 17 recorded above was
       one draw from that. It is on the convergence BOUNDARY -- width <= 0.0025
       stalls on every arm, width >= 0.004 converges in five or six on every
       arm -- and 0.0032 restores 17 steps on all sixteen arms, which is the
       width the unit cases pin.

       **That is not a defect in this operator, and the check that says so is
       worth reusing.** Normalise the computed step to unit length and
       finite-difference NPCResidual() along it: the elimination claims
       J.delta = -r, so the true Jacobian must map the unit d to -r/|delta|.
       It does, to six to eight digits at every healthy iterate, so the
       gradient and the elimination algebra are right. What the same
       measurement then shows is that |J d| falls to 1.3e-09 while |d| = 1 --
       the Jacobian is going SINGULAR along the direction Newton is heading
       in. The problem has several roots (states of norm 44.923, 44.922 and
       44.827 have all been reached from one start), the iterate approaches a
       fold between them, and the residual freezes to seven digits while the
       step length doubles each iteration to 1.5e+06. The condensation route
       reaches a root there in 7 to 9 steps and NPCResidual() sees that root at
       1.8e-12, so the two routes never disagreed about the discrete problem.

       **A step-norm cap was built and rejected, which is worth knowing before
       building it again.** Bounding |dx| at 10 converges in 15 steps on all
       twelve arms of the perturbation sweep, looking exactly like a trust
       region earning its keep. Sweeping the cap kills it: 10 works, 15 and 30
       do not, and any cap BREAKS the two configurations that converge without
       one. The cap does not globalise anything near a fold -- it just reshuffles
       which trajectory round-off selects, so it is one more draw.

       The operational consequence outlives the test: **a configuration on this
       boundary fails only when something pins the BLAS thread count**, because
       UMFPack's reduction order is where the round-off enters. Three unit
       cases sitting on it were green for months and failed the first time a
       run set MKL_NUM_THREADS=1, with no library change behind it.

       **Do not read that as a general recommendation of a line search.** The
       backtracking above is an l2 merit over all three blocks. Where the
       nonlinearity sits in the potential block and the flux and trace rows are
       linear -- the shape of every DarcyForm problem whose nonlinearity is in
       Mnl_p, and measured true here: a full step takes the flux row to 4e-17
       and the trace row to 3e-14 -- a full step is exactly optimal for two
       blocks and any damping restores part of them. Whether backtracking then
       helps or hurts turns on how badly the full step treats the potential
       block: here it improves it and the search helps, on meq's
       Grad-Shafranov discretisation it multiplies it by 77 and the search made
       every case worse.

       **Do not write a line search to fix that.** KINSolver(KIN_LINESEARCH) is
       KINSOL's Dennis & Schnabel search, with the sufficient-decrease and
       curvature conditions and a minimum-step test, and meq report it failing
       on exactly the same cases -- so a correct backtracking implementation
       does not rescue this, and a block-weighted merit does not either:
       accepting a full step on meq's published residuals needs the LINEAR flux
       row weighted 79x above the nonlinear potential row, which is a merit
       that a Newton step annihilates by construction and hence no line search
       at all. What the evidence points at is non-monotonicity -- undamped
       Newton converges cases every monotone search kills -- for which KINSOL
       offers Anderson-accelerated KIN_PICARD / KIN_FP. See section 6 of
       doc/HDG-ORDERING-API.md.

       A line search here is well defined for a
       reason worth keeping in view -- the fields and the trace scale together
       because both are state, where a line search on a trace-only operator
       scales the trace and leaves the field update to whatever the
       substitution makes of it.

       @note **The flux space must be discontinuous, and the reason is
       representational rather than a matter of sign conventions -- an earlier
       version of this note said it was, and had not been measured.** The state
       NPC iterates on is the BROKEN one: each element owns its own copy of the
       flux dofs on a shared face, and the trace row is what makes the two
       copies agree. A conforming H(div) space has no room for that -- summing
       the element vdof counts of an RT space gives 192 against a space size of
       144 on a 4x4 quad mesh at order 1, one dof short per interior face -- so
       both elements read the SAME value, their Ct blocks carry opposite signs,
       and their contributions cancel identically. Measured on RT at five
       random states of (q, u, lambda): |C' q| = 3.2e-16 against a flux row of
       9.8, where the same problem gives 7.9-9.0 with an L2 flux and 7.1-8.0
       with a broken-RT one. **So the trace row is annihilated for every
       conforming state, not merely for the ones an iteration visits**, lambda
       is never driven, and NPC stalls: on a 6x6 triangle mesh it stops at
       |F| = 0.0804 and lands 12% off in the flux and 44% off in the trace,
       while the reduced route on the same problem converges in 6 steps.

       The route to an H(div)-shaped discretisation under NPC is therefore
       BrokenRT_FECollection, which is already admitted -- its GetContType()
       is DISCONTINUOUS, its element vdof counts sum to its space size, and it
       carries the same RT element on a space with room for the broken state.
       Measured on the problem above: quadratic convergence in 7 steps to the
       same potential and the same trace as conforming RT's reduced route,
       3.08934 and 3.31561 both ways. Lifting the guard would need the flux
       unknown carried in the hat space instead, which is a different operator
       size and a different caller contract, not a bug fix.

       @note **A load assembled on the TRACE has no slot**, here or in the
       reduced route: DarcyForm offers GetFluxRHS() and GetPotentialRHS() and
       nothing for the skeleton, and @a b is (flux, potential). The caller
       carries it instead, and where it goes is measured rather than inferred:
       subtract it from @a r_tr between NPCResidual() and NPCReduce(), which is
       the same r = F(x) - b convention NewtonSolver::Mult(b, x) applies on the
       reduced route. Checked against that route at three load scales spanning
       20x, |lambda| running 4.69 to 64.2 against 3.31 unloaded: the two agree
       to 1.2e-13 - 2.9e-12 relative.

       @note **A parity test against the reduced trace operator must run on a
       resolved mesh.** The two routes reach the same discrete solution, and
       the tests here assert it -- but on an UNDER-resolved mesh a semilinear
       source can give the discrete system more than one solution, and which
       one a solve lands on is then a property of the route rather than of the
       discretisation. Reported by meq on a Grad-Shafranov benchmark: three
       fully converged solves of the same system to rel_tol 1e-12 gave
       max psi_h of 3.1831e-01 (NPC), 3.4779e-01 (reduced operator) and
       3.1514e-01 (NPC after Anderson-accelerated Picard) -- a spread of 9.4%,
       collapsing under one refinement. So a disagreement between the two
       orderings on a coarse mesh is not evidence that either is wrong, and a
       test that asserts agreement to a tight tolerance has to be posed where
       both converge quickly or it measures this instead. */
   ///@{
   /** @brief The residual of the full system at the given state: no local
       solve, no substitution, no linearisation.

       @a r's potential block carries the sign convention of the symmetrized
       system when that is in force, which is what NPCReduce() and
       NPCRecover() consume; its norm is unaffected and nothing else should
       read it. */
   void NPCResidual(const BlockVector &b, const BlockVector &x,
                    const Vector &x_tr, BlockVector &r, Vector &r_tr);

   /** @brief The same residual with the element-local nonlinearity supplied
       by the CALLER as a device callable, falling back to the element loop.

       This is the entry point a caller with a device integrand actually uses:
       it is NPCResidual() with NonlinearResidualBatched() tried first, so the
       prolongation, the state, the skeleton load and the essential-row
       convention are shared line for line rather than restated. When
       CanBatchNonlinearResidual() declines -- which is most configurations --
       this IS NPCResidual(), and a caller can hold one call site either way.

       @a f is called as `f(element, point, x, y, u)`; see
       HDGNLResidualQuadrature() for what it must return and why it is taken
       by value. */
   template <typename Integrand>
   void NPCResidual(const BlockVector &b, const BlockVector &x,
                    const Vector &x_tr, BlockVector &r, Vector &r_tr,
                    Integrand f)
   {
      Vector x_tr_l, r_tr_l;
      NPCResidualStart(x, x_tr, r, r_tr, x_tr_l, r_tr_l);
      if (!NonlinearResidualBatched(x_tr_l, b.GetBlock(0), b.GetBlock(1), r,
                                    r_tr_l, f))
      {
         MultNL(MultNlMode::AtFields, b, x_tr_l, r_tr_l, &r);
      }
      NPCResidualFinish(r_tr_l, r_tr);
   }
   /** @brief Assemble and factor the Jacobian at the same state, and return
       the reduced trace operator S = H - C' M^-1 [C; E].

       **Whether S is assembled at all is SetGradientMode()'s choice**, exactly
       as it is for GetGradient(). GradientMode::Assembled returns a
       SparseMatrix, so a direct solve or an algebraic preconditioner works;
       GradientMode::MatrixFree returns an Operator that applies S one element
       at a time with nothing stored, for a Krylov method that needs only the
       action. NPCReduce() and NPCRecover() work identically either way --
       both modes factor the local blocks and form the Schur complement, and
       it is only the global trace matrix that the matrix-free mode declines to
       build.

       Every LocalOpType works in both modes. LocalOpType::FluxNL did not
       until @a Sf_data was added; see SetGradientMode().

       The returned reference does not outlive the next call. */
   Operator &NPCGradient(const BlockVector &x, const Vector &x_tr);

   /** @brief NPCGradient() with the element potential-mass Jacobian supplied
       by the CALLER, as NPCResidual()'s templated overload supplies the
       residual's.

       The pair is the point: a caller holds ONE call site for each and gets
       the batched route when it applies and the element loop when it does
       not, with no predicate of its own to ask. @a g is
       `g(element, point, x, y, u)` returning @f$\partial f/\partial u@f$;
       see NonlinearGradientBatched().

       **Give this the derivative of the SAME law NPCResidual() was given.**
       Nothing checks it and nothing can -- the two callables are opaque -- so
       a mismatched pair is a Newton that converges linearly, or onto
       something else, with every norm looking healthy. That is this branch's
       oldest recorded failure shape and it is why the two entry points name
       each other here. */
   template <typename Integrand>
   Operator &NPCGradient(const BlockVector &x, const Vector &x_tr, Integrand g)
   {
      NPCCheck();

      // Before the batched pass, which reads @a darcy_p as its linearisation
      // point -- that is what GradAtFields means.
      darcy_u = x.GetBlock(0);
      darcy_p = x.GetBlock(1);

#ifdef MFEM_USE_MPI
      if (ParallelC())
      {
         // CanBatchNonlinearGradient() refuses ParallelC() through the
         // residual predicate, so there is nothing to try here: this is the
         // element loop and the integrand it evaluates is the integrator's.
         return ParReducedGradient(MultNlMode::GradAtFields, x_tr);
      }
#endif
      const bool done = NonlinearGradientBatched(g);
      return ReducedGradient(MultNlMode::GradAtFields, x_tr, done);
   }

   /** @brief Add @a dD to element @a el's potential-potential block of the
       local Jacobian, for the NEXT NPCGradient() only.

       The block is assembled and then factored inside one NPCGradient()
       call, and on the batched route (NonlinearGradientBatched()) nothing
       outside the library runs between the two. This is the seam between
       them. The correction is added after the block is assembled, on either
       route, and before the element is factored, so the Schur complement,
       the reduced operator, NPCReduce() and NPCRecover() all see the
       corrected block.

       @a dD means what the same matrix added to the elmat of the potential
       mass integrator's AssembleElementGrad() means: a square matrix of the
       element's potential dofs, in the integrator's element dof order, with
       no sign change. Asked for by meq, whose bordered Newton adds one row
       `kappa e_j b_p^T` to the one element that carries the border row's
       potential half. That removes a cancellation in the bordered step
       without changing the step. On the element loop meq used to add it
       inside their integrator, keyed on `ElementTransformation::ElementNo`.
       Under a Device they could not add it at all.

       **The queue is consumed by the next NPCGradient() and emptied there**,
       whether or not that call is the batched one. A caller who decides
       per step queues per step. Several corrections may be queued, and two
       on one element add. With nothing queued, NPCGradient() is unchanged to
       the bit.

       The correction is added wherever the assembled block already is: a
       one-block kernel on the device after the batched route, and on the
       host after the element loop. Neither moves D between memory spaces.

       Refused under LocalOpType::FluxNL. The potential block there is the
       linear mass, factored once and not reassembled, so there is no
       assembled block for the correction to join. Element numbers are
       local: on a ParMesh they are this rank's. */
   void AddNextGradientCorrection(int el, const DenseMatrix &dD);

   /// Empty the queue AddNextGradientCorrection() fills, without a gradient.
   void ClearNextGradientCorrections() const
   { grad_corr_el.SetSize(0); grad_corr_val.clear(); }

   /// How many corrections are queued for the next NPCGradient().
   int NumNextGradientCorrections() const { return grad_corr_el.Size(); }
   /// @brief The right-hand side of eq (18) for the trace increment.
   void NPCReduce(const BlockVector &r, const Vector &r_tr,
                  Vector &b_tr) const;
   /// @brief The local increments implied by a trace increment @a dtr.
   void NPCRecover(const BlockVector &r, const Vector &dtr,
                   BlockVector &dx) const;

   /** @brief NPCReduce() for several right-hand sides against one Jacobian.

       One pass over the mesh for all of them rather than one pass each. The
       element-local solve becomes MultInvBlocked() and every face product
       becomes a matrix-matrix one; what is actually saved is the traversal --
       GetElementFaces(), GetFaceElements(), GetCtFaceMatrix(),
       GetFaceVDofs() and the gathers happen once instead of once per column.
       The arithmetic on any one column is unchanged. In a build without
       LAPACK -- where mfem::Mult(DenseMatrix, ...) and LUFactors::Solve()
       are the native column loops -- that makes the answer BIT-IDENTICAL to
       the single-vector routine, measured at 0 relative difference on every
       column at three mesh sizes. With LAPACK the dense products become GEMM
       against the single-vector GEMV and only round-off agreement is claimed;
       do not write a bitwise assertion that a LAPACK build would fail.

       **What it is worth, measured rather than argued**: 1.24x end to end on
       4802 triangles at order 2 with 13 columns, against thirteen calls to
       the single-vector route, with the trace solve unblocked in both arms
       (UMFPackSolver does not override ArrayMult, so that leg is identical by
       construction and this is the two HDG legs alone). 1.36x at 2048
       elements, 1.15x at 4 columns. A trace solver that DOES block adds to
       this; see DarcyNPCSolver::ArrayMult().

       Every column must belong to the SAME Jacobian, i.e. to one
       NPCGradient() call, which is the case this exists for: a bordered,
       deflated, parameter-continued or adjoint solve has several right-hand
       sides known at once and one operator. Requested by MEQ for a bordered
       free-boundary Grad-Shafranov Newton, whose border columns are all
       known before the first application.

       @a r, @a r_tr and @a b_tr must have the same length. */
   void NPCReduce(const Array<const BlockVector *> &r,
                  const Array<const Vector *> &r_tr,
                  Array<Vector *> &b_tr) const;
   /// @brief NPCRecover() for several trace increments; see the blocked
   /// NPCReduce().
   void NPCRecover(const Array<const BlockVector *> &r,
                   const Array<const Vector *> &dtr,
                   Array<BlockVector *> &dx) const;
   ///@}

   /** @brief The number of local nonlinear iterations performed, summed over
       elements and over every residual and gradient evaluation.

       Zero for a linear problem, and zero under NPC -- see NPCResidual() --
       which is the acceptance signal that NPC really is running a single
       local linear solve per outer step rather than a condensation. */
   long GetNumLocalNLIterations() const { return num_local_nl_iters; }

   /// N/A, use SetConstraintIntegrators()
   void SetConstraintIntegrator(BilinearFormIntegrator *c_integ) = delete;

   /// Sets the constraint integrators
   /** Set the integrators that will be used to construct the constraint
       matrices for fluxes @a C and (if provided) stabilization contributions
       to @a E, @a G, @a D and @a H for potentials. Note the potential
       integrator is required to implement the method
       BilinearFormIntegrator::AssembleHDGFaceMatrix(). The DarcyHybridization
       object assumes ownership of the integrators, i.e. it will delete the
       integrators when destroyed. */
   void SetConstraintIntegrators(BilinearFormIntegrator *c_flux_integ,
                                 BilinearFormIntegrator *c_pot_integ);

   void SetConstraintIntegrators(BilinearFormIntegrator *c_flux_integ,
                                 NonlinearFormIntegrator *c_pot_integ);

   void SetConstraintIntegrators(BilinearFormIntegrator *c_flux_integ,
                                 BlockNonlinearFormIntegrator *c_integ);

   void SetFluxMassNonlinearIntegrator(NonlinearFormIntegrator *flux_integ,
                                       bool own = true);

   void SetPotMassNonlinearIntegrator(NonlinearFormIntegrator *pot_integ,
                                      bool own = true);

   /** @brief Add a face constraint that is EVALUATED at every residual and
       gradient, beside whatever SetConstraintIntegrators() froze.

       The route for a face integrator that is bilinear in the unknowns and
       whose COEFFICIENT is not: an upwinded convection whose drift velocity
       is a function of another field of the coupled system moves at every
       residual, and folding it into @a c_bfi_p assembles it once and then
       never looks at the coefficient again. Nothing about that is visible --
       the answer is simply the one the first assembly implied -- and the only
       way out was Update() + Assemble() + Finalize(), which for a caller
       measuring it was 92.7 ms against 8.9 ms for the gradient it wanted.

       Unlike SetConstraintIntegrators(), this does NOT clear @a c_bfi_p. The
       two then hold different halves of the same face term -- the frozen
       stabilization and the live convection, which is the configuration that
       occurs -- and Finalize() keeps a copy of the frozen half in
       @a E_lin_data, @a G_lin_data and @a H_lin_data because the live pass
       overwrites those blocks.

       @note Call it AFTER SetConstraintIntegrators(), which resets this slot.
             DarcyForm::EnableHybridization() does, and is the normal caller;
             set DarcyForm::SetFaceConstraintMode() to ask it to.
       @note Ownership is taken, as it is for every constraint integrator. */
   void SetPotConstraintNonlinearIntegrator(NonlinearFormIntegrator *c_pot_integ);

   void SetBlockNonlinearIntegrator(BlockNonlinearFormIntegrator *block_integ,
                                    bool own = true);

   /// Returns the flux constraint integrator
   BilinearFormIntegrator* GetFluxConstraintIntegrator() const { return c_bfi.get(); }

   /// Returns the potential constraint integrator
   BilinearFormIntegrator* GetPotConstraintIntegrator() const { return c_bfi_p.get(); }

   /** @brief Assemble the INTERIOR-face potential term with one batched
       kernel, scattering straight into E, G, H and D.

       @returns false, having done nothing, whenever it does not apply --
       AssemblyMode is not Batched, the face term is not a single
       pure-diffusion HDGDiffusionIntegrator, or the spaces do not admit the
       batched form. The caller then takes the per-face loop, so this is an
       optimisation and never a restriction.

       Boundary faces are NOT covered and still go through
       ComputeAndAssemblePotBdrFaceMatrix(). */
   bool AssemblePotFaceMatricesBatched();
   /// The integrators the batched face assembly would apply, sum unwrapped.
   void PotFaceConstraintIntegrators(
      Array<BilinearFormIntegrator*> &integs) const;
   /// The boundary constraint integrators, and the faces each marker admits.
   void PotBdrFaceLists(std::vector<Array<int>> &lists, Array<int> &all,
                        Array<BilinearFormIntegrator*> &integs) const;
   bool AssemblePotBdrFaceMatricesBatched();
   /// Shared by CanBatchFluxMass() and CanBatchPotMass().
   bool CanBatchElementMass(BilinearForm *M,
                            const FiniteElementSpace &f) const;
   /** @brief The (boundary element, integrator, element) triples
       DarcyForm::AssembleFluxMassBdrFaces() would visit, in that loop's own
       order -- attribute markers applied per integrator and the periodic
       leftovers dropped by the same null-transformation test. */
   void FluxMassBdrWork(BilinearForm *M_u, Array<int> &bdr_els,
                        Array<int> &integs, Array<int> &elems) const;
   /// The local index of each free / essential hat dof, and where each
   /// element's essential run starts; see AssembleFluxMassMatricesBatched().
   void HatDofMaps(Array<int> &free_map, Array<int> &ess_map,
                   Array<int> &ess_offsets) const;
   /// The interior faces, which is what the batched face assembly covers.
   void InteriorFaceList(Array<int> &flist) const;
   /** @brief Whether this rank's SHARED faces admit the batched pair route;
       true in serial and where there are none, so a caller can ask
       unconditionally. @a residual picks the stricter of the two predicates. */
   bool SharedNLFacesCanBatch(
      const Array<NonlinearFormIntegrator*> &integs,
      const Array<BlockNonlinearFormIntegrator*> &bintegs,
      bool residual) const;
   /// The uniformity the pair gather needs, over @a flist at @a ns sides.
   bool NLFacePairsUniform(const Array<int> &flist, int ns, int LDD,
                           int LDC) const;
#ifdef MFEM_USE_MPI
   /** @brief This rank's shared faces, by LOCAL face index -- the faces
       ParDarcyForm::AssemblePotHDGSharedFaces() visits, which are exactly the
       ones InteriorFaceList() cannot see (Mesh::FaceIsInterior() is
       `Elem2No >= 0` and a shared face has no local Elem2). */
   void SharedFaceList(Array<int> &flist) const;
   /** @brief Assemble the SHARED-face potential term with one batched kernel.

       @returns false, having done nothing, whenever it does not apply; the
       caller then takes the per-face loop. Serial builds and serial runs have
       no shared faces, so this is a parallel-only route and returns true
       having done nothing when the list is empty. */
   bool AssemblePotSharedFaceMatricesBatched();
   bool CanBatchPotSharedFaceAssembly() const;
#endif
   NonlinearFormIntegrator* GetPotConstraintNonlinearIntegrator() const { return c_nlfi_p.get(); }

   /** @brief The nonlinear flux mass integrator, or NULL.

       @note **This returned the *potential* mass integrator until it was
       fixed** -- the same defect, in the same words, as
       DarcyReduction::GetFluxMassNonlinearIntegrator(); the two classes each
       carried their own copy. See that one for why no deprecated alias is
       offered. */
   NonlinearFormIntegrator* GetFluxMassNonlinearIntegrator() const { return m_nlfi_u; }
   NonlinearFormIntegrator* GetPotMassNonlinearIntegrator() const { return m_nlfi_p; }

   /** @brief Not available, use AddBdrFluxConstraintIntegrator()
       or AddBdrPotConstraintIntegrator(). */
   void AddBdrConstraintIntegrator(BilinearFormIntegrator *c_integ) = delete;

   /** @brief Not available, use AddBdrFluxConstraintIntegrator()
       or AddBdrPotConstraintIntegrator(). */
   void AddBdrConstraintIntegrator(BilinearFormIntegrator *c_integ,
                                   Array<int> &bdr_marker) = delete;

   /** @brief Not available, use GetBdrFluxConstraintIntegrator()
       or GetBdrPotConstraintIntegrator(). */
   Array<BilinearFormIntegrator*> *GetBCBFI() = delete;

   /** @brief Not available, use GetBdrFluxConstraintIntegratorMarker()
       or GetBdrPotConstraintIntegratorMarker(). */
   Array<Array<int>*> *GetBCBFI_Marker() = delete;

   /// Adds flux boundary constraint integrator
   /** Add the boundary face integrator that will be used to construct the
       constraint matrix @a C. The DarcyHybridization object assumes ownership
       of the integrator, i.e. it will delete the integrator when destroyed. */
   void AddBdrFluxConstraintIntegrator(BilinearFormIntegrator *c_integ)
   { Hybridization::AddBdrConstraintIntegrator(c_integ); }

   /// Adds flux boundary constraint integrator (with a boundary marker)
   /** Add the boundary face integrator that will be used to construct the
       constraint matrix @a C. The DarcyHybridization object assumes ownership
       of the integrator, i.e. it will delete the integrator when destroyed.
       The boundary attribute marker array is referenced and must remain valid
       over the lifetime. */
   void AddBdrFluxConstraintIntegrator(BilinearFormIntegrator *c_integ,
                                       Array<int> &bdr_marker)
   { Hybridization::AddBdrConstraintIntegrator(c_integ, bdr_marker); }

   /// Get number of all integrators added with AddBdrFluxConstraintIntegrator().
   inline int NumBdrFluxConstraintIntegrators() const { return Hybridization::NumBdrConstraintIntegrators(); }

   /// Access all integrators added with AddBdrFluxConstraintIntegrator().
   BilinearFormIntegrator& GetBdrFluxConstraintIntegrator(int i) { return Hybridization::GetBdrConstraintIntegrator(i); }

   /// Access all boundary markers added with AddBdrFluxConstraintIntegrator().
   /** If no marker was specified when the integrator was added, the
       corresponding pointer (to Array<int>) will be NULL. */
   Array<int>* GetBdrFluxConstraintIntegratorMarker(int i) { return Hybridization::GetBdrConstraintIntegratorMarker(i); }

   /// Adds potential boundary constraint integrator
   /** Add the boundary face integrator that will be used to construct the
       constraint stabilization matrices @a E, @a G, @a D and @a H. Note the
       integrator is required to implement the method
       BilinearFormIntegrator::AssembleHDGFaceMatrix(). The DarcyHybridization
       object assumes ownership of the integrator, i.e. it will delete the
       integrator when destroyed. */
   void AddBdrPotConstraintIntegrator(BilinearFormIntegrator *c_integ)
   {
      boundary_constraint_pot_integs.push_back(c_integ);
      boundary_constraint_pot_integs_marker.push_back(
         NULL); // NULL marker means apply everywhere
   }

   /// Adds potential boundary constraint integrator (with a boundary marker)
   /** Add the boundary face integrator that will be used to construct the
       constraint stabilization matrices @a E, @a G, @a D and @a H. Note the
       integrator is required to implement the method
       BilinearFormIntegrator::AssembleHDGFaceMatrix(). The DarcyHybridization
       object assumes ownership of the integrator, i.e. it will delete the
       integrator when destroyed. The boundary attribute marker array is
       referenced and must remain valid over the lifetime. */
   void AddBdrPotConstraintIntegrator(BilinearFormIntegrator *c_integ,
                                      Array<int> &bdr_marker)
   {
      boundary_constraint_pot_integs.push_back(c_integ);
      boundary_constraint_pot_integs_marker.push_back(&bdr_marker);
   }

   /// Get number of all integrators added with AddBdrPotConstraintIntegrator().
   inline int NumBdrPotConstraintIntegrators() const { return boundary_constraint_pot_integs.size(); }

   /// Access all integrators added with AddBdrPotConstraintIntegrator().
   BilinearFormIntegrator& GetBdrPotConstraintIntegrator(int i) { return *boundary_constraint_pot_integs[i]; }

   /// Access all boundary markers added with AddBdrPotConstraintIntegrator().
   /** If no marker was specified when the integrator was added, the
       corresponding pointer (to Array<int>) will be NULL. */
   Array<int>* GetBdrPotConstraintIntegratorMarker(int i) { return boundary_constraint_pot_integs_marker[i]; }

   void AddBdrPotConstraintIntegrator(NonlinearFormIntegrator *c_integ)
   {
      boundary_constraint_pot_nonlin_integs.push_back(c_integ);
      boundary_constraint_pot_nonlin_integs_marker.push_back(
         NULL); // NULL marker means apply everywhere
   }
   void AddBdrPotConstraintIntegrator(NonlinearFormIntegrator *c_integ,
                                      Array<int> &bdr_marker)
   {
      boundary_constraint_pot_nonlin_integs.push_back(c_integ);
      boundary_constraint_pot_nonlin_integs_marker.push_back(&bdr_marker);
   }

   /// Get number of all non-linear integrators added with AddBdrPotConstraintIntegrator().
   inline int NumBdrPotConstraintNLIntegrators() const { return boundary_constraint_pot_nonlin_integs.size(); }

   /// Access all non-linear integrators added with AddBdrPotConstraintIntegrator().
   NonlinearFormIntegrator& GetBdrPotConstraintNLIntegrator(int i) { return *boundary_constraint_pot_nonlin_integs[i]; }

   /// Access all boundary markers added with AddBdrPotConstraintIntegrator().
   /** If no marker was specified when the non-linear integrator was added, the
       corresponding pointer (to Array<int>) will be NULL. */
   Array<int>* GetBdrPotConstraintNLIntegratorMarker(int i) { return boundary_constraint_pot_nonlin_integs_marker[i]; }

   void AddBdrConstraintIntegrator(BlockNonlinearFormIntegrator *c_integ)
   {
      boundary_constraint_nonlin_integs.push_back(c_integ);
      boundary_constraint_nonlin_integs_marker.push_back(
         NULL); // NULL marker means apply everywhere
   }
   void AddBdrConstraintIntegrator(BlockNonlinearFormIntegrator *c_integ,
                                   Array<int> &bdr_marker)
   {
      boundary_constraint_nonlin_integs.push_back(c_integ);
      boundary_constraint_nonlin_integs_marker.push_back(&bdr_marker);
   }

   /// Get number of all non-linear integrators added with AddBdrConstraintIntegrator().
   inline int NumBdrConstraintNLIntegrators() const { return boundary_constraint_pot_integs.size(); }

   /// Access all non-linear integrators added with AddBdrConstraintIntegrator().
   BlockNonlinearFormIntegrator& GetBdrConstraintNLIntegrator(int i) { return *boundary_constraint_nonlin_integs[i]; }

   /// Access all boundary markers added with AddBdrConstraintIntegrator().
   /** If no marker was specified when the non-linear integrator was added, the
       corresponding pointer (to Array<int>) will be NULL. */
   Array<int>* GetBdrConstraintNLIntegratorMarker(int i) { return boundary_constraint_nonlin_integs_marker[i]; }

   void UseExternalBdrConstraintIntegrators() = delete;

   /// Indicate that boundary flux constraint integrators are not owned
   void UseExternalBdrFluxConstraintIntegrators() { Hybridization::UseExternalBdrConstraintIntegrators(); }

   /// Indicate that boundary potential constraint integrators are not owned
   void UseExternalBdrPotConstraintIntegrators() { extern_bdr_constr_pot_integs = 1; }

   /// Prepare the DarcyHybridization object for assembly.
   /** @param ess_flux_tdof_list    essential true DOFs of the flux */
   void Init(const Array<int> &ess_flux_tdof_list) override;

   /// Specify essential boundary conditions on the trace.
   /** Takes a *boundary attribute* marker and produces essential true DOFs of
       the constraint (trace) space, not of the flux -- the flux ones are
       Init()'s argument. See @a ess_tdof_list. */
   void SetEssentialBC(const Array<int> &bdr_attr_is_ess);

   /** @brief Boundary attributes whose trace row is a BOUNDARY CONDITION and
       not the conservativity condition, so the flux constraint is left out of
       it.

       On an ordinary boundary face the trace row reads

       \verbatim
           < (F^ + q^) . n , mu >  =  0,
       \endverbatim

       and the `q^.n` half of it comes from the flux constraint C -- the same
       block whose transpose puts `<lambda, v.n>` on the FLUX row. A condition
       that REPLACES the row rather than supplying a datum to it therefore
       cannot be expressed by any integrator: a boundary face integrator on the
       potential mass form can add to the row, but it is a form in (p, lambda)
       and cannot cancel a term in q. This is the asymmetry that lets it:
       @a C is dropped from the trace row on these attributes while @a C^T
       stays on the flux row, which is what a boundary condition IS -- the
       transmission condition is replaced, and the trace still enters the
       element's local problem exactly as before.

       **The row is then whatever the potential mass form's boundary face
       integrators put there, and NOTHING ELSE.** A caller that marks an
       attribute here owes it a complete condition; marking one and supplying
       no boundary integrator leaves an empty row and a singular system, which
       is a different failure from the zero-numerical-flux row that an
       unmarked, un-datumed attribute produces.

       Pass an empty array to clear. Call before Finalize(); the marker is read
       there and at every trace-row assembly after it.

       The caller this exists for is a CHARACTERISTIC outflow,
       `B^ = A+_n (u - u^) - A-_n (u_inf - u^)`, which is a condition on the
       trace alone and carries no viscous flux -- see miniapps/hdg/nsflux.hpp's
       HDGCharacteristicBdrFlux and the boundary-condition step of
       navierstokes.cpp. */
   void SetTraceBCAttributes(const Array<int> &bdr_attr_marker);

   /// The marker SetTraceBCAttributes() was given; empty when none was.
   const Array<int> &GetTraceBCAttributes() const { return bc_trace_marker; }

   /// Specify essential VDOFs of the constraint (trace) space.
   /** Use either SetEssentialBC() or SetEssentialTrueDofs() if possible. */
   void SetEssentialVDofs(const Array<int> &ess_vdofs_list);

   /// Specify essential true DOFs of the constraint (trace) space.
   void SetEssentialTrueDofs(const Array<int> &ess_tdof_list_)
   { ess_tdof_list_.Copy(ess_tdof_list); ess_tdof_list_.Copy(ess_tdof_user); }

   /// Return a (read-only) list of the essential *trace* true DOFs.
   /** These index the reduced system, so this is the list a caller needs to
       compare GetGradient() against a finite difference of Mult(): the
       residual is masked on them and the Jacobian carries a unit row, so they
       have to be left out or the comparison is meaningless. See
       @a ess_tdof_list. */
   const Array<int> &GetEssentialTrueDofs() const { return ess_tdof_list; }

   /// How a face's trace degree follows the degrees of its two elements.
   enum class TraceOrderRule
   {
      /** @brief The lower of the two. It is the configuration every existing
          case uses, and it is **not** the safe choice it looks like: it gets
          WORSE as the degree jump across an interface grows. Measured on
          `convdiff -o 1 -nx 8 -pref n`, which raises half the domain by `n`
          degrees, the relative flux error goes

              n = 1      n = 2      n = 3
              1.066e-2   1.620e-2   1.519e-2

          -- adding degrees to the refined half makes the answer worse, because
          the interface trace is held at the coarse degree and becomes a
          tighter bottleneck the richer its other side gets. Max holds
          1.051e-2, 1.051e-2, 1.054e-2 over the same three. */
      Min,
      /** @brief The higher of the two, which is the usual choice in the
          literature. It needs the HDG face quadrature to take the trace
          element's order into account, or the trace-trace block comes back
          rank-deficient -- that is fixed, see the note at the top of
          bilininteg_hdg.cpp. Measured to be *exactly redundant* on a face
          whose two neighbours agree, so it can only pay at a genuine
          `p`-interface -- and it does.

          **Where the interface is put decides whether the rule matters, and
          the two studies that look contradictory are not.** On a PRESCRIBED
          interface (`convdiff -pref`, half the domain by geometry) Max costs
          0.2 to 2.6 per cent more active dofs and returns, at a one-degree
          jump, nothing outside the last digits -- net dof efficiency 0.998 to
          1.004, exactly the wash the redundancy argument predicts. At a jump
          of two or more it reaches Min's *flux* error at 12 to 35 per cent
          fewer active dofs, and the *potential* still never sees it: ratios
          0.997 to 1.0005 at every jump and mesh. Neither rule changes the
          convergence rate; Min's penalty is an `O(h^(r+1))` term on an
          `O(h^r)` error, so the gap closes -- 0.598, 0.640, 0.758, 0.857 over
          four meshes at `-o 2 -pref 2`.

          In an ADAPTIVE loop the potential does see it, by 21 to 27 per cent
          of the dofs at fixed error over four decades
          (`anisodiff -p 5 -ks 1e2 --hp-adaptivity`, table in that file). The
          difference is not the metric but the placement: a geometric interface
          lands wherever the domain is cut, which on that problem is away from
          the layer, while an adaptive one is put exactly on the feature. A
          rule can only matter where its interface does. */
      Max,
   };

   /** @brief Derive one trace degree per face from one degree per element.

       @a cap is the ceiling, normally the degree the constraint space was
       built at. Boundary faces take their single neighbour's degree.

       **On a nonconforming mesh the unit is the hanging-node family, not the
       face.** A family carries one trace unknown, on its master face, so it
       has one degree: the rule taken over the slaves, each of which has seen
       a fine element and the coarse one. The master's own entry says nothing,
       because a master face is never integrated directly.

       This used to force the family to the ceiling instead, and that was a
       limitation of the retired route rather than a choice -- see
       SetTraceOrders() for why, and for the 0.284 / 1.06 / 3.67 it cost to
       ignore. Constraining the surplus removes it.

       **In parallel a shared face takes both ranks' degrees**, obtained by one
       exchange over face neighbours. The mesh is taken by non-const reference
       for that reason: the exchange builds face-neighbour data on it. The rule
       is applied to the same pair of degrees on both sides and min and max are
       both symmetric, so the two ranks agree by construction rather than by
       convention. */
   static void FaceOrdersFromElementOrders(Mesh &mesh,
                                           const Array<int> &elem_order,
                                           TraceOrderRule rule, int cap,
                                           Array<int> &face_order);

   /** @brief Give each face its own trace polynomial degree, for `p`-adaptivity.

       @a face_order holds one degree per mesh face, and every entry must lie
       between 0 and the degree of the constraint space, which stays a uniform
       space at the maximum degree. A face of degree `p_f` then carries
       `nt(p_f)` unknowns instead of the `nt(p_max)` slots it owns, and the
       difference is **constrained away, not retired**: the slots hold the
       ceiling-basis coefficients of a function that happens to be of degree
       `p_f`, and

           E(j, i) = phi_i^{p_f}( node_j^{p_max} )

       is the per-face matrix that says so. The reduced system is in the
       constrained unknowns, so its size really is the sum of `nt(p_f)`; there
       are no unit rows standing in for the surplus.

       **Why it is constrained rather than truncated, which is what this used
       to do.** Storing a coarse function as coarse coefficients in the first
       `nt(p_f)` slots is well defined inside DarcyHybridization, which knows
       the degrees, and wrong for every reader that does not -- and the
       space's own machinery is exactly such a reader. Three things followed
       from that and all three are gone:

       | was refused | because | cost measured then | and now |
       |---|---|---|---|
       | a hanging-node family below the ceiling | the conforming prolongation interpolates master onto slave in the ceiling basis | the error went 0.284, 1.06, 3.67 as the mesh refined | works; the demonstrator runs families below the ceiling every cycle |
       | an essential datum on a coarsened boundary face | the caller projects the datum in the ceiling basis | 21x, 0.0124 to 0.259 as the ceiling went 2, 3, 5, 8 | 0.0124172 at every one of those ceilings, to every printed digit |
       | a face shared between ranks | the two ranks order its slots by their own view of the orientation | 144 retired true DOFs at one rank, 152 at two, 162 at three | 432 constrained true DOFs at every rank count, and `pconvdiff --p-refine 1 --p-refine-x 0` gives 5.867e-4 at 1, 2, 3 and 4 ranks where it gave 5.9e-4 and 0.56 |

       All three are one cause and the constraint is one repair. The stored
       coefficients are the ceiling's own, so the prolongation, the dof
       ordering and the true-dof numbering all apply unchanged, and
       orientation stops being this route's problem: only the OWNER of a face
       builds an `E` for it, in its own ordering, and every other rank
       receives values through the space's map as it always did.

       **That it cannot change the discretisation is measured, not argued.**
       `phi_i^lo = sum_j E(j,i) phi_j^hi` POINTWISE, so a face matrix
       assembled against the coarse trace equals the ceiling one restricted by
       `E`, under any quadrature rule applied to both. The unit tests
       `"A coarse trace basis is an exact combination of the ceiling's"` and
       `"The constrained ceiling system IS the coarse system"` check the two
       halves -- the identity on one face, and `E^T H(ceiling) E == H(coarse)`
       on the assembled reduced system, with a plain slot selection alongside
       as the control that says the comparison can fail.

       The low-order face is a genuine subspace of the high-order one here,
       and BuildTraceConstraint() verifies that once per (geometry, degree)
       rather than assuming it: `R E == I`, where `R` reads a ceiling function
       back at the coarse nodes. A collection that is modal, or whose lower
       degrees are not nested, fails loudly instead of quietly discretising
       something else.

       The degrees are stored, not derived. A `p`-adaptive driver picks them
       from the neighbouring element degrees by whatever rule it wants; note
       that a face richer than *both* its neighbours is exactly redundant
       rather than wrong (measured -- see the test "A trace richer than both
       its elements is exactly redundant"), so `min` costs nothing that `max`
       would have bought there. A hanging-node family is the exception to
       "one degree per face": it carries one unknown, on its master, so
       FaceOrdersFromElementOrders() gives the whole family one degree and
       only the master's is ever read.

       **The constraint space's degree is a ceiling, not a starting point.**
       This route reuses that space's storage, so a face can only go *down*
       from the degree it was constructed with; there is no way to enrich past
       it. A driver that means to raise degrees must therefore build the
       constraint space at the highest degree the run will ever reach and start
       its faces below that, not at the degree it happens to start from. Every
       caller in the tree today builds it at the element degree, which is
       coarsening-only.

       **What that ceiling costs, measured rather than predicted.** The stored
       blocks and a constraint-space vector go as `nt(p_max)` per face
       whatever the degrees are, and so do the local blocks: they are sized
       from TraceFE(), which is the ceiling's element. The trace-dependent
       element-local work is `n_el^2 n_c + n_el n_c^2`, so it should grow as
       `r` and `r^2` in `r = nt(p_max)/nt(p_f)` -- a predicted 4.0x at the
       demonstrator's configuration, 2D quads at element degree 2 under a
       ceiling of 7, where `r = 8/3`.

       The controlled experiment is a fixed mesh with every face at degree 2,
       sweeping only the ceiling, so the answer cannot move and does not
       (t_err 1.6883e-05 at every one). Seconds, 64x64:

       | ceiling | hybridization | assembly | trace solve |
       |---|---|---|---|
       | 2 | 0.056 | 0.204 | 0.0626 |
       | 3 | 0.071 | 0.268 | 0.0627 |
       | 5 | 0.079 | 0.392 | 0.0623 |
       | 7 | 0.100 | 0.659 | 0.0634 |

       So 3.2x on assembly at the extreme ceiling -- the prediction's shape,
       a little below its size, since assembly also carries trace-independent
       element work -- and **the solve is flat to the third digit**, which is
       what constraining bought: the reduced system is the same size at every
       ceiling. Under the retired route it was not, and the ceiling cost 1.03
       to 1.19x there instead.

       Keeping the local blocks at `p_f` is possible -- it means applying `E`
       at each of the twelve gather and scatter sites instead of once in the
       prolongation. **The claim that stood here, that the measurement does
       not justify it, is WITHDRAWN**: it rested on an end-to-end comparison
       that could not resolve anything (0.67 s against 0.81 s, while an
       unchanged path moved as far on the same machine), and the right answer
       to a comparison inside the drift is a better instrument, not a weaker
       conclusion.

       **The end-to-end measurement is resolvable, and the ceiling does cost
       the demonstrator.** What makes it controlled is a demonstrator run that
       never REACHES its ceiling -- the sensor stops enriching at degree 6 --
       so every ceiling at or above 6 takes identical decisions and the two
       arms differ in stored size and in nothing else. `-nx 16 -amr 11` at
       ceilings 6, 7 and 10 agrees in element count, `M`, degree range and
       both error norms in every one of the twelve cycles.

       Interleaved pairs on that arm, order alternated, sign test over twelve
       pairs:

       | arms | ratio | pairs | p |
       |---|---|---|---|
       | ceiling 7 against itself (the control) | 0.991x | 8/12 | 0.19, NOT resolved |
       | ceiling 6 -> 7, one degree | **1.097x** | 12/12 | 0.0002 |
       | ceiling 6 -> 10, four degrees | **1.423x** | 12/12 | 0.0002 |

       and a round-robin over 6..10 gives 1.000, 1.037, 1.124, 1.281, 1.370 --
       monotone and compounding at 1.04 to 1.09 per degree, which is the shape
       `r` and `r^2` predict.

       **And all of it is in assembly**, which is what says it is the local
       blocks and therefore Shape B's to recover: summed over the run, assembly
       goes 0.536 / 0.631 / 0.888 s at ceilings 6 / 7 / 10 while the
       preconditioner (0.092 / 0.087 / 0.081) and the trace solve (0.037 /
       0.036 / 0.038) do not move. Assembly is 93% of the demonstrator's
       measured legs.

       So one degree of unused ceiling costs about a tenth of the run, the
       default ceiling is `order + 5`, and the demonstrator's faces sit at 2
       to 6 under it. Shape B is justified by that; what is NOT measured is
       Shape B itself, only the ceiling it would remove. At the modest ceiling
       a coarsening-only driver needs, `p_max = order + 1`, the gap is one
       degree and the same instrument would be needed to see it.

       **A face richer than its element is a real thing and stays one.** The
       measured claim that a trace above both its elements is exactly
       redundant was taken on a conforming mesh and does not carry over to a
       hanging node: a master face sees several fine elements, which between
       them do reach the higher modes, so the extra degrees are determined
       rather than annihilated. On the same mesh that is BETTER -- 0.0818
       against 0.0982 on an identical hanging-node mesh whose only difference
       is four family faces at degree 3 instead of 2. An earlier note said the
       opposite, comparing two runs that had refined differently by then; it
       is withdrawn.

       **Call it straight after DarcyForm::EnableHybridization() and before
       Assemble().** C, E, G and H are sized from the trace element and this
       rebuilds them, calling Reset() and discarding anything assembled. Under
       the present sizing that rebuild is a no-op, since those blocks follow
       the ceiling either way; the contract is kept as it is because keeping
       the local blocks at `p_f` would need it back, and a contract that
       tightens again later is worse than one that never loosened. Passing an
       empty array returns to a uniform trace on the same terms.

       **A nonlinear hybridized solve works under a per-face trace, and the
       thing that makes it work is worth knowing.** The element loop addresses
       the trace by TraceVDofs() -- the constraint space's face VDOFs --
       throughout, and the vector a nonlinear solver hands the operator is in
       the CONSTRAINED unknowns, which are fewer. So every entry point
       prolongs to the ceiling, runs the loop there, and restricts the result
       back with the transpose: ParMultNL() is that wrapper, the linear
       route's ReduceRHS() and ComputeSolution() do the same by hand, and
       ComputeH() does it as a RAP. Three sites called the element loop
       directly instead -- Mult(), GetGradient() and the matrix-free
       Gradient -- and read a vector of constrained unknowns with the
       ceiling's numbers. They go through the wrapper now.

       **That was never safe, and a per-face trace is only the second way to
       reach it.** Those sites were right while the trace prolongation is
       NULL, which is a conforming serial mesh and nothing else:
       DG_Interface_FECollection derives from RT_FECollection and so reports
       GetContType() == NORMAL rather than DISCONTINUOUS, which means
       FiniteElementSpace::BuildConformingInterpolation() does not take its
       early exit for a DG trace -- so on a nonconforming mesh with hanging
       nodes the space has a real prolongation and the direct call reads past
       the end of the reduced vector there too. Measured on amr-quad.mesh
       with one refinement, order 3: VSize 1056, TrueVSize 928, cP non-null.
       That half is not this route's and belongs on the trunk. */
   void SetTraceOrders(const Array<int> &face_order);

   /// Return the per-face trace degrees, empty when the trace is uniform.
   const Array<int> &GetTraceOrders() const { return tr_order; }

   /** @brief The trace finite element the DOFs of face @a f live in.

       Always the constraint space's own, which is the CEILING's element, and
       that is the point: a face's coefficients are ceiling-basis
       coefficients, of a function constrained to its own degree. Anything
       reading a trace vector -- an estimator, a reconstruction, a
       GridFunction -- therefore reads the right function without knowing the
       degrees, which is what constraining buys over truncating.

       The face's own DEGREE, which is a different question, is
       GetTraceOrders()[f]; FaceDegreeFE() is its element and is private,
       being needed only to build the constraint. The local blocks are sized
       from here, so they follow the ceiling; see SetTraceOrders() for what
       that costs and what the alternative would be. */
   const FiniteElement *TraceFE(int f) const;

   /** @brief The constraint-space VDOFs of face @a f.

       All of them: a face's storage is the ceiling's whatever its degree.
       Kept as a method rather than folded back into
       FiniteElementSpace::GetFaceVDofs() so that the twelve places which
       gather and scatter a face's block have one door to go through, which is
       where the per-face degree would re-enter if the local blocks were ever
       taken back down to it. */
   void TraceVDofs(int f, Array<int> &vdofs) const;

   /** @brief The number of trace unknowns the reduced system solves for.

       The sum over the faces this process owns of nt(p_f) * vdim -- one per
       degree of freedom a face actually has, the ceiling's surplus
       constrained away rather than retired into a unit row. With no per-face
       degrees set it is the constraint space's own true size. */
   int GetTraceTrueVSize() const;

   /** @brief The trace prolongation, from the constrained true DOFs to the
       constraint space's VDOFs, or NULL when it would be the identity.

       This is what every reader of a trace prolongation goes through once a
       per-face degree is in play. With no degrees set it returns **exactly**
       what FiniteElementSpace::GetConformingProlongation() returns, the same
       pointer and the same null, so the uniform path is unchanged by
       construction rather than by test.

       Serial only; see GetParTraceProlongation() for the parallel one. */
   const SparseMatrix *GetTraceProlongationMatrix() const;

#ifdef MFEM_USE_MPI
   /** @brief The trace prolongation in parallel: constrained true DOFs to
       constraint-space LDOFs, i.e. Dof_TrueDof composed with the per-face
       constraint. Never null.

       The composition is in that order, and that ordering is the whole of the
       shared-face repair: the constraint acts on TRUE DOFs, so only the owner
       of a face builds an E for it, in its own ordering, and every other rank
       receives LDOF values through the space's map exactly as it did for a
       uniform trace. Orientation stops being this route's problem, which is
       what the retired route could never manage -- there the meaning of a
       slot was per-rank, and no exchange repairs a disagreement about what a
       number means. */
   HypreParMatrix *GetParTraceProlongation() const;
#endif

   /** @brief GetTraceProlongationMatrix(), or its parallel counterpart, as an
       Operator. NULL only in the serial case where it would be the
       identity. */
   const Operator *GetTraceProlongation() const;

   /** @brief Prolong a constrained trace vector to constraint-space VDOFs.

       The result is a genuine ceiling-basis representation of a function of
       each face's own degree, so anything reading the trace space generically
       -- a GridFunction, the error estimator, the reconstruction, GLVis --
       reads the right function. That is what constraining buys over
       truncating, where the same slots would have held a coarser basis's
       coefficients and every such reader would have been silently wrong. */
   void ProlongTrace(const Vector &X_c, Vector &x) const;

   /** @brief Read a constraint-space VDOF vector back at each face's own
       degree.

       A left inverse of ProlongTrace(), and **not** its transpose: it
       interpolates at the coarse nodes where the transpose would integrate
       against the ceiling's. Use it for DATA -- a boundary datum, an initial
       guess -- and the transpose for residuals.

       The difference is why the essential-datum refusal existed. This map is
       a function of the FUNCTION, so restricting one function represented at
       two different ceilings gives the same answer twice; a least-squares
       pseudo-inverse is a function of the ceiling's NODE SET and does not.
       Both are left inverses, so only the second property separates them, and
       it is measured in "A coarse trace basis is an exact combination of the
       ceiling's". */
   void RestrictTrace(const Vector &x, Vector &X_c) const;

   /// Not available, use a specific Assemble*MassMatrix() instead.
   void AssembleMatrix(int el, const DenseMatrix &A) override
   { MFEM_ABORT("Not supported, system part must be specified"); }

   /// Assemble element matrix of @a Mu
   void AssembleFluxMassMatrix(int el, const DenseMatrix &A);

   /// Assemble element matrix of @a Mp
   void AssemblePotMassMatrix(int el, const DenseMatrix &D);

   /// Assemble element matrix of @a B
   void AssembleDivMatrix(int el, const DenseMatrix &B);

   /// Computes and assembles potential face matrix
   /** The provided provided potential constraint integrator (see
       SetConstraintIntegrators()) is used to compute the HDG face matrix,
       which contributes to @a D, @a E, @a G and @a H. The element
       contributions to @a D are returned in @p elmat1 and @p elmat2 together
       with the VDOFs lists @p vdofs1 and @p vdofs2. The flag for skipping
       zeros for contributions of @a H to the hybridized matrix can be set in
       @p skip_zeros. */
   void ComputeAndAssemblePotFaceMatrix(int face,
                                        DenseMatrix & elmat1, DenseMatrix & elmat2,
                                        Array<int>& vdofs1, Array<int>& vdofs2, int skip_zeros = 1);

   /// Computes and assembles potential boundary face matrix
   /** The provided provided potential constraint integrator (see
       SetConstraintIntegrators()) is used to compute the HDG boundary face
       matrix, which contributes to @a D, @a E, @a G and @a H. The element
       contributions to @a D are returned in @p elmat together with the VDOFs
       list @p vdofs. The flag for skipping zeros for contributions of @a H to
       the hybridized matrix can be set in @p skip_zeros. */
   void ComputeAndAssemblePotBdrFaceMatrix(int bface, DenseMatrix & elmat,
                                           Array<int>& vdofs, int skip_zeros = 1);

   /// Assemble the boundary element matrix A into the hybridized system matrix.
   //void AssembleBdrMatrix(int bdr_el, const DenseMatrix &A);

   /** @brief Whether the mesh carries a nonconforming MASTER face at all --
       the coarse side of a hanging node.

       The gate on every route that assembles a face term in that face's own
       trace dofs; see GetNCMasterSlaves() for why a master cannot be one of
       those. O(number of masters) and short-circuits on a conforming mesh.

       Public, and it was private here until the trunk lifted it: on the trunk
       it is the trigger of Finalize()'s refusal of a nonlinear face
       constraint on such a mesh, so a caller -- or a test -- has to be able
       to ask without tripping that. This branch implements the expansion the
       trunk refuses and admits the configuration, but both branches ask the
       question the same way, which is the point of lifting the accessor
       rather than copying it. */
   bool HasNCMasterFaces() const;

   /// Finalize the construction of the hybridized matrix.
   void Finalize() override;

   /// Use the stored eliminated part of the system to modify the r.h.s.
   /** @param vdofs_flux   list of VDOFs of flux @a u
       @param x            solution vector providing the VDOF values
       @param b            right hand side vector
   */
   void EliminateVDofsInRHS(const Array<int> &vdofs_flux,
                            const BlockVector &x, BlockVector &b);

   /// Use the stored eliminated part of the system to modify the r.h.s.
   /** @param tdofs_flux   list of true DOFs of flux @a u
       @param X            solution vector providing the true DOF values
       @param B            (true) right hand side vector
   */
   virtual void EliminateTrueDofsInRHS(const Array<int> &tdofs_flux,
                                       const BlockVector &X, BlockVector &B);

   /// Eliminate the given true DOFs, storing the eliminated part internally.
   /** This method works in conjunction with EliminateTraceTrueDofsInRHS() and
       allows elimination of boundary conditions in multiple right-hand sides.
       In this method, @p tdofs is a list of true DOFs. */
   void EliminateTraceTrueDofs(const Array<int> &tdofs,
                               DiagonalPolicy dpolicy = DIAG_ONE);

   /// Eliminate the essential true DOFs.
   /** This method eliminates the essential true DOFs set previously through
       SetEssentialTrueDofs() (or derived methods). See EliminateTraceTrueDofs(
       const Array<int> &, DiagonalPolicy) for details. */
   void EliminateTraceTrueDofs(DiagonalPolicy dpolicy = DIAG_ONE);

   /// Use the stored eliminated part of the hybridized matrix to modify r.h.s.
   /** This method works in conjunction with EliminateTraceTrueDofs(
       const Array<int>&, DiagonalPolicy) to modify the r.h.s.
       @param vdofs     list of true DOFs (non-directional, i.e. >= 0)
       @param x         solution vector providing the true DOF values
       @param b         right hand side vector
    */
   void EliminateTraceTrueDofsInRHS(const Array<int> &vdofs, const Vector &x,
                                    Vector &b);

   /// Use the stored eliminated part of the hybridized matrix to modify r.h.s.
   /** This method works in conjunction with EliminateTraceTrueDofs(
       DiagonalPolicy) to modify the r.h.s.
       @param x         solution vector providing the  true DOF values
       @param b         right hand side vector
    */
   void EliminateTraceTrueDofsInRHS(const Vector &x, Vector &b);

   /// Return the eliminated part of the hybridized matrix.
   /** See EliminateTraceTrueDofs() for generation of this matrix. */
   SparseMatrix& GetMatrixElim() const { return *He; };

#ifdef MFEM_USE_MPI
   /// Return the parallel hybridized operator.
   void GetParallelOperator(OperatorHandle &H_h) const { H_h = pOp; }
#endif //MFEM_USE_MPI

   /// Not available, use ReduceRHS(const BlockVector &, Vector &) instead.
   void ReduceRHS(const Vector &b, Vector &b_r) const override
   { MFEM_ABORT("Use BlockVector version instead"); }

   /// Hybridize r.h.s. of the mixed system.
   /** @param b      r.h.s. of the mixed system (VDOFs)
       @param b_r    r.h.s of the hybridized system (TDOFs), IN and out

       If @a b_r already has the size of the hybridized system, the reduction
       of @a b is ADDED to what it holds; otherwise it is resized and zeroed
       first. Every branch does this: the direct scatter when the trace has
       no prolongation, AddMultTranspose() when it has one or in parallel.
       It is deliberate. A caller brings a load assembled on the trace (a
       Neumann condition, say) in by preloading @a b_r, which is what
       miniapps/hdg/darcyop.cpp does. So a fresh right-hand side needs @a b_r
       empty or zeroed, and calling twice on the same vector returns the sum.
       Hybridization::ReduceRHS(), behind BilinearForm, and the Darcy
       reductions overwrite; this one does not. On the nonlinear route nothing
       is added: @a b_r is only sized, and what it holds is left in place.
    */
   void ReduceRHS(const BlockVector &b, Vector &b_r) const;

   /// Projects trace of the solution onto the trace variable
   /** @note The trace projection performs simple averaging of the face values,
       which may not be consistent with the implicit definition in the
       hybridized system. Therefore, the values should serve only as an
       approximation or an initial guess.
       @param sol    solution of the mixed system (VDOFs)
       @param sol_r  solution of the hybridized system (VDOFs)
    */
   void ProjectSolution(const BlockVector &sol, Vector &sol_r) const;

   /// Apply the hybridized operator.
   /** @note The DarcyHybridization object must be finalized by Finalize(). */
   void Mult(const Vector &x, Vector &y) const override;

   /// Evaluate the gradient operator at the point @a x.
   Operator &GetGradient(const Vector &x) const override;

   /** @brief Not available, use ComputeSolution(const BlockVector &,
       const Vector &, BlockVector &) instead. */
   void ComputeSolution(const Vector &b, const Vector &sol_r,
                        Vector &sol) const override
   { MFEM_ABORT("Use BlockVector version instead"); }

   /// Compute solution of the mixed system.
   /** @param b      r.h.s. of the mixed system (VDOFs)
       @param sol_r  solution of the hybridized system (TDOFs)
       @param sol    solution of the mixed system (TDOFs)
    */
   void ComputeSolution(const BlockVector &b, const Vector &sol_r,
                        BlockVector &sol) const;

   /// Total flux function, for one field.
   /** @param Tr  element transformation (with set integration point)
       @param u   flux at the integration point
       @param p   potential at the integration point
       @param ut  total flux at the integration point
   */
   using total_flux_fun =
      std::function<void(ElementTransformation &Tr, const Vector &u, real_t p,
                         Vector &ut)>;

   /** @brief The flux law of a *system*, evaluated at a quadrature point.

       As #total_flux_fun, but the potential is a vector with one entry per
       equation. @a u is the flux, @a p the potential and @a ut the total
       flux, all of them per equation: for `neq` equations in `dim` dimensions
       @a p has `neq` entries and @a u and @a ut have `neq*dim`, with the block
       of equation `e` occupying `[e*dim, (e+1)*dim)`. That is the layout the
       block integrators build; see VectorBlockDiagonalIntegrator.

       This is a second type rather than a widening of #total_flux_fun so that
       a single-field caller written against the scalar signature goes on
       compiling. The two overloads of ReconstructTotalFlux() below cannot be
       ambiguous: `Vector(int)` is `explicit`, so a `real_t` argument never
       converts to a `const Vector &`, and a callable taking one is viable for
       exactly one of them. */
   using total_flux_sys_fun =
      std::function<void(ElementTransformation &Tr, const Vector &u,
                         const Vector &p, Vector &ut)>;

   /// Reconstruct the total flux from the provided solution.
   /** The total flux function is normally continuous and its finite element
       space is assumed to have equal number of DOFs at faces as the trace
       variable. For the interiors of elements, the quadrature function must
       be provided to calculate the total flux from the provided flux and
       potential values.

       @note **Systems are supported**, and this note used to say they were
       not. @a ut, the constraint space and the potential must all carry the
       same number of fields; every block below is that many copies of a
       scalar one, field outermost. Two things carried the generalisation and
       both are worth knowing about, because neither shows up as an abort.
       The face solve inverts the *scalar* face mass, which MassIntegrator
       builds regardless of vdim, so it is factored once and applied once per
       field rather than once against an neq-times-too-long right-hand side.
       And the element interior's boundary/interior split is a scalar count:
       GetNumElementInteriorDofs() counts one field's, and the interior dofs
       are the tail of each field's block of the vdof list, not the tail of
       the list -- which is the same thing only when there is one field.
       @param sol    solution of the mixed system
       @param sol_r  solution of the hybridized system
       @param ut_fx  total flux function
       @param ut     total flux
   */
   void ReconstructTotalFlux(const BlockVector &sol, const Vector &sol_r,
                             total_flux_sys_fun ut_fx, GridFunction &ut) const;

   /** @brief Whether ReconstructTotalFluxBatched() can serve @a fes_ut.

       AssemblyMode::Batched; serial; a flux constraint
       HDGNormalTraceJumpCanBatch() takes and a potential constraint -- interior
       and boundary -- HDGFaceSideCanBatch() takes, none of it nonlinear; no
       essential flux dofs; and a scalar RT total flux space without a
       DofTransformation. Asked of the spaces and the integrators, so it
       builds no transformation. */
   bool CanBatchReconstructTotalFlux(const FiniteElementSpace &fes_ut) const;

   /** @brief ReconstructTotalFlux() for the IDENTITY flux law -- the total
       flux is the flux, as on a problem with no convective or hyperbolic
       term -- in two kernels: one thread per face for the face dofs, one per
       element for the interior ones.

       The face pass is the host route's arithmetic: C^T u from the stored C
       (a kernel-built block on a boundary face, where ConstructC() stores
       none), the potential constraint's trace rows G p + H x one side at a
       time, then the face mass solve and the halving of an interior face.
       One difference, and it is round-off: the face mass is inverted ONCE.
       MassIntegrator between the INTEGRAL-mapped total-flux face element and
       the VALUE-mapped trace one has the transformation weight cancel between
       the map and the measure, so every face of one geometry has that matrix.

       The element pass takes its geometry from HDGElementGeometry and its
       shapes from reference tables, so nothing in it is a host loop over
       elements but the index maps and, on a mesh without nodes, the packing
       of each element's vertices.

       Leaves @a ut wherever the kernels ran -- on the device under one -- and
       returns false, having done nothing, unless
       CanBatchReconstructTotalFlux(). */
   bool ReconstructTotalFluxBatched(const BlockVector &sol, const Vector &sol_r,
                                    GridFunction &ut) const;

   /// Reconstruct the total flux of a single field.
   /** Equivalent to the overload above with a one-entry potential; provided so
       that a scalar flux law needs no reshaping. Aborts if the problem carries
       more than one field, rather than silently using the first.
       @param sol    solution of the mixed system
       @param sol_r  solution of the hybridized system
       @param ut_fx  total flux function
       @param ut     total flux
   */
   void ReconstructTotalFlux(const BlockVector &sol, const Vector &sol_r,
                             total_flux_fun ut_fx, GridFunction &ut) const;

   /// Resets the assembled data
   /** @note Assumes topology of the mesh does not change, otherwise recreate
       the object. */
   void Reset() override;
};

/** @brief The NPC method as an ordinary MFEM Operator on the full
    (q, u, lambda) system, so that NewtonSolver or KINSolver can drive it.

    Pair it with DarcyNPCSolver, which inverts the Jacobian by hybridized
    elimination:

        DarcyNPCOperator npc(*darcy.GetHybridization(), offsets, load);
        DarcyNPCSolver   lin(trace_solver);
        NewtonSolver newton;
        newton.SetOperator(npc);
        newton.SetSolver(lin);
        newton.Mult(zero, x);        // x = (q, u, lambda)

    @a offsets is size 4 and partial-summed over the flux, potential and trace
    vector sizes; @a load is the (flux, potential) right-hand side.

    **The unknown is the whole system, and that is the point.** A reduced
    operator on the trace alone cannot be NPC, because the fields would have to
    be a function of the trace and here they are Newton state -- which is
    exactly what an Operator over all three blocks gives them. So the outer
    solver needs no special support, its convergence test is on the full
    residual by construction, and its line search scales the fields and the
    trace together. An earlier version of this comment said NewtonSolver could
    not drive NPC "because it has nowhere to keep the fields"; it keeps them in
    @a x, and the claim was about a trace-only operator rather than about
    NewtonSolver. */
class DarcyNPCOperator : public Operator
{
public:
   DarcyNPCOperator(DarcyHybridization &dh, const Array<int> &offsets,
                    const BlockVector &load);

   /// The full residual F(q, u, lambda). No local solve of any kind.
   void Mult(const Vector &x, Vector &y) const override;

   /** @brief Assemble and factor the Jacobian at @a x.

       The returned handle is **solve-only** and its Mult() aborts: after
       ComputeH() the local arrays hold the FACTORED blocks and the Schur
       complement, so the Jacobian can no longer be applied out of them. Only
       DarcyNPCSolver understands it. That is a real constraint of hybridized
       elimination and not an oversight -- applying J would need unfactored
       copies of every local block. */
   Operator &GetGradient(const Vector &x) const override;

   /// The (q, u) sub-blocks of the full vector, for a caller splitting it.
   const Array<int> &LocalOffsets() const { return loc_offsets; }

   /// The handle GetGradient() returns; DarcyNPCSolver takes it.
   class Jacobian : public Operator
   {
   public:
      Jacobian(DarcyHybridization &dh_, Operator &S_, const Array<int> &offs,
               const Array<int> &loc_offs)
         : Operator(offs.Last()), dh(dh_), S(S_), offsets(offs),
           loc_offsets(loc_offs) { }
      void Mult(const Vector &, Vector &) const override
      {
         MFEM_ABORT("The NPC Jacobian is solve-only: the local blocks are "
                    "factored in place, so it cannot be applied. Use "
                    "DarcyNPCSolver.");
      }
      DarcyHybridization &dh;
      Operator &S;                  ///< reduced trace operator, assembled or not
      const Array<int> &offsets;    ///< {0, flux, potential, trace}
      const Array<int> &loc_offsets;///< {0, flux, potential}
   };

private:
   DarcyHybridization *dh;
   const BlockVector &load;
   Array<int> offsets, loc_offsets;
   mutable BlockVector r_loc, x_loc;
   mutable Vector r_tr;
   mutable std::unique_ptr<Jacobian> jac;
};

/** @brief Solves the Jacobian of DarcyNPCOperator by hybridized elimination:
    reduce to the trace, solve there, recover the local increments.

    The trace solve is the caller's: pass any Solver. With
    DarcyHybridization::GradientMode::Assembled it receives a SparseMatrix, so
    a direct solver or an algebraic preconditioner works; with MatrixFree it
    receives an operator that only applies S, so it must be a Krylov method
    that needs no matrix. */
class DarcyNPCSolver : public Solver
{
public:
   /// @a trace_solver is used for the reduced trace system, once per step.
   explicit DarcyNPCSolver(Solver &trace_solver);

   /// Expects the handle from DarcyNPCOperator::GetGradient().
   void SetOperator(const Operator &op) override;

   /// @a b is the outer residual; @a x comes back as the Newton CORRECTION,
   /// in that solver's convention of x_new = x - correction.
   void Mult(const Vector &b, Vector &x) const override;

   /** @brief Apply the same factored Jacobian to several right-hand sides in
       one pass.

       The three legs block together: one pass over the mesh in
       DarcyHybridization::NPCReduce(), one call to the trace solver's
       ArrayMult(), one pass back in NPCRecover(). A trace solver that
       overrides ArrayMult() -- MUMPSSolver, SuperLUSolver, STRUMPACKSolver,
       CuDSSSolver, PardisoSolver -- then walks its factors once as well;
       Operator::ArrayMult()'s base implementation loops Mult(), so any other
       solver still works and simply saves nothing in that leg.

       This is the entry point for a bordered, deflated, parameter-continued
       or adjoint solve, which is where several right-hand sides share one
       Jacobian. Because it is an Operator method, such a caller is written
       against Operator and is not tied to hybridized Darcy at all.

       @a X.Size() == 1 forwards to Mult() and allocates nothing, so the
       single-right-hand-side path is exactly what it was.

       **Outputs that should stay on the device must be device-valid on
       entry** -- `Y[j]->Write()` before the call. Each column is written
       through its blocks and then reconciled with
       BlockVector::SyncFromBlocks(), which moves every block to wherever the
       PARENT's flags say the data lives; a host-valid @a Y therefore has
       every block downloaded on the spot (meq: 504 copies and 140 MB per run
       on a 14-column case), and a device-valid one has none. Mult() is the
       same for its one output. */
   void ArrayMult(const Array<const Vector *> &X,
                  Array<Vector *> &Y) const override;

   /** @brief Seconds Mult() and ArrayMult() have spent OUTSIDE the two
       traversal halves and the trace solve, since ResetStagingTime().

       That is, moving each column between the caller's layout and the
       local one, and the sign flip into the Newton convention. With the
       halves (DarcyHybridization::GetNPCReduceTime() and
       GetNPCRecoverTime()) and a caller's own timing of the trace solver,
       it closes the leg. A bordered Newton that subtracted the other three
       from the whole call found a remainder flat in the thread count, a
       third of the leg at eight threads, and this is the timer that
       measures it rather than infers it.

       Static, process-wide, NOT thread safe and wall clock, for
       DarcyHybridization::GetNPCTraversalTime()'s reasons. */
   static real_t GetStagingTime();
   /// Zero the staging accumulator; see GetStagingTime().
   static void ResetStagingTime();

private:
   Solver &trace_solver;
   const DarcyNPCOperator::Jacobian *jac{nullptr};
   mutable BlockVector r_loc, dx_loc;
   mutable Vector r_tr, b_tr, dtr;
};

}

#endif
