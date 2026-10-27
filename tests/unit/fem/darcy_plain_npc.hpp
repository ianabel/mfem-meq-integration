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

#ifndef MFEM_UNIT_FEM_DARCY_PLAIN_NPC_HPP
#define MFEM_UNIT_FEM_DARCY_PLAIN_NPC_HPP

#include "mfem.hpp"

/** @file The PLAIN NPC path -- every batched mode left at its default -- with
    an answer that is arithmetic rather than a second run of the same code.

    **Why this exists, and why it is a header shared by two binaries.** meq
    reports (their M-79) that a whole NPC solve does not survive an
    mfem::Device: Newton takes zero steps where it needs four, and the flux
    disagrees by 4.006e-02, identically across three different trace solvers --
    two of which never touch the device at all. Their configuration turns on
    NONE of this class's batched modes.

    Every existing [DebugDevice] case compares two device-configured arms of
    the same problem against each other, because the Device belongs to the
    binary and no no-Device arm is available inside it; and every one of them
    switches a batched mode on somewhere. Two arms agreeing says nothing when
    the fault is common to both, which is exactly what three trace solvers
    returning the same wrong answer looks like. So the check here is against
    an EXACT solution: the discrete answer is known before the code runs.

    Compiled into unit_tests (no Device) and into debug_device_tests (which
    configures Device("debug")) from this one header, so the two differ in the
    Device and in nothing else. That is the "two TEST_CASEs over one static
    helper" shape this tree already uses for the SuiteSparse BLAS check.

    **The problem, and why the answer is exact.** The mixed Darcy system

        u + grad p = 0        - div u = 0

    on the unit square, with p = x + 2y and therefore u = -(1, 2) constant.
    Both lie in the discrete spaces at any order >= 1 and p restricted to a
    face lies in the trace space, so by consistency of the HDG method the
    discrete solution IS the exact one, to round-off, on ANY mesh. There is no
    discretisation error to hide a defect behind and no mesh refinement to run.

    **The structure is meq's, deliberately.** A domain integrator on the
    potential mass NONLINEAR form with the HDG face stabilization beside it,
    which is what sends DarcyForm::EnableHybridization() down its
    `Mnl_p && FaceIntegratorsAreLinear(...)` branch and installs the face
    constraint as the LINEAR c_bfi_p -- the route meq takes. That is this
    branch's own "look for the inert knob" note used as a construction rather
    than as a probe.

    **The inert knob is a blind spot as well as a construction, which is why
    @a mass exists.** A contribution whose correct value is exactly zero is
    indistinguishable from a contribution that was never read back. With
    `mass = 0` the domain term above is arithmetically inert, so on its own
    this case cannot tell a device that evaluates it from one that silently
    drops it -- which is the failure mode it exists to catch.
    Their own fixture puts a constant non-zero F on that integrator and gets
    exactly zero back.

    So `mass = a != 0` adds `a (p, phi)` to the potential row and the matching
    load `a (p_exact, phi)` to g, and BOTH halves are then individually
    detectable:

      * drop the domain term and the discrete problem becomes `lap p = a
        p_exact`, whose solution is not `x + 2y`;
      * drop the load and it becomes `lap p + a p = 0`, whose solution is not
        `x + 2y` either.

    Either way `err_p` leaves round-off, where with `mass = 0` neither could
    move it. **Both halves were checked by dropping them**, which is the only
    thing that makes this arm worth its build: with the domain term dropped
    `err_u` comes back 0.835 to 0.858, with the load dropped 0.739 to 0.754,
    against a threshold of 1e-10 -- and with `mass = 0` dropping either is
    invisible, which is the whole point meq was making. The exact solution is unchanged because `u = -grad p` and
    `-div u + a p = a p_exact` hold pointwise for `p = x + 2y`, and both
    integrands are polynomials the default rules integrate exactly on an
    affine mesh.

    The problem is affine, so ONE NPC step from any starting point is the
    answer; nothing here iterates, and a step that fails to land is a defect
    and not slow convergence.

    **Which route this runs, measured rather than assumed.**
    `CanBatchLinearResidual()` comes back FALSE here at either `mass`, and
    that is the right answer rather than a disappointment: a non-null
    `Mnl_p` makes `IsNonlinear()` true with no flux or block nonlinearity
    beside it, so `Finalize()` picks `LocalOpType::PotNL`, and the predicate
    refuses PotNL outright -- it cannot freeze a gradient to match a frozen
    residual there. So the element loop is the per-element one, which is
    **the same local operator meq runs**, their source integrator being
    genuinely nonlinear and refused by the same predicate one line earlier.
    The case pins it with REQUIRE_FALSE so that a change which silently moved
    this onto the batched route would fail here rather than quietly stop
    being meq's configuration. */
namespace darcy_plain_npc
{
using namespace mfem;

inline real_t PExact(const Vector &x) { return x(0) + 2.0 * x(1); }

inline void UExact(const Vector &, Vector &u) { u(0) = -1.0; u(1) = -2.0; }

/// What one plain NPC step produced, and enough to tell HOW it failed.
struct Result
{
   real_t err_u{};      ///< L2 error of the flux against UExact
   real_t err_p{};      ///< L2 error of the potential against PExact
   real_t r0_fields{};  ///< ||initial residual|| on the flux+potential blocks
   real_t r0_trace{};   ///< ||initial residual|| on the trace block
   real_t step{};       ///< ||the Newton increment||, fields and trace
   real_t datum{};      ///< max |essential trace datum|, so it is known nonzero
   real_t load{};       ///< max |g|, so a non-zero source is known to be live
   bool batched_residual{};  ///< whether LinearResidualBatched() is taken
   real_t r0_operator{}; ///< ||F(x0)|| taken THROUGH DarcyNPCOperator::Mult
   int its{-1};          ///< Newton iterations, when driven through the operator
   bool converged{};     ///< ... and whether it said so
   bool r_dev_valid{};   ///< is F(x0) still DEVICE-valid straight out of Mult()
   int nonfinite{};     ///< CheckFinite() summed over everything returned
   int trace_size{};
};

/** One NPC step on the problem above at @a order on an @a n by @a n mesh.

    Nothing is set on the hybridization but NPC and the essential boundary:
    AssemblyMode, LocalFactorMode, TraceAssemblyMode and GradientMode are all
    left at their defaults, which is the whole point. @a geom is a parameter
    because meq runs TRIANGLES and every other Darcy device case in this tree
    runs quadrilaterals; the exact solution above is exact on both.

    @a mass is the coefficient on the potential-mass domain term and on the
    load that balances it; see the note above on why zero is not enough on its
    own. The routing does not depend on it -- MassIntegrator is a
    BilinearFormIntegrator at either value -- so the two arms differ in the
    arithmetic and in nothing else. */
/** @a via_operator drives the step through **DarcyNPCOperator and
    DarcyNPCSolver under a NewtonSolver**, which is the layer a caller
    actually runs: DarcyNPCOperator::Mult() is the residual a Newton
    evaluates. Calling NPCResidual() directly skips it, and a fault inside it
    is then invisible to every assertion here. */
inline Result Solve(int order, int n, Element::Type geom, real_t mass = 0.0,
                    bool via_operator = false)
{
   const int dim = 2;
   Mesh mesh = Mesh::MakeCartesian2D(n, n, geom);

   L2_FECollection u_coll(order, dim, BasisType::GaussLobatto);
   L2_FECollection p_coll(order, dim);
   DG_Interface_FECollection t_coll(order, dim);
   FiniteElementSpace Vh(&mesh, &u_coll, dim), Wh(&mesh, &p_coll),
                      Mh(&mesh, &t_coll);

   DarcyForm darcy(&Vh, &Wh);
   ConstantCoefficient one(1.0), mass_coeff(mass);
   FunctionCoefficient load_coeff([mass](const Vector &x)
   {
      return mass * PExact(x);
   });

   Array<int> all(mesh.bdr_attributes.Max());
   all = 1;

   darcy.GetFluxMassForm()->AddDomainIntegrator(new VectorMassIntegrator(one));

   // The boundary face integrator is read for its MARKER and for nothing
   // else: DarcyForm::Assemble() installs the constraint on those attributes,
   // and without one the constraint is interior-only and every element
   // touching the boundary is wrong. Same trap as the Navier-Stokes miniapp's.
   MixedBilinearForm *B = darcy.GetFluxDivForm();
   B->AddDomainIntegrator(new VectorDivergenceIntegrator());
   B->AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)), all);

   // Everything on the potential block goes on the NONLINEAR form, which is
   // meq's structural decision and is what selects the c_bfi_p route here.
   // The domain term is decisive in the routing at either coefficient, and
   // arithmetically live only at a non-zero one.
   NonlinearForm *Mnl_p = darcy.GetPotentialMassNonlinearForm();
   Mnl_p->AddDomainIntegrator(new MassIntegrator(mass_coeff));
   Mnl_p->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));
   Mnl_p->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));

   // The load that balances it. Asked for ONLY when it is wanted: the
   // non-const accessors CONSTRUCT the form they return, and merely calling
   // GetPotentialMassForm() is enough to silence a face constraint placed on
   // the nonlinear form.
   if (mass != 0.0)
   {
      darcy.GetPotentialRHS()->AddDomainIntegrator(
         new DomainLFIntegrator(load_coeff));
   }

   Array<int> ess_flux;
   darcy.EnableHybridization(&Mh, new NormalTraceJumpIntegrator(), ess_flux);

   DarcyHybridization *dh = darcy.GetHybridization();
   dh->EnableNPC();
   dh->SetEssentialBC(all);

   darcy.Assemble();
   darcy.Finalize();

   Result res;
   res.trace_size = Mh.GetVSize();

   // The Dirichlet datum, on the boundary trace dofs. ProjectBdrCoefficient
   // and not ProjectCoefficient: the latter segfaults on a face-based space.
   GridFunction tr(&Mh);
   tr = 0.0;
   FunctionCoefficient pcoeff(PExact);
   tr.ProjectBdrCoefficient(pcoeff, all);

   Vector x_tr(Mh.GetVSize());
   x_tr = 0.0;
   Array<int> bdr_dofs;
   Mh.GetEssentialTrueDofs(all, bdr_dofs);
   for (int i = 0; i < bdr_dofs.Size(); i++)
   {
      const int d = bdr_dofs[i];
      x_tr(d) = tr(d);
      res.datum = std::max(res.datum, std::abs(tr(d)));
   }

   BlockVector b(darcy.GetOffsets()), x(darcy.GetOffsets());
   b = 0.0;   // f = 0; with mass = 0, g = 0 too and the datum drives it alone

   if (mass != 0.0)
   {
      // **MINUS, and the sign is pinned by the residual's own convention
      // rather than by taste.** LinearResidualBatched() states the potential
      // row as `rp = B u + D p + E x - sgn bp` with `sgn = bsym ? -1 : 1`,
      // and DarcyForm's bsymmetrize defaults to TRUE -- so a term that
      // arrives in D with a plus has to arrive in g with a minus. Measured
      // both ways before it was believed: the other sign gives err_u 1.53 and
      // err_p 0.327 where this one gives round-off.
      //
      // SyncAliasMemory() because a block of a BlockVector is an alias with
      // its own validity flags, and a host write through it leaves the base
      // unaware.
      darcy.GetPotentialRHS()->Assemble();
      b.GetBlock(1) -= *darcy.GetPotentialRHS();
      b.GetBlock(1).SyncAliasMemory(b);
      res.load = b.GetBlock(1).Normlinf();
   }

   // A NON-ZERO start, and it is not arbitrary. From x = 0 with the datum on
   // the boundary trace alone, the trace block of the residual is EXACTLY
   // zero -- the fields are zero, so the interior numerical flux is zero, so
   // the conservativity rows have nothing in them -- and a check on it would
   // assert what the starting point guarantees rather than what the code
   // does. Measured: r0_trace came back 0 at every order and mesh. Starting
   // away from zero puts all three blocks in play. The problem is affine, so
   // one step lands on the answer from anywhere.
   x = 1.0;

   if (via_operator)
   {
      // The recipe documented on DarcyNPCOperator, followed exactly.
      Array<int> npc_offs(4);
      npc_offs[0] = 0;
      npc_offs[1] = npc_offs[0] + Vh.GetVSize();
      npc_offs[2] = npc_offs[1] + Wh.GetVSize();
      npc_offs[3] = npc_offs[2] + Mh.GetVSize();

      BlockVector X(npc_offs);
      X.GetBlock(0) = x.GetBlock(0);
      X.GetBlock(1) = x.GetBlock(1);
      X.GetBlock(2) = x_tr;
      X.SyncFromBlocks();

      DarcyNPCOperator npc(*dh, npc_offs, b);

      // **The initial residual taken THROUGH the operator**, which is the one
      // quantity meq reports as identically zero. Taken before the solve,
      // because a Newton handed a zero residual reports convergence at
      // iteration zero and leaves the fields at the initial iterate -- so
      // this and @a its below are the two halves of M-79 as assertions.
      Vector R(X.Size());
      // **UseDevice(true), because that is what the caller does.**
      // NewtonSolver::SetOperator() sets it unconditionally on its residual
      // and correction, so a bare Vector here would take the host path and
      // exercise none of what the operator has to get right -- the probe
      // would be measuring a configuration nobody runs.
      R.UseDevice(true);
      R = 0.0;
      npc.Mult(X, R);
      // **Sampled BEFORE any host read, and it asserts where the data IS
      // rather than what it says.** meq's point, and it is a good one: a
      // check that ends in a host read cannot tell "the flags were repaired"
      // from "the data was dragged to the host", because SyncAlias() will
      // migrate device-valid data down and unprotect it. So the one-hop
      // version of DarcyNPCOperator::Mult() is accidentally correct at the
      // price of a d2h round trip per residual, and no value-based assertion
      // anywhere can see the difference. This one can.
      res.r_dev_valid = R.GetMemory().DeviceIsValid();
      R.HostRead();
      res.r0_operator = R.Norml2();
      res.nonfinite += R.CheckFinite();

      UMFPackSolver umf;
      DarcyNPCSolver lin(umf);
      NewtonSolver newton;
      newton.SetOperator(npc);
      newton.SetSolver(lin);
      newton.SetRelTol(1e-13);
      newton.SetAbsTol(1e-14);
      newton.SetMaxIter(20);
      newton.SetPrintLevel(-1);
      Vector zero(X.Size());
      zero = 0.0;
      newton.Mult(zero, X);
      res.its = newton.GetNumIterations();
      res.converged = newton.GetConverged();

      X.HostRead();
      res.nonfinite += X.CheckFinite();
      res.batched_residual = dh->CanBatchLinearResidual();

      GridFunction qo(&Vh), po(&Wh);
      qo.MakeRef(&Vh, X.GetBlock(0), 0);
      po.MakeRef(&Wh, X.GetBlock(1), 0);
      VectorFunctionCoefficient uc(dim, UExact);
      res.err_u = qo.ComputeL2Error(uc);
      res.err_p = po.ComputeL2Error(pcoeff);
      return res;
   }

   BlockVector r(darcy.GetOffsets());
   Vector r_tr, b_tr, dtr;
   dh->NPCResidual(b, x, x_tr, r, r_tr);
   // AFTER the residual: the route builds its cache on first use, so asking
   // before reports the question rather than the answer. Recorded because
   // WHICH route this case exercises is the point of it -- meq's own
   // configuration cannot take this one, their source integrator being
   // genuinely nonlinear.
   res.batched_residual = dh->CanBatchLinearResidual();
   res.r0_fields = r.Norml2();
   res.r0_trace = r_tr.Norml2();

   Operator &S = dh->NPCGradient(x, x_tr);
   dh->NPCReduce(r, r_tr, b_tr);

   dtr.SetSize(b_tr.Size());
   dtr = 0.0;
   {
      SparseMatrix *Sm = dynamic_cast<SparseMatrix*>(&S);
#ifdef MFEM_USE_SUITESPARSE
      if (Sm)
      {
         UMFPackSolver umf(*Sm);
         umf.Mult(b_tr, dtr);
      }
      else
#endif
      {
         std::unique_ptr<GSSmoother> prec;
         if (Sm) { prec.reset(new GSSmoother(*Sm)); }
         GMRESSolver gmres;
         gmres.SetOperator(S);
         if (prec) { gmres.SetPreconditioner(*prec); }
         gmres.SetKDim(200);
         gmres.SetMaxIter(5000);
         gmres.SetRelTol(1e-14);
         gmres.SetAbsTol(0.0);
         gmres.SetPrintLevel(-1);
         gmres.Mult(b_tr, dtr);
      }
   }

   BlockVector dx(darcy.GetOffsets());
   dx = 0.0;
   dh->NPCRecover(r, dtr, dx);

   // NPCReduce()/NPCRecover() give the increment to ADD; the sign flip lives
   // in DarcyNPCSolver, which is not on this path.
   x += dx;
   x_tr += dtr;
   res.step = std::sqrt(dx.Norml2() * dx.Norml2() + dtr.Norml2() * dtr.Norml2());

   // Norml2() CANNOT see a NaN -- it guards its reduction with fabs(v) > 0,
   // which is false for NaN, so it returns the norm of the remainder and an
   // all-NaN vector reads as zero. Every norm above is therefore worthless
   // until this says so.
   res.nonfinite = x.CheckFinite() + x_tr.CheckFinite() + r.CheckFinite()
                   + r_tr.CheckFinite() + dtr.CheckFinite();

   GridFunction q(&Vh), p(&Wh);
   q.MakeRef(&Vh, x.GetBlock(0), 0);
   p.MakeRef(&Wh, x.GetBlock(1), 0);
   VectorFunctionCoefficient ucoeff(dim, UExact);
   res.err_u = q.ComputeL2Error(ucoeff);
   res.err_p = p.ComputeL2Error(pcoeff);

   return res;
}

/// A potential mass `(c p^3, w)`: nonlinear, so the Jacobian depends on the
/// state and a Jacobian taken at the wrong state is a wrong Jacobian. The
/// affine problem above cannot tell the two apart.
class CubicMass : public NonlinearFormIntegrator
{
public:
   explicit CubicMass(real_t c_) : c(c_) { }

   void AssembleElementVector(const FiniteElement &el,
                              ElementTransformation &Tr,
                              const Vector &elfun, Vector &elvect) override
   {
      const int dof = el.GetDof();
      shape.SetSize(dof);
      elvect.SetSize(dof);
      elvect = 0.0;
      const IntegrationRule &ir = IntRules.Get(el.GetGeomType(),
                                               4*el.GetOrder() + 2);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         el.CalcPhysShape(Tr, shape);
         const real_t u = shape * elfun;
         elvect.Add(ip.weight * Tr.Weight() * c * u * u * u, shape);
      }
   }

   void AssembleElementGrad(const FiniteElement &el, ElementTransformation &Tr,
                            const Vector &elfun, DenseMatrix &elmat) override
   {
      const int dof = el.GetDof();
      shape.SetSize(dof);
      elmat.SetSize(dof);
      elmat = 0.0;
      const IntegrationRule &ir = IntRules.Get(el.GetGeomType(),
                                               4*el.GetOrder() + 2);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         el.CalcPhysShape(Tr, shape);
         const real_t u = shape * elfun;
         AddMult_a_VVt(ip.weight * Tr.Weight() * 3.0 * c * u * u, shape, elmat);
      }
   }

private:
   real_t c;
   Vector shape;
};

/// The two Newton runs NonlinearNewton() compares.
struct NewtonAgreement
{
   int its_operator{-1}, its_legs{-1};
   bool converged_operator{}, converged_legs{};
   real_t first_step_operator{}, first_step_legs{};
   real_t diff{};    ///< ||X_operator - X_legs||_inf at the end
   real_t scale{};   ///< ||X_legs||_inf
   int nonfinite{};
};

/** @brief A NONLINEAR NPC solve two ways in one process: through
    DarcyNPCOperator + DarcyNPCSolver under a NewtonSolver, and through a
    Newton loop written out on the legs (NPCResidual, NPCGradient, NPCReduce,
    a direct trace solve, NPCRecover) on host-synced vectors.

    **This is meq's defect, and the affine Solve() above could not see it.**
    DarcyNPCOperator::Mult() and GetGradient() copy the state into @a x_loc's
    BLOCKS; a block is an alias with its own validity flags, so under a Device
    @a x_loc itself went on claiming a stale copy, and the Jacobian was
    assembled at the wrong state. On an affine problem the Jacobian does not
    depend on the state, so a Jacobian at the wrong state is still the right
    one and every assertion passes. meq measured the symptom on their real
    case as a Newton increment 36x too small and 4x the iterations, with the
    answer right in the end -- so a convergence assertion cannot catch it
    either. What catches it is agreement STEP BY STEP with a route that does
    not go through the operator: the first increment and the iteration count.
    Under no Device the two arms are the same arithmetic. */
inline NewtonAgreement NonlinearNewton(int order, int n, Element::Type geom)
{
   const int dim = 2;
   Mesh mesh = Mesh::MakeCartesian2D(n, n, geom);

   L2_FECollection u_coll(order, dim, BasisType::GaussLobatto);
   L2_FECollection p_coll(order, dim);
   DG_Interface_FECollection t_coll(order, dim);
   FiniteElementSpace Vh(&mesh, &u_coll, dim), Wh(&mesh, &p_coll),
                      Mh(&mesh, &t_coll);

   DarcyForm darcy(&Vh, &Wh);
   ConstantCoefficient one(1.0);
   Array<int> all(mesh.bdr_attributes.Max());
   all = 1;

   darcy.GetFluxMassForm()->AddDomainIntegrator(new VectorMassIntegrator(one));
   MixedBilinearForm *B = darcy.GetFluxDivForm();
   B->AddDomainIntegrator(new VectorDivergenceIntegrator());
   B->AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)), all);
   NonlinearForm *Mnl_p = darcy.GetPotentialMassNonlinearForm();
   Mnl_p->AddDomainIntegrator(new CubicMass(4.0));
   Mnl_p->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));
   Mnl_p->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));

   Array<int> ess_flux;
   darcy.EnableHybridization(&Mh, new NormalTraceJumpIntegrator(), ess_flux);
   DarcyHybridization *dh = darcy.GetHybridization();
   dh->EnableNPC();
   dh->SetEssentialBC(all);
   darcy.Assemble();
   darcy.Finalize();

   // The datum drives it: p = x + 2y on the boundary, a unit source inside.
   GridFunction tr(&Mh);
   tr = 0.0;
   FunctionCoefficient pcoeff(PExact);
   tr.ProjectBdrCoefficient(pcoeff, all);
   Vector x_tr0(Mh.GetVSize());
   x_tr0 = 0.0;
   Array<int> bdr_dofs;
   Mh.GetEssentialTrueDofs(all, bdr_dofs);
   for (int i = 0; i < bdr_dofs.Size(); i++) { x_tr0(bdr_dofs[i]) = tr(bdr_dofs[i]); }

   BlockVector b(darcy.GetOffsets());
   b = 0.0;
   b.GetBlock(1) = -1.0;   // minus: see Solve()'s note on the sign
   b.GetBlock(1).SyncAliasMemory(b);

   Array<int> npc_offs(4);
   npc_offs[0] = 0;
   npc_offs[1] = Vh.GetVSize();
   npc_offs[2] = npc_offs[1] + Wh.GetVSize();
   npc_offs[3] = npc_offs[2] + Mh.GetVSize();

   NewtonAgreement res;
   const real_t rtol = 1e-12;
   const int max_it = 30;

   // ---- the legs, on host-synced vectors: the reference.
   Vector X_legs;
   {
      BlockVector x(darcy.GetOffsets()), r(darcy.GetOffsets()),
                  dx(darcy.GetOffsets());
      x = 1.0;
      Vector x_tr(x_tr0), r_tr, b_tr, dtr;
      real_t r0 = -1.0;
      UMFPackSolver umf;
      for (int it = 0; it <= max_it; it++)
      {
         dh->NPCResidual(b, x, x_tr, r, r_tr);
         r.HostRead(); r_tr.HostRead();
         const real_t nr = std::sqrt(r*r + r_tr*r_tr);
         if (it == 0) { r0 = nr; }
         if (nr <= rtol * r0) { res.its_legs = it; res.converged_legs = true; break; }
         if (it == max_it) { res.its_legs = it; break; }
         Operator &S = dh->NPCGradient(x, x_tr);
         dh->NPCReduce(r, r_tr, b_tr);
         umf.SetOperator(S);
         dtr.SetSize(b_tr.Size());
         dtr = 0.0;
         umf.Mult(b_tr, dtr);
         dh->NPCRecover(r, dtr, dx);
         dx.HostRead(); dtr.HostRead();
         if (it == 0)
         {
            res.first_step_legs = std::sqrt(dx*dx + dtr*dtr);
         }
         x.HostReadWrite();
         x += dx;
         x_tr.HostReadWrite();
         x_tr += dtr;
      }
      X_legs.SetSize(npc_offs.Last());
      BlockVector Xb(X_legs, npc_offs);
      x.HostRead(); x_tr.HostRead();
      Xb.GetBlock(0) = x.GetBlock(0);
      Xb.GetBlock(1) = x.GetBlock(1);
      Xb.GetBlock(2) = x_tr;
      Xb.SyncFromBlocks();
      X_legs.HostRead();
      res.nonfinite += X_legs.CheckFinite();
   }

   // ---- the operator, the layer a caller runs.
   BlockVector X(npc_offs);
   X.GetBlock(0) = 1.0;
   X.GetBlock(1) = 1.0;
   X.GetBlock(2) = x_tr0;
   X.SyncFromBlocks();
   // The Newton state lives where NewtonSolver's work vectors live, as it
   // does for any caller: its update x -= c is a device operation under a
   // Device, so every iterate after the first reaches the operator
   // device-resident.
   X.UseDevice(true);

   DarcyNPCOperator npc(*dh, npc_offs, b);
   UMFPackSolver umf;
   DarcyNPCSolver lin(umf);
   {
      // The first increment through the operator's own Jacobian.
      Vector c(X.Size()), R(X.Size());
      c.UseDevice(true);
      R.UseDevice(true);
      npc.Mult(X, R);
      lin.SetOperator(npc.GetGradient(X));
      c = 0.0;
      lin.Mult(R, c);
      c.HostRead();
      res.first_step_operator = c.Norml2();
      res.nonfinite += c.CheckFinite();
   }
   NewtonSolver newton;
   newton.SetOperator(npc);
   newton.SetSolver(lin);
   newton.SetRelTol(rtol);
   newton.SetAbsTol(0.0);
   newton.SetMaxIter(max_it);
   newton.SetPrintLevel(-1);
   Vector zero(X.Size());
   zero = 0.0;
   newton.Mult(zero, X);
   res.its_operator = newton.GetNumIterations();
   res.converged_operator = newton.GetConverged();

   X.HostRead();
   res.nonfinite += X.CheckFinite();
   Vector d(X);
   d -= X_legs;
   res.diff = d.Normlinf();
   res.scale = X_legs.Normlinf();
   return res;
}

// MFEM_HOST_DEVICE, and that is not decoration: the law is called from the
// device lambdas below, and nvcc only WARNS (#20011-D) when a host function is
// called from a __host__ __device__ one -- the device then runs whatever it
// runs. Found when the route case was first tagged [GPU]: under CUDA the
// batched arm's potential residual came back entirely non-finite, while
// Device("debug"), which does its arithmetic on the host, saw nothing.
MFEM_HOST_DEVICE inline real_t CallerLaw(real_t u, real_t c)
{ return u + c * u * u * u; }
MFEM_HOST_DEVICE inline real_t CallerDLaw(real_t u, real_t c)
{ return 1.0 + 3.0 * c * u * u; }

/// The caller-integrand fixture's potential mass, `F(u) = u + c u^3` on an
/// explicit rule -- the shape NonlinearGradientBatched() accepts.
class CallerCubicNLF : public NonlinearFormIntegrator
{
   const real_t c;
public:
   CallerCubicNLF(real_t c_, const IntegrationRule *ir)
      : NonlinearFormIntegrator(ir), c(c_) { }
   void AssembleElementVector(const FiniteElement &el, ElementTransformation &Tr,
                              const Vector &elfun, Vector &elvect) override
   {
      const int nd = el.GetDof();
      Vector shape(nd);
      elvect.SetSize(nd);
      elvect = 0.0;
      for (int q = 0; q < IntRule->GetNPoints(); q++)
      {
         const IntegrationPoint &ip = IntRule->IntPoint(q);
         Tr.SetIntPoint(&ip);
         el.CalcShape(ip, shape);
         elvect.Add(ip.weight * Tr.Weight() * CallerLaw(shape * elfun, c), shape);
      }
   }
   void AssembleElementGrad(const FiniteElement &el, ElementTransformation &Tr,
                            const Vector &elfun, DenseMatrix &elmat) override
   {
      const int nd = el.GetDof();
      Vector shape(nd);
      elmat.SetSize(nd);
      elmat = 0.0;
      for (int q = 0; q < IntRule->GetNPoints(); q++)
      {
         const IntegrationPoint &ip = IntRule->IntPoint(q);
         Tr.SetIntPoint(&ip);
         el.CalcShape(ip, shape);
         AddMult_a_VVt(ip.weight * Tr.Weight() * CallerDLaw(shape * elfun, c),
                       shape, elmat);
      }
   }
};

/// What the caller-integrand gradient gave against the element loop.
struct CallerGradientOutcome
{
   bool took_route{};    ///< CanBatchNonlinearGradient() after the call
   real_t worst{}, scale{};
   int nonfinite{};
};

/** @brief The caller-integrand NPC gradient against the element loop's, on
    one discrete problem built twice, compared by applying both operators.

    A copy of test_darcy_batched_residual.cpp's fixture, here because that
    file is compiled into unit_tests alone and meq's report is that this
    route FAULTS under Device("debug") -- in NonlinearGradientFinish()'s
    `Df_data += Df_lin_data` -- which only a debug_device_tests case can see. */
inline CallerGradientOutcome CallerGradient(int order,
                                            bool batched_factor = false,
                                            int calls = 1)
{
   const real_t c = 0.35;
   struct Fx
   {
      std::unique_ptr<Mesh> mesh;
      std::unique_ptr<L2_FECollection> u_coll, p_coll;
      std::unique_ptr<DG_Interface_FECollection> t_coll;
      std::unique_ptr<FiniteElementSpace> Vh, Wh, Mh;
      std::unique_ptr<DarcyForm> darcy;
      ConstantCoefficient one{1.0};
      DarcyHybridization *dh{nullptr};
      std::unique_ptr<BlockVector> x;
      Vector x_tr;
   };
   auto build = [&](Fx &F)
   {
      const int dim = 2, n = 4;
      F.mesh.reset(new Mesh(Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL,
                                                  false, 0.8, 1.2)));
      F.u_coll.reset(new L2_FECollection(order, dim, BasisType::GaussLobatto));
      F.p_coll.reset(new L2_FECollection(order, dim));
      F.t_coll.reset(new DG_Interface_FECollection(order, dim));
      F.Vh.reset(new FiniteElementSpace(F.mesh.get(), F.u_coll.get(), dim));
      F.Wh.reset(new FiniteElementSpace(F.mesh.get(), F.p_coll.get()));
      F.Mh.reset(new FiniteElementSpace(F.mesh.get(), F.t_coll.get()));
      F.darcy.reset(new DarcyForm(F.Vh.get(), F.Wh.get()));
      F.darcy->GetFluxDivForm()->AddDomainIntegrator(
         new VectorDivergenceIntegrator());
      F.darcy->GetFluxDivForm()->AddBdrFaceIntegrator(
         new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));
      F.darcy->GetFluxMassForm()->AddDomainIntegrator(
         new VectorMassIntegrator(F.one));
      const IntegrationRule &ir = IntRules.Get(Geometry::SQUARE, 2 * order + 3);
      F.darcy->GetPotentialMassNonlinearForm()->AddDomainIntegrator(
         new CallerCubicNLF(c, &ir));
      BilinearForm *M_p = F.darcy->GetPotentialMassForm();
      M_p->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(F.one, 1.0));
      M_p->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(F.one, 1.0));
      Array<int> ess_flux;
      F.darcy->EnableHybridization(F.Mh.get(), new NormalTraceJumpIntegrator(),
                                   ess_flux);
      F.dh = F.darcy->GetHybridization();
      F.dh->EnableNPC();
      if (batched_factor)
      { F.dh->SetLocalFactorMode(DarcyHybridization::LocalFactorMode::Batched); }
      Array<int> all(F.mesh->bdr_attributes.Max());
      all = 1;
      F.dh->SetEssentialBC(all);
      F.darcy->Assemble();
      F.darcy->Finalize();
      F.x.reset(new BlockVector(F.darcy->GetOffsets()));
      for (int i = 0; i < F.x->Size(); i++)
      { (*F.x)(i) = std::sin(0.71 * i + 0.5) + 0.5 * std::cos(0.113 * i); }
      F.x_tr.SetSize(F.Mh->GetVSize());
      for (int i = 0; i < F.x_tr.Size(); i++)
      { F.x_tr(i) = std::sin(0.71 * i + 2.1) + 0.5 * std::cos(0.113 * i); }
   };
   Fx ref, bat;
   build(ref);
   build(bat);

   CallerGradientOutcome out;
   // @a calls > 1 is a Newton's shape: the gradient re-entered at a moved
   // state, with a residual between, on the same hybridization.
   Operator *pG_ref = nullptr, *pG_bat = nullptr;
   for (int k = 0; k < calls; k++)
   {
      if (k > 0)
      {
         *ref.x *= 1.1; *bat.x *= 1.1;
         BlockVector b(ref.darcy->GetOffsets()), r(ref.darcy->GetOffsets());
         b = 0.0;
         Vector r_tr;
         ref.dh->NPCResidual(b, *ref.x, ref.x_tr, r, r_tr);
         BlockVector rb(bat.darcy->GetOffsets());
         Vector rb_tr;
         bat.dh->NPCResidual(b, *bat.x, bat.x_tr, rb, rb_tr,
                             [c] MFEM_HOST_DEVICE (int, int, real_t, real_t,
                                                   real_t u)
         { return CallerLaw(u, c); });
      }
      pG_ref = &ref.dh->NPCGradient(*ref.x, ref.x_tr);
      pG_bat = &bat.dh->NPCGradient(*bat.x, bat.x_tr,
                                    [c] MFEM_HOST_DEVICE (int, int, real_t,
                                                          real_t, real_t u)
      {
         return CallerDLaw(u, c);
      });
   }
   Operator &G_ref = *pG_ref;
   Operator &G_bat = *pG_bat;
   out.took_route = bat.dh->CanBatchNonlinearGradient();
   for (int k = 0; k < 2; k++)
   {
      Vector v(G_ref.Width()), y_ref(G_ref.Height()), y_bat(G_ref.Height());
      for (int i = 0; i < v.Size(); i++) { v(i) = std::sin(0.3 * i + 1.7 * k); }
      G_ref.Mult(v, y_ref);
      G_bat.Mult(v, y_bat);
      y_ref.HostRead();
      y_bat.HostRead();
      out.nonfinite += y_bat.CheckFinite();
      for (int i = 0; i < y_ref.Size(); i++)
      {
         out.worst = std::max(out.worst, std::abs(y_ref(i) - y_bat(i)));
         out.scale = std::max(out.scale, std::abs(y_ref(i)));
      }
   }
   return out;
}


/// What meq's whole route gave against the element loop, step by step.
struct CallerRouteOutcome
{
   bool took_residual{}, took_gradient{}, took_traversal{};
   real_t worst_r{}, scale_r{}, worst_y{}, scale_y{};
   int nonfinite{};
};

/** @brief meq's fully batched NPC route -- the caller's integrand for the
    residual and the gradient, LocalFactorMode::Batched, the blocked
    DarcyNPCSolver::ArrayMult() -- against the element loop on the same
    discrete problem, over several Newton-shaped steps.

    CallerGradient() compares the GRADIENT; this compares what is solved with
    it, and does so at a moved state every step, on the same hybridization.
    That is the shape in which a factor or a block left stale-but-valid on one
    side shows -- the batched recover and reduce read the factors the
    gradient left, and a sync that stopped taking them over would hand the
    next step the previous one's. The route's outputs are made device-valid
    on entry, as meq has them, so the copies the solver does NOT make are
    the ones being trusted. */
inline CallerRouteOutcome CallerRoute(int order, int steps, int ncols)
{
   const real_t c = 0.35;
   struct Fx
   {
      std::unique_ptr<Mesh> mesh;
      std::unique_ptr<L2_FECollection> u_coll, p_coll;
      std::unique_ptr<DG_Interface_FECollection> t_coll;
      std::unique_ptr<FiniteElementSpace> Vh, Wh, Mh;
      std::unique_ptr<DarcyForm> darcy;
      ConstantCoefficient one{1.0};
      DarcyHybridization *dh{nullptr};
      std::unique_ptr<BlockVector> x;
      Vector x_tr;
   };
   auto build = [&](Fx &F, bool batched)
   {
      const int dim = 2, n = 4;
      F.mesh.reset(new Mesh(Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL,
                                                  false, 0.8, 1.2)));
      F.u_coll.reset(new L2_FECollection(order, dim, BasisType::GaussLobatto));
      F.p_coll.reset(new L2_FECollection(order, dim));
      F.t_coll.reset(new DG_Interface_FECollection(order, dim));
      F.Vh.reset(new FiniteElementSpace(F.mesh.get(), F.u_coll.get(), dim));
      F.Wh.reset(new FiniteElementSpace(F.mesh.get(), F.p_coll.get()));
      F.Mh.reset(new FiniteElementSpace(F.mesh.get(), F.t_coll.get()));
      F.darcy.reset(new DarcyForm(F.Vh.get(), F.Wh.get()));
      F.darcy->GetFluxDivForm()->AddDomainIntegrator(
         new VectorDivergenceIntegrator());
      F.darcy->GetFluxDivForm()->AddBdrFaceIntegrator(
         new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));
      F.darcy->GetFluxMassForm()->AddDomainIntegrator(
         new VectorMassIntegrator(F.one));
      const IntegrationRule &ir = IntRules.Get(Geometry::SQUARE, 2 * order + 3);
      F.darcy->GetPotentialMassNonlinearForm()->AddDomainIntegrator(
         new CallerCubicNLF(c, &ir));
      BilinearForm *M_p = F.darcy->GetPotentialMassForm();
      M_p->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(F.one, 1.0));
      M_p->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(F.one, 1.0));
      Array<int> ess_flux;
      F.darcy->EnableHybridization(F.Mh.get(), new NormalTraceJumpIntegrator(),
                                   ess_flux);
      F.dh = F.darcy->GetHybridization();
      F.dh->EnableNPC();
      if (batched)
      { F.dh->SetLocalFactorMode(DarcyHybridization::LocalFactorMode::Batched); }
      Array<int> all(F.mesh->bdr_attributes.Max());
      all = 1;
      F.dh->SetEssentialBC(all);
      F.darcy->Assemble();
      F.darcy->Finalize();
      F.x.reset(new BlockVector(F.darcy->GetOffsets()));
      for (int i = 0; i < F.x->Size(); i++)
      { (*F.x)(i) = std::sin(0.71 * i + 0.5) + 0.5 * std::cos(0.113 * i); }
      F.x_tr.SetSize(F.Mh->GetVSize());
      for (int i = 0; i < F.x_tr.Size(); i++)
      { F.x_tr(i) = std::sin(0.71 * i + 2.1) + 0.5 * std::cos(0.113 * i); }
   };
   Fx ref, bat;
   build(ref, false);
   build(bat, true);

   const Array<int> &doffs = ref.darcy->GetOffsets();
   const int nu = doffs[1], np = doffs[2] - doffs[1], nt = ref.x_tr.Size();
   Array<int> offs({0, nu, nu + np, nu + np + nt}), loc_offs({0, nu, nu + np});
   const int N = offs.Last();

   std::vector<Vector> X(ncols), Yr(ncols), Yb(ncols);
   Array<const Vector *> XP(ncols);
   Array<Vector *> YRP(ncols), YBP(ncols);
   for (int j = 0; j < ncols; j++)
   {
      X[j].SetSize(N);
      for (int i = 0; i < N; i++)
      { X[j](i) = std::sin(0.37 * i + 1.7 * j) + 0.25 * (j + 1) * std::cos(0.11 * i); }
      Yr[j].SetSize(N);
      Yb[j].SetSize(N);
      Yb[j].UseDevice(true);
      XP[j] = &X[j];
      YRP[j] = &Yr[j];
      YBP[j] = &Yb[j];
   }

   UMFPackSolver trace_ref, trace_bat;
   DarcyNPCSolver lin_ref(trace_ref), lin_bat(trace_bat);
   BlockVector b(doffs), r_ref(doffs), r_bat(doffs);
   b = 0.0;
   Vector rt_ref, rt_bat;

   CallerRouteOutcome out;
   auto worst = [](const Vector &got, const Vector &want, real_t &w, real_t &s)
   {
      for (int i = 0; i < want.Size(); i++)
      {
         w = std::max(w, std::abs(got(i) - want(i)));
         s = std::max(s, std::abs(want(i)));
      }
   };
   for (int k = 0; k < steps; k++)
   {
      ref.dh->NPCResidual(b, *ref.x, ref.x_tr, r_ref, rt_ref);
      bat.dh->NPCResidual(b, *bat.x, bat.x_tr, r_bat, rt_bat,
                          [c] MFEM_HOST_DEVICE (int, int, real_t, real_t,
                                                real_t u)
      { return CallerLaw(u, c); });
      out.took_residual = bat.dh->CanBatchNonlinearResidual();

      Operator &S_ref = ref.dh->NPCGradient(*ref.x, ref.x_tr);
      Operator &S_bat = bat.dh->NPCGradient(*bat.x, bat.x_tr,
                                            [c] MFEM_HOST_DEVICE (int, int,
                                                                  real_t, real_t,
                                                                  real_t u)
      { return CallerDLaw(u, c); });
      out.took_gradient = bat.dh->CanBatchNonlinearGradient();

      DarcyNPCOperator::Jacobian J_ref(*ref.dh, S_ref, offs, loc_offs);
      DarcyNPCOperator::Jacobian J_bat(*bat.dh, S_bat, offs, loc_offs);
      lin_ref.SetOperator(J_ref);
      lin_bat.SetOperator(J_bat);
      for (int j = 0; j < ncols; j++) { Yb[j].Write(); }
      lin_ref.ArrayMult(XP, YRP);
      lin_bat.ArrayMult(XP, YBP);
      out.took_traversal = bat.dh->CanBatchNPCTraversal();

      r_ref.HostRead(); r_bat.HostRead(); rt_ref.HostRead(); rt_bat.HostRead();
      out.nonfinite += r_bat.CheckFinite() + rt_bat.CheckFinite();
      worst(r_bat, r_ref, out.worst_r, out.scale_r);
      worst(rt_bat, rt_ref, out.worst_r, out.scale_r);
      for (int j = 0; j < ncols; j++)
      {
         Yr[j].HostRead();
         Yb[j].HostRead();
         out.nonfinite += Yb[j].CheckFinite();
         worst(Yb[j], Yr[j], out.worst_y, out.scale_y);
      }

      // A moved state for the next step, on the same hybridizations.
      *ref.x *= 1.1;
      *bat.x *= 1.1;
   }
   return out;
}

/** @brief CallerCubicNLF with a block added to ONE element's gradient, the
    way meq added its border fold before AddNextGradientCorrection() existed:
    inside AssembleElementGrad(), keyed on ElementTransformation::ElementNo.
    Both pointers are BORROWED, so the case can move the fold between steps;
    an element of -1 is no fold. */
class CallerFoldedNLF : public CallerCubicNLF
{
   const int *fold_el;
   const DenseMatrix *fold;
public:
   CallerFoldedNLF(real_t c_, const IntegrationRule *ir, const int *el,
                   const DenseMatrix *dD)
      : CallerCubicNLF(c_, ir), fold_el(el), fold(dD) { }
   void AssembleElementGrad(const FiniteElement &el, ElementTransformation &Tr,
                            const Vector &elfun, DenseMatrix &elmat) override
   {
      CallerCubicNLF::AssembleElementGrad(el, Tr, elfun, elmat);
      if (Tr.ElementNo == *fold_el) { elmat += *fold; }
   }
};

/// What a queued gradient correction gave against the integrator's own fold.
struct CorrectedGradientOutcome
{
   bool took_gradient{}, took_traversal{};
   /// Corrections still queued after an NPCGradient(), summed over calls.
   int left_queued{};
   int nonfinite{};
   /// The corrected steps: each queued arm against the folded integrator.
   real_t worst_loop{}, worst_bat{}, scale{};
   /// The corrected steps: the uncorrected operator against the folded one.
   real_t moved{};
   /// The final step, nothing queued: each queued arm against an arm that
   /// never queued anything.
   real_t last_loop{}, last_bat{}, last_scale{};
};

/** @brief AddNextGradientCorrection() against the fold meq used to put in
    its integrator, compared through what is SOLVED with the gradient:
    DarcyNPCSolver::ArrayMult(), so NPCReduce(), the trace solve and
    NPCRecover() all read the corrected factors.

    Four hybridizations of one problem. `ref` carries the fold in its
    integrator, on the element loop. `loop` and `bat` queue it: `loop` on the
    element loop, `bat` on the caller-integrand batched gradient with
    @a batched_factor choosing the local factorisation. `plain` never
    corrects. Over @a steps steps the fold moves to another element, row and
    weight each step, as meq's does. Then one more step queues nothing. That
    step catches a correction that outlived its call. */
inline CorrectedGradientOutcome CorrectedGradient(int order,
                                                  bool batched_factor,
                                                  int steps, int ncols)
{
   const real_t c = 0.35;
   int fold_el = -1;
   DenseMatrix fold;
   struct Fx
   {
      std::unique_ptr<Mesh> mesh;
      std::unique_ptr<L2_FECollection> u_coll, p_coll;
      std::unique_ptr<DG_Interface_FECollection> t_coll;
      std::unique_ptr<FiniteElementSpace> Vh, Wh, Mh;
      std::unique_ptr<DarcyForm> darcy;
      ConstantCoefficient one{1.0};
      DarcyHybridization *dh{nullptr};
      std::unique_ptr<BlockVector> x;
      Vector x_tr;
   };
   auto build = [&](Fx &F, bool folded, bool batched)
   {
      const int dim = 2, n = 4;
      F.mesh.reset(new Mesh(Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL,
                                                  false, 0.8, 1.2)));
      F.u_coll.reset(new L2_FECollection(order, dim, BasisType::GaussLobatto));
      F.p_coll.reset(new L2_FECollection(order, dim));
      F.t_coll.reset(new DG_Interface_FECollection(order, dim));
      F.Vh.reset(new FiniteElementSpace(F.mesh.get(), F.u_coll.get(), dim));
      F.Wh.reset(new FiniteElementSpace(F.mesh.get(), F.p_coll.get()));
      F.Mh.reset(new FiniteElementSpace(F.mesh.get(), F.t_coll.get()));
      F.darcy.reset(new DarcyForm(F.Vh.get(), F.Wh.get()));
      F.darcy->GetFluxDivForm()->AddDomainIntegrator(
         new VectorDivergenceIntegrator());
      F.darcy->GetFluxDivForm()->AddBdrFaceIntegrator(
         new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));
      F.darcy->GetFluxMassForm()->AddDomainIntegrator(
         new VectorMassIntegrator(F.one));
      const IntegrationRule &ir = IntRules.Get(Geometry::SQUARE, 2 * order + 3);
      F.darcy->GetPotentialMassNonlinearForm()->AddDomainIntegrator(
         folded ? new CallerFoldedNLF(c, &ir, &fold_el, &fold)
         : new CallerCubicNLF(c, &ir));
      BilinearForm *M_p = F.darcy->GetPotentialMassForm();
      M_p->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(F.one, 1.0));
      M_p->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(F.one, 1.0));
      Array<int> ess_flux;
      F.darcy->EnableHybridization(F.Mh.get(), new NormalTraceJumpIntegrator(),
                                   ess_flux);
      F.dh = F.darcy->GetHybridization();
      F.dh->EnableNPC();
      if (batched)
      { F.dh->SetLocalFactorMode(DarcyHybridization::LocalFactorMode::Batched); }
      Array<int> all(F.mesh->bdr_attributes.Max());
      all = 1;
      F.dh->SetEssentialBC(all);
      F.darcy->Assemble();
      F.darcy->Finalize();
      F.x.reset(new BlockVector(F.darcy->GetOffsets()));
      for (int i = 0; i < F.x->Size(); i++)
      { (*F.x)(i) = std::sin(0.71 * i + 0.5) + 0.5 * std::cos(0.113 * i); }
      F.x_tr.SetSize(F.Mh->GetVSize());
      for (int i = 0; i < F.x_tr.Size(); i++)
      { F.x_tr(i) = std::sin(0.71 * i + 2.1) + 0.5 * std::cos(0.113 * i); }
   };
   Fx ref, loop, bat, plain;
   build(ref, true, false);
   build(loop, false, false);
   build(bat, false, batched_factor);
   build(plain, false, false);

   const Array<int> &doffs = ref.darcy->GetOffsets();
   const int nu = doffs[1], np = doffs[2] - doffs[1], nt = ref.x_tr.Size();
   Array<int> offs({0, nu, nu + np, nu + np + nt}), loc_offs({0, nu, nu + np});
   const int N = offs.Last();
   const int NE = ref.mesh->GetNE();
   const int nd = ref.Wh->GetFE(0)->GetDof();

   std::vector<Vector> X(ncols), Y[4];
   Array<const Vector *> XP(ncols);
   Array<Vector *> YP[4];
   for (int a = 0; a < 4; a++) { Y[a].resize(ncols); YP[a].SetSize(ncols); }
   for (int j = 0; j < ncols; j++)
   {
      X[j].SetSize(N);
      for (int i = 0; i < N; i++)
      { X[j](i) = std::sin(0.37 * i + 1.7 * j) + 0.25 * (j + 1) * std::cos(0.11 * i); }
      XP[j] = &X[j];
      for (int a = 0; a < 4; a++)
      {
         Y[a][j].SetSize(N);
         Y[a][j].UseDevice(a == 2);
         YP[a][j] = &Y[a][j];
      }
   }

   Fx *arms[4] = {&ref, &loop, &bat, &plain};
   UMFPackSolver trace[4];
   std::unique_ptr<DarcyNPCSolver> lin[4];
   for (int a = 0; a < 4; a++) { lin[a].reset(new DarcyNPCSolver(trace[a])); }
   BlockVector b(doffs), r(doffs);
   b = 0.0;
   Vector r_tr;

   CorrectedGradientOutcome out;
   auto worst = [](const Vector &got, const Vector &want, real_t &w, real_t *s)
   {
      for (int i = 0; i < want.Size(); i++)
      {
         w = std::max(w, std::abs(got(i) - want(i)));
         if (s) { *s = std::max(*s, std::abs(want(i))); }
      }
   };
   for (int k = 0; k <= steps; k++)
   {
      const bool corrected = (k < steps);
      // meq's shape, kappa e_j b^T: one row of one element, re-decided every
      // step. A transposed or misplaced block lands in another row.
      fold_el = corrected ? (5 + 7 * k) % NE : -1;
      fold.SetSize(nd);
      fold = 0.0;
      if (corrected)
      {
         const int row = (1 + 2 * k) % nd;
         for (int i = 0; i < nd; i++)
         { fold(row, i) = 0.02 * (1 + k) * std::sin(0.9 * i + 0.4 * k + 0.3); }
      }

      std::unique_ptr<DarcyNPCOperator::Jacobian> J[4];
      for (int a = 0; a < 4; a++)
      {
         Fx &F = *arms[a];
         if (a == 2)
         {
            F.dh->NPCResidual(b, *F.x, F.x_tr, r, r_tr,
                              [c] MFEM_HOST_DEVICE (int, int, real_t, real_t,
                                                    real_t u)
            { return CallerLaw(u, c); });
         }
         else { F.dh->NPCResidual(b, *F.x, F.x_tr, r, r_tr); }

         if (corrected && (a == 1 || a == 2))
         { F.dh->AddNextGradientCorrection(fold_el, fold); }
         Operator *S = nullptr;
         if (a == 2)
         {
            S = &F.dh->NPCGradient(*F.x, F.x_tr,
                                   [c] MFEM_HOST_DEVICE (int, int, real_t, real_t,
                                                         real_t u)
            { return CallerDLaw(u, c); });
            out.took_gradient = F.dh->CanBatchNonlinearGradient();
         }
         else { S = &F.dh->NPCGradient(*F.x, F.x_tr); }
         out.left_queued += F.dh->NumNextGradientCorrections();

         J[a].reset(new DarcyNPCOperator::Jacobian(*F.dh, *S, offs, loc_offs));
         lin[a]->SetOperator(*J[a]);
         for (int j = 0; j < ncols; j++) { Y[a][j].Write(); }
         lin[a]->ArrayMult(XP, YP[a]);
         if (a == 2) { out.took_traversal = F.dh->CanBatchNPCTraversal(); }
         for (int j = 0; j < ncols; j++)
         {
            Y[a][j].HostRead();
            out.nonfinite += Y[a][j].CheckFinite();
         }
      }

      for (int j = 0; j < ncols; j++)
      {
         if (corrected)
         {
            worst(Y[1][j], Y[0][j], out.worst_loop, &out.scale);
            worst(Y[2][j], Y[0][j], out.worst_bat, nullptr);
            worst(Y[3][j], Y[0][j], out.moved, nullptr);
         }
         else
         {
            worst(Y[1][j], Y[3][j], out.last_loop, &out.last_scale);
            worst(Y[2][j], Y[3][j], out.last_bat, nullptr);
         }
      }

      for (int a = 0; a < 4; a++) { *arms[a]->x *= 1.1; }
   }
   return out;
}


} // namespace darcy_plain_npc

#endif // MFEM_UNIT_FEM_DARCY_PLAIN_NPC_HPP
