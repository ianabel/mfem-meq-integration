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

#ifndef MFEM_NLRESIDUAL_HDG
#define MFEM_NLRESIDUAL_HDG

#include "../../config/config.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/vector.hpp"

namespace mfem
{

/** @brief The quadrature data a caller's element-local integrand is evaluated
    against, and the ONLY thing DarcyHybridization puts in a header for it.

    **This struct exists so that the template does not.** A caller supplying a
    device integrand has to instantiate the loop that calls it, and that loop
    therefore has to be visible -- but nothing else does. Everything that knows
    about `Af_f_offsets`, the element dof maps, the face gathers, the NPC
    ordering and the block layout stays in the source file, on either side of
    this: DarcyHybridization::NonlinearResidualPrepare() fills this in and
    DarcyHybridization::NonlinearResidualFinish() scatters what the kernel
    leaves behind. What is exposed is eight pointers and four sizes, so the
    internals stay free to move.

    Every pointer is DEVICE-side and read (or, for #rp, read-written) by the
    kernel below. They are valid between a Prepare() and the Finish() that
    follows it and not after, and nothing here owns any of them.

    The layouts are the element-blocked ones the batched routes use
    throughout: element e's potential dofs are `p + nd*e`, and the shape table
    is one table for the mesh, which is what makes it (nq, nd) rather than
    (nq, nd, NE). That single table is the same condition
    HDGElementMassCanBatch() asks -- one geometry and one dof count. */
struct HDGNLResidualWork
{
   int NE {0};          ///< elements
   int nd {0};          ///< potential dofs per element
   int nq {0};          ///< quadrature points per element
   int dim {0};         ///< space dimension, so a 2-D caller reads y and a 1-D one does not

   const real_t *shape {nullptr};   ///< (nq, nd), ONE table for the mesh
   const real_t *coord {nullptr};   ///< (dim, nq, NE), the physical point
   const real_t *weight {nullptr};  ///< (nq, NE), the quadrature weight times |J|
   const real_t *p {nullptr};       ///< (nd, NE), the element potential dofs
   real_t *rp {nullptr};            ///< (nd, NE), ACCUMULATED into -- Residual
   /** @brief (nd, nd, NE), ASSIGNED -- Gradient only.

       Column-major per element, so element e's block is `gD + nd*nd*e` and
       its (i, j) entry is `[i + nd*j]` -- a DenseMatrix built on that
       pointer, which is what ConstructGrad() builds over the same bytes of
       Df_data. ASSIGNED rather than accumulated because that is what the
       per-element route does, zeroing D before adding the element mass
       Jacobian, and because the previous linearisation is still in there. */
   real_t *gD {nullptr};
};


/// @brief Which contraction HDGNLElementQuadrature() applies.
enum class HDGNLAccum
{
   Residual,   ///< @f$(f, \varphi_i)_K@f$ into HDGNLResidualWork::rp
   Gradient    ///< @f$(f \varphi_j, \varphi_i)_K@f$ into HDGNLResidualWork::gD
};

/** @brief Add @f$(f(x, u_h), \varphi_i)_K@f$ to every element's potential row
    at once, with @a f supplied by the caller and called on the device.

    @a f is invoked once per element per quadrature point as

        real_t f( int element, int point, real_t x, real_t y, real_t u )

    and returns the integrand at that point -- NOT multiplied by the
    quadrature weight, which this applies, and NOT contracted with the basis,
    which this also applies. So a caller writing @f$-F(R, z, \psi)/R@f$ returns
    exactly that and nothing else.

    **`element` and `point` are passed because a caller's law has state this
    cannot see.** meq's source reads a per-element plasma-connectivity flag and
    a per-point cached field, neither of which is a function of the coordinate,
    so an interface offering only `(x, y, u)` would force them to recover the
    index from the position. They are the two loop counters and they cost
    nothing.

    **`f` is taken BY VALUE and captured by value.** It has to be trivially
    copyable and callable from device code; a lambda capturing a POD by value
    is the intended shape and is what meq's `SourceView` was built to be. A
    capture by reference would be a host pointer inside a device kernel, which
    is the fault class both projects have paid for more than once.

    @note **The rule is the CALLER's and this does not choose one.**
    NonlinearResidualPrepare() takes it from the integrator's own
    Integrator::GetIntRule() and refuses when there is none, because a kernel
    that quadratured differently from the element loop would disagree with it
    by an amount no test states. */
template <HDGNLAccum Acc, typename Integrand>
void HDGNLElementQuadrature(const HDGNLResidualWork &w, Integrand f)
{
   const int nd = w.nd, nq = w.nq, dim = w.dim;
   const real_t *d_sh = w.shape;
   const real_t *d_x = w.coord;
   const real_t *d_w = w.weight;
   const real_t *d_p = w.p;
   real_t *d_rp = w.rp;
   real_t *d_gD = w.gD;

   mfem::forall(w.NE, [=] MFEM_HOST_DEVICE (int e)
   {
      const real_t *p_e = d_p + nd * e;

      /* **BOTH DESTINATIONS ARE NAMED HERE, OUTSIDE THE `if constexpr`, AND
         THAT IS REQUIRED RATHER THAN TIDY.** nvcc refuses to let an extended
         __host__ __device__ lambda FIRST-capture a variable inside a
         constexpr-if context -- "cannot first-capture variable in constexpr-if
         context" -- so a tag whose branches are each the only mention of their
         own pointer compiles under g++ and fails under nvcc, which is the one
         compiler this seam exists for. Naming both here first-captures them;
         the unused one costs a register the dead branch drops. */
      real_t *const rp_all = d_rp;
      real_t *const gD_all = d_gD;

      // Each element owns its destination -- an L2 element's dofs are its
      // own and the gradient block is this element's alone -- so the zero
      // below races with nothing and wants no colouring.
      if constexpr (Acc == HDGNLAccum::Gradient)
      {
         real_t *gD_e = gD_all + nd * nd * e;
         for (int k = 0; k < nd * nd; k++) { gD_e[k] = 0.; }
      }

      for (int q = 0; q < nq; q++)
      {
         // u_h at the point, from this element's own dofs.
         real_t u = 0.;
         for (int j = 0; j < nd; j++) { u += d_sh[q + nq * j] * p_e[j]; }

         const real_t *X = d_x + dim * (q + nq * e);
         const real_t x0 = X[0];
         const real_t x1 = (dim > 1) ? X[1] : real_t(0.);

         // The weight is applied HERE and the basis below, so that what the
         // caller returns is the integrand and nothing else. THE TWO TAGS
         // DIFFER IN NOTHING ABOVE THIS LINE, which is why they are one
         // template: the gather, the geometry and the call into the caller's
         // law are the shared part and the contraction is the difference.
         const real_t v = d_w[q + nq * e] * f(e, q, x0, x1, u);

         if constexpr (Acc == HDGNLAccum::Residual)
         {
            real_t *rp_e = rp_all + nd * e;
            for (int i = 0; i < nd; i++) { rp_e[i] += d_sh[q + nq * i] * v; }
         }
         else
         {
            real_t *gD_e = gD_all + nd * nd * e;
            for (int j = 0; j < nd; j++)
            {
               const real_t sj = d_sh[q + nq * j] * v;
               for (int i = 0; i < nd; i++)
               { gD_e[i + nd * j] += d_sh[q + nq * i] * sj; }
            }
         }
      }
   });
}


/// @brief HDGNLElementQuadrature() with the residual contraction; see there.
template <typename Integrand>
void HDGNLResidualQuadrature(const HDGNLResidualWork &w, Integrand f)
{
   HDGNLElementQuadrature<HDGNLAccum::Residual>(w, f);
}


/** @brief Write every element's potential-mass JACOBIAN block from the
    caller's derivative -- the counterpart of HDGNLResidualQuadrature().

    @a g is invoked exactly as @a f is there, `g(element, point, x, y, u)`,
    and returns @f$\partial f/\partial u@f$ at that point, again with neither
    the quadrature weight nor the basis applied. So a caller writing
    @f$-F(R,z,\psi)/R@f$ for the residual writes
    @f$-(\partial F/\partial\psi)/R@f$ here and the two differ in the
    integrand alone -- which is what lets one callable serve both when the
    caller's law can report its own derivative.

    **The block is ASSIGNED and is the element mass Jacobian ONLY.** The
    linear part of D -- a linear face constraint, a linear potential domain
    mass, or both -- is added afterwards by
    DarcyHybridization::NonlinearGradientFinish(), exactly as ConstructGrad()
    adds Df_lin_data after the integrator's own block. A caller reaching this
    template directly therefore gets the mass Jacobian and nothing else. */
template <typename Integrand>
void HDGNLGradientQuadrature(const HDGNLResidualWork &w, Integrand g)
{
   HDGNLElementQuadrature<HDGNLAccum::Gradient>(w, g);
}

} // namespace mfem

#endif
