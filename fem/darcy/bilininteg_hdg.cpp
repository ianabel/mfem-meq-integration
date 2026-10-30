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

#include "bilininteg_hdg.hpp"
#include "../linearform.hpp"
#include "../hyperbolic.hpp"
#include "../nonlininteg_mixed.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/dtensor.hpp"
#ifdef MFEM_USE_MPI
#include "../../mesh/pmesh.hpp"
#endif

#include <map>
#include <typeinfo>

using std::min;
using std::max;

/** @file
    Quadrature for the HDG face terms, and why the trace element's order is in
    every rule below.

    Each routine here that takes a @a trace_el assembles three kinds of block on
    one face: element-element, element-trace, and trace-trace. The last is a
    mass-like matrix in the trace basis alone -- it is where the stabilization
    `<tau*uhat, mu>` lands -- so it is of degree `2*p_trace`, and a rule chosen
    from the element orders alone under-integrates it whenever the trace outruns
    its elements.

    That is not a lost digit, it is a rank failure. A Gauss rule of degree
    `2*p_el` on a face carries `p_el + 1` points, and a Gram matrix built from
    `n` points has rank at most `n`; the trace-trace block is
    `(p_trace + 1) x (p_trace + 1)`. At `p_trace = p_el + 1` it is therefore
    rank-deficient by exactly one per face and the reduced trace system is
    singular. Measured before the fix: `convdiff -p 1 -dg -hb -o 2` with an
    order-3 trace diverges (GMRES residual 1e21, relative errors of 1e4);
    raising the rule to degree `2*p_trace` -- and nothing else -- converges it.
    The threshold is exactly `2*p_trace`, degree `2*p_el` still fails.

    So the max below spans the trace element too, which is what
    `fem/hyperbolic.cpp`'s HDG routines and `NormalTraceJumpIntegrator` in
    `fem/bilininteg.cpp` already do. Where `p_trace <= p_el` the expression is
    arithmetically the old one, so no existing configuration moves; the one
    committed case that does is `--order 0 --trace-H1`, whose trace collection is
    `max(order, 1)` and so outruns its elements (errors move by 4-5% and no
    regression reference covers it).

    This matters beyond tidiness: a trace order that may exceed an element order
    is exactly what `p`-adaptivity's max rule needs on a face between elements of
    different degree, and this was what blocked it.

    `AssembleFaceMatrix` below takes no trace element and is left alone. */

namespace mfem
{


const IntegrationRule &HDGConvectionCenteredIntegrator::GetHDGFaceIntRule(
   const FiniteElement &trace_el, const FiniteElement &el1,
   const FiniteElement &el2, FaceElementTransformations &Trans) const
{
   int order;
   // Assuming order(u)==order(mesh)
   if (Trans.Elem2No >= 0)
      order = (min(Trans.Elem1->OrderW(), Trans.Elem2->OrderW()) +
               2*max(max(el1.GetOrder(), el2.GetOrder()), trace_el.GetOrder()));
   else
   {
      order = Trans.Elem1->OrderW() + 2*max(el1.GetOrder(), trace_el.GetOrder());
   }
   if (el1.Space() == FunctionSpace::Pk)
   {
      order++;
   }
   return IntRules.Get(Trans.GetGeometryType(), order);
}

const IntegrationRule &HDGConvectionCenteredIntegrator::GetHDGFaceIntRule(
   int side, const FiniteElement &trace_el, const FiniteElement &el,
   FaceElementTransformations &Trans) const
{
   // The one-sided rule, and it is NOT the two-sided one with el2 == el1:
   // OrderW() is taken from the side being integrated rather than min'd over
   // the two, and the max runs over one element. On a shared face -- the case
   // the batched pass exists for -- Trans.Elem2No is >= 0 and yet only side 0
   // is this rank's, so asking the two-sided overload would sample the wrong
   // rule the moment the neighbour's order differed.
   // Assuming order(u)==order(mesh)
   int order = (side != 0) ? Trans.Elem2->OrderW() : Trans.Elem1->OrderW();
   order += 2*max(el.GetOrder(), trace_el.GetOrder());
   if (el.Space() == FunctionSpace::Pk)
   {
      order++;
   }
   return IntRules.Get(Trans.GetGeometryType(), order);
}

const IntegrationRule &HDGConvectionUpwindedIntegrator::GetHDGFaceIntRule(
   const FiniteElement &trace_el, const FiniteElement &el1,
   const FiniteElement &el2, FaceElementTransformations &Trans) const
{
   // The same rule as the centred form; they differ in the weights, not in
   // where they are sampled.
   int order;
   if (Trans.Elem2No >= 0)
      order = (min(Trans.Elem1->OrderW(), Trans.Elem2->OrderW()) +
               2*max(max(el1.GetOrder(), el2.GetOrder()), trace_el.GetOrder()));
   else
   {
      order = Trans.Elem1->OrderW() + 2*max(el1.GetOrder(), trace_el.GetOrder());
   }
   if (el1.Space() == FunctionSpace::Pk)
   {
      order++;
   }
   return IntRules.Get(Trans.GetGeometryType(), order);
}

const IntegrationRule &HDGConvectionUpwindedIntegrator::GetHDGFaceIntRule(
   int side, const FiniteElement &trace_el, const FiniteElement &el,
   FaceElementTransformations &Trans) const
{
   // The one-sided rule, and it is NOT the two-sided one with el2 == el1:
   // OrderW() is taken from the side being integrated rather than min'd over
   // the two, and the max runs over one element. On a shared face -- the case
   // the batched pass exists for -- Trans.Elem2No is >= 0 and yet only side 0
   // is this rank's, so asking the two-sided overload would sample the wrong
   // rule the moment the neighbour's order differed.
   // Assuming order(u)==order(mesh)
   int order = (side != 0) ? Trans.Elem2->OrderW() : Trans.Elem1->OrderW();
   order += 2*max(el.GetOrder(), trace_el.GetOrder());
   if (el.Space() == FunctionSpace::Pk)
   {
      order++;
   }
   return IntRules.Get(Trans.GetGeometryType(), order);
}

const IntegrationRule &HDGDiffusionIntegrator::GetHDGFaceIntRule(
   const FiniteElement &trace_el, const FiniteElement &el1,
   const FiniteElement &el2, FaceElementTransformations &Trans) const
{
   // Degree 2*max(element, trace): see the note at the top of this file for
   // why the trace element has to be in the max.
   const int order = (Trans.Elem2No >= 0)
                     ? 2*max(max(el1.GetOrder(), el2.GetOrder()),
                             trace_el.GetOrder())
                     : 2*max(el1.GetOrder(), trace_el.GetOrder());
   return IntRules.Get(Trans.GetGeometryType(), order);
}

const IntegrationRule &HDGDiffusionIntegrator::GetHDGFaceIntRule(
   const FiniteElement &trace_el, const FiniteElement &el,
   FaceElementTransformations &Trans) const
{
   // Degree 2*max(element, trace): see the note at the top of this file for
   // why the trace element has to be in the max.
   const int order = 2*max(el.GetOrder(), trace_el.GetOrder());
   return IntRules.Get(Trans.GetGeometryType(), order);
}

void HDGConvectionCenteredIntegrator::AssembleHDGFaceMatrix(
   const FiniteElement &trace_el, const FiniteElement &el1,
   const FiniteElement &el2, FaceElementTransformations &Trans,
   DenseMatrix &elmat)
{
#ifdef MFEM_THREAD_SAFE
   Vector tr_shape, shape1, shape2;
#endif
   MFEM_VERIFY(trace_el.GetMapType() == FiniteElement::VALUE, "");

   const int el_dim = el1.GetDim();
   const int tr_ndof = trace_el.GetDof();
   const int ndof1 = el1.GetDof();
   const int ndof2 = (Trans.Elem2No >= 0)?(el2.GetDof()):(0);
   const int el_ndof = ndof1 + ndof2;

   Vector vu(el_dim), nor(el_dim);

   tr_shape.SetSize(tr_ndof);
   shape1.SetSize(ndof1);
   shape2.SetSize(ndof2);

   elmat.SetSize(el_ndof + tr_ndof);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      ir = &GetHDGFaceIntRule(trace_el, el1, el2, Trans);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration point
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();

      trace_el.CalcShape(ip, tr_shape);
      el1.CalcPhysShape(*Trans.Elem1, shape1);
      if (ndof2)
      {
         el2.CalcPhysShape(*Trans.Elem2, shape2);
      }

      u->Eval(vu, *Trans.Elem1, eip1);

      if (el_dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      const real_t un = vu * nor;
      const real_t a = alpha * un;
      const real_t b = fabs(alpha * un);
      // note: if |alpha/2|==|beta| then |a|==|b|, i.e. (a==b) or (a==-b)
      //       and therefore two blocks in the element matrix contribution
      //       (from the current quadrature point) are 0

      real_t w = ip.weight * b;
      if (w != 0.0)
      {
         // assemble the element matrix
         for (int i = 0; i < ndof1; i++)
            for (int j = 0; j < ndof1; j++)
            {
               elmat(i, j) += w * shape1(i) * shape1(j);
            }
         for (int i = 0; i < ndof2; i++)
            for (int j = 0; j < ndof2; j++)
            {
               elmat(ndof1+i, ndof1+j) += w * shape2(i) * shape2(j);
            }

         // assemble the constraint matrix
         for (int i = 0; i < tr_ndof; i++)
            for (int j = 0; j < ndof1; j++)
            {
               elmat(el_ndof+i, j) += w * tr_shape(i) * shape1(j);
            }
         for (int i = 0; i < tr_ndof; i++)
            for (int j = 0; j < ndof2; j++)
            {
               elmat(el_ndof+i, ndof1+j) += w * tr_shape(i) * shape2(j);
            }

      }

      // assemble the face matrix
      // note: that this term must be non-zero at the boundary for stability
      //       reasons, so the advective part is intentionally dropped here
      //       and must be compensated elsewhere
      w = ip.weight * ((ndof2)?(2.*b):(b));//<-- single face integration
      if (w != 0.0)
      {
         for (int i = 0; i < tr_ndof; i++)
            for (int j = 0; j < tr_ndof; j++)
            {
               elmat(el_ndof+i, el_ndof+j) -= w * tr_shape(i) * tr_shape(j);
            }
      }

      w = ip.weight * (b-a);
      if (w != 0.0)
      {
         // assemble the trace matrix (elem1)
         for (int i = 0; i < ndof1; i++)
            for (int j = 0; j < tr_ndof; j++)
            {
               elmat(i, el_ndof+j) -= w * shape1(i) * tr_shape(j);
            }
      }

      w = ip.weight * (b+a);
      if (w != 0.0)
      {
         // assemble the trace matrix (elem2)
         for (int i = 0; i < ndof2; i++)
            for (int j = 0; j < tr_ndof; j++)
            {
               elmat(ndof1+i, el_ndof+j) -= w * shape2(i) * tr_shape(j);
            }
      }
   }
}

void HDGConvectionCenteredIntegrator::AssembleHDGFaceMatrix(
   int side, const FiniteElement &trace_el, const FiniteElement &el,
   FaceElementTransformations &Trans, DenseMatrix &elmat)
{
#ifdef MFEM_THREAD_SAFE
   Vector tr_shape, shape1;
#endif
   MFEM_VERIFY(trace_el.GetMapType() == FiniteElement::VALUE, "");

   const int el_dim = el.GetDim();
   const int tr_ndof = trace_el.GetDof();
   const int el_ndof = el.GetDof();

   Vector vu(el_dim), nor(el_dim);
   Vector &el_shape = shape1;

   tr_shape.SetSize(tr_ndof);
   el_shape.SetSize(el_ndof);

   elmat.SetSize(el_ndof + tr_ndof);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      ir = &GetHDGFaceIntRule(side, trace_el, el, Trans);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration point
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();

      trace_el.CalcShape(ip, tr_shape);

      if (side != 0)
      {
         el.CalcPhysShape(*Trans.Elem2, el_shape);
      }
      else
      {
         el.CalcPhysShape(*Trans.Elem1, el_shape);
      }

      u->Eval(vu, *Trans.Elem1, eip1);

      if (el_dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      if (side != 0) { nor.Neg(); }

      const real_t un = vu * nor;
      const real_t a = alpha * un;
      const real_t b = fabs(alpha * un);
      // note: if |alpha/2|==|beta| then |a|==|b|, i.e. (a==b) or (a==-b)
      //       and therefore two blocks in the element matrix contribution
      //       (from the current quadrature point) are 0

      real_t w = ip.weight * b;
      if (w != 0.0)
      {
         // assemble the element matrix
         for (int i = 0; i < el_ndof; i++)
            for (int j = 0; j < el_ndof; j++)
            {
               elmat(i, j) += w * el_shape(i) * el_shape(j);
            }

         // assemble the constraint matrix
         for (int i = 0; i < tr_ndof; i++)
            for (int j = 0; j < el_ndof; j++)
            {
               elmat(el_ndof+i, j) += w * tr_shape(i) * el_shape(j);
            }
      }

      // assemble the face matrix
      // note: that this term must be non-zero at the boundary for stability
      //       reasons, so the advective part is intentionally dropped here
      //       and must be compensated elsewhere
      w = ip.weight * ((Trans.Elem2No >= 0)?(b-a):(b));
      if (w != 0.0)
      {
         for (int i = 0; i < tr_ndof; i++)
            for (int j = 0; j < tr_ndof; j++)
            {
               elmat(el_ndof+i, el_ndof+j) -= w * tr_shape(i) * tr_shape(j);
            }
      }

      w = ip.weight * (b-a);
      if (w != 0.0)
      {
         // assemble the trace matrix
         for (int i = 0; i < el_ndof; i++)
            for (int j = 0; j < tr_ndof; j++)
            {
               elmat(i, el_ndof+j) -= w * el_shape(i) * tr_shape(j);
            }
      }
   }
}

void HDGConvectionCenteredIntegrator::AssembleHDGFaceVector(
   int type, const FiniteElement &trace_el, const FiniteElement &el,
   FaceElementTransformations &Trans, const Vector &trfun, const Vector &elfun,
   Vector &elvec)
{
#ifdef MFEM_THREAD_SAFE
   Vector tr_shape, shape1;
#endif
   MFEM_VERIFY(trace_el.GetMapType() == FiniteElement::VALUE, "");

   if (Trans.Elem2No < 0) { type &= ~1; }

   const int dim = el.GetDim();
   const int tr_ndof = trace_el.GetDof();
   const int el_ndof = el.GetDof();
   const int ioff = (type & (HDGFaceType::ELEM | HDGFaceType::TRACE))?
                    (el_ndof):(0);

   Vector vu(dim), nor(dim);
   Vector &el_shape = shape1;

   tr_shape.SetSize(tr_ndof);
   el_shape.SetSize(el_ndof);

   int ndofs = 0;
   if (type & (HDGFaceType::ELEM | HDGFaceType::TRACE)) { ndofs += el_ndof; }
   if (type & (HDGFaceType::CONSTR | HDGFaceType::FACE)) { ndofs += tr_ndof; }
   elvec.SetSize(ndofs);
   elvec = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      // Assuming order(u)==order(mesh)
      if (type & 1)
      {
         order = Trans.Elem2->OrderW();
      }
      else
      {
         order = Trans.Elem1->OrderW();
      }
      order += 2*max(el.GetOrder(), trace_el.GetOrder());
      if (el.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration point
      const IntegrationPoint &eip = Trans.GetElement1IntPoint();

      trace_el.CalcShape(ip, tr_shape);
      if (type & 1)
      {
         el.CalcPhysShape(*Trans.Elem2, el_shape);
      }
      else
      {
         el.CalcPhysShape(*Trans.Elem1, el_shape);
      }

      u->Eval(vu, *Trans.Elem1, eip);

      if (dim == 1)
      {
         nor(0) = 2*eip.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      if (type & 1) { nor.Neg(); }

      const real_t un = vu * nor;
      const real_t a = alpha * un;
      const real_t b = fabs(alpha * un);
      // note: if |alpha/2|==|beta| then |a|==|b|, i.e. (a==b) or (a==-b)
      //       and therefore two blocks in the element matrix contribution
      //       (from the current quadrature point) are 0

      if (type & (HDGFaceType::ELEM | HDGFaceType::CONSTR))
      {
         const real_t v_q = el_shape * elfun;
         const real_t w = ip.weight * b * v_q;

         if (w != 0.0)
         {
            if (type & HDGFaceType::ELEM)
            {
               // assemble the element term
               for (int i = 0; i < el_ndof; i++)
               {
                  elvec(i) += w * el_shape(i);
               }
            }

            if (type & HDGFaceType::CONSTR)
            {
               // assemble the constraint term
               for (int i = 0; i < tr_ndof; i++)
               {
                  elvec(ioff+i) += w * tr_shape(i);
               }
            }
         }
      }

      if (type & (HDGFaceType::TRACE | HDGFaceType::FACE))
      {
         const real_t tr_q = tr_shape * trfun;

         if (type & HDGFaceType::TRACE)
         {
            const real_t w = ip.weight * (b-a) * tr_q;
            if (w != 0.0)
            {
               // assemble the trace term
               for (int i = 0; i < el_ndof; i++)
               {
                  elvec(i) -= w * el_shape(i);
               }
            }
         }

         if (type & HDGFaceType::FACE)
         {
            // assemble the face term
            // note: that this term must be non-zero at the boundary for stability
            //       reasons, so the advective part is intentionally dropped here
            //       and must be compensated elsewhere
            const real_t w = ip.weight * tr_q * ((Trans.Elem2No >= 0)?(b-a):(b));
            if (w != 0.0)
            {
               for (int i = 0; i < tr_ndof; i++)
               {
                  elvec(ioff+i) -= w * tr_shape(i);
               }
            }
         }
      }
   }
}

void HDGConvectionUpwindedIntegrator::AssembleHDGFaceMatrix(
   const FiniteElement &trace_el, const FiniteElement &el1,
   const FiniteElement &el2, FaceElementTransformations &Trans,
   DenseMatrix &elmat)
{
#ifdef MFEM_THREAD_SAFE
   Vector tr_shape, shape1, shape2;
#endif
   MFEM_VERIFY(trace_el.GetMapType() == FiniteElement::VALUE, "");

   const int el_dim = el1.GetDim();
   const int tr_ndof = trace_el.GetDof();
   const int ndof1 = el1.GetDof();
   const int ndof2 = (Trans.Elem2No >= 0)?(el2.GetDof()):(0);
   const int el_ndof = ndof1 + ndof2;

   Vector vu(el_dim), nor(el_dim);

   tr_shape.SetSize(tr_ndof);
   shape1.SetSize(ndof1);
   shape2.SetSize(ndof2);

   elmat.SetSize(el_ndof + tr_ndof);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      ir = &GetHDGFaceIntRule(trace_el, el1, el2, Trans);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration point
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();

      trace_el.CalcShape(ip, tr_shape);
      el1.CalcPhysShape(*Trans.Elem1, shape1);
      if (ndof2)
      {
         el2.CalcPhysShape(*Trans.Elem2, shape2);
      }

      u->Eval(vu, *Trans.Elem1, eip1);

      if (el_dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      const real_t un = vu * nor;
      const real_t a = 0.5 * alpha * un;
      const real_t b = beta * fabs(un);
      // note: if |alpha/2|==|beta| then |a|==|b|, i.e. (a==b) or (a==-b)
      //       and therefore two blocks in the element matrix contribution
      //       (from the current quadrature point) are 0

      real_t w = ip.weight * (b+a);
      if (w != 0.0)
      {
         // assemble the element matrix (elem1)
         for (int i = 0; i < ndof1; i++)
            for (int j = 0; j < ndof1; j++)
            {
               elmat(i, j) += w * shape1(i) * shape1(j);
            }
         // assemble the constraint matrix (elem1)
         for (int i = 0; i < tr_ndof; i++)
            for (int j = 0; j < ndof1; j++)
            {
               elmat(el_ndof+i, j) += w * tr_shape(i) * shape1(j);
            }
         // assemble the trace matrix (elem2)
         for (int i = 0; i < ndof2; i++)
            for (int j = 0; j < tr_ndof; j++)
            {
               elmat(ndof1+i, el_ndof+j) -= w * shape2(i) * tr_shape(j);
            }
      }

      w = ip.weight * (b-a);
      if (w != 0.0)
      {
         // assemble the element matrix (elem2)
         for (int i = 0; i < ndof2; i++)
            for (int j = 0; j < ndof2; j++)
            {
               elmat(ndof1+i, ndof1+j) += w * shape2(i) * shape2(j);
            }
         // assemble the constraint matrix (elem2)
         for (int i = 0; i < tr_ndof; i++)
            for (int j = 0; j < ndof2; j++)
            {
               elmat(el_ndof+i, ndof1+j) += w * tr_shape(i) * shape2(j);
            }
         // assemble the trace matrix (elem1)
         for (int i = 0; i < ndof1; i++)
            for (int j = 0; j < tr_ndof; j++)
            {
               elmat(i, el_ndof+j) -= w * shape1(i) * tr_shape(j);
            }
      }

      // assemble the face matrix
      // note: that this term must be non-zero at the boundary for stability
      //       reasons, so the advective part is intentionally dropped here
      //       and must be compensated elsewhere
      w = ip.weight * 2.*b;//<-- single face integration
      if (w != 0.0)
      {
         for (int i = 0; i < tr_ndof; i++)
            for (int j = 0; j < tr_ndof; j++)
            {
               elmat(el_ndof+i, el_ndof+j) -= w * tr_shape(i) * tr_shape(j);
            }
      }
   }
}

void HDGConvectionUpwindedIntegrator::AssembleHDGFaceMatrix(
   int side, const FiniteElement &trace_el, const FiniteElement &el,
   FaceElementTransformations &Trans, DenseMatrix &elmat)
{
#ifdef MFEM_THREAD_SAFE
   Vector tr_shape, shape1;
#endif
   MFEM_VERIFY(trace_el.GetMapType() == FiniteElement::VALUE, "");

   const int el_dim = el.GetDim();
   const int tr_ndof = trace_el.GetDof();
   const int el_ndof = el.GetDof();

   Vector vu(el_dim), nor(el_dim);
   Vector &el_shape = shape1;

   tr_shape.SetSize(tr_ndof);
   el_shape.SetSize(el_ndof);

   elmat.SetSize(el_ndof + tr_ndof);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      ir = &GetHDGFaceIntRule(side, trace_el, el, Trans);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration point
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();

      trace_el.CalcShape(ip, tr_shape);

      if (side != 0)
      {
         el.CalcPhysShape(*Trans.Elem2, el_shape);
      }
      else
      {
         el.CalcPhysShape(*Trans.Elem1, el_shape);
      }

      u->Eval(vu, *Trans.Elem1, eip1);

      if (el_dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      if (side != 0) { nor.Neg(); }

      const real_t un = vu * nor;
      const real_t a = 0.5 * alpha * un;
      const real_t b = beta * fabs(un);
      // note: if |alpha/2|==|beta| then |a|==|b|, i.e. (a==b) or (a==-b)
      //       and therefore two blocks in the element matrix contribution
      //       (from the current quadrature point) are 0

      real_t w = ip.weight * (b+a);
      if (w != 0.0)
      {
         // assemble the element matrix
         for (int i = 0; i < el_ndof; i++)
            for (int j = 0; j < el_ndof; j++)
            {
               elmat(i, j) += w * el_shape(i) * el_shape(j);
            }
         // assemble the constraint matrix
         for (int i = 0; i < tr_ndof; i++)
            for (int j = 0; j < el_ndof; j++)
            {
               elmat(el_ndof+i, j) += w * tr_shape(i) * el_shape(j);
            }
      }

      w = ip.weight * (b-a);
      if (w != 0.0)
      {
         // assemble the trace matrix (elem1)
         for (int i = 0; i < el_ndof; i++)
            for (int j = 0; j < tr_ndof; j++)
            {
               elmat(i, el_ndof+j) -= w * el_shape(i) * tr_shape(j);
            }
      }

      // assemble the face matrix
      // note: that this term must be non-zero at the boundary for stability
      //       reasons, so the advective part is intentionally dropped here
      //       and must be compensated elsewhere
      w = ip.weight * ((Trans.Elem2No >= 0)?(b-a):(2.*b));
      if (w != 0.0)
      {
         for (int i = 0; i < tr_ndof; i++)
            for (int j = 0; j < tr_ndof; j++)
            {
               elmat(el_ndof+i, el_ndof+j) -= w * tr_shape(i) * tr_shape(j);
            }
      }
   }
}

void HDGConvectionUpwindedIntegrator::AssembleHDGFaceVector(
   int type, const FiniteElement &trace_el, const FiniteElement &el,
   FaceElementTransformations &Trans, const Vector &trfun, const Vector &elfun,
   Vector &elvec)
{
#ifdef MFEM_THREAD_SAFE
   Vector tr_shape, shape1;
#endif
   MFEM_VERIFY(trace_el.GetMapType() == FiniteElement::VALUE, "");

   if (Trans.Elem2No < 0) { type &= ~1; }

   int tr_ndof, el_ndof;

   dim = el.GetDim();
   tr_ndof = trace_el.GetDof();
   el_ndof = el.GetDof();
   const int ioff = (type & (HDGFaceType::ELEM | HDGFaceType::TRACE))?
                    (el_ndof):(0);
   Vector vu(dim), nor(dim);

   tr_shape.SetSize(tr_ndof);
   Vector &el_shape = shape1;
   el_shape.SetSize(el_ndof);

   int ndofs = 0;
   if (type & (HDGFaceType::ELEM | HDGFaceType::TRACE)) { ndofs += el_ndof; }
   if (type & (HDGFaceType::CONSTR | HDGFaceType::FACE)) { ndofs += tr_ndof; }
   elvec.SetSize(ndofs);
   elvec = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      // Assuming order(u)==order(mesh)
      if (type & 1)
      {
         order = Trans.Elem2->OrderW();
      }
      else
      {
         order = Trans.Elem1->OrderW();
      }
      order += 2*max(el.GetOrder(), trace_el.GetOrder());
      if (el.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration point
      const IntegrationPoint &eip = Trans.GetElement1IntPoint();

      trace_el.CalcShape(ip, tr_shape);
      if (type & 1)
      {
         el.CalcPhysShape(*Trans.Elem2, el_shape);
      }
      else
      {
         el.CalcPhysShape(*Trans.Elem1, el_shape);
      }

      u->Eval(vu, *Trans.Elem1, eip);

      if (dim == 1)
      {
         nor(0) = 2*eip.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      if (type & 1) { nor.Neg(); }

      const real_t un = vu * nor;
      const real_t a = 0.5 * alpha * un;
      const real_t b = beta * fabs(un);
      // note: if |alpha/2|==|beta| then |a|==|b|, i.e. (a==b) or (a==-b)
      //       and therefore two blocks in the element matrix contribution
      //       (from the current quadrature point) are 0

      if (type & (HDGFaceType::ELEM | HDGFaceType::CONSTR))
      {
         const real_t v_q = el_shape * elfun;
         const real_t w = ip.weight * (b+a) * v_q;

         if (w != 0.0)
         {
            if (type & HDGFaceType::ELEM)
            {
               // assemble the element term
               for (int i = 0; i < el_ndof; i++)
               {
                  elvec(i) += w * el_shape(i);
               }
            }

            if (type & HDGFaceType::CONSTR)
            {
               // assemble the constraint term
               for (int i = 0; i < tr_ndof; i++)
               {
                  elvec(ioff+i) += w * tr_shape(i);
               }
            }
         }
      }

      if (type & (HDGFaceType::TRACE | HDGFaceType::FACE))
      {
         const real_t tr_q = tr_shape * trfun;
         if (type & HDGFaceType::TRACE)
         {
            // assemble the trace term
            const real_t w = ip.weight * (b-a) * tr_q;
            if (w != 0.0)
            {
               for (int i = 0; i < el_ndof; i++)
               {
                  elvec(i) -= w * el_shape(i);
               }
            }
         }

         if (type & HDGFaceType::FACE)
         {
            // assemble the face term
            // note: that this term must be non-zero at the boundary for stability
            //       reasons, so the advective part is intentionally dropped here
            //       and must be compensated elsewhere
            const real_t w = ip.weight * tr_q * (((Trans.Elem2No >= 0)?(b-a):(2.*b)));
            if (w != 0.0)
            {
               for (int i = 0; i < tr_ndof; i++)
               {
                  elvec(ioff+i) -= w * tr_shape(i);
               }
            }
         }
      }
   }
}

void HDGDiffusionIntegrator::AssembleFaceMatrix(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Trans, DenseMatrix &elmat)
{
#ifdef MFEM_THREAD_SAFE
   Vector shape1, shape2, nor, nh, ni;
   DenseMatrix mq;
#endif
   int dim, ndof1, ndof2, ndofs;

   dim = el1.GetDim();
   ndof1 = el1.GetDof();

   nor.SetSize(dim);
   nh.SetSize(dim);
   ni.SetSize(dim);
   if (MQ)
   {
      mq.SetSize(dim);
   }

   shape1.SetSize(ndof1);
   if (Trans.Elem2No >= 0)
   {
      ndof2 = el2.GetDof();
      shape2.SetSize(ndof2);
   }
   else
   {
      ndof2 = 0;
   }

   ndofs = ndof1 + ndof2;
   elmat.SetSize(ndofs);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // No trace element on this route, so the element orders are the whole rule.
      int order;
      if (ndof2)
      {
         order = 2*max(el1.GetOrder(), el2.GetOrder());
      }
      else
      {
         order = 2*el1.GetOrder();
      }
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   // assemble: alpha < {h^{-1} Q} [u],[v] >
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      el1.CalcPhysShape(*Trans.Elem1, shape1);
      {
         real_t wn = ip.weight/Trans.Elem1->Weight();
         if (ndof2)
         {
            wn /= 2;
         }
         if (!MQ)
         {
            if (Q)
            {
               wn *= Q->Eval(*Trans.Elem1, eip1);
            }
            ni.Set(wn, nor);
         }
         else
         {
            nh.Set(wn, nor);
            MQ->Eval(mq, *Trans.Elem1, eip1);
            mq.MultTranspose(nh, ni);
         }
      }
      real_t wq = ni * nor;
      // Note: in the jump term, we use 1/h1 = |nor|/det(J1) which is
      // independent of Loc1 and always gives the size of element 1 in
      // direction perpendicular to the face. Indeed, for linear transformation
      //     |nor|=measure(face)/measure(ref. face),
      //   det(J1)=measure(element)/measure(ref. element),
      // and the ratios measure(ref. element)/measure(ref. face) are
      // compatible for all element/face pairs.
      // For example: meas(ref. tetrahedron)/meas(ref. triangle) = 1/3, and
      // for any tetrahedron vol(tet)=(1/3)*height*area(base).
      // For interior faces: q_e/h_e=(q1/h1+q2/h2)/2.

      if (ndof2)
      {
         el2.CalcPhysShape(*Trans.Elem2, shape2);
         real_t wn = ip.weight/2/Trans.Elem2->Weight();
         if (!MQ)
         {
            if (Q)
            {
               wn *= Q->Eval(*Trans.Elem2, eip2);
            }
            ni.Set(wn, nor);
         }
         else
         {
            nh.Set(wn, nor);
            MQ->Eval(mq, *Trans.Elem2, eip2);
            mq.MultTranspose(nh, ni);
         }
         wq += ni * nor;
      }

      wq *= 0.5 * beta;

      // only assemble the lower triangular part
      for (int i = 0; i < ndof1; i++)
      {
         const real_t wsi = wq*shape1(i);
         for (int j = 0; j <= i; j++)
         {
            elmat(i, j) += wsi * shape1(j);
         }
      }
      if (ndof2)
      {
         for (int i = 0; i < ndof2; i++)
         {
            const int i2 = ndof1 + i;
            const real_t wsi = wq*shape2(i);
            for (int j = 0; j < ndof1; j++)
            {
               elmat(i2, j) -= wsi * shape1(j);
            }
            for (int j = 0; j <= i; j++)
            {
               elmat(i2, ndof1 + j) += wsi * shape2(j);
            }
         }
      }
   }

   // complete the upper triangular part
   for (int i = 0; i < ndofs; i++)
      for (int j = 0; j < i; j++)
      {
         elmat(j,i) = elmat(i,j);
      }
}

void HDGDiffusionIntegrator::AssembleHDGFaceMatrix(
   const FiniteElement &trace_el, const FiniteElement &el1,
   const FiniteElement &el2, FaceElementTransformations &Trans,
   DenseMatrix &elmat)
{
#ifdef MFEM_THREAD_SAFE
   Vector tr_shape, shape1, shape2, vu, nor, nh, ni;
   DenseMatrix mq;
#endif
   MFEM_VERIFY(trace_el.GetMapType() == FiniteElement::VALUE, "");
   MFEM_VERIFY(!stab || stab->IsConstant(),
               "A state dependent stabilization makes the face term nonlinear; "
               "assemble it through AssembleHDGFaceVector/Grad instead of as a "
               "bilinear form.");

   const int dim = el1.GetDim();
   const int tr_ndof = trace_el.GetDof();
   const int ndof1 = el1.GetDof();
   const int ndof2 = (Trans.Elem2No >= 0)?(el2.GetDof()):(0);

   vu.SetSize(dim);
   nor.SetSize(dim);
   nh.SetSize(dim);
   ni.SetSize(dim);
   if (MQ)
   {
      mq.SetSize(dim);
   }

   tr_shape.SetSize(tr_ndof);
   shape1.SetSize(ndof1);
   if (ndof2)
   {
      shape2.SetSize(ndof2);
   }
   const int el_ndof = ndof1 + ndof2;

   elmat.SetSize(el_ndof + tr_ndof);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      ir = &GetHDGFaceIntRule(trace_el, el1, el2, Trans);
   }

   // assemble: alpha < {h^{-1} Q} [u],[v] >
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      trace_el.CalcShape(ip, tr_shape);

      // evaluate the normal velocity
      real_t un;
      if (v)
      {
         v->Eval(vu, *Trans.Elem1, eip1);
         un = vu * nor;
      }
      else
      {
         un = 0.0;
      }

      // calculate the stabilization coefficient on side 1
      {
         el1.CalcPhysShape(*Trans.Elem1, shape1);
         real_t wn = ip.weight/Trans.Elem1->Weight();
         if (!MQ)
         {
            if (Q)
            {
               wn *= Q->Eval(*Trans.Elem1, eip1);
            }
            ni.Set(wn, nor);
         }
         else
         {
            nh.Set(wn, nor);
            MQ->Eval(mq, *Trans.Elem1, eip1);
            mq.MultTranspose(nh, ni);
         }
      }
      const real_t wq1 = ni * nor;
      // Note: in the jump term, we use 1/h1 = |nor|/det(J1) which is
      // independent of Loc1 and always gives the size of element 1 in
      // direction perpendicular to the face. Indeed, for linear transformation
      //     |nor|=measure(face)/measure(ref. face),
      //   det(J1)=measure(element)/measure(ref. element),
      // and the ratios measure(ref. element)/measure(ref. face) are
      // compatible for all element/face pairs.
      // For example: meas(ref. tetrahedron)/meas(ref. triangle) = 1/3, and
      // for any tetrahedron vol(tet)=(1/3)*height*area(base).
      // For interior faces: q_e/h_e=(q1/h1+q2/h2)/2.

      // calculate the stabilization coefficient on side 2
      real_t wq2;
      if (ndof2)
      {
         el2.CalcPhysShape(*Trans.Elem2, shape2);
         real_t wn = ip.weight/Trans.Elem2->Weight();
         if (!MQ)
         {
            if (Q)
            {
               wn *= Q->Eval(*Trans.Elem2, eip2);
            }
            ni.Set(wn, nor);
         }
         else
         {
            nh.Set(wn, nor);
            MQ->Eval(mq, *Trans.Elem2, eip2);
            mq.MultTranspose(nh, ni);
         }
         wq2 = ni * nor;
      }

      const real_t un_raw = un;
      real_t a, b;
      if (un != 0.)
      {
         un /= fabs(un);
         a = 0.5 * alpha * un;
         b = beta * fabs(un);
      }
      else
      {
         a = 0.0;
         b = beta;
      }
      const real_t face_w = ip.weight * nor.Norml2();

      // assemble side 1
      real_t w = StabValue(wq1, b+a, un_raw, face_w, 0., 0., *Trans.Elem1);
      if (w != 0.0)
      {
         // assemble the element matrix
         // (only the lower triangular part)
         for (int i = 0; i < ndof1; i++)
         {
            const real_t wsi = w*shape1(i);
            for (int j = 0; j <= i; j++)
            {
               elmat(i, j) += wsi * shape1(j);
            }
         }

         // assemble the constraint matrix
         for (int i = 0; i < ndof1; i++)
         {
            const real_t wsi = w*shape1(i);
            for (int j = 0; j < tr_ndof; j++)
            {
               elmat(i, el_ndof+j) -= wsi * tr_shape(j);
            }
         }
      }

      // assemble side 2
      if (ndof2)
      {
         w = StabValue(wq2, b-a, un_raw, face_w, 0., 0., *Trans.Elem2);
         if (w != 0.0)
         {
            // assemble the element matrix
            // (only the lower triangular part)
            for (int i = 0; i < ndof2; i++)
            {
               const real_t wsi = w*shape2(i);
               for (int j = 0; j <= i; j++)
               {
                  elmat(ndof1+i, ndof1+j) += wsi * shape2(j);
               }
            }

            // assemble the constraint matrix
            for (int i = 0; i < ndof2; i++)
            {
               const real_t wsi = w*shape2(i);
               for (int j = 0; j < tr_ndof; j++)
               {
                  elmat(i+ndof1, el_ndof+j) -= wsi * tr_shape(j);
               }
            }
         }
      }

      w = StabValue(wq1, b+a, un_raw, face_w, 0., 0., *Trans.Elem1);
      //<-- single face integration
      if (ndof2)
      {
         w += StabValue(wq2, b-a, un_raw, face_w, 0., 0., *Trans.Elem2);
      }
      if (w != 0.0)
      {
         // assemble the trace matrix
         for (int i = 0; i < tr_ndof; i++)
         {
            const real_t wsi = w*tr_shape(i);
            for (int j = 0; j <= i; j++)
            {
               elmat(el_ndof+i, el_ndof+j) -= wsi * tr_shape(j);
            }
         }
      }
   }

   // complete the element matrices
   // (the upper triangular part)
   for (int i = 0; i < ndof1; i++)
      for (int j = 0; j < i; j++)
      {
         elmat(j, i) = elmat(i, j);
      }

   for (int i = 0; i < ndof2; i++)
      for (int j = 0; j < i; j++)
      {
         elmat(ndof1+j, ndof1+i) = elmat(ndof1+i, ndof1+j);
      }

   // complete the constraint matrix
   for (int i = 0; i < el_ndof; i++)
      for (int j = 0; j < tr_ndof; j++)
      {
         elmat(el_ndof+j, i) = -elmat(i, el_ndof+j);
      }

   // complete the trace matrix
   for (int i = 0; i < tr_ndof; i++)
      for (int j = 0; j < i; j++)
      {
         elmat(el_ndof+j, el_ndof+i) = elmat(el_ndof+i, el_ndof+j);
      }
}

void HDGDiffusionIntegrator::AssembleHDGFaceMatrix(
   int side, const FiniteElement &trace_el, const FiniteElement &el,
   FaceElementTransformations &Trans, DenseMatrix &elmat)
{
#ifdef MFEM_THREAD_SAFE
   Vector tr_shape, shape1, vu, nor, nh, ni;
   DenseMatrix mq;
#endif
   MFEM_VERIFY(trace_el.GetMapType() == FiniteElement::VALUE, "");
   MFEM_VERIFY(!stab || stab->IsConstant(),
               "A state dependent stabilization makes the face term nonlinear; "
               "assemble it through AssembleHDGFaceVector/Grad instead of as a "
               "bilinear form.");

   const int dim = el.GetDim();
   const int tr_ndof = trace_el.GetDof();
   const int el_ndof = el.GetDof();

   vu.SetSize(dim);
   nor.SetSize(dim);
   nh.SetSize(dim);
   ni.SetSize(dim);
   if (MQ)
   {
      mq.SetSize(dim);
   }

   Vector &el_shape = shape1;
   tr_shape.SetSize(tr_ndof);
   el_shape.SetSize(el_ndof);

   elmat.SetSize(el_ndof + tr_ndof);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      ir = &GetHDGFaceIntRule(trace_el, el, Trans);
   }

   // assemble: alpha < {h^{-1} Q} [u],[v] >
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      if (side != 0) { nor.Neg(); }

      trace_el.CalcShape(ip, tr_shape);

      // evaluate the normal velocity
      real_t un;
      if (v)
      {
         v->Eval(vu, *Trans.Elem1, eip1);
         un = vu * nor;
      }
      else
      {
         un = 0.0;
      }

      // calculate the stabilization coefficient
      ElementTransformation *ElTr = (side != 0)?(Trans.Elem2):(Trans.Elem1);
      const IntegrationPoint &eip = (side != 0)?(eip2):(eip1);

      el.CalcPhysShape(*ElTr, el_shape);
      real_t wn = ip.weight/ElTr->Weight();
      if (!MQ)
      {
         if (Q)
         {
            wn *= Q->Eval(*ElTr, eip);
         }
         ni.Set(wn, nor);
      }
      else
      {
         nh.Set(wn, nor);
         MQ->Eval(mq, *ElTr, eip);
         mq.MultTranspose(nh, ni);
      }
      const real_t wq = ni * nor;
      // Note: in the jump term, we use 1/h1 = |nor|/det(J1) which is
      // independent of Loc1 and always gives the size of element 1 in
      // direction perpendicular to the face. Indeed, for linear transformation
      //     |nor|=measure(face)/measure(ref. face),
      //   det(J1)=measure(element)/measure(ref. element),
      // and the ratios measure(ref. element)/measure(ref. face) are
      // compatible for all element/face pairs.
      // For example: meas(ref. tetrahedron)/meas(ref. triangle) = 1/3, and
      // for any tetrahedron vol(tet)=(1/3)*height*area(base).
      // For interior faces: q_e/h_e=(q1/h1+q2/h2)/2.

      const real_t un_raw = un;
      real_t a, b;
      if (un != 0.)
      {
         un /= fabs(un);
         a = 0.5 * alpha * un;
         b = beta * fabs(un);
      }
      else
      {
         a = 0.0;
         b = beta;
      }

      const real_t face_w = ip.weight * nor.Norml2();
      real_t w = StabValue(wq, b+a, un_raw, face_w, 0., 0., *ElTr);
      if (w != 0.0)
      {
         // assemble the element matrix
         // (only the lower triangular part)
         for (int i = 0; i < el_ndof; i++)
         {
            const real_t wsi = w*el_shape(i);
            for (int j = 0; j <= i; j++)
            {
               elmat(i, j) += wsi * el_shape(j);
            }
         }

         // assemble the constraint matrix
         for (int i = 0; i < el_ndof; i++)
         {
            const real_t wsi = w*el_shape(i);
            for (int j = 0; j < tr_ndof; j++)
            {
               elmat(i, el_ndof+j) -= wsi * tr_shape(j);
            }
         }

         // assemble the trace matrix
         for (int i = 0; i < tr_ndof; i++)
         {
            const real_t wsi = w*tr_shape(i);
            for (int j = 0; j <= i; j++)
            {
               elmat(el_ndof+i, el_ndof+j) -= wsi * tr_shape(j);
            }
         }
      }
   }

   // complete the element matrix
   // (the upper triangular part)
   for (int i = 0; i < el_ndof; i++)
      for (int j = 0; j < i; j++)
      {
         elmat(j, i) = elmat(i, j);
      }

   // complete the constraint matrix
   for (int i = 0; i < el_ndof; i++)
      for (int j = 0; j < tr_ndof; j++)
      {
         elmat(el_ndof+j, i) = -elmat(i, el_ndof+j);
      }

   // complete the trace matrix
   for (int i = 0; i < tr_ndof; i++)
      for (int j = 0; j < i; j++)
      {
         elmat(el_ndof+j, el_ndof+i) = elmat(el_ndof+i, el_ndof+j);
      }
}

void HDGDiffusionIntegrator::AssembleHDGFaceVector(
   int type, const FiniteElement &trace_el, const FiniteElement &el,
   FaceElementTransformations &Trans, const Vector &trfun, const Vector &elfun,
   Vector &elvec)
{
#ifdef MFEM_THREAD_SAFE
   Vector tr_shape, shape1, vu, nor, nh, ni;
   DenseMatrix mq;
#endif
   MFEM_VERIFY(trace_el.GetMapType() == FiniteElement::VALUE, "");

   if (Trans.Elem2No < 0) { type &= ~1; }

   const int dim = el.GetDim();
   const int tr_ndof = trace_el.GetDof();
   const int el_ndof = el.GetDof();
   const int ioff = (type & (HDGFaceType::ELEM | HDGFaceType::TRACE))?
                    (el_ndof):(0);

   vu.SetSize(dim);
   nor.SetSize(dim);
   nh.SetSize(dim);
   ni.SetSize(dim);
   if (MQ)
   {
      mq.SetSize(dim);
   }

   tr_shape.SetSize(tr_ndof);
   Vector &el_shape = shape1;
   el_shape.SetSize(el_ndof);

   int ndofs = 0;
   if (type & (HDGFaceType::ELEM | HDGFaceType::TRACE)) { ndofs += el_ndof; }
   if (type & (HDGFaceType::CONSTR | HDGFaceType::FACE)) { ndofs += tr_ndof; }
   elvec.SetSize(ndofs);
   elvec = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // Degree 2*max(element, trace): see the note at the top of this file
      // for why the trace element has to be in the max.
      int order;
      order = 2*max(el.GetOrder(), trace_el.GetOrder());
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   // assemble: alpha < {h^{-1} Q} [u],[v] >
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      if (type & 1) { nor.Neg(); }

      trace_el.CalcShape(ip, tr_shape);

      // evaluate the normal velocity
      real_t un;
      if (v)
      {
         v->Eval(vu, *Trans.Elem1, eip1);
         un = vu * nor;
      }
      else
      {
         un = 0.0;
      }

      // calculate the stabilization coefficient
      ElementTransformation *ElTr = (type & 1)?(Trans.Elem2):(Trans.Elem1);
      const IntegrationPoint &eip = (type & 1)?(eip2):(eip1);

      el.CalcPhysShape(*ElTr, el_shape);
      real_t wn = ip.weight/ElTr->Weight();
      if (!MQ)
      {
         if (Q)
         {
            wn *= Q->Eval(*ElTr, eip);
         }
         ni.Set(wn, nor);
      }
      else
      {
         nh.Set(wn, nor);
         MQ->Eval(mq, *ElTr, eip);
         mq.MultTranspose(nh, ni);
      }
      const real_t wq = ni * nor;
      // Note: in the jump term, we use 1/h1 = |nor|/det(J1) which is
      // independent of Loc1 and always gives the size of element 1 in
      // direction perpendicular to the face. Indeed, for linear transformation
      //     |nor|=measure(face)/measure(ref. face),
      //   det(J1)=measure(element)/measure(ref. element),
      // and the ratios measure(ref. element)/measure(ref. face) are
      // compatible for all element/face pairs.
      // For example: meas(ref. tetrahedron)/meas(ref. triangle) = 1/3, and
      // for any tetrahedron vol(tet)=(1/3)*height*area(base).
      // For interior faces: q_e/h_e=(q1/h1+q2/h2)/2.

      const real_t un_raw = un;
      real_t a, b;
      if (un != 0.)
      {
         un /= fabs(un);
         a = 0.5 * alpha * un;
         b = beta * fabs(un);
      }
      else
      {
         a = 0.0;
         b = beta;
      }

      // Here wq = (ip.weight |nor|)(Q/h), so face_w below is the physical
      // measure at the point and the rest is the stabilization itself.
      real_t w;
      if (!stab)
      {
         w = wq * (b+a);
      }
      else
      {
         const real_t face_w = ip.weight * nor.Norml2();
         w = StabValue(wq, b+a, un_raw, face_w, el_shape * elfun,
                       tr_shape * trfun, *ElTr);
      }
      if (w == 0.) { continue; }

      if (type & (HDGFaceType::ELEM | HDGFaceType::CONSTR))
      {
         const real_t v_q = el_shape * elfun;
         const real_t wv = w * v_q;
         if (wv != 0.0)
         {
            if (type & HDGFaceType::ELEM)
            {
               // assemble the element term
               for (int i = 0; i < el_ndof; i++)
               {
                  elvec(i) += wv * el_shape(i);
               }
            }

            if (type & HDGFaceType::CONSTR)
            {
               // assemble the constraint term
               for (int i = 0; i < tr_ndof; i++)
               {
                  elvec(ioff+i) += wv * tr_shape(i);
               }
            }
         }
      }

      if (type & (HDGFaceType::TRACE | HDGFaceType::FACE))
      {
         const real_t tr_q = tr_shape * trfun;
         const real_t wt = w * tr_q;
         if (wt != 0.0)
         {
            if (type & HDGFaceType::TRACE)
            {
               // assemble the trace term
               for (int i = 0; i < el_ndof; i++)
               {
                  elvec(i) -= wt * el_shape(i);
               }
            }

            if (type & HDGFaceType::FACE)
            {
               // assemble the face term
               for (int i = 0; i < tr_ndof; i++)
               {
                  elvec(ioff+i) -= wt * tr_shape(i);
               }
            }
         }
      }
   }
}

void HDGDiffusionIntegrator::AssembleHDGFaceGrad(
   int type, const FiniteElement &trace_el, const FiniteElement &el,
   FaceElementTransformations &Trans, const Vector &trfun, const Vector &elfun,
   DenseMatrix &grad)
{
#ifdef MFEM_THREAD_SAFE
   Vector tr_shape, shape1, vu, nor, nh, ni;
   DenseMatrix mq;
#endif
   if (!stab || stab->IsConstant())
   {
      // Nothing state dependent to carry, so the base class building the
      // gradient out of the face matrix is exactly right. This is the path
      // every existing caller takes and it is left untouched.
      BilinearFormIntegrator::AssembleHDGFaceGrad(type, trace_el, el, Trans,
                                                  trfun, elfun, grad);
      return;
   }

   MFEM_VERIFY(trace_el.GetMapType() == FiniteElement::VALUE, "");

   if (Trans.Elem2No < 0) { type &= ~1; }

   const int dim = el.GetDim();
   const int tr_ndof = trace_el.GetDof();
   const int el_ndof = el.GetDof();

   MFEM_VERIFY(elfun.Size() == el_ndof && trfun.Size() == tr_ndof,
               "A state dependent stabilization is implemented for a single "
               "equation; for a system s becomes a matrix over the equations, "
               "which is a question about the formulation.");

   vu.SetSize(dim);
   nor.SetSize(dim);
   nh.SetSize(dim);
   ni.SetSize(dim);
   if (MQ) { mq.SetSize(dim); }

   tr_shape.SetSize(tr_ndof);
   Vector &el_shape = shape1;
   el_shape.SetSize(el_ndof);

   // Accumulate the full one-sided matrix in the [element | trace] layout that
   // AssembleHDGFaceMatrix() produces, so that the extraction at the end can
   // follow the base class exactly.
   DenseMatrix full(el_ndof + tr_ndof);
   full = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      ir = &GetHDGFaceIntRule(trace_el, el, Trans);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      Trans.SetAllIntPoints(&ip);

      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

      if (dim == 1) { nor(0) = 2*eip1.x - 1.0; }
      else { CalcOrtho(Trans.Jacobian(), nor); }

      if (type & 1) { nor.Neg(); }

      trace_el.CalcShape(ip, tr_shape);

      real_t un;
      if (v)
      {
         v->Eval(vu, *Trans.Elem1, eip1);
         un = vu * nor;
      }
      else
      {
         un = 0.0;
      }

      ElementTransformation *ElTr = (type & 1)?(Trans.Elem2):(Trans.Elem1);
      const IntegrationPoint &eip = (type & 1)?(eip2):(eip1);

      el.CalcPhysShape(*ElTr, el_shape);
      real_t wn = ip.weight/ElTr->Weight();
      if (!MQ)
      {
         if (Q) { wn *= Q->Eval(*ElTr, eip); }
         ni.Set(wn, nor);
      }
      else
      {
         nh.Set(wn, nor);
         MQ->Eval(mq, *ElTr, eip);
         mq.MultTranspose(nh, ni);
      }
      const real_t wq = ni * nor;

      const real_t un_raw = un;
      real_t a, b;
      if (un != 0.)
      {
         un /= fabs(un);
         a = 0.5 * alpha * un;
         b = beta * fabs(un);
      }
      else
      {
         a = 0.0;
         b = beta;
      }

      const real_t face_w = ip.weight * nor.Norml2();
      if (face_w == 0.) { continue; }
      const real_t s_diff = wq * (b+a) / face_w;

      const real_t u_q = el_shape * elfun;
      const real_t t_q = tr_shape * trfun;

      const real_t s = stab->Eval(s_diff, un_raw, u_q, t_q, *ElTr);
      real_t d1s, d2s;
      stab->EvalGrad(s_diff, un_raw, u_q, t_q, *ElTr, d1s, d2s);

      // The residual contributes s(u,uhat)(u - uhat) against both test spaces,
      // so its derivatives are s + d1s (u - uhat) with respect to the potential
      // and -s + d2s (u - uhat) with respect to the trace. Those are the
      // coefficients of the G, and of the E and H, blocks of Eq. (15) of
      // Nguyen, Peraire and Cockburn, once the convective term F'(uhat).n
      // carried by the convection integrators is set aside.
      const real_t jump = u_q - t_q;
      const real_t cu = face_w * ( s + d1s * jump);
      const real_t ct = face_w * (-s + d2s * jump);

      for (int i = 0; i < el_ndof; i++)
      {
         const real_t si = el_shape(i);
         for (int j = 0; j < el_ndof; j++)
         {
            full(i, j) += cu * si * el_shape(j);
         }
         for (int j = 0; j < tr_ndof; j++)
         {
            full(i, el_ndof + j) += ct * si * tr_shape(j);
         }
      }
      for (int i = 0; i < tr_ndof; i++)
      {
         const real_t si = tr_shape(i);
         for (int j = 0; j < el_ndof; j++)
         {
            full(el_ndof + i, j) += cu * si * el_shape(j);
         }
         for (int j = 0; j < tr_ndof; j++)
         {
            full(el_ndof + i, el_ndof + j) += ct * si * tr_shape(j);
         }
      }
   }

   int h = 0, w = 0;
   if (type & (HDGFaceType::ELEM | HDGFaceType::TRACE))  { h += el_ndof; }
   if (type & (HDGFaceType::CONSTR | HDGFaceType::FACE)) { h += tr_ndof; }
   if (type & (HDGFaceType::ELEM | HDGFaceType::CONSTR)) { w += el_ndof; }
   if (type & (HDGFaceType::TRACE | HDGFaceType::FACE))  { w += tr_ndof; }

   grad.SetSize(h, w);
   grad = 0.;

   int ioff = 0, joff = 0;
   if (type & HDGFaceType::ELEM)
   {
      grad.CopyMN(full, el_ndof, el_ndof, 0, 0, ioff, joff);
   }
   if (type & (HDGFaceType::ELEM | HDGFaceType::CONSTR)) { joff += el_ndof; }
   if (type & HDGFaceType::TRACE)
   {
      grad.CopyMN(full, el_ndof, tr_ndof, 0, el_ndof, ioff, joff);
   }
   if (type & (HDGFaceType::ELEM | HDGFaceType::TRACE)) { ioff += el_ndof; }
   joff = 0;
   if (type & HDGFaceType::CONSTR)
   {
      grad.CopyMN(full, tr_ndof, el_ndof, el_ndof, 0, ioff, joff);
   }
   if (type & (HDGFaceType::ELEM | HDGFaceType::CONSTR)) { joff += el_ndof; }
   if (type & HDGFaceType::FACE)
   {
      grad.CopyMN(full, tr_ndof, tr_ndof, el_ndof, el_ndof, ioff, joff);
   }
}

real_t HDGDiffusionIntegrator::ComputeHDGFaceEnergy(
   int side, const FiniteElement &trace_el, const FiniteElement &el,
   FaceElementTransformations &Trans, const Vector &trfun, const Vector &elfun,
   Vector *d_energy)
{
#ifdef MFEM_THREAD_SAFE
   Vector tr_shape, shape1, vu, nor, nh, ni;
   Vector nor_Jt, nor_Ji, ni_Jt, ni_Ji;
   DenseMatrix mq;
#endif
   MFEM_VERIFY(trace_el.GetMapType() == FiniteElement::VALUE, "");

   if (Trans.Elem2No < 0) { side = 0; }

   int dim, tr_ndof, el_ndof;

   dim = el.GetDim();
   tr_ndof = trace_el.GetDof();
   el_ndof = el.GetDof();

   vu.SetSize(dim);
   nor.SetSize(dim);
   nh.SetSize(dim);
   ni.SetSize(dim);
   if (MQ)
   {
      mq.SetSize(dim);
   }

   tr_shape.SetSize(tr_ndof);
   Vector &el_shape = shape1;
   el_shape.SetSize(el_ndof);

   if (d_energy)
   {
      d_energy->SetSize(dim);
      *d_energy = 0.;

      ni_Ji.SetSize(dim);
      nor_Jt.SetSize(dim);
      if (MQ)
      {
         nor_Ji.SetSize(dim);
         ni_Jt.SetSize(dim);
      }
   }
   real_t energy = 0.;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // Degree 2*max(element, trace): see the note at the top of this file
      // for why the trace element has to be in the max.
      int order;
      order = 2*max(el.GetOrder(), trace_el.GetOrder());
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   // assemble: alpha < {h^{-1} Q} [u],[v] >
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      if (side != 0) { nor.Neg(); }

      trace_el.CalcShape(ip, tr_shape);

      // evaluate the normal velocity
      real_t un;
      if (v)
      {
         v->Eval(vu, *Trans.Elem1, eip1);
         un = vu * nor;
      }
      else
      {
         un = 0.0;
      }

      // calculate the stabilization coefficient
      ElementTransformation *ElTr = (side != 0)?(Trans.Elem2):(Trans.Elem1);
      const IntegrationPoint &eip = (side != 0)?(eip2):(eip1);

      el.CalcPhysShape(*ElTr, el_shape);
      real_t wn = ip.weight/ElTr->Weight();
      if (!MQ)
      {
         if (Q)
         {
            wn *= Q->Eval(*ElTr, eip);
         }
         ni.Set(wn, nor);
      }
      else
      {
         nh.Set(wn, nor);
         MQ->Eval(mq, *ElTr, eip);
         mq.MultTranspose(nh, ni);
      }
      const real_t wq = ni * nor;
      // Note: in the jump term, we use 1/h1 = |nor|/det(J1) which is
      // independent of Loc1 and always gives the size of element 1 in
      // direction perpendicular to the face. Indeed, for linear transformation
      //     |nor|=measure(face)/measure(ref. face),
      //   det(J1)=measure(element)/measure(ref. element),
      // and the ratios measure(ref. element)/measure(ref. face) are
      // compatible for all element/face pairs.
      // For example: meas(ref. tetrahedron)/meas(ref. triangle) = 1/3, and
      // for any tetrahedron vol(tet)=(1/3)*height*area(base).
      // For interior faces: q_e/h_e=(q1/h1+q2/h2)/2.

      const real_t un_raw = un;
      real_t a, b;
      if (un != 0.)
      {
         un /= fabs(un);
         a = 0.5 * alpha * un;
         b = beta * fabs(un);
      }
      else
      {
         a = 0.0;
         b = beta;
      }

      const real_t v_q = el_shape * elfun;
      const real_t tr_q = tr_shape * trfun;
      const real_t d_q = v_q - tr_q;

      // The weighted stabilization, taken exactly as the four assembly paths
      // take it. With no hook StabValue() returns wq*(b+a), so the energy is
      // the same number as before to the last bit; with one installed this is
      // the only place the hook's value can enter, and the estimator reported
      // the built-in expression instead until it did.
      const real_t face_w = ip.weight * nor.Norml2();
      const real_t s_w = StabValue(wq, b+a, un_raw, face_w, v_q, tr_q, *ElTr);

      const real_t w = d_q*d_q * fabs(s_w);
      if (w == 0.) { continue; }

      energy += w;

      if (d_energy)
      {
         // The split is geometry: the weights below sum to ni.nor over the
         // directions, so dividing by it distributes exactly the energy just
         // added, whatever the stabilization was. A face where the diffusion
         // vanishes has no directions to split along and gets none of it.
         const real_t wg = (wq != 0.) ? (w / wq) : 0.;

         ElTr->InverseJacobian().Mult(ni, ni_Ji);
         ElTr->Jacobian().MultTranspose(nor, nor_Jt);
         if (!MQ)
         {
            for (int d = 0; d < dim; d++)
            {
               (*d_energy)(d) += wg * ni_Ji(d) * nor_Jt(d);
            }
         }
         else
         {
            // symmetrize the product as:
            // 1/2 * (n J J^-1 MQ J^-T J^T n + n J^-T J^T MQ J^-1 J n) =
            // 1/2 * (J^T n . J^-1 MQ n + J^-1 n . J^T MQ n)
            ElTr->Jacobian().MultTranspose(ni, ni_Jt);
            ElTr->InverseJacobian().Mult(nor, nor_Ji);

            for (int d = 0; d < dim; d++)
            {
               (*d_energy)(d) += 0.5 * wg * (ni_Jt(d) * nor_Ji(d) + ni_Ji(d) * nor_Jt(d));
            }
         }
      }
   }

   return energy;
}

namespace
{

/** @brief Whether one integration rule, asked of the FIRST element (or face),
    is every element's (or face's) rule -- so that a batched predicate need
    build no transformation beyond that one.

    An integrator's rule is a function of its element's order and geometry and
    of the transformation's ElementTransformation::OrderW(), which is a
    function of the transformation's own finite element: the mesh's linear one
    per geometry without nodes, the nodal space's with them. One element
    geometry, one order of @a fes and nodes of one order make all three the
    same everywhere. This replaces loops that built a transformation per
    element or per face to ask each the same question -- the predicates' share
    of the host time meq measured under a Device, paid twice per assembly
    because every batched routine re-asks its predicate. */
bool HDGOneRule(const FiniteElementSpace &fes)
{
   const Mesh &mesh = *fes.GetMesh();
   if (mesh.GetNE() == 0) { return false; }
   if (mesh.GetNumGeometries(mesh.Dimension()) != 1) { return false; }
   if (fes.IsVariableOrder()) { return false; }
   const GridFunction *nodes = mesh.GetNodes();
   if (nodes && nodes->FESpace()->IsVariableOrder()) { return false; }
   return true;
}

/** @brief Whether every transformation of geometry @a g on @a mesh is
    AFFINE, so that its Jacobian -- and with it a face's normal and an
    element's weight -- is one value per entity rather than one per point.

    A mesh without nodes maps each entity by the linear element of its
    geometry, and on a simplex (a segment, a triangle, a tetrahedron) that
    element's derivatives are constants: the Jacobian is the same NUMBER at
    every point, bit for bit, so computing it once per entity changes nothing
    but the cost. Not a quadrilateral or a hexahedron, which are bilinear. */
bool HDGAffine(const Mesh &mesh, Geometry::Type g)
{
   if (mesh.GetNodes()) { return false; }
   return g == Geometry::SEGMENT || g == Geometry::TRIANGLE
          || g == Geometry::TETRAHEDRON;
}

} // namespace

real_t HDGRuleStabilization::Eval(real_t s_diff, real_t un, real_t u,
                                  real_t uhat, ElementTransformation &Tr) const
{
   MFEM_CONTRACT_VAR(s_diff);
   MFEM_CONTRACT_VAR(un);
   MFEM_CONTRACT_VAR(u);
   MFEM_CONTRACT_VAR(uhat);
   real_t x[3] = { 0.0, 0.0, 0.0 };
   if (rule.coord >= 0)
   {
      MFEM_ASSERT(rule.coord < Tr.GetSpaceDim(), "coordinate out of range");
      Vector X(x, Tr.GetSpaceDim());
      Tr.Transform(Tr.GetIntPoint(), X);
   }
   return rule.Eval(x);
}

namespace
{

/// The weight of a Jacobian @a m, column-major sd x d, in DenseMatrix::Weight()'s
/// expressions case by case.
MFEM_HOST_DEVICE inline real_t HDGJacobianWeight(const real_t *m, int sd, int d)
{
   if (sd == d && d == 1) { return m[0]; }
   if (sd == d && d == 2) { return m[0] * m[3] - m[1] * m[2]; }
   if (sd == d && d == 3)
   {
      return m[0] * (m[4] * m[8] - m[5] * m[7]) +
             m[3] * (m[2] * m[7] - m[1] * m[8]) +
             m[6] * (m[1] * m[5] - m[2] * m[4]);
   }
   if (d == 1 && sd == 2) { return sqrt(m[0] * m[0] + m[1] * m[1]); }
   if (d == 1) { return sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]); }
   const real_t E = m[0] * m[0] + m[1] * m[1] + m[2] * m[2];
   const real_t G = m[3] * m[3] + m[4] * m[4] + m[5] * m[5];
   const real_t F = m[0] * m[3] + m[1] * m[4] + m[2] * m[5];
   return sqrt(E * G - F * F);
}

/** @brief Every element's point matrix, (NV, SDIM, NE) -- the layout an
    ElementRestriction gives -- and the transformation element it goes with.

    The values are what Mesh::GetElementTransformation() puts in PointMat: the
    vertices and the geometry's linear element on a mesh without nodes, the
    nodes and their space's element on a mesh with them. Gathered on the host,
    a read and a copy per entry, so that the caller's nodes are not read on
    the device by a routine that only wanted their values. One element
    geometry and nodes (if any) of one order, which the callers establish. */
const FiniteElement &HDGElementPointMatrices(const Mesh &mesh, Vector &X)
{
   const int NE = mesh.GetNE(), sdim = mesh.SpaceDimension();
   const GridFunction *nodes = mesh.GetNodes();
   const FiniteElement &tfe = nodes ? *nodes->FESpace()->GetFE(0)
                              : *Mesh::GetTransformationFEforElementType(
                                 mesh.GetElementType(0));
   const int NV = tfe.GetDof();
   X.UseDevice(true);
   X.SetSize(NV * sdim * NE);
   real_t *px = X.HostWrite();
   if (!nodes)
   {
      for (int e = 0; e < NE; e++)
      {
         const int *v = mesh.GetElement(e)->GetVertices();
         for (int k = 0; k < NV; k++)
         {
            const real_t *x = mesh.GetVertex(v[k]);
            for (int i = 0; i < sdim; i++) { px[k + NV * (i + sdim * e)] = x[i]; }
         }
      }
   }
   else
   {
      const real_t *pn = nodes->HostRead();
      Array<int> vdofs;
      for (int e = 0; e < NE; e++)
      {
         nodes->FESpace()->GetElementVDofs(e, vdofs);
         MFEM_ASSERT(vdofs.Size() == NV * sdim, "nodes of one order");
         for (int i = 0; i < sdim; i++)
            for (int k = 0; k < NV; k++)
            {
               px[k + NV * (i + sdim * e)] = pn[vdofs[NV * i + k]];
            }
      }
   }
   return tfe;
}

/// HDGElementGeometry's kernel, a free function because nvcc refuses an
/// extended lambda inside a constructor.
void HDGVertexGeometryKernel(int NQ, int NV, int sdim, int dim, int NE,
                             const Vector &dsh, const Vector &X, Vector &J,
                             Vector &detJ, bool want_J, bool want_det)
{
   const auto d_ds = Reshape(dsh.Read(), NQ, NV, dim);
   const auto d_X = Reshape(X.Read(), NV, sdim, NE);
   real_t *d_J = want_J ? J.Write() : nullptr;
   real_t *d_det = want_det ? detJ.Write() : nullptr;
   const int nq = NQ, nv = NV, sd = sdim, d = dim;
   mfem::forall(NQ * NE, [=] MFEM_HOST_DEVICE (int qe)
   {
      const int q = qe % nq, e = qe / nq;
      // Column-major sd x d, as DenseMatrix holds dFdx. Each entry summed
      // over the vertices from zero in order, as kernels::AddMult() does.
      real_t Jq[9];
      for (int j = 0; j < d; j++)
         for (int i = 0; i < sd; i++)
         {
            real_t s = 0.0;
            for (int k = 0; k < nv; k++) { s += d_ds(q, k, j) * d_X(k, i, e); }
            Jq[i + sd * j] = s;
         }
      if (d_J)
      {
         for (int j = 0; j < d; j++)
            for (int i = 0; i < sd; i++)
            {
               d_J[q + nq * (i + sd * (j + d * e))] = Jq[i + sd * j];
            }
      }
      if (d_det) { d_det[qe] = HDGJacobianWeight(Jq, sd, d); }
   });
}

} // namespace

HDGElementGeometry::HDGElementGeometry(Mesh &mesh, const IntegrationRule &ir,
                                       int flags)
{
   if (mesh.GetNodes())
   {
      gf = mesh.GetGeometricFactors(ir, flags);
      return;
   }

   const int NE = mesh.GetNE(), NQ = ir.GetNPoints();
   const int sdim = mesh.SpaceDimension(), dim = mesh.Dimension();
   const bool want_J = (flags & GeometricFactors::JACOBIANS);
   const bool want_det = (flags & GeometricFactors::DETERMINANTS);
   J_.UseDevice(true);
   detJ_.UseDevice(true);
   if (want_J) { J_.SetSize(NQ * sdim * dim * NE); }
   if (want_det) { detJ_.SetSize(NQ * NE); }
   if (NE == 0 || !(want_J || want_det)) { return; }
   MFEM_VERIFY(mesh.GetNumGeometries(dim) == 1,
               "one element geometry, as the batched kernels require");
   MFEM_VERIFY(sdim == dim || (sdim == 2 && dim == 1) ||
               (sdim == 3 && dim <= 2), "unsupported dimensions");

   // The reference derivatives of the transformation element, one table, and
   // every element's vertices, which is all Mesh::GetPointMatrix() reads.
   Vector X;
   const FiniteElement &tfe = HDGElementPointMatrices(mesh, X);
   const int NV = tfe.GetDof();
   Vector dsh(NQ * NV * dim);
   dsh.UseDevice(true);
   {
      DenseMatrix ds(NV, dim);
      real_t *pd = dsh.HostWrite();
      for (int q = 0; q < NQ; q++)
      {
         tfe.CalcDShape(ir.IntPoint(q), ds);
         for (int j = 0; j < dim; j++)
            for (int k = 0; k < NV; k++) { pd[q + NQ * (k + NV * j)] = ds(k, j); }
      }
   }

   HDGVertexGeometryKernel(NQ, NV, sdim, dim, NE, dsh, X, J_, detJ_,
                           want_J, want_det);
}

bool HDGElementPointsOnDevice(const Mesh &mesh)
{
   // A NURBS element's shapes depend on the element through its knots, so one
   // table cannot serve the mesh.
   if (mesh.GetNE() == 0 || mesh.NURBSext) { return false; }
   const int dim = mesh.Dimension(), sdim = mesh.SpaceDimension();
   if (mesh.GetNumGeometries(dim) != 1) { return false; }
   if (!(sdim == dim || (sdim == 2 && dim == 1) || (sdim == 3 && dim <= 2)))
   {
      return false;
   }
   const GridFunction *nodes = mesh.GetNodes();
   return !nodes || !nodes->FESpace()->IsVariableOrder();
}

void HDGElementPointsAndWeights(const Mesh &mesh, const IntegrationRule &ir,
                                Vector &X, Vector &W)
{
   MFEM_VERIFY(HDGElementPointsOnDevice(mesh),
               "element points and weights on the host for this mesh");
   const int NE = mesh.GetNE(), NQ = ir.GetNPoints();
   const int sdim = mesh.SpaceDimension(), dim = mesh.Dimension();
   Vector PM;
   const FiniteElement &tfe = HDGElementPointMatrices(mesh, PM);
   const int NV = tfe.GetDof();

   Vector sh(NQ * NV), ds(NQ * NV * dim), qw(NQ);
   sh.UseDevice(true);
   ds.UseDevice(true);
   qw.UseDevice(true);
   {
      Vector s(NV);
      DenseMatrix dsk(NV, dim);
      real_t *ps = sh.HostWrite(), *pd = ds.HostWrite(), *pw = qw.HostWrite();
      for (int q = 0; q < NQ; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         tfe.CalcShape(ip, s);
         tfe.CalcDShape(ip, dsk);
         for (int k = 0; k < NV; k++) { ps[q + NQ * k] = s(k); }
         for (int j = 0; j < dim; j++)
            for (int k = 0; k < NV; k++) { pd[q + NQ * (k + NV * j)] = dsk(k, j); }
         pw[q] = ip.weight;
      }
   }

   X.UseDevice(true);
   W.UseDevice(true);
   X.SetSize(sdim * NQ * NE);
   W.SetSize(NQ * NE);
   const auto d_sh = Reshape(sh.Read(), NQ, NV);
   const auto d_ds = Reshape(ds.Read(), NQ, NV, dim);
   const auto d_P = Reshape(PM.Read(), NV, sdim, NE);
   const real_t *d_qw = qw.Read();
   real_t *d_x = X.Write(), *d_w = W.Write();
   const int nq = NQ, nv = NV, sd = sdim, d = dim;
   mfem::forall(NQ * NE, [=] MFEM_HOST_DEVICE (int qe)
   {
      const int q = qe % nq, e = qe / nq;
      // Transform(): PointMat * shape, each coordinate summed over the
      // vertices in order. Weight(): J = PointMat * DShape the same way, and
      // its weight as DenseMatrix::Weight() takes it.
      for (int i = 0; i < sd; i++)
      {
         real_t s = 0.0;
         for (int k = 0; k < nv; k++) { s += d_P(k, i, e) * d_sh(q, k); }
         d_x[i + sd * qe] = s;
      }
      real_t Jq[9];
      for (int j = 0; j < d; j++)
         for (int i = 0; i < sd; i++)
         {
            real_t s = 0.0;
            for (int k = 0; k < nv; k++) { s += d_ds(q, k, j) * d_P(k, i, e); }
            Jq[i + sd * j] = s;
         }
      d_w[qe] = d_qw[q] * HDGJacobianWeight(Jq, sd, d);
   });
}

bool HDGCoefficientOnDevice(Coefficient *Q, const Mesh &mesh)
{
   if (!Q) { return false; }
   if (dynamic_cast<ConstantCoefficient*>(Q)) { return true; }
   auto *gq = dynamic_cast<GridFunctionCoefficient*>(Q);
   if (!gq || !gq->GetGridFunction()) { return false; }
   const FiniteElementSpace &gfes = *gq->GetGridFunction()->FESpace();
   // The coefficient's own mesh, element for element: GridFunctionCoefficient
   // takes a different route on a mesh of another size.
   if (gfes.GetMesh() != &mesh || mesh.GetNE() == 0) { return false; }
   if (mesh.GetNumGeometries(mesh.Dimension()) != 1) { return false; }
   if (gfes.IsVariableOrder()) { return false; }
   const int comp = gq->GetComponent();
   if (comp < 1 || comp > gfes.GetVDim()) { return false; }
   const FiniteElement &fe = *gfes.GetFE(0);
   return fe.GetMapType() == FiniteElement::VALUE
          && fe.GetRangeType() == FiniteElement::SCALAR;
}

namespace
{

/** @brief @a Q at points of elements, on the device: @a out(q, k) is @a Q at
    point q of @a pts[@a item_slot[k]] on element @a item_el[k].

    @a Q must pass HDGCoefficientOnDevice(). A GridFunctionCoefficient is
    gathered by its space's ElementRestriction and summed against one shape
    table per rule in @a pts, which is GridFunction::GetValue()'s arithmetic
    at a VALUE-mapped element. */
void HDGEvalCoefficient(Coefficient &Q, int NQ,
                        const std::vector<IntegrationRule> &pts,
                        const Array<int> &item_el, const Array<int> &item_slot,
                        Vector &out)
{
   const int NI = item_el.Size();
   out.UseDevice(true);
   out.SetSize(NQ * NI);
   if (auto *cq = dynamic_cast<ConstantCoefficient*>(&Q))
   {
      out = cq->constant;
      return;
   }
   auto &gq = static_cast<GridFunctionCoefficient&>(Q);
   const GridFunction &gf = *gq.GetGridFunction();
   const FiniteElementSpace &gfes = *gf.FESpace();
   const FiniteElement &fe = *gfes.GetFE(0);
   const int ND = fe.GetDof(), VD = gfes.GetVDim(), NE = gfes.GetNE();
   const int NS = (int)pts.size(), comp = gq.GetComponent() - 1;

   Vector tab(NQ * ND * NS);
   tab.UseDevice(true);
   {
      Vector sh(ND);
      real_t *pt = tab.HostWrite();
      for (int s = 0; s < NS; s++)
         for (int q = 0; q < NQ; q++)
         {
            fe.CalcShape(pts[s].IntPoint(q), sh);
            for (int i = 0; i < ND; i++) { pt[q + NQ * (i + ND * s)] = sh(i); }
         }
   }
   const Operator *R = gfes.GetElementRestriction(ElementDofOrdering::NATIVE);
   Vector ev(R->Height());
   ev.UseDevice(true);
   R->Mult(gf, ev);

   const auto d_t = Reshape(tab.Read(), NQ, ND, NS);
   const auto d_v = Reshape(ev.Read(), ND, VD, NE);
   const int *d_el = item_el.Read(), *d_sl = item_slot.Read();
   real_t *d_o = out.Write();
   const int nq = NQ, nd = ND, c = comp;
   mfem::forall(NQ * NI, [=] MFEM_HOST_DEVICE (int qk)
   {
      const int q = qk % nq, k = qk / nq;
      const int e = d_el[k], s = d_sl[k];
      real_t sum = 0.0;
      for (int i = 0; i < nd; i++) { sum += d_t(q, i, s) * d_v(i, c, e); }
      d_o[qk] = sum;
   });
}

/** @brief Whether the face kernels can take a face's geometry on the device
    on @a mesh: dimension 2 or 3 equal to the space dimension, one element
    geometry with one face geometry (not a prism or a pyramid), and no nodes
    or CONTINUOUS nodes of one order -- the cases in which
    Mesh::GetFaceTransformation() reads a face's own vertices or node dofs.
    Discontinuous nodes, a periodic mesh's, go through an element there, and
    stay on the host here. A NURBS mesh is refused, as its shapes vary by
    element; the per-face route does not run there either (ConstructC()
    faults in NURBS1DFiniteElement::CalcDShape(), probed 2026-10-28). */
bool HDGFaceGeometryOnDevice(const Mesh &mesh)
{
   if (mesh.GetNE() == 0 || mesh.NURBSext) { return false; }
   const int dim = mesh.Dimension();
   if ((dim != 2 && dim != 3) || mesh.SpaceDimension() != dim) { return false; }
   if (mesh.GetNumGeometries(dim) != 1) { return false; }
   const Geometry::Type g = mesh.GetElementBaseGeometry(0);
   if (g == Geometry::PRISM || g == Geometry::PYRAMID) { return false; }
   const GridFunction *nodes = mesh.GetNodes();
   if (!nodes) { return true; }
   const FiniteElementSpace &nfes = *nodes->FESpace();
   return !nfes.IsVariableOrder() && nfes.FEColl()->GetContType()
          == FiniteElementCollection::CONTINUOUS;
}

/** @brief The point matrices of @a faces, (NVF, SDIM, NF), and the
    transformation element they go with -- what
    Mesh::GetFaceTransformation() puts in PointMat on a mesh that
    HDGFaceGeometryOnDevice() admits: the face's vertices without nodes, its
    node dofs with them. Gathered on the host, as HDGElementPointMatrices()
    gathers elements. */
const FiniteElement &HDGFacePointMatrices(const Mesh &mesh,
                                          const Array<int> &faces, Vector &X)
{
   const int NF = faces.Size(), sdim = mesh.SpaceDimension();
   const GridFunction *nodes = mesh.GetNodes();
   const FiniteElement &tfe = nodes
                              ? *nodes->FESpace()->GetFaceElement(faces[0])
                              : *Mesh::GetTransformationFEforElementType(
                                 mesh.GetFaceElementType(faces[0]));
   const int NV = tfe.GetDof();
   X.UseDevice(true);
   X.SetSize(NV * sdim * NF);
   real_t *px = X.HostWrite();
   if (!nodes)
   {
      Array<int> v;
      for (int k = 0; k < NF; k++)
      {
         mesh.GetFaceVertices(faces[k], v);
         MFEM_ASSERT(v.Size() == NV, "one face geometry");
         for (int j = 0; j < NV; j++)
         {
            const real_t *x = mesh.GetVertex(v[j]);
            for (int i = 0; i < sdim; i++) { px[j + NV * (i + sdim * k)] = x[i]; }
         }
      }
   }
   else
   {
      const real_t *pn = nodes->HostRead();
      Array<int> vdofs;
      for (int k = 0; k < NF; k++)
      {
         nodes->FESpace()->GetFaceVDofs(faces[k], vdofs);
         MFEM_ASSERT(vdofs.Size() == NV * sdim, "nodes of one order");
         for (int i = 0; i < sdim; i++)
            for (int j = 0; j < NV; j++)
            {
               px[j + NV * (i + sdim * k)] = pn[vdofs[NV * i + j]];
            }
      }
   }
   return tfe;
}

/** @brief The geometry of a set of one-sided face items, packed for a
    kernel: everything a face routine used to ask a face transformation and
    an element transformation for, at every quadrature point of every item,
    with neither built.

    An item is (face, element, slot): a face of @a faces, the element whose
    side of it is integrated, and that side's (local face, orientation) slot.
    The NORMAL is taken from the face's own map, in the order
    IsoparametricTransformation::EvalJacobian() takes it, so it is
    CalcOrtho() of the face transformation's Jacobian exactly. The element's
    JACOBIAN and PHYSICAL POINT are taken from the element's map at the slot's
    reference points, as the element transformation the per-face routines
    set to those points gives them. On the host the arithmetic is theirs in
    their order, so the weights come out as the per-face loops' bits; a GPU
    contracts to FMAs and gives round-off instead.

    Tables are O(slots) and O(points); the point matrices are the only
    O(elements) and O(faces) parts. */
struct HDGSideMaps
{
   int NQ = 0, dim = 0;
   int NV = 0, NS = 0, NE = 0;   // the element maps
   int NVF = 0, NF = 0;          // the face maps
   Vector X;    ///< (NV, dim, NE) element point matrices
   Vector tsh;  ///< (NQ, NV, NS) element transformation shapes at the slots
   Vector tds;  ///< (NQ, NV, dim, NS) and their reference derivatives
   Vector XF;   ///< (NVF, dim, NF) face point matrices
   Vector fds;  ///< (NQ, NVF, dim-1) face transformation derivatives

   /// @a elements false packs the face maps alone, for the normals.
   HDGSideMaps(const Mesh &mesh, const IntegrationRule &ir,
               const std::vector<IntegrationRule> &slot_eip,
               const Array<int> &faces, bool elements = true)
   {
      MFEM_VERIFY(HDGFaceGeometryOnDevice(mesh), "face geometry on the host");
      NQ = ir.GetNPoints();
      NS = (int)slot_eip.size();
      dim = mesh.Dimension();
      NE = mesh.GetNE();
      NF = faces.Size();
      X.UseDevice(true);
      tsh.UseDevice(true);
      tds.UseDevice(true);
      fds.UseDevice(true);
      if (elements) { PackElements(mesh, slot_eip); }
      if (NF == 0) { return; }
      const FiniteElement &ffe = HDGFacePointMatrices(mesh, faces, XF);
      NVF = ffe.GetDof();
      fds.SetSize(NQ * NVF * (dim - 1));
      DenseMatrix ds(NVF, dim - 1);
      real_t *pd = fds.HostWrite();
      for (int q = 0; q < NQ; q++)
      {
         ffe.CalcDShape(ir.IntPoint(q), ds);
         for (int j = 0; j < dim - 1; j++)
            for (int k = 0; k < NVF; k++) { pd[q + NQ * (k + NVF * j)] = ds(k, j); }
      }
   }

private:
   void PackElements(const Mesh &mesh,
                     const std::vector<IntegrationRule> &slot_eip)
   {
      const FiniteElement &tfe = HDGElementPointMatrices(mesh, X);
      NV = tfe.GetDof();
      tsh.SetSize(NQ * NV * NS);
      tds.SetSize(NQ * NV * dim * NS);
      {
         Vector sh(NV);
         DenseMatrix ds(NV, dim);
         real_t *ps = tsh.HostWrite(), *pd = tds.HostWrite();
         for (int s = 0; s < NS; s++)
            for (int q = 0; q < NQ; q++)
            {
               const IntegrationPoint &eip = slot_eip[s].IntPoint(q);
               tfe.CalcShape(eip, sh);
               tfe.CalcDShape(eip, ds);
               for (int k = 0; k < NV; k++)
               {
                  ps[q + NQ * (k + NV * s)] = sh(k);
                  for (int j = 0; j < dim; j++)
                  {
                     pd[q + NQ * (k + NV * (j + dim * s))] = ds(k, j);
                  }
               }
            }
      }
   }
};

/** @brief CalcOrtho() of face @a f's Jacobian at point @a q, the Jacobian
    summed over the face's points in order as EvalJacobian() sums it. */
MFEM_HOST_DEVICE inline void HDGFaceNormal(
   int q, int f, int nvf, int d, const DeviceTensor<3, const real_t> &fds,
   const DeviceTensor<3, const real_t> &XF, real_t *n)
{
   real_t Jf[6];
   for (int c = 0; c < d - 1; c++)
      for (int i = 0; i < d; i++)
      {
         real_t sum = 0.0;
         for (int v = 0; v < nvf; v++) { sum += XF(v, i, f) * fds(q, v, c); }
         Jf[i + d * c] = sum;
      }
   if (d == 2)
   {
      n[0] = Jf[1];
      n[1] = -Jf[0];
   }
   else
   {
      n[0] = Jf[1] * Jf[5] - Jf[2] * Jf[4];
      n[1] = Jf[2] * Jf[3] - Jf[0] * Jf[5];
      n[2] = Jf[0] * Jf[4] - Jf[1] * Jf[3];
   }
}

/** @brief The face normal of every item at every point, on the device, as
    `CalcOrtho(face transformation Jacobian)` gives it, negated where
    @a item_neg is nonzero. @a nor is (dim, NQ, NI). See HDGSideMaps. */
void HDGFaceNormalsDevice(const HDGSideMaps &m, const Array<int> &item_face,
                          const Array<int> &item_neg, Vector &nor)
{
   const int NI = item_face.Size(), NQ = m.NQ, dim = m.dim;
   nor.UseDevice(true);
   nor.SetSize(dim * NQ * NI);
   if (NI == 0) { return; }
   const auto d_fds = Reshape(m.fds.Read(), NQ, m.NVF, dim - 1);
   const auto d_XF = Reshape(m.XF.Read(), m.NVF, dim, m.NF);
   const int *d_f = item_face.Read(), *d_ng = item_neg.Read();
   real_t *d_n = nor.Write();
   const int nq = NQ, nvf = m.NVF, d = dim;
   mfem::forall(NQ * NI, [=] MFEM_HOST_DEVICE (int qk)
   {
      const int q = qk % nq, k = qk / nq;
      real_t n[3];
      HDGFaceNormal(q, d_f[k], nvf, d, d_fds, d_XF, n);
      // -x where the per-face routine calls Vector::Neg(): the same number.
      for (int i = 0; i < d; i++) { d_n[i + d * qk] = d_ng[k] ? -n[i] : n[i]; }
   });
}

/** @brief HDGDiffusionIntegrator's one-sided face weight -- what its
    StabValue() returns, at `b + a = beta` -- at every point of every item,
    on the device. See HDGSideMaps for what an item is.

    Without a velocity the normal's sign is never read -- only `n.n` and `|n|`
    enter the weight -- so one routine serves either side of a face.

    @a dif must pass HDGFaceWeightsOnDevice(). @a w is (NQ, NI). */
void HDGFaceWeightsDevice(const HDGDiffusionIntegrator &dif,
                          const HDGSideMaps &m, const IntegrationRule &ir,
                          const std::vector<IntegrationRule> &slot_eip,
                          const Array<int> &item_face,
                          const Array<int> &item_el,
                          const Array<int> &item_slot, Vector &w)
{
   const int NQ = m.NQ, NI = item_el.Size();
   w.UseDevice(true);
   w.SetSize(NQ * NI);
   if (NI == 0) { return; }

   HDGStabilizationRule rule;
   const HDGStabilization *st = dif.GetStabilization();
   const bool has_rule = st && st->GetRule(rule);
   MFEM_VERIFY(!st || has_rule, "a stabilization without a rule");

   Vector qw(NQ);
   qw.UseDevice(true);
   {
      real_t *pw = qw.HostWrite();
      for (int q = 0; q < NQ; q++) { pw[q] = ir.IntPoint(q).weight; }
   }

   // The coefficient, where the built-in stabilization reads it. A rule does
   // not: it never sees s_diff, which is the only place the coefficient goes.
   Coefficient *Q = has_rule ? nullptr : dif.GetCoefficient();
   Vector qv;
   if (Q) { HDGEvalCoefficient(*Q, NQ, slot_eip, item_el, item_slot, qv); }

   const int dim = m.dim;
   const auto d_sh = Reshape(m.tsh.Read(), NQ, m.NV, m.NS);
   const auto d_ds = Reshape(m.tds.Read(), NQ, m.NV, dim, m.NS);
   const auto d_X = Reshape(m.X.Read(), m.NV, dim, m.NE);
   const auto d_fds = Reshape(m.fds.Read(), NQ, m.NVF, dim - 1);
   const auto d_XF = Reshape(m.XF.Read(), m.NVF, dim, m.NF);
   const real_t *d_qw = qw.Read();
   const real_t *d_q = Q ? qv.Read() : nullptr;
   const int *d_f = item_face.Read();
   const int *d_el = item_el.Read(), *d_sl = item_slot.Read();
   real_t *d_w = w.Write();
   const int nq = NQ, nv = m.NV, nvf = m.NVF, d = dim;
   const real_t beta = dif.GetBeta();
   const bool rl = has_rule, wx = has_rule && rule.coord >= 0;

   mfem::forall(NQ * NI, [=] MFEM_HOST_DEVICE (int qk)
   {
      const int q = qk % nq, k = qk / nq;
      const int e = d_el[k], s = d_sl[k];
      real_t n[3];
      HDGFaceNormal(q, d_f[k], nvf, d, d_fds, d_XF, n);

      const real_t ipw = d_qw[q];
      if (rl)
      {
         // Vector::Norml2()'s scaled sum, in order.
         real_t scale = 0.0, ssq = 0.0;
         for (int i = 0; i < d; i++)
         {
            const real_t a = fabs(n[i]);
            if (a > 0.0)
            {
               if (scale <= a)
               {
                  const real_t r = scale / a;
                  ssq = ssq * (r * r) + 1.0;
                  scale = a;
               }
               else
               {
                  const real_t r = a / scale;
                  ssq += r * r;
               }
            }
         }
         const real_t face_w = ipw * (scale * sqrt(ssq));
         real_t x[3] = { 0.0, 0.0, 0.0 };
         if (wx)
         {
            // The physical point on the element, as
            // IsoparametricTransformation::Transform() takes it: what a
            // stabilization's Eval() reads off the transformation.
            for (int i = 0; i < d; i++)
            {
               real_t sum = 0.0;
               for (int v = 0; v < nv; v++) { sum += d_X(v, i, e) * d_sh(q, v, s); }
               x[i] = sum;
            }
         }
         d_w[qk] = face_w * rule.Eval(x);
      }
      else
      {
         // The element's Jacobian at the face point, and its weight.
         real_t J[9];
         for (int j = 0; j < d; j++)
            for (int i = 0; i < d; i++)
            {
               real_t sum = 0.0;
               for (int v = 0; v < nv; v++) { sum += d_X(v, i, e) * d_ds(q, v, j, s); }
               J[i + d * j] = sum;
            }
         const real_t wn = ipw / HDGJacobianWeight(J, d, d);
         const real_t c = d_q ? (wn * d_q[qk]) : wn;
         real_t wq = 0.0;
         for (int i = 0; i < d; i++) { wq += (c * n[i]) * n[i]; }
         d_w[qk] = wq * beta;
      }
   });
}

} // namespace

bool HDGFaceWeightsOnDevice(const HDGDiffusionIntegrator &dif,
                            const FiniteElementSpace &el_fes)
{
   const Mesh *mesh = el_fes.GetMesh();
   if (!mesh || !HDGFaceGeometryOnDevice(*mesh)) { return false; }
   if (dif.GetVelocity() || dif.GetMatrixCoefficient()) { return false; }
   if (const HDGStabilization *st = dif.GetStabilization())
   {
      HDGStabilizationRule r;
      return st->IsConstant() && st->GetRule(r)
             && r.coord < mesh->SpaceDimension();
   }
   Coefficient *Q = dif.GetCoefficient();
   return !Q || HDGCoefficientOnDevice(Q, *mesh);
}

bool HDGFaceSpacesCanBatch(const FiniteElementSpace &tr_fes,
                           const FiniteElementSpace &el_fes)
{
   const Mesh *mesh = el_fes.GetMesh();
   if (!mesh || mesh->GetNE() == 0) { return false; }
   if (mesh->Nonconforming()) { return false; }
   // ONE vector dimension across the two spaces, and it may be > 1.
   //
   // The face constraint of a system is block diagonal in the equation index
   // -- equation k's trace multiplies equation k's field and nothing else --
   // so a trace space of a different vector dimension from the field space is
   // a different operator shape, not a harder case of this one. No caller in
   // the tree has one: navierstokes, examples/hdg/ex17 and ex21 and
   // tests/unit/fem/test_darcy_system.cpp all carry vdim on both.
   if (tr_fes.GetVDim() != el_fes.GetVDim()) { return false; }
   if (el_fes.GetVDim() < 1) { return false; }
   if (!dynamic_cast<const DG_Interface_FECollection*>(tr_fes.FEColl()))
   { return false; }

   // One element geometry and one order, so every face matrix is the same
   // size and the reference trace shapes are one table -- and one rule.
   return HDGOneRule(el_fes);
}

bool HDGDiffusionFaceMatricesCanBatch(const FiniteElementSpace &tr_fes,
                                      const FiniteElementSpace &el_fes)
{
   // The two routines this gates -- HDGDiffusionFaceMatricesBatched() and
   // HDGDiffusionFaceScatterBatched() -- take a bare Coefficient and a beta
   // rather than an integrator, so they have no way to be told what each
   // equation's coefficient is and stay SCALAR. The vdim > 1 generalisation
   // lives on HDGFaceScatterCanBatch()/HDGFaceScatterBatched(), which take
   // the integrators themselves.
   return HDGFaceSpacesCanBatch(tr_fes, el_fes) && el_fes.GetVDim() == 1;
}

void HDGDiffusionFaceMatricesBatched(const FiniteElementSpace &tr_fes,
                                     const FiniteElementSpace &el_fes,
                                     Coefficient *Q, real_t beta,
                                     const HDGStabilization *stab,
                                     DenseTensor &elmats)
{
   MFEM_VERIFY(HDGDiffusionFaceMatricesCanBatch(tr_fes, el_fes),
               "the spaces do not admit the batched face assembly");
   MFEM_VERIFY(!stab || stab->IsConstant(),
               "a state dependent stabilization makes the face term nonlinear");

   Mesh *mesh = el_fes.GetMesh();
   const int nfaces = mesh->GetNumFaces();

   Array<int> flist;
   for (int f = 0; f < nfaces; f++)
   {
      if (mesh->FaceIsInterior(f)) { flist.Append(f); }
   }
   const int NF = flist.Size();

   const FiniteElement *tr_fe0 = tr_fes.GetFaceElement(flist.Size() ? flist[0] :
                                                       0);
   const int TRD = tr_fe0->GetDof();
   const int ND = el_fes.GetFE(0)->GetDof();
   const int SZ = 2 * ND + TRD;

   elmats.SetSize(SZ, SZ, NF);
   // Without this the kernel below runs on the HOST whatever the Device is
   // configured as -- Read() hands back a host pointer and mfem::forall
   // degrades to a plain loop. It is the same trap as the raw-pointer
   // DenseTensor in DarcyHybridization::InvertA(), measured there and walked
   // into again here: the first version of this kernel was silently host-only
   // and 1.5x SLOWER under -d cuda than under -d cpu.
   elmats.GetMemory().UseDevice(true);
   if (NF == 0) { return; }

   // The rule the per-face path would pick, so the two agree by construction.
   const int order = 2 * std::max(el_fes.GetFE(0)->GetOrder(),
                                  tr_fe0->GetOrder());
   FaceElementTransformations *ftr0 = mesh->GetInteriorFaceTransformations(
                                         flist[0]);
   MFEM_VERIFY(ftr0, "no interior face transformation");
   const IntegrationRule &ir = IntRules.Get(ftr0->GetGeometryType(), order);
   const int NQ = ir.GetNPoints();

   // ---- host precompute: two weights and the shapes per quadrature point ----
   Vector w1(NQ * NF), w2(NQ * NF);
   Vector sh1(NQ * ND * NF), sh2(NQ * ND * NF);
   Vector trs(NQ * TRD);
   w1.UseDevice(true);
   w2.UseDevice(true);
   sh1.UseDevice(true);
   sh2.UseDevice(true);
   trs.UseDevice(true);

   {
      Vector t(TRD);
      for (int q = 0; q < NQ; q++)
      {
         tr_fe0->CalcShape(ir.IntPoint(q), t);
         for (int i = 0; i < TRD; i++) { trs(q + i * NQ) = t(i); }
      }
   }

   {
      const int dim = mesh->Dimension();
      Vector nor(dim), s1(ND), s2(ND);
      real_t *pw1 = w1.HostWrite(), *pw2 = w2.HostWrite();
      real_t *ps1 = sh1.HostWrite(), *ps2 = sh2.HostWrite();

      for (int fi = 0; fi < NF; fi++)
      {
         FaceElementTransformations *ftr =
            mesh->GetInteriorFaceTransformations(flist[fi]);
         const FiniteElement &e1 = *el_fes.GetFE(ftr->Elem1No);
         const FiniteElement &e2 = *el_fes.GetFE(ftr->Elem2No);

         for (int q = 0; q < NQ; q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            ftr->SetAllIntPoints(&ip);
            const IntegrationPoint &eip1 = ftr->GetElement1IntPoint();
            const IntegrationPoint &eip2 = ftr->GetElement2IntPoint();

            if (dim == 1) { nor(0) = 2 * eip1.x - 1.0; }
            else { CalcOrtho(ftr->Jacobian(), nor); }

            e1.CalcPhysShape(*ftr->Elem1, s1);
            e2.CalcPhysShape(*ftr->Elem2, s2);

            const real_t nn = nor * nor;
            const real_t face_w = ip.weight * nor.Norml2();

            real_t wn1 = ip.weight / ftr->Elem1->Weight();
            if (Q) { wn1 *= Q->Eval(*ftr->Elem1, eip1); }
            real_t wn2 = ip.weight / ftr->Elem2->Weight();
            if (Q) { wn2 *= Q->Eval(*ftr->Elem2, eip2); }

            // v is null on this path, so un == 0 and a == 0, b == beta.
            const real_t wq1 = wn1 * nn, wq2 = wn2 * nn;
            const real_t v1 = stab
                              ? face_w * stab->Eval((face_w != 0.) ? (wq1 * beta / face_w) : 0.,
                                                    0., 0., 0., *ftr->Elem1)
                              : wq1 * beta;
            const real_t v2 = stab
                              ? face_w * stab->Eval((face_w != 0.) ? (wq2 * beta / face_w) : 0.,
                                                    0., 0., 0., *ftr->Elem2)
                              : wq2 * beta;
            pw1[q + fi * NQ] = v1;
            pw2[q + fi * NQ] = v2;
            for (int i = 0; i < ND; i++)
            {
               ps1[q + NQ * (i + ND * fi)] = s1(i);
               ps2[q + NQ * (i + ND * fi)] = s2(i);
            }
         }
      }
   }

   // ---- the kernel ----
   // Measured, 2-D quads, steady state (first call is 100x on either backend,
   // allocation and first touch -- do not time one call):
   //
   //   n=64 order 3   precompute   kernel   copy-back
   //     -d cpu         16.8 ms    26.3 ms     0.0 ms
   //     -d cuda        16.9 ms    32.7 ms    10.9 ms
   //
   // THE COPY-BACK IS THE FINDING, not the kernel time. This routine's output
   // is consumed by host code (ComputeAndAssemblePotFaceMatrix splitting it
   // into E/G/H/D), so every face matrix has to come back -- at n=96 order 3
   // that is 193 MB and 183 ms against an 85 ms kernel. An isolated device
   // kernel whose consumer is on the host pays more in transfer than it saves
   // in arithmetic, which is the plan's own gate argument arriving from the
   // integrator side. The kernel is worth having; it is not worth switching on
   // until the consumers are device-resident too.
   //
   // The kernel share is what makes it worth having at all: 47% of this
   // routine at order 1, 72% at order 2, 83% at order 3, because the outer
   // products are O(NQ*SZ^2) against O(NQ*ND) for the shapes.
   //
   // GPU timings here are from a consumer card shared with a desktop under
   // WSL2 and are not a verdict on the approach.
   // ---- one face per thread ----
   const auto d_w1 = Reshape(w1.Read(), NQ, NF);
   const auto d_w2 = Reshape(w2.Read(), NQ, NF);
   const auto d_s1 = Reshape(sh1.Read(), NQ, ND, NF);
   const auto d_s2 = Reshape(sh2.Read(), NQ, ND, NF);
   const auto d_tr = Reshape(trs.Read(), NQ, TRD);
   auto d_M = Reshape(elmats.Write(), SZ, SZ, NF);

   const int nd = ND, trd = TRD, nq = NQ, sz = SZ;

   mfem::forall(NF, [=] MFEM_HOST_DEVICE (int f)
   {
      for (int j = 0; j < sz; j++)
         for (int i = 0; i < sz; i++)
         {
            d_M(i, j, f) = 0.0;
         }

      for (int q = 0; q < nq; q++)
      {
         const real_t a1 = d_w1(q, f), a2 = d_w2(q, f);

         // D blocks, both sides
         for (int i = 0; i < nd; i++)
         {
            const real_t b1 = a1 * d_s1(q, i, f);
            const real_t b2 = a2 * d_s2(q, i, f);
            for (int j = 0; j < nd; j++)
            {
               d_M(i, j, f)           += b1 * d_s1(q, j, f);
               d_M(nd + i, nd + j, f) += b2 * d_s2(q, j, f);
            }
            // E blocks, and G = -E^T
            for (int j = 0; j < trd; j++)
            {
               const real_t e1 = b1 * d_tr(q, j);
               const real_t e2 = b2 * d_tr(q, j);
               d_M(i, 2 * nd + j, f)      -= e1;
               d_M(nd + i, 2 * nd + j, f) -= e2;
               d_M(2 * nd + j, i, f)      += e1;
               d_M(2 * nd + j, nd + i, f) += e2;
            }
         }

         // the face block, once, with both sides' weights
         const real_t aa = a1 + a2;
         for (int i = 0; i < trd; i++)
         {
            const real_t t = aa * d_tr(q, i);
            for (int j = 0; j < trd; j++)
            {
               d_M(2 * nd + i, 2 * nd + j, f) -= t * d_tr(q, j);
            }
         }
      }
   });

}


namespace
{

/// Which of the three face terms the batched kernel implements @a bfi is.
enum class HDGBatchKind { Diffusion, ConvCentered, ConvUpwinded, Unsupported };

HDGBatchKind HDGBatchKindOf(const BilinearFormIntegrator *bfi)
{
   // Order matters only in that the convection pair are siblings; a
   // dynamic_cast to either cannot match the other.
   if (dynamic_cast<const HDGDiffusionIntegrator*>(bfi))
   { return HDGBatchKind::Diffusion; }
   if (dynamic_cast<const HDGConvectionCenteredIntegrator*>(bfi))
   { return HDGBatchKind::ConvCentered; }
   if (dynamic_cast<const HDGConvectionUpwindedIntegrator*>(bfi))
   { return HDGBatchKind::ConvUpwinded; }
   return HDGBatchKind::Unsupported;
}

/** @brief The rule @a bfi integrates this face at, asked of the integrator
    rather than reconstructed. A rule the caller set explicitly wins, exactly
    as it does in AssembleHDGFaceMatrix(). */
const IntegrationRule *HDGBatchRule(const BilinearFormIntegrator *bfi,
                                    const FiniteElement &tr_fe,
                                    const FiniteElement &e1,
                                    const FiniteElement &e2,
                                    FaceElementTransformations &ftr)
{
   if (const IntegrationRule *ir = bfi->GetIntRule()) { return ir; }
   if (auto *d = dynamic_cast<const HDGDiffusionIntegrator*>(bfi))
   { return &d->GetHDGFaceIntRule(tr_fe, e1, e2, ftr); }
   if (auto *c = dynamic_cast<const HDGConvectionCenteredIntegrator*>(bfi))
   { return &c->GetHDGFaceIntRule(tr_fe, e1, e2, ftr); }
   if (auto *u = dynamic_cast<const HDGConvectionUpwindedIntegrator*>(bfi))
   { return &u->GetHDGFaceIntRule(tr_fe, e1, e2, ftr); }
   return NULL;
}

/** @brief The rule @a bfi integrates ONE SIDE of this face at, asked of the
    integrator exactly as HDGBatchRule() asks for the two-sided one.

    Not the same rule, and the difference is not cosmetic: the one-sided
    routines take their OrderW() from the side being integrated and their max
    over one element, so on a mesh whose neighbours differ in order the two
    expressions part company. The shared-face pass is one-sided from this rank
    and must sample where ComputeAndAssemblePotFaceMatrix()'s own shared
    branch does. */
const IntegrationRule *HDGBatchSideRule(const BilinearFormIntegrator *bfi,
                                        int side,
                                        const FiniteElement &tr_fe,
                                        const FiniteElement &el,
                                        FaceElementTransformations &ftr)
{
   if (const IntegrationRule *ir = bfi->GetIntRule()) { return ir; }
   if (auto *d = dynamic_cast<const HDGDiffusionIntegrator*>(bfi))
   { return &d->GetHDGFaceIntRule(tr_fe, el, ftr); }
   if (auto *c = dynamic_cast<const HDGConvectionCenteredIntegrator*>(bfi))
   { return &c->GetHDGFaceIntRule(side, tr_fe, el, ftr); }
   if (auto *u = dynamic_cast<const HDGConvectionUpwindedIntegrator*>(bfi))
   { return &u->GetHDGFaceIntRule(side, tr_fe, el, ftr); }
   return NULL;
}

/** @brief One diagonal block of a face constraint: which equation it belongs
    to and the SCALAR integrator that supplies it. */
struct HDGFaceBlock
{
   int eq;                             ///< the equation index, 0 at vdim == 1
   const BilinearFormIntegrator *bfi;  ///< the scalar integrator for it
};

/** @brief Resolve @a bfi into one HDGFaceBlock per equation, or refuse.

    At vdim == 1 that is the integrator itself, on equation 0. At vdim > 1 the
    only shape any caller in the tree installs is a
    VectorBlockDiagonalIntegrator over the scalar HDG face families, in both
    of its forms: one integrator per equation (navierstokes,
    test_darcy_system) or one repeated (examples/hdg/ex17, ex21). Both are
    admitted; anything else is refused.

    THE NULL BLOCK IS REFUSED AND THE REASON IS NOT FASTIDIOUSNESS.
    VectorBlockDiagonalIntegrator::AssembleMat() accumulates its row and
    column extents only for non-null blocks, so a null one SHRINKS the
    element matrix rather than zeroing that block -- the per-face route then
    reads a matrix one block short against offsets sized for vdim of them,
    and MFEM_ASSERT is compiled out in a release build. The batched route
    would be wrong in a different way from the per-face route, which is worse
    than either being wrong. A caller wanting a zero block passes a
    zero-valued integrator, which is what navierstokes' pressure row does. */
bool HDGFaceBlocksOf(const BilinearFormIntegrator *bfi, int vdim,
                     std::vector<HDGFaceBlock> &out)
{
   out.clear();
   if (vdim == 1 && !dynamic_cast<const VectorBlockDiagonalIntegrator*>(bfi))
   {
      if (HDGBatchKindOf(bfi) == HDGBatchKind::Unsupported) { return false; }
      out.push_back({0, bfi});
      return true;
   }

   auto *vb = dynamic_cast<const VectorBlockDiagonalIntegrator*>(bfi);
   if (!vb) { return false; }
   if (vb->GetNumBlocks() != vdim) { return false; }

   const int ni = vb->GetNumIntegrators();
   const bool repeated = (ni < vdim);
   if (repeated && ni != 1) { return false; }

   for (int k = 0; k < vdim; k++)
   {
      const BilinearFormIntegrator *b = vb->GetIntegrator(repeated ? 0 : k);
      if (!b) { return false; }
      if (HDGBatchKindOf(b) == HDGBatchKind::Unsupported) { return false; }
      out.push_back({k, b});
   }
   return true;
}

/// The seven per-(point, face) weight streams of HDGFaceScatterBatched().
struct HDGFaceWeights
{
   Vector d1, d2, e1, e2, g1, g2, h;

   void Init(int n)
   {
      Vector *all[7] = { &d1, &d2, &e1, &e2, &g1, &g2, &h };
      for (Vector *v : all) { v->SetSize(n); *v = 0.; v->UseDevice(true); }
   }
};

} // namespace

namespace
{

/** @brief Everything the face weight loop used to ask a
    FaceElementTransformations for, precomputed per FACE or per CODE.

    **The whole point is that the loop no longer builds a
    FaceElementTransformations per face**, which is where that loop's time
    was: ablation inside it found the shape evaluation, the coefficient
    evaluation, the per-point transformation update and the shape-table
    stores ALL innocent, and the cost survived deleting the entire quadrature
    loop -- so it was the per-face frame, not the integrand. The ablation is
    the attribution; the wall-clock pairs behind it are on
    HDGFaceScatterBatched() in the header, which is the routine this serves.

    Two facts make this exact, and both were measured rather than reasoned:

    - **The element reference points at a face's quadrature points depend only
      on (local face id, orientation, q).** So one table per CODE serves the
      mesh: 4 distinct codes on 2-D quads and 6 on 3-D hexes, against one per
      face. Checked by observing every face's own CalcPhysShape and requiring
      faces that share a code to agree -- 0.0 at every order and dimension
      tried.
    - **The normal comes from the FACE's own transformation**, which is what
      a FaceElementTransformations' Jacobian() already is -- so
      `Mesh::GetFaceTransformation()` gives the identical `CalcOrtho()` input
      with one transformation object instead of three plus the two element
      frames.

    **This deliberately does NOT use FaceGeometricFactors, and an earlier
    version did.** `Mesh::GetFaceGeometricFactors()` SEGFAULTS on a simplex
    mesh (inside ConformingFaceRestriction, via GetFaceRestriction), which
    would have made this whole path tensor-only, and at the time the weight
    loop could not be a kernel anyway: its weight went through host virtuals.
    **A restriction that buys nothing is not a trade-off, it is a defect**,
    and the version that had it was wrong. The weights did become a kernel
    since (HDGFaceWeightsDevice()), and still not through the factors: the
    normal comes from each face's own point matrix and the element weight from
    the element's, at this table's reference points, which serves simplices
    and gives the per-face routines' own bits on the host.

    The shape tables need `map_type == VALUE`, where CalcPhysShape reduces to
    CalcShape; an INTEGRAL space divides by the transformation weight and is
    refused -- that is the one refusal left, and it is about the SPACE, not
    the mesh. */
struct HDGFaceTables
{
   int NQ = 0, ND = 0, dim = 0;
   bool have_shape = false;       ///< false unless map_type == VALUE

   Array<int> el1, el2;           ///< (NF) the two elements of each face
   Array<int> c1, c2;             ///< (NF) each side's (local face, orient) slot
   /// (NQ, ND, ncode) element shape at the face points, one table per slot
   Vector sref;
   /// (ncode) the element reference points, one rule per slot
   std::vector<IntegrationRule> eip;
};

/// Build @a t for @a face_list at @a ir. Never fails: the two `have_*` flags
/// say which halves are usable and the caller falls back for the rest.
void HDGFaceBuildTables(Mesh &mesh, const FiniteElementSpace &el_fes,
                        const Array<int> &face_list,
                        const IntegrationRule &ir, HDGFaceTables &t)
{
   const int NF = face_list.Size();
   t.NQ = ir.GetNPoints();
   t.ND = el_fes.GetFE(0)->GetDof();
   t.dim = mesh.Dimension();
   t.el1.SetSize(NF);
   t.el2.SetSize(NF);
   t.c1.SetSize(NF);
   t.c2.SetSize(NF);

   // The slot table, keyed on the (local face id, orientation) info word each
   // side carries. Populated from the FIRST face exhibiting a slot, which is
   // what makes this O(ncode) transformations instead of O(NF).
   std::map<int,int> slot_of;
   std::vector<int> slot_info;
   auto slot = [&](int info) -> int
   {
      auto it = slot_of.find(info);
      if (it != slot_of.end()) { return it->second; }
      const int s = (int)slot_info.size();
      slot_of[info] = s;
      slot_info.push_back(info);
      return s;
   };

   for (int fi = 0; fi < NF; fi++)
   {
      const int f = face_list[fi];
      const Mesh::FaceInformation fin = mesh.GetFaceInformation(f);
      t.el1[fi] = fin.element[0].index;
      t.el2[fi] = fin.element[1].index;
      t.c1[fi] = slot(fin.element[0].local_face_id * 64
                      + fin.element[0].orientation);
      t.c2[fi] = slot(fin.element[1].local_face_id * 64
                      + fin.element[1].orientation);
   }

   // The shape tables, one transformation per SLOT. Only valid where
   // CalcPhysShape is CalcShape.
   const int ncode = (int)slot_info.size();
   t.have_shape = (el_fes.GetFE(0)->GetMapType() == FiniteElement::VALUE);
   if (t.have_shape)
   {
      t.eip.assign(ncode, IntegrationRule(t.NQ));
      t.sref.SetSize(t.NQ * t.ND * ncode);
      t.sref.UseDevice(true);
      t.sref = 0.0;
      std::vector<bool> done(ncode, false);
      Vector sh(t.ND);
      real_t *ps = t.sref.HostWrite();
      for (int fi = 0; fi < NF; fi++)
      {
         for (int side = 0; side < 2; side++)
         {
            const int c = side ? t.c2[fi] : t.c1[fi];
            if (done[c]) { continue; }
            FaceElementTransformations *ftr =
               mesh.GetInteriorFaceTransformations(face_list[fi]);
            if (!ftr) { t.have_shape = false; return; }
            const FiniteElement &fe =
               *el_fes.GetFE(side ? ftr->Elem2No : ftr->Elem1No);
            for (int q = 0; q < t.NQ; q++)
            {
               ftr->SetAllIntPoints(&ir.IntPoint(q));
               const IntegrationPoint &e =
                  side ? ftr->GetElement2IntPoint() : ftr->GetElement1IntPoint();
               t.eip[c].IntPoint(q) = e;
               fe.CalcPhysShape(side ? *ftr->Elem2 : *ftr->Elem1, sh);
               for (int i = 0; i < t.ND; i++)
               { ps[q + t.NQ*(i + t.ND*c)] = sh(i); }
            }
            done[c] = true;
         }
      }
   }
}

/** @brief The element side of a set of one-sided face items, tabulated.

    Each item (face, side) is reduced to a SLOT, the (local face id,
    orientation) of that side's element, and the element's shapes at the face
    quadrature points depend on the slot alone -- the fact HDGFaceTables rests
    on, which it measured. Unlike HDGFaceTables this takes boundary faces
    (side 0) as well as either side of an interior one. */
struct HDGSideTables
{
   int NQ = 0, ND = 0, nslot = 0;
   Array<int> slot;                   ///< (NI) each item's slot
   Array<int> el;                     ///< (NI) each item's element
   Vector sref;                       ///< (NQ, ND, nslot) element shapes
   std::vector<IntegrationRule> eip;  ///< (nslot) element reference points
};

void HDGBuildSideTables(Mesh &mesh, const FiniteElementSpace &el_fes,
                        const Array<int> &faces, const Array<int> &sides,
                        const IntegrationRule &ir, HDGSideTables &t)
{
   const int NI = faces.Size();
   t.NQ = ir.GetNPoints();
   t.ND = el_fes.GetFE(0)->GetDof();
   t.slot.SetSize(NI);
   t.el.SetSize(NI);

   std::map<int, int> slot_of;
   std::vector<int> rep;            // the first item showing each slot
   for (int k = 0; k < NI; k++)
   {
      const Mesh::FaceInformation fin = mesh.GetFaceInformation(faces[k]);
      const auto &e = fin.element[sides[k]];
      t.el[k] = e.index;
      const int key = e.local_face_id * 64 + e.orientation;
      auto it = slot_of.find(key);
      if (it == slot_of.end())
      {
         it = slot_of.emplace(key, (int)rep.size()).first;
         rep.push_back(k);
      }
      t.slot[k] = it->second;
   }

   t.nslot = (int)rep.size();
   t.eip.assign(t.nslot, IntegrationRule(t.NQ));
   t.sref.SetSize(t.NQ * t.ND * t.nslot);
   t.sref.UseDevice(true);
   real_t *ps = t.sref.HostWrite();
   Vector sh(t.ND);
   for (int s = 0; s < t.nslot; s++)
   {
      const int k = rep[s];
      FaceElementTransformations *ftr =
         mesh.GetFaceElementTransformations(faces[k]);
      const int side = sides[k];
      MFEM_VERIFY(side == 0 || ftr->Elem2No >= 0,
                  "side 2 of face " << faces[k] << ", which has none");
      const FiniteElement &fe =
         *el_fes.GetFE(side ? ftr->Elem2No : ftr->Elem1No);
      for (int q = 0; q < t.NQ; q++)
      {
         ftr->SetAllIntPoints(&ir.IntPoint(q));
         const IntegrationPoint &e =
            side ? ftr->GetElement2IntPoint() : ftr->GetElement1IntPoint();
         t.eip[s].IntPoint(q) = e;
         // VALUE-mapped, so CalcPhysShape() is CalcShape() at the point.
         fe.CalcShape(e, sh);
         for (int i = 0; i < t.ND; i++) { ps[q + t.NQ * (i + t.ND * s)] = sh(i); }
      }
   }
}

} // namespace

bool HDGFaceScatterCanBatch(const FiniteElementSpace &tr_fes,
                            const FiniteElementSpace &el_fes,
                            const Array<BilinearFormIntegrator*> &integs,
                            const Array<int> &face_list)
{
   if (integs.Size() == 0) { return false; }
   if (!HDGFaceSpacesCanBatch(tr_fes, el_fes)) { return false; }
   if (face_list.Size() == 0) { return true; }

   Mesh *mesh = el_fes.GetMesh();
   const int VD = el_fes.GetVDim();
   std::vector<HDGFaceBlock> blocks;
   for (BilinearFormIntegrator *bfi : integs)
   {
      // One SCALAR integrator per equation; see HDGFaceBlocksOf(). At
      // vdim == 1 this is the integrator itself and the loop runs once.
      if (!HDGFaceBlocksOf(bfi, VD, blocks)) { return false; }
      for (const HDGFaceBlock &blk : blocks)
      {
         // A state dependent stabilization is not a bilinear form at all; the
         // per-face routine refuses it too.
         if (auto *d = dynamic_cast<const HDGDiffusionIntegrator*>(blk.bfi))
         {
            const HDGStabilization *st = d->GetStabilization();
            if (st && !st->IsConstant()) { return false; }
         }

         // ONE rule for the whole face list, PER BLOCK -- each equation's
         // integrator is asked separately, because that is how the per-face
         // route calls them: VectorBlockDiagonalIntegrator invokes each
         // block's AssembleHDGFaceMatrix() in turn, so each picks its own
         // rule. The convection forms take ElementTransformation::OrderW()
         // into theirs, so a mesh whose elements do not all report the same
         // one would want a different number of points on different faces --
         // which the kernel, sampling every face at one rule, cannot do.
         // Asked of the faces because that is where the answer is;
         // HDGFaceSpacesCanBatch() checks the geometry and the dof counts and
         // could not see this.
         //
         // Asked of the FIRST face: HDGFaceSpacesCanBatch() has established
         // what makes it every face's rule (see HDGOneRule()). The faces are
         // checked to be interior without building their transformations.
         FaceElementTransformations *ftr =
            mesh->GetInteriorFaceTransformations(face_list[0]);
         if (!ftr) { return false; }
         if (!HDGBatchRule(blk.bfi, *tr_fes.GetFaceElement(face_list[0]),
                           *el_fes.GetFE(ftr->Elem1No),
                           *el_fes.GetFE(ftr->Elem2No), *ftr))
         { return false; }
      }
   }
   for (int fi = 0; fi < face_list.Size(); fi++)
   {
      if (!mesh->FaceIsInterior(face_list[fi])) { return false; }
   }
   return true;
}

void HDGFaceScatterBatched(const FiniteElementSpace &tr_fes,
                           const FiniteElementSpace &el_fes,
                           const Array<BilinearFormIntegrator*> &integs,
                           const Array<int> &face_list,
                           const Array<int> &E_offsets,
                           const Array<int> &H_offsets,
                           const Array<int> &Df_offsets,
                           Vector &E_data, Vector &G_data,
                           Vector &H_data, Vector &Df_data)
{
   MFEM_VERIFY(HDGFaceScatterCanBatch(tr_fes, el_fes, integs, face_list),
               "these integrators do not admit the batched face assembly");

   Mesh *mesh = el_fes.GetMesh();
   const int NF = face_list.Size();
   if (NF == 0) { return; }

   const FiniteElement *tr_fe0 = tr_fes.GetFaceElement(face_list[0]);
   const int TRD = tr_fe0->GetDof();
   const int ND = el_fes.GetFE(0)->GetDof();
   // The vector dimension, and the block dimensions it multiplies.
   //
   // E, G, H and D are stored at the sizes DarcyHybridization's offsets carry,
   // which are the SPACE's: Df_offsets[el] holds (ND*VD)^2, E_offsets[f] holds
   // (TRD*VD)*(ND*VD) per side and H_offsets[f] holds (TRD*VD)^2. Within a
   // block the layout is field-outermost -- equation k's dofs occupy the
   // contiguous run [k*ND, (k+1)*ND) -- because that is what
   // VectorBlockDiagonalIntegrator::AssembleMat() produces and what
   // ComputeAndAssemblePotFaceMatrix() then CopyMN()s out of, and it is
   // Ordering::byNODES's layout as well.
   const int VD = el_fes.GetVDim();
   const int NDT = ND * VD, TRDT = TRD * VD;

   // Face-indexed offsets, shared by every pass. The two elements are the
   // mesh's face table's, which is what a FaceElementTransformations reports
   // for Elem1No and Elem2No -- without building one per face, which was a
   // measurable share of this routine's host time on meq's mesh.
   Array<int> eo(NF), ho(NF), d1o(NF), d2o(NF);
   for (int fi = 0; fi < NF; fi++)
   {
      const int face = face_list[fi];
      int el1, el2;
      mesh->GetFaceElements(face, &el1, &el2);
      eo[fi] = E_offsets[face];
      ho[fi] = H_offsets[face];
      d1o[fi] = Df_offsets[el1];
      d2o[fi] = Df_offsets[el2];
   }

   {
      // E, G and H belong to a face alone, so they are zeroed once here and
      // every integrator's pass then accumulates. D is NOT zeroed: it
      // accumulates across the faces of an element and across the potential
      // mass form as well, so whoever owns it zeroes it.
      const int *d_eo = eo.Read(), *d_ho = ho.Read();
      real_t *d_E = E_data.ReadWrite();
      real_t *d_G = G_data.ReadWrite();
      real_t *d_H = H_data.ReadWrite();
      // The WHOLE block, every equation, including the off-diagonal ones the
      // passes below never touch. That is what makes zero-then-accumulate
      // reproduce the per-face route: it CopyMN()s the whole (NDT x TRDT)
      // sub-block out of an element matrix whose off-diagonal equation
      // entries are exactly zero.
      const int nd = NDT, trd = TRDT;
      mfem::forall(NF, [=] MFEM_HOST_DEVICE (int f)
      {
         const int e0 = d_eo[f], h0 = d_ho[f];
         const int e2off = e0 + trd * nd;      // side 2 follows side 1
         for (int j = 0; j < trd; j++)
            for (int i = 0; i < nd; i++)
            {
               d_E[e0 + i + nd * j] = 0.0;
               d_E[e2off + i + nd * j] = 0.0;
               d_G[e0 + j + trd * i] = 0.0;
               d_G[e2off + j + trd * i] = 0.0;
            }
         for (int j = 0; j < trd; j++)
            for (int i = 0; i < trd; i++)
            {
               d_H[h0 + i + trd * j] = 0.0;
            }
      });
   }

   const int dim = mesh->Dimension();

   // ONE PASS PER (integrator, equation) PAIR, rather than one per integrator
   // with an equation loop inside the kernel. Two reasons, and the first is
   // correctness: each equation's integrator picks its OWN integration rule,
   // exactly as it does on the per-face route, so a pass is the largest unit
   // that can share a rule. The second is that it keeps the kernel index
   // arithmetic at one thread per FACE with plain inner loops -- adding an
   // equation component to a flat index decomposition is the shape that cost
   // 0.37 s of a 0.95 s loop in ComputeElementsHBatched().
   std::vector<HDGFaceBlock> blocks;
   for (BilinearFormIntegrator *bfi_v : integs)
   {
      MFEM_VERIFY(HDGFaceBlocksOf(bfi_v, VD, blocks),
                  "integrator does not resolve into scalar diagonal blocks");
      for (const HDGFaceBlock &blk : blocks)
      {
         const BilinearFormIntegrator *bfi = blk.bfi;
         const int EQ = blk.eq;
         const HDGBatchKind kind = HDGBatchKindOf(bfi);
         auto *dif = dynamic_cast<const HDGDiffusionIntegrator*>(bfi);
         auto *tr_bfi = dynamic_cast<const DGTraceIntegrator*>(bfi);

         FaceElementTransformations *ftr0 =
            mesh->GetInteriorFaceTransformations(face_list[0]);
         const IntegrationRule &ir =
            *HDGBatchRule(bfi, *tr_fe0, *el_fes.GetFE(ftr0->Elem1No),
                          *el_fes.GetFE(ftr0->Elem2No), *ftr0);
         const int NQ = ir.GetNPoints();

         HDGFaceWeights w;
         w.Init(NQ * NF);
         Vector trs(NQ * TRD);
         trs.UseDevice(true);

         {
            Vector t(TRD);
            for (int q = 0; q < NQ; q++)
            {
               tr_fe0->CalcShape(ir.IntPoint(q), t);
               for (int i = 0; i < TRD; i++) { trs(q + i * NQ) = t(i); }
            }
         }

         // The tables that replace a FaceElementTransformations per face; see
         // HDGFaceTables. Built per PASS because each pass has its own rule,
         // and cheap because it costs one transformation per (local face,
         // orientation) slot rather than one per face.
         HDGFaceTables tab;
         HDGFaceBuildTables(*mesh, el_fes, face_list, ir, tab);
         // One refusal now, and it is about the SPACE rather than the mesh:
         // the shape tables need map_type == VALUE. An earlier version ANDed
         // this with a tensor-mesh test it had no need of -- see HDGFaceTables
         // -- which silently sent every simplex mesh down the slow path.
         const bool fast = tab.have_shape;
         //
         // **Measured, A/B in one build against an env gate that is now
         // removed**, Assemble() on the host, n=64:
         //
         //     order  1     2     2 (n=128)  3
         //     quads  1.42x 1.06x 1.13x      1.12x
         //     tris   1.34x 1.29x --         1.28x
         //
         // Triangles gain MORE than quads at order 2 and 3, which is worth
         // knowing because the first version of this used
         // FaceGeometricFactors and so refused simplices outright -- it would
         // have delivered nothing on the meshes that benefit most.
         //
         // The first A/B run of this was against a STALE probe binary whose
         // makefile rule depended on the .cpp and not on libmfem.a; it
         // reported 0.95x at order 3 and the trend backwards. Relink before
         // believing a probe.

         // The element shapes, read by SLOT: the tables' own on the fast path,
         // and otherwise one slot per (face, side), filled by the loop below.
         // The fast path used to copy its table into a per-face array on the
         // host, NQ*ND*NF reals a side, for the kernel to read back.
         Vector sh_face;
         Array<int> fs1, fs2;
         if (!fast)
         {
            sh_face.SetSize(NQ * ND * 2 * NF);
            sh_face.UseDevice(true);
            fs1.SetSize(NF);
            fs2.SetSize(NF);
            for (int fi = 0; fi < NF; fi++)
            {
               fs1[fi] = 2 * fi;
               fs2[fi] = 2 * fi + 1;
            }
         }
         const Vector &sh = fast ? tab.sref : sh_face;
         const Array<int> &sl1 = fast ? tab.c1 : fs1;
         const Array<int> &sl2 = fast ? tab.c2 : fs2;
         const int NSL = fast ? (int)tab.eip.size() : 2 * NF;

         // The weights on the DEVICE where HDGFaceWeightsOnDevice() admits the
         // integrator: two items per face, one per side, from the element maps
         // and the slot tables. Nothing below then runs per face on the host.
         const bool dev_w = fast && kind == HDGBatchKind::Diffusion
                            && HDGFaceWeightsOnDevice(*dif, el_fes);
         if (dev_w)
         {
            Array<int> item_face(2 * NF), item_el(2 * NF), item_slot(2 * NF);
            for (int fi = 0; fi < NF; fi++)
            {
               item_face[2 * fi] = item_face[2 * fi + 1] = fi;
               item_el[2 * fi] = tab.el1[fi];
               item_el[2 * fi + 1] = tab.el2[fi];
               item_slot[2 * fi] = tab.c1[fi];
               item_slot[2 * fi + 1] = tab.c2[fi];
            }
            const HDGSideMaps maps(*mesh, ir, tab.eip, face_list);
            Vector wi;
            HDGFaceWeightsDevice(*dif, maps, ir, tab.eip, item_face, item_el,
                                 item_slot, wi);
            // Side 1's weight on D, E and G of side 1, side 2's on side 2's,
            // and their sum on H, as the host loop below accumulates them.
            const auto d_wi = Reshape(wi.Read(), NQ, 2, NF);
            real_t *d_d1 = w.d1.Write(), *d_d2 = w.d2.Write();
            real_t *d_e1 = w.e1.Write(), *d_e2 = w.e2.Write();
            real_t *d_g1 = w.g1.Write(), *d_g2 = w.g2.Write();
            real_t *d_hh = w.h.Write();
            const int nq = NQ;
            mfem::forall(NQ * NF, [=] MFEM_HOST_DEVICE (int qf)
            {
               const int q = qf % nq, f = qf / nq;
               const real_t w1 = d_wi(q, 0, f), w2 = d_wi(q, 1, f);
               d_d1[qf] = w1;  d_e1[qf] = w1;  d_g1[qf] = w1;
               d_d2[qf] = w2;  d_e2[qf] = w2;  d_g2[qf] = w2;
               d_hh[qf] = w1 + w2;
            });
         }

         if (!dev_w)
         {
            Vector nor(dim), vu(dim), nh(dim), ni(dim), s1(ND), s2(ND);
            DenseMatrix mq(dim);
            // CALLER-PROVIDED and not Mesh::GetElementTransformation(i),
            // which returns a pointer to the mesh's ONE shared
            // Transformation -- two calls would alias and both sides would
            // read the second element's geometry.
            IsoparametricTransformation tr1, tr2, ftr_f;
            real_t *pd1 = w.d1.HostWrite(), *pd2 = w.d2.HostWrite();
            real_t *pe1 = w.e1.HostWrite(), *pe2 = w.e2.HostWrite();
            real_t *pg1 = w.g1.HostWrite(), *pg2 = w.g2.HostWrite();
            real_t *ph = w.h.HostWrite();
            real_t *psf = fast ? NULL : sh_face.HostWrite();

            for (int fi = 0; fi < NF; fi++)
            {
               // On the fast path NOTHING here builds a face transformation:
               // the elements come from the tables and their transformations
               // are the mesh's own, which is one object rather than three
               // plus the face geometry.
               FaceElementTransformations *ftr =
                  fast ? NULL : mesh->GetInteriorFaceTransformations(face_list[fi]);
               const int e1no = fast ? tab.el1[fi] : ftr->Elem1No;
               const int e2no = fast ? tab.el2[fi] : ftr->Elem2No;
               const FiniteElement &e1 = *el_fes.GetFE(e1no);
               const FiniteElement &e2 = *el_fes.GetFE(e2no);
               if (fast)
               {
                  mesh->GetElementTransformation(e1no, &tr1);
                  mesh->GetElementTransformation(e2no, &tr2);
                  // The face frame, which is what a FaceElementTransformations
                  // IS for the purposes of Jacobian() -- one object instead of
                  // three, and it exists for every element geometry.
                  mesh->GetFaceTransformation(face_list[fi], &ftr_f);
               }
               ElementTransformation *t1 = fast ? (ElementTransformation*)&tr1
                                           : (ElementTransformation*)ftr->Elem1;
               ElementTransformation *t2 = fast ? (ElementTransformation*)&tr2
                                           : (ElementTransformation*)ftr->Elem2;
               const int cc1 = fast ? tab.c1[fi] : 0;
               const int cc2 = fast ? tab.c2[fi] : 0;

               for (int q = 0; q < NQ; q++)
               {
                  const IntegrationPoint &ip = ir.IntPoint(q);
                  if (!fast) { ftr->SetAllIntPoints(&ip); }
                  const IntegrationPoint &eip1 =
                     fast ? tab.eip[cc1].IntPoint(q) : ftr->GetElement1IntPoint();
                  const IntegrationPoint &eip2 =
                     fast ? tab.eip[cc2].IntPoint(q) : ftr->GetElement2IntPoint();
                  if (fast)
                  {
                     // The coefficient and stabilization calls below take an
                     // ElementTransformation, and Weight() needs the point set
                     // on it. Measured free: ablating the per-point
                     // transformation update changed nothing.
                     t1->SetIntPoint(&eip1);
                     t2->SetIntPoint(&eip2);
                  }

                  if (dim == 1) { nor(0) = 2 * eip1.x - 1.0; }
                  else if (fast)
                  {
                     ftr_f.SetIntPoint(&ip);
                     CalcOrtho(ftr_f.Jacobian(), nor);
                  }
                  else { CalcOrtho(ftr->Jacobian(), nor); }

                  if (!fast)
                  {
                     e1.CalcPhysShape(*ftr->Elem1, s1);
                     e2.CalcPhysShape(*ftr->Elem2, s2);
                     for (int i = 0; i < ND; i++)
                     {
                        psf[q + NQ * (i + ND * (2 * fi))] = s1(i);
                        psf[q + NQ * (i + ND * (2 * fi + 1))] = s2(i);
                     }
                  }

                  const int o = q + fi * NQ;
                  if (kind == HDGBatchKind::Diffusion)
                  {
                     // The velocity enters only through sign(u.n) and |u.n|; see
                     // HDGDiffusionIntegrator::AssembleHDGFaceMatrix().
                     real_t un = 0.;
                     if (VectorCoefficient *v = dif->GetVelocity())
                     {
                        v->Eval(vu, *t1, eip1);
                        un = vu * nor;
                     }
                     const real_t un_raw = un;
                     real_t a, b;
                     if (un != 0.)
                     {
                        un /= std::fabs(un);
                        a = 0.5 * dif->GetAlpha() * un;
                        b = dif->GetBeta() * std::fabs(un);
                     }
                     else { a = 0.; b = dif->GetBeta(); }

                     Coefficient *Q = dif->GetCoefficient();
                     MatrixCoefficient *MQ = dif->GetMatrixCoefficient();
                     real_t wq[2];
                     ElementTransformation *el[2] = { t1, t2 };
                     const IntegrationPoint *eip[2] = { &eip1, &eip2 };
                     for (int side = 0; side < 2; side++)
                     {
                        const real_t wn = ip.weight / el[side]->Weight();
                        if (!MQ)
                        {
                           ni.Set(Q ? (wn * Q->Eval(*el[side], *eip[side])) : wn,
                                  nor);
                        }
                        else
                        {
                           nh.Set(wn, nor);
                           MQ->Eval(mq, *el[side], *eip[side]);
                           mq.MultTranspose(nh, ni);
                        }
                        wq[side] = ni * nor;
                     }

                     const real_t face_w = ip.weight * nor.Norml2();
                     const real_t w1 = dif->EvalStabilization(
                                          wq[0], b + a, un_raw, face_w, 0., 0., *t1);
                     const real_t w2 = dif->EvalStabilization(
                                          wq[1], b - a, un_raw, face_w, 0., 0., *t2);
                     pd1[o] += w1;  pe1[o] += w1;  pg1[o] += w1;
                     pd2[o] += w2;  pe2[o] += w2;  pg2[o] += w2;
                     ph[o] += w1 + w2;
                  }
                  else
                  {
                     tr_bfi->GetVelocity()->Eval(vu, *t1, eip1);
                     const real_t un = vu * nor;
                     const real_t alpha = tr_bfi->GetAlpha();
                     real_t wp, wm, wd;
                     if (kind == HDGBatchKind::ConvCentered)
                     {
                        // D and G take |a u.n| on both sides; E takes b-a on
                        // side 1 and b+a on side 2.
                        const real_t a = alpha * un, b = std::fabs(alpha * un);
                        wd = ip.weight * b;
                        wm = ip.weight * (b - a);
                        wp = ip.weight * (b + a);
                        pd1[o] += wd;  pg1[o] += wd;  pe1[o] += wm;
                        pd2[o] += wd;  pg2[o] += wd;  pe2[o] += wp;
                        ph[o] += 2. * wd;
                     }
                     else
                     {
                        // Upwinded: D and G take their own side's weight and E
                        // takes the OTHER side's. That crossing is the whole
                        // difference from the centred form.
                        const real_t a = 0.5 * alpha * un;
                        const real_t b = tr_bfi->GetBeta() * std::fabs(un);
                        wp = ip.weight * (b + a);
                        wm = ip.weight * (b - a);
                        pd1[o] += wp;  pg1[o] += wp;  pe1[o] += wm;
                        pd2[o] += wm;  pg2[o] += wm;  pe2[o] += wp;
                        ph[o] += wp + wm;
                     }
                  }
               }
            }
         }

         const auto d_d1 = Reshape(w.d1.Read(), NQ, NF);
         const auto d_d2 = Reshape(w.d2.Read(), NQ, NF);
         const auto d_e1 = Reshape(w.e1.Read(), NQ, NF);
         const auto d_e2 = Reshape(w.e2.Read(), NQ, NF);
         const auto d_g1 = Reshape(w.g1.Read(), NQ, NF);
         const auto d_g2 = Reshape(w.g2.Read(), NQ, NF);
         const auto d_h  = Reshape(w.h.Read(),  NQ, NF);
         const auto d_S = Reshape(sh.Read(), NQ, ND, NSL);
         const int *d_c1 = sl1.Read(), *d_c2 = sl2.Read();
         const auto d_tr = Reshape(trs.Read(), NQ, TRD);
         const int *d_eo = eo.Read(), *d_ho = ho.Read();
         const int *d_o1 = d1o.Read(), *d_o2 = d2o.Read();

         real_t *d_E = E_data.ReadWrite();
         real_t *d_G = G_data.ReadWrite();
         real_t *d_H = H_data.ReadWrite();
         real_t *d_D = Df_data.ReadWrite();

         // The SCALAR shape counts, and the BLOCK strides they are written at.
         // A pass writes only equation EQ's diagonal sub-block, which sits at
         // row/column offset EQ*nd inside a block of leading dimension ndt (and
         // EQ*trd inside trdt for the trace). At VD == 1 these collapse to the
         // scalar kernel exactly -- roff and toff are 0, ndt == nd, trdt == trd
         // -- so the arithmetic is bit-for-bit what it was.
         const int nd = ND, trd = TRD, nq = NQ;
         const int ndt = NDT, trdt = TRDT;
         const int roff = EQ * ND, toff = EQ * TRD;

         mfem::forall(NF, [=] MFEM_HOST_DEVICE (int f)
         {
            const int e0 = d_eo[f], h0 = d_ho[f];
            const int o1 = d_o1[f], o2 = d_o2[f];
            const int k1 = d_c1[f], k2 = d_c2[f];
            const int e2off = e0 + trdt * ndt;

            for (int q = 0; q < nq; q++)
            {
               const real_t ad1 = d_d1(q, f), ad2 = d_d2(q, f);
               const real_t ae1 = d_e1(q, f), ae2 = d_e2(q, f);
               const real_t ag1 = d_g1(q, f), ag2 = d_g2(q, f);

               for (int i = 0; i < nd; i++)
               {
                  const real_t s1i = d_S(q, i, k1), s2i = d_S(q, i, k2);
                  const int ri = roff + i;

                  // D accumulates per ELEMENT, and two faces of one element
                  // collide, so it is the one block that needs atomics.
                  for (int j = 0; j < nd; j++)
                  {
                     const int rj = roff + j;
                     AtomicAdd(d_D[o1 + ri + ndt * rj], ad1 * s1i * d_S(q, j, k1));
                     AtomicAdd(d_D[o2 + ri + ndt * rj], ad2 * s2i * d_S(q, j, k2));
                  }
                  for (int j = 0; j < trd; j++)
                  {
                     const real_t t = d_tr(q, j);
                     const int tj = toff + j;
                     d_E[e0 + ri + ndt * tj]    -= ae1 * s1i * t;
                     d_E[e2off + ri + ndt * tj] -= ae2 * s2i * t;
                     d_G[e0 + tj + trdt * ri]    += ag1 * t * s1i;
                     d_G[e2off + tj + trdt * ri] += ag2 * t * s2i;
                  }
               }

               const real_t ah = d_h(q, f);
               for (int i = 0; i < trd; i++)
               {
                  const real_t t = ah * d_tr(q, i);
                  for (int j = 0; j < trd; j++)
                  {
                     d_H[h0 + (toff + i) + trdt * (toff + j)] -= t * d_tr(q, j);
                  }
               }
            }
         });
      }
   }
}



namespace
{

/// The rule @a bfi integrates @a el at, asked of the integrator.
const IntegrationRule *HDGMassRule(const BilinearFormIntegrator *bfi,
                                   const FiniteElement &el,
                                   ElementTransformation &Tr)
{
   if (auto *m = dynamic_cast<const MassIntegrator*>(bfi))
   { return &m->GetElementIntRule(el, Tr); }
   if (auto *v = dynamic_cast<const VectorMassIntegrator*>(bfi))
   { return &v->GetElementIntRule(el, Tr); }
   return NULL;
}

} // namespace

bool HDGElementMassCanBatch(const FiniteElementSpace &fes,
                            const Array<BilinearFormIntegrator*> &integs)
{
   if (integs.Size() == 0) { return false; }

   Mesh *mesh = fes.GetMesh();
   const int NE = fes.GetNE();
   if (!mesh || NE == 0) { return false; }

   // One element geometry and one dof count, so every block is the same size
   // and one shape table serves the mesh -- and one rule; see HDGOneRule().
   if (!HDGOneRule(fes)) { return false; }

   const int vd = fes.GetVDim();
   for (BilinearFormIntegrator *bfi : integs)
   {
      auto *m = dynamic_cast<const MassIntegrator*>(bfi);
      auto *v = dynamic_cast<const VectorMassIntegrator*>(bfi);
      if (!m && !v) { return false; }
      // A scalar MassIntegrator on a vector space would assemble one block
      // where the space wants vd of them; the per-element route would too, so
      // this is a configuration nobody has, not a case to support.
      if (m && vd != 1) { return false; }
      if (v)
      {
         // vdim == -1 is VectorMassIntegrator's LAZY DEFAULT, not an error:
         // AssembleElementMatrix() resolves it to Trans.GetSpaceDim() on
         // first use, so before any assembly the accessor reports -1 on every
         // integrator built from a scalar Coefficient. Refusing on that would
         // refuse the flux mass of every miniapp in the tree -- which it did,
         // silently, until the report said "flux n" on a case that plainly
         // qualified.
         const int ivd = (v->GetVDim() == -1)
                         ? mesh->SpaceDimension() : v->GetVDim();
         if (ivd != vd) { return false; }
      }

      ElementTransformation *Tr = mesh->GetElementTransformation(0);
      if (!HDGMassRule(bfi, *fes.GetFE(0), *Tr)) { return false; }
   }
   return true;
}

bool HDGElementMassOnDevice(const BilinearFormIntegrator &bfi,
                            const FiniteElementSpace &fes)
{
   const Mesh *mesh = fes.GetMesh();
   if (!mesh) { return false; }
   const Coefficient *Q = nullptr;
   if (auto *v = dynamic_cast<const VectorMassIntegrator*>(&bfi))
   {
      auto *vm = const_cast<VectorMassIntegrator*>(v);
      if (vm->GetVectorCoefficient() || vm->GetMatrixCoefficient())
      { return false; }
      Q = vm->GetCoefficient();
   }
   else if (auto *m = dynamic_cast<const MassIntegrator*>(&bfi))
   {
      Q = m->GetCoefficient();
   }
   else { return false; }
   return !Q || HDGCoefficientOnDevice(const_cast<Coefficient*>(Q), *mesh);
}

void HDGElementMassBatched(const FiniteElementSpace &fes,
                           const Array<BilinearFormIntegrator*> &integs,
                           Vector &emat)
{
   MFEM_VERIFY(HDGElementMassCanBatch(fes, integs),
               "these integrators do not admit the batched element assembly");

   Mesh *mesh = fes.GetMesh();
   const int NE = fes.GetNE();
   const int ND = fes.GetFE(0)->GetDof();
   const int VD = fes.GetVDim();
   const int N = ND * VD;

   emat.SetSize(N * N * NE);
   emat.UseDevice(true);
   emat = 0.;
   if (NE == 0) { return; }

   // VALUE-mapped elements -- every L2 space the Darcy forms carry -- have
   // CalcPhysShape() equal to CalcShape(), so ONE reference table serves the
   // mesh. The per-element table this replaces was NE*NQ*ND reals built on the
   // host and uploaded, 59 MB at meq's 103k triangles at k = 2.
   const FiniteElement &fe0 = *fes.GetFE(0);
   const bool tabled = (fe0.GetMapType() == FiniteElement::VALUE);

   for (BilinearFormIntegrator *bfi : integs)
   {
      auto *vm = dynamic_cast<const VectorMassIntegrator*>(bfi);
      auto *sm = dynamic_cast<const MassIntegrator*>(bfi);

      ElementTransformation *Tr0 = mesh->GetElementTransformation(0);
      const IntegrationRule &ir = *HDGMassRule(bfi, fe0, *Tr0);
      const int NQ = ir.GetNPoints();

      // MassIntegrator::GetCoefficient() is const-qualified on its return and
      // VectorMassIntegrator's is not, so the two cannot meet in a ternary;
      // and Coefficient::Eval() is non-const, so the const one has to be cast.
      // Upstream's asymmetry, not ours.
      const int ivd = vm ? ((vm->GetVDim() == -1) ? mesh->SpaceDimension()
                            : vm->GetVDim()) : 1;
      MFEM_VERIFY(ivd == VD, "vdim mismatch");
      Coefficient *Q = vm ? vm->GetCoefficient()
                       : const_cast<Coefficient*>(sm->GetCoefficient());
      VectorCoefficient *VQ = vm ? vm->GetVectorCoefficient() : NULL;
      MatrixCoefficient *MQ = vm ? vm->GetMatrixCoefficient() : NULL;

      // The weights' SHAPE: one per (point, element) for a scalar or no
      // coefficient, VD for a diagonal one, VD*VD for a coupled one -- the
      // last is what a MatrixCoefficient needs and the only one that costs.
      const int WK = MQ ? VD * VD : (VQ ? VD : 1);
      // And whether the host need be involved at all: without a coefficient,
      // or with one HDGCoefficientOnDevice() takes, a weight is ip.weight *
      // det J * Q, with det J from HDGElementGeometry, computed wherever the
      // mesh's nodes or vertices are, and Q from HDGEvalCoefficient().
      auto *cq = dynamic_cast<ConstantCoefficient*>(Q);
      const bool dev_q = Q && !cq && HDGElementMassOnDevice(*bfi, fes);
      const bool geom_only = !MQ && !VQ && (!Q || cq || dev_q);

      Vector wt(WK * NQ * NE), sh(NQ * ND * (tabled ? 1 : NE));
      wt.UseDevice(true);
      sh.UseDevice(true);

      {
         Vector shape(ND), vec(VD);
         DenseMatrix mc(VD);
         real_t *ps = sh.HostWrite();
         if (tabled)
         {
            for (int q = 0; q < NQ; q++)
            {
               fe0.CalcShape(ir.IntPoint(q), shape);
               for (int i = 0; i < ND; i++) { ps[q + NQ * i] = shape(i); }
            }
         }
         if (geom_only)
         {
            // Not Mesh::GetGeometricFactors(), which would add nodes to a
            // mesh without them; see HDGElementGeometry.
            const HDGElementGeometry geom(*mesh, ir,
                                          GeometricFactors::DETERMINANTS);
            Vector wq(NQ);
            wq.UseDevice(true);
            {
               real_t *pq = wq.HostWrite();
               for (int q = 0; q < NQ; q++) { pq[q] = ir.IntPoint(q).weight; }
            }
            const real_t c = cq ? cq->constant : 1.0;
            // The coefficient at every point of every element, one rule.
            Vector qv;
            if (dev_q)
            {
               Array<int> els(NE), slot(NE);
               for (int e = 0; e < NE; e++) { els[e] = e; }
               slot = 0;
               HDGEvalCoefficient(*Q, NQ, std::vector<IntegrationRule>(1, ir),
                                  els, slot, qv);
            }
            const auto d_det = Reshape(geom.DetJ().Read(), NQ, NE);
            const real_t *d_wq = wq.Read();
            const real_t *d_qv = dev_q ? qv.Read() : nullptr;
            real_t *d_wt = wt.Write();
            const int nq = NQ;
            mfem::forall(NQ * NE, [=] MFEM_HOST_DEVICE (int qe)
            {
               const int q = qe % nq, e = qe / nq;
               d_wt[qe] = (d_wq[q] * d_det(q, e)) * (d_qv ? d_qv[qe] : c);
            });
         }
         if (!tabled || !geom_only)
         {
            real_t *pw = geom_only ? NULL : wt.HostWrite();
            // One weight per element where it is affine; see HDGAffine().
            const bool affine = HDGAffine(*mesh, fe0.GetGeomType());
            real_t el_w = 0.0;
            for (int e = 0; e < NE; e++)
            {
               const FiniteElement &el = *fes.GetFE(e);
               ElementTransformation *Tr = mesh->GetElementTransformation(e);
               for (int q = 0; q < NQ; q++)
               {
                  const IntegrationPoint &ip = ir.IntPoint(q);
                  Tr->SetIntPoint(&ip);
                  if (!geom_only && (q == 0 || !affine)) { el_w = Tr->Weight(); }
                  if (!tabled)
                  {
                     el.CalcPhysShape(*Tr, shape);
                     for (int i = 0; i < ND; i++)
                     {
                        ps[q + NQ * (i + ND * e)] = shape(i);
                     }
                  }
                  if (geom_only) { continue; }

                  const real_t norm = ip.weight * el_w;
                  real_t *w = pw + WK * (q + NQ * e);
                  if (MQ)
                  {
                     MQ->Eval(mc, *Tr, ip);
                     for (int k = 0; k < VD; k++)
                        for (int l = 0; l < VD; l++)
                        {
                           w[k + VD * l] = norm * mc(k, l);
                        }
                  }
                  else if (VQ)
                  {
                     VQ->Eval(vec, *Tr, ip);
                     for (int k = 0; k < VD; k++) { w[k] = norm * vec(k); }
                  }
                  else
                  {
                     w[0] = Q ? (norm * Q->Eval(*Tr, ip)) : norm;
                  }
               }
            }
         }
      }

      const real_t *d_w = wt.Read();
      const real_t *d_s = sh.Read();
      real_t *d_M = emat.ReadWrite();

      const int nd = ND, vd = VD, nq = NQ, n = N, wk = WK;
      const int sstride = tabled ? 0 : NQ * ND;
      const bool coupled = (MQ != NULL), diagonal = (VQ != NULL);

      // ONE THREAD PER ENTRY, its sum over the points held in a register and
      // written once. The one-thread-per-element version this replaces read
      // and wrote its element's whole block from global memory at every
      // point, which a GPU does slowly: 180 ms a call on meq's enriched flux
      // (103k triangles, 20 dofs) on an RTX 2070, nsys. The sum runs over the
      // points in the same order, so each entry is the same number.
      mfem::forall(NE * n * n, [=] MFEM_HOST_DEVICE (int idx)
      {
         const int r = idx % n, c = (idx / n) % n, e = idx / (n * n);
         const int k = r / nd, a = r % nd, l = c / nd, b = c % nd;
         // The block (k, l): a coupled weight, or the diagonal one -- per
         // component, or one shared -- and nothing off it.
         if (!coupled && k != l) { return; }
         const real_t *S = d_s + sstride * e;
         real_t sum = 0.0;
         for (int q = 0; q < nq; q++)
         {
            const real_t *W = d_w + wk * (q + nq * e);
            const real_t w = coupled ? W[k + vd * l] : (diagonal ? W[k] : W[0]);
            if (w == 0.0) { continue; }
            // Field-outermost, native dof order: block (k,l) at rows nd*k
            // and columns nd*l, which is what
            // VectorMassIntegrator::AssembleElementMatrix() writes and what
            // Ordering::byNODES indexes.
            sum += (w * S[q + nq * b]) * S[q + nq * a];
         }
         d_M[(nd * k + a) + n * (nd * l + b) + n * n * e] += sum;
      });
   }
}


namespace
{

/// The MixedConductionNLFIntegrator @a integ is, or NULL if it is not one.
MixedConductionNLFIntegrator *AsMixedConduction(
   BlockNonlinearFormIntegrator *integ)
{
   return dynamic_cast<MixedConductionNLFIntegrator*>(integ);
}

} // namespace

bool HDGMixedConductionResidualCanBatch(const FiniteElementSpace &fes,
                                        BlockNonlinearFormIntegrator *integ)
{
   MixedConductionNLFIntegrator *mc = AsMixedConduction(integ);
   if (!mc) { return false; }

   // The law, and this is the one condition that is about the PHYSICS rather
   // than about the mesh. See the doxygen.
   const MixedFluxFunction &fun = mc->GetFluxFunction();
   if (!dynamic_cast<const LinearDiffusionFlux*>(&fun)) { return false; }
   if (fun.num_equations != 1) { return false; }

   Mesh *mesh = fes.GetMesh();
   const int NE = fes.GetNE();
   if (!mesh || NE == 0) { return false; }

   const int sdim = mesh->SpaceDimension();
   if (fes.GetVDim() != sdim) { return false; }
   if (fun.dim != sdim) { return false; }
   // The kernel holds the point values of the flux and the dual flux in
   // fixed-size registers, which is what keeps them off the stack on a
   // device. Three is every mesh MFEM builds, so this is a bound on the
   // kernel rather than a restriction on the caller.
   if (sdim > 3) { return false; }

   // One geometry and one order, so one rule and one range type; see
   // HDGOneRule(). GetElementIntRule() always answers, so the rule needs no
   // asking here.
   if (!HDGOneRule(fes)) { return false; }
   return fes.GetFE(0)->GetRangeType() == FiniteElement::SCALAR;
}

void HDGMixedConductionResidualBatched(const FiniteElementSpace &fes,
                                       BlockNonlinearFormIntegrator *integ,
                                       const Vector &u_all, Vector &ru_all)
{
   MFEM_VERIFY(HDGMixedConductionResidualCanBatch(fes, integ),
               "this integrator does not admit the batched local residual");

   MixedConductionNLFIntegrator *mc = AsMixedConduction(integ);
   const MixedFluxFunction &fun = mc->GetFluxFunction();

   Mesh *mesh = fes.GetMesh();
   const int NE = fes.GetNE();
   const int ND = fes.GetFE(0)->GetDof();
   const int SDIM = mesh->SpaceDimension();
   const int N = ND * SDIM;

   ElementTransformation *Tr0 = mesh->GetElementTransformation(0);
   const IntegrationRule &ir = mc->GetElementIntRule(*fes.GetFE(0), *Tr0);
   const int NQ = ir.GetNPoints();

   MFEM_VERIFY(u_all.Size() == N * NE, "Incompatible size");
   ru_all.SetSize(N * NE);
   ru_all.UseDevice(true);
   ru_all = 0.;
   if (NE == 0) { return; }

   // CalcShape and NOT CalcPhysShape, because that is what
   // AssembleElementVector() calls -- and it is what makes one table serve
   // the mesh. An L2 collection with map_type INTEGRAL divides the physical
   // shape by detJ, so the physical table would be per element while the
   // reference one is not, and the two answers would differ.
   Vector sh(NQ * ND), wt(NQ * NE), kk(SDIM * SDIM * NQ * NE);
   sh.UseDevice(true);
   wt.UseDevice(true);
   kk.UseDevice(true);

   {
      Vector shape(ND);
      real_t *ps = sh.HostWrite();
      const FiniteElement &fe0 = *fes.GetFE(0);
      for (int q = 0; q < NQ; q++)
      {
         fe0.CalcShape(ir.IntPoint(q), shape);
         for (int i = 0; i < ND; i++) { ps[q + NQ * i] = shape(i); }
      }
   }

   {
      // One conductivity read per point per element -- the same number of
      // coefficient evaluations AssembleElementVector() makes -- and it is
      // read through the LAW rather than reconstructed from its coefficient.
      // ComputeDualFluxJacobian() is exactly the linear map ComputeDualFlux()
      // applies: dualFlux(0,d) = sum_d' flux(0,d') J_F(d,d'), which is what
      // Set(), RightScaling() and MultABt() do in the law's three branches.
      Vector state(1);
      DenseMatrix flux0(1, SDIM), J_u, J_F;
      state = 0.;
      flux0 = 0.;
      real_t *pw = wt.HostWrite(), *pk = kk.HostWrite();
      for (int e = 0; e < NE; e++)
      {
         ElementTransformation *Tr = mesh->GetElementTransformation(e);
         for (int q = 0; q < NQ; q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            Tr->SetIntPoint(&ip);
            fun.ComputeDualFluxJacobian(state, flux0, *Tr, J_u, J_F);
            // Checked at the first point and only there, so it runs in a
            // release build without costing the loop: a VERIFY per point
            // would be NQ*NE norms. The gate admits the law by TYPE, and
            // this is the same statement by VALUE -- J_u is d(dual
            // flux)/d(state), and it has to vanish or the per-point map
            // frozen here is not the law. A guard nothing ever reaches is
            // worse than none, and this one at least runs.
            if (e == 0 && q == 0)
            {
               MFEM_VERIFY(J_F.Height() == SDIM && J_F.Width() == SDIM,
                           "J_F must be dim squared for one equation");
               MFEM_VERIFY(J_u.MaxMaxNorm() == 0.0,
                           "the law depends on the state and cannot be frozen");
            }
            MFEM_ASSERT(J_F.Height() == SDIM && J_F.Width() == SDIM,
                        "J_F must be dim squared for one equation");
            pw[q + NQ * e] = ip.weight * Tr->Weight();
            for (int d = 0; d < SDIM; d++)
               for (int dp = 0; dp < SDIM; dp++)
               {
                  pk[d + SDIM * (dp + SDIM * (q + NQ * e))] = J_F(d, dp);
               }
         }
      }
   }

   const auto d_s = Reshape(sh.Read(), NQ, ND);
   const auto d_w = Reshape(wt.Read(), NQ, NE);
   const auto d_k = Reshape(kk.Read(), SDIM, SDIM, NQ, NE);
   const auto d_u = u_all.Read();
   real_t *d_r = ru_all.ReadWrite();

   const int nd = ND, sd = SDIM, nq = NQ, n = N;

   // One thread per ELEMENT with the quadrature and dof loops inside it. One
   // thread per output ENTRY would need three integer divisions to recover
   // (i, component, element), and this branch has already paid for learning
   // that such arithmetic IS the kernel -- 0.37 s of a 0.95 s loop in
   // ComputeElementsHBatched()'s first draft.
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int MAX_SDIM = 3;
      real_t mu[MAX_SDIM], mF[MAX_SDIM];
      const real_t *u = d_u + n * e;
      real_t *r = d_r + n * e;

      for (int q = 0; q < nq; q++)
      {
         for (int d = 0; d < sd; d++)
         {
            real_t t = 0.;
            for (int i = 0; i < nd; i++) { t += u[d * nd + i] * d_s(q, i); }
            mu[d] = t;
         }
         for (int d = 0; d < sd; d++)
         {
            real_t t = 0.;
            for (int dp = 0; dp < sd; dp++) { t += mu[dp] * d_k(d, dp, q, e); }
            mF[d] = t;
         }
         const real_t w = d_w(q, e);
         for (int d = 0; d < sd; d++)
            for (int i = 0; i < nd; i++)
            {
               // `w * shape(i) * mF` and not `(w * mF) * shape(i)`: the
               // per-element route writes it in this order and floating-point
               // multiplication is not associative, so the other grouping
               // costs the bitwise agreement for nothing.
               r[d * nd + i] += w * d_s(q, i) * mF[d];
            }
      }
   });
}


bool HDGElementDivCanBatch(const FiniteElementSpace &trial_fes,
                           const FiniteElementSpace &test_fes,
                           const Array<BilinearFormIntegrator*> &integs)
{
   if (integs.Size() == 0) { return false; }

   Mesh *mesh = trial_fes.GetMesh();
   const int NE = trial_fes.GetNE();
   if (!mesh || NE == 0) { return false; }
   if (mesh != test_fes.GetMesh()) { return false; }

   // The trial space carries one scalar basis per space dimension, which is
   // what makes the block (test dofs) x (sdim * trial dofs) and what the hat
   // dof mask assumes. An H(div) flux is a different element entirely and
   // takes VectorFEDivergenceIntegrator, which this does not implement.
   if (trial_fes.GetVDim() != mesh->SpaceDimension()) { return false; }
   if (test_fes.GetVDim() != 1) { return false; }

   // One geometry and one order of each space, so one rule; see
   // HDGOneRule(). The rule itself is asked of nothing: GetElementIntRule()
   // always answers, so there is nothing per integrator to refuse on but the
   // kind.
   if (!HDGOneRule(trial_fes) || test_fes.IsVariableOrder()) { return false; }

   for (BilinearFormIntegrator *bfi : integs)
   {
      if (!dynamic_cast<const VectorDivergenceIntegrator*>(bfi)) { return false; }
   }
   return true;
}

void HDGElementDivBatched(const FiniteElementSpace &trial_fes,
                          const FiniteElementSpace &test_fes,
                          const Array<BilinearFormIntegrator*> &integs,
                          Vector &emat)
{
   MFEM_VERIFY(HDGElementDivCanBatch(trial_fes, test_fes, integs),
               "these integrators do not admit the batched divergence "
               "assembly");

   Mesh *mesh = trial_fes.GetMesh();
   const int NE = trial_fes.GetNE();
   const int NDU = trial_fes.GetFE(0)->GetDof();
   const int NDP = test_fes.GetFE(0)->GetDof();
   const int SDIM = mesh->SpaceDimension();
   const int DIM = mesh->Dimension();
   const int W = NDU * SDIM;

   emat.SetSize(NDP * W * NE);
   emat.UseDevice(true);
   emat = 0.;
   if (NE == 0) { return; }

   const FiniteElement &fu0 = *trial_fes.GetFE(0);
   const FiniteElement &fp0 = *test_fes.GetFE(0);

   for (BilinearFormIntegrator *bfi : integs)
   {
      auto *dv = dynamic_cast<const VectorDivergenceIntegrator*>(bfi);

      ElementTransformation *Tr0 = mesh->GetElementTransformation(0);
      const IntegrationRule &ir = dv->GetElementIntRule(fu0, fp0, *Tr0);
      const int NQ = ir.GetNPoints();
      Coefficient *Q = dv->GetCoefficient();
      real_t *d_B = NULL;
      const int ndu = NDU, ndp = NDP, sdim = SDIM, dim = DIM, nq = NQ, w = W;

      // WITHOUT a coefficient, on a VALUE-mapped test space of full dimension
      // -- every Darcy divergence form in the tree -- nothing per element is
      // needed from the host: the trial gradients are a reference table
      // times adj(J), and J comes from HDGElementGeometry, computed wherever
      // the mesh's nodes or vertices are. The host-packed route below it
      // uploaded NE*NQ*NDU*SDIM gradients, 264 MB on meq's enriched flux.
      if (!Q && DIM == SDIM && (DIM == 2 || DIM == 3)
          && fp0.GetMapType() == FiniteElement::VALUE)
      {
         Vector dsh(NQ * NDU * DIM), psh(NQ * NDP), wq(NQ);
         for (Vector *v : {&dsh, &psh, &wq}) { v->UseDevice(true); }
         {
            DenseMatrix dshape(NDU, DIM);
            Vector shape(NDP);
            real_t *pd = dsh.HostWrite(), *pp = psh.HostWrite();
            real_t *pw = wq.HostWrite();
            for (int q = 0; q < NQ; q++)
            {
               const IntegrationPoint &ip = ir.IntPoint(q);
               fu0.CalcDShape(ip, dshape);
               fp0.CalcShape(ip, shape);
               pw[q] = ip.weight;
               for (int d = 0; d < DIM; d++)
                  for (int a = 0; a < NDU; a++)
                  {
                     pd[q + NQ * (a + NDU * d)] = dshape(a, d);
                  }
               for (int i = 0; i < NDP; i++) { pp[q + NQ * i] = shape(i); }
            }
         }
         const HDGElementGeometry geom(*mesh, ir, GeometricFactors::JACOBIANS);
         const auto d_J = Reshape(geom.J().Read(), NQ, DIM, DIM, NE);
         const auto d_ds = Reshape(dsh.Read(), NQ, NDU, DIM);
         const auto d_ps = Reshape(psh.Read(), NQ, NDP);
         const real_t *d_wq = wq.Read();
         d_B = emat.ReadWrite();

         mfem::forall(NE * ndp * w, [=] MFEM_HOST_DEVICE (int idx)
         {
            const int i = idx % ndp, col = (idx / ndp) % w;
            const int e = idx / (ndp * w);
            const int k = col / ndu, a = col % ndu;
            real_t sum = 0.0;
            for (int q = 0; q < nq; q++)
            {
               // Column k of adj(J), CalcAdjugate()'s expressions.
               real_t g = 0.0;
               if (dim == 2)
               {
                  const real_t J00 = d_J(q, 0, 0, e), J01 = d_J(q, 0, 1, e);
                  const real_t J10 = d_J(q, 1, 0, e), J11 = d_J(q, 1, 1, e);
                  const real_t a0 = (k == 0) ? J11 : -J01;
                  const real_t a1 = (k == 0) ? -J10 : J00;
                  g = d_ds(q, a, 0) * a0 + d_ds(q, a, 1) * a1;
               }
               else
               {
                  // adj(J)(d, k) is the (k, d) cofactor of J.
                  for (int d = 0; d < 3; d++)
                  {
                     const int k1 = (k + 1) % 3, k2 = (k + 2) % 3;
                     const int d1 = (d + 1) % 3, d2 = (d + 2) % 3;
                     const real_t cof = d_J(q, k1, d1, e) * d_J(q, k2, d2, e)
                                        - d_J(q, k1, d2, e) * d_J(q, k2, d1, e);
                     g += d_ds(q, a, d) * cof;
                  }
               }
               sum += (d_wq[q] * g) * d_ps(q, i);
            }
            d_B[i + ndp * col + ndp * w * e] += sum;
         });
         continue;
      }

      // The physical gradient of every trial basis function, and the test
      // shape, at every point of every element -- plus one scalar weight.
      // That is all the kernel needs: the divergence of the vector basis
      // function (a, k) is d_k phi_a, so gshape IS the divergence table,
      // which is what DenseMatrix::GradToDiv() says by copying it verbatim.
      Vector gsh(NQ * NDU * SDIM * NE), psh(NQ * NDP * NE), wt(NQ * NE);
      gsh.UseDevice(true);
      psh.UseDevice(true);
      wt = 0.;
      wt.UseDevice(true);

      {
         DenseMatrix dshape(NDU, DIM), gshape(NDU, SDIM), Jadj(DIM, SDIM);
         Vector shape(NDP);
         real_t *pg = gsh.HostWrite(), *pp = psh.HostWrite();
         real_t *pw = wt.HostWrite();

         for (int e = 0; e < NE; e++)
         {
            const FiniteElement &fu = *trial_fes.GetFE(e);
            const FiniteElement &fp = *test_fes.GetFE(e);
            ElementTransformation *Tr = mesh->GetElementTransformation(e);

            for (int q = 0; q < NQ; q++)
            {
               const IntegrationPoint &ip = ir.IntPoint(q);
               Tr->SetIntPoint(&ip);

               fu.CalcDShape(ip, dshape);
               fp.CalcPhysShape(*Tr, shape);
               CalcAdjugate(Tr->Jacobian(), Jadj);
               Mult(dshape, Jadj, gshape);

               real_t c = ip.weight;
               if (DIM != SDIM) { c /= Tr->Weight(); }
               if (Q) { c *= Q->Eval(*Tr, ip); }
               pw[q + NQ * e] += c;

               for (int k = 0; k < SDIM; k++)
                  for (int a = 0; a < NDU; a++)
                  {
                     pg[q + NQ * (a + NDU * (k + SDIM * e))] = gshape(a, k);
                  }
               for (int i = 0; i < NDP; i++)
               {
                  pp[q + NQ * (i + NDP * e)] = shape(i);
               }
            }
         }
      }

      const auto d_g = Reshape(gsh.Read(), NQ, NDU, SDIM, NE);
      const auto d_p = Reshape(psh.Read(), NQ, NDP, NE);
      const auto d_w = Reshape(wt.Read(), NQ, NE);
      d_B = emat.ReadWrite();

      // One thread per entry; see HDGElementMassBatched().
      mfem::forall(NE * ndp * w, [=] MFEM_HOST_DEVICE (int idx)
      {
         const int i = idx % ndp, col = (idx / ndp) % w;
         const int e = idx / (ndp * w);
         const int k = col / ndu, a = col % ndu;
         real_t sum = 0.0;
         for (int q = 0; q < nq; q++)
         {
            sum += (d_w(q, e) * d_g(q, a, k, e)) * d_p(q, i, e);
         }
         d_B[i + ndp * col + ndp * w * e] += sum;
      });
      (void)sdim;
   }
}

bool HDGBdrFaceScatterCanBatch(const FiniteElementSpace &tr_fes,
                               const FiniteElementSpace &el_fes,
                               const Array<BilinearFormIntegrator*> &integs,
                               const std::vector<Array<int>> &face_lists)
{
   if (integs.Size() == 0) { return false; }
   MFEM_ASSERT((size_t)integs.Size() == face_lists.size(), "one list each");
   if (!HDGFaceSpacesCanBatch(tr_fes, el_fes)) { return false; }

   Mesh *mesh = el_fes.GetMesh();
   const int VD = el_fes.GetVDim();
   std::vector<HDGFaceBlock> blocks;
   for (int k = 0; k < integs.Size(); k++)
   {
      if (!HDGFaceBlocksOf(integs[k], VD, blocks)) { return false; }
      for (const HDGFaceBlock &blk : blocks)
      {
         if (auto *d = dynamic_cast<const HDGDiffusionIntegrator*>(blk.bfi))
         {
            const HDGStabilization *st = d->GetStabilization();
            if (st && !st->IsConstant()) { return false; }
         }

         // One rule across this integrator's own face list; see
         // HDGFaceScatterCanBatch(). Elem2 is absent, so the rule takes only
         // Elem1's OrderW() -- but that can still vary element to element.
         // Asked of the first face, for the reason HDGOneRule() gives; the
         // faces are checked to be boundary ones without a transformation.
         if (face_lists[k].Size() == 0) { continue; }
         const int face = face_lists[k][0];
         FaceElementTransformations *ftr =
            mesh->GetFaceElementTransformations(face);
         if (!ftr || ftr->Elem2No >= 0) { return false; }
         const FiniteElement &fe = *el_fes.GetFE(ftr->Elem1No);
         if (!HDGBatchRule(blk.bfi, *tr_fes.GetFaceElement(face), fe, fe, *ftr))
         { return false; }
      }
      for (int fi = 0; fi < face_lists[k].Size(); fi++)
      {
         if (mesh->FaceIsInterior(face_lists[k][fi])) { return false; }
      }
   }
   return true;
}

void HDGBdrFaceScatterBatched(const FiniteElementSpace &tr_fes,
                              const FiniteElementSpace &el_fes,
                              const Array<BilinearFormIntegrator*> &integs,
                              const std::vector<Array<int>> &face_lists,
                              const Array<int> &all_faces,
                              const Array<int> &E_offsets,
                              const Array<int> &H_offsets,
                              const Array<int> &Df_offsets,
                              Vector &E_data, Vector &G_data,
                              Vector &H_data, Vector &Df_data)
{
   MFEM_VERIFY(HDGBdrFaceScatterCanBatch(tr_fes, el_fes, integs, face_lists),
               "these integrators do not admit the batched boundary assembly");

   Mesh *mesh = el_fes.GetMesh();
   if (all_faces.Size() == 0) { return; }

   const int TRD = tr_fes.GetFaceElement(all_faces[0])->GetDof();
   const int ND = el_fes.GetFE(0)->GetDof();
   const int dim = mesh->Dimension();
   // See HDGFaceScatterBatched(): the stored blocks carry the space's vector
   // dimension and a pass writes one equation's diagonal sub-block of them.
   const int VD = el_fes.GetVDim();
   const int NDT = ND * VD, TRDT = TRD * VD;

   {
      // E, G and H over the union of the lists. The per-face route ASSIGNS
      // them with CopyMN, so zeroing here and accumulating below reproduces
      // it; D accumulates in both.
      Array<int> eo(all_faces.Size()), ho(all_faces.Size());
      for (int fi = 0; fi < all_faces.Size(); fi++)
      {
         eo[fi] = E_offsets[all_faces[fi]];
         ho[fi] = H_offsets[all_faces[fi]];
      }
      const int *d_eo = eo.Read(), *d_ho = ho.Read();
      real_t *d_E = E_data.ReadWrite();
      real_t *d_G = G_data.ReadWrite();
      real_t *d_H = H_data.ReadWrite();
      // The whole block, every equation; see HDGFaceScatterBatched().
      const int nd = NDT, trd = TRDT;
      mfem::forall(all_faces.Size(), [=] MFEM_HOST_DEVICE (int f)
      {
         const int e0 = d_eo[f], h0 = d_ho[f];
         for (int j = 0; j < trd; j++)
            for (int i = 0; i < nd; i++)
            {
               d_E[e0 + i + nd * j] = 0.0;
               d_G[e0 + j + trd * i] = 0.0;
            }
         for (int j = 0; j < trd; j++)
            for (int i = 0; i < trd; i++)
            {
               d_H[h0 + i + trd * j] = 0.0;
            }
      });
   }

   std::vector<HDGFaceBlock> blocks;
   for (int k = 0; k < integs.Size(); k++)
   {
      const Array<int> &flist = face_lists[k];
      const int NF = flist.Size();
      if (NF == 0) { continue; }

      MFEM_VERIFY(HDGFaceBlocksOf(integs[k], VD, blocks),
                  "integrator does not resolve into scalar diagonal blocks");
      for (const HDGFaceBlock &blk : blocks)
      {
         const BilinearFormIntegrator *bfi = blk.bfi;
         const int EQ = blk.eq;
         const HDGBatchKind kind = HDGBatchKindOf(bfi);
         auto *dif = dynamic_cast<const HDGDiffusionIntegrator*>(bfi);
         auto *tr_bfi = dynamic_cast<const DGTraceIntegrator*>(bfi);

         const FiniteElement *tr_fe0 = tr_fes.GetFaceElement(flist[0]);
         FaceElementTransformations *ftr0 =
            mesh->GetFaceElementTransformations(flist[0]);
         const FiniteElement &fe0 = *el_fes.GetFE(ftr0->Elem1No);
         const IntegrationRule &ir =
            *HDGBatchRule(bfi, *tr_fe0, fe0, fe0, *ftr0);
         const int NQ = ir.GetNPoints();

         // ZEROED, and not merely sized. Vector(int) does not initialise, the
         // loop below only accumulates into these, and an uninitialised weight
         // is not a crash -- it is a plausible-looking face integral. Measured:
         // the E block of a face whose true weight is exactly zero came back
         // carrying another face's upwinded values, and the answer was 46% out.
         // HDGFaceWeights::Init() does this for the interior kernel; this loop
         // was open-coded and dropped it.
         Vector wd(NQ * NF), we(NQ * NF), wg(NQ * NF), wh(NQ * NF);
         Vector sh, trs(NQ * TRD);
         Array<int> eo(NF), ho(NF), dof(NF), slot(NF);
         wd = 0.;
         we = 0.;
         wg = 0.;
         wh = 0.;
         wd.UseDevice(true);
         we.UseDevice(true);
         wg.UseDevice(true);
         wh.UseDevice(true);
         sh.UseDevice(true);
         trs.UseDevice(true);
         int NSL = NF;

         {
            Vector t(TRD);
            for (int q = 0; q < NQ; q++)
            {
               tr_fe0->CalcShape(ir.IntPoint(q), t);
               for (int i = 0; i < TRD; i++) { trs(q + i * NQ) = t(i); }
            }
         }

         // On the DEVICE where HDGFaceWeightsOnDevice() admits the integrator
         // and the shapes can be tabled: one item per face, side 1, its shapes
         // from one table per (local face, orientation) slot and its weight
         // from the element maps. Otherwise per face on the host, below, with
         // each face its own slot.
         const bool dev_w =
            kind == HDGBatchKind::Diffusion
            && el_fes.GetFE(0)->GetMapType() == FiniteElement::VALUE
            && HDGFaceWeightsOnDevice(*dif, el_fes);
         if (dev_w)
         {
            Array<int> sides(NF);
            sides = 0;
            HDGSideTables tab;
            HDGBuildSideTables(*mesh, el_fes, flist, sides, ir, tab);
            for (int fi = 0; fi < NF; fi++)
            {
               eo[fi] = E_offsets[flist[fi]];
               ho[fi] = H_offsets[flist[fi]];
               dof[fi] = Df_offsets[tab.el[fi]];
            }
            slot = tab.slot;
            NSL = tab.nslot;
            sh = tab.sref;
            Array<int> item_face(NF);
            for (int fi = 0; fi < NF; fi++) { item_face[fi] = fi; }
            const HDGSideMaps maps(*mesh, ir, tab.eip, flist);
            Vector wi;
            HDGFaceWeightsDevice(*dif, maps, ir, tab.eip, item_face, tab.el,
                                 tab.slot, wi);
            // One weight for all four, as below.
            wd = wi;
            we = wi;
            wg = wi;
            wh = wi;
         }
         else
         {
            sh.SetSize(NQ * ND * NF);
            for (int fi = 0; fi < NF; fi++) { slot[fi] = fi; }

            Vector nor(dim), vu(dim), nh(dim), ni(dim), sv(ND);
            DenseMatrix mq(dim);
            real_t *pd = wd.HostWrite(), *pe = we.HostWrite();
            real_t *pg = wg.HostWrite(), *ph = wh.HostWrite();
            real_t *ps = sh.HostWrite();

            for (int fi = 0; fi < NF; fi++)
            {
               const int face = flist[fi];
               FaceElementTransformations *ftr =
                  mesh->GetFaceElementTransformations(face);
               const FiniteElement &fe = *el_fes.GetFE(ftr->Elem1No);
               eo[fi] = E_offsets[face];
               ho[fi] = H_offsets[face];
               dof[fi] = Df_offsets[ftr->Elem1No];

               for (int q = 0; q < NQ; q++)
               {
                  const IntegrationPoint &ip = ir.IntPoint(q);
                  ftr->SetAllIntPoints(&ip);
                  const IntegrationPoint &eip1 = ftr->GetElement1IntPoint();

                  if (dim == 1) { nor(0) = 2 * eip1.x - 1.0; }
                  else { CalcOrtho(ftr->Jacobian(), nor); }

                  fe.CalcPhysShape(*ftr->Elem1, sv);
                  for (int i = 0; i < ND; i++)
                  {
                     ps[q + NQ * (i + ND * fi)] = sv(i);
                  }

                  const int o = q + fi * NQ;
                  if (kind == HDGBatchKind::Diffusion)
                  {
                     real_t un = 0.;
                     if (VectorCoefficient *v = dif->GetVelocity())
                     {
                        v->Eval(vu, *ftr->Elem1, eip1);
                        un = vu * nor;
                     }
                     const real_t un_raw = un;
                     real_t a, b;
                     if (un != 0.)
                     {
                        un /= std::fabs(un);
                        a = 0.5 * dif->GetAlpha() * un;
                        b = dif->GetBeta() * std::fabs(un);
                     }
                     else { a = 0.; b = dif->GetBeta(); }

                     Coefficient *Q = dif->GetCoefficient();
                     MatrixCoefficient *MQ = dif->GetMatrixCoefficient();
                     const real_t wn = ip.weight / ftr->Elem1->Weight();
                     if (!MQ)
                     {
                        ni.Set(Q ? (wn * Q->Eval(*ftr->Elem1, eip1)) : wn, nor);
                     }
                     else
                     {
                        nh.Set(wn, nor);
                        MQ->Eval(mq, *ftr->Elem1, eip1);
                        mq.MultTranspose(nh, ni);
                     }
                     const real_t wq1 = ni * nor;
                     const real_t face_w = ip.weight * nor.Norml2();
                     const real_t w1 = dif->EvalStabilization(
                                          wq1, b + a, un_raw, face_w, 0., 0., *ftr->Elem1);
                     // One weight for all four: the trace block takes side 1's
                     // alone, side 2 being absent.
                     pd[o] += w1;  pe[o] += w1;  pg[o] += w1;  ph[o] += w1;
                  }
                  else
                  {
                     tr_bfi->GetVelocity()->Eval(vu, *ftr->Elem1, eip1);
                     const real_t un = vu * nor;
                     const real_t alpha = tr_bfi->GetAlpha();
                     if (kind == HDGBatchKind::ConvCentered)
                     {
                        const real_t a = alpha * un, b = std::fabs(alpha * un);
                        pd[o] += ip.weight * b;
                        pg[o] += ip.weight * b;
                        pe[o] += ip.weight * (b - a);
                        // b and not 2b: the centred form halves its trace term
                        // at a boundary, where the upwinded one does not.
                        ph[o] += ip.weight * b;
                     }
                     else
                     {
                        const real_t a = 0.5 * alpha * un;
                        const real_t b = tr_bfi->GetBeta() * std::fabs(un);
                        pd[o] += ip.weight * (b + a);
                        pg[o] += ip.weight * (b + a);
                        pe[o] += ip.weight * (b - a);
                        // 2b even here -- "this term must be non-zero at the
                        // boundary for stability reasons, so the advective part
                        // is intentionally dropped", says the per-face routine.
                        ph[o] += ip.weight * 2. * b;
                     }
                  }
               }
            }
         }

         const auto d_wd = Reshape(wd.Read(), NQ, NF);
         const auto d_we = Reshape(we.Read(), NQ, NF);
         const auto d_wg = Reshape(wg.Read(), NQ, NF);
         const auto d_wh = Reshape(wh.Read(), NQ, NF);
         const auto d_s = Reshape(sh.Read(), NQ, ND, NSL);
         const auto d_tr = Reshape(trs.Read(), NQ, TRD);
         const int *d_eo = eo.Read(), *d_ho = ho.Read(), *d_do = dof.Read();
         const int *d_sl = slot.Read();

         real_t *d_E = E_data.ReadWrite();
         real_t *d_G = G_data.ReadWrite();
         real_t *d_H = H_data.ReadWrite();
         real_t *d_D = Df_data.ReadWrite();

         const int nd = ND, trd = TRD, nq = NQ;
         const int ndt = NDT, trdt = TRDT;
         const int roff = EQ * ND, toff = EQ * TRD;

         mfem::forall(NF, [=] MFEM_HOST_DEVICE (int f)
         {
            const int e0 = d_eo[f], h0 = d_ho[f], o1 = d_do[f];
            const int c = d_sl[f];

            for (int q = 0; q < nq; q++)
            {
               const real_t ad = d_wd(q, f), ae = d_we(q, f);
               const real_t ag = d_wg(q, f), ah = d_wh(q, f);

               for (int i = 0; i < nd; i++)
               {
                  const real_t si = d_s(q, i, c);
                  const int ri = roff + i;
                  // Two boundary faces of one element collide in D -- any
                  // corner element has them -- so the atomics are needed here
                  // for the same reason as on the interior.
                  for (int j = 0; j < nd; j++)
                  {
                     AtomicAdd(d_D[o1 + ri + ndt * (roff + j)],
                               ad * si * d_s(q, j, c));
                  }
                  for (int j = 0; j < trd; j++)
                  {
                     const real_t t = d_tr(q, j);
                     const int tj = toff + j;
                     d_E[e0 + ri + ndt * tj] -= ae * si * t;
                     d_G[e0 + tj + trdt * ri] += ag * t * si;
                  }
               }

               for (int i = 0; i < trd; i++)
               {
                  const real_t t = ah * d_tr(q, i);
                  for (int j = 0; j < trd; j++)
                  {
                     d_H[h0 + (toff + i) + trdt * (toff + j)] -= t * d_tr(q, j);
                  }
               }
            }
         });
      }
   }
}

#ifdef MFEM_USE_MPI

bool HDGSharedFaceScatterCanBatch(const FiniteElementSpace &tr_fes,
                                  const FiniteElementSpace &el_fes,
                                  ParMesh &pmesh,
                                  const Array<BilinearFormIntegrator*> &integs,
                                  const Array<int> &face_list)
{
   if (integs.Size() == 0) { return false; }
   if (!HDGFaceSpacesCanBatch(tr_fes, el_fes)) { return false; }
   if (face_list.Size() == 0) { return true; }

   const int VD = el_fes.GetVDim();
   std::vector<HDGFaceBlock> blocks;
   for (BilinearFormIntegrator *bfi : integs)
   {
      if (!HDGFaceBlocksOf(bfi, VD, blocks)) { return false; }
      for (const HDGFaceBlock &blk : blocks)
      {
         // A state dependent stabilization is not a bilinear form at all; the
         // per-face routine refuses it too.
         if (auto *d = dynamic_cast<const HDGDiffusionIntegrator*>(blk.bfi))
         {
            const HDGStabilization *st = d->GetStabilization();
            if (st && !st->IsConstant()) { return false; }
         }

         // ONE rule across the shared list, PER BLOCK, and taken from the
         // ONE-SIDED accessor because that is what the routine this replaces
         // calls. See HDGSharedFaceScatterBatched().
         const IntegrationRule *ir0 = NULL;
         for (int fi = 0; fi < face_list.Size(); fi++)
         {
            const int face = face_list[fi];
            if (!pmesh.FaceIsTrueInterior(face)) { return false; }
            FaceElementTransformations *ftr =
               pmesh.GetSharedFaceTransformationsByLocalIndex(face);
            if (!ftr) { return false; }
            const IntegrationRule *ir =
               HDGBatchSideRule(blk.bfi, 0, *tr_fes.GetFaceElement(face),
                                *el_fes.GetFE(ftr->Elem1No), *ftr);
            if (!ir) { return false; }
            if (!ir0) { ir0 = ir; }
            else if (ir != ir0) { return false; }
         }
      }
   }
   return true;
}

void HDGSharedFaceScatterBatched(const FiniteElementSpace &tr_fes,
                                 const FiniteElementSpace &el_fes,
                                 ParMesh &pmesh,
                                 const Array<BilinearFormIntegrator*> &integs,
                                 const Array<int> &face_list,
                                 const Array<int> &E_offsets,
                                 const Array<int> &H_offsets,
                                 const Array<int> &Df_offsets,
                                 Vector &E_data, Vector &G_data,
                                 Vector &H_data, Vector &Df_data)
{
   MFEM_VERIFY(HDGSharedFaceScatterCanBatch(tr_fes, el_fes, pmesh, integs,
                                            face_list),
               "these integrators do not admit the batched shared-face "
               "assembly");

   const int NF = face_list.Size();
   if (NF == 0) { return; }

   const int TRD = tr_fes.GetFaceElement(face_list[0])->GetDof();
   const int ND = el_fes.GetFE(0)->GetDof();
   const int dim = pmesh.Dimension();
   // See HDGFaceScatterBatched(): the stored blocks carry the space's vector
   // dimension and a pass writes one equation's diagonal sub-block of them.
   const int VD = el_fes.GetVDim();
   const int NDT = ND * VD, TRDT = TRD * VD;

   {
      // E, G and H, zeroed once. The stored E and G of a shared face are
      // ONE-SIDED -- AllocEG() sizes a face from its local elements alone and
      // a shared face has one -- so this is the boundary kernel's zero pass
      // and not the interior one's, which has a second side to clear.
      Array<int> eo(NF), ho(NF);
      for (int fi = 0; fi < NF; fi++)
      {
         eo[fi] = E_offsets[face_list[fi]];
         ho[fi] = H_offsets[face_list[fi]];
      }
      const int *d_eo = eo.Read(), *d_ho = ho.Read();
      real_t *d_E = E_data.ReadWrite();
      real_t *d_G = G_data.ReadWrite();
      real_t *d_H = H_data.ReadWrite();
      const int nd = NDT, trd = TRDT;
      mfem::forall(NF, [=] MFEM_HOST_DEVICE (int f)
      {
         const int e0 = d_eo[f], h0 = d_ho[f];
         for (int j = 0; j < trd; j++)
            for (int i = 0; i < nd; i++)
            {
               d_E[e0 + i + nd * j] = 0.0;
               d_G[e0 + j + trd * i] = 0.0;
            }
         for (int j = 0; j < trd; j++)
            for (int i = 0; i < trd; i++)
            {
               d_H[h0 + i + trd * j] = 0.0;
            }
      });
   }

   std::vector<HDGFaceBlock> blocks;
   for (BilinearFormIntegrator *bfi_v : integs)
   {
      MFEM_VERIFY(HDGFaceBlocksOf(bfi_v, VD, blocks),
                  "integrator does not resolve into scalar diagonal blocks");
      for (const HDGFaceBlock &blk : blocks)
      {
         const BilinearFormIntegrator *bfi = blk.bfi;
         const int EQ = blk.eq;
         const HDGBatchKind kind = HDGBatchKindOf(bfi);
         auto *dif = dynamic_cast<const HDGDiffusionIntegrator*>(bfi);
         auto *tr_bfi = dynamic_cast<const DGTraceIntegrator*>(bfi);

         const FiniteElement *tr_fe0 = tr_fes.GetFaceElement(face_list[0]);
         FaceElementTransformations *ftr0 =
            pmesh.GetSharedFaceTransformationsByLocalIndex(face_list[0]);
         const IntegrationRule &ir =
            *HDGBatchSideRule(bfi, 0, *tr_fe0, *el_fes.GetFE(ftr0->Elem1No),
                              *ftr0);
         const int NQ = ir.GetNPoints();

         // Zeroed as well as sized; Vector(int) does not initialise and these
         // are accumulated into. The boundary kernel's own comment records
         // what an uninitialised weight looked like: a plausible face integral
         // and an answer 46% out.
         Vector wd(NQ * NF), we(NQ * NF), wg(NQ * NF), wh(NQ * NF);
         Vector sh(NQ * ND * NF), trs(NQ * TRD);
         Array<int> eo(NF), ho(NF), dof(NF);
         wd = 0.;
         we = 0.;
         wg = 0.;
         wh = 0.;
         wd.UseDevice(true);
         we.UseDevice(true);
         wg.UseDevice(true);
         wh.UseDevice(true);
         sh.UseDevice(true);
         trs.UseDevice(true);

         {
            Vector t(TRD);
            for (int q = 0; q < NQ; q++)
            {
               tr_fe0->CalcShape(ir.IntPoint(q), t);
               for (int i = 0; i < TRD; i++) { trs(q + i * NQ) = t(i); }
            }
         }

         {
            Vector nor(dim), vu(dim), nh(dim), ni(dim), sv(ND);
            DenseMatrix mq(dim);
            // Caller-allocated, for the reason at the call below.
            FaceElementTransformations ws_face;
            IsoparametricTransformation ws_e1, ws_e2;
            real_t *pd = wd.HostWrite(), *pe = we.HostWrite();
            real_t *pg = wg.HostWrite(), *ph = wh.HostWrite();
            real_t *ps = sh.HostWrite();

            for (int fi = 0; fi < NF; fi++)
            {
               const int face = face_list[fi];
               // CALLER-ALLOCATED, and that is not a preference: the
               // no-argument overload returns a pointer into the ParMesh's ONE
               // cached FaceElementTransformations, so a second call would
               // move the first face's geometry underneath this loop. The
               // interior kernel makes the same point about
               // Mesh::GetElementTransformation().
               pmesh.GetSharedFaceTransformationsByLocalIndex(face, ws_face,
                                                              ws_e1, ws_e2);
               FaceElementTransformations *ftr = &ws_face;
               const FiniteElement &fe = *el_fes.GetFE(ftr->Elem1No);
               eo[fi] = E_offsets[face];
               ho[fi] = H_offsets[face];
               dof[fi] = Df_offsets[ftr->Elem1No];

               for (int q = 0; q < NQ; q++)
               {
                  const IntegrationPoint &ip = ir.IntPoint(q);
                  ftr->SetAllIntPoints(&ip);
                  const IntegrationPoint &eip1 = ftr->GetElement1IntPoint();

                  if (dim == 1) { nor(0) = 2 * eip1.x - 1.0; }
                  else { CalcOrtho(ftr->Jacobian(), nor); }

                  fe.CalcPhysShape(*ftr->Elem1, sv);
                  for (int i = 0; i < ND; i++)
                  {
                     ps[q + NQ * (i + ND * fi)] = sv(i);
                  }

                  const int o = q + fi * NQ;
                  if (kind == HDGBatchKind::Diffusion)
                  {
                     real_t un = 0.;
                     if (VectorCoefficient *v = dif->GetVelocity())
                     {
                        v->Eval(vu, *ftr->Elem1, eip1);
                        un = vu * nor;
                     }
                     const real_t un_raw = un;
                     real_t a, b;
                     if (un != 0.)
                     {
                        un /= std::fabs(un);
                        a = 0.5 * dif->GetAlpha() * un;
                        b = dif->GetBeta() * std::fabs(un);
                     }
                     else { a = 0.; b = dif->GetBeta(); }

                     Coefficient *Q = dif->GetCoefficient();
                     MatrixCoefficient *MQ = dif->GetMatrixCoefficient();
                     const real_t wn = ip.weight / ftr->Elem1->Weight();
                     if (!MQ)
                     {
                        ni.Set(Q ? (wn * Q->Eval(*ftr->Elem1, eip1)) : wn, nor);
                     }
                     else
                     {
                        nh.Set(wn, nor);
                        MQ->Eval(mq, *ftr->Elem1, eip1);
                        mq.MultTranspose(nh, ni);
                     }
                     const real_t wq1 = ni * nor;
                     const real_t face_w = ip.weight * nor.Norml2();
                     const real_t w1 = dif->EvalStabilization(
                                          wq1, b + a, un_raw, face_w, 0., 0., *ftr->Elem1);
                     // One weight for all four, the diffusion form's one-sided
                     // routine writing side 1's stabilization into every block
                     // -- which is the BOUNDARY kernel's arithmetic exactly.
                     // The other rank supplies the trace row's other half from
                     // its own element, and the two sum to the interior
                     // kernel's w1 + w2 under the global assembly.
                     pd[o] += w1;  pe[o] += w1;  pg[o] += w1;  ph[o] += w1;
                  }
                  else
                  {
                     tr_bfi->GetVelocity()->Eval(vu, *ftr->Elem1, eip1);
                     const real_t un = vu * nor;
                     const real_t alpha = tr_bfi->GetAlpha();
                     if (kind == HDGBatchKind::ConvCentered)
                     {
                        const real_t a = alpha * un, b = std::fabs(alpha * un);
                        pd[o] += ip.weight * b;
                        pg[o] += ip.weight * b;
                        pe[o] += ip.weight * (b - a);
                        // **(b - a), where the BOUNDARY kernel takes b and the
                        // interior one takes 2b.** The one-sided routine
                        // branches on Trans.Elem2No, and a shared face has one
                        // -- the neighbour's shifted index -- so it is the
                        // interior expression that fires, halved. The advective
                        // part is dropped only where there is genuinely no
                        // second element, and a partition boundary is not that.
                        ph[o] += ip.weight * (b - a);
                     }
                     else
                     {
                        const real_t a = 0.5 * alpha * un;
                        const real_t b = tr_bfi->GetBeta() * std::fabs(un);
                        pd[o] += ip.weight * (b + a);
                        pg[o] += ip.weight * (b + a);
                        pe[o] += ip.weight * (b - a);
                        // (b - a) and not 2b, for the reason above. The other
                        // rank's own pass sees the opposite normal and so
                        // contributes (b + a); the two sum to the interior
                        // kernel's wp + wm on the trace row they share.
                        ph[o] += ip.weight * (b - a);
                     }
                  }
               }
            }
         }

         const auto d_wd = Reshape(wd.Read(), NQ, NF);
         const auto d_we = Reshape(we.Read(), NQ, NF);
         const auto d_wg = Reshape(wg.Read(), NQ, NF);
         const auto d_wh = Reshape(wh.Read(), NQ, NF);
         const auto d_s = Reshape(sh.Read(), NQ, ND, NF);
         const auto d_tr = Reshape(trs.Read(), NQ, TRD);
         const int *d_eo = eo.Read(), *d_ho = ho.Read(), *d_do = dof.Read();

         real_t *d_E = E_data.ReadWrite();
         real_t *d_G = G_data.ReadWrite();
         real_t *d_H = H_data.ReadWrite();
         real_t *d_D = Df_data.ReadWrite();

         const int nd = ND, trd = TRD, nq = NQ;
         const int ndt = NDT, trdt = TRDT;
         const int roff = EQ * ND, toff = EQ * TRD;

         mfem::forall(NF, [=] MFEM_HOST_DEVICE (int f)
         {
            const int e0 = d_eo[f], h0 = d_ho[f], o1 = d_do[f];

            for (int q = 0; q < nq; q++)
            {
               const real_t ad = d_wd(q, f), ae = d_we(q, f);
               const real_t ag = d_wg(q, f), ah = d_wh(q, f);

               for (int i = 0; i < nd; i++)
               {
                  const real_t si = d_s(q, i, f);
                  const int ri = roff + i;
                  // Two shared faces of one element collide in D -- a corner
                  // element of a partition has them -- so the atomics are
                  // needed here for the same reason as on the interior, and
                  // D is accumulated into by the interior and boundary passes
                  // as well.
                  for (int j = 0; j < nd; j++)
                  {
                     AtomicAdd(d_D[o1 + ri + ndt * (roff + j)],
                               ad * si * d_s(q, j, f));
                  }
                  for (int j = 0; j < trd; j++)
                  {
                     const real_t t = d_tr(q, j);
                     const int tj = toff + j;
                     d_E[e0 + ri + ndt * tj] -= ae * si * t;
                     d_G[e0 + tj + trdt * ri] += ag * t * si;
                  }
               }

               for (int i = 0; i < trd; i++)
               {
                  const real_t t = ah * d_tr(q, i);
                  for (int j = 0; j < trd; j++)
                  {
                     d_H[h0 + (toff + i) + trdt * (toff + j)] -= t * d_tr(q, j);
                  }
               }
            }
         });
      }
   }
}

#endif // MFEM_USE_MPI

void HDGDiffusionFaceScatterBatched(const FiniteElementSpace &tr_fes,
                                    const FiniteElementSpace &el_fes,
                                    Coefficient *Q, real_t beta,
                                    const HDGStabilization *stab,
                                    const Array<int> &face_list,
                                    const Array<int> &E_offsets,
                                    const Array<int> &H_offsets,
                                    const Array<int> &Df_offsets,
                                    Vector &E_data, Vector &G_data,
                                    Vector &H_data, Vector &Df_data)
{
   MFEM_VERIFY(HDGDiffusionFaceMatricesCanBatch(tr_fes, el_fes),
               "the spaces do not admit the batched face assembly");
   MFEM_VERIFY(!stab || stab->IsConstant(),
               "a state dependent stabilization makes the face term nonlinear");

   Mesh *mesh = el_fes.GetMesh();
   const int NF = face_list.Size();
   if (NF == 0) { return; }

   const FiniteElement *tr_fe0 = tr_fes.GetFaceElement(face_list[0]);
   const int TRD = tr_fe0->GetDof();
   const int ND = el_fes.GetFE(0)->GetDof();

   const int order = 2 * std::max(el_fes.GetFE(0)->GetOrder(),
                                  tr_fe0->GetOrder());
   FaceElementTransformations *ftr0 =
      mesh->GetInteriorFaceTransformations(face_list[0]);
   MFEM_VERIFY(ftr0, "no interior face transformation");
   const IntegrationRule &ir = IntRules.Get(ftr0->GetGeometryType(), order);
   const int NQ = ir.GetNPoints();

   // Same host precompute as the dense form; see the note there on what it
   // would take to remove it.
   Vector w1(NQ * NF), w2(NQ * NF);
   Vector sh1(NQ * ND * NF), sh2(NQ * ND * NF);
   Vector trs(NQ * TRD);
   Array<int> eo(NF), ho(NF), d1(NF), d2(NF);
   w1.UseDevice(true);
   w2.UseDevice(true);
   sh1.UseDevice(true);
   sh2.UseDevice(true);
   trs.UseDevice(true);

   {
      Vector t(TRD);
      for (int q = 0; q < NQ; q++)
      {
         tr_fe0->CalcShape(ir.IntPoint(q), t);
         for (int i = 0; i < TRD; i++) { trs(q + i * NQ) = t(i); }
      }
   }

   {
      const int dim = mesh->Dimension();
      Vector nor(dim), s1(ND), s2(ND);
      real_t *pw1 = w1.HostWrite(), *pw2 = w2.HostWrite();
      real_t *ps1 = sh1.HostWrite(), *ps2 = sh2.HostWrite();

      for (int fi = 0; fi < NF; fi++)
      {
         const int face = face_list[fi];
         FaceElementTransformations *ftr =
            mesh->GetInteriorFaceTransformations(face);
         const FiniteElement &e1 = *el_fes.GetFE(ftr->Elem1No);
         const FiniteElement &e2 = *el_fes.GetFE(ftr->Elem2No);
         eo[fi] = E_offsets[face];
         ho[fi] = H_offsets[face];
         d1[fi] = Df_offsets[ftr->Elem1No];
         d2[fi] = Df_offsets[ftr->Elem2No];

         for (int q = 0; q < NQ; q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            ftr->SetAllIntPoints(&ip);
            const IntegrationPoint &eip1 = ftr->GetElement1IntPoint();
            const IntegrationPoint &eip2 = ftr->GetElement2IntPoint();

            if (dim == 1) { nor(0) = 2 * eip1.x - 1.0; }
            else { CalcOrtho(ftr->Jacobian(), nor); }

            e1.CalcPhysShape(*ftr->Elem1, s1);
            e2.CalcPhysShape(*ftr->Elem2, s2);

            const real_t nn = nor * nor;
            const real_t face_w = ip.weight * nor.Norml2();
            real_t wn1 = ip.weight / ftr->Elem1->Weight();
            if (Q) { wn1 *= Q->Eval(*ftr->Elem1, eip1); }
            real_t wn2 = ip.weight / ftr->Elem2->Weight();
            if (Q) { wn2 *= Q->Eval(*ftr->Elem2, eip2); }
            const real_t wq1 = wn1 * nn, wq2 = wn2 * nn;
            pw1[q + fi * NQ] = stab
                               ? face_w * stab->Eval((face_w != 0.) ? (wq1 * beta / face_w) : 0.,
                                                     0., 0., 0., *ftr->Elem1)
                               : wq1 * beta;
            pw2[q + fi * NQ] = stab
                               ? face_w * stab->Eval((face_w != 0.) ? (wq2 * beta / face_w) : 0.,
                                                     0., 0., 0., *ftr->Elem2)
                               : wq2 * beta;
            for (int i = 0; i < ND; i++)
            {
               ps1[q + NQ * (i + ND * fi)] = s1(i);
               ps2[q + NQ * (i + ND * fi)] = s2(i);
            }
         }
      }
   }

   const auto d_w1 = Reshape(w1.Read(), NQ, NF);
   const auto d_w2 = Reshape(w2.Read(), NQ, NF);
   const auto d_s1 = Reshape(sh1.Read(), NQ, ND, NF);
   const auto d_s2 = Reshape(sh2.Read(), NQ, ND, NF);
   const auto d_tr = Reshape(trs.Read(), NQ, TRD);
   const int *d_eo = eo.Read(), *d_ho = ho.Read();
   const int *d_d1 = d1.Read(), *d_d2 = d2.Read();

   real_t *d_E = E_data.ReadWrite();
   real_t *d_G = G_data.ReadWrite();
   real_t *d_H = H_data.ReadWrite();
   real_t *d_D = Df_data.ReadWrite();

   const int nd = ND, trd = TRD, nq = NQ;

   mfem::forall(NF, [=] MFEM_HOST_DEVICE (int f)
   {
      const int e0 = d_eo[f], h0 = d_ho[f];
      const int o1 = d_d1[f], o2 = d_d2[f];
      const int e2off = e0 + trd * nd;      // side 2 follows side 1

      // E, G and H belong to this face alone: overwrite, no atomics.
      for (int j = 0; j < trd; j++)
         for (int i = 0; i < nd; i++)
         {
            d_E[e0 + i + nd * j] = 0.0;
            d_E[e2off + i + nd * j] = 0.0;
            d_G[e0 + j + trd * i] = 0.0;
            d_G[e2off + j + trd * i] = 0.0;
         }
      for (int j = 0; j < trd; j++)
         for (int i = 0; i < trd; i++)
         {
            d_H[h0 + i + trd * j] = 0.0;
         }

      for (int q = 0; q < nq; q++)
      {
         const real_t a1 = d_w1(q, f), a2 = d_w2(q, f);

         for (int i = 0; i < nd; i++)
         {
            const real_t b1 = a1 * d_s1(q, i, f);
            const real_t b2 = a2 * d_s2(q, i, f);

            // D accumulates per ELEMENT, and two faces of one element
            // collide, so it is the one block that needs atomics.
            for (int j = 0; j < nd; j++)
            {
               AtomicAdd(d_D[o1 + i + nd * j], b1 * d_s1(q, j, f));
               AtomicAdd(d_D[o2 + i + nd * j], b2 * d_s2(q, j, f));
            }
            for (int j = 0; j < trd; j++)
            {
               const real_t e1v = b1 * d_tr(q, j);
               const real_t e2v = b2 * d_tr(q, j);
               d_E[e0 + i + nd * j]    -= e1v;
               d_E[e2off + i + nd * j] -= e2v;
               d_G[e0 + j + trd * i]    += e1v;
               d_G[e2off + j + trd * i] += e2v;
            }
         }

         const real_t aa = a1 + a2;
         for (int i = 0; i < trd; i++)
         {
            const real_t t = aa * d_tr(q, i);
            for (int j = 0; j < trd; j++)
            {
               d_H[h0 + i + trd * j] -= t * d_tr(q, j);
            }
         }
      }
   });
}


namespace
{

/// What HDGNLFaceGradScatterBatched() knows how to weigh.
enum class HDGNLGradKind { Unsupported, Diffusion, MixedConduction, Hyperbolic };

HDGNLGradKind HDGNLGradKindOf(const NonlinearFormIntegrator *nlfi)
{
   if (auto *d = dynamic_cast<const HDGDiffusionIntegrator*>(nlfi))
   {
      // A state dependent stabilization takes a different pair of weights --
      // s + d1s*jump and -s + d2s*jump rather than +w and -w -- and nothing
      // in this tree installs one, so it is refused rather than guessed at.
      const HDGStabilization *st = d->GetStabilization();
      if (st && !st->IsConstant()) { return HDGNLGradKind::Unsupported; }
      return HDGNLGradKind::Diffusion;
   }
   if (dynamic_cast<const HyperbolicFormIntegrator*>(nlfi))
   { return HDGNLGradKind::Hyperbolic; }
   return HDGNLGradKind::Unsupported;
}

HDGNLGradKind HDGNLGradKindOf(const BlockNonlinearFormIntegrator *nlfi)
{
   if (dynamic_cast<const MixedConductionNLFIntegrator*>(nlfi))
   { return HDGNLGradKind::MixedConduction; }
   return HDGNLGradKind::Unsupported;
}

/// The one-sided rule @a nlfi integrates this pair at, asked of it.
const IntegrationRule *HDGNLGradRule(const NonlinearFormIntegrator *nlfi,
                                     const FiniteElement &tr_fe,
                                     const FiniteElement &el_fe,
                                     const FiniteElement *fl_fe,
                                     FaceElementTransformations &ftr)
{
   MFEM_CONTRACT_VAR(fl_fe);
   if (auto *d = dynamic_cast<const HDGDiffusionIntegrator*>(nlfi))
   {
      if (const IntegrationRule *ir = d->GetIntRule()) { return ir; }
      return &d->GetHDGFaceIntRule(tr_fe, el_fe, ftr);
   }
   if (auto *h = dynamic_cast<const HyperbolicFormIntegrator*>(nlfi))
   { return &h->GetHDGFaceIntRule(tr_fe, el_fe, ftr); }
   return NULL;
}

const IntegrationRule *HDGNLGradRule(const BlockNonlinearFormIntegrator *nlfi,
                                     const FiniteElement &tr_fe,
                                     const FiniteElement &el_fe,
                                     const FiniteElement *fl_fe,
                                     FaceElementTransformations &ftr)
{
   if (auto *m = dynamic_cast<const MixedConductionNLFIntegrator*>(nlfi))
   {
      if (!fl_fe) { return NULL; }
      return &m->GetHDGFaceIntRule(*fl_fe, el_fe, ftr);
   }
   return NULL;
}

/// The geometry, dof-count and space conditions the kernel needs.
/** @brief Where a batched face pass gets its transformations, and how many
    PAIRS one face of its list contributes.

    Two sources rather than two kernels, and the reason is arithmetic: a
    shared face is one-sided from this rank -- a local Elem1, no local Elem2 --
    so it contributes ONE pair where an interior face contributes two, and
    Mesh::GetInteriorFaceTransformations() returns NULL for it because
    Mesh::FaceIsInterior() is `Elem2No >= 0`. Nothing else about the two
    passes differs: the weights are the one-sided routines' either way, the
    destinations are caller-supplied per pair, and the side-2 arithmetic is
    reached only through the side loop this bounds. Duplicating the two
    kernels for the shared case would have been six hundred lines of near
    copy, and the copy is where the two would then have parted company.

    The transformation pointer is the mesh's own cached object in both cases
    and so must be consumed before the next call, which is what every caller
    here does. */
struct HDGFacePairSource
{
   Mesh *mesh;
#ifdef MFEM_USE_MPI
   ParMesh *pmesh;
   HDGFacePairSource(Mesh *m, ParMesh *pm) : mesh(m), pmesh(pm) { }
   int NumSides() const { return pmesh ? 1 : 2; }
   FaceElementTransformations *Get(int face) const
   {
      if (pmesh)
      {
         return pmesh->FaceIsTrueInterior(face)
                ? pmesh->GetSharedFaceTransformationsByLocalIndex(face) : NULL;
      }
      return mesh->GetInteriorFaceTransformations(face);
   }
#else
   HDGFacePairSource(Mesh *m, ParMesh *) : mesh(m) { }
   int NumSides() const { return 2; }
   FaceElementTransformations *Get(int face) const
   { return mesh->GetInteriorFaceTransformations(face); }
#endif
};

bool HDGNLFaceGradGeometryOK(const FiniteElementSpace &tr_fes,
                             const FiniteElementSpace &el_fes,
                             const Array<int> &face_list,
                             int &ND, int &TRD)
{
   const Mesh *mesh = el_fes.GetMesh();
   if (!mesh || mesh->GetNE() == 0) { return false; }
   if (mesh->Nonconforming()) { return false; }
   // The weight precompute sizes its normal by Dimension(); an embedded mesh
   // needs SpaceDimension() and is refused rather than mis-sized.
   if (mesh->Dimension() != mesh->SpaceDimension()) { return false; }
   if (!dynamic_cast<const DG_Interface_FECollection*>(tr_fes.FEColl()))
   { return false; }
   if (tr_fes.GetVDim() != el_fes.GetVDim()) { return false; }

   const Geometry::Type g = mesh->GetElementBaseGeometry(0);
   ND = el_fes.GetFE(0)->GetDof();
   for (int e = 1; e < mesh->GetNE(); e++)
   {
      if (mesh->GetElementBaseGeometry(e) != g) { return false; }
      if (el_fes.GetFE(e)->GetDof() != ND) { return false; }
   }
   if (face_list.Size() == 0) { TRD = 0; return true; }
   TRD = tr_fes.GetFaceElement(face_list[0])->GetDof();
   for (int fi = 0; fi < face_list.Size(); fi++)
   {
      const FiniteElement *tfe = tr_fes.GetFaceElement(face_list[fi]);
      if (tfe->GetDof() != TRD) { return false; }
      if (tfe->GetMapType() != FiniteElement::VALUE) { return false; }
   }
   return true;
}

} // namespace

bool HDGNLFaceGradCanBatch(const FiniteElementSpace &tr_fes,
                           const FiniteElementSpace &el_fes,
                           const FiniteElementSpace *fl_fes,
                           const Array<NonlinearFormIntegrator*> &integs,
                           const Array<BlockNonlinearFormIntegrator*> &bintegs,
                           const Array<int> &face_list, ParMesh *pmesh)
{
   if (integs.Size() + bintegs.Size() == 0) { return false; }

   int ND = 0, TRD = 0;
   if (!HDGNLFaceGradGeometryOK(tr_fes, el_fes, face_list, ND, TRD))
   { return false; }

   const int neq = el_fes.GetVDim();
   const HDGFacePairSource src(el_fes.GetMesh(), pmesh);
   const int ns = src.NumSides();

   for (int k = 0; k < integs.Size(); k++)
   {
      const HDGNLGradKind kind = HDGNLGradKindOf(integs[k]);
      if (kind == HDGNLGradKind::Unsupported) { return false; }
      // HDGDiffusionIntegrator builds an (el + tr) square with no notion of
      // vdim; a system wraps it, and the wrapper is not one of these kinds.
      if (kind == HDGNLGradKind::Diffusion && neq != 1) { return false; }
   }
   for (int k = 0; k < bintegs.Size(); k++)
   {
      if (HDGNLGradKindOf(bintegs[k]) == HDGNLGradKind::Unsupported)
      { return false; }
      if (!fl_fes) { return false; }
   }
   if (face_list.Size() == 0) { return true; }

   // ONE rule per integrator across every pair, asked of the integrator on
   // each pair because that is where the answer is -- the one-sided rules
   // take the max over one element and the trace, so two neighbours of
   // different order want different rules on the same face.
   for (int k = 0; k < integs.Size() + bintegs.Size(); k++)
   {
      const IntegrationRule *ir0 = NULL;
      for (int fi = 0; fi < face_list.Size(); fi++)
      {
         FaceElementTransformations *ftr = src.Get(face_list[fi]);
         if (!ftr) { return false; }
         const FiniteElement &tr_fe = *tr_fes.GetFaceElement(face_list[fi]);
         // Elem2No on a SHARED face is the neighbour's shifted index and is
         // not an el_fes element at all; ns == 1 is what keeps it unread.
         const int els[2] = { ftr->Elem1No, (ns == 2) ? ftr->Elem2No : -1 };
         for (int side = 0; side < ns; side++)
         {
            const FiniteElement &el_fe = *el_fes.GetFE(els[side]);
            const FiniteElement *fl_fe =
               fl_fes ? fl_fes->GetFE(els[side]) : NULL;
            const IntegrationRule *ir = (k < integs.Size())
                                        ? HDGNLGradRule(integs[k], tr_fe, el_fe, fl_fe, *ftr)
                                        : HDGNLGradRule(bintegs[k - integs.Size()], tr_fe, el_fe, fl_fe,
                                                        *ftr);
            if (!ir) { return false; }
            if (!ir0) { ir0 = ir; }
            else if (ir != ir0) { return false; }
         }
      }
   }
   return true;
}

bool HDGNLFaceResidualCanBatch(
   const FiniteElementSpace &tr_fes, const FiniteElementSpace &el_fes,
   const FiniteElementSpace *fl_fes,
   const Array<NonlinearFormIntegrator*> &integs,
   const Array<BlockNonlinearFormIntegrator*> &bintegs,
   const Array<int> &face_list, ParMesh *pmesh)
{
   // Everything the GRADIENT pass needs, because the families and the
   // geometry conditions are the same and there must not be two dispatches.
   if (!HDGNLFaceGradCanBatch(tr_fes, el_fes, fl_fes, integs, bintegs,
                              face_list, pmesh)) { return false; }

   // **And one refusal the gradient does not have.** That pass implements
   // MixedConductionNLFIntegrator and this one does not, deliberately: the
   // integrator is unreachable as a hybridized face constraint from every
   // miniapp in the tree, established by printing which slot
   // EnableHybridization() fills across all 152 serial references. A kernel
   // for a slot nothing fills is untested code, and this file has recorded
   // that mistake twice. Refused here rather than aborted in the kernel.
   for (int k = 0; k < bintegs.Size(); k++)
   {
      if (HDGNLGradKindOf(bintegs[k]) == HDGNLGradKind::MixedConduction)
      { return false; }
   }
   for (int k = 0; k < integs.Size(); k++)
   {
      if (HDGNLGradKindOf(integs[k]) == HDGNLGradKind::MixedConduction)
      { return false; }
   }
   return true;
}

void HDGNLFaceResidualBatched(
   const FiniteElementSpace &tr_fes, const FiniteElementSpace &el_fes,
   const FiniteElementSpace *fl_fes,
   const Array<NonlinearFormIntegrator*> &integs,
   const Array<BlockNonlinearFormIntegrator*> &bintegs,
   const Array<int> &face_list,
   const Vector &el_state, const Vector &tr_state,
   Vector &r_el, Vector &r_tr, ParMesh *pmesh)
{
   MFEM_VERIFY(HDGNLFaceResidualCanBatch(tr_fes, el_fes, fl_fes, integs,
                                         bintegs, face_list, pmesh),
               "these integrators do not admit the batched face residual");

   Mesh *mesh = el_fes.GetMesh();
   const HDGFacePairSource src(mesh, pmesh);
   const int ns = src.NumSides();
   const int NF = face_list.Size();
   if (NF == 0) { r_el.SetSize(0); r_tr.SetSize(0); return; }
   const int NP = ns * NF;
   const int neq = el_fes.GetVDim();

   int ND = 0, TRD = 0;
   HDGNLFaceGradGeometryOK(tr_fes, el_fes, face_list, ND, TRD);
   const int LDD = neq * ND, LDC = neq * TRD;

   MFEM_ASSERT(el_state.Size() == NP * LDD && tr_state.Size() == NP * LDC, "");

   r_el.SetSize(NP * LDD);
   r_tr.SetSize(NP * LDC);
   r_el.UseDevice(true);
   r_tr.UseDevice(true);
   r_el = 0.;
   r_tr = 0.;

   const int dim = mesh->Dimension();
   const int NI = integs.Size() + bintegs.Size();

   for (int k = 0; k < NI; k++)
   {
      NonlinearFormIntegrator *nlfi = (k < integs.Size()) ? integs[k] : NULL;
      BlockNonlinearFormIntegrator *bnlfi =
         nlfi ? NULL : bintegs[k - integs.Size()];
      const HDGNLGradKind kind = nlfi ? HDGNLGradKindOf(nlfi)
                                 : HDGNLGradKindOf(bnlfi);

      FaceElementTransformations *ftr0 = src.Get(face_list[0]);
      const FiniteElement &tr_fe0 = *tr_fes.GetFaceElement(face_list[0]);
      const IntegrationRule &ir = nlfi
                                  ? *HDGNLGradRule(nlfi, tr_fe0, *el_fes.GetFE(ftr0->Elem1No),
                                                   fl_fes ? fl_fes->GetFE(ftr0->Elem1No) : NULL, *ftr0)
                                  : *HDGNLGradRule(bnlfi, tr_fe0, *el_fes.GetFE(ftr0->Elem1No),
                                                   fl_fes ? fl_fes->GetFE(ftr0->Elem1No) : NULL, *ftr0);
      const int NQ = ir.GetNPoints();

      // ONE weight VECTOR per (point, pair) for each row, where the gradient
      // pass carries a matrix. Sized AND ZEROED, for the reason its own
      // comment gives: Vector(int) does not initialise, and a face whose
      // weight is exactly zero would otherwise carry heap -- which read as a
      // plausible face integral with a 46% wrong answer once already.
      Vector wel(NQ * neq * NP), wtr(NQ * neq * NP);
      Vector sel(NQ * ND * NP), str(NQ * TRD);
      wel = 0.; wtr = 0.; sel = 0.; str = 0.;
      wel.UseDevice(true); wtr.UseDevice(true);
      sel.UseDevice(true); str.UseDevice(true);

      {
         Vector t(TRD);
         real_t *ps = str.HostWrite();
         for (int q = 0; q < NQ; q++)
         {
            tr_fe0.CalcShape(ir.IntPoint(q), t);
            for (int i = 0; i < TRD; i++) { ps[q + i * NQ] = t(i); }
         }
      }

      {
         auto *dif = dynamic_cast<HDGDiffusionIntegrator*>(nlfi);
         auto *hyp = dynamic_cast<const HyperbolicFormIntegrator*>(nlfi);
         auto *mix = dynamic_cast<const MixedConductionNLFIntegrator*>(bnlfi);

         Vector nor(dim), vu(dim), nh(dim), ni(dim), s(ND);
         Vector st_el(neq), st_tr(neq), tq(TRD), fl(neq);
         DenseMatrix mq(dim);

         real_t *pwel = wel.HostWrite(), *pwtr = wtr.HostWrite();
         real_t *psel = sel.HostWrite();
         const real_t *pes = el_state.HostRead(), *pts = tr_state.HostRead();
         // Hoisted: this sat inside the quadrature loop, so the Memory<>
         // validity check ran NQ*2*NF times for a table that never moves.
         const real_t *pstr = str.HostRead();

         for (int fi = 0; fi < NF; fi++)
         {
            FaceElementTransformations *ftr = src.Get(face_list[fi]);
            // Elem2No on a SHARED face is the neighbour's shifted index, not
            // an el_fes element; ns == 1 is what keeps it unread.
            const int els[2] = { ftr->Elem1No, (ns == 2) ? ftr->Elem2No : -1 };

            for (int side = 0; side < ns; side++)
            {
               const int p = ns * fi + side;
               const FiniteElement &el_fe = *el_fes.GetFE(els[side]);
               const DenseMatrix el_mat(const_cast<real_t*>(pes + p * LDD),
                                        ND, neq);
               const DenseMatrix tr_mat(const_cast<real_t*>(pts + p * LDC),
                                        TRD, neq);

               for (int q = 0; q < NQ; q++)
               {
                  const IntegrationPoint &ip = ir.IntPoint(q);
                  ftr->SetAllIntPoints(&ip);
                  const IntegrationPoint &eip1 = ftr->GetElement1IntPoint();
                  const IntegrationPoint &eip2 = ftr->GetElement2IntPoint();
                  const IntegrationPoint &eip = side ? eip2 : eip1;
                  ElementTransformation *ElTr = side ? ftr->Elem2 : ftr->Elem1;

                  if (dim == 1) { nor(0) = 2 * eip1.x - 1.0; }
                  else { CalcOrtho(ftr->Jacobian(), nor); }

                  // The trace table is (NQ x TRD) with stride NQ, so a row is
                  // not contiguous; copy rather than alias, per the gradient
                  // pass's note about Vector's move assignment.
                  for (int i = 0; i < TRD; i++)
                  { tq(i) = pstr[q + i * NQ]; }

                  real_t *we = pwel + (q + NQ * neq * p);
                  real_t *wt = pwtr + (q + NQ * neq * p);
                  const int wstride = NQ;

                  if (kind == HDGNLGradKind::Hyperbolic)
                  {
                     el_fe.CalcShape(eip, s);
                     if (side) { nor.Neg(); }
                     el_mat.MultTranspose(s, st_el);
                     tr_mat.MultTranspose(tq, st_tr);
                     const real_t w = -ip.weight * hyp->GetSign();
                     // Average() where the gradient pass calls AverageGrad():
                     // the numerical flux AT the state, a neq-vector, instead
                     // of its Jacobian. That substitution IS this routine.
                     const NumericalFlux &nf = hyp->GetNumericalFlux();
                     nf.Average(st_tr, st_el, nor, *ftr, fl);
                     for (int di = 0; di < neq; di++)
                     {
                        we[wstride * di] += w * fl(di);
                        wt[wstride * di] += w * fl(di);
                     }
                  }
                  else if (kind == HDGNLGradKind::Diffusion)
                  {
                     if (side) { nor.Neg(); }
                     real_t un = 0.;
                     if (VectorCoefficient *v = dif->GetVelocity())
                     { v->Eval(vu, *ftr->Elem1, eip1); un = vu * nor; }
                     el_fe.CalcPhysShape(*ElTr, s);
                     real_t wn = ip.weight / ElTr->Weight();
                     Coefficient *Q = dif->GetCoefficient();
                     MatrixCoefficient *MQ = dif->GetMatrixCoefficient();
                     if (!MQ)
                     {
                        if (Q) { wn *= Q->Eval(*ElTr, eip); }
                        ni.Set(wn, nor);
                     }
                     else
                     {
                        nh.Set(wn, nor);
                        MQ->Eval(mq, *ElTr, eip);
                        mq.MultTranspose(nh, ni);
                     }
                     const real_t wq = ni * nor;
                     const real_t un_raw = un;
                     real_t a, b;
                     if (un != 0.)
                     {
                        const real_t su = un / std::fabs(un);
                        a = 0.5 * dif->GetAlpha() * su;
                        b = dif->GetBeta() * std::fabs(un);
                     }
                     else { a = 0.; b = dif->GetBeta(); }

                     // The scalar state at the point, which the stabilization
                     // is a function of -- this is where a residual differs
                     // from a Jacobian by more than a rank.
                     // An explicit dot and NOT `s * Vector(el_mat.GetData(),
                     // ND)`: Vector has a move assignment, so a prvalue binds
                     // to it and ALIASES the pointer -- a trap this branch
                     // has already paid for with a silently corrupted
                     // residual. Read-only here, but not worth writing in a
                     // form that has to be argued safe.
                     real_t v_q = 0., t_q = 0.;
                     for (int i = 0; i < ND; i++)
                     { v_q += s(i) * el_mat.GetData()[i]; }
                     for (int i = 0; i < TRD; i++)
                     { t_q += tq(i) * tr_mat.GetData()[i]; }
                     const real_t face_w = ip.weight * nor.Norml2();
                     const real_t w = dif->EvalStabilization(
                                         wq, b + a, un_raw, face_w, v_q, t_q, *ElTr);
                     // AssembleHDGFaceVector() adds w*v_q against both shapes
                     // and subtracts w*t_q against both, so the weight is one
                     // scalar times the jump.
                     const real_t wj = w * (v_q - t_q);
                     we[0] += wj;
                     wt[0] += wj;
                  }
                  else if (kind == HDGNLGradKind::MixedConduction)
                  {
                     // Deliberately not implemented, and refused rather than
                     // approximated. MixedConductionNLFIntegrator is
                     // unreachable as a hybridized face constraint from every
                     // miniapp -- established by printing which slot
                     // EnableHybridization() fills across all 152 serial
                     // references -- so a kernel for it would be untested
                     // code for a slot nothing fills. See the plan.
                     MFEM_ABORT("the batched face residual does not implement "
                                "MixedConductionNLFIntegrator; "
                                "HDGNLFaceResidualCanBatch() should have "
                                "refused it");
                  }
                  MFEM_CONTRACT_VAR(mix);

                  for (int i = 0; i < ND; i++)
                  { psel[q + NQ * (i + ND * p)] = s(i); }
               }
            }
         }
      }

      // ---- the contraction. One thread per PAIR, the rank-one accumulation
      // by inner loops: NQ*neq*(ND+TRD) per pair, and no index decomposition
      // beyond the pair itself.
      {
         const auto d_we = wel.Read(), d_wt = wtr.Read();
         const auto d_se = sel.Read(), d_st = str.Read();
         auto d_re = r_el.ReadWrite();
         auto d_rt = r_tr.ReadWrite();
         const int nq = NQ, nd_ = ND, trd = TRD, ne = neq;
         const int ldd = LDD, ldc = LDC;
         mfem::forall(NP, [=] MFEM_HOST_DEVICE (int p)
         {
            for (int c = 0; c < ne; c++)
            {
               for (int q = 0; q < nq; q++)
               {
                  const real_t we = d_we[q + nq * (c + ne * p)];
                  const real_t wt = d_wt[q + nq * (c + ne * p)];
                  if (we != 0.0)
                  {
                     for (int i = 0; i < nd_; i++)
                     { d_re[p * ldd + c * nd_ + i] += we * d_se[q + nq * (i + nd_ * p)]; }
                  }
                  if (wt != 0.0)
                  {
                     for (int j = 0; j < trd; j++)
                     { d_rt[p * ldc + c * trd + j] += wt * d_st[q + nq * j]; }
                  }
               }
            }
         });
      }
   }
}

void HDGNLFaceGradScatterBatched(
   const FiniteElementSpace &tr_fes, const FiniteElementSpace &el_fes,
   const FiniteElementSpace *fl_fes,
   const Array<NonlinearFormIntegrator*> &integs,
   const Array<BlockNonlinearFormIntegrator*> &bintegs,
   const Array<int> &face_list,
   const Vector &el_state, const Vector &tr_state,
   const Array<int> &D_off, const Array<int> &E_off,
   const Array<int> &G_off, const Array<int> &H_off,
   Vector &Df_data, Vector &E_data, Vector &G_data, Vector &H_data,
   ParMesh *pmesh)
{
   MFEM_VERIFY(HDGNLFaceGradCanBatch(tr_fes, el_fes, fl_fes, integs, bintegs,
                                     face_list, pmesh),
               "these integrators do not admit the batched face gradient");

   Mesh *mesh = el_fes.GetMesh();
   const HDGFacePairSource src(mesh, pmesh);
   const int ns = src.NumSides();
   const int NF = face_list.Size();
   if (NF == 0) { return; }
   const int NP = ns * NF;
   const int neq = el_fes.GetVDim();

   int ND = 0, TRD = 0;
   HDGNLFaceGradGeometryOK(tr_fes, el_fes, face_list, ND, TRD);
   const int LDD = neq * ND, LDC = neq * TRD;

   MFEM_ASSERT(el_state.Size() == NP * LDD && tr_state.Size() == NP * LDC &&
               D_off.Size() == NP && E_off.Size() == NP &&
               G_off.Size() == NP && H_off.Size() == NP, "");

   const int *d_Do = D_off.Read(), *d_Eo = E_off.Read();
   const int *d_Go = G_off.Read(), *d_Ho = H_off.Read();

   {
      // E and G hold one block per (face, side) and are not reset between
      // gradient evaluations, so they are cleared once here and every
      // integrator's pass then accumulates. That IS ConstructGrad()'s
      // eg_written rule -- clear on the first writer of a face and side --
      // without the flag, and getting it wrong either way has been paid for:
      // accumulating unconditionally made GetGradient() depend on how many
      // times it had been called, overwriting kept only the last integrator.
      real_t *d_E = E_data.ReadWrite();
      real_t *d_G = G_data.ReadWrite();
      const int ldd = LDD, ldc = LDC;
      mfem::forall(NP, [=] MFEM_HOST_DEVICE (int p)
      {
         const int e0 = d_Eo[p], g0 = d_Go[p];
         for (int j = 0; j < ldc; j++)
            for (int i = 0; i < ldd; i++)
            { d_E[e0 + i + ldd * j] = 0.0; }
         for (int j = 0; j < ldd; j++)
            for (int i = 0; i < ldc; i++)
            { d_G[g0 + i + ldc * j] = 0.0; }
      });
   }

   const int dim = mesh->Dimension();
   const int NI = integs.Size() + bintegs.Size();

   for (int k = 0; k < NI; k++)
   {
      NonlinearFormIntegrator *nlfi = (k < integs.Size()) ? integs[k] : NULL;
      BlockNonlinearFormIntegrator *bnlfi =
         nlfi ? NULL : bintegs[k - integs.Size()];
      const HDGNLGradKind kind = nlfi ? HDGNLGradKindOf(nlfi)
                                 : HDGNLGradKindOf(bnlfi);

      FaceElementTransformations *ftr0 = src.Get(face_list[0]);
      const FiniteElement &tr_fe0 = *tr_fes.GetFaceElement(face_list[0]);
      const IntegrationRule &ir = nlfi
                                  ? *HDGNLGradRule(nlfi, tr_fe0, *el_fes.GetFE(ftr0->Elem1No),
                                                   fl_fes ? fl_fes->GetFE(ftr0->Elem1No) : NULL, *ftr0)
                                  : *HDGNLGradRule(bnlfi, tr_fe0, *el_fes.GetFE(ftr0->Elem1No),
                                                   fl_fes ? fl_fes->GetFE(ftr0->Elem1No) : NULL, *ftr0);
      const int NQ = ir.GetNPoints();

      // Two weight matrices per (point, pair) and the two shape tables. Sized
      // AND ZEROED: Vector(int) does not initialise and a face whose weight
      // is exactly zero would otherwise carry whatever was in the heap -- a
      // defect already paid for in the boundary kernel, where it read as a
      // plausible face integral with the right sparsity and a 46% wrong
      // answer.
      Vector Wel(NQ * neq * neq * NP), Wtr(NQ * neq * neq * NP);
      Vector sel(NQ * ND * NP), str(NQ * TRD);
      Wel = 0.; Wtr = 0.; sel = 0.; str = 0.;
      Wel.UseDevice(true); Wtr.UseDevice(true);
      sel.UseDevice(true); str.UseDevice(true);

      {
         Vector t(TRD);
         real_t *ps = str.HostWrite();
         for (int q = 0; q < NQ; q++)
         {
            tr_fe0.CalcShape(ir.IntPoint(q), t);
            for (int i = 0; i < TRD; i++) { ps[q + i * NQ] = t(i); }
         }
      }

      {
         auto *dif = dynamic_cast<const HDGDiffusionIntegrator*>(nlfi);
         auto *hyp = dynamic_cast<const HyperbolicFormIntegrator*>(nlfi);
         auto *mix = dynamic_cast<const MixedConductionNLFIntegrator*>(bnlfi);

         Vector nor(dim), vu(dim), nh(dim), ni(dim), s(ND);
         Vector st_el(neq), st_tr(neq), pv(1), tq(TRD);
         DenseMatrix mq(dim), JDotN(neq), J_u, J_F, uzero(1, dim);
         DenseMatrixInverse J_Fi;
         uzero = 0.;

         real_t *pWel = Wel.HostWrite(), *pWtr = Wtr.HostWrite();
         real_t *psel = sel.HostWrite();
         const real_t *pes = el_state.HostRead(), *pts = tr_state.HostRead();

         for (int fi = 0; fi < NF; fi++)
         {
            FaceElementTransformations *ftr = src.Get(face_list[fi]);
            // Elem2No on a SHARED face is the neighbour's shifted index, not
            // an el_fes element; ns == 1 is what keeps it unread.
            const int els[2] = { ftr->Elem1No, (ns == 2) ? ftr->Elem2No : -1 };

            for (int side = 0; side < ns; side++)
            {
               const int p = ns * fi + side;
               const FiniteElement &el_fe = *el_fes.GetFE(els[side]);
               const DenseMatrix el_mat(const_cast<real_t*>(pes + p * LDD),
                                        ND, neq);
               const DenseMatrix tr_mat(const_cast<real_t*>(pts + p * LDC),
                                        TRD, neq);

               for (int q = 0; q < NQ; q++)
               {
                  const IntegrationPoint &ip = ir.IntPoint(q);
                  ftr->SetAllIntPoints(&ip);
                  const IntegrationPoint &eip1 = ftr->GetElement1IntPoint();
                  const IntegrationPoint &eip2 = ftr->GetElement2IntPoint();
                  const IntegrationPoint &eip = side ? eip2 : eip1;
                  ElementTransformation *ElTr = side ? ftr->Elem2 : ftr->Elem1;

                  if (dim == 1) { nor(0) = 2 * eip1.x - 1.0; }
                  else { CalcOrtho(ftr->Jacobian(), nor); }

                  // The trace shape table is (NQ x TRD) with stride NQ in the
                  // dof, so a row is not contiguous; copy it out rather than
                  // aliasing a Vector onto it. Vector has a move assignment,
                  // so a prvalue on the right ALIASES the pointer rather than
                  // copying -- which has corrupted a stored residual on this
                  // branch before.
                  for (int i = 0; i < TRD; i++)
                  { tq(i) = str.HostRead()[q + i * NQ]; }

                  real_t *wel = pWel + (q + NQ * neq * neq * p);
                  real_t *wtr = pWtr + (q + NQ * neq * neq * p);
                  const int wstride = NQ;

                  if (kind == HDGNLGradKind::Hyperbolic)
                  {
                     // fe.CalcShape at the SIDE's element integration point;
                     // the element basis belongs to the side being assembled.
                     el_fe.CalcShape(eip, s);
                     if (side) { nor.Neg(); }
                     el_mat.MultTranspose(s, st_el);
                     tr_mat.MultTranspose(tq, st_tr);
                     const real_t w = -ip.weight * hyp->GetSign();
                     const NumericalFlux &nf = hyp->GetNumericalFlux();
                     nf.AverageGrad(2, st_tr, st_el, nor, *ftr, JDotN);
                     for (int dj = 0; dj < neq; dj++)
                        for (int di = 0; di < neq; di++)
                        { wel[wstride * (di + neq * dj)] += w * JDotN(di, dj); }
                     nf.AverageGrad(1, st_tr, st_el, nor, *ftr, JDotN);
                     for (int dj = 0; dj < neq; dj++)
                        for (int di = 0; di < neq; di++)
                        { wtr[wstride * (di + neq * dj)] += w * JDotN(di, dj); }
                  }
                  else if (kind == HDGNLGradKind::Diffusion)
                  {
                     if (side) { nor.Neg(); }
                     real_t un = 0.;
                     if (VectorCoefficient *v = dif->GetVelocity())
                     { v->Eval(vu, *ftr->Elem1, eip1); un = vu * nor; }
                     el_fe.CalcPhysShape(*ElTr, s);
                     real_t wn = ip.weight / ElTr->Weight();
                     Coefficient *Q = dif->GetCoefficient();
                     MatrixCoefficient *MQ = dif->GetMatrixCoefficient();
                     if (!MQ)
                     {
                        if (Q) { wn *= Q->Eval(*ElTr, eip); }
                        ni.Set(wn, nor);
                     }
                     else
                     {
                        nh.Set(wn, nor);
                        MQ->Eval(mq, *ElTr, eip);
                        mq.MultTranspose(nh, ni);
                     }
                     const real_t wq = ni * nor;
                     const real_t un_raw = un;
                     real_t a, b;
                     if (un != 0.)
                     {
                        un /= std::fabs(un);
                        a = 0.5 * dif->GetAlpha() * un;
                        b = dif->GetBeta() * std::fabs(un);
                     }
                     else { a = 0.; b = dif->GetBeta(); }
                     const real_t face_w = ip.weight * nor.Norml2();
                     const real_t w = dif->EvalStabilization(
                                         wq, b + a, un_raw, face_w, 0., 0., *ElTr);
                     wel[0] += w;
                     wtr[0] -= w;
                  }
                  else // MixedConduction
                  {
                     real_t un = 0.;
                     if (VectorCoefficient *v = mix->GetVelocity())
                     { v->Eval(vu, *ftr->Elem1, eip1); un = vu * nor; }
                     if (side) { un *= -1.; }
                     el_fe.CalcPhysShape(*ElTr, s);
                     const real_t wn = ip.weight / ElTr->Weight();
                     real_t wq = 0.;
                     if (neq == 1)
                     {
                        el_mat.MultTranspose(s, st_el);
                        pv(0) = st_el(0);
                        nh.Set(wn, nor);
                        mix->GetFluxFunction().ComputeDualFluxJacobian(
                           pv, uzero, *ftr, J_u, J_F);
                        J_Fi.Factor(J_F);
                        J_Fi.Mult(nh, ni);
                        wq = ni * nor;
                     }
                     // The commented-out upwinding in the integrator: a = 0
                     // and b = beta, whatever the velocity. Kept identical
                     // here -- the residual defines the operator.
                     const real_t a = 0., b = mix->GetBeta();
                     const real_t face_w = ip.weight * nor.Norml2();
                     for (int e = 0; e < neq; e++)
                     {
                        const real_t w_e = (neq == 1)
                                           ? (wq * (b + a))
                                           : (face_w * mix->GetVariableStabilization(e) * (b + a));
                        wel[wstride * (e + neq * e)] += w_e;
                        wtr[wstride * (e + neq * e)] -= w_e;
                     }
                  }

                  for (int i = 0; i < ND; i++)
                  { psel[q + NQ * (i + ND * p)] = s(i); }
               }
            }
         }
      }

      const auto d_Wel = Reshape(Wel.Read(), NQ, neq * neq, NP);
      const auto d_Wtr = Reshape(Wtr.Read(), NQ, neq * neq, NP);
      const auto d_sel = Reshape(sel.Read(), NQ, ND, NP);
      const auto d_str = Reshape(str.Read(), NQ, TRD);

      real_t *d_D = Df_data.ReadWrite();
      real_t *d_E = E_data.ReadWrite();
      real_t *d_G = G_data.ReadWrite();
      real_t *d_H = H_data.ReadWrite();

      const int nd = ND, trd = TRD, nq = NQ, ne = neq;
      const int ldd = LDD, ldc = LDC;

      // ONE THREAD PER FACE, both sides inside, and the inner indices are
      // loops rather than a decomposed flat index. Both halves of that are
      // deliberate: a thread per output entry needs four integer divisions to
      // recover (i, j, point, pair) and that arithmetic was 0.37 s of a 0.95 s
      // loop in ComputeElementsHBatched()'s first version; and a thread per
      // FACE rather than per PAIR makes H exclusive, since only the two sides
      // of one face write it.
      const int nsides = ns;
      mfem::forall(NF, [=] MFEM_HOST_DEVICE (int f)
      {
         for (int side = 0; side < nsides; side++)
         {
            const int p = nsides * f + side;
            const int dO = d_Do[p], eO = d_Eo[p];
            const int gO = d_Go[p], hO = d_Ho[p];

            for (int q = 0; q < nq; q++)
               for (int dj = 0; dj < ne; dj++)
                  for (int di = 0; di < ne; di++)
                  {
                     const real_t wel = d_Wel(q, di + ne * dj, p);
                     const real_t wtr = d_Wtr(q, di + ne * dj, p);
                     if (wel == 0.0 && wtr == 0.0) { continue; }

                     for (int i = 0; i < nd; i++)
                     {
                        const real_t si = d_sel(q, i, p);
                        // D is the one block two faces of an element collide
                        // on, so it is the one that needs an atomic.
                        for (int j = 0; j < nd; j++)
                        {
                           AtomicAdd(d_D[dO + (di * nd + i) + ldd * (dj * nd + j)],
                                     wel * si * d_sel(q, j, p));
                        }
                        for (int j = 0; j < trd; j++)
                        {
                           d_E[eO + (di * nd + i) + ldd * (dj * trd + j)] +=
                              wtr * si * d_str(q, j);
                        }
                     }
                     for (int i = 0; i < trd; i++)
                     {
                        const real_t ti = d_str(q, i);
                        for (int j = 0; j < nd; j++)
                        {
                           d_G[gO + (di * trd + i) + ldc * (dj * nd + j)] +=
                              wel * ti * d_sel(q, j, p);
                        }
                        for (int j = 0; j < trd; j++)
                        {
                           d_H[hO + (di * trd + i) + ldc * (dj * trd + j)] +=
                              wtr * ti * d_str(q, j);
                        }
                     }
                  }
         }
      });
   }
}


const Array<int> *GetRestrictedFluxComponents(const BilinearFormIntegrator *bfi)
{
   if (auto *d = dynamic_cast<const RestrictedVectorDivergenceIntegrator*>(bfi))
   {
      return &d->GetFluxComponents();
   }
   if (auto *c = dynamic_cast<const RestrictedNormalTraceJumpIntegrator*>(bfi))
   {
      return &c->GetFluxComponents();
   }
   return NULL;
}

void CheckFluxComponents(const Array<int> &comps, int sdim, const char *what)
{
   for (int c = 0; c < comps.Size(); c++)
   {
      MFEM_VERIFY(comps[c] >= 0 && comps[c] < sdim,
                  what << ": flux component " << c << " names direction "
                  << comps[c] << ", which is not in [0, " << sdim << ")");
      MFEM_VERIFY(c == 0 || comps[c] > comps[c-1],
                  what << ": the flux components must be strictly increasing, "
                  "the order being the layout of the flux space itself; got "
                  << comps[c-1] << " then " << comps[c]);
   }
   MFEM_VERIFY(comps.Size() <= sdim,
               what << ": " << comps.Size() << " flux components in "
               << sdim << " dimensions");
}

void RestrictedVectorDivergenceIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   ElementTransformation &Trans, DenseMatrix &elmat)
{
#ifdef MFEM_THREAD_SAFE
   DenseMatrix full;
#endif
   MFEM_VERIFY(trial_fe.GetRangeType() == FiniteElement::SCALAR,
               "RestrictedVectorDivergenceIntegrator needs a SCALAR-range flux "
               "space -- one scalar basis per direction. An H(div) element's "
               "components are intrinsic and there is no column block to drop; "
               "use VectorFEDivergenceIntegrator and a full flux.");
   const int sdim = Trans.GetSpaceDim();
   CheckFluxComponents(comps, sdim, "RestrictedVectorDivergenceIntegrator");

   base.AssembleElementMatrix2(trial_fe, test_fe, Trans, full);

   const int nd = trial_fe.GetDof();
   const int m = comps.Size();
   MFEM_ASSERT(full.Width() == sdim * nd, "the base block is not sdim wide");
   elmat.SetSize(full.Height(), m * nd);
   for (int c = 0; c < m; c++)
   {
      const int off = comps[c] * nd;
      for (int j = 0; j < nd; j++)
         for (int i = 0; i < full.Height(); i++)
         {
            elmat(i, c * nd + j) = full(i, off + j);
         }
   }
}

void RestrictedNormalTraceJumpIntegrator::AssembleFaceMatrix(
   const FiniteElement &trial_face_fe, const FiniteElement &test_fe1,
   const FiniteElement &test_fe2, FaceElementTransformations &Trans,
   DenseMatrix &elmat)
{
#ifdef MFEM_THREAD_SAFE
   DenseMatrix full;
#endif
   MFEM_VERIFY(test_fe1.GetRangeType() == FiniteElement::SCALAR,
               "RestrictedNormalTraceJumpIntegrator needs a SCALAR-range flux "
               "space -- one scalar basis per direction. An H(div) element's "
               "components are intrinsic and there are no rows to drop; use "
               "NormalTraceJumpIntegrator and a full flux.");
   // test_fe1.GetDim() and not Trans.GetSpaceDim(): the base indexes its rows
   // by the ELEMENT's dimension, which is what its own `const int dim =
   // test_fe1.GetDim()` says, so the slice has to agree with that and not with
   // the space dimension it may differ from on a surface mesh.
   const int dim = test_fe1.GetDim();
   CheckFluxComponents(comps, dim, "RestrictedNormalTraceJumpIntegrator");

   base.AssembleFaceMatrix(trial_face_fe, test_fe1, test_fe2, Trans, full);

   const int nd1 = test_fe1.GetDof();
   const int nd2 = (Trans.Elem2No >= 0) ? test_fe2.GetDof() : 0;
   const int nf = trial_face_fe.GetDof();
   const int m = comps.Size();
   MFEM_ASSERT(full.Height() == (nd1 + nd2) * dim,
               "the base block is not dim tall");
   elmat.SetSize((nd1 + nd2) * m, nf);
   elmat = 0.0;
   for (int c = 0; c < m; c++)
   {
      const int off1 = comps[c] * nd1;
      for (int i = 0; i < nd1; i++)
         for (int j = 0; j < nf; j++)
         {
            elmat(c * nd1 + i, j) = full(off1 + i, j);
         }
      // Element 2 sits at dim*nd1 in the base and at m*nd1 here, which is the
      // whole reason the unrestricted block cannot simply be truncated.
      const int off2 = dim * nd1 + comps[c] * nd2;
      for (int i = 0; i < nd2; i++)
         for (int j = 0; j < nf; j++)
         {
            elmat(m * nd1 + c * nd2 + i, j) = full(off2 + i, j);
         }
   }
}

void RestrictedVectorBoundaryFluxLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   MFEM_CONTRACT_VAR(el);
   MFEM_CONTRACT_VAR(Tr);
   MFEM_CONTRACT_VAR(elvect);
   MFEM_ABORT("RestrictedVectorBoundaryFluxLFIntegrator carries a normal, so "
              "it belongs on the FACES: use "
              "LinearForm::AddBdrFaceIntegrator() and not "
              "AddBoundaryIntegrator().");
}

void RestrictedVectorBoundaryFluxLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
#ifdef MFEM_THREAD_SAFE
   Vector full;
#endif
   MFEM_VERIFY(el.GetRangeType() == FiniteElement::SCALAR,
               "RestrictedVectorBoundaryFluxLFIntegrator needs a SCALAR-range "
               "flux space -- one scalar basis per direction. An H(div) "
               "element's components are intrinsic and there is nothing to "
               "slice; use VectorFEBoundaryFluxLFIntegrator and a full flux.");
   // el.GetDim() and not Tr.GetSpaceDim(), for the reason
   // RestrictedNormalTraceJumpIntegrator::AssembleFaceMatrix() gives: the base
   // indexes by the ELEMENT dimension and the slice has to agree with it.
   const int dim = el.GetDim();
   CheckFluxComponents(comps, dim, "RestrictedVectorBoundaryFluxLFIntegrator");

   base.AssembleRHSElementVect(el, Tr, full);

   const int dof = el.GetDof();
   const int m = comps.Size();
   MFEM_ASSERT(full.Size() == dim * dof, "the base vector is not dim tall");
   elvect.SetSize(m * dof);
   for (int c = 0; c < m; c++)
   {
      const int off = comps[c] * dof;
      for (int j = 0; j < dof; j++)
      {
         elvect(c * dof + j) = full(off + j);
      }
   }
}

void CheckRestrictedFluxLoad(LinearForm *b_u, int vdim, int sdim)
{
   if (!b_u || vdim >= sdim) { return; }

   // The boundary FACE list is the one a DG flux space uses; the boundary
   // ELEMENT list cannot carry a normal-trace datum at all and the class
   // above refuses it by name, so there is nothing to check there.
   const Array<LinearFormIntegrator*> *bfli = b_u->GetBFLFI();
   if (!bfli) { return; }
   for (int i = 0; i < bfli->Size(); i++)
   {
      MFEM_VERIFY(!dynamic_cast<VectorBoundaryFluxLFIntegrator*>((*bfli)[i]),
                  "the flux space carries " << vdim << " of " << sdim
                  << " directions, and VectorBoundaryFluxLFIntegrator sizes "
                  "its element vector from the MESH dimension: the load would "
                  "be read into the wrong component and no size check would "
                  "notice. Use RestrictedVectorBoundaryFluxLFIntegrator with "
                  "the same component list the divergence and the constraint "
                  "carry.");
   }
}

// ---------------------------------------------------------------------------
// The ONE-SIDED face kernels: NormalTraceJumpIntegrator's constraint block
// and HDGDiffusionIntegrator's one-sided HDG matrix, per (face, side) item.
// ---------------------------------------------------------------------------

namespace
{

/** @brief One element geometry, one order, and nodes (if any) of one order:
    what lets one integration rule and one shape table per (local face,
    orientation) slot serve every face of the mesh.

    Asked of the spaces and the mesh rather than face by face, so it builds no
    transformation. The face rules these kernels use take their OrderW() from
    the face's transformation, and with nodes of one order that is one number
    for every face of one geometry. */
bool HDGUniformMesh(const Mesh &mesh, const FiniteElementSpace &el_fes)
{
   if (mesh.GetNE() == 0 || mesh.Nonconforming()) { return false; }
   const int dim = mesh.Dimension();
   if (dim < 2 || mesh.SpaceDimension() != dim) { return false; }
   if (mesh.GetNumGeometries(dim) != 1) { return false; }
   if (el_fes.IsVariableOrder()) { return false; }
   const GridFunction *nodes = mesh.GetNodes();
   if (nodes && nodes->FESpace()->IsVariableOrder()) { return false; }
   return true;
}

/// The rule NormalTraceJumpIntegrator::AssembleFaceMatrix() takes on @a f.
const IntegrationRule &HDGNormalTraceJumpRule(const BilinearFormIntegrator &bfi,
                                              const FiniteElementSpace &tr_fes,
                                              const FiniteElementSpace &el_fes,
                                              Mesh &mesh, int f)
{
   if (const IntegrationRule *ir = bfi.GetIntRule()) { return *ir; }
   FaceElementTransformations *ftr = mesh.GetFaceElementTransformations(f);
   // max(o1, o2) on an interior face and o1 on a boundary one, which is the
   // same number at one order.
   const int order = el_fes.GetFE(ftr->Elem1No)->GetOrder()
                     + tr_fes.GetFaceElement(f)->GetOrder() + ftr->OrderW();
   return IntRules.Get(ftr->GetGeometryType(), order);
}

} // namespace

bool HDGNormalTraceJumpCanBatch(const FiniteElementSpace &tr_fes,
                                const FiniteElementSpace &el_fes,
                                const BilinearFormIntegrator *bfi)
{
   // The integrator itself, not anything deriving from it: a subclass may
   // assemble something else under the same AssembleFaceMatrix().
   if (!bfi || typeid(*bfi) != typeid(NormalTraceJumpIntegrator))
   { return false; }

   const Mesh *mesh = el_fes.GetMesh();
   if (!mesh || !HDGUniformMesh(*mesh, el_fes)) { return false; }
   const int dim = mesh->Dimension();
   if (el_fes.GetVDim() != dim || tr_fes.GetVDim() != 1) { return false; }
   if (!dynamic_cast<const DG_Interface_FECollection*>(tr_fes.FEColl()))
   { return false; }
   const FiniteElement *fe = el_fes.GetFE(0);
   return fe->GetRangeType() == FiniteElement::SCALAR
          && fe->GetMapType() == FiniteElement::VALUE;
}

void HDGNormalTraceJumpBatched(const FiniteElementSpace &tr_fes,
                               const FiniteElementSpace &el_fes,
                               const BilinearFormIntegrator &bfi,
                               const Array<int> &faces,
                               const Array<int> &sides,
                               Vector &blocks)
{
   MFEM_ASSERT(HDGNormalTraceJumpCanBatch(tr_fes, el_fes, &bfi),
               "this integrator does not admit the batched constraint");
   MFEM_VERIFY(faces.Size() == sides.Size(), "one side per face item");

   Mesh &mesh = *el_fes.GetMesh();
   const int NI = faces.Size();
   const int dim = mesh.Dimension();
   const int ND = el_fes.GetFE(0)->GetDof();
   const int NR = ND * dim;
   blocks.UseDevice(true);
   if (NI == 0) { blocks.SetSize(0); return; }

   const FiniteElement &tr_fe = *tr_fes.GetFaceElement(faces[0]);
   const int TRD = tr_fe.GetDof();
   blocks.SetSize(NR * TRD * NI);

   const IntegrationRule &ir =
      HDGNormalTraceJumpRule(bfi, tr_fes, el_fes, mesh, faces[0]);
   const int NQ = ir.GetNPoints();

   HDGSideTables tab;
   HDGBuildSideTables(mesh, el_fes, faces, sides, ir, tab);

   // The trace shape scaled by `ip.weight * sign` -- the integrator's own
   // `face_shape *= ip.weight * sign`, done once rather than per face.
   const real_t sign =
      static_cast<const NormalTraceJumpIntegrator&>(bfi).GetSign();
   Vector ftw(NQ * TRD);
   ftw.UseDevice(true);
   {
      Vector fs(TRD);
      real_t *p = ftw.HostWrite();
      for (int q = 0; q < NQ; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         tr_fe.CalcShape(ip, fs);
         fs *= ip.weight * sign;
         for (int j = 0; j < TRD; j++) { p[q + NQ * j] = fs(j); }
      }
   }

   // The normals: CalcOrtho() of the face transformation's Jacobian, as
   // the integrator takes it, side 2's negated so that both sides are
   // written with a plus. -(a*b*c) and a*b*(-c) are the same number in IEEE
   // arithmetic, so the subtraction the integrator does for side 2 is
   // reproduced exactly. On the device from the faces' own point matrices
   // where HDGFaceGeometryOnDevice() admits the mesh -- the same arithmetic
   // in the same order, so the same bits on the host -- and otherwise one
   // face transformation per face on the host.
   Vector nor(dim * NQ * NI);
   nor.UseDevice(true);
   Array<int> slot(tab.slot);
   if (HDGFaceGeometryOnDevice(mesh))
   {
      Array<int> item_face(NI), neg(NI);
      for (int k = 0; k < NI; k++)
      {
         item_face[k] = k;
         neg[k] = sides[k] ? 1 : 0;
      }
      const HDGSideMaps maps(mesh, ir, tab.eip, faces, false);
      HDGFaceNormalsDevice(maps, item_face, neg, nor);
   }
   else
   {
      real_t *pn = nor.HostWrite();
      IsoparametricTransformation ftr;
      Vector n(dim);
      int f_last = -1;
      // An affine face has one normal; see HDGAffine().
      const bool affine = HDGAffine(mesh, tr_fe.GetGeomType());
      for (int k = 0; k < NI; k++)
      {
         if (faces[k] != f_last)
         {
            mesh.GetFaceTransformation(faces[k], &ftr);
            f_last = faces[k];
         }
         for (int q = 0; q < NQ; q++)
         {
            if (q == 0 || !affine)
            {
               ftr.SetIntPoint(&ir.IntPoint(q));
               CalcOrtho(ftr.Jacobian(), n);
               if (sides[k]) { n.Neg(); }
            }
            for (int d = 0; d < dim; d++) { pn[d + dim * (q + NQ * k)] = n(d); }
         }
      }
   }

   const auto d_s = Reshape(tab.sref.Read(), NQ, ND, tab.nslot);
   const auto d_t = Reshape(ftw.Read(), NQ, TRD);
   const auto d_n = Reshape(nor.Read(), dim, NQ, NI);
   const int *d_slot = slot.Read();
   real_t *d_B = blocks.Write();
   const int nq = NQ, nd = ND, trd = TRD, nr = NR, sdim = dim;

   // One thread per entry, summed over the points in a register in the
   // integrator's order -- `+= shape * face_shape * normal` per point -- so
   // each entry is the integrator's number.
   mfem::forall(NI * nr * trd, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int r = idx % nr, j = (idx / nr) % trd, k = idx / (nr * trd);
      const int d = r / nd, i = r % nd;
      const int c = d_slot[k];
      real_t sum = 0.0;
      for (int q = 0; q < nq; q++)
      {
         sum += d_s(q, i, c) * d_t(q, j) * d_n(d, q, k);
      }
      d_B[r + nr * j + nr * trd * k] = sum;
   });
   (void)sdim;
}

bool HDGFaceSideCanBatch(const FiniteElementSpace &tr_fes,
                         const FiniteElementSpace &el_fes,
                         const Array<BilinearFormIntegrator*> &integs)
{
   if (integs.Size() == 0) { return false; }
   const Mesh *mesh = el_fes.GetMesh();
   if (!mesh || !HDGUniformMesh(*mesh, el_fes)) { return false; }
   if (el_fes.GetVDim() != 1 || tr_fes.GetVDim() != 1) { return false; }
   if (!dynamic_cast<const DG_Interface_FECollection*>(tr_fes.FEColl()))
   { return false; }
   const FiniteElement *fe = el_fes.GetFE(0);
   if (fe->GetRangeType() != FiniteElement::SCALAR
       || fe->GetMapType() != FiniteElement::VALUE) { return false; }
   for (const BilinearFormIntegrator *bfi : integs)
   {
      auto *d = dynamic_cast<const HDGDiffusionIntegrator*>(bfi);
      if (!d) { return false; }
      const HDGStabilization *st = d->GetStabilization();
      if (st && !st->IsConstant()) { return false; }
   }
   return true;
}

void HDGFaceSideBatched(const FiniteElementSpace &tr_fes,
                        const FiniteElementSpace &el_fes,
                        const Array<BilinearFormIntegrator*> &integs,
                        const Array<int> &faces,
                        const Array<int> &sides,
                        Vector &blocks)
{
   MFEM_ASSERT(HDGFaceSideCanBatch(tr_fes, el_fes, integs),
               "these integrators do not admit the batched one-sided matrix");
   MFEM_VERIFY(faces.Size() == sides.Size(), "one side per face item");

   Mesh &mesh = *el_fes.GetMesh();
   const int NI = faces.Size();
   const int dim = mesh.Dimension();
   const int ND = el_fes.GetFE(0)->GetDof();
   blocks.UseDevice(true);
   if (NI == 0) { blocks.SetSize(0); return; }

   const FiniteElement &tr_fe = *tr_fes.GetFaceElement(faces[0]);
   const int TRD = tr_fe.GetDof();
   const int N = ND + TRD;
   blocks.SetSize(N * N * NI);

   Vector pass;
   pass.UseDevice(true);
   for (int ii = 0; ii < integs.Size(); ii++)
   {
      auto *dif = static_cast<const HDGDiffusionIntegrator*>(integs[ii]);

      // The one-sided rule, asked of the integrator on the first item; one
      // order and one geometry make it every item's.
      const IntegrationRule *ir = dif->GetIntRule();
      if (!ir)
      {
         FaceElementTransformations *ftr0 =
            mesh.GetFaceElementTransformations(faces[0]);
         const FiniteElement &el0 =
            *el_fes.GetFE(sides[0] ? ftr0->Elem2No : ftr0->Elem1No);
         ir = &dif->GetHDGFaceIntRule(tr_fe, el0, *ftr0);
      }
      const int NQ = ir->GetNPoints();

      HDGSideTables tab;
      HDGBuildSideTables(mesh, el_fes, faces, sides, *ir, tab);

      Vector trs(NQ * TRD);
      trs.UseDevice(true);
      {
         Vector t(TRD);
         real_t *p = trs.HostWrite();
         for (int q = 0; q < NQ; q++)
         {
            tr_fe.CalcShape(ir->IntPoint(q), t);
            for (int j = 0; j < TRD; j++) { p[q + NQ * j] = t(j); }
         }
      }

      // The weights: on the device from the element maps where
      // HDGFaceWeightsOnDevice() admits the integrator, and otherwise on the
      // host, HDGDiffusionIntegrator's one-sided AssembleHDGFaceMatrix() line
      // for line up to the point where it has its w. Everything after that is
      // the kernel's.
      Vector wt(NQ * NI);
      wt.UseDevice(true);
      if (HDGFaceWeightsOnDevice(*dif, el_fes))
      {
         Array<int> item_face(NI);
         for (int k = 0; k < NI; k++) { item_face[k] = k; }
         const HDGSideMaps maps(mesh, *ir, tab.eip, faces);
         HDGFaceWeightsDevice(*dif, maps, *ir, tab.eip, item_face, tab.el,
                              tab.slot, wt);
      }
      else
      {
         real_t *pw = wt.HostWrite();
         Coefficient *Q = dif->GetCoefficient();
         MatrixCoefficient *MQ = dif->GetMatrixCoefficient();
         VectorCoefficient *V = dif->GetVelocity();
         IsoparametricTransformation ftr, etr, etr1;
         Vector nor(dim), vu(dim), nh(dim), ni(dim);
         DenseMatrix mq(dim);
         const bool face_affine = HDGAffine(mesh, tr_fe.GetGeomType());
         const bool el_affine =
            HDGAffine(mesh, el_fes.GetFE(0)->GetGeomType());
         for (int k = 0; k < NI; k++)
         {
            const int f = faces[k], side = sides[k];
            const Mesh::FaceInformation fin = mesh.GetFaceInformation(f);
            const int el = fin.element[side].index;
            mesh.GetFaceTransformation(f, &ftr);
            mesh.GetElementTransformation(el, &etr);
            const IntegrationRule &eips = tab.eip[tab.slot[k]];
            // The velocity is evaluated on side 1's element whatever the
            // side, as the integrator does.
            if (V && side != 0)
            {
               mesh.GetElementTransformation(fin.element[0].index, &etr1);
            }
            real_t el_w = 0.0;
            for (int q = 0; q < NQ; q++)
            {
               const IntegrationPoint &ip = ir->IntPoint(q);
               // The normal and the element weight are constants on an affine
               // face and element; see HDGAffine(). The element point is still
               // set for the coefficient.
               if (q == 0 || !face_affine)
               {
                  ftr.SetIntPoint(&ip);
                  CalcOrtho(ftr.Jacobian(), nor);
                  if (side != 0) { nor.Neg(); }
               }

               const IntegrationPoint &eip = eips.IntPoint(q);
               etr.SetIntPoint(&eip);
               if (q == 0 || !el_affine) { el_w = etr.Weight(); }

               real_t un = 0.;
               if (V)
               {
                  if (side != 0)
                  {
                     // Side 1's reference point at this face point, which
                     // the side-2 table does not hold.
                     FaceElementTransformations *fet =
                        mesh.GetFaceElementTransformations(f);
                     fet->SetAllIntPoints(&ip);
                     const IntegrationPoint &e1 = fet->GetElement1IntPoint();
                     etr1.SetIntPoint(&e1);
                     V->Eval(vu, etr1, e1);
                  }
                  else { V->Eval(vu, etr, eip); }
                  un = vu * nor;
               }

               real_t wn = ip.weight / el_w;
               if (!MQ)
               {
                  if (Q) { wn *= Q->Eval(etr, eip); }
                  ni.Set(wn, nor);
               }
               else
               {
                  nh.Set(wn, nor);
                  MQ->Eval(mq, etr, eip);
                  mq.MultTranspose(nh, ni);
               }
               const real_t wq = ni * nor;

               const real_t un_raw = un;
               real_t a, b;
               if (un != 0.)
               {
                  un /= std::fabs(un);
                  a = 0.5 * dif->GetAlpha() * un;
                  b = dif->GetBeta() * std::fabs(un);
               }
               else { a = 0.0; b = dif->GetBeta(); }

               const real_t face_w = ip.weight * nor.Norml2();
               pw[q + NQ * k] = dif->EvalStabilization(wq, b + a, un_raw,
                                                       face_w, 0., 0., etr);
            }
         }
      }

      Vector &out = (ii == 0) ? blocks : pass;
      out.SetSize(N * N * NI);
      Array<int> slot(tab.slot);
      const auto d_s = Reshape(tab.sref.Read(), NQ, ND, tab.nslot);
      const auto d_t = Reshape(trs.Read(), NQ, TRD);
      const auto d_w = Reshape(wt.Read(), NQ, NI);
      const int *d_slot = slot.Read();
      real_t *d_B = out.Write();
      const int nq = NQ, nd = ND, trd = TRD, n = N;

      // One thread per entry, each summed over the points in a register in
      // the association the integrator uses -- `(w * s_i) * s_j` over the
      // lower triangle, mirrored -- so each entry is the integrator's number.
      mfem::forall(NI * n * n, [=] MFEM_HOST_DEVICE (int idx)
      {
         const int r = idx % n, cl = (idx / n) % n, k = idx / (n * n);
         const int c = d_slot[k];
         real_t sum = 0.0;
         if (r < nd && cl < nd)
         {
            // D, lower triangle accumulated, upper mirrored.
            const int i = (r >= cl) ? r : cl, j = (r >= cl) ? cl : r;
            for (int q = 0; q < nq; q++)
            {
               const real_t w = d_w(q, k);
               if (w == 0.0) { continue; }
               sum += (w * d_s(q, i, c)) * d_s(q, j, c);
            }
         }
         else if (r < nd)
         {
            // E = -w s t^T.
            const int i = r, j = cl - nd;
            for (int q = 0; q < nq; q++)
            {
               const real_t w = d_w(q, k);
               if (w == 0.0) { continue; }
               sum -= (w * d_s(q, i, c)) * d_t(q, j);
            }
         }
         else if (cl < nd)
         {
            // G = -E^T, which is how the integrator completes it.
            const int i = cl, j = r - nd;
            real_t e = 0.0;
            for (int q = 0; q < nq; q++)
            {
               const real_t w = d_w(q, k);
               if (w == 0.0) { continue; }
               e -= (w * d_s(q, i, c)) * d_t(q, j);
            }
            sum = -e;
         }
         else
         {
            // H = -w t t^T, lower triangle accumulated, upper mirrored.
            const int ri = r - nd, ci = cl - nd;
            const int i = (ri >= ci) ? ri : ci, j = (ri >= ci) ? ci : ri;
            for (int q = 0; q < nq; q++)
            {
               const real_t w = d_w(q, k);
               if (w == 0.0) { continue; }
               sum -= (w * d_t(q, i)) * d_t(q, j);
            }
         }
         d_B[r + n * cl + n * n * k] = sum;
      });

      if (ii > 0) { blocks += pass; }
   }
}

}
