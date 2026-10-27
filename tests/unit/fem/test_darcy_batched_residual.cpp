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

#include "mfem.hpp"
#include "unit_tests.hpp"

#include <memory>

using namespace mfem;

namespace darcy_batched_residual
{

/// Which conductivity shape LinearDiffusionFlux is built over.
enum class Kappa { Scalar, Vector, Matrix };

/** @brief A LinearDiffusionFlux over each of the three coefficient shapes its
    ComputeDualFlux() branches on.

    Worth covering all three: the batched route reads the law through
    ComputeDualFluxJacobian(), and the equivalence of that with
    ComputeDualFlux() is a separate claim per branch -- Set() for a scalar,
    RightScaling() for a vector, MultABt() for a matrix. The matrix one is the
    only branch where the dual flux couples directions, so it is the only one
    that can catch a transposed contraction. */
struct Law
{
   std::unique_ptr<Coefficient> c;
   std::unique_ptr<VectorCoefficient> vc;
   std::unique_ptr<MatrixCoefficient> mc;
   std::unique_ptr<MixedFluxFunction> fun;

   Law(Kappa k, int dim)
   {
      switch (k)
      {
         case Kappa::Scalar:
            c.reset(new FunctionCoefficient([](const Vector &X)
            {
               return 1.3 + 0.4 * std::sin(M_PI * X(0)) * X(1);
            }));
            fun.reset(new LinearDiffusionFlux(dim, *c));
            break;
         case Kappa::Vector:
            vc.reset(new VectorFunctionCoefficient(dim,
                                                   [](const Vector &X, Vector &v)
            {
               v(0) = 0.9 + 0.2 * X(1);
               v(1) = 1.7 - 0.3 * X(0);
            }));
            fun.reset(new LinearDiffusionFlux(*vc));
            break;
         case Kappa::Matrix:
            mc.reset(new MatrixFunctionCoefficient(dim,
                                                   [](const Vector &X, DenseMatrix &m)
            {
               // Deliberately NOT symmetric: a symmetric conductivity cannot
               // tell K from K^T, and the contraction order is exactly what
               // the batched route has to get right.
               m(0,0) = 1.1 + 0.2*X(0);
               m(0,1) = 0.6 + 0.1*X(1);
               m(1,0) = -0.35;
               m(1,1) = 1.4 - 0.25*X(1);
            }));
            fun.reset(new LinearDiffusionFlux(*mc));
            break;
      }
   }
};

/// A field with structure rather than a constant, reproducibly.
void FillWavy(Vector &v, real_t shift)
{
   for (int i = 0; i < v.Size(); i++)
   {
      v(i) = std::sin(0.71 * i + shift) + 0.5 * std::cos(0.113 * i);
   }
}

} // namespace darcy_batched_residual

/** @brief The batched local residual is the integrator's own element vector,
    entry for entry.

    This is the ISOLATED comparison and the one that discriminates: it calls
    HDGMixedConductionResidualBatched() for the whole mesh and
    MixedConductionNLFIntegrator::AssembleElementVector() one element at a
    time, on the same field, and demands the same bits. Nothing else in the
    problem varies, unlike the in-situ case below where AssemblyMode::Batched
    switches several kernels at once.

    BITWISE, and the kernel is written to earn that rather than hoping for it:
    the accumulation is grouped `w * shape(i) * mF` in the integrator's own
    order, and the dual flux is contracted in the k-order the law's MultABt()
    uses. Deliberately regrouping it as `(w * mF) * shape(i)` fails this case.

    The potential field handed to the integrator is NONZERO on purpose. A
    LinearDiffusionFlux ignores its state -- that is exactly the property
    HDGMixedConductionResidualCanBatch() admits it for -- so a law that did
    not would show up here as a disagreement rather than as a silently
    different answer in a solve. */
TEST_CASE("The batched local residual reproduces the element integrator",
          "[DarcyHybridization][BatchedLinAlg][GPU]")
{
   using namespace darcy_batched_residual;

   const int order = GENERATE(0, 1, 2);
   const auto etype = GENERATE(Element::QUADRILATERAL, Element::TRIANGLE);
   const auto kappa = GENERATE(Kappa::Scalar, Kappa::Vector, Kappa::Matrix);
   CAPTURE(order, (int)etype, (int)kappa);

   const int dim = 2;
   Mesh mesh = Mesh::MakeCartesian2D(3, 4, etype, false, 0.8, 1.2);
   L2_FECollection u_coll(order, dim, BasisType::GaussLobatto);
   L2_FECollection p_coll(order, dim);
   FiniteElementSpace Vh(&mesh, &u_coll, dim), Wh(&mesh, &p_coll);

   Law law(kappa, dim);
   MixedConductionNLFIntegrator integ(*law.fun);

   REQUIRE(HDGMixedConductionResidualCanBatch(Vh, &integ));

   const int NE = mesh.GetNE();
   const int ND = Vh.GetFE(0)->GetDof();
   const int N = ND * dim;

   Vector u_all(N * NE), ru_all;
   FillWavy(u_all, 0.3);
   HDGMixedConductionResidualBatched(Vh, &integ, u_all, ru_all);
   // BOTH, and u_all is the one that was missing. The kernel READ u_all on the
   // device, which leaves its device copy valid -- and Memory::operator[],
   // which Vector::operator() reaches, asserts VALID_HOST *and not*
   // VALID_DEVICE. So indexing u_all(...) in the loop below touches a page
   // Device("debug") has mprotect-ed and dies in MmuError(); on CUDA it reads
   // the host copy, which happens to be right here, so the fault is invisible
   // there. A host read of an input is as necessary as one of an output.
   u_all.HostRead();
   ru_all.HostRead();

   const int NDP = Wh.GetFE(0)->GetDof();
   Vector p_e(NDP);
   FillWavy(p_e, 1.9);

   Vector u_e(N), out_u, out_p;
   int checked = 0;
   for (int e = 0; e < NE; e++)
   {
      for (int i = 0; i < N; i++) { u_e(i) = u_all(N * e + i); }
      Array<const FiniteElement*> fe_arr({Vh.GetFE(e), Wh.GetFE(e)});
      Array<const Vector*> x_arr({&u_e, &p_e});
      Array<Vector*> y_arr({&out_u, &out_p});
      integ.AssembleElementVector(fe_arr, *mesh.GetElementTransformation(e),
                                  x_arr, y_arr);
      REQUIRE(out_u.Size() == N);
      for (int i = 0; i < N; i++)
      {
         // Exact equality on the host, where the kernel and the integrator
         // are the same arithmetic on the same data and any difference is a
         // defect. Under a Device they are NOT: the kernel runs on the
         // device and AssembleElementVector() on the host, so the criterion
         // has to be a tolerance -- measured on an RTX 2070 SUPER, this
         // failed printing `0.137810123068043 == 0.137810123068043`, a
         // last-bit difference. The same widening RequireSame() in
         // test_darcy_batched_factor.cpp needed, and for the same reason;
         // its comment carries the argument.
         if (Device::Allows(Backend::DEVICE_MASK))
         {
            REQUIRE(ru_all(N * e + i) ==
                    MFEM_Approx(out_u(i), 1e-12, 1e-12));
         }
         else
         {
            REQUIRE(ru_all(N * e + i) == out_u(i));
         }
         checked++;
      }
   }
   // A comparison of nothing passes; say how much was compared.
   REQUIRE(checked == N * NE);
   REQUIRE(ru_all.Normlinf() > 1e-3);
}

/** @brief The gate refuses what it cannot take, and each refusal is reached.

    A guard written from reading the code is not a guard until something
    arrives at it; three of the four conditions here have a live caller in
    miniapps/hdg/regress_test/ and the fourth is the H(div) flux space. */
TEST_CASE("The batched local residual refuses what it cannot take",
          "[DarcyHybridization][BatchedLinAlg][GPU]")
{
   using namespace darcy_batched_residual;

   const int dim = 2, order = 1;
   Mesh mesh = Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL, false,
                                     0.8, 1.2);
   L2_FECollection u_coll(order, dim, BasisType::GaussLobatto);
   RT_FECollection rt_coll(order, dim);
   FiniteElementSpace Vh(&mesh, &u_coll, dim), Rh(&mesh, &rt_coll);

   SECTION("A state-dependent conductivity")
   {
      // `-p 8`: ikappa = 1/(k+T). The conductivity is a host std::function of
      // the CURRENT potential, so there is no per-point weight to compute
      // ahead of the kernel and no device form of the callback. 5 of the 88
      // hybridized references are this.
      auto ik = [](const Vector &, real_t T) -> real_t { return 1.0/(0.3+T); };
      auto dik = [](const Vector &, real_t T) -> real_t
      { return -1.0/((0.3+T)*(0.3+T)); };
      FunctionDiffusionFlux fun(dim, ik, dik);
      MixedConductionNLFIntegrator integ(fun);
      REQUIRE_FALSE(HDGMixedConductionResidualCanBatch(Vh, &integ));
   }

   SECTION("An H(div) flux space")
   {
      // A vector-valued basis: CalcVShape() is per element and the dofs are
      // laid out per equation, not per (equation, direction).
      Law law(Kappa::Scalar, dim);
      MixedConductionNLFIntegrator integ(*law.fun);
      REQUIRE_FALSE(HDGMixedConductionResidualCanBatch(Rh, &integ));
   }

   SECTION("A flux space whose vdim is not the space dimension")
   {
      FiniteElementSpace Sh(&mesh, &u_coll);   // vdim 1
      Law law(Kappa::Scalar, dim);
      MixedConductionNLFIntegrator integ(*law.fun);
      REQUIRE_FALSE(HDGMixedConductionResidualCanBatch(Sh, &integ));
   }

   SECTION("An integrator that is not a MixedConductionNLFIntegrator")
   {
      SumBlockNLFIntegrator sum(false);
      REQUIRE_FALSE(HDGMixedConductionResidualCanBatch(Vh, &sum));
   }
}

namespace darcy_batched_residual
{

struct ResidualOutcome
{
   BlockVector r;
   Vector r_tr;
   Vector Sv;
   bool taken = false;
   int integ_calls = 0;
};

/** @brief One NPC residual evaluation on the `-nld` shape: the flux law is a
    MixedConductionNLFIntegrator on the BlockNonlinearForm and the HDG
    stabilization is a plain HDGDiffusionIntegrator on the linear potential
    mass form.

    That is `convdiff -nld -hb -npc` and it is the configuration 15 of the 88
    hybridized regression references have -- established by printing which
    integrator pointers DarcyHybridization ends up holding for each reference,
    not by reading the miniapp. */
void NPCResidualOnce(DarcyHybridization::AssemblyMode am, int order,
                     ResidualOutcome &out)
{
   const int dim = 2, n = 4;
   Mesh mesh = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, false,
                                     0.8, 1.2);
   L2_FECollection u_coll(order, dim, BasisType::GaussLobatto);
   L2_FECollection p_coll(order, dim);
   DG_Interface_FECollection t_coll(order, dim);
   FiniteElementSpace Vh(&mesh, &u_coll, dim), Wh(&mesh, &p_coll),
                      Mh(&mesh, &t_coll);

   DarcyForm darcy(&Vh, &Wh);
   ConstantCoefficient one(1.0), src(1.0);
   FunctionCoefficient ikappa([](const Vector &X)
   {
      return 1.3 + 0.4 * std::sin(M_PI * X(0)) * X(1);
   });
   LinearDiffusionFlux law(dim, ikappa);

   darcy.GetPotentialRHS()->AddDomainIntegrator(new DomainLFIntegrator(src));
   darcy.GetFluxDivForm()->AddDomainIntegrator(
      new VectorDivergenceIntegrator());
   darcy.GetFluxDivForm()->AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));

   // The flux mass comes from the LAW and from nowhere else, which is what
   // makes m_nlfi the only element integrator on the local residual.
   darcy.GetBlockNonlinearForm()->AddDomainIntegrator(
      new MixedConductionNLFIntegrator(law));

   BilinearForm *M_p = darcy.GetPotentialMassForm();
   M_p->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));
   M_p->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));

   Array<int> ess_flux;
   darcy.EnableHybridization(&Mh, new NormalTraceJumpIntegrator(), ess_flux);
   DarcyHybridization *dh = darcy.GetHybridization();
   dh->SetAssemblyMode(am);
   dh->EnableNPC();
   Array<int> all(mesh.bdr_attributes.Max());
   all = 1;
   dh->SetEssentialBC(all);

   darcy.Assemble();
   darcy.Finalize();
   out.taken = dh->CanBatchLocalResidual();

   BlockVector b(darcy.GetOffsets()), x(darcy.GetOffsets());
   b = 0.0;
   darcy.GetPotentialRHS()->Assemble();
   b.GetBlock(1) += *darcy.GetPotentialRHS();
   b.GetBlock(1).SyncAliasMemory(b);

   // A state with structure. At x = 0 the flux row is exactly zero for a law
   // linear in the flux, so a zero state would compare nothing.
   FillWavy(x, 0.5);
   Vector x_tr(Mh.GetVSize());
   FillWavy(x_tr, 2.1);

   out.r.Update(darcy.GetOffsets());
   dh->NPCResidual(b, x, x_tr, out.r, out.r_tr);
   out.r.HostRead();
   out.r_tr.HostRead();
}

} // namespace darcy_batched_residual

/** @brief In situ: the NPC residual is the same with the kernel and without.

    Reachability is the point of this case, and it is what a batched route
    most often lacks -- this branch lost AssemblyMode::Batched's face kernel
    to a dead `dynamic_cast` for three commits with nothing noticing. The two
    REQUIREs on @a taken are therefore load-bearing: two fallbacks agree
    perfectly and test nothing.

    A TOLERANCE and not the bits, unlike the isolated case above, and the
    reason is a confound rather than the kernel: AssemblyMode::Batched also
    switches the face, mass and divergence assembly kernels on, and those
    accumulate point by point where the per-element route adds one element
    matrix. The isolated case is what pins the kernel itself. */
TEST_CASE("The batched local residual reaches an NPC caller",
          "[DarcyHybridization][BatchedLinAlg][GPU]")
{
   using namespace darcy_batched_residual;
   using AM = DarcyHybridization::AssemblyMode;

   const int order = GENERATE(1, 2);
   CAPTURE(order);

   ResidualOutcome ref, bat;
   NPCResidualOnce(AM::Serial, order, ref);
   NPCResidualOnce(AM::Batched, order, bat);

   REQUIRE_FALSE(ref.taken);
   REQUIRE(bat.taken);

   REQUIRE(ref.r.GetBlock(0).Normlinf() > 1e-3);
   REQUIRE(ref.r.GetBlock(1).Normlinf() > 1e-3);
   REQUIRE(ref.r_tr.Normlinf() > 1e-3);

   auto close = [](const Vector &a, const Vector &b)
   {
      REQUIRE(a.Size() == b.Size());
      Vector d(a);
      d -= b;
      // Floored: an equality test between two routes must not compare
      // round-off relatively, which is a standing note on this branch.
      REQUIRE(d.Normlinf() <= 1e-11 * std::max(a.Normlinf(), 1e-30));
   };
   close(ref.r.GetBlock(0), bat.r.GetBlock(0));
   close(ref.r.GetBlock(1), bat.r.GetBlock(1));
   close(ref.r_tr, bat.r_tr);
}

namespace darcy_batched_residual
{

/** @brief One NPC residual on a problem whose mass terms sit either on the
    BILINEAR forms or on the NONLINEAR ones, the two being the same discrete
    problem by construction.

    `class BilinearFormIntegrator : public NonlinearFormIntegrator`, so
    installing one on a nonlinear form compiles, and that is what
    `convdiff -nl` does: a VectorMassIntegrator on the flux mass nonlinear
    form. DarcyForm::EnableHybridization() then wraps it in a
    SumNLFIntegrator -- unconditionally, even for a single integrator -- and
    the element loop calls BilinearFormIntegrator::AssembleElementVector(),
    which re-assembles a constant element matrix and multiplies, on every
    residual evaluation.

    TWO integrators on the potential mass, and one of them NON-SYMMETRIC.
    Both are deliberate: a single-member sum cannot see a routine that stops
    after the first member, and a symmetric block cannot see a transposed
    write. */
void NPCResidualMassSlots(bool nonlinear_slots, int order,
                          ResidualOutcome &out)
{
   const int dim = 2, n = 4;
   Mesh mesh = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, false,
                                     0.8, 1.2);
   L2_FECollection u_coll(order, dim, BasisType::GaussLobatto);
   L2_FECollection p_coll(order, dim);
   DG_Interface_FECollection t_coll(order, dim);
   FiniteElementSpace Vh(&mesh, &u_coll, dim), Wh(&mesh, &p_coll),
                      Mh(&mesh, &t_coll);

   DarcyForm darcy(&Vh, &Wh);
   ConstantCoefficient one(1.0), src(1.0);
   FunctionCoefficient ikappa([](const Vector &X)
   {
      return 1.3 + 0.4 * std::sin(M_PI * X(0)) * X(1);
   });
   Vector bvec(dim);
   bvec(0) = 0.9;
   bvec(1) = -0.6;
   VectorConstantCoefficient bcoeff(bvec);

   darcy.GetPotentialRHS()->AddDomainIntegrator(new DomainLFIntegrator(src));
   darcy.GetFluxDivForm()->AddDomainIntegrator(
      new VectorDivergenceIntegrator());
   darcy.GetFluxDivForm()->AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));

   if (nonlinear_slots)
   {
      darcy.GetFluxMassNonlinearForm()->AddDomainIntegrator(
         new VectorMassIntegrator(ikappa));
      NonlinearForm *Mnl_p = darcy.GetPotentialMassNonlinearForm();
      Mnl_p->AddDomainIntegrator(new MassIntegrator(one));
      Mnl_p->AddDomainIntegrator(new ConservativeConvectionIntegrator(bcoeff));
   }
   else
   {
      darcy.GetFluxMassForm()->AddDomainIntegrator(
         new VectorMassIntegrator(ikappa));
      BilinearForm *M_p = darcy.GetPotentialMassForm();
      M_p->AddDomainIntegrator(new MassIntegrator(one));
      M_p->AddDomainIntegrator(new ConservativeConvectionIntegrator(bcoeff));
   }

   // The face constraint, on the LINEAR potential mass form either way --
   // CanBatchLinearResidual() requires c_bfi_p, and this is also what keeps
   // Df_lin_data non-empty when the domain terms have moved off it.
   BilinearForm *M_p_face = darcy.GetPotentialMassForm();
   M_p_face->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));
   M_p_face->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));

   Array<int> ess_flux;
   darcy.EnableHybridization(&Mh, new NormalTraceJumpIntegrator(), ess_flux);
   DarcyHybridization *dh = darcy.GetHybridization();
   dh->EnableNPC();
   Array<int> all(mesh.bdr_attributes.Max());
   all = 1;
   dh->SetEssentialBC(all);

   darcy.Assemble();
   darcy.Finalize();

   BlockVector b(darcy.GetOffsets()), x(darcy.GetOffsets());
   b = 0.0;
   darcy.GetPotentialRHS()->Assemble();
   b.GetBlock(1) += *darcy.GetPotentialRHS();
   b.GetBlock(1).SyncAliasMemory(b);

   FillWavy(x, 0.5);
   Vector x_tr(Mh.GetVSize());
   FillWavy(x_tr, 2.1);

   out.r.Update(darcy.GetOffsets());
   dh->NPCResidual(b, x, x_tr, out.r, out.r_tr);
   // AFTER the residual: the route builds its cache on first use, and asking
   // before would report the question rather than the answer.
   out.taken = dh->CanBatchLinearResidual();
   out.r.HostRead();
   out.r_tr.HostRead();

   // **And the GRADIENT on the same state, which is the half this case was
   // missing.** It compared residuals only, and passed for a whole session
   // while the nonlinear-slot gradient was DOUBLE COUNTED: ConstructGrad()'s
   // `if (m_nlfi_u)` branch carried no !ad_done guard, so the element loop
   // added the element flux mass on top of CopyLinearGradBlocks()'s copy of
   // the same matrix. The residual was bit-identical throughout -- it is
   // computed by a different route -- so no residual comparison of any
   // tightness could have caught it, while 20 of the 152 serial references
   // did. Taken here, before dh goes out of scope, because the returned
   // Operator is owned by it.
   Operator &S = dh->NPCGradient(x, x_tr);
   Vector v(x_tr.Size());
   FillWavy(v, 3.7);
   out.Sv.SetSize(x_tr.Size());
   S.Mult(v, out.Sv);
   out.Sv.HostRead();
}

} // namespace darcy_batched_residual

/** @brief A mass term on the NONLINEAR form is the same problem as one on the
    bilinear form, and the batched residual takes both.

    The REQUIRE on the nonlinear arm's @a taken is the whole reachability
    half, and it is what fails if the route goes back to refusing a nonlinear
    mass slot: measured over the unit suite, no fixture reached this before
    the case existed, and every `convdiff -nl` run did.

    The equality half pins the assembled copy. Discriminated rather than
    assumed: truncating the sum to its first member, and transposing each
    element matrix, each fail this case on the first comparison. */
TEST_CASE("A bilinear integrator on a nonlinear mass form is assembled once",
          "[DarcyHybridization][BatchedLinAlg][NPC][GPU]")
{
   using namespace darcy_batched_residual;

   const int order = GENERATE(1, 2);
   CAPTURE(order);

   ResidualOutcome lin, nlin;
   NPCResidualMassSlots(false, order, lin);
   NPCResidualMassSlots(true, order, nlin);

   REQUIRE(lin.taken);
   REQUIRE(nlin.taken);

   REQUIRE(lin.r.GetBlock(0).Normlinf() > 1e-3);
   REQUIRE(lin.r.GetBlock(1).Normlinf() > 1e-3);
   REQUIRE(lin.r_tr.Normlinf() > 1e-3);

   auto close = [](const Vector &a, const Vector &b)
   {
      REQUIRE(a.Size() == b.Size());
      Vector d(a);
      d -= b;
      REQUIRE(d.Normlinf() <= 1e-11 * std::max(a.Normlinf(), 1e-30));
   };
   close(lin.r.GetBlock(0), nlin.r.GetBlock(0));
   close(lin.r.GetBlock(1), nlin.r.GetBlock(1));
   close(lin.r_tr, nlin.r_tr);

   // The gradient has to agree too, and this is the assertion that fails on
   // the double count -- by a factor near two on |S v|, not by round-off.
   REQUIRE(lin.Sv.Normlinf() > 1e-3);
   close(lin.Sv, nlin.Sv);
}

#ifdef MFEM_USE_MPI

namespace darcy_batched_residual
{

/** @brief NPCResidualOnce()'s problem on a ParMesh.

    The same `-nld` shape -- the flux law a MixedConductionNLFIntegrator on the
    BlockNonlinearForm, the stabilization a plain HDGDiffusionIntegrator on the
    linear potential mass -- because that is the shape
    DarcyHybridization::CanBatchLocalResidual() admits, and the only thing this
    twin varies is that the three spaces are parallel. */
void ParNPCResidualOnce(DarcyHybridization::AssemblyMode am, int order,
                        ResidualOutcome &out)
{
   const int dim = 2, n = 4;
   Mesh serial = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, false,
                                       0.8, 1.2);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   L2_FECollection u_coll(order, dim, BasisType::GaussLobatto);
   L2_FECollection p_coll(order, dim);
   DG_Interface_FECollection t_coll(order, dim);
   ParFiniteElementSpace Vh(&mesh, &u_coll, dim), Wh(&mesh, &p_coll),
                         Mh(&mesh, &t_coll);

   ParDarcyForm darcy(&Vh, &Wh);
   ConstantCoefficient one(1.0), src(1.0);
   FunctionCoefficient ikappa([](const Vector &X)
   {
      return 1.3 + 0.4 * std::sin(M_PI * X(0)) * X(1);
   });
   LinearDiffusionFlux law(dim, ikappa);

   darcy.GetPotentialRHS()->AddDomainIntegrator(new DomainLFIntegrator(src));
   darcy.GetFluxDivForm()->AddDomainIntegrator(
      new VectorDivergenceIntegrator());
   darcy.GetFluxDivForm()->AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));

   darcy.GetBlockNonlinearForm()->AddDomainIntegrator(
      new MixedConductionNLFIntegrator(law));

   BilinearForm *M_p = darcy.GetPotentialMassForm();
   M_p->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));
   M_p->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));

   Array<int> ess_flux;
   darcy.EnableHybridization(&Mh, new NormalTraceJumpIntegrator(), ess_flux);
   DarcyHybridization *dh = darcy.GetHybridization();
   dh->SetAssemblyMode(am);
   dh->EnableNPC();
   Array<int> all(mesh.bdr_attributes.Max());
   all = 1;
   dh->SetEssentialBC(all);

   darcy.Assemble();
   darcy.Finalize();
   out.taken = dh->CanBatchLocalResidual();

   // The trace block is sized on TRUE dofs, which is what NPC's interface
   // takes; the flux and potential are L2 and are the same either way.
   Array<int> offs(4);
   offs[0] = 0;
   offs[1] = Vh.GetVSize();
   offs[2] = Wh.GetVSize();
   offs[3] = Mh.GetTrueVSize();
   offs.PartialSum();

   BlockVector b(darcy.GetOffsets()), x(darcy.GetOffsets());
   b = 0.0;
   darcy.GetPotentialRHS()->Assemble();
   b.GetBlock(1) += *darcy.GetPotentialRHS();
   b.GetBlock(1).SyncAliasMemory(b);

   // A state with structure, as in the serial twin: at x = 0 the flux row is
   // exactly zero for a law linear in the flux, so a zero state would compare
   // nothing. **Seeded from the GLOBAL dof index and not the local one**, or
   // the two arms would be comparing different states the moment the
   // partition changed -- and it would still pass, both arms being wrong the
   // same way.
   auto fill_global = [](const ParFiniteElementSpace &fes, Vector &v,
                         real_t shift)
   {
      const HYPRE_BigInt *off = fes.GetTrueDofOffsets();
      const HYPRE_BigInt base = off ? off[0] : 0;
      for (int i = 0; i < v.Size(); i++)
      {
         const real_t g = real_t(base + i);
         v(i) = std::sin(0.7 * g + shift) + 0.25 * std::cos(0.13 * g);
      }
   };
   fill_global(Vh, x.GetBlock(0), 0.5);
   fill_global(Wh, x.GetBlock(1), 0.5);
   Vector x_tr(Mh.GetTrueVSize());
   fill_global(Mh, x_tr, 2.1);

   out.r.Update(darcy.GetOffsets());
   dh->NPCResidual(b, x, x_tr, out.r, out.r_tr);
   out.r.HostRead();
   out.r_tr.HostRead();
}

} // namespace darcy_batched_residual

/** @brief The batched local residual on more than one rank.

    **This case is the whole content of lifting a refusal**, and the refusal
    said so: CanBatchLocalResidual() declined `ParallelU() || ParallelP()`
    while stating that the kernel needs nothing from a neighbour -- the flux
    space is L2 and NPCCheck() refuses a conforming one, so every element's
    flux dofs are its own and the loop is rank-local -- and that it was
    refused only because it had not been RUN on more than one rank. That is a
    claim about coverage rather than about behaviour, and the answer to it is
    a test.

    What could have made it wrong, and what this therefore compares: the
    kernel gathers `darcy_u` through el_u_dofs, a map built from
    GetElementVDofs() over the LOCAL elements, and writes one flat slice per
    local element. On a ParMesh both the element count and the dof numbering
    change under the partition, so a map built with a serial assumption would
    hand the element loop another element's slice -- which is a wrong answer
    and not a crash.

    The two REQUIREs on @a taken are load-bearing for the usual reason: the
    route falls back silently, and two fallbacks agree perfectly while testing
    nothing. Here they also state the result -- `Batched` is taken in parallel,
    which before this it was not.

    A tolerance rather than the bits, as in the serial in-situ case:
    AssemblyMode::Batched flips whichever other kernels do not refuse in
    parallel, and those accumulate point by point where the per-element route
    adds one element matrix. */
TEST_CASE("The batched local residual reaches an NPC caller in parallel",
          "[DarcyHybridization][BatchedLinAlg][NPC][Parallel]")
{
   using namespace darcy_batched_residual;
   using AM = DarcyHybridization::AssemblyMode;

   const int order = GENERATE(0, 1, 2);
   CAPTURE(order, Mpi::WorldSize());

   ResidualOutcome ref, got;
   ParNPCResidualOnce(AM::Serial, order, ref);
   ParNPCResidualOnce(AM::Batched, order, got);

   REQUIRE_FALSE(ref.taken);
   REQUIRE(got.taken);

   // There is a residual to compare, and it is not the zero one. Taken as a
   // GLOBAL norm: a rank whose slice happened to be zero would otherwise let
   // this pass while comparing nothing on that rank.
   const real_t n_u = std::sqrt(InnerProduct(MPI_COMM_WORLD,
                                             ref.r.GetBlock(0),
                                             ref.r.GetBlock(0)));
   const real_t n_tr = std::sqrt(InnerProduct(MPI_COMM_WORLD, ref.r_tr,
                                              ref.r_tr));
   CAPTURE(n_u, n_tr);
   REQUIRE(n_u > 1e-3);
   REQUIRE(n_tr > 1e-6);

   auto close = [](const Vector &a, const Vector &b)
   {
      REQUIRE(a.Size() == b.Size());
      Vector d(a);
      d -= b;
      REQUIRE(d.Normlinf() <= 1e-11 * std::max(a.Normlinf(), 1e-30));
   };
   close(ref.r.GetBlock(0), got.r.GetBlock(0));
   close(ref.r.GetBlock(1), got.r.GetBlock(1));
   close(ref.r_tr, got.r_tr);
}

#endif // MFEM_USE_MPI

namespace darcy_caller_integrand
{

using darcy_batched_residual::FillWavy;

/// The largest entrywise difference, which is what the comparisons below are.
inline real_t MaxDiff(const Vector &a, const Vector &b)
{
   REQUIRE(a.Size() == b.Size());
   real_t m = 0.;
   for (int i = 0; i < a.Size(); i++)
   { m = std::max(m, std::abs(a(i) - b(i))); }
   return m;
}

/// The law, in ONE place, so the integrator below and the device callable in
/// the case cannot drift apart: `F(u) = u + c u^3`, a genuine nonlinearity.
// MFEM_HOST_DEVICE: called from device lambdas; see CallerLaw() in
// darcy_plain_npc.hpp for what nvcc does with a plain host function there.
MFEM_HOST_DEVICE inline real_t Law(real_t u, real_t c)
{ return u + c * u * u * u; }
MFEM_HOST_DEVICE inline real_t DLaw(real_t u, real_t c)
{ return 1.0 + 3.0 * c * u * u; }

/** @brief A genuinely nonlinear potential mass, of the shape a caller's own
    integrator has: pointwise in the potential, with an explicit rule.

    The rule is set by the fixture rather than left to the element, because
    NonlinearResidualBatched() refuses an integrator whose
    Integrator::GetIntRule() is null -- one mesh-wide table cannot reproduce a
    rule chosen per element. */
class CubicReactionNLF : public NonlinearFormIntegrator
{
   const real_t c;
public:
   CubicReactionNLF(real_t c_, const IntegrationRule *ir)
      : NonlinearFormIntegrator(ir), c(c_) { }

   void AssembleElementVector(const FiniteElement &el,
                              ElementTransformation &Tr,
                              const Vector &elfun, Vector &elvect) override
   {
      const int nd = el.GetDof();
      Vector shape(nd);
      elvect.SetSize(nd);
      elvect = 0.0;
      const IntegrationRule &ir = *IntRule;
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         el.CalcShape(ip, shape);
         const real_t u = shape * elfun;
         elvect.Add(ip.weight * Tr.Weight() * Law(u, c), shape);
      }
   }

   void AssembleElementGrad(const FiniteElement &el,
                            ElementTransformation &Tr,
                            const Vector &elfun, DenseMatrix &elmat) override
   {
      const int nd = el.GetDof();
      Vector shape(nd);
      elmat.SetSize(nd);
      elmat = 0.0;
      const IntegrationRule &ir = *IntRule;
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         el.CalcShape(ip, shape);
         const real_t u = shape * elfun;
         AddMult_a_VVt(ip.weight * Tr.Weight() * DLaw(u, c), shape, elmat);
      }
   }
};

/// Everything one residual needs, kept so two arms can be driven on ONE
/// DarcyForm -- the point being that the discrete problem is identical by
/// construction and not merely by agreement.
struct Fixture
{
   std::unique_ptr<Mesh> mesh;
   std::unique_ptr<L2_FECollection> u_coll, p_coll;
   std::unique_ptr<DG_Interface_FECollection> t_coll;
   std::unique_ptr<FiniteElementSpace> Vh, Wh, Mh;
   std::unique_ptr<DarcyForm> darcy;
   std::unique_ptr<ConstantCoefficient> one, src;
   DarcyHybridization *dh{nullptr};
   std::unique_ptr<BlockVector> b, x;
   Vector x_tr;
};

std::unique_ptr<Fixture> Build(int order, real_t c, bool linear_slot = false)
{
   auto F = std::unique_ptr<Fixture>(new Fixture);
   const int dim = 2, n = 4;
   F->mesh.reset(new Mesh(Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL,
                                                false, 0.8, 1.2)));
   F->u_coll.reset(new L2_FECollection(order, dim, BasisType::GaussLobatto));
   F->p_coll.reset(new L2_FECollection(order, dim));
   F->t_coll.reset(new DG_Interface_FECollection(order, dim));
   F->Vh.reset(new FiniteElementSpace(F->mesh.get(), F->u_coll.get(), dim));
   F->Wh.reset(new FiniteElementSpace(F->mesh.get(), F->p_coll.get()));
   F->Mh.reset(new FiniteElementSpace(F->mesh.get(), F->t_coll.get()));

   F->darcy.reset(new DarcyForm(F->Vh.get(), F->Wh.get()));
   F->one.reset(new ConstantCoefficient(1.0));
   F->src.reset(new ConstantCoefficient(1.0));

   F->darcy->GetPotentialRHS()->AddDomainIntegrator(
      new DomainLFIntegrator(*F->src));
   F->darcy->GetFluxDivForm()->AddDomainIntegrator(
      new VectorDivergenceIntegrator());
   F->darcy->GetFluxDivForm()->AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));

   // A LINEAR flux mass: LocalOpType::PotNL is the mode in which the
   // potential is the only nonlinearity, and it is the mode this route is
   // for.
   F->darcy->GetFluxMassForm()->AddDomainIntegrator(
      new VectorMassIntegrator(*F->one));

   const IntegrationRule &ir =
      IntRules.Get(Geometry::SQUARE, 2 * order + 3);
   if (linear_slot)
   {
      // The SAME discrete problem at c = 0, on the bilinear slot -- so
      // lop_type is FullNL and both the element loop and
      // LinearResidualBatched() are the verified ones. The arbiter.
      MassIntegrator *mi = new MassIntegrator(*F->one);
      mi->SetIntRule(&ir);
      F->darcy->GetPotentialMassForm()->AddDomainIntegrator(mi);
   }
   else
   {
      F->darcy->GetPotentialMassNonlinearForm()->AddDomainIntegrator(
         new CubicReactionNLF(c, &ir));
   }

   // The face constraint stays LINEAR and on the bilinear form, which is what
   // keeps c_bfi_p live and Df_lin_data non-empty.
   BilinearForm *M_p = F->darcy->GetPotentialMassForm();
   M_p->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(*F->one, 1.0));
   M_p->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(*F->one, 1.0));

   Array<int> ess_flux;
   F->darcy->EnableHybridization(F->Mh.get(), new NormalTraceJumpIntegrator(),
                                 ess_flux);
   F->dh = F->darcy->GetHybridization();
   F->dh->EnableNPC();
   Array<int> all(F->mesh->bdr_attributes.Max());
   all = 1;
   F->dh->SetEssentialBC(all);

   F->darcy->Assemble();
   F->darcy->Finalize();

   F->b.reset(new BlockVector(F->darcy->GetOffsets()));
   F->x.reset(new BlockVector(F->darcy->GetOffsets()));
   *F->b = 0.0;
   F->darcy->GetPotentialRHS()->Assemble();
   F->b->GetBlock(1) += *F->darcy->GetPotentialRHS();
   F->b->GetBlock(1).SyncAliasMemory(*F->b);

   FillWavy(*F->x, 0.5);
   F->x_tr.SetSize(F->Mh->GetVSize());
   FillWavy(F->x_tr, 2.1);
   return F;
}

} // namespace darcy_caller_integrand

TEST_CASE("A caller's integrand reproduces the element loop",
          "[DarcyHybridization][NPC][Batched][GPU]")
{
   using namespace darcy_caller_integrand;

   /* **The route MFEM does not own the integrand of.** Every other batched
      residual here replaces an integrator the library can see; this one
      replaces one it cannot, and the caller hands it in as a callable. So the
      case drives ONE DarcyForm two ways -- the element loop through
      NPCResidual(), and the kernel through the templated overload -- and the
      discrete problem is identical by construction rather than by agreement.

      They are compared to a TOLERANCE and not to the bit, and that is not
      slack: the element loop contracts each element's quadrature through
      Vector::Add() in dof order while the kernel accumulates per point, so
      the two sum the same terms in different orders. */
   /* **The one batched route whose integrand MFEM does not own**, so the
      case drives ONE discrete problem two ways -- the element loop through
      NPCResidual(), and the kernel through the templated overload with the
      same law as a device callable -- and the two fixtures are identical by
      construction rather than by agreement.

      Compared to a tolerance and not to the bit, and that is not slack: the
      element loop contracts each element's quadrature through Vector::Add()
      in dof order while the kernel accumulates per point, so the two sum the
      same terms in different orders.

      **Order 2 is the arm that discriminates and order 1 nearly hid a real
      defect.** The scatter ACCUMULATES -- an interior trace dof takes a
      contribution from each of its two elements -- and the destination was
      zeroed only by MultNL(), which this route does not go through. At order
      1 the output vector came back from a fresh allocation that was already
      zero and the case passed; at order 2 it did not, and every interior face
      was wrong. Keep both orders. */
   const int order = GENERATE(1, 2);
   const real_t c = 0.35;
   CAPTURE(order, c);

   auto ref = Build(order, c);
   auto bat = Build(order, c);

   // The route has to be TAKEN, and asking before the residual would report
   // the question rather than the answer -- the caches are built on first use.
   BlockVector r_ref(ref->darcy->GetOffsets()), r_bat(bat->darcy->GetOffsets());
   Vector rt_ref, rt_bat;

   ref->dh->NPCResidual(*ref->b, *ref->x, ref->x_tr, r_ref, rt_ref);

   bat->dh->NPCResidual(*bat->b, *bat->x, bat->x_tr, r_bat, rt_bat,
                        [c] MFEM_HOST_DEVICE (int, int, real_t, real_t,
                                              real_t u)
   {
      return Law(u, c);
   });
   REQUIRE(bat->dh->CanBatchNonlinearResidual());

   // And the linear route must NOT apply here, or the case would be measuring
   // something else: its two refusals are this route's two preconditions.
   REQUIRE_FALSE(bat->dh->CanBatchLinearResidual());

   r_ref.HostRead();
   r_bat.HostRead();
   rt_ref.HostRead();
   rt_bat.HostRead();

   const real_t scale = std::max(r_ref.Normlinf(), rt_ref.Normlinf());
   CAPTURE(scale);
   REQUIRE(scale > 0.0);
   REQUIRE(MaxDiff(r_ref.GetBlock(0), r_bat.GetBlock(0)) < 1e-10 * scale);
   REQUIRE(MaxDiff(r_ref.GetBlock(1), r_bat.GetBlock(1)) < 1e-10 * scale);
   REQUIRE(MaxDiff(rt_ref, rt_bat) < 1e-10 * scale);

   /* **The discriminator, and without it the case would pass on an integrand
      that was never called.** A kernel that added nothing to the potential row
      would agree with the element loop everywhere the reaction is small, and
      at c = 0.35 on this state it is not small -- so a WRONG law has to move
      the answer by much more than the tolerance above. Dropping the cubic term
      is that law. */
   auto wrong = Build(order, c);
   BlockVector r_wrong(wrong->darcy->GetOffsets());
   Vector rt_wrong;
   wrong->dh->NPCResidual(*wrong->b, *wrong->x, wrong->x_tr, r_wrong, rt_wrong,
                          [] MFEM_HOST_DEVICE (int, int, real_t, real_t,
   real_t u) { return u; });
   r_wrong.HostRead();
   REQUIRE(MaxDiff(r_ref.GetBlock(1), r_wrong.GetBlock(1)) > 1e-6 * scale);
}

TEST_CASE("A caller's derivative reproduces the element loop's Jacobian",
          "[DarcyHybridization][NPC][Batched][GPU]")
{
   using namespace darcy_caller_integrand;

   /* **The gradient counterpart of the case above, and it shares its
      fixture's law on purpose.** `Law()` and `DLaw()` sit together in one
      place so the integrator's AssembleElementGrad(), the element loop's
      reference answer and the device callable cannot drift apart -- which is
      the failure this pair of entry points is most exposed to, the two
      callables being opaque to the library.

      ONE discrete problem, two routes: `ref` takes the element loop through
      the plain NPCGradient() -- CopyLinearGradBlocks() declines for
      LocalOpType::PotNL, so ConstructGrad() calls the integrator per element
      -- and `bat` takes the templated overload, which writes every D block in
      one kernel and tells the loop not to write them again.

      **The comparison is on the REDUCED operator and not on D**, deliberately.
      D is internal, and what a caller is owed is that the trace operator it
      gets back is the same one; that also puts the element factorisation and
      the Schur complement inside the claim, which is where a
      wrongly-transposed or wrongly-zeroed block would surface. */
   const int order = GENERATE(1, 2);
   const real_t c = 0.35;
   CAPTURE(order, c);

   auto ref = Build(order, c);
   auto bat = Build(order, c);

   Operator &G_ref = ref->dh->NPCGradient(*ref->x, ref->x_tr);
   Operator &G_bat = bat->dh->NPCGradient(
                        *bat->x, bat->x_tr,
                        [c] MFEM_HOST_DEVICE (int, int, real_t, real_t,
                                              real_t u)
   {
      return DLaw(u, c);
   });

   // The route has to be TAKEN. Asking before the call would report the
   // question rather than the answer -- D is allocated on first use.
   REQUIRE(bat->dh->CanBatchNonlinearGradient());

   REQUIRE(G_ref.Height() == G_bat.Height());
   REQUIRE(G_ref.Width() == G_bat.Width());
   REQUIRE(G_ref.Width() > 0);

   /* Two directions rather than one. A single probe can miss a block that is
      wrong only where that vector happens to be small, and the second costs
      one Mult(). */
   real_t worst = 0., scale = 0.;
   for (int k = 0; k < 2; k++)
   {
      Vector v(G_ref.Width()), y_ref(G_ref.Height()), y_bat(G_ref.Height());
      FillWavy(v, 0.3 + 1.7 * k);
      G_ref.Mult(v, y_ref);
      G_bat.Mult(v, y_bat);
      y_ref.HostRead();
      y_bat.HostRead();
      worst = std::max(worst, MaxDiff(y_ref, y_bat));
      scale = std::max(scale, y_ref.Normlinf());
   }
   CAPTURE(worst, scale);
   REQUIRE(scale > 0.0);

   /* A tolerance and not the bit, for the reason the residual case gives:
      the element loop contracts through AddMult_a_VVt() in dof order while
      the kernel accumulates per quadrature point, so the two sum the same
      terms in a different order. MEASURED at exactly 0 on both orders here,
      the two orders happening to coincide for this rule -- which is a fact
      about this fixture and not a property to assert. */
   REQUIRE(worst < 1e-10 * scale);

   /* **DO NOT ADD A FINITE-DIFFERENCE CHECK OF THIS OPERATOR AGAINST
      NPCResidual(), which is where the obvious next test goes and it is
      differencing the wrong function.** Built, run, and thrown away: it
      disagreed by 76% of the operator's own scale, and the arm that settled
      it was the ELEMENT LOOP's operator, which disagreed by the identical
      0.5234487811234837 while agreeing with the batched one to exactly zero.

      The reason is the ordering, not the kernel. What NPCGradient() returns
      is the REDUCED operator -- the Schur complement, with the element-local
      unknowns eliminated -- while NPCResidual() at a fixed field state
      differentiates to the direct trace-trace block alone: under NPC the
      fields are Newton STATE and do not follow the trace, which is the whole
      difference between this ordering and the condensation. So the two are
      not a function and its derivative, and no tolerance makes them one. */

   /* **The discriminator, and the case is worth little without it.** A kernel
      that wrote a zero block, or one the loop then overwrote with the linear
      copy, would still produce an operator of the right shape -- and at
      order 1 on a mild state it could sit inside the tolerance above. So a
      WRONG derivative has to move the answer by far more than that. Dropping
      the cubic term is that derivative, and it is the exact mistake a caller
      makes by handing NPCGradient() the residual's own law instead of its
      derivative. */
   auto wrong = Build(order, c);
   Operator &G_wrong = wrong->dh->NPCGradient(
                          *wrong->x, wrong->x_tr,
                          [] MFEM_HOST_DEVICE (int, int, real_t, real_t,
                                               real_t)
   {
      return real_t(1.0);
   });
   Vector v(G_ref.Width()), y_ref(G_ref.Height()), y_wrong(G_ref.Height());
   FillWavy(v, 0.3);
   G_ref.Mult(v, y_ref);
   G_wrong.Mult(v, y_wrong);
   y_ref.HostRead();
   y_wrong.HostRead();
   REQUIRE(MaxDiff(y_ref, y_wrong) > 1e-6 * scale);
}


TEST_CASE("The batched Jacobian refuses what it cannot take",
          "[DarcyHybridization][NPC][Batched]")
{
   using namespace darcy_caller_integrand;

   /* The predicate is public so a caller can find out whether the route was
      TAKEN, and a predicate nobody falsifies is a predicate that widens. This
      is the arm that must say NO.

      `linear_slot` puts the SAME discrete problem at c = 0 on the BILINEAR
      potential mass, which makes lop_type FullNL rather than PotNL -- and a
      bilinear slot belongs on the linear route, which assembles once and is
      strictly better than any per-linearisation kernel. */
   auto F = Build(1, 0.0, /*linear_slot=*/true);

   Operator &G = F->dh->NPCGradient(
                    *F->x, F->x_tr,
                    [] MFEM_HOST_DEVICE (int, int, real_t, real_t, real_t)
   {
      // Deliberately absurd: if the route were taken despite the refusal this
      // would land in D and the operator below could not be the linear one.
      return real_t(1e6);
   });

   REQUIRE_FALSE(F->dh->CanBatchNonlinearGradient());

   /* And the refusal has to be INERT, which is the half a boolean cannot
      show. The templated overload passes `ad_external = false` when the batch
      declines, so the element loop writes A and D exactly as the plain
      NPCGradient() would -- same operator, integrand never consulted. */
   auto R = Build(1, 0.0, /*linear_slot=*/true);
   Operator &G_ref = R->dh->NPCGradient(*R->x, R->x_tr);

   REQUIRE(G.Height() == G_ref.Height());
   Vector v(G.Width()), y(G.Height()), y_ref(G_ref.Height());
   FillWavy(v, 0.9);
   G.Mult(v, y);
   G_ref.Mult(v, y_ref);
   y.HostRead();
   y_ref.HostRead();
   REQUIRE(MaxDiff(y, y_ref) == MFEM_Approx(0.0));
}
