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

namespace darcy_batched_weights
{

/** @brief A coefficient the device route does not recognise, with exactly
    the values of the one it wraps: what forces a batched face or mass
    routine onto its host weights without changing a single number. */
class HostOnly : public Coefficient
{
   Coefficient &c;
public:
   explicit HostOnly(Coefficient &c_) : c(c_) { }
   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   { return c.Eval(T, ip); }
};

/** @brief HDGRuleStabilization's Eval() without its GetRule(): the same
    stabilization, which the batched routes must weigh on the host. */
class NoRule : public HDGStabilization
{
   HDGRuleStabilization s;
public:
   explicit NoRule(const HDGStabilizationRule &r) : s(r) { }
   real_t Eval(real_t s_diff, real_t un, real_t u, real_t uhat,
               ElementTransformation &Tr) const override
   { return s.Eval(s_diff, un, u, uhat, Tr); }
};

/// What the face integrators weigh their points with.
enum class Weight
{
   Plain,        ///< no coefficient, the built-in stabilization
   Constant,     ///< a ConstantCoefficient, the built-in stabilization
   GridFn,       ///< x_0 as a GridFunctionCoefficient, the built-in one
   TauConstant,  ///< a constant HDGRuleStabilization
   TauInvR       ///< HDGRuleStabilization tau0/max(x_0, R_f)
};

const char *Name(Weight w)
{
   switch (w)
   {
      case Weight::Plain:       return "no coefficient";
      case Weight::Constant:    return "constant coefficient";
      case Weight::GridFn:      return "GridFunctionCoefficient";
      case Weight::TauConstant: return "constant tau rule";
      case Weight::TauInvR:     return "tau0/max(R, R_f) rule";
   }
   return "?";
}

/** @brief The coefficients and stabilizations of one configuration, and
    their host-only twins. The coordinate field is the mesh's own nodes when
    it has them -- meq's case -- and a separate degree-1 H1 field otherwise. */
struct Weights
{
   std::unique_ptr<FiniteElementCollection> fec;
   std::unique_ptr<FiniteElementSpace> fes;
   std::unique_ptr<GridFunction> X;
   std::unique_ptr<GridFunctionCoefficient> R;
   ConstantCoefficient c{2.5};
   std::unique_ptr<VectorConstantCoefficient> zv;
   std::unique_ptr<HostOnly> host_c, host_R;
   HDGStabilizationRule tau_const, tau_invr;
   std::unique_ptr<HDGRuleStabilization> rule_const, rule_invr;
   std::unique_ptr<NoRule> norule_const, norule_invr;

   explicit Weights(Mesh &mesh)
   {
      const int sdim = mesh.SpaceDimension();
      if (mesh.GetNodes())
      {
         R.reset(new GridFunctionCoefficient(mesh.GetNodes(), 1));
      }
      else
      {
         fec.reset(new H1_FECollection(1, mesh.Dimension()));
         fes.reset(new FiniteElementSpace(&mesh, fec.get(), sdim));
         X.reset(new GridFunction(fes.get()));
         VectorFunctionCoefficient xc(sdim, [](const Vector &x, Vector &y)
         { y = x; });
         X->ProjectCoefficient(xc);
         R.reset(new GridFunctionCoefficient(X.get(), 1));
      }
      Vector z(sdim);
      z = 0.0;
      zv.reset(new VectorConstantCoefficient(z));
      host_c.reset(new HostOnly(c));
      host_R.reset(new HostOnly(*R));
      tau_const.c0 = 1.7;
      tau_invr.c0 = 3.0;
      tau_invr.coord = 0;
      // Inside the mesh, so that some points take the floor and some do not.
      tau_invr.floor = 0.35;
      rule_const.reset(new HDGRuleStabilization(tau_const));
      rule_invr.reset(new HDGRuleStabilization(tau_invr));
      norule_const.reset(new NoRule(tau_const));
      norule_invr.reset(new NoRule(tau_invr));
   }

   /** One face integrator of configuration @a w: the device route's when
       @a host is false, the host-only twin's when it is true. The twin of
       "no coefficient" carries a ZERO velocity, which makes u.n = 0 and so
       the same weight, and takes the host route because a velocity is read
       there. */
   HDGDiffusionIntegrator *Make(Weight w, bool host)
   {
      HDGDiffusionIntegrator *d = nullptr;
      switch (w)
      {
         case Weight::Plain:
            // The velocity constructor takes alpha and sets beta = alpha/2,
            // so 2.6 is beta = 1.3; alpha never enters, since u.n = 0.
            d = host ? new HDGDiffusionIntegrator(*zv, 2.6)
                : new HDGDiffusionIntegrator(1.3);
            break;
         case Weight::Constant:
            d = host ? new HDGDiffusionIntegrator(*host_c, 1.3)
                : new HDGDiffusionIntegrator(c, 1.3);
            break;
         case Weight::GridFn:
            d = host ? new HDGDiffusionIntegrator(*host_R, 1.3)
                : new HDGDiffusionIntegrator(*R, 1.3);
            break;
         case Weight::TauConstant:
            // The coefficient is dead weight under a rule, which is meq's
            // configuration: R handed to an integrator whose stabilization
            // never reads it.
            d = new HDGDiffusionIntegrator(*host_R, 1.3);
            if (host) { d->SetStabilization(*norule_const); }
            else { d->SetStabilization(*rule_const); }
            break;
         case Weight::TauInvR:
            d = new HDGDiffusionIntegrator(*host_R, 1.3);
            if (host) { d->SetStabilization(*norule_invr); }
            else { d->SetStabilization(*rule_invr); }
            break;
      }
      return d;
   }
};

struct Outcome
{
   Array<int> I, J;
   Vector data;
   bool taken = false, bdr_taken = false, on_device = false;
};

/** The NPC trace gradient of a diffusion problem whose potential face term is
    configuration @a w, assembled in @a am. As in test_darcy_batched_face.cpp:
    the GRADIENT, because an assembled operator cannot hide a mis-weighted
    face the way a solve at a finite tolerance can. */
void AssembleGradient(Mesh &mesh, int order, Weight w, bool host,
                      DarcyHybridization::AssemblyMode am, Outcome &out)
{
   const int dim = mesh.Dimension();
   L2_FECollection u_coll(order, dim, BasisType::GaussLobatto);
   L2_FECollection p_coll(order, dim);
   DG_Interface_FECollection t_coll(order, dim);
   FiniteElementSpace Vh(&mesh, &u_coll, dim), Wh(&mesh, &p_coll),
                      Mh(&mesh, &t_coll);

   Weights ws(mesh);
   DarcyForm darcy(&Vh, &Wh);
   darcy.GetFluxMassForm()->AddDomainIntegrator(new VectorMassIntegrator());
   darcy.GetFluxDivForm()->AddDomainIntegrator(new VectorDivergenceIntegrator());
   darcy.GetFluxDivForm()->AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));
   BilinearForm *M_p = darcy.GetPotentialMassForm();
   HDGDiffusionIntegrator *interior = ws.Make(w, host);
   M_p->AddInteriorFaceIntegrator(interior);
   M_p->AddBdrFaceIntegrator(ws.Make(w, host));
   out.on_device = HDGFaceWeightsOnDevice(*interior, Wh);

   darcy.EnableHybridization(&Mh, new NormalTraceJumpIntegrator(),
                             Array<int>());
   DarcyHybridization *dh = darcy.GetHybridization();
   dh->SetAssemblyMode(am);
   dh->EnableNPC();
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;
   dh->SetEssentialBC(ess_bdr);

   darcy.Assemble();
   darcy.Finalize();
   out.taken = dh->CanBatchPotFaceAssembly();
   out.bdr_taken = dh->CanBatchPotBdrFaceAssembly();

   BlockVector x(darcy.GetOffsets());
   x = 0.0;
   Vector x_tr(Mh.GetVSize());
   x_tr = 0.0;
   Operator &S = dh->NPCGradient(x, x_tr);
   SparseMatrix *Sm = dynamic_cast<SparseMatrix*>(&S);
   REQUIRE(Sm != nullptr);
   Sm->HostReadI();
   Sm->HostReadJ();
   Sm->HostReadData();
   const int nrows = Sm->Height(), nnz = Sm->NumNonZeroElems();
   out.I.SetSize(nrows + 1);
   std::copy(Sm->GetI(), Sm->GetI() + nrows + 1, out.I.begin());
   out.J.SetSize(nnz);
   std::copy(Sm->GetJ(), Sm->GetJ() + nnz, out.J.begin());
   out.data.SetSize(nnz);
   std::copy(Sm->GetData(), Sm->GetData() + nnz, out.data.GetData());
}

/// A warped mesh of one of four element shapes; see the cases.
Mesh MakeMesh(int geom, bool nodes)
{
   Mesh mesh;
   switch (geom)
   {
      case 0:
         mesh = Mesh::MakeCartesian2D(4, 3, Element::TRIANGLE, true, 1.3, 0.9);
         break;
      case 1:
         mesh = Mesh::MakeCartesian2D(4, 3, Element::QUADRILATERAL, true,
                                      1.3, 0.9);
         break;
      case 2:
         mesh = Mesh::MakeCartesian3D(2, 2, 2, Element::TETRAHEDRON,
                                      1.0, 1.2, 0.8);
         break;
      default:
         mesh = Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON,
                                      1.0, 1.2, 0.8);
         break;
   }
   if (nodes) { mesh.EnsureNodes(); }
   // Off the axes and not affine, so that no two elements share a Jacobian,
   // no normal is axis-aligned, and a quadrilateral's weight varies along a
   // face: a kernel reading the wrong element, slot or face cannot agree.
   mesh.Transform([](const Vector &x, Vector &y)
   {
      y = x;
      y(0) += 0.13 * x(1) + 0.05 * x(0) * x(1);
      y(1) += 0.07 * x(0) * x(0);
   });
   return mesh;
}

const char *GeomName(int geom)
{
   const char *names[4] = {"triangles", "quadrilaterals", "tetrahedra",
                           "hexahedra"
                          };
   return names[geom];
}

void RequireSameSparsity(const Outcome &a, const Outcome &b)
{
   REQUIRE(a.I.Size() == b.I.Size());
   REQUIRE(a.J.Size() == b.J.Size());
   for (int i = 0; i < a.I.Size(); i++) { REQUIRE(a.I[i] == b.I[i]); }
   for (int i = 0; i < a.J.Size(); i++) { REQUIRE(a.J[i] == b.J[i]); }
}

} // namespace darcy_batched_weights

/**
 * @brief The batched face routes' device weights are the host weights.
 *
 * Under HDGFaceWeightsOnDevice() the interior and boundary scatters take
 * every point's weight from a kernel over the faces' and elements' point
 * matrices instead of from a FaceElementTransformations per face. Two
 * comparisons, and they test different things:
 *
 *  - against the SAME batched assembly forced onto its host weights by a
 *    twin integrator that the device route does not recognise but which
 *    has the same numbers (a wrapped coefficient, a stabilization without
 *    GetRule(), a zero velocity). The scatter kernel is common to both, so
 *    on the host the two operators agree BIT FOR BIT: the device geometry is
 *    the transformations' arithmetic in their order. A GPU contracts to
 *    FMAs, so there the bound is round-off.
 *  - against the per-face (Serial) route, to round-off, which is what pins
 *    the twin itself.
 *
 * Warped meshes of all four shapes, with and without nodes: the nodes arm
 * is meq's (their device integrand adds degree-1 nodes, and their R is the
 * nodes' first component), and the meshes without nodes must come out
 * without nodes.
 */
TEST_CASE("The batched HDG face weights on the device are the host weights",
          "[DarcyHybridization][BatchedLinAlg][GPU]")
{
   using namespace darcy_batched_weights;
   using AM = DarcyHybridization::AssemblyMode;

   const int geom = GENERATE(0, 1, 2, 3);
   const int order = GENERATE(0, 1, 2);
   const bool nodes = GENERATE(false, true);
   const Weight w = GENERATE(Weight::Plain, Weight::Constant, Weight::GridFn,
                             Weight::TauConstant, Weight::TauInvR);
   CAPTURE(GeomName(geom), order, nodes, Name(w));
   if (geom >= 2 && order == 2) { return; }

   Mesh mesh_s = MakeMesh(geom, nodes), mesh_h = MakeMesh(geom, nodes),
        mesh_d = MakeMesh(geom, nodes);

   Outcome ser, hst, dev;
   AssembleGradient(mesh_s, order, w, false, AM::Serial, ser);
   AssembleGradient(mesh_h, order, w, true, AM::Batched, hst);
   AssembleGradient(mesh_d, order, w, false, AM::Batched, dev);

   // The routes taken: both batched arms batched, on the weights intended.
   REQUIRE(hst.taken);
   REQUIRE(hst.bdr_taken);
   REQUIRE(dev.taken);
   REQUIRE(dev.bdr_taken);
   REQUIRE_FALSE(hst.on_device);
   REQUIRE(dev.on_device);
   REQUIRE((mesh_d.GetNodes() != nullptr) == nodes);

   REQUIRE(ser.data.Size() > 0);
   REQUIRE(ser.data.Normlinf() > 1e-6);
   RequireSameSparsity(ser, dev);
   RequireSameSparsity(hst, dev);

   const real_t scale = ser.data.Normlinf();
   Vector d(dev.data);
   d -= hst.data;
   CAPTURE(d.Normlinf(), scale);
   if (Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK))
   {
      REQUIRE(d.Normlinf() <= 1e-14 * scale);
   }
   else
   {
      REQUIRE(d.Normlinf() == 0.0);
   }

   d = dev.data;
   d -= ser.data;
   CAPTURE(d.Normlinf());
   REQUIRE(d.Normlinf() <= 1e-14 * scale);
}

/**
 * @brief HDGFaceSideBatched()'s device weights are its host weights, and its
 * blocks the integrator's one-sided matrices.
 *
 * The one-sided route serves both reconstructions. Every interior face on
 * both sides and every boundary face, so that both sides' slots and the
 * boundary's are all read.
 */
TEST_CASE("The batched one-sided HDG face matrices weigh on the device",
          "[DarcyHybridization][BatchedLinAlg][GPU]")
{
   using namespace darcy_batched_weights;

   const int geom = GENERATE(0, 1, 2, 3);
   const int order = GENERATE(0, 1, 2);
   const bool nodes = GENERATE(false, true);
   const Weight w = GENERATE(Weight::Plain, Weight::Constant, Weight::GridFn,
                             Weight::TauConstant, Weight::TauInvR);
   CAPTURE(GeomName(geom), order, nodes, Name(w));
   if (geom >= 2 && order == 2) { return; }

   Mesh mesh = MakeMesh(geom, nodes);
   const int dim = mesh.Dimension();
   L2_FECollection p_coll(order, dim);
   DG_Interface_FECollection t_coll(order, dim);
   FiniteElementSpace Wh(&mesh, &p_coll), Mh(&mesh, &t_coll);
   Weights ws(mesh);

   std::unique_ptr<HDGDiffusionIntegrator> d_int(ws.Make(w, false));
   std::unique_ptr<HDGDiffusionIntegrator> h_int(ws.Make(w, true));
   Array<BilinearFormIntegrator*> d_list({d_int.get()}), h_list({h_int.get()});
   REQUIRE(HDGFaceSideCanBatch(Mh, Wh, d_list));
   REQUIRE(HDGFaceSideCanBatch(Mh, Wh, h_list));
   REQUIRE(HDGFaceWeightsOnDevice(*d_int, Wh));
   REQUIRE_FALSE(HDGFaceWeightsOnDevice(*h_int, Wh));

   Array<int> faces, sides;
   for (int f = 0; f < mesh.GetNumFaces(); f++)
   {
      faces.Append(f);
      sides.Append(0);
      if (mesh.FaceIsInterior(f))
      {
         faces.Append(f);
         sides.Append(1);
      }
   }

   Vector bd, bh;
   HDGFaceSideBatched(Mh, Wh, d_list, faces, sides, bd);
   HDGFaceSideBatched(Mh, Wh, h_list, faces, sides, bh);
   REQUIRE(bd.Size() == bh.Size());
   REQUIRE(bd.Size() > 0);
   bd.HostRead();
   bh.HostRead();

   // The integrator's own one-sided matrix, item by item.
   Vector ref(bd.Size());
   {
      const int NI = faces.Size(), N = (int)std::lround(
                                          std::sqrt((double)(bd.Size() / NI)));
      DenseMatrix elmat;
      for (int k = 0; k < NI; k++)
      {
         FaceElementTransformations *ftr =
            mesh.GetFaceElementTransformations(faces[k]);
         const int el = sides[k] ? ftr->Elem2No : ftr->Elem1No;
         d_int->AssembleHDGFaceMatrix(sides[k], *Mh.GetFaceElement(faces[k]),
                                      *Wh.GetFE(el), *ftr, elmat);
         REQUIRE(elmat.Height() == N);
         for (int j = 0; j < N * N; j++) { ref(j + N * N * k) = elmat.GetData()[j]; }
      }
   }

   const real_t scale = ref.Normlinf();
   REQUIRE(scale > 1e-6);
   Vector d(bd);
   d -= bh;
   CAPTURE(d.Normlinf(), scale);
   if (Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK))
   {
      REQUIRE(d.Normlinf() <= 1e-14 * scale);
   }
   else
   {
      REQUIRE(d.Normlinf() == 0.0);
   }
   d = bd;
   d -= ref;
   CAPTURE(d.Normlinf());
   REQUIRE(d.Normlinf() <= 1e-14 * scale);
   REQUIRE((mesh.GetNodes() != nullptr) == nodes);
}

/**
 * @brief The batched element mass weighs a GridFunctionCoefficient on the
 * device, and gets the per-element blocks.
 *
 * meq's flux mass is (R q, v), and R is the coordinate field's first
 * component. Taken through HDGCoefficientOnDevice() and
 * HDGElementGeometry, so no host loop and no ElementTransformation.
 */
TEST_CASE("The batched HDG element mass weighs a GridFunctionCoefficient on "
          "the device", "[DarcyHybridization][BatchedLinAlg][GPU]")
{
   using namespace darcy_batched_weights;

   const int geom = GENERATE(0, 1, 2, 3);
   const int order = GENERATE(0, 1, 2);
   const bool nodes = GENERATE(false, true);
   CAPTURE(GeomName(geom), order, nodes);
   if (geom >= 2 && order == 2) { return; }

   Mesh mesh = MakeMesh(geom, nodes);
   const int dim = mesh.Dimension();
   L2_FECollection u_coll(order, dim, BasisType::GaussLobatto);
   FiniteElementSpace Vh(&mesh, &u_coll, dim);
   Weights ws(mesh);

   VectorMassIntegrator dev_m(*ws.R), host_m(*ws.host_R);
   Array<BilinearFormIntegrator*> d_list({&dev_m}), h_list({&host_m});
   REQUIRE(HDGElementMassCanBatch(Vh, d_list));
   REQUIRE(HDGElementMassOnDevice(dev_m, Vh));
   REQUIRE_FALSE(HDGElementMassOnDevice(host_m, Vh));

   Vector md, mh;
   HDGElementMassBatched(Vh, d_list, md);
   HDGElementMassBatched(Vh, h_list, mh);
   REQUIRE(md.Size() == mh.Size());
   md.HostRead();
   mh.HostRead();

   // Against the integrator's own element matrices.
   const int N = Vh.GetFE(0)->GetDof() * dim;
   Vector ref(md.Size());
   DenseMatrix elmat;
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      dev_m.AssembleElementMatrix(*Vh.GetFE(e),
                                  *mesh.GetElementTransformation(e), elmat);
      REQUIRE(elmat.Height() == N);
      for (int j = 0; j < N * N; j++) { ref(j + N * N * e) = elmat.GetData()[j]; }
   }
   const real_t scale = ref.Normlinf();
   REQUIRE(scale > 1e-6);
   Vector d(md);
   d -= ref;
   CAPTURE(d.Normlinf(), scale);
   REQUIRE(d.Normlinf() <= 1e-14 * scale);
   d = md;
   d -= mh;
   CAPTURE(d.Normlinf());
   REQUIRE(d.Normlinf() <= 1e-14 * scale);
   REQUIRE((mesh.GetNodes() != nullptr) == nodes);
}

/**
 * @brief HDGRuleStabilization is its rule, and the device predicates say
 * what they admit.
 */
TEST_CASE("HDGRuleStabilization and the device predicates",
          "[DarcyHybridization][BatchedLinAlg]")
{
   using namespace darcy_batched_weights;

   Mesh mesh = Mesh::MakeCartesian2D(3, 3, Element::TRIANGLE, true, 2.0, 1.0);
   HDGStabilizationRule r;
   r.c0 = 3.0;
   r.coord = 0;
   r.floor = 0.5;
   HDGRuleStabilization st(r);

   HDGStabilizationRule got;
   REQUIRE(st.IsConstant());
   REQUIRE(st.GetRule(got));
   REQUIRE(got.c0 == r.c0);
   REQUIRE(got.coord == r.coord);
   REQUIRE(got.floor == r.floor);
   NoRule nr(r);
   REQUIRE_FALSE(nr.GetRule(got));

   // tau0/max(R, R_f) at points either side of the floor, through Eval() as a
   // face routine calls it.
   ElementTransformation *T = mesh.GetElementTransformation(0);
   const IntegrationRule &ir = IntRules.Get(Geometry::TRIANGLE, 3);
   Vector x;
   int below = 0, above = 0;
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      T = mesh.GetElementTransformation(e);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         T->SetIntPoint(&ir.IntPoint(q));
         T->Transform(ir.IntPoint(q), x);
         const real_t want = 3.0 / std::max(x(0), real_t(0.5));
         (x(0) < 0.5 ? below : above)++;
         REQUIRE(st.Eval(0.0, 0.0, 0.0, 0.0, *T) == want);
      }
   }
   REQUIRE(below > 0);
   REQUIRE(above > 0);
   HDGStabilizationRule c;
   c.c0 = 1.25;
   REQUIRE(HDGRuleStabilization(c).Eval(9.0, 9.0, 9.0, 9.0, *T) == 1.25);

   // HDGCoefficientOnDevice().
   ConstantCoefficient one(1.0);
   FunctionCoefficient fc([](const Vector &v) { return v(0); });
   H1_FECollection h1(1, 2);
   L2_FECollection l2(2, 2);
   FiniteElementSpace vfes(&mesh, &h1, 2), sfes(&mesh, &l2);
   GridFunction X(&vfes), s(&sfes);
   X = 0.0;
   s = 0.0;
   GridFunctionCoefficient gx(&X, 2), gs(&s), gbad(&X, 3);
   Mesh other = Mesh::MakeCartesian2D(3, 3, Element::TRIANGLE);
   FiniteElementSpace ofes(&other, &l2);
   GridFunction og(&ofes);
   og = 0.0;
   GridFunctionCoefficient go(&og);
   REQUIRE(HDGCoefficientOnDevice(&one, mesh));
   REQUIRE(HDGCoefficientOnDevice(&gx, mesh));
   REQUIRE(HDGCoefficientOnDevice(&gs, mesh));
   REQUIRE_FALSE(HDGCoefficientOnDevice(nullptr, mesh));
   REQUIRE_FALSE(HDGCoefficientOnDevice(&fc, mesh));
   REQUIRE_FALSE(HDGCoefficientOnDevice(&gbad, mesh));
   REQUIRE_FALSE(HDGCoefficientOnDevice(&go, mesh));
   REQUIRE(mesh.GetNodes() == nullptr);
}

/**
 * @brief HDGElementPointsAndWeights() is Transform() and ip.weight * Weight(),
 * bit for bit on the host.
 *
 * The caller-integrand residual and gradient routes evaluate at these points
 * and weigh with these weights (DarcyHybridization's element quadrature
 * tables). Warped meshes of the four shapes with and without nodes, quadratic
 * nodes, and a surface mesh in 3-D, against a loop over
 * Mesh::GetElementTransformation().
 */
TEST_CASE("Element points and weights on the device are the transformations'",
          "[DarcyHybridization][BatchedLinAlg][GPU]")
{
   using namespace darcy_batched_weights;

   const int geom = GENERATE(0, 1, 2, 3, 4);
   const int nodes = GENERATE(0, 1, 2);
   CAPTURE(geom, nodes);

   Mesh mesh;
   if (geom < 4) { mesh = MakeMesh(geom, nodes > 0); }
   else
   {
      // A surface in 3-D: the weight is sqrt(det(J^T J)).
      mesh = Mesh::MakeCartesian2D(3, 2, Element::QUADRILATERAL, true, 1.0, 0.7,
                                   false);
      mesh.SetCurvature(1, false, 3, Ordering::byVDIM);
      mesh.Transform([](const Vector &x, Vector &y)
      {
         y(0) = x(0) + 0.1 * x(1);
         y(1) = x(1);
         y(2) = 0.3 * x(0) * x(1) + 0.2 * x(0);
      });
      if (nodes == 0) { return; }
   }
   if (nodes == 2) { mesh.SetCurvature(2); }
   REQUIRE(HDGElementPointsOnDevice(mesh));

   const int dim = mesh.Dimension(), sdim = mesh.SpaceDimension();
   const int NE = mesh.GetNE();
   const IntegrationRule &ir = IntRules.Get(mesh.GetElementBaseGeometry(0), 5);
   const int NQ = ir.GetNPoints();
   Vector X, W;
   HDGElementPointsAndWeights(mesh, ir, X, W);
   REQUIRE(X.Size() == sdim * NQ * NE);
   REQUIRE(W.Size() == NQ * NE);
   const real_t *px = X.HostRead(), *pw = W.HostRead();

   const bool gpu = Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK);
   Vector xq;
   real_t err_x = 0.0, err_w = 0.0, scale = 0.0;
   for (int e = 0; e < NE; e++)
   {
      ElementTransformation *T = mesh.GetElementTransformation(e);
      for (int q = 0; q < NQ; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T->SetIntPoint(&ip);
         T->Transform(ip, xq);
         for (int i = 0; i < sdim; i++)
         {
            err_x = std::max(err_x, std::abs(px[i + sdim * (q + NQ * e)] - xq(i)));
            scale = std::max(scale, std::abs(xq(i)));
         }
         err_w = std::max(err_w, std::abs(pw[q + NQ * e] - ip.weight * T->Weight()));
      }
   }
   CAPTURE(err_x, err_w, scale);
   REQUIRE(scale > 0.5);
   if (gpu)
   {
      REQUIRE(err_x <= 1e-14 * scale);
      REQUIRE(err_w <= 1e-14);
   }
   else
   {
      REQUIRE(err_x == 0.0);
      REQUIRE(err_w == 0.0);
   }
   REQUIRE((mesh.GetNodes() != nullptr) == (nodes > 0));

   // What it refuses: NURBS, and more than one element geometry.
   if (geom == 0 && nodes == 0)
   {
      Mesh nurbs("../../data/disc-nurbs.mesh");
      REQUIRE_FALSE(HDGElementPointsOnDevice(nurbs));
      Mesh mixed("../../data/star-mixed.mesh");
      REQUIRE_FALSE(HDGElementPointsOnDevice(mixed));
   }
}
