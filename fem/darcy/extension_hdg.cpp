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

#include "extension_hdg.hpp"

#include <limits>
#include "../geom.hpp"
#include "../../mesh/submesh/submesh.hpp"

#include <algorithm>
#include <array>
#include <exception>
#include <map>
#include <vector>

namespace mfem
{

namespace
{

/** @brief Whether the level-set setup loops below may run threaded.

    The caller's promise about @a phi first, since nothing here can audit a
    std::function. Then OpenMP, without which a pragma is a serial loop. Then
    no Device: MFEM's memory bookkeeping is live only when one is configured,
    and it is not thread-safe -- a Vector allocated inside the region would
    reach it. A threaded element loop under Device("debug") was measured to
    abort in MmuHostMemorySpace::Dealloc for exactly that reason, on the NPC
    branch. And, where the loop maps reference points through
    Mesh::GetElementTransformation(), no nodes: with nodes that call reads
    them through the nodal space's one-argument GetElementVDofs(), which
    writes the space's shared DofTransformation. */
bool LevelSetLoopsThread(const Mesh &mesh, bool phi_thread_safe,
                         bool maps_points)
{
#ifdef MFEM_USE_OPENMP
   if (!phi_thread_safe) { return false; }
   if (Device::Allows(Backend::DEVICE_MASK)) { return false; }
   if (maps_points && mesh.GetNodes()) { return false; }
   return true;
#else
   (void)mesh; (void)phi_thread_safe; (void)maps_points;
   return false;
#endif
}

/** @brief The first exception thrown inside an OpenMP region, carried out of
    it. A throw may not leave the region -- it is std::terminate, even with
    the region's `if` false -- so each iteration catches, the first is kept,
    and it is rethrown after the region closes. */
struct LevelSetLoopTrap
{
   std::exception_ptr first;
   void Capture()
   {
#ifdef MFEM_USE_OPENMP
      #pragma omp critical (LevelSetLoopTrap)
#endif
      { if (!first) { first = std::current_exception(); } }
   }
   void Rethrow() { if (first) { std::rethrow_exception(first); } }
};

} // namespace


void TransferPath::Endpoint(FaceElementTransformations &FTr,
                            const IntegrationPoint &ip, Vector &xbar) const
{
   const int sdim = FTr.GetSpaceDim();
   Vector x(sdim), nor(sdim);

   FTr.SetAllIntPoints(&ip);
   FTr.Transform(ip, x);

   if (sdim == 1)
   {
      // The face is a point, whose own reference coordinate is always zero;
      // which end of the element it is has to be read from the element's.
      nor(0) = 2. * FTr.GetElement1IntPoint().x - 1.;
   }
   else
   {
      CalcOrtho(FTr.Jacobian(), nor);
      nor /= nor.Norml2();
   }

   Endpoint(x, nor, xbar);
}

VectorPositionFunction ClosestPointPath::Sphere(const Vector &c, real_t R)
{
   const Vector cc(c);
   return [cc, R](const Vector &x, Vector &xbar)
   {
      const int dim = x.Size();
      MFEM_ASSERT(cc.Size() == dim, "dimension mismatch");
      xbar.SetSize(dim);

      real_t r = 0.;
      for (int d = 0; d < dim; d++)
      {
         const real_t s = x(d) - cc(d);
         r += s * s;
      }
      r = sqrt(r);
      MFEM_ASSERT(r > 0., "the centre of the sphere has no closest point on it");

      for (int d = 0; d < dim; d++)
      {
         xbar(d) = cc(d) + R * (x(d) - cc(d)) / r;
      }
   };
}

namespace
{

/// What the format resolves of a crossing along a ray of @a length from @a x,
/// which is where FirstCrossing()'s bisection stops.
real_t BisectionResolution(const Vector &x, real_t length)
{
   return std::numeric_limits<real_t>::epsilon() * (x.Normlinf() + length);
}

/// How far outside @f$\Gamma@f$ an origin may lie and still count as on it:
/// @a tol times @a length, or BisectionResolution(), whichever is the larger.
/// This is the only thing @a tol sets.
real_t CrossingResolution(const Vector &x, real_t length, real_t tol)
{
   return std::max(tol * length, BisectionResolution(x, length));
}

/** @brief March from @a x along the unit direction @a dir and bisect the first
    crossing of the zero level set, returning its parameter in @a t.

    Returns false when there is no crossing within @a length -- which is a real
    outcome and not an error: a ray can leave the domain without ever meeting
    the boundary, and on a shape with a thin feature many of them do.

    An origin outside @f$\Gamma@f$ by less than CrossingResolution() -- the
    level set positive there and not one resolution step back along the ray --
    is ON @f$\Gamma@f$, and gets @a t = 0. Where @f$\Gamma@f$ meets the
    background's boundary at a vertex, or a face lies along it, the level set
    is zero by construction and its computed sign is round-off: at +1e-16 every
    ray used to be refused and the caller aborted (gffp). A point outside by
    more than that is still refused. */
bool FirstCrossing(const PositionFunction &phi, const Vector &x,
                   const Vector &dir, real_t length, int steps, real_t tol,
                   int max_iter, real_t &t)
{
   const int dim = x.Size();
   Vector y(dim);
   auto phi_at = [&](real_t u) -> real_t
   {
      for (int d = 0; d < dim; d++) { y(d) = x(d) + u * dir(d); }
      return phi(y);
   };

   if (phi_at(0.) > 0.)
   {
      if (phi_at(-CrossingResolution(x, length, tol)) > 0.) { return false; }
      t = 0.;
      return true;
   }

   real_t ta = 0., tb = -1.;
   const real_t du = length / steps;
   for (int i = 1; i <= steps; i++)
   {
      const real_t u = i * du;
      if (phi_at(u) > 0.) { tb = u; break; }
      ta = u;
   }
   if (tb < 0.) { return false; }

   // BISECT UNTIL THE MIDPOINT STOPS MOVING, WHATEVER tol IS. The endpoint
   // is differenced along the face when a family has no closed-form
   // derivative, and a bisection stopped at a bracket of tol * length makes
   // it a staircase in the face point, with steps up to that bracket: at
   // 1e-13 times six element sizes, 2.4e-13, which a central difference at a
   // physical step near 1e-7 turns into an error near 1e-6 in da/dxi. MEQ
   // measured psi_h moving by 1e-6 to 1e-5 with the step alone at tol =
   // 1e-13, and by 1e-8 at tol = 0, with no measurable change in wall time.
   // So tol keeps only its other job, CrossingResolution()'s allowance for an
   // origin just outside Gamma. In single precision the old bracket target
   // was below what the format can represent, which is where the midpoint
   // test came from (MEQ: 27 s of setup against 9 with every ray run to
   // max_iter); the default max_iter, 100, is ample for double, whose
   // bisection needs about 45 halvings from a 1/32 bracket.
   for (int it = 0; it < max_iter; it++)
   {
      const real_t tm = 0.5 * (ta + tb);
      if (tm <= ta || tm >= tb) { break; }
      if (phi_at(tm) > 0.) { tb = tm; }
      else { ta = tm; }
   }

   t = 0.5 * (ta + tb);
   return true;
}

/** @brief The outward unit normal @a n of the face of @a FTr at its set
    point, as TransferPath::Endpoint() takes it, and its derivative @a dn
    along the face's reference coordinates, SDIM x (SDIM - 1).

    Zero on a face whose Jacobian is constant. On a curved one it comes from
    the face map's Hessian, whose columns are ordered xx, xy, yy: with
    @f$\nu@f$ CalcOrtho()'s unnormalised normal, @f$n' = (\nu' - n(n\cdot
    \nu'))/|\nu|@f$, where @f$\nu' = (x_{\xi\xi,1}, -x_{\xi\xi,0})@f$ in
    two dimensions and @f$\nu = x_{\xi_0}\times x_{\xi_1}@f$ is
    differentiated by the product rule in three. */
void FaceNormalDerivative(FaceElementTransformations &FTr, Vector &n,
                          DenseMatrix &dn)
{
   const int sdim = FTr.GetSpaceDim(), fdim = sdim - 1;
   Vector nu(sdim);
   CalcOrtho(FTr.Jacobian(), nu);
   const real_t len = nu.Norml2();
   n = nu;
   n /= len;

   dn.SetSize(sdim, fdim);
   dn = 0.;
   if (FTr.OrderJ() == 0) { return; }

   const DenseMatrix &J = FTr.Jacobian();
   const DenseMatrix &H = FTr.Hessian();
   Vector dnu(sdim);
   for (int k = 0; k < fdim; k++)
   {
      if (sdim == 2)
      {
         dnu(0) = H(1, 0);
         dnu(1) = -H(0, 0);
      }
      else
      {
         // d(c0 x c1) = dc0 x c1 + c0 x dc1, with c_j = J(:, j) and
         // d c_j / d xi_k = H(:, j + k).
         auto cross_add = [&](const real_t *a, const real_t *b)
         {
            dnu(0) += a[1] * b[2] - a[2] * b[1];
            dnu(1) += a[2] * b[0] - a[0] * b[2];
            dnu(2) += a[0] * b[1] - a[1] * b[0];
         };
         const real_t c0[3] = { J(0, 0), J(1, 0), J(2, 0) };
         const real_t c1[3] = { J(0, 1), J(1, 1), J(2, 1) };
         const real_t d0[3] = { H(0, k), H(1, k), H(2, k) };
         const real_t d1[3] = { H(0, k + 1), H(1, k + 1), H(2, k + 1) };
         dnu = 0.;
         cross_add(d0, c1);
         cross_add(c0, d1);
      }
      const real_t nd = n * dnu;
      for (int d = 0; d < sdim; d++) { dn(d, k) = (dnu(d) - nd * n(d)) / len; }
   }
}

/** @brief @f$\partial a/\partial\xi@f$ for the endpoint @a xbar @f$= x +
    s\,t@f$ of a ray along the unit direction @a t, given the face's
    Jacobian @a dx and the direction's derivative @a dt, both SDIM x FDIM.

    Differentiating @f$\phi(x + s\,t) = 0@f$ along the face gives
    @f$s' = -\nabla\phi\cdot(x' + s\,t')/(\nabla\phi\cdot t)@f$, and
    @f$a' = x' + s\,t' + s'\,t@f$.

    **A ray grazing @f$\Gamma@f$ is refused, and aborts.** As
    @f$\nabla\phi\cdot t \to 0@f$, @f$s'@f$ diverges: the crossing slides
    along @f$\Gamma@f$ without bound as the start moves, and past tangency
    the ray misses that piece of @f$\Gamma@f$ and jumps to a farther one.
    That is the path map's, not the formula's, so a central difference there
    is no better, only quieter. Refused at @f$|\nabla\phi\cdot t| \le
    \sqrt\epsilon|\nabla\phi|@f$, where the quotient has no digits left,
    and for a gradient that is not finite. CheckPathIncidence() finds the
    shallow paths before a solve does. @a be is FTr.ElementNo, for the
    message. */
void RayEndpointJacobian(const VectorPositionFunction &grad_phi,
                         const Vector &xbar, const Vector &t, real_t s,
                         const DenseMatrix &dx, const DenseMatrix &dt,
                         int be, DenseMatrix &dadxi)
{
   const int sdim = xbar.Size(), fdim = dx.Width();
   Vector g(sdim);
   grad_phi(xbar, g);
   MFEM_VERIFY(g.Size() == sdim, "the level set's gradient has "
               << g.Size() << " components in " << sdim << " dimensions");
   const real_t gt = g * t;
   const real_t eps = std::numeric_limits<real_t>::epsilon();
   MFEM_VERIFY(std::abs(gt) > std::sqrt(eps) * g.Norml2() && g.CheckFinite() == 0,
               "the path from boundary element (or face) " << be << " meets "
               "Gamma at "
               "xbar = (" << xbar(0) << ", " << xbar(1)
               << (sdim > 2 ? ", " + std::to_string(xbar(2)) : std::string())
               << ") at a grazing angle, |grad phi . t| / |grad phi| = "
               << std::abs(gt) / g.Norml2() << " (or the gradient is not "
               "finite): its endpoint has no derivative along the face there. "
               "The family's directions need changing; CheckPathIncidence() "
               "finds such paths before a solve.");

   dadxi.SetSize(sdim, fdim);
   Vector v(sdim);
   for (int i = 0; i < fdim; i++)
   {
      for (int d = 0; d < sdim; d++) { v(d) = dx(d, i) + s * dt(d, i); }
      const real_t ds = -(g * v) / gt;
      for (int d = 0; d < sdim; d++) { dadxi(d, i) = v(d) + ds * t(d); }
   }
}

} // namespace

std::function<void(const Vector &, DenseMatrix &)>
ClosestPointPath::SphereJacobian(const Vector &c, real_t R)
{
   const Vector cc(c);
   return [cc, R](const Vector &x, DenseMatrix &J)
   {
      const int dim = x.Size();
      MFEM_ASSERT(cc.Size() == dim, "dimension mismatch");
      J.SetSize(dim);

      Vector u(dim);
      real_t r = 0.;
      for (int d = 0; d < dim; d++)
      {
         u(d) = x(d) - cc(d);
         r += u(d) * u(d);
      }
      r = std::sqrt(r);
      MFEM_VERIFY(r > 0., "the closest point to a sphere's own centre is not "
                  "defined, so neither is the Jacobian there");
      u /= r;

      // d a/d x = ( R/|x-c| )( I - u u^T ).  The projector annihilates the
      // radial direction because every point of a ray shares a closest point.
      const real_t s = R / r;
      for (int i = 0; i < dim; i++)
         for (int j = 0; j < dim; j++)
         {
            J(i, j) = s * ((i == j ? 1. : 0.) - u(i) * u(j));
         }
   };
}

bool ClosestPointPath::EndpointJacobian(FaceElementTransformations &FTr,
                                        const IntegrationPoint &ip,
                                        DenseMatrix &dadxi) const
{
   if (!dcp) { return false; }

   const int sdim = FTr.GetSpaceDim();
   const int fdim = sdim - 1;
   if (fdim < 1) { return false; }

   Vector x(sdim);
   FTr.SetAllIntPoints(&ip);
   FTr.Transform(ip, x);

   // The chain rule: d a/d xi = ( d a/d x )( d x/d xi ), the second factor
   // being the face's own Jacobian.  Exact, where the difference is not.
   DenseMatrix dadx(sdim);
   dcp(x, dadx);

   dadxi.SetSize(sdim, fdim);
   Mult(dadx, FTr.Jacobian(), dadxi);
   return true;
}

void LevelSetPath::Endpoint(const Vector &x, const Vector &n,
                            Vector &xbar) const
{
   const int dim = x.Size();
   xbar.SetSize(dim);

   real_t t;
   MFEM_VERIFY(FirstCrossing(phi, x, n, search_length, search_steps, tol,
                             max_iter, t),
               "the ray along the outward normal never meets the level set "
               "within the search length " << search_length << ": either it is "
               "too short, the normal does not leave the domain, or the "
               "boundary has a feature the normal passes outside -- for which "
               "VertexConePath is the family to use");

   for (int d = 0; d < dim; d++) { xbar(d) = x(d) + t * n(d); }
}

bool LevelSetPath::EndpointJacobian(FaceElementTransformations &FTr,
                                    const IntegrationPoint &ip,
                                    DenseMatrix &dadxi) const
{
   const int sdim = FTr.GetSpaceDim();
   if (!grad_phi || sdim < 2) { return false; }

   // The point, the normal and the crossing exactly as Endpoint() takes them,
   // so the derivative is of the map the quadrature evaluates.
   Vector x(sdim), n(sdim), xbar(sdim);
   DenseMatrix dn;
   FTr.SetAllIntPoints(&ip);
   FTr.Transform(ip, x);
   FaceNormalDerivative(FTr, n, dn);
   const DenseMatrix dx(FTr.Jacobian());
   real_t s;
   MFEM_VERIFY(FirstCrossing(phi, x, n, search_length, search_steps, tol,
                             max_iter, s),
               "the ray along the outward normal never meets the level set "
               "within the search length " << search_length);
   for (int d = 0; d < sdim; d++) { xbar(d) = x(d) + s * n(d); }

   RayEndpointJacobian(grad_phi, xbar, n, s, dx, dn,
                       FTr.ElementNo, dadxi);
   return true;
}

VertexConePath::VertexConePath(const Mesh &mesh_, int gamma_h_attr,
                               PositionFunction phi_, real_t search_length_,
                               int n_rays_, int n_keep_, int search_steps_,
                               real_t tol_, int max_iter_, bool use_cone_,
                               bool phi_thread_safe_)
   : mesh(&mesh_), phi(std::move(phi_)), search_length(search_length_),
     n_rays(n_rays_), n_keep(n_keep_), search_steps(search_steps_),
     max_iter(max_iter_), tol(tol_)
{
   MFEM_VERIFY(mesh->Dimension() == 2 && mesh->SpaceDimension() == 2,
               "VertexConePath is a two-dimensional construction");
   MFEM_VERIFY(n_rays >= 2 && n_keep >= 1, "invalid search parameters");

   Mesh &m = const_cast<Mesh &>(*mesh);
   const int nfaces = m.GetNumFaces();

   tang.SetSize(4 * nfaces); tang = 0.;
   has_tangent.SetSize(nfaces); has_tangent = 0;

   // The outward normals of the faces of Gamma_h meeting at each vertex.
   std::map<int, Array<real_t>> vertex_normals;
   Array<int> vs;
   Vector nor(2);

   for (int be = 0; be < m.GetNBE(); be++)
   {
      if (m.GetBdrAttribute(be) != gamma_h_attr) { continue; }

      FaceElementTransformations *FTr = m.GetBdrFaceTransformations(be);
      if (!FTr) { continue; }

      IntegrationPoint mid;
      mid.Set1w(0.5, 1.0);
      FTr->SetAllIntPoints(&mid);
      CalcOrtho(FTr->Jacobian(), nor);
      nor /= nor.Norml2();

      m.GetFaceVertices(m.GetBdrElementFaceIndex(be), vs);
      for (int k = 0; k < vs.Size(); k++)
      {
         Array<real_t> &ns = vertex_normals[vs[k]];
         ns.Append(nor(0));
         ns.Append(nor(1));
      }
   }

   // The cone C(x) of CS-Extensions section 2.4.1, when there is a background
   // mesh to build it from. It is the angular hull of the BACKGROUND mesh's
   // edges at the vertex that leave D_h, and restricting the search to it is
   // what keeps the paths of neighbouring faces from crossing: a path stays
   // inside the fan of background elements at its own vertex, and those fans
   // tile. Without it only the half space H(x) is applied, the swept regions
   // of neighbouring faces overlap wherever the boundary carries a feature
   // thinner than a mesh width, and the flux loses its order while the
   // potential keeps it.
   //
   // The background mesh is not an extra argument: a SubMesh knows its parent
   // and its own vertex numbering within it. Handed anything else, there is no
   // cone and the behaviour is what it was.
   std::map<int, std::array<real_t, 2>> vertex_cone;   // centre, half width
   const SubMesh *sub = use_cone_ ? dynamic_cast<const SubMesh *>(mesh) : NULL;
   if (sub && sub->GetParent())
   {
      have_cone = true;
      const Mesh &bg = *sub->GetParent();
      const Array<int> &pv = sub->GetParentVertexIDMap();

      // Background edges incident on each parent vertex, once, as a flat
      // vertex-to-edge table: two passes over the edges in edge order, so each
      // vertex lists its edges in the order the std::map this replaces did.
      // That map cost a node allocation per vertex of the WHOLE background --
      // 244k elements in meq's case -- to serve the few Gamma_h touches.
      const int nbv = bg.GetNV();
      Array<int> eoff(nbv + 1), eat;
      eoff = 0;
      Array<int> ev;
      for (int e = 0; e < bg.GetNEdges(); e++)
      {
         const_cast<Mesh &>(bg).GetEdgeVertices(e, ev);
         eoff[ev[0] + 1]++;
         eoff[ev[1] + 1]++;
      }
      eoff.PartialSum();
      eat.SetSize(eoff[nbv]);
      {
         Array<int> fill(eoff);
         for (int e = 0; e < bg.GetNEdges(); e++)
         {
            const_cast<Mesh &>(bg).GetEdgeVertices(e, ev);
            eat[fill[ev[0]]++] = e;
            eat[fill[ev[1]]++] = e;
         }
      }

      for (auto &kv : vertex_normals)
      {
         const int sv = kv.first;
         if (sv >= pv.Size()) { continue; }
         const int bv = pv[sv];
         if (eoff[bv + 1] == eoff[bv]) { continue; }

         const real_t *c = bg.GetVertex(pv[sv]);
         const Array<real_t> &ns = kv.second;
         const int m_n = ns.Size() / 2;

         // Keep the edge directions that leave D_h through every face of
         // Gamma_h meeting here -- the same admissibility the half space
         // uses, applied to the background's own edges rather than to a
         // continuum of angles.
         std::vector<real_t> ang;
         for (int ei = eoff[bv]; ei < eoff[bv + 1]; ei++)
         {
            const int e = eat[ei];
            const_cast<Mesh &>(bg).GetEdgeVertices(e, ev);
            const int other = (ev[0] == pv[sv]) ? ev[1] : ev[0];
            const real_t *o = bg.GetVertex(other);
            real_t d0 = o[0] - c[0], d1 = o[1] - c[1];
            const real_t dn = std::sqrt(d0 * d0 + d1 * d1);
            if (dn < 1e-14) { continue; }
            d0 /= dn; d1 /= dn;

            // Against the MEAN normal, not against every one of them. The
            // stricter test empties the cone at exactly the vertices that need
            // it -- a reentrant corner has no direction leaving through both
            // its faces -- which was measured: it restricted 10 vertices and
            // moved neither the tiling error nor a single rate.
            real_t mx = 0., my = 0.;
            for (int i = 0; i < m_n; i++)
            {
               mx += ns[2 * i]; my += ns[2 * i + 1];
            }
            if (d0 * mx + d1 * my > 0.) { ang.push_back(atan2(d1, d0)); }
         }
         if (ang.empty()) { continue; }

         // The smallest arc containing them. Sorting and taking the largest
         // gap is what makes this right across the branch cut of atan2.
         std::sort(ang.begin(), ang.end());
         const int na = (int)ang.size();
         real_t best_gap = 2. * M_PI + ang[0] - ang[na - 1];
         int best_at = na - 1;
         for (int i = 0; i + 1 < na; i++)
         {
            const real_t g = ang[i + 1] - ang[i];
            if (g > best_gap) { best_gap = g; best_at = i; }
         }
         const real_t span = 2. * M_PI - best_gap;
         const real_t lo = ang[(best_at + 1) % na];
         // real_t() on the halves for the same reason as the clamps below,
         // and this one is a WARNING rather than an error -- narrowing in a
         // braced initialiser -- so a single-precision build passes it and
         // the report MEQ sent did not name it. -Werror or a stricter
         // compiler would refuse it.
         vertex_cone[sv] = {{lo + real_t(0.5) * span, real_t(0.5) * span}};
      }
   }

   n_gamma_h_vertices = (int)vertex_normals.size();

   // A direction at each of those vertices. This is where the constructor's
   // time goes -- a fan of rays per vertex, each a march and a root find
   // through the level set -- and each vertex is searched alone, so with the
   // caller's promise about phi it runs threaded. The vertices are taken out
   // of the map first because an OpenMP loop wants an index; the maps are
   // only READ inside the region. The tallies VertexDirection() keeps are
   // per thread and summed after, and a vertex that finds no ray is reported
   // after the region, since nothing may be thrown out of one.
   std::vector<int> vlist;
   std::vector<const Array<real_t>*> nlist;
   vlist.reserve(vertex_normals.size());
   nlist.reserve(vertex_normals.size());
   for (auto &kv : vertex_normals)
   {
      vlist.push_back(kv.first);
      nlist.push_back(&kv.second);
   }
   const int nvx = (int)vlist.size();
   std::vector<std::array<real_t, 2>> tv(nvx);
   std::vector<char> found(nvx, 0);
   int widened = 0, coned = 0, tighter = 0;
   const bool threaded = LevelSetLoopsThread(m, phi_thread_safe_, false);
   (void)threaded;   // read only by the pragma
   LevelSetLoopTrap trap;
#ifdef MFEM_USE_OPENMP
   #pragma omp parallel if (threaded) reduction(+:widened, coned, tighter)
#endif
   {
      Vector xv(2), tt(2);
      int counts[3] = {0, 0, 0};
#ifdef MFEM_USE_OPENMP
      // Dynamic: a vertex whose fan has to be widened costs several times
      // one that does not, and they cluster where Gamma_h is rough.
      #pragma omp for schedule(dynamic, 16)
#endif
      for (int k = 0; k < nvx; k++)
      {
         try
         {
            const real_t *c = m.GetVertex(vlist[k]);
            xv(0) = c[0]; xv(1) = c[1];
            real_t cc = 0., ch = -1.;
            auto ic = vertex_cone.find(vlist[k]);
            if (ic != vertex_cone.end()) { cc = ic->second[0]; ch = ic->second[1]; }
            found[k] = VertexDirection(xv, *nlist[k], cc, ch, tt, counts) ? 1 : 0;
            if (found[k]) { tv[k] = {{tt(0), tt(1)}}; }
         }
         catch (...) { trap.Capture(); }
      }
      widened += counts[0];
      coned += counts[1];
      tighter += counts[2];
   }
   trap.Rethrow();
   n_widened += widened;
   n_coned += coned;
   n_tighter += tighter;

   std::map<int, std::array<real_t, 2>> vertex_tangent;
   Vector x(2);
   for (int k = 0; k < nvx; k++)
   {
      const real_t *c = m.GetVertex(vlist[k]);
      MFEM_VERIFY(found[k],
                  "no ray from the vertex (" << c[0] << ", " << c[1]
                  << ") of Gamma_h reaches Gamma within the search length "
                  << search_length);
      vertex_tangent[vlist[k]] = tv[k];
   }

   // Per face, those two tangents in the order its reference coordinate visits
   // the vertices. Which vertex sits at xi = 0 is read off the transformation
   // rather than assumed to follow the mesh's own ordering.
   IntegrationPoint zero;
   zero.Set1w(0.0, 1.0);
   for (int be = 0; be < m.GetNBE(); be++)
   {
      if (m.GetBdrAttribute(be) != gamma_h_attr) { continue; }

      FaceElementTransformations *FTr = m.GetBdrFaceTransformations(be);
      if (!FTr) { continue; }

      const int f = m.GetBdrElementFaceIndex(be);
      m.GetFaceVertices(f, vs);
      MFEM_VERIFY(vs.Size() == 2, "a face of a two-dimensional mesh has two "
                  "vertices");

      FTr->SetAllIntPoints(&zero);
      FTr->Transform(zero, x);

      const real_t *c0 = m.GetVertex(vs[0]);
      const real_t d0 = (x(0) - c0[0]) * (x(0) - c0[0])
                        + (x(1) - c0[1]) * (x(1) - c0[1]);
      const real_t *c1 = m.GetVertex(vs[1]);
      const real_t d1 = (x(0) - c1[0]) * (x(0) - c1[0])
                        + (x(1) - c1[1]) * (x(1) - c1[1]);

      const int at0 = (d0 <= d1) ? vs[0] : vs[1];
      const int at1 = (d0 <= d1) ? vs[1] : vs[0];

      tang[4 * f + 0] = vertex_tangent[at0][0];
      tang[4 * f + 1] = vertex_tangent[at0][1];
      tang[4 * f + 2] = vertex_tangent[at1][0];
      tang[4 * f + 3] = vertex_tangent[at1][1];
      has_tangent[f] = 1;
   }
}

bool VertexConePath::VertexDirection(const Vector &x,
                                     const Array<real_t> &normals,
                                     real_t cone_centre, real_t cone_half,
                                     Vector &t,
                                     int counts[3]) const
{
   const int m = normals.Size() / 2;
   MFEM_VERIFY(m >= 1, "a vertex of Gamma_h with no face");

   // The admissible directions are those leaving D_h through every face
   // meeting here. In two dimensions each such condition is a half-circle, so
   // their intersection is the arc about the mean normal whose half-width is
   // pi/2 less the largest angle between the mean and any one of them.
   Vector mean(2);
   mean = 0.;
   for (int i = 0; i < m; i++)
   {
      mean(0) += normals[2 * i];
      mean(1) += normals[2 * i + 1];
   }
   const real_t mn = mean.Norml2();
   real_t centre_angle, half_width;
   if (mn > 1e-12)
   {
      mean /= mn;
      centre_angle = atan2(mean(1), mean(0));
      real_t worst = 0.;
      for (int i = 0; i < m; i++)
      {
         const real_t dot = mean(0) * normals[2 * i] + mean(1) * normals[2 * i + 1];
         // real_t() on the clamps, not bare -1. and 1. -- those literals are
         // doubles, so under MFEM_PRECISION = single std::max cannot deduce
         // against a real_t and this whole translation unit fails to compile.
         // Reported by MEQ, who build the integration tree that way, and the
         // idiom is estimators_hdg.cpp's. The clamp is there because a dot
         // product of two unit vectors can leave [-1, 1] by a round-off and
         // acos() of that is a NaN, which matters MORE in single, not less.
         worst = std::max(worst, real_t(acos(std::min(std::max(dot, real_t(-1.)),
                                                      real_t(1.)))));
      }
      half_width = M_PI / 2. - worst;
   }
   else
   {
      centre_angle = atan2(normals[1], normals[0]);
      half_width = -1.;
   }

   // The cone restricts the first pass and nothing else: it is a subset of the
   // half space by construction, so where it exists it is the tighter of the
   // two, and where it does not the arc is what it always was. The widenings
   // below deliberately leave both behind.
   if (cone_half > 0.)
   {
      if (half_width <= 0. || cone_half < half_width) { counts[2]++; }
      centre_angle = cone_centre;
      half_width = cone_half;
      counts[1]++;
   }

   // A vertex ON Gamma -- where Gamma meets the background's boundary, or
   // passes exactly through a mesh vertex, the level set is zero there and its
   // sign is round-off -- has a path of no length, and every outward ray meets
   // Gamma inside the bisection's bracket. The rays then tie, and the nearest
   // few below were the first few in the fan's order: +41 degrees off the
   // normal where a vertex 1e-6 inside got -29.5 (the unit test "a vertex of
   // Gamma_h lying on Gamma"), and on the unit tests' own disc a foot map that
   // ran backwards (see the note on the class). At +1e-16 no ray was marched
   // at all, and the constructor aborted (gffp). Such a vertex instead takes the LIMIT of the rule
   // below as it approaches Gamma: a crossing at s = |phi| / slope ranks the
   // rays by the slope of phi along them, and the mean of the nearest
   // endpoints weights each direction by 1 / slope. The switch is where the
   // bisection's error in s (BisectionResolution(), r) and the limit's own
   // (s times the curvature, taken on the scale of the search) are equal,
   // sqrt(r L); either side of it the two agree to that, and the direction
   // is continuous through it.
   const real_t phi0 = phi(x);
   const real_t eps = std::numeric_limits<real_t>::epsilon();
   const real_t on_gamma = std::sqrt(BisectionResolution(x, search_length)
                                     * search_length);
   const real_t ds = std::sqrt(eps) * search_length;

   // Search the admissible fan, then the half-circle about the mean normal,
   // then everything. The widenings leave Assumption P.1 behind and are
   // counted so that a driver can say so.
   const real_t widths[3] = { half_width, M_PI / 2., M_PI };
   for (int pass = 0; pass < 3; pass++)
   {
      const real_t w = widths[pass];
      if (w <= 0.) { continue; }
      const real_t use = (pass == 0) ? 0.98 * w : 0.98 * w;
      auto ray = [&](int i, Vector &d)
      {
         const real_t a = centre_angle
                          + ((n_rays == 1) ? 0.
                             : (-use + 2. * use * i / (n_rays - 1)));
         d(0) = cos(a); d(1) = sin(a);
      };

      // The slope of phi along each ray, steepest first; only rays along
      // which phi increases leave through Gamma.
      std::vector<std::pair<real_t, int>> slopes;
      Vector d(2), y(2);
      for (int i = 0; i < n_rays; i++)
      {
         ray(i, d);
         add(x, ds, d, y);
         const real_t slope = (phi(y) - phi0) / ds;
         if (slope > 0.) { slopes.push_back({slope, i}); }
      }
      std::sort(slopes.begin(), slopes.end(),
                [](const std::pair<real_t, int> &a,
                   const std::pair<real_t, int> &b)
      { return a.first > b.first; });

      if (!slopes.empty() && std::abs(phi0) <= on_gamma * slopes[0].first)
      {
         const int keep = std::min<int>(n_keep, slopes.size());
         t.SetSize(2);
         t = 0.;
         for (int k = 0; k < keep; k++)
         {
            ray(slopes[k].second, d);
            t.Add(1. / slopes[k].first, d);
         }
         const real_t len = t.Norml2();
         if (len <= 0.) { continue; }
         t /= len;
         if (pass > 0) { counts[0]++; }
         return true;
      }

      // Endpoints found, sorted by distance, keeping the nearest few.
      std::vector<std::pair<real_t, std::array<real_t, 2>>> hits;
      for (int i = 0; i < n_rays; i++)
      {
         ray(i, d);
         real_t s;
         if (!FirstCrossing(phi, x, d, search_length, search_steps, tol,
                            max_iter, s)) { continue; }
         hits.push_back({s, {{x(0) + s * d(0), x(1) + s * d(1)}}});
      }
      if (hits.empty()) { continue; }

      if (pass > 0) { counts[0]++; }

      std::sort(hits.begin(), hits.end(),
                [](const std::pair<real_t, std::array<real_t, 2>> &a,
                   const std::pair<real_t, std::array<real_t, 2>> &b)
      { return a.first < b.first; });

      const int keep = std::min<int>(n_keep, hits.size());
      real_t mx = 0., my = 0.;
      for (int i = 0; i < keep; i++)
      {
         mx += hits[i].second[0];
         my += hits[i].second[1];
      }
      mx /= keep; my /= keep;

      t.SetSize(2);
      t(0) = mx - x(0); t(1) = my - x(1);
      const real_t len = t.Norml2();
      if (len <= 0.) { continue; }
      t /= len;

      // One more shot along the mean, so that the endpoint lands on Gamma and
      // not merely near it.
      real_t s;
      if (!FirstCrossing(phi, x, t, search_length, search_steps, tol, max_iter,
                         s)) { continue; }
      return true;
   }

   return false;
}

void VertexConePath::Endpoint(const Vector &x, const Vector &n,
                              Vector &xbar) const
{
   MFEM_ABORT("VertexConePath is defined face by face: use the "
              "FaceElementTransformations overload. Falling back to the "
              "normal would reintroduce the two failures this family repairs.");
}

void VertexConePath::FaceDirection(FaceElementTransformations &FTr,
                                   const IntegrationPoint &ip, Vector &t,
                                   DenseMatrix *dt) const
{
   const int f = (FTr.ElementType == ElementTransformation::BDR_FACE)
                 ? mesh->GetBdrElementFaceIndex(FTr.ElementNo)
                 : FTr.ElementNo;
   MFEM_VERIFY(f >= 0 && f < has_tangent.Size() && has_tangent[f],
               "no path was built for this face: it is not on the boundary "
               "the family was constructed for");

   const real_t th = ip.x;
   t.SetSize(2);
   t(0) = (1. - th) * tang[4 * f + 0] + th * tang[4 * f + 2];
   t(1) = (1. - th) * tang[4 * f + 1] + th * tang[4 * f + 3];
   const real_t len = t.Norml2();
   MFEM_VERIFY(len > 1e-12, "the paths of the two vertices of a face oppose "
               "each other, so no direction can be interpolated between them");
   t /= len;

   if (dt)
   {
      // t = u/|u| with u linear in th: t' = (u' - t (t . u'))/|u|.
      const real_t du0 = tang[4 * f + 2] - tang[4 * f + 0];
      const real_t du1 = tang[4 * f + 3] - tang[4 * f + 1];
      const real_t tdu = t(0) * du0 + t(1) * du1;
      dt->SetSize(2, 1);
      (*dt)(0, 0) = (du0 - tdu * t(0)) / len;
      (*dt)(1, 0) = (du1 - tdu * t(1)) / len;
   }
}

void VertexConePath::Endpoint(FaceElementTransformations &FTr,
                              const IntegrationPoint &ip, Vector &xbar) const
{
   Vector t;
   FaceDirection(FTr, ip, t, nullptr);

   Vector x(2);
   FTr.SetAllIntPoints(&ip);
   FTr.Transform(ip, x);

   real_t s;
   MFEM_VERIFY(FirstCrossing(phi, x, t, search_length, search_steps, tol,
                             max_iter, s),
               "the interpolated path from a point of Gamma_h never meets "
               "Gamma within the search length " << search_length);

   xbar.SetSize(2);
   xbar(0) = x(0) + s * t(0);
   xbar(1) = x(1) + s * t(1);
}

bool VertexConePath::EndpointJacobian(FaceElementTransformations &FTr,
                                      const IntegrationPoint &ip,
                                      DenseMatrix &dadxi) const
{
   if (!grad_phi) { return false; }

   Vector t;
   DenseMatrix dt;
   FaceDirection(FTr, ip, t, &dt);

   Vector x(2), xbar(2);
   FTr.SetAllIntPoints(&ip);
   FTr.Transform(ip, x);
   const DenseMatrix dx(FTr.Jacobian());
   real_t s;
   MFEM_VERIFY(FirstCrossing(phi, x, t, search_length, search_steps, tol,
                             max_iter, s),
               "the interpolated path from a point of Gamma_h never meets "
               "Gamma within the search length " << search_length);
   xbar(0) = x(0) + s * t(0);
   xbar(1) = x(1) + s * t(1);

   RayEndpointJacobian(grad_phi, xbar, t, s, dx, dt,
                       FTr.ElementNo, dadxi);
   return true;
}

TiltedPath::TiltedPath(const TransferPath &base_, PositionFunction phi_,
                       real_t gamma_, real_t search_length_, int search_steps_,
                       real_t tol_, int max_iter_)
   : base(base_), phi(std::move(phi_)), gamma(gamma_),
     search_length(search_length_), search_steps(search_steps_), tol(tol_),
     max_iter(max_iter_)
{
   MFEM_VERIFY(gamma > 0. && gamma < 1., "TiltedPath: gamma must lie in "
               "(0, 1), not " << gamma);
}

bool TiltedPath::TiltDirection(const Vector &x, const Vector &n,
                               const Vector &xbar, Vector &t) const
{
   Vector m(xbar);
   m -= x;
   const real_t L = m.Norml2();
   if (L == 0.) { return false; }    // already on Gamma
   m /= L;
   const real_t c = m * n;
   if (c >= gamma) { return false; }

   // t = gamma n + sqrt(1 - gamma^2) w, with w the unit part of m orthogonal
   // to n. A path pointing straight back along -n has no such part, and is
   // replaced by the normal itself.
   Vector w(m);
   w.Add(-c, n);
   const real_t wn = w.Norml2();
   t = n;
   if (wn > 0.)
   {
      t *= gamma;
      t.Add(std::sqrt(1. - gamma * gamma) / wn, w);
   }
   return true;
}

bool TiltedPath::Tilt(const Vector &x, const Vector &n, Vector &xbar) const
{
   const int dim = x.Size();
   Vector t;
   if (!TiltDirection(x, n, xbar, t)) { return false; }

   real_t s;
   MFEM_VERIFY(FirstCrossing(phi, x, t, search_length, search_steps, tol,
                             max_iter, s),
               "TiltedPath: the tilted ray never meets the level set within "
               "the search length " << search_length);
   for (int d = 0; d < dim; d++) { xbar(d) = x(d) + s * t(d); }
   return true;
}

void TiltedPath::Endpoint(const Vector &x, const Vector &n, Vector &xbar) const
{
   base.Endpoint(x, n, xbar);
   Vector nn(n);
   nn /= nn.Norml2();
   Tilt(x, nn, xbar);
}

namespace
{
/// The physical point and the unit outward normal of a face point, read
/// before any family's Endpoint() is allowed to move the transformations.
void FacePointAndNormal(FaceElementTransformations &FTr,
                        const IntegrationPoint &ip, Vector &x, Vector &nor)
{
   const int sdim = FTr.GetSpaceDim();
   x.SetSize(sdim);
   nor.SetSize(sdim);
   FTr.SetAllIntPoints(&ip);
   FTr.Transform(ip, x);
   if (sdim == 1)
   {
      nor(0) = 2. * FTr.GetElement1IntPoint().x - 1.;
   }
   else
   {
      CalcOrtho(FTr.Jacobian(), nor);
      nor /= nor.Norml2();
   }
}
} // namespace

void TiltedPath::Endpoint(FaceElementTransformations &FTr,
                          const IntegrationPoint &ip, Vector &xbar) const
{
   Vector x, nor;
   FacePointAndNormal(FTr, ip, x, nor);
   base.Endpoint(FTr, ip, xbar);
   Tilt(x, nor, xbar);
}

bool TiltedPath::EndpointJacobian(FaceElementTransformations &FTr,
                                  const IntegrationPoint &ip,
                                  DenseMatrix &dadxi) const
{
   Vector x, nor, xbar, t;
   FacePointAndNormal(FTr, ip, x, nor);
   base.Endpoint(FTr, ip, xbar);
   if (!TiltDirection(x, nor, xbar, t))
   {
      return base.EndpointJacobian(FTr, ip, dadxi);
   }

   // Tilted. t is built from the wrapped path's direction m and the face
   // normal n, so t' needs the wrapped endpoint's derivative and n'.
   const int sdim = FTr.GetSpaceDim(), fdim = sdim - 1;
   if (!grad_phi || fdim < 1) { return false; }
   DenseMatrix dab;
   if (!base.EndpointJacobian(FTr, ip, dab)) { return false; }

   FTr.SetAllIntPoints(&ip);
   Vector n;
   DenseMatrix dn;
   FaceNormalDerivative(FTr, n, dn);
   const DenseMatrix dx(FTr.Jacobian());

   Vector m(xbar);
   m -= x;
   const real_t L = m.Norml2();
   m /= L;
   const real_t c = m * n;
   Vector w(m);
   w.Add(-c, n);
   const real_t wn = w.Norml2();
   const real_t sg = std::sqrt(1. - gamma * gamma);

   DenseMatrix dt(sdim, fdim);
   Vector dm(sdim), dw(sdim);
   for (int k = 0; k < fdim; k++)
   {
      // m = (a_b - x)/L: m' = (r' - m (m . r'))/L with r' = a_b' - x'.
      for (int d = 0; d < sdim; d++) { dm(d) = dab(d, k) - dx(d, k); }
      const real_t mr = m * dm;
      for (int d = 0; d < sdim; d++) { dm(d) = (dm(d) - mr * m(d)) / L; }
      if (wn == 0.)
      {
         for (int d = 0; d < sdim; d++) { dt(d, k) = dn(d, k); }
         continue;
      }
      // c = m . n, w = m - c n, t = gamma n + sg w/|w|.
      real_t dc = dm * n;
      for (int d = 0; d < sdim; d++) { dc += m(d) * dn(d, k); }
      for (int d = 0; d < sdim; d++)
      {
         dw(d) = dm(d) - dc * n(d) - c * dn(d, k);
      }
      real_t wdw = 0.;
      for (int d = 0; d < sdim; d++) { wdw += w(d) * dw(d); }
      wdw /= wn;
      for (int d = 0; d < sdim; d++)
      {
         dt(d, k) = gamma * dn(d, k) + sg * (dw(d) - wdw * w(d) / wn) / wn;
      }
   }

   real_t s;
   MFEM_VERIFY(FirstCrossing(phi, x, t, search_length, search_steps, tol,
                             max_iter, s),
               "TiltedPath: the tilted ray never meets the level set within "
               "the search length " << search_length);
   Vector a(x);
   a.Add(s, t);
   RayEndpointJacobian(grad_phi, a, t, s, dx, dt, FTr.ElementNo,
                       dadxi);
   return true;
}

ElementExtension::ElementExtension()
{
   // The default #NewtonElementProject solver projects every iterate back into
   // the reference element, which is precisely what must not happen here.
   inv_tr.SetSolverType(InverseElementTransformation::Newton);
   inv_tr.SetInitialGuessType(InverseElementTransformation::Center);
}

bool ElementExtension::TransformBack(const Vector &y,
                                     IntegrationPoint &ip) const
{
   // The reference point is written before it is classified, so an #Outside
   // return still carries the out-of-element coordinates that are wanted.
   const int res = inv_tr.Transform(y, ip);
   return res != InverseElementTransformation::Unknown;
}

real_t PathIntegral(const VectorPositionFunction &Cu, const Vector &x,
                    const Vector &xbar, const IntegrationRule &line_ir)
{
   const int dim = x.Size();
   Vector m(dim), y(dim), v(dim);

   // The unit tangent is m/|m| and ds = |m| dt, so the length cancels.
   subtract(xbar, x, m);

   real_t val = 0.;
   for (int q = 0; q < line_ir.GetNPoints(); q++)
   {
      const IntegrationPoint &qip = line_ir.IntPoint(q);
      for (int d = 0; d < dim; d++) { y(d) = x(d) + qip.x * m(d); }
      Cu(y, v);
      val += qip.weight * (v * m);
   }

   return val;
}

real_t PathTraceCoefficient::Eval(ElementTransformation &T,
                                  const IntegrationPoint &ip)
{
   FaceElementTransformations *FTr =
      dynamic_cast<FaceElementTransformations *>(&T);
   MFEM_VERIFY(FTr, "PathTraceCoefficient must be evaluated on a face: the "
               "path family may need the outward normal of Gamma_h");

   path.Endpoint(*FTr, ip, xbar);
   return g(xbar);
}

void HDGExtensionIntegrator::SetLineRule(const IntegrationRule *ir)
{
   if (ir)
   {
      // A rule on some other interval, or one that does not sum to one, is a
      // lifting scaled or shifted along the path, and every result built from
      // it is wrong without anything aborting.
      real_t wsum = 0.;
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         MFEM_VERIFY(ip.x >= 0. && ip.x <= 1.,
                     "the line rule's point " << i << " is at t = " << ip.x
                     << ", outside the path [0,1]");
         wsum += ip.weight;
      }
      MFEM_VERIFY(std::abs(wsum - 1.)
                  < 100 * std::numeric_limits<real_t>::epsilon(),
                  "the line rule's weights sum to " << wsum << ", not to one");
   }
   line_ir = ir;
}

void HDGExtensionIntegrator::AssembleFaceMatrix(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Trans, DenseMatrix &elmat)
{
   MFEM_VERIFY(Trans.Elem2No < 0,
               "the extension term lives on a boundary face only");

   const int dof = el1.GetDof();
   const int dim = Trans.GetSpaceDim();

   elmat.SetSize(dof * dim);
   elmat = 0.;

#ifdef MFEM_THREAD_SAFE
   Vector shape, shape_ext, nor, x, xbar, m, y, CTm;
   DenseMatrix Cmat, L;
   ElementExtension ext;
#endif
   shape.SetSize(dof);
   shape_ext.SetSize(dof);
   nor.SetSize(dim);
   m.SetSize(dim);
   y.SetSize(dim);
   CTm.SetSize(dim);
   L.SetSize(dof, dim);
   if (MC) { Cmat.SetSize(dim); }

   // The datum's linear form must integrate with this same rule; see
   // GetFaceRule().
   const IntegrationRule *ir = &GetFaceRule(el1, Trans.GetGeometryType());
   const IntegrationRule &lir = LineRule(el1);

   // The projection reads the element's transformation at its own points, so
   // it is formed before the face loop resets them.
   DenseMatrix P;
   if (projected_lift) { ProjectionMatrix(el1, *Trans.Elem1, P); }

   ext.SetElement(*Trans.Elem1);

   for (int q = 0; q < ir->GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir->IntPoint(q);
      Trans.SetAllIntPoints(&ip);

      // The scaled normal already carries the face Jacobian, so the face
      // weight is the quadrature weight alone.
      if (dim == 1)
      {
         nor(0) = 2 * Trans.GetElement1IntPoint().x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      el1.CalcPhysShape(*Trans.Elem1, shape);
      Trans.Transform(ip, x);

      // Endpoint() resets the transformations, which is why the shape and the
      // normal above are taken into arrays of our own first.
      path.Endpoint(Trans, ip, xbar);
      subtract(xbar, x, m);

      // The lifting of every basis function of the element, at this point of
      // the face.
      LiftBasis(el1, *Trans.Elem1, x, m, lir, ext, shape_ext, y, CTm, Cmat, L);
      if (projected_lift) { ApplyProjection(P, L); }

      const real_t w = sign * ip.weight;
      for (int di = 0; di < dim; di++)
         for (int i = 0; i < dof; i++)
         {
            const real_t ti = w * nor(di) * shape(i);
            for (int dj = 0; dj < dim; dj++)
               for (int j = 0; j < dof; j++)
               {
                  elmat(dof * di + i, dof * dj + j) += ti * L(j, dj);
               }
         }
   }
}

void HDGExtensionIntegrator::LiftBasis(
   const FiniteElement &el1, ElementTransformation &elem_tr, const Vector &x,
   const Vector &m, const IntegrationRule &lir, ElementExtension &ext_w,
   Vector &shape_w, Vector &y_w, Vector &CTm_w, DenseMatrix &Cmat_w,
   DenseMatrix &Lmat)
{
   const int dof = el1.GetDof();
   const int dim = m.Size();

   shape_w.SetSize(dof);
   y_w.SetSize(dim);
   CTm_w.SetSize(dim);
   if (MC) { Cmat_w.SetSize(dim); }

   // The unit tangent is m/|m| and ds = |m| dt, so the length of the path
   // cancels and never appears.
   Lmat.SetSize(dof, dim);
   Lmat = 0.;
   for (int t = 0; t < lir.GetNPoints(); t++)
   {
      const IntegrationPoint &tip = lir.IntPoint(t);
      for (int d = 0; d < dim; d++) { y_w(d) = x(d) + tip.x * m(d); }

      IntegrationPoint eip;
      MFEM_VERIFY(ext_w.TransformBack(y_w, eip),
                  "the inverse element transformation did not converge on "
                  "the extension of the element beyond Gamma_h");
      elem_tr.SetIntPoint(&eip);
      el1.CalcPhysShape(elem_tr, shape_w);

      // (C phi_j) . m = phi_j . (C^T m), and phi_j = shape_j e_d. With the
      // projected lifting C has already been applied inside the element.
      if (projected_lift)
      {
         CTm_w = m;
      }
      else if (MC)
      {
         MC->Eval(Cmat_w, elem_tr, eip);
         Cmat_w.MultTranspose(m, CTm_w);
      }
      else
      {
         CTm_w = m;
         CTm_w *= C->Eval(elem_tr, eip);
      }

      for (int d = 0; d < dim; d++)
      {
         const real_t wd = tip.weight * CTm_w(d);
         for (int j = 0; j < dof; j++)
         {
            Lmat(j, d) += wd * shape_w(j);
         }
      }
   }
}

void HDGExtensionIntegrator::ProjectionMatrix(const FiniteElement &el1,
                                              ElementTransformation &Tr,
                                              DenseMatrix &P)
{
   const int dof = el1.GetDof();
   const int dim = Tr.GetSpaceDim();

   // M_C as the flux mass form assembles it, rule and all, so that
   // (q_h, v) = (C u_h, v) holds for the system actually solved.
   DenseMatrix MCm;
   if (MC)
   {
      VectorMassIntegrator vmi(*MC);
      vmi.AssembleElementMatrix(el1, Tr, MCm);
   }
   else
   {
      VectorMassIntegrator vmi(*C);
      vmi.AssembleElementMatrix(el1, Tr, MCm);
   }
   MFEM_VERIFY(MCm.Height() == dof * dim, "unexpected C-weighted mass size");

   DenseMatrix M;
   MassIntegrator mi;
   mi.AssembleElementMatrix(el1, Tr, M);
   DenseMatrixInverse Minv(M);

   // P = blockdiag(M^-1) M_C, one dof x dof block per pair of components.
   P.SetSize(dof * dim);
   DenseMatrix blk(dof), res(dof);
   for (int a = 0; a < dim; a++)
      for (int b = 0; b < dim; b++)
      {
         MCm.GetSubMatrix(dof * a, dof * (a + 1), dof * b, dof * (b + 1), blk);
         Minv.Mult(blk, res);
         P.SetSubMatrix(dof * a, dof * b, res);
      }
}

void HDGExtensionIntegrator::ApplyProjection(const DenseMatrix &P,
                                             DenseMatrix &Lmat)
{
   const int dof = Lmat.Height(), dim = Lmat.Width();
   Vector l0(dof * dim), l1(dof * dim);
   for (int d = 0; d < dim; d++)
      for (int j = 0; j < dof; j++) { l0(dof * d + j) = Lmat(j, d); }
   P.MultTranspose(l0, l1);
   for (int d = 0; d < dim; d++)
      for (int j = 0; j < dof; j++) { Lmat(j, d) = l1(dof * d + j); }
}

real_t HDGExtensionIntegrator::ComputeLift(
   const FiniteElement &el1, FaceElementTransformations &Trans,
   const IntegrationPoint &ip, const Vector &elfun)
{
   MFEM_VERIFY(Trans.Elem2No < 0,
               "the extension term lives on a boundary face only");

   const int dof = el1.GetDof();
   const int dim = Trans.GetSpaceDim();

   MFEM_VERIFY(elfun.Size() == dof * dim,
               "the flux dofs of the element owning the face are expected, "
               "with vdim equal to the space dimension");

#ifdef MFEM_THREAD_SAFE
   Vector shape_ext, x, xbar, m, y, CTm;
   DenseMatrix Cmat, L;
   ElementExtension ext;
#endif
   m.SetSize(dim);

   Trans.SetAllIntPoints(&ip);
   Trans.Transform(ip, x);

   // Endpoint() resets the transformations, so nothing read from them before
   // this point may be relied on afterwards -- which is why the element is
   // handed to the extension only below.
   path.Endpoint(Trans, ip, xbar);
   subtract(xbar, x, m);

   DenseMatrix P;
   if (projected_lift) { ProjectionMatrix(el1, *Trans.Elem1, P); }
   ext.SetElement(*Trans.Elem1);
   LiftBasis(el1, *Trans.Elem1, x, m, LineRule(el1), ext, shape_ext, y, CTm,
             Cmat, L);
   if (projected_lift) { ApplyProjection(P, L); }

   real_t lift = 0.;
   for (int d = 0; d < dim; d++)
      for (int j = 0; j < dof; j++)
      {
         lift += L(j, d) * elfun(dof * d + j);
      }
   return lift;
}

void ExtensionDatumLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   MFEM_ABORT("ExtensionDatumLFIntegrator is a boundary-face integrator: add "
              "it with LinearForm::AddBdrFaceIntegrator()");
}

void ExtensionDatumLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   const int dim = el.GetDim();
   const int dof = el.GetDof();
#ifdef MFEM_THREAD_SAFE
   Vector shape, nor;
#endif
   shape.SetSize(dof);
   nor.SetSize(dim);
   elvect.SetSize(dim * dof);
   elvect = 0.0;

   // The extension integrator's rule, so that the two halves of the datum are
   // integrated alike; see HDGExtensionIntegrator::GetFaceRule().
   const IntegrationRule &ir = ext.GetFaceRule(el, Tr.GetGeometryType());

   for (int q = 0; q < ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      Tr.SetAllIntPoints(&ip);

      if (dim == 1)
      {
         nor(0) = 2 * Tr.GetElement1IntPoint().x - 1.0;
      }
      else
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }
      el.CalcPhysShape(*Tr.Elem1, shape);

      // g is evaluated last: a PathTraceCoefficient moves the transformations
      // to find the end of the path, and everything above reads them first.
      const real_t w = sign * ip.weight * g.Eval(Tr, ip);
      for (int d = 0; d < dim; d++)
         for (int j = 0; j < dof; j++)
         {
            elvect(dof * d + j) += w * nor(d) * shape(j);
         }
   }
}

real_t PathLiftCoefficient::Eval(ElementTransformation &T,
                                 const IntegrationPoint &ip)
{
   FaceElementTransformations *FTr =
      dynamic_cast<FaceElementTransformations *>(&T);
   MFEM_VERIFY(FTr, "PathLiftCoefficient must be evaluated on a face, as the "
               "lifting is defined along a path issuing from one");

   const FiniteElementSpace &fes = *u.FESpace();
   fes.GetElementVDofs(FTr->Elem1No, vdofs);
   u.GetSubVector(vdofs, elfun);

   return integ.ComputeLift(*fes.GetFE(FTr->Elem1No), *FTr, ip, elfun);
}

// The derivative of the path's endpoint along reference coordinate @a i of
// the face at @a ip, for a family without EndpointJacobian(), by a
// second-order difference whose stencil STAYS ON THE FACE.
//
// The stencil used to be ip +- fd_step whatever ip was. Within fd_step of a
// face's end that asked Endpoint() about a point extrapolated past the vertex,
// off the face the family is defined on; where the vertex lies on or near
// Gamma that point is outside Gamma, and VertexConePath aborts there. Double
// at its default step (6e-6) never came that close to an end, but single's
// (4.9e-3) does on an ordinary rule -- the first point of a 21-point Gauss
// rule is at 3.1e-3. Reported by MEQ.
//
// So: the centred difference wherever both of its points are on the face --
// unchanged, bit for bit, from what every such point got before -- and
// otherwise the one-sided second-order difference (-3 a(xi) + 4 a(xi +- h)
// - a(xi +- 2h)) / (+-2h) into the face. Near a corner of a triangular face
// neither may fit; the step is then halved, which costs a factor of two in
// round-off each time, and after a few halvings the old extrapolated stencil
// is used rather than refusing a point the code used to accept.
static void FaceEndpointDerivative(FaceElementTransformations &FTr,
                                   const TransferPath &path,
                                   const IntegrationPoint &ip,
                                   const Vector &xbar, int i, real_t fd_step,
                                   Vector &xbar_p, Vector &xbar_m, Vector &da)
{
   const Geometry::Type fgeom = FTr.GetGeometryType();
   auto shifted = [&](real_t d)
   {
      IntegrationPoint q = ip;
      real_t *c = (i == 0) ? &q.x : ((i == 1) ? &q.y : &q.z);
      *c += d;
      return q;
   };
   auto on_face = [&](real_t d)
   { return Geometry::CheckPoint(fgeom, shifted(d)); };
   auto centred = [&](real_t h)
   {
      path.Endpoint(FTr, shifted(h), xbar_p);
      path.Endpoint(FTr, shifted(-h), xbar_m);
      subtract(xbar_p, xbar_m, da);
      da /= 2. * h;
   };

   real_t h = fd_step;
   for (int halvings = 0; halvings <= 4; halvings++, h *= 0.5)
   {
      if (on_face(h) && on_face(-h)) { centred(h); return; }
      for (const real_t sgn : {-1., 1.})
      {
         if (!on_face(sgn * h) || !on_face(2. * sgn * h)) { continue; }
         path.Endpoint(FTr, shifted(sgn * h), xbar_p);
         path.Endpoint(FTr, shifted(2. * sgn * h), xbar_m);
         da.Set(-3., xbar);
         da.Add(4., xbar_p);
         da.Add(-1., xbar_m);
         da /= 2. * sgn * h;
         return;
      }
   }
   centred(fd_step);
}

void ExtensionRegionQuadrature(
   FaceElementTransformations &FTr, const TransferPath &path,
   const IntegrationRule &face_ir, const IntegrationRule &line_ir,
   const std::function<void(const ExtensionPoint &)> &visit, real_t fd_step)
{
   const int sdim = FTr.GetSpaceDim();
   const int fdim = sdim - 1;

   Vector x(sdim), xbar(sdim), m(sdim), y(sdim);
   Vector xbar_p(sdim), xbar_m(sdim), da(sdim);
   DenseMatrix J(sdim), dxdxi(sdim, std::max(fdim, 1));
   DenseMatrix dadxi(sdim, std::max(fdim, 1));

   for (int q = 0; q < face_ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = face_ir.IntPoint(q);

      path.Endpoint(FTr, ip, xbar);
      FTr.SetAllIntPoints(&ip);
      FTr.Transform(ip, x);
      subtract(xbar, x, m);

      // The face's own Jacobian, and the derivative of the path's endpoint
      // along the face.  A point face -- the boundary of a one-dimensional
      // mesh -- has neither, and the region is the path itself.
      if (fdim > 0) { dxdxi = FTr.Jacobian(); }

      // The family's own derivative where it has one, as
      // ExtensionBoundaryQuadrature() asks; the difference otherwise.
      const bool analytic = fdim > 0 && path.EndpointJacobian(FTr, ip, dadxi);
      for (int i = 0; i < fdim && !analytic; i++)
      {
         FaceEndpointDerivative(FTr, path, ip, xbar, i, fd_step,
                                xbar_p, xbar_m, da);
         for (int d = 0; d < sdim; d++) { dadxi(d, i) = da(d); }
      }
      // Endpoint() -- or EndpointJacobian() -- moved the transformations; put
      // them back.
      FTr.SetAllIntPoints(&ip);

      // THE WEIGHT IS SIGNED, FOR THE REASON ExtensionBoundaryQuadrature() IS.
      // The map y(xi,t) is not required to be injective, and where a boundary
      // feature is thinner than a mesh width it is not: near a cusp the paths
      // of one face fan across the tip and the region folds back on itself. An
      // UNSIGNED weight integrates the swept area WITH MULTIPLICITY, so a fold
      // is counted twice instead of cancelling, and the sum overcounts
      // |D_h^c| by O(h) -- 1.1e-2, 1.1e-2, 5.3e-3, 1.1e-3 on the reference's
      // aerofoil as n runs 16 to 128, which is the accuracy the transfer
      // technique exists to buy, thrown away in the quadrature. It is the same
      // defect the sibling had, found there first.
      //
      // The reference orientation is the face's own at t = 0, where J is
      // (dx/dxi, m): CalcOrtho()'s difficulty does not arise because nothing
      // here needs an outward normal, only a consistent sign to measure the
      // fold against. Where the map reverses, det J changes sign relative to
      // it and the segment is subtracted; what survives is the net
      // multiplicity, which is one.
      for (int i = 0; i < fdim; i++)
         for (int d = 0; d < sdim; d++) { J(d, i) = dxdxi(d, i); }
      for (int d = 0; d < sdim; d++) { J(d, sdim - 1) = m(d); }
      const real_t det0 = J.Det();
      // A face whose reference cell is already degenerate contributes nothing
      // and has no sign to take; the sibling skips the analogous case.
      if (det0 == 0.) { continue; }
      const real_t orient = (det0 > 0.) ? 1. : -1.;

      for (int t = 0; t < line_ir.GetNPoints(); t++)
      {
         const IntegrationPoint &tip = line_ir.IntPoint(t);

         for (int d = 0; d < sdim; d++) { y(d) = x(d) + tip.x * m(d); }

         // dy/dxi_i = (1-t) dx/dxi_i + t da/dxi_i, and dy/dt = m.
         for (int i = 0; i < fdim; i++)
            for (int d = 0; d < sdim; d++)
            {
               J(d, i) = (1. - tip.x) * dxdxi(d, i) + tip.x * dadxi(d, i);
            }
         for (int d = 0; d < sdim; d++) { J(d, sdim - 1) = m(d); }

         const ExtensionPoint pt{y, xbar, ip, tip.x,
                                 ip.weight * tip.weight * orient * J.Det()};
         visit(pt);
      }
   }
}

void ExtensionBoundaryQuadrature(
   FaceElementTransformations &FTr, const TransferPath &path,
   const IntegrationRule &face_ir,
   const std::function<void(const ExtensionBoundaryPoint &)> &visit,
   real_t fd_step)
{
   const int sdim = FTr.GetSpaceDim();
   const int fdim = sdim - 1;

   Vector x(sdim), xbar(sdim), m(sdim), nu(sdim);
   Vector xbar_p(sdim), xbar_m(sdim), da(sdim);
   DenseMatrix dadxi(sdim, std::max(fdim, 1));

   for (int q = 0; q < face_ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = face_ir.IntPoint(q);

      path.Endpoint(FTr, ip, xbar);
      FTr.SetAllIntPoints(&ip);
      FTr.Transform(ip, x);
      subtract(xbar, x, m);

      // The derivative of the path's endpoint along the face, which at t = 1
      // is the WHOLE Jacobian of the map: the face's own dx/dxi is multiplied
      // by (1-t) and has gone.
      //
      // ASK THE FAMILY FIRST.  A family with a closed form gives an exact
      // Jacobian and saves 2(d-1) Endpoint() calls per point; one without says
      // so and the central difference below is taken instead, which is the
      // reference behaviour rather than a fallback in the pejorative sense.
      // Note the difference's own floor: O(fd_step^2) truncation against
      // O(eps/fd_step) round-off puts it near 1e-10 at the default step, and a
      // closed form removes that entirely.
      const bool analytic = path.EndpointJacobian(FTr, ip, dadxi);

      for (int i = 0; i < fdim && !analytic; i++)
      {
         FaceEndpointDerivative(FTr, path, ip, xbar, i, fd_step,
                                xbar_p, xbar_m, da);
         for (int d = 0; d < sdim; d++) { dadxi(d, i) = da(d); }
      }
      // Endpoint() -- or EndpointJacobian() -- moved the transformations; put
      // them back.
      FTr.SetAllIntPoints(&ip);

      // A point face -- the boundary of a one-dimensional mesh -- has no
      // tangent, and its image is a point of unit weight.  CalcOrtho() is not
      // defined for that shape, so it is taken separately rather than guarded
      // inside the loop above.
      real_t measure = 1.;
      if (fdim > 0)
      {
         CalcOrtho(dadxi, nu);
         measure = nu.Norml2();

         // A FACE WHOSE IMAGE IS A POINT CONTRIBUTES NOTHING, AND THAT IS
         // ORDINARY RATHER THAN AN ERROR.  A closest-point family collapses
         // every point of a face lying along a ray to the same foot -- for
         // ClosestPointPath::Sphere, a face of the staircase Gamma_h that
         // happens to be radial, which near the poles of the circle is every
         // second face.  Then da/dxi vanishes, Gamma has no normal to report,
         // and the right contribution is none: the face covers zero arc, its
         // two endpoint feet coincide, and its neighbours still cover Gamma
         // between them.
         //
         // THE TEST IS RELATIVE TO THE FACE, AND AN EXACT-ZERO TEST IS NOT
         // ENOUGH -- measured, not reasoned.  With ClosestPointPath's analytic
         // Jacobian the radial direction is annihilated by (I - uu^T), which is
         // exactly zero only in exact arithmetic; in floating point a radial
         // face gives |da/dxi| ~ 1e-16 pointing ANYWHERE.  An exact-zero test
         // passes that through, and the normal it yields is garbage -- measured
         // at 90 degrees from the true one, |nu - exact| = sqrt(2).  The
         // WEIGHT is ~1e-16 too, so a quadrature sum is unharmed and the fault
         // hides; a caller reading nu on its own is not so lucky.
         //
         // Comparing against the face's own |dx/dxi| is the defensible test:
         // both are lengths per unit reference coordinate, so the ratio is
         // dimensionless, and an image twelve orders shorter than the face that
         // produced it is degenerate on any reading.  A map that merely
         // compresses -- a distant sphere of small radius, where |da/dxi| is
         // R/|x-c| times |dx/dxi| -- is nowhere near this and is kept.
         // AND THE THRESHOLD HAS TO MATCH HOW THE JACOBIAN WAS OBTAINED, which
         // is the second half of the same lesson.  A closed form is accurate to
         // rounding, so a residue of order eps|dx/dxi| is degeneracy.  A central
         // difference is not: it divides by fd_step, so its own noise floor is
         // eps|a|/fd_step -- about 1e-10 at the default step, SIX ORDERS above
         // eps.  Measured, a radial face under the difference gives |da/dxi| ~
         // 5e-11 against a face scale of 0.1, sails through a 1e-12 test, and
         // hands back the same garbage normal at 90 degrees.
         //
         // So the two branches get their own floors, each a hundred times its
         // own noise.  A legitimate face here measures ~0.1, five orders clear
         // of the looser of them, so nothing real is at risk of being skipped.
         real_t face_scale = 0.;
         for (int i = 0; i < fdim; i++)
            for (int d = 0; d < sdim; d++)
            {
               face_scale += FTr.Jacobian()(d, i) * FTr.Jacobian()(d, i);
            }
         face_scale = std::sqrt(face_scale);

         const real_t eps = std::numeric_limits<real_t>::epsilon();
         const real_t floor = analytic
                              ? 1e-12 * face_scale
                              : 100. * (eps / fd_step) * xbar.Norml2();

         if (measure <= floor)
         {
            // A floor at or above the face's OWN scale rejects every image,
            // however good, so the skip is no longer a degeneracy test -- it
            // is the routine quietly doing nothing, and the caller cannot tell
            // an empty rule from a face that legitimately contributed nothing.
            // That is precisely what a double-tuned fd_step does in a
            // single-precision build: eps/fd_step is 0.119 there against
            // 2.2e-10 here, so the floor lands at ~12|xbar| and no point
            // survives it. Refuse, and name the step, rather than return empty.
            MFEM_VERIFY(floor < face_scale,
                        "The central-difference step " << fd_step
                        << " is too small for this build's epsilon (" << eps
                        << "): the degeneracy floor " << floor
                        << " is at or above the face's own scale "
                        << face_scale << ", so no quadrature point can pass "
                        "it and the sweep would silently yield none. Pass a "
                        "larger fd_step -- the default, ExtensionFDStep(), "
                        "follows real_t.");
            continue;
         }
         nu /= measure;
      }
      else
      {
         // In 1D the outward direction is the path's own.
         nu = m;
         const real_t length = nu.Norml2();
         MFEM_VERIFY(length > 0., "the path has zero length");
         nu /= length;
      }

      // THE ORIENTATION, AND IT IS THE PATHS THAT FIX IT.  CalcOrtho()'s sign
      // follows the ordering of dadxi's columns, which is the face's
      // parametrisation and carries no information about which side D_h is on.
      // The paths do: they run outward, so a normal agreeing with them is the
      // outward normal of Gamma.
      //
      // AND THE SAME TEST GIVES THE WEIGHT ITS SIGN, WHICH IS NOT COSMETIC.
      // The map xi -> a(x(xi)) is not required to be monotone along Gamma.
      // Where it runs forward, back and forward again, an UNSIGNED weight
      // integrates the traversed length, counting that stretch two and three
      // times over. The case first measured here -- a "pinch" face on a
      // staircase circle, overcounting |Gamma| by O(h) -- turned out to be
      // VertexConePath's defect at a vertex lying on Gamma, not the staircase
      // (see VertexConePath); the sign is what this routine owes a family that
      // does reverse, and the unit test builds one.
      //
      // A SIGNED weight cancels the backtracking exactly: where the map
      // reverses, dadxi reverses, CalcOrtho()'s normal turns inward, this test
      // fires, and the segment is subtracted.  What survives is the net
      // multiplicity, which is one.  So a caller integrating f over Gamma gets
      // the integral over Gamma however the family wanders, and the tiling
      // check becomes a statement about coverage rather than about
      // monotonicity.
      //
      // The normal handed to the visitor is ALWAYS the outward one; only the
      // weight carries the sign.  A visitor that ignores the sign -- summing
      // std::abs(weight), say -- measures the traversed length instead, which
      // is a different and occasionally useful quantity, but it is not a
      // quadrature over Gamma.
      real_t signed_weight = ip.weight * measure;
      if (nu * m < 0.) { nu.Neg(); signed_weight = -signed_weight; }

      const ExtensionBoundaryPoint pt{xbar, nu, ip, signed_weight};
      visit(pt);
   }
}

int MarkLevelSetSubdomain(const Mesh &mesh, const PositionFunction &phi,
                          real_t offset, Array<int> &marker, int extra_refine,
                          bool phi_thread_safe)
{
   const int NE = mesh.GetNE();
   const int sdim = mesh.SpaceDimension();

   marker.SetSize(NE);
   marker = 0;

   // The sampling lattices, one per element geometry, refined BEFORE the loop.
   // GeometryRefiner::Refine() is a critical section in a thread-safe build
   // and a race in any other, and its answer depends on the geometry alone,
   // so there is nothing to refine per element.
   std::array<RefinedGeometry*, Geometry::NUM_GEOMETRIES> lattice{};
   if (extra_refine >= 1)
   {
      for (int g = 0; g < Geometry::NUM_GEOMETRIES; g++)
      {
         const Geometry::Type gt = Geometry::Type(g);
         if (Geometry::Dimension[g] == mesh.Dimension() && mesh.HasGeometry(gt))
         {
            lattice[g] = GlobGeometryRefiner.Refine(gt, extra_refine);
         }
      }
   }

   const bool threaded =
      LevelSetLoopsThread(mesh, phi_thread_safe, extra_refine >= 1);
   (void)threaded;   // read only by the pragma
   int count = 0;
   LevelSetLoopTrap trap;

   // Every element is decided alone and writes only its own marker, so the
   // threaded loop is the serial one with its iterations reordered.
#ifdef MFEM_USE_OPENMP
   #pragma omp parallel if (threaded) reduction(+:count)
#endif
   {
      Array<int> vert;
      Vector x(sdim);
      IsoparametricTransformation Tr;

#ifdef MFEM_USE_OPENMP
      #pragma omp for schedule(static)
#endif
      for (int i = 0; i < NE; i++)
      {
         try
         {
            bool inside = true;

            mesh.GetElementVertices(i, vert);
            for (int v = 0; v < vert.Size() && inside; v++)
            {
               const real_t *c = mesh.GetVertex(vert[v]);
               for (int d = 0; d < sdim; d++) { x(d) = c[d]; }
               if (phi(x) > -offset) { inside = false; }
            }

            if (inside && extra_refine >= 1)
            {
               RefinedGeometry *RefG = lattice[mesh.GetElementGeometry(i)];
               mesh.GetElementTransformation(i, &Tr);
               for (int q = 0; q < RefG->RefPts.GetNPoints() && inside; q++)
               {
                  Tr.Transform(RefG->RefPts.IntPoint(q), x);
                  if (phi(x) > -offset) { inside = false; }
               }
            }

            if (inside) { marker[i] = 1; count++; }
         }
         catch (...) { trap.Capture(); }
      }
   }
   trap.Rethrow();

   return count;
}


int MarkLevelSetSubdomain(const Mesh &mesh, const PositionFunction &phi,
                          real_t offset, Array<int> &marker,
                          const TransferPath &path, real_t min_cos,
                          int extra_refine, int max_sweeps,
                          bool phi_thread_safe)
{
   int count = MarkLevelSetSubdomain(mesh, phi, offset, marker, extra_refine,
                                     phi_thread_safe);
   const int sdim = mesh.SpaceDimension();
   if (sdim == 1 || mesh.Dimension() != sdim) { return count; }

   FaceElementTransformations FTr;
   IsoparametricTransformation T1, T2;
   Vector x(sdim), nor(sdim), xbar(sdim), m(sdim);
   Array<int> drop;

   for (int sweep = 0; sweep < max_sweeps; sweep++)
   {
      drop.SetSize(0);
      for (int f = 0; f < mesh.GetNumFaces(); f++)
      {
         int e1, e2;
         mesh.GetFaceElements(f, &e1, &e2);
         if (e2 < 0) { continue; }                 // the background's boundary
         if (marker[e1] == marker[e2]) { continue; }
         const int inner = marker[e1] ? e1 : e2;
         const real_t flip = marker[e1] ? 1. : -1.;

         mesh.GetFaceElementTransformations(f, FTr, T1, T2);
         const IntegrationRule &ir = IntRules.Get(FTr.GetGeometryType(), 4);
         for (int q = 0; q < ir.GetNPoints(); q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            FTr.SetAllIntPoints(&ip);
            FTr.Transform(ip, x);
            CalcOrtho(FTr.Jacobian(), nor);
            nor *= flip / nor.Norml2();
            path.Endpoint(x, nor, xbar);
            subtract(xbar, x, m);
            const real_t L = m.Norml2();
            if (L > 0. && (m * nor) / L < min_cos)
            {
               drop.Append(inner);
               break;
            }
         }
      }
      if (drop.Size() == 0) { break; }
      for (int k = 0; k < drop.Size(); k++)
      {
         if (marker[drop[k]]) { marker[drop[k]] = 0; count--; }
      }
   }
   return count;
}

PathIncidenceReport CheckPathIncidence(Mesh &D_h, int gamma_h_attr,
                                       const TransferPath &path,
                                       const VectorPositionFunction &grad_phi,
                                       real_t threshold, int order)
{
   PathIncidenceReport r;
   Vector x, nor, xbar, m, g;
   for (int be = 0; be < D_h.GetNBE(); be++)
   {
      if (D_h.GetBdrAttribute(be) != gamma_h_attr) { continue; }
      FaceElementTransformations *FTr = D_h.GetBdrFaceTransformations(be);
      if (!FTr) { continue; }
      const IntegrationRule &ir = IntRules.Get(FTr->GetGeometryType(), order);
      bool face_below = false;
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         FacePointAndNormal(*FTr, ip, x, nor);
         path.Endpoint(*FTr, ip, xbar);
         m = xbar;
         m -= x;
         const real_t L = m.Norml2();
         if (L == 0.) { continue; }
         g.SetSize(x.Size());
         grad_phi(xbar, g);
         const real_t gn = g.Norml2();
         // A gradient that vanishes or is not finite has no incidence; it is
         // as bad as a tangent ray, and counts as one.
         const real_t c = (gn > 0. && g.CheckFinite() == 0)
                          ? (m * g) / (L * gn) : 0.;
         r.points++;
         if (c < threshold) { r.below++; face_below = true; }
         if (c < r.min_cos) { r.min_cos = c; r.worst_bdr = be; }
      }
      if (face_below) { r.faces_below++; }
   }
   return r;
}

PathAngleReport CheckPathAngles(Mesh &D_h, int gamma_h_attr,
                                const TransferPath &path, real_t threshold,
                                int order)
{
   PathAngleReport r;
   Vector x, nor, xbar, m;
   for (int be = 0; be < D_h.GetNBE(); be++)
   {
      if (D_h.GetBdrAttribute(be) != gamma_h_attr) { continue; }
      FaceElementTransformations *FTr = D_h.GetBdrFaceTransformations(be);
      if (!FTr) { continue; }
      const IntegrationRule &ir = IntRules.Get(FTr->GetGeometryType(), order);
      bool face_below = false;
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         FacePointAndNormal(*FTr, ip, x, nor);
         path.Endpoint(*FTr, ip, xbar);
         m = xbar;
         m -= x;
         const real_t L = m.Norml2();
         if (L == 0.) { continue; }
         const real_t c = (m * nor) / L;
         r.points++;
         if (c < threshold) { r.below++; face_below = true; }
         if (c < r.min_cos) { r.min_cos = c; r.worst_bdr = be; }
      }
      if (face_below) { r.faces_below++; }
   }
   return r;
}

} // namespace mfem
