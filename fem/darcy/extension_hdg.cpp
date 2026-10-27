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
#include <map>
#include <vector>

namespace mfem
{

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

/// The distance along a ray from @a x below which a crossing cannot be told
/// from the origin: the bisection's bracket, or what the format resolves about
/// @a x, whichever is the larger.
real_t CrossingResolution(const Vector &x, real_t length, real_t tol)
{
   return std::max(tol * length, std::numeric_limits<real_t>::epsilon()
                   * (x.Normlinf() + length));
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

   // The bracket target tol * length is absolute, and with the 1e-13 default
   // it is below what single precision can represent. So the loop also stops
   // when the midpoint no longer moves: the crossing is then located as well
   // as the format allows. Without that, every ray in a single build ran to
   // max_iter at the right answer (MEQ: 27 s of setup against 9). In double
   // the bracket target is reached first, so nothing changes there.
   const real_t ttol = tol * length;
   for (int it = 0; it < max_iter && tb - ta > ttol; it++)
   {
      const real_t tm = 0.5 * (ta + tb);
      if (tm <= ta || tm >= tb) { break; }
      if (phi_at(tm) > 0.) { tb = tm; }
      else { ta = tm; }
   }

   t = 0.5 * (ta + tb);
   return true;
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

VertexConePath::VertexConePath(const Mesh &mesh_, int gamma_h_attr,
                               PositionFunction phi_, real_t search_length_,
                               int n_rays_, int n_keep_, int search_steps_,
                               real_t tol_, int max_iter_, bool use_cone_)
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

      // Background edges incident on each parent vertex, once.
      std::map<int, std::vector<int>> edges_at;
      Array<int> ev;
      for (int e = 0; e < bg.GetNEdges(); e++)
      {
         const_cast<Mesh &>(bg).GetEdgeVertices(e, ev);
         edges_at[ev[0]].push_back(e);
         edges_at[ev[1]].push_back(e);
      }

      for (auto &kv : vertex_normals)
      {
         const int sv = kv.first;
         if (sv >= pv.Size()) { continue; }
         auto it = edges_at.find(pv[sv]);
         if (it == edges_at.end()) { continue; }

         const real_t *c = bg.GetVertex(pv[sv]);
         const Array<real_t> &ns = kv.second;
         const int m_n = ns.Size() / 2;

         // Keep the edge directions that leave D_h through every face of
         // Gamma_h meeting here -- the same admissibility the half space
         // uses, applied to the background's own edges rather than to a
         // continuum of angles.
         std::vector<real_t> ang;
         for (int e : it->second)
         {
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

   // A direction at each of those vertices.
   std::map<int, std::array<real_t, 2>> vertex_tangent;
   Vector x(2), t(2);
   for (auto &kv : vertex_normals)
   {
      const real_t *c = m.GetVertex(kv.first);
      x(0) = c[0]; x(1) = c[1];
      real_t cc = 0., ch = -1.;
      auto ic = vertex_cone.find(kv.first);
      if (ic != vertex_cone.end()) { cc = ic->second[0]; ch = ic->second[1]; }
      MFEM_VERIFY(VertexDirection(x, kv.second, cc, ch, t),
                  "no ray from the vertex (" << x(0) << ", " << x(1)
                  << ") of Gamma_h reaches Gamma within the search length "
                  << search_length);
      vertex_tangent[kv.first] = {{t(0), t(1)}};
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
                                     Vector &t)
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
      if (half_width <= 0. || cone_half < half_width) { n_tighter++; }
      centre_angle = cone_centre;
      half_width = cone_half;
      n_coned++;
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
   // bisection's error in s (CrossingResolution(), r) and the limit's own
   // (s times the curvature, taken on the scale of the search) are equal,
   // sqrt(r L); either side of it the two agree to that, and the direction
   // is continuous through it.
   const real_t phi0 = phi(x);
   const real_t eps = std::numeric_limits<real_t>::epsilon();
   const real_t on_gamma = std::sqrt(CrossingResolution(x, search_length, tol)
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
         if (pass > 0) { n_widened++; }
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

      if (pass > 0) { n_widened++; }

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

void VertexConePath::Endpoint(FaceElementTransformations &FTr,
                              const IntegrationPoint &ip, Vector &xbar) const
{
   const int f = (FTr.ElementType == ElementTransformation::BDR_FACE)
                 ? mesh->GetBdrElementFaceIndex(FTr.ElementNo)
                 : FTr.ElementNo;
   MFEM_VERIFY(f >= 0 && f < has_tangent.Size() && has_tangent[f],
               "no path was built for this face: it is not on the boundary "
               "the family was constructed for");

   const real_t th = ip.x;
   Vector t(2);
   t(0) = (1. - th) * tang[4 * f + 0] + th * tang[4 * f + 2];
   t(1) = (1. - th) * tang[4 * f + 1] + th * tang[4 * f + 3];
   const real_t len = t.Norml2();
   MFEM_VERIFY(len > 1e-12, "the paths of the two vertices of a face oppose "
               "each other, so no direction can be interpolated between them");
   t /= len;

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

bool TiltedPath::Tilt(const Vector &x, const Vector &n, Vector &xbar) const
{
   const int dim = x.Size();
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
   Vector t(n);
   if (wn > 0.)
   {
      t *= gamma;
      t.Add(std::sqrt(1. - gamma * gamma) / wn, w);
   }

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
   Vector x, nor, xbar;
   FacePointAndNormal(FTr, ip, x, nor);
   base.Endpoint(FTr, ip, xbar);
   if (Tilt(x, nor, xbar)) { return false; }
   return base.EndpointJacobian(FTr, ip, dadxi);
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

      for (int i = 0; i < fdim; i++)
      {
         FaceEndpointDerivative(FTr, path, ip, xbar, i, fd_step,
                                xbar_p, xbar_m, da);
         for (int d = 0; d < sdim; d++) { dadxi(d, i) = da(d); }
      }
      // Endpoint() moved the transformations; put them back.
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
                          real_t offset, Array<int> &marker, int extra_refine)
{
   const int NE = mesh.GetNE();
   const int sdim = mesh.SpaceDimension();

   marker.SetSize(NE);
   marker = 0;

   Array<int> vert;
   Vector x(sdim);
   IsoparametricTransformation Tr;
   int count = 0;

   for (int i = 0; i < NE; i++)
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
         RefinedGeometry *RefG =
            GlobGeometryRefiner.Refine(mesh.GetElementGeometry(i), extra_refine);
         mesh.GetElementTransformation(i, &Tr);
         for (int q = 0; q < RefG->RefPts.GetNPoints() && inside; q++)
         {
            Tr.Transform(RefG->RefPts.IntPoint(q), x);
            if (phi(x) > -offset) { inside = false; }
         }
      }

      if (inside) { marker[i] = 1; count++; }
   }

   return count;
}


int MarkLevelSetSubdomain(const Mesh &mesh, const PositionFunction &phi,
                          real_t offset, Array<int> &marker,
                          const TransferPath &path, real_t min_cos,
                          int extra_refine, int max_sweeps)
{
   int count = MarkLevelSetSubdomain(mesh, phi, offset, marker, extra_refine);
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
