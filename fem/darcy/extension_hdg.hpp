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

#ifndef MFEM_EXTENSION_HDG
#define MFEM_EXTENSION_HDG

#include "../../config/config.hpp"
#include "../bilininteg.hpp"
#include "../coefficient.hpp"
#include "../eltrans.hpp"
#include "../gridfunc.hpp"
#include "../lininteg.hpp"
#include "../../mesh/mesh.hpp"

#include <functional>
#include <cmath>
#include <limits>

namespace mfem
{

/** @brief Solving a boundary-value problem on a polyhedral subdomain of the
    true domain, by extension from that subdomain.

    The construction is Cockburn & Solano's. Given a domain @f$\Omega@f$ with a
    boundary @f$\Gamma@f$ that the mesh does not follow, one meshes a polyhedral
    subdomain @f$D_h \subset \Omega@f$ and solves there, with the Dirichlet
    datum given on @f$\Gamma@f$ transferred to the computational boundary
    @f$\Gamma_h = \partial D_h@f$ along a family of *transferring paths*. For
    the model problem

    @f[ \boldsymbol{u} + K \nabla p = 0, \qquad -\nabla\cdot\boldsymbol{u} = f
        \text{ in } \Omega, \qquad p = g \text{ on } \Gamma, @f]

    integrating the first equation along a path @f$\sigma@f$ running from
    @f$x \in \Gamma_h@f$ to @f$\bar{x} = a(x) \in \Gamma@f$ gives

    @f[ p(x) = g(\bar{x}) + \int_\sigma C\,\boldsymbol{u}\cdot\boldsymbol{m}\,ds,
        \qquad C = K^{-1}, @f]

    with @f$\boldsymbol{m}@f$ the unit tangent of @f$\sigma@f$ pointing from
    @f$x@f$ towards @f$\bar{x}@f$. The value is independent of the path because
    @f$C\boldsymbol{u} = -\nabla p@f$ is a gradient. The discrete method
    replaces @f$\boldsymbol{u}@f$ outside @f$D_h@f$ by the *extension*
    @f$E_h(\boldsymbol{u}_h)@f$ -- the polynomial of the element owning the
    face, evaluated outside it -- and takes the resulting

    @f[ \varphi_h(x) = g(a(x))
        + \int_\sigma C\,E_h(\boldsymbol{u}_h)\cdot\boldsymbol{m}\,ds @f]

    as the Dirichlet datum on @f$\Gamma_h@f$. The curved boundary is thereby
    reduced to the evaluation of line integrals; nothing depends on how
    @f$\Gamma@f$ is represented, on the space dimension, or on the method used
    inside @f$D_h@f$.

    The point of the construction is that the orders of convergence are those
    of the boundary-fitted method even when @f$\operatorname{dist}(\Gamma_h,
    \Gamma)@f$ is only @f$O(h)@f$, where earlier techniques needed
    @f$O(h^{k+1})@f$.

    References:
    - Cockburn, B., & Solano, M. (2012). Solving Dirichlet boundary-value
      problems on curved domains by extensions from subdomains. SIAM J. Sci.
      Comput. 34(1), A497-A519.
    - Cockburn, B., & Solano, M. (2014). Solving convection-diffusion problems
      on curved domains by extensions from subdomains. J. Sci. Comput. 59,
      512-543. The lifting is unchanged by the convection: it integrates the
      constitutive law, in which the convective field does not appear.
    - Cockburn, B., Qiu, W., & Solano, M. (2014). A priori error analysis for
      HDG methods using extensions from subdomains to achieve boundary
      conformity. Math. Comp. 83(286), 665-699. */

/// A scalar function of the physical position alone.
/** The level set and the Dirichlet datum are both evaluated at points that lie
    *outside* the mesh, where a Coefficient cannot be evaluated: a Coefficient
    reaches its position through an ElementTransformation, and no element
    contains these points. They are therefore taken as functions of position. */
using PositionFunction = std::function<real_t(const Vector &)>;

/// A vector-valued function of the physical position alone.
using VectorPositionFunction = std::function<void(const Vector &, Vector &)>;


/** @brief A family of transferring paths joining the computational boundary
    @f$\Gamma_h@f$ to the true boundary @f$\Gamma@f$.

    Each path is the straight segment from a point @f$x@f$ of @f$\Gamma_h@f$ to
    its image @f$a(x)@f$ on @f$\Gamma@f$, so a family is determined by the map
    @f$a@f$ alone. Two conditions are needed for the analysis to apply, and
    both should be checked when a new family is written:
    - @f$(a(x) - x)\cdot n_e > 0@f$ on every boundary face @f$e@f$, with
      @f$n_e@f$ the outward normal;
    - the paths do not cross before reaching @f$\Gamma@f$.

    **A family is built once and read afterwards, and every routine that
    evaluates one is `const`.** That is a contract and not an accident:
    HDGExtensionIntegrator holds a `const TransferPath &` and, in a
    MFEM_THREAD_SAFE build, is entered from several threads at once, so a
    family that cached anything per call would put back the race the
    integrator's guard takes out. The three families here were checked against
    it rather than assumed to meet it -- ClosestPointPath and LevelSetPath hold
    configuration only, and VertexConePath writes #tang, #has_tangent and its
    four counters in its CONSTRUCTOR and nowhere else. A family that cannot
    keep the property owes a guard of its own. */
class TransferPath
{
public:
   /** @brief The endpoint @f$a(x) \in \Gamma@f$ of the path issuing from the
       point @a x of @f$\Gamma_h@f$, whose outward unit normal is @a n. */
   virtual void Endpoint(const Vector &x, const Vector &n,
                         Vector &xbar) const = 0;

   /** @brief The endpoint at an integration point of a boundary face.

       The default implementation evaluates the physical point and the outward
       unit normal and defers to Endpoint(x, n). A family that depends on the
       face itself -- on its vertices, or on data precomputed per face --
       overrides this instead.

       @a FTr must have its integration points set to @a ip already. */
   virtual void Endpoint(FaceElementTransformations &FTr,
                         const IntegrationPoint &ip, Vector &xbar) const;

   /** @brief @f$\partial a/\partial\xi@f$ along the face, if the family has
       it in closed form.

       **Returns false by default, and a family is under no obligation to
       override it.** A path family is a map; requiring a derivative of every
       one would exclude the families defined by a search, which are the
       general case. ExtensionBoundaryQuadrature() falls back to a central
       difference when this returns false, and that fallback is the reference
       behaviour rather than a degraded one.

       **A family that CAN supply it should**, because the difference is not
       free in either sense: it costs @f$2(d-1)@f$ extra Endpoint() calls per
       quadrature point, and it carries an @f$O(\texttt{fd_step}^2)@f$
       truncation against an @f$O(\epsilon/\texttt{fd_step})@f$ round-off,
       which at the default step floors the quadrature near @f$10^{-10}@f$
       where a closed form is exact.

       **The level-set families have one when given the level set's
       gradient** (LevelSetPath::SetLevelSetGradient() and its namesakes).
       Their endpoint is @f$a = x + s\,t@f$ with @f$\phi(a) = 0@f$, so
       @f$s' = -\nabla\phi\cdot(x' + s\,t')/(\nabla\phi\cdot t)@f$ and
       @f$a' = x' + s\,t' + s'\,t@f$, with @f$x'@f$ the face's Jacobian and
       @f$t'@f$ the derivative of the path's direction. Without the gradient
       they return false.

       @a dadxi is @f$d\times(d-1)@f$, its column @a i being
       @f$\partial a/\partial\xi_i@f$. @a FTr must have its integration
       points set to @a ip already. */
   virtual bool EndpointJacobian(FaceElementTransformations &FTr,
                                 const IntegrationPoint &ip,
                                 DenseMatrix &dadxi) const
   { return false; }

   virtual ~TransferPath() = default;
};


/** @brief The path family of a boundary with a closed-form closest-point map.

    When @f$\Gamma@f$ is an analytic surface whose closest point is known in
    closed form, the whole path construction collapses to that map: for a
    sphere of centre @f$c@f$ and radius @f$R@f$,
    @f$a(x) = c + R(x-c)/|x-c|@f$. This is worth having directly and not only
    as a fast path through the general construction. */
class ClosestPointPath : public TransferPath
{
   VectorPositionFunction cp;
   /// @f$\partial a/\partial x@f$, if the caller supplied it. May be empty.
   std::function<void(const Vector &, DenseMatrix &)> dcp;

public:
   /// @param cp_  the closest-point map onto @f$\Gamma@f$.
   ClosestPointPath(VectorPositionFunction cp_) : cp(std::move(cp_)) { }

   /** @brief With the map's own Jacobian, so the quadrature needs no
       differences.

       @param dcp_  fills a @f$d\times d@f$ matrix with @f$\partial
                    a/\partial x@f$ at the given point. The derivative along a
                    face then follows by the chain rule, @f$\partial
                    a/\partial\xi = (\partial a/\partial x)(\partial
                    x/\partial\xi)@f$, which is exact.

       Worth supplying wherever the map is analytic, which is the case this
       class exists for. SphereJacobian() is the one for Sphere(). */
   ClosestPointPath(VectorPositionFunction cp_,
                    std::function<void(const Vector &, DenseMatrix &)> dcp_)
      : cp(std::move(cp_)), dcp(std::move(dcp_)) { }

   /// The closest-point map onto a sphere of centre @a c and radius @a R.
   static VectorPositionFunction Sphere(const Vector &c, real_t R);

   /** @brief Its Jacobian, @f$\partial a/\partial x = \frac{R}{|x-c|}
       (I - uu^T)@f$ with @f$u = (x-c)/|x-c|@f$.

       Singular in the radial direction, which is the map rather than a defect:
       every point of a ray has the same closest point, so @f$\partial
       a/\partial x@f$ annihilates @f$u@f$. A face lying along a ray therefore
       has @f$\partial a/\partial\xi = 0@f$ and covers nothing of
       @f$\Gamma@f$ -- which ExtensionBoundaryQuadrature() skips, and which the
       closed form makes an identity rather than a near-cancellation. */
   static std::function<void(const Vector &, DenseMatrix &)>
   SphereJacobian(const Vector &c, real_t R);

   using TransferPath::Endpoint;

   void Endpoint(const Vector &x, const Vector &n,
                 Vector &xbar) const override { cp(x, xbar); }

   bool EndpointJacobian(FaceElementTransformations &FTr,
                         const IntegrationPoint &ip,
                         DenseMatrix &dadxi) const override;
};


/** @brief The path family issuing along the outward normal of @f$\Gamma_h@f$,
    terminated on the zero level set by bisection.

    This is the family the a priori error analysis is written for: its paths
    are parallel to the face normal, which satisfies the sign condition
    @f$(a(x)-x)\cdot n_e > 0@f$ by construction and, for a boundary that is
    resolved by the mesh, does not cross. It needs no representation of
    @f$\Gamma@f$ beyond the level set that selected the subdomain. */
class LevelSetPath : public TransferPath
{
   PositionFunction phi;
   VectorPositionFunction grad_phi;
   real_t search_length;
   int search_steps;
   real_t tol;
   int max_iter;

public:
   /** @param phi_            the level set: negative inside @f$\Omega@f$, zero
                              on @f$\Gamma@f$, positive outside.
       @param search_length_  how far along the normal to look for the sign
                              change. It should exceed the largest expected
                              distance from @f$\Gamma_h@f$ to @f$\Gamma@f$ and
                              is normally a small multiple of the mesh size.
       @param search_steps_   how many equal steps the search interval is cut
                              into while bracketing the sign change. More than
                              one is needed only where @f$\Gamma@f$ can be
                              crossed twice within @a search_length_.
       @param tol_            how far outside @f$\Gamma@f$, as a fraction of
                              @a search_length_, the start of a path may lie
                              and still count as on it. **It does not stop the
                              bisection**, which always runs until its
                              midpoint stops moving: a bisection stopped short
                              makes the endpoint a staircase in the face
                              point, which a central difference along the
                              face amplifies (MEQ, 2026-10-28). The same holds
                              for every family here that takes a @a tol_.
       @param max_iter_       a cap on the bisection's halvings; double needs
                              about 45 from a bracket of a thirty-second of
                              the search. */
   LevelSetPath(PositionFunction phi_, real_t search_length_,
                int search_steps_ = 1, real_t tol_ = 1e-13,
                int max_iter_ = 100)
      : phi(std::move(phi_)), search_length(search_length_),
        search_steps(search_steps_), tol(tol_), max_iter(max_iter_) { }

   /** @brief Give the gradient of the level set, and EndpointJacobian()
       returns @f$\partial a/\partial\xi@f$ in closed form instead of
       false.

       @a grad_phi_ fills its second argument, of the size of the first, with
       @f$\nabla\phi@f$ of the SAME level set this family was built on, at
       any point it is asked about -- the endpoints lie on @f$\Gamma@f$,
       outside the mesh. Nothing here can check that the two agree, and a
       wrong gradient gives a wrong derivative with no error; the unit test
       compares the two against a central difference, which is the check to
       copy.

       The derivative is exact in the face point, where the central difference
       it replaces carries an @f$O(\epsilon|a|/\texttt{fd_step})@f$
       round-off and so a dependence on the step. It needs the face map's
       second derivatives where the face is curved.

       **A path meeting @f$\Gamma@f$ at a grazing angle ABORTS**, naming the
       face, the endpoint and @f$|\nabla\phi\cdot t|/|\nabla\phi|@f$, at
       @f$|\nabla\phi\cdot t| \le \sqrt\epsilon\,|\nabla\phi|@f$ or a
       gradient that is not finite. There @f$s'@f$ diverges -- the crossing
       slides along @f$\Gamma@f$ without bound, and past tangency jumps to a
       farther piece of it -- and that is the path map's, so a central
       difference would only hide it. The fix is the family's directions.
       CheckPathIncidence() finds the shallow paths before a solve.

       **At a corner of a polygonal @f$\Gamma@f$** the endpoint is smooth
       on either side and kinks where the ray passes the corner. Which side's
       derivative a point exactly there gets is @a grad_phi_'s choice: a
       GridFunction's gradient at a point on an element boundary is one
       element's. A face whose image straddles a corner loses its rule's
       order either way, which only splitting the face's rule at the corner
       would recover; that needs @f$\Gamma@f$'s corners, and nothing here
       has them.

       Call it before the family is used: it is configuration, and the
       family is read-only afterwards (see TransferPath). */
   void SetLevelSetGradient(VectorPositionFunction grad_phi_)
   { grad_phi = std::move(grad_phi_); }

   using TransferPath::Endpoint;

   void Endpoint(const Vector &x, const Vector &n,
                 Vector &xbar) const override;

   /// Along the outward unit normal, @f$t' = n'@f$: zero on a straight face,
   /// from the face map's Hessian on a curved one. False without a gradient.
   bool EndpointJacobian(FaceElementTransformations &FTr,
                         const IntegrationPoint &ip,
                         DenseMatrix &dadxi) const override;
};


/** @brief The general path family: a direction chosen at each vertex of
    @f$\Gamma_h@f$ by a search, and interpolated along the faces. **The
    cone of CS-Extensions section 2.4.1 is OFF unless @a use_cone_ is set**:
    by default the search is restricted by the half space alone, and
    HasCone() reports which.

    The two closed-form families above are enough where @f$\Gamma@f$ has a
    closest-point map, or where every face normal reaches it. Neither holds in
    general, and the Joukowsky airfoil of CS-Extensions §3.4 is where both give
    out. Measured there: **between four and fifteen per cent of the face
    normals never meet @f$\Gamma@f$ at all**, passing outside the thin tail or
    the nose, so LevelSetPath has no endpoint to return; and a family following
    each face's own normal does not tile the region beyond @f$\Gamma_h@f$,
    adjacent faces disagreeing on the path through the vertex they share.

    This is CS-Extensions' own construction, and it repairs both at once. A
    direction is chosen at each *vertex* of @f$\Gamma_h@f$ by searching a fan
    of rays for the one reaching @f$\Gamma@f$ nearest, restricted to the
    directions that leave @f$D_h@f$ through both faces meeting there; along a
    face, the unit tangents of its two vertices are interpolated. The search is
    what makes a path exist where a normal would miss, and the sharing of the
    vertex path is what makes the swept regions tile — which is the property
    the vertex-first construction is *for*, and which reading it as merely a
    way to find a nearby point on @f$\Gamma@f$ misses.

    Two dimensions only. The construction generalises, and the reference says
    so, but nothing here has been run in three.

    **Both halves of the reference's crossing restriction are built, and the
    cone is OFF BY DEFAULT because neither reason for wanting it survived
    measurement.** CS-Extensions section 2.4.1 restricts the direction at a
    vertex to a half space @f$H(x)@f$ intersected with a cone @f$C(x)@f$
    assembled from the *background* mesh's edges there. @f$C(x)@f$ needs a
    background mesh; rather than take one as an argument, this class recovers
    it when the mesh it is given is a SubMesh, and @a use_cone asks for it.

    **It does not close the aerofoil's flux order**, which is what it was built
    for and which this branch asserted for several sessions was the only thing
    standing between the extension and the reference's Table 6. Measured, the
    cone restricts every vertex of @f$\Gamma_h@f$ at every refinement and is
    strictly tighter than the half space at most of them, @f$\pi/2@f$ becoming
    @f$\pi/8@f$; the tiling residual is 1.13e-2 either way and the flux rates
    agree to the fourth digit. That order loss turned out to be pre-asymptotic
    and to recover on its own -- see the miniapp's header comment.

    **WITHDRAWN: "it costs the far face's quadrature".** This note used to say
    that, summed over a circle at a fixed rule, ExtensionBoundaryQuadrature()'s
    residual against @f$|\Gamma|@f$ was a 1e-11 floor without the cone and as
    much as 6e-5 with it (MEQ), that a 40th- or 80th-order rule recovered it,
    and so that the cone roughened the foot map along a face and a caller
    wanting it should raise the face rule. The numbers were right and the
    mechanism was not. The circle (radius 0.45 about the centre of the unit
    square) passes exactly through four mesh vertices at n = 20 and 40 and
    through none at n = 24 -- which is the "mesh dependence" this note
    recorded -- and at such a vertex the direction search below was broken.
    Moving the radius by 1e-9 takes the 6.1e-5 to 9.7e-11 with the old code,
    and the fix takes it to 2e-10 or better at every n and every rule, cone or
    no cone. The cone costs no quadrature that has been measured.

    **A vertex of @f$\Gamma_h@f$ lying on @f$\Gamma@f$** -- where
    @f$\Gamma@f$ meets the background's boundary, or passes exactly through a
    mesh vertex, so the level set there is zero and its computed sign is
    round-off -- takes the limit of the nearest-endpoint rule as the vertex
    approaches @f$\Gamma@f$: the rays are ranked by the slope of the level set
    along them, rather than by a bisection that can no longer tell them apart.
    Before that, such a vertex aborted at +1e-16 (gffp). At 0 or -1e-16 it
    built, but took the first rays in the fan's order, 70 degrees from the
    direction a vertex 1e-6 inside gets. The interpolated foot map then ran
    backwards along the faces there, which is what the withdrawn note above
    measured, and what ExtensionBoundaryQuadrature()'s note used to call a
    pinch face.

    Must be evaluated through the FaceElementTransformations overload: a
    direction interpolated along a face is not a function of the point alone.
    The mesh must outlive the path. */
class VertexConePath : public TransferPath
{
   const Mesh *mesh;
   PositionFunction phi;
   VectorPositionFunction grad_phi;
   real_t search_length;
   int n_rays, n_keep, search_steps, max_iter;
   real_t tol;

   /// Per face, the unit tangents of its two vertices, in the order the face's
   /// reference coordinate visits them; four entries per face.
   Array<real_t> tang;
   Array<int> has_tangent;
   int n_widened{};
   int n_coned{};
   int n_gamma_h_vertices{};
   int n_tighter{};
   bool have_cone{};

   /// Choose the direction at one vertex, given the outward normals of the
   /// faces of @f$\Gamma_h@f$ meeting there and, when a background mesh
   /// supplied one, the cone @f$C(x)@f$ as a centre angle and half width.
   /// Returns false if no ray reached @f$\Gamma@f$ even after the admissible
   /// fan was widened.
   /// @a counts accumulates the widened, cone-restricted and tighter tallies
   /// for this vertex, in that order, so that several vertices can be searched
   /// at once without sharing the members they end up in.
   bool VertexDirection(const Vector &x, const Array<real_t> &normals,
                        real_t cone_centre, real_t cone_half, Vector &t,
                        int counts[3]) const;

   /// The interpolated unit direction at @a ip of the face of @a FTr, and,
   /// if @a dt is not null, its derivative along the face (2 x 1).
   void FaceDirection(FaceElementTransformations &FTr,
                      const IntegrationPoint &ip, Vector &t,
                      DenseMatrix *dt) const;

public:
   /** @param mesh_           the mesh of @f$D_h@f$.
       @param gamma_h_attr    the boundary attribute of @f$\Gamma_h@f$.
       @param phi_            the level set, negative inside @f$\Omega@f$.
       @param search_length_  how far along a ray to look for @f$\Gamma@f$.
       @param n_rays_         rays in the fan searched at each vertex.
       @param n_keep_         how many of the nearest endpoints found to
                              average before shooting once more along their
                              mean, which is what the reference does and what
                              keeps the direction from jumping between two
                              nearly equal rays.
       @param use_cone_       apply CS-Extensions section 2.4.1's cone
                              @f$C(x)@f$ as well as the half space
                              @f$H(x)@f$. **Off by default, and the reasons are
                              measured rather than stylistic** -- see the note
                              above the class. It needs the mesh to be a
                              SubMesh with a live parent; given anything else
                              there is no cone whatever this says.
       @param phi_thread_safe_ the caller's PROMISE that @a phi_ may be
                              called from several threads at once. With it,
                              and with OpenMP and no Device configured, the
                              per-vertex search -- the ray fans and their
                              root finds, which is where the constructor's
                              time goes -- runs threaded. The answer is the
                              same either way: each vertex is searched alone
                              and the tallies are summed after. Off by
                              default, because nothing here can check it. */
   VertexConePath(const Mesh &mesh_, int gamma_h_attr, PositionFunction phi_,
                  real_t search_length_, int n_rays_ = 16, int n_keep_ = 3,
                  int search_steps_ = 32, real_t tol_ = 1e-13,
                  int max_iter_ = 100, bool use_cone_ = false,
                  bool phi_thread_safe_ = false);

   using TransferPath::Endpoint;

   /** @brief Not available: this family is defined face by face.

       The base class's point-and-normal form cannot express an interpolated
       direction, and silently falling back to the normal would reintroduce
       exactly the two failures this family exists to repair. */
   void Endpoint(const Vector &x, const Vector &n, Vector &xbar) const override;

   void Endpoint(FaceElementTransformations &FTr, const IntegrationPoint &ip,
                 Vector &xbar) const override;

   /** @brief Give the gradient of the level set, and EndpointJacobian()
       returns @f$\partial a/\partial\xi@f$ in closed form instead of
       false.

       @a grad_phi_ fills its second argument, of the size of the first, with
       @f$\nabla\phi@f$ of the SAME level set this family was built on, at
       any point it is asked about -- the endpoints lie on @f$\Gamma@f$,
       outside the mesh. Nothing here can check that the two agree, and a
       wrong gradient gives a wrong derivative with no error; the unit test
       compares the two against a central difference, which is the check to
       copy.

       The derivative is exact in the face point, where the central difference
       it replaces carries an @f$O(\epsilon|a|/\texttt{fd_step})@f$
       round-off and so a dependence on the step. It needs the face map's
       second derivatives where the face is curved.

       **A path meeting @f$\Gamma@f$ at a grazing angle ABORTS**, naming the
       face, the endpoint and @f$|\nabla\phi\cdot t|/|\nabla\phi|@f$, at
       @f$|\nabla\phi\cdot t| \le \sqrt\epsilon\,|\nabla\phi|@f$ or a
       gradient that is not finite. There @f$s'@f$ diverges -- the crossing
       slides along @f$\Gamma@f$ without bound, and past tangency jumps to a
       farther piece of it -- and that is the path map's, so a central
       difference would only hide it. The fix is the family's directions.
       CheckPathIncidence() finds the shallow paths before a solve.

       **At a corner of a polygonal @f$\Gamma@f$** the endpoint is smooth
       on either side and kinks where the ray passes the corner. Which side's
       derivative a point exactly there gets is @a grad_phi_'s choice: a
       GridFunction's gradient at a point on an element boundary is one
       element's. A face whose image straddles a corner loses its rule's
       order either way, which only splitting the face's rule at the corner
       would recover; that needs @f$\Gamma@f$'s corners, and nothing here
       has them.

       Call it before the family is used: it is configuration, and the
       family is read-only afterwards (see TransferPath). */
   void SetLevelSetGradient(VectorPositionFunction grad_phi_)
   { grad_phi = std::move(grad_phi_); }

   /// With the interpolated direction's own derivative for @f$t'@f$. False
   /// without a gradient.
   bool EndpointJacobian(FaceElementTransformations &FTr,
                         const IntegrationPoint &ip,
                         DenseMatrix &dadxi) const override;

   /** @brief How many vertices needed the admissible fan widened past the
       condition that the path leave @f$D_h@f$ through both adjacent faces.

       Non-zero means Assumption P.1 of the analysis is violated somewhere, so
       it is worth reporting rather than hiding: the method may still run, and
       the estimate no longer covers it. */
   int NumWidened() const { return n_widened; }

   /// Whether the cone @f$C(x)@f$ was available, i.e. whether the mesh handed
   /// to the constructor was a SubMesh with a parent to read edges from.
   bool HasCone() const { return have_cone; }

   /// How many vertices had their search restricted by @f$C(x)@f$ rather than
   /// by the half space alone. Zero with no background mesh.
   int NumConeRestricted() const { return n_coned; }
   int NumTighter() const { return n_tighter; }

   /// Vertices of @f$\Gamma_h@f$ a direction was searched at.
   int NumVertices() const { return n_gamma_h_vertices; }
};


/** @brief A family that keeps the paths of another, except where a path runs
    too close to the plane of the face it leaves; there it is tilted towards
    the face normal. **Opt-in, and it costs the exact tiling of
    @f$D_h^c@f$.**

    The analysis asks of a path direction @f$t@f$ that
    @f$t\cdot n_e \ge \gamma > 0@f$ (Cockburn, Qiu and Solano 2014, section
    5). Nothing in a closest-point map guarantees it. Onto a sphere the
    closest-point paths are radial, so on a face lying in a plane through the
    centre every path runs inside the face. Background meshes built on the
    centre's coordinate planes have such faces, and whether one ends up on
    @f$\Gamma_h@f$ depends on which elements were selected. CheckPathAngles()
    reports it.

    It is not harmless. On the extension miniapp's ball (problem 4, order 1,
    closest-point paths), the elements with such a face carry 65% to 92% of
    the transfer's excess flux error. There the flux error is 1.82 to 2.58
    times that of the boundary-fitted solve on the same @f$D_h@f$, against
    0.77 to 0.90 at offsets that select no such face. Tilting only the paths
    with @f$t\cdot n_e < \gamma@f$ brings every offset to 0.69 to 0.90, for
    @f$\gamma@f$ from 0.15 to 0.5, and leaves the potential's ratio at 1.00
    to 1.02.

    A path with @f$\hat m\cdot n_e < \gamma@f$, where @f$\hat m@f$ is the
    unit direction of the wrapped family's path, is replaced by the ray along
    @f$\gamma n_e + \sqrt{1-\gamma^2}\,\hat w@f$. Here @f$\hat w@f$ is the
    part of @f$\hat m@f$ orthogonal to @f$n_e@f$. The ray is run to the first
    crossing of the level set. Every other path is the wrapped family's,
    unchanged.

    **The cost:** tilted paths no longer tile the region beyond
    @f$\Gamma_h@f$ exactly. On the ball the measure of @f$D_h^c@f$ is off by
    0.7% at @f$\gamma = 0.15@f$ and 4% at 0.5, against 2e-12 untilted. The
    solve on @f$D_h@f$ does not use that measure. ExtensionRegionQuadrature()
    and the estimator terms built on it do. Take the smallest @f$\gamma@f$
    that clears the faces CheckPathAngles() reports.
    MarkLevelSetSubdomain()'s path-checking overload is the alternative that
    keeps the tiling, by choosing @f$D_h@f$ so that no such face arises.

    Like every family here it is `const` after construction and holds no
    per-call state. It holds a reference to @a base, which must outlive it. */
class TiltedPath : public TransferPath
{
   const TransferPath &base;
   PositionFunction phi;
   VectorPositionFunction grad_phi;
   real_t gamma;
   real_t search_length;
   int search_steps;
   real_t tol;
   int max_iter;

   /// Replace @a xbar by the tilted path's end if the path from @a x is too
   /// close to the face. Returns whether it was.
   bool Tilt(const Vector &x, const Vector &n, Vector &xbar) const;

   /// The tilted direction for the wrapped family's endpoint @a xbar, if the
   /// path from @a x is too close to the face; returns whether it is.
   bool TiltDirection(const Vector &x, const Vector &n, const Vector &xbar,
                      Vector &t) const;

public:
   /** @param base_           the family whose paths are kept where they are
                              steep enough.
       @param phi_            the level set: negative inside @f$\Omega@f$.
       @param gamma_          the smallest cosine between a path and the
                              outward face normal that is kept, in (0, 1).
       @param search_length_  how far along a tilted ray to look for
                              @f$\Gamma@f$, as for LevelSetPath.
       @param search_steps_   how many steps bracket the crossing. */
   TiltedPath(const TransferPath &base_, PositionFunction phi_, real_t gamma_,
              real_t search_length_, int search_steps_ = 32,
              real_t tol_ = 1e-13, int max_iter_ = 100);

   /// The wrapped family's path from @a x, tilted if need be.
   void Endpoint(const Vector &x, const Vector &n,
                 Vector &xbar) const override;

   /// The wrapped family's face-based path, tilted if need be.
   void Endpoint(FaceElementTransformations &FTr, const IntegrationPoint &ip,
                 Vector &xbar) const override;

   /** @brief Give the gradient of the level set, and EndpointJacobian()
       returns @f$\partial a/\partial\xi@f$ in closed form instead of
       false.

       @a grad_phi_ fills its second argument, of the size of the first, with
       @f$\nabla\phi@f$ of the SAME level set this family was built on, at
       any point it is asked about -- the endpoints lie on @f$\Gamma@f$,
       outside the mesh. Nothing here can check that the two agree, and a
       wrong gradient gives a wrong derivative with no error; the unit test
       compares the two against a central difference, which is the check to
       copy.

       The derivative is exact in the face point, where the central difference
       it replaces carries an @f$O(\epsilon|a|/\texttt{fd_step})@f$
       round-off and so a dependence on the step. It needs the face map's
       second derivatives where the face is curved.

       **A path meeting @f$\Gamma@f$ at a grazing angle ABORTS**, naming the
       face, the endpoint and @f$|\nabla\phi\cdot t|/|\nabla\phi|@f$, at
       @f$|\nabla\phi\cdot t| \le \sqrt\epsilon\,|\nabla\phi|@f$ or a
       gradient that is not finite. There @f$s'@f$ diverges -- the crossing
       slides along @f$\Gamma@f$ without bound, and past tangency jumps to a
       farther piece of it -- and that is the path map's, so a central
       difference would only hide it. The fix is the family's directions.
       CheckPathIncidence() finds the shallow paths before a solve.

       **At a corner of a polygonal @f$\Gamma@f$** the endpoint is smooth
       on either side and kinks where the ray passes the corner. Which side's
       derivative a point exactly there gets is @a grad_phi_'s choice: a
       GridFunction's gradient at a point on an element boundary is one
       element's. A face whose image straddles a corner loses its rule's
       order either way, which only splitting the face's rule at the corner
       would recover; that needs @f$\Gamma@f$'s corners, and nothing here
       has them.

       Call it before the family is used: it is configuration, and the
       family is read-only afterwards (see TransferPath). */
   void SetLevelSetGradient(VectorPositionFunction grad_phi_)
   { grad_phi = std::move(grad_phi_); }

   /** @brief The wrapped family's Jacobian where the path is not tilted.
       Where it is, the tilted ray's in closed form if this family was given
       the gradient AND the wrapped family has a Jacobian of its own at that
       point (the tilted direction is built from the wrapped path's), and
       false otherwise, so that the caller falls back to a central
       difference. The wrapped family's gradient is the wrapped family's to
       be given. */
   bool EndpointJacobian(FaceElementTransformations &FTr,
                         const IntegrationPoint &ip,
                         DenseMatrix &dadxi) const override;
};


/** @brief The extension operator @f$E_h@f$: an element's own polynomial,
    evaluated outside the element.

    Mathematically trivial -- @f$E_h(q_h)|_{K^{ext}}(y) := q_h|_K(y)@f$ -- and
    the only thing to get right is that the reference coordinates of a point
    outside the element are not clamped to the reference element.
    ElementTransformation::TransformBack() does clamp: it uses
    InverseElementTransformation's default #NewtonElementProject solver, which
    projects every iterate back into the reference element, so a point outside
    silently comes back as the nearest point of the element boundary and the
    extension degenerates into a constant. This class configures the
    unrestricted #Newton solver instead.

    For an affine element the composition of the reference basis with the
    inverse map is a polynomial in the physical coordinates, so this is the
    polynomial extension the method is written against. For an element with a
    non-affine map -- a general quadrilateral, hexahedron or wedge -- it is the
    reference-space extension instead, which is the natural generalisation but
    not the same object.

    **What it holds is per-element STATE rather than scratch, so it cannot be
    shared between threads.** SetElement() points the inverse transformation at
    one element and TransformBack() reads it; two threads working on two
    elements do not merely spoil each other's arithmetic, they take each
    other's ELEMENT, and what comes back is a well-formed reference point for
    the wrong map. Nothing aborts. HDGExtensionIntegrator therefore keeps its
    one behind `#ifndef MFEM_THREAD_SAFE` and hands it to LiftBasis() as an
    argument, so that the thread-safe arm gets one per call. */
class ElementExtension
{
   mutable InverseElementTransformation inv_tr;

public:
   ElementExtension();

   /// Set the element whose polynomials are to be extended.
   void SetElement(ElementTransformation &Tr) { inv_tr.SetTransformation(Tr); }

   /** @brief Reference coordinates of the physical point @a y under the
       element map, not restricted to the reference element.

       Returns false if the Newton solve failed to converge, in which case
       @a ip is not meaningful. Note that the element transformation's own
       integration point is modified by the solve. */
   bool TransformBack(const Vector &y, IntegrationPoint &ip) const;
};


/** @brief The line integral @f$\int_\sigma C\,\boldsymbol{u}\cdot
    \boldsymbol{m}\,ds@f$ along the straight path from @a x to @a xbar.

    @param Cu       evaluates @f$C(y)\,\boldsymbol{u}(y)@f$ at a physical point
                    @a y, which lies outside the mesh. Its output is passed in
                    already sized, as MFEM's vector functions expect.
    @param line_ir  a rule on Geometry::SEGMENT. On a straight-sided element the
                    flux is a polynomial of the degree of the flux space along
                    the path, so while @f$C@f$ is constant a rule exact to that
                    degree integrates the integrand exactly; the outer integral
                    over the face is the one that is not exact, because
                    @f$a(x)@f$ is not polynomial in @f$x@f$. **A @f$C@f$ that
                    varies along the path is not covered by that**, and no
                    Gauss rule of fixed order is exact for it;
                    HDGExtensionIntegrator::SetLineRule() takes a rule built for
                    the integrand instead. See the note on
                    HDGExtensionIntegrator for what that does and does not buy
                    when @f$C@f$ is near-singular.

    This is the whole of the lifting apart from the datum @f$g(a(x))@f$, and it
    is stated as a free function because that makes it checkable on its own:
    fed the exact flux, it must return @f$p(x) - p(a(x))@f$ to quadrature
    accuracy, whatever the path. */
real_t PathIntegral(const VectorPositionFunction &Cu, const Vector &x,
                    const Vector &xbar, const IntegrationRule &line_ir);


/** @brief The Dirichlet datum of @f$\Gamma@f$, seen from @f$\Gamma_h@f$.

    Evaluates @f$g(a(x))@f$ at a point @a x of a boundary face, which is the
    part of the lifting that does not depend on the solution. Paired with
    VectorBoundaryFluxLFIntegrator it supplies @f$\langle g\circ a, v\cdot
    n\rangle_e@f$, which is exactly how the miniapps already give a Dirichlet
    datum to the flux equation -- the only difference being where the datum is
    read from.

    Must be evaluated on a FaceElementTransformations, since the path family
    may need the face normal.

    @warning **Not reentrant, and that is MFEM's convention for Coefficient
    rather than an omission here.** `fem/coefficient.hpp` carries 26 `mutable`
    scratch members over 19 classes -- InnerProductCoefficient,
    MatrixProductCoefficient, the six Cartesian, cylindrical and spherical
    ones among them -- and neither it nor `coefficient.cpp` contains a single
    MFEM_THREAD_SAFE. A threaded assembly therefore has to give each thread its
    own coefficient whatever this class does, and guarding these members alone
    would buy nothing while advertising a guarantee the objects beside them do
    not offer. Give each thread its own. The TransferPath behind it may be
    shared, and so may a coefficient whose Eval() uses only locals -- which
    FunctionCoefficient's does. */
class PathTraceCoefficient : public Coefficient
{
   const TransferPath &path;
   PositionFunction g;
   mutable Vector xbar;

public:
   PathTraceCoefficient(const TransferPath &path_, PositionFunction g_)
      : path(path_), g(std::move(g_)) { }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override;
};


/** @brief The solution-dependent part of the transferred Dirichlet datum, as a
    contribution to the flux mass form on a face of @f$\Gamma_h@f$.

    The transferred datum is @f$\varphi_h = g\circ a + L_e(\boldsymbol{u}_h)@f$
    and the flux equation reads it as @f$\langle \varphi_h, v\cdot n
    \rangle_e@f$. Only the first term is data; the second depends on the
    unknown flux, so it belongs on the left, and this integrator is it:

    @f[ \langle L_e(\boldsymbol\varphi_j), \boldsymbol\varphi_i \cdot n
        \rangle_e, \qquad
        L_e(\boldsymbol\varphi_j)(x) = \int_{\sigma(x)}
        C\,E_h(\boldsymbol\varphi_j)\cdot\boldsymbol{m}\,ds. @f]

    **The block is element-local.** A face of @f$\Gamma_h@f$ belongs to one
    element, and the extension @f$E_h@f$ on the region beyond it is that
    element's own polynomial, so nothing outside the element is read. The term
    is therefore an addition to the element's flux mass block and leaves the
    hybridization -- the constraint, its transpose, the static condensation --
    untouched. That is why the weak boundary route costs nothing structural
    here and the essential-trace route would.

    @a C is the same coefficient the flux mass form carries: the *inverse* of
    the diffusion tensor, as in @f$(C\boldsymbol{u}, v)@f$. It is evaluated at
    points outside the element, through that element's transformation, so a
    coefficient defined by a function of position extends as that function and
    one defined by a grid function extends as that grid function's own
    polynomial.

    The space is assumed fully discontinuous with @a vdim equal to the space
    dimension -- the arrangement `VectorMassIntegrator` also assumes, and the
    one the HDG configuration of `DarcyForm` uses.

    @note **What a @a C that grows without bound towards @f$\Gamma@f$ costs
    depends on how @f$\Gamma@f$ meets the place where it blows up.** Three
    configurations have been measured. In all of them the datum was
    consistent with the exact solution at every end of every path, which is a
    precondition for reading any of these numbers: a datum wrong only where
    paths end on the degenerate set was, in an application, the whole of an
    order loss first attributed to the transfer.

    **Tangent contact: the line rule is not where the order is lost.** This
    was measured on the disc of the unit tests (radius 0.45, centre
    (0.5, 0.5)). Its lowest point touches the line @f$y = y_0@f$, and the line
    otherwise lies outside @f$\Omega@f$. The tensor is
    @f$K = \mathrm{diag}(1, y - y_0 + \epsilon)@f$, which degenerates linearly
    on the line. Uniform doublings, n = 16 to 256:
    - The lifted datum of the projected flux, integrated exactly (a graded
      rule that a finer one reproduces to three digits), converges at about
      k+1 at orders 1 and 2, against k+2 for @f$K = I@f$.
    - In the solve at order 1 (both face rules on the shared 2k default) the
      potential keeps k+1 throughout, and the flux stalls between n = 128 and
      256: rate 0.83 with the default line rule, 0.73 with Gauss 20 and with a
      converged graded rule. The excess lies next to the tangent point. With
      @f$K = I@f$ the transfer matches the boundary-fitted solve on the same
      @f$D_h@f$ to n = 256. At order 2 the flux rate falls to 1.81 by n = 128.
    - The default Gauss rule, Gauss of order 20 and the converged graded rule
      give the same rates to n = 128, and at n = 256 the default's flux error
      is the SMALLER, 5.28e-6 against 5.65e-6. @f$\epsilon@f$ from 1e-9 to
      1e-6 changes nothing, and 1e-3 changes little. (These figures were
      7.55e-6, 1.01e-5 and rates 0.47, 0.05 before the face rules were
      shared; see GetFaceRule().)
    - The mechanism is the lifting's amplification: the element's flux error
      is multiplied by @f$\int C\,ds@f$ along the path. Here that levels off
      near 7 while the paths shorten like h, so relative to @f$K = I@f$ it
      grows like 1/h. The quadrature points never come within about
      @f$h^2@f$ of the line, which is why @f$\epsilon@f$ is not the knob.
      SetProjectedLift() removes it and restores k+1; see there.

    **A corner on the degenerate set, interior angle 90 degrees or less: the
    rule is irrelevant.** The domain is a disc segment above @f$y = y_0@f$
    with the same @f$K@f$. The fitted chord carries zero normal flux, and the
    arc is @f$\Gamma@f$. Corners of 15, 45 and 90 degrees were measured at
    orders 1 and 2, with closest-point paths onto the arc. Five line rules
    (the default, Gauss 20 and 80, two graded rules) agree to about 1% at
    every n. At such a corner the paths run AWAY from the degenerate line, so
    @a C falls along each of them. A 15-degree corner costs a constant, about
    20 times the no-transfer error at n = 512, but not a rate.

    **A corner on the degenerate set, interior angle above 90 degrees: the
    line rule does not help, and the converged graded rule can hurt.** Paths
    close to the outward normal would cross into @f$y < y_0@f$, where @a K is
    not defined. So the paths next to the corner must end at the corner or on
    the chord, and their end-point map has a kink inside a face. That makes
    GetFaceRule()'s requirement bite: with mismatched face rules, every such
    family loses the flux order there, already with @f$K = I@f$ (rate 0.3 to
    1.5 at 120 to 165 degrees). With a shared rule, a clamped closest-point
    family and VertexConePath on the segment both keep rate 2.00. With
    degenerate @a K, shared rules, VertexConePath at 165 degrees, the chord
    in @f$\Gamma@f$ and an exact datum:
    - At order 1 to n = 256, the Gauss rules keep the potential at rate 2,
      within 8% of the no-transfer error. The graded rule breaks at n = 256,
      where the potential's rate falls to 0.75.
    - At order 2 to n = 128, every rule gives the potential rate about 3,
      and the rules differ by at most 1.4 times.

    An application has this obtuse configuration. A loss parabola meets the
    degenerate boundary @f$\mu = 0@f$ at about 15 degrees, so the domain's
    interior angle is about 165 degrees. The parabola is Dirichlet and the
    axis carries regularity. With a mesh line on the parabola's foot, so that
    no element of @f$\Gamma_h@f$ reaches out over the axis, no path comes near
    the degeneracy. There the default and a converged graded line rule agree
    to four figures, and the potential converges at k+1 for k = 1, 2, 3
    (3.10, 3.03 at k = 2; 4.27, 4.10 at k = 3).

    So what decides whether the rule matters is whether the paths approach
    the degeneracy, not the corner angle: at the acute corners above the
    angle is what keeps them away, and here the mesh is. Where they do
    approach it, as at a tangent contact, SetProjectedLift() is the remedy.
    The application's earlier report of a graded rule lifting the rate from
    0.76 to 1.82 was taken while paths from a misplaced face of
    @f$\Gamma_h@f$ ended on the axis with a datum imposed there. Its authors
    have withdrawn it. */
class HDGExtensionIntegrator : public BilinearFormIntegrator
{
   const TransferPath &path;
   Coefficient *C{};
   MatrixCoefficient *MC{};
   real_t sign;
   int line_order;
   const IntegrationRule *line_ir{};
   bool projected_lift{};

   /** @brief Per-face and per-point scratch, behind MFEM's guard.

       The convention is that an integrator's working set lives here, so that
       a build setting MFEM_THREAD_SAFE has no shared mutable members left and
       one integrator object may be entered from several threads at once. This
       class declared all eleven of these outside the guard and so ignored the
       option silently.

       #ext is the one that makes the failure more than a scrambled sum.
       AssembleFaceMatrix() points its InverseElementTransformation at the
       element owning the face and LiftBasis() then inverts that map once per
       point of every path; two threads on faces of two different elements do
       not merely overwrite each other's arithmetic, they invert each other's
       ELEMENT. The extension that comes back is some other element's
       polynomial, and nothing aborts. */
#ifndef MFEM_THREAD_SAFE
   Vector shape, shape_ext, nor, x, xbar, m, y, CTm;
   DenseMatrix Cmat, L;
   ElementExtension ext;
#endif

public:
   /** @param path_        the transferring paths.
       @param C_           the inverse diffusion tensor, as a scalar.
       @param sign_        **the negative of the sign with which the datum
                           enters the flux right-hand side**, since that is
                           where the term comes from. The default, `+1`, is
                           therefore the one to use with the branch's
                           convention of giving the *negated* potential to
                           VectorBoundaryFluxLFIntegrator, as `pNatural` does
                           in the harnesses. Measured rather than argued: with
                           the sign reversed the solve diverges outright, the
                           error growing by four orders between the two finest
                           meshes.
       @param line_order_  order of the rule along the path; negative takes
                           twice the element order plus two. */
   HDGExtensionIntegrator(const TransferPath &path_, Coefficient &C_,
                          real_t sign_ = +1., int line_order_ = -1)
      : path(path_), C(&C_), sign(sign_), line_order(line_order_) { }

   HDGExtensionIntegrator(const TransferPath &path_, MatrixCoefficient &C_,
                          real_t sign_ = +1., int line_order_ = -1)
      : path(path_), MC(&C_), sign(sign_), line_order(line_order_) { }

   /** @brief Use @a ir along every path instead of the Gauss rule the order
       selects; nullptr restores that rule.

       @a ir is on Geometry::SEGMENT, @f$t = 0@f$ at the face point of
       @f$\Gamma_h@f$ and @f$t = 1@f$ at its image on @f$\Gamma@f$, and it must
       integrate the constant exactly: its points lie in @f$[0,1]@f$ and its
       weights sum to one. Both are checked here. The rule is not copied, so it
       must outlive this object, as one passed to SetIntRule() must.

       An order selects a Gauss rule, which is exact only while @a C is a
       polynomial along the path; see PathIntegral(). This is for a rule built
       to the caller's integrand instead -- a composite one graded towards
       @f$t = 1@f$ where @a C grows there. Pass the same rule to any
       PathLiftCoefficient or TransferredDatumCoefficient that is to report
       the datum this one imposes. */
   void SetLineRule(const IntegrationRule *ir);

   /// The rule set by SetLineRule(), or nullptr if the order selects it.
   const IntegrationRule *GetLineRule() const { return line_ir; }

   /** @brief Lift @f$q_h = \Pi_K(C\boldsymbol{u}_h)@f$, the element's own
       @f$L^2@f$ projection of @f$C\boldsymbol{u}_h@f$, instead of integrating
       @a C along the path. **Opt-in.**

       This is the three-field form @f$q = -\nabla p@f$,
       @f$C\boldsymbol{u} = q@f$, @f$-\nabla\cdot\boldsymbol{u} = g@f$, with
       @f$q_h@f$ in the flux space and @f$q@f$ as the extended variable. Its
       constitutive equation @f$(q_h, v) = (C\boldsymbol{u}_h, v)@f$ makes
       @f$q_h@f$ that projection, so the discrete system is the two-field
       one, and only the lifting changes. The lifting becomes
       @f$\int_\sigma E_h(q_h)\cdot\boldsymbol{m}\,ds@f$. @a C is evaluated
       only inside the element, through the same @f$C@f$-weighted mass matrix
       the flux mass form uses, and never along the path. The integrand along
       a straight path is then a polynomial, so the order's Gauss rule is
       exact and SetLineRule() is not needed.

       With @a C constant on the element the projection is the identity, and
       this is the default lifting. It differs where @a C varies strongly
       beyond @f$\Gamma_h@f$.

       **This is the fix to use where @a C diverges like 1/x towards the
       Dirichlet boundary**, that is, where @f$K@f$ degenerates on or near
       @f$\Gamma@f$. The default lifting multiplies the element's flux error
       by @f$\int_\sigma C\,ds@f$. With @f$C \sim 1/x@f$ that is O(1) along a
       path of length O(h) instead of O(h), so it costs the flux an order.
       The projected lifting integrates @f$E_h(q_h)@f$, which approximates the
       smooth @f$-\nabla p@f$, so no divergent factor appears along the path.

       Measured on the disc tangent to @f$y = y_0@f$, with
       @f$K = \mathrm{diag}(1, y - y_0 + \epsilon)@f$, closest-point paths,
       and the flux error on @f$D_h@f$ against the boundary-fitted solve on the
       same @f$D_h@f$:

       | order | n   | default lifting      | projected lifting    | fitted   |
       |-------|-----|----------------------|----------------------|----------|
       | 1     | 256 | 5.28e-06 (rate 0.83) | 1.14e-06 (rate 1.64) | 7.92e-07 |
       | 1     | 512 | 9.20e-07 (rate 2.52) | 2.71e-07 (rate 2.08) | 1.99e-07 |
       | 2     | 128 | 5.16e-08 (rate 1.81) | 5.44e-09 (rate 3.06) | 5.12e-09 |
       | 2     | 256 | 2.89e-08 (rate 0.84) | 6.83e-10 (rate 2.99) | 6.43e-10 |

       At order 1 the default lifting's rate is erratic (1.84, 0.83, 2.52 from
       n = 64 to 512), at 3 to 7 times the fitted error. That is because the
       amplification depends on how close the nearest quadrature point comes
       to the tangent point. At order 2 it loses the order outright: 45 times
       the fitted error at n = 256, and its potential falls to rate 2.44 as
       well. The projected lifting stays within 1.06 to 1.44 times the fitted
       error throughout, and its potential keeps k+1. Its excess over the
       fitted solve sits next to the tangent point, and away from it the two
       agree. Every line rule (the default, Gauss 20,
       graded) gives the same numbers here, as the polynomial integrand
       predicts. With @f$K = I@f$ it reproduces the default lifting to every
       printed digit.

       The projection uses the flux mass form's default rule, and the two
       must agree. Raising the projection's rule alone broke the solve at
       n = 256 (3.8e-05). Raising both made it no better than the defaults. */
   void SetProjectedLift(bool on = true) { projected_lift = on; }

   /// Whether SetProjectedLift() is on.
   bool GetProjectedLift() const { return projected_lift; }

   /** @brief The rule over a face of @f$\Gamma_h@f$ that AssembleFaceMatrix()
       uses: the one given to SetIntRule(), or else order @f$2k@f$ on
       @a face_geom, with @f$k@f$ the order of @a el1.

       **The linear form that carries the other half of the datum,
       @f$\langle g\circ a, v\cdot n\rangle_e@f$, must integrate with this
       same rule.** For the exact solution the two halves sum to the
       potential, which is smooth whatever the paths. Each half alone is
       smooth only where the end-point map @f$a(x)@f$ is, and that map has a
       kink inside any face whose paths are split between the corner of
       @f$\Omega@f$ and an arc beside it. There the two quadrature errors
       cancel only under one rule. Measured on a disc segment with a
       120-degree corner, @f$K = I@f$ and an exact datum, at n = 32:
       - Mismatched rules make the flux error 8 times the no-transfer error
         at order 1, and 845 times at order 2.
       - A shared rule, of any order, makes it 1.04 and 1.34 times.

       The @f$2k@f$ default is chosen to be VectorBoundaryFluxLFIntegrator's,
       so two integrators left on their defaults agree. Once either is given a
       rule of its own they no longer do, so the safe route is
       ExtensionDatumLFIntegrator, which reads this rule face by face. */
   const IntegrationRule &GetFaceRule(const FiniteElement &el1,
                                      Geometry::Type face_geom) const
   {
      if (IntRule) { return *IntRule; }
      return IntRules.Get(face_geom, 2 * el1.GetOrder());
   }

   void AssembleFaceMatrix(const FiniteElement &el1, const FiniteElement &el2,
                           FaceElementTransformations &Trans,
                           DenseMatrix &elmat) override;

   using BilinearFormIntegrator::AssembleFaceMatrix;

   /** @brief The lifting @f$L_e(\boldsymbol{u}_h)(x)@f$ at one point of a face
       of @f$\Gamma_h@f$, which is the half of the transferred datum that the
       assembled block above absorbs and never reports.

       @param el1    the element owning the face, as in AssembleFaceMatrix().
       @param Trans  the face transformations; @a Trans.Elem2No must be
                     negative.
       @param ip     a point of the *face* reference element.
       @param elfun  the element's flux dofs, ordered as AssembleFaceMatrix()
                     orders the block it builds -- `Ordering::byNODES`, so
                     component @a d of basis function @a j is entry
                     `dof*d + j`.

       Same quadrature and same geometry as the assembly: the block that
       function forms is @f$\pm\sum_q w_q\,(\boldsymbol\varphi_i\cdot n)\,
       L_e(\boldsymbol\varphi_j)@f$, and this returns the second factor
       against a given @a elfun rather than against each basis function in
       turn. Costs one line integral per call, so a caller sweeping a face
       should sweep its quadrature points rather than call it pointwise from
       elsewhere. */
   real_t ComputeLift(const FiniteElement &el1,
                      FaceElementTransformations &Trans,
                      const IntegrationPoint &ip, const Vector &elfun);

private:
   /// The projection @f$P = M^{-1}M_C@f$ of SetProjectedLift(), for the
   /// element behind @a Tr. @f$M@f$ is the vector mass matrix and @f$M_C@f$
   /// the @f$C@f$-weighted one, both on Ordering::byNODES.
   void ProjectionMatrix(const FiniteElement &el1, ElementTransformation &Tr,
                         DenseMatrix &P);

   /// Replace each lifted basis function by the lifting of its projection:
   /// @a Lmat(j,d) becomes the sum over (i,e) of @a Lmat(i,e) P(dof*e+i,
   /// dof*d+j).
   static void ApplyProjection(const DenseMatrix &P, DenseMatrix &Lmat);

   /// The rule along the path, so that the assembly and ComputeLift() cannot
   /// drift apart on it -- a different rule is a different lifting.
   const IntegrationRule &LineRule(const FiniteElement &el1) const
   {
      if (line_ir) { return *line_ir; }
      return IntRules.Get(Geometry::SEGMENT,
                          (line_order >= 0) ? line_order
                          : (2 * el1.GetOrder() + 2));
   }

   /** @brief @f$L_e@f$ of every basis function of @a el1, at the face point
       whose image is @a x and whose path displacement is @a m. Fills
       @a Lmat(j,d) for the basis function @f$\text{shape}_j e_d@f$.

       Expects @a ext_w to have been set to the element and leaves
       @a elem_tr's integration point wherever the last step of the path put
       it.

       The extension and the four pieces of per-point scratch are the
       CALLER'S, rather than this object's, so that a MFEM_THREAD_SAFE build
       allocates them once per face instead of once per quadrature point of
       it -- and, for @a ext_w, because the element it is pointed at has to
       survive from AssembleFaceMatrix() into here. Without the guard they are
       the members of the same name, so the two arms run the same arithmetic
       on the same objects. */
   void LiftBasis(const FiniteElement &el1, ElementTransformation &elem_tr,
                  const Vector &x, const Vector &m,
                  const IntegrationRule &lir, ElementExtension &ext_w,
                  Vector &shape_w, Vector &y_w, Vector &CTm_w,
                  DenseMatrix &Cmat_w, DenseMatrix &Lmat);
};


/** @brief The datum half of the transferred Dirichlet datum,
    @f$s\,\langle g, v\cdot n\rangle_e@f$ on a face of @f$\Gamma_h@f$,
    integrated with the face rule of the HDGExtensionIntegrator that carries
    the other half.

    It is VectorBoundaryFluxLFIntegrator's arithmetic, a scalar @a g against
    the normal component of a vector test function, with the rule taken from
    HDGExtensionIntegrator::GetFaceRule() on every face. A later
    SetIntRule() on the extension integrator therefore moves both halves
    together. See GetFaceRule() for why the rule must be shared, and what a
    mismatch costs.

    @a g is normally a PathTraceCoefficient. The sign convention is
    VectorBoundaryFluxLFIntegrator's: the harnesses hand it the *negated*
    potential with @a s = 1, and pair that with the extension integrator's
    default sign of +1.

    Add it with LinearForm::AddBdrFaceIntegrator(), on the same boundary
    marker as the extension integrator. It holds a reference to @a ext, so it
    must not outlive the form that owns @a ext. */
class ExtensionDatumLFIntegrator : public LinearFormIntegrator
{
   const HDGExtensionIntegrator &ext;
   Coefficient &g;
   real_t sign;
#ifndef MFEM_THREAD_SAFE
   Vector shape, nor;
#endif

public:
   ExtensionDatumLFIntegrator(const HDGExtensionIntegrator &ext_,
                              Coefficient &g_, real_t s = 1.0)
      : ext(ext_), g(g_), sign(s) { }

   /// Not available: the datum lives on faces of @f$\Gamma_h@f$ only.
   void AssembleRHSElementVect(const FiniteElement &el,
                               ElementTransformation &Tr,
                               Vector &elvect) override;

   void AssembleRHSElementVect(const FiniteElement &el,
                               FaceElementTransformations &Tr,
                               Vector &elvect) override;

   using LinearFormIntegrator::AssembleRHSElementVect;
};


/** @brief The solution-dependent half of the transferred datum, evaluable
    after a solve.

    @f$L_e(\boldsymbol{u}_h)@f$ is formed inside
    HDGExtensionIntegrator::AssembleFaceMatrix() and contracted straight into
    the element's flux mass block, so once the solve is over there is no way to
    ask what was imposed on a given face. That defeats any face indicator that
    compares a computed trace against the datum actually imposed -- @f$\eta_5@f$
    of Sanchez-Vizuet, Solano and Cerfon eq. (20) among them, which on
    @f$\Gamma_h@f$ otherwise compares the postprocessed potential against zero
    and picks up an @f$O(\text{dist}(\Gamma_h,\Gamma))@f$ term that swamps the
    rest. This coefficient is that term, at no new quadrature and no new
    geometry.

    Must be evaluated on a FaceElementTransformations of a face of
    @f$\Gamma_h@f$, as PathTraceCoefficient must, and for the same reason.

    @warning Evaluating it **moves both transformations**: the path endpoint
    resets them and the line quadrature then walks the element's own
    transformation out beyond @f$\Gamma_h@f$. Anything a caller needs from the
    geometry at the face point -- the normal, the shape functions, the weight --
    must be taken before the call, exactly as
    HDGExtensionIntegrator::AssembleFaceMatrix() takes them.

    @warning **Not reentrant, and that is MFEM's convention for Coefficient
    rather than an omission here.** `fem/coefficient.hpp` carries 26 `mutable`
    scratch members over 19 classes -- InnerProductCoefficient,
    MatrixProductCoefficient, the six Cartesian, cylindrical and spherical
    ones among them -- and neither it nor `coefficient.cpp` contains a single
    MFEM_THREAD_SAFE. A threaded assembly therefore has to give each thread its
    own coefficient whatever this class does, and guarding these members alone
    would buy nothing while advertising a guarantee the objects beside them do
    not offer. Give each thread its own. The TransferPath behind it may be
    shared, and so may a coefficient whose Eval() uses only locals -- which
    FunctionCoefficient's does. The HDGExtensionIntegrator held here by value IS
    guarded, so in a MFEM_THREAD_SAFE build the shared state left is #vdofs
    and #elfun and nothing deeper. */
class PathLiftCoefficient : public Coefficient
{
   const GridFunction &u;
   HDGExtensionIntegrator integ;
   mutable Array<int> vdofs;
   mutable Vector elfun;

public:
   /** @param path_        the transferring paths, the same family the
                           integrator was given.
       @param u_           the computed flux.
       @param C_           the inverse diffusion tensor, as the integrator
                           takes it.
       @param line_order_  order of the rule along the path; pass what the
                           integrator was given, since a different rule is a
                           different lifting. */
   PathLiftCoefficient(const TransferPath &path_, const GridFunction &u_,
                       Coefficient &C_, int line_order_ = -1)
      : u(u_), integ(path_, C_, +1., line_order_) { }

   PathLiftCoefficient(const TransferPath &path_, const GridFunction &u_,
                       MatrixCoefficient &C_, int line_order_ = -1)
      : u(u_), integ(path_, C_, +1., line_order_) { }

   /// As HDGExtensionIntegrator::SetProjectedLift(); pass what the integrator
   /// was given.
   void SetProjectedLift(bool on = true) { integ.SetProjectedLift(on); }

   /// The rule along the path; pass what the integrator was given. See
   /// HDGExtensionIntegrator::SetLineRule().
   void SetLineRule(const IntegrationRule *ir) { integ.SetLineRule(ir); }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override;
};


/** @brief The transferred datum @f$\varphi_h = g\circ a +
    L_e(\boldsymbol{u}_h)@f$ itself, which is what a face indicator on
    @f$\Gamma_h@f$ needs and what neither half is on its own.

    @a g is **the Dirichlet datum, not its negation.** The branch's harnesses
    hand `VectorBoundaryFluxLFIntegrator` the negated potential and pair that
    with the integrator's default `sign = +1`; this class states
    @f$\varphi_h@f$ in its own terms instead, so the caller passes @f$g@f$ and
    reads @f$\varphi_h@f$ back. Fed the exact flux it returns @f$p@f$ on
    @f$\Gamma_h@f$ to quadrature accuracy -- @f$g(a(x)) = p(a(x))@f$ and
    @f$L_e@f$ of the exact flux is @f$p(x) - p(a(x))@f$, which is the identity
    PathIntegral() is documented by and the one this is tested against. */
class TransferredDatumCoefficient : public Coefficient
{
   PathTraceCoefficient trace;
   PathLiftCoefficient lift;

public:
   TransferredDatumCoefficient(const TransferPath &path_, PositionFunction g_,
                               const GridFunction &u_, Coefficient &C_,
                               int line_order_ = -1)
      : trace(path_, std::move(g_)), lift(path_, u_, C_, line_order_) { }

   TransferredDatumCoefficient(const TransferPath &path_, PositionFunction g_,
                               const GridFunction &u_, MatrixCoefficient &C_,
                               int line_order_ = -1)
      : trace(path_, std::move(g_)), lift(path_, u_, C_, line_order_) { }

   /// The rule along the path; pass what the integrator was given. See
   /// HDGExtensionIntegrator::SetLineRule().
   void SetLineRule(const IntegrationRule *ir) { lift.SetLineRule(ir); }

   /// As HDGExtensionIntegrator::SetProjectedLift(); pass what the integrator
   /// was given.
   void SetProjectedLift(bool on = true) { lift.SetProjectedLift(on); }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   { return trace.Eval(T, ip) + lift.Eval(T, ip); }
};


/** @brief A point of the region beyond a face of @f$\Gamma_h@f$, with the
    quadrature weight that goes with it. */
struct ExtensionPoint
{
   const Vector &y;                  ///< the physical point, outside @f$D_h@f$
   const Vector &xbar;               ///< the end @f$a(x)@f$ of its path
   const IntegrationPoint &face_ip;  ///< where that path leaves @f$\Gamma_h@f$
   real_t t;                         ///< along the path; 0 on @f$\Gamma_h@f$
   real_t weight;                    ///< quadrature weight, Jacobian included
};


/** @brief The default central-difference step for the two quadrature sweeps
    below, @f$\epsilon^{1/3}@f$: 6.1e-06 in double and 4.9e-03 in single.

    A central difference balances an @f$O(h^2)@f$ truncation against an
    @f$O(\epsilon/h)@f$ round-off, so @f$\epsilon^{1/3}@f$ is the optimum --
    but ACCURACY IS NOT WHY THIS FOLLOWS @a real_t. The literal `1e-6` this
    replaced was tuned for double, where it is the right order and where
    @f$\epsilon/h@f$ is 2.2e-10; in a single-precision build the same literal
    makes it **0.119**, and ExtensionBoundaryQuadrature()'s degeneracy floor is
    a hundred times that, so the floor exceeds every measure a real face can
    have and EVERY quadrature point is skipped. The routine then returns
    normally having called the caller's callback zero times -- no error, no
    warning, and any accumulator it was filling keeps its initialised zero.
    Measured by MEQ on a DIII-D case: 21 points per face in double, 0 in
    single, a coupling silently dropped from a Jacobian, and a Newton that
    converged at full rate onto a 10% wrong answer.

    A caller may still pass its own step; the floor is checked against the
    face's own scale either way, so a step too small for the build's precision
    is now refused rather than silently emptying the rule.

    **THE STENCIL STAYS ON THE FACE.** TransferPath::Endpoint() is only ever
    asked about points of the face: where @f$\xi \pm h@f$ would leave it, the
    sweeps take the one-sided second-order difference into the face instead
    (and near a corner of a triangular face, a halved step). They used to step
    off the end. A path from a point extrapolated past a vertex lying near
    @f$\Gamma@f$ starts outside @f$\Gamma@f$, and VertexConePath aborts there
    -- unreachable in double at this default, reached by MEQ in single, where
    the step is 4.9e-03 and the first point of a 21-point Gauss rule is at
    3.1e-03. Every point whose centred stencil already stayed on the face gets
    the same difference, bit for bit, as before.

    THIS CHANGED THE DOUBLE DEFAULT TOO, from the literal `1e-6` to 6.06e-06,
    and a default is not observable from a call site -- so it is written here
    rather than only in the commit that did it. The move was measured before it
    was taken and is an improvement in the one quantity that can say so without
    a solve or an exact solution: on `extension -o 1 -n 8 -r 1 -no-ctl` every
    error norm is identical to every printed digit while the relative error in
    the measure of @f$D_h^c@f$ falls from 2.80e-11 to 8.46e-12.

    AND THE TRAP THAT FOLLOWS IS NOT ABOUT ACCURACY. Where a caller computes
    one quantity two ways and compares them -- a row of an operator against a
    direct quadrature of the integral it claims to be -- BOTH SIDES MUST BE
    GIVEN THE SAME STEP. The difference's error is a deterministic function of
    the step, so with one step it is common to the two sides and cancels
    exactly, and such a comparison reads at round-off however coarse the step
    is; with two it does not cancel, and the full difference appears at once.
    That is a property of the comparison and not of either sweep, and it is
    much larger than the change in the step suggests: MEQ pinned a transmission
    row at `1e-6` and left the reference contraction on this default, and the
    case moved from 7.8e-16 to 1.27e-06 -- seven orders for a 6x step, which no
    @f$O(h^2)@f$ truncation can account for. Their own case had already
    recorded the same shape one axis over, for the quadrature ORDER, and
    pinning it on both sides is what they had done about that.

    **WHERE THE STEP STOPS MATTERING.** A family that returns its
    TransferPath::EndpointJacobian() -- ClosestPointPath with its map's
    Jacobian, a level-set family given SetLevelSetGradient() -- is never
    differenced, and both sweeps ignore @a fd_step for it, so two such sweeps
    have no step to share. */
inline real_t ExtensionFDStep()
{
   return std::cbrt(std::numeric_limits<real_t>::epsilon());
}


/** @brief Quadrature over @f$K^{ext}_e@f$, the region swept by the paths
    issuing from one face of @f$\Gamma_h@f$.

    The region is the image of the face times the unit interval under

    @f[ y(\xi,t) = x(\xi) + t\,(a(x(\xi)) - x(\xi)), @f]

    and integrating in @f$(\xi,t)@f$ against the Jacobian of that map is
    integrating over @f$K^{ext}_e@f$. The Jacobian's columns are
    @f$(1-t)\,\partial x/\partial\xi_i + t\,\partial a/\partial\xi_i@f$ and
    @f$a(x)-x@f$. The derivative of @f$a@f$ along the face is the family's
    own TransferPath::EndpointJacobian() where it has one, and a central
    difference otherwise, because a path family is a map and not required to
    supply one. (Until 2026-10-28 this routine always took the difference,
    and only ExtensionBoundaryQuadrature() asked the family.)

    Summed over the faces this is a quadrature over @f$D_h^c = \Omega\setminus
    D_h@f$ **provided the regions tile it**, which is a property of the path
    family and not of this routine: adjacent faces must agree on the path
    through a shared vertex. The closest-point map does agree, since it depends
    on the point and not on the face. A family whose paths follow each face's
    own normal does not, and leaves gaps and overlaps at the vertices -- which
    is why CS-Extensions builds its general family by interpolating the paths
    of the vertices rather than taking each face's normal.

    **The tiling is checkable and should be checked**: the weights sum to
    @f$|K^{ext}_e|@f$, so summing them over the faces must give
    @f$|\Omega| - |D_h|@f$.

    **Measured in THREE dimensions as well as two, and it holds.** A ball of
    radius 0.45 carved from a tetrahedral background mesh, with
    ClosestPointPath onto the sphere and @a face_ir a rule on
    Geometry::TRIANGLE where the two-dimensional case passes one on
    Geometry::SEGMENT: the swept measure reproduces
    @f$\tfrac{4}{3}\pi R^3 - |D_h|@f$ to 3.7e-11, 2.0e-11, 3.5e-11 and
    7.5e-11 at n = 8, 16, 24 and 32, against the 1.6e-10 the same check gives
    in two dimensions. So nothing in the sweep, the Jacobian's columns or the
    central-difference derivative along the face is two-dimensional, and this
    routine needs no dimension-specific treatment. The last figure is from a
    mesh whose trace solve did NOT converge, which is the point of quoting it:
    the tiling is a property of the geometry alone and is measurable where the
    discretisation is not.

    @param FTr       a boundary face of @f$\Gamma_h@f$.
    @param path      the transferring paths.
    @param face_ir   a rule on the face.
    @param line_ir   a rule on Geometry::SEGMENT, for the path direction.
    @param visit     called once per quadrature point. */
void ExtensionRegionQuadrature(
   FaceElementTransformations &FTr, const TransferPath &path,
   const IntegrationRule &face_ir, const IntegrationRule &line_ir,
   const std::function<void(const ExtensionPoint &)> &visit,
   real_t fd_step = ExtensionFDStep());


/** @brief A point of @f$\Gamma@f$ reached by the paths from one face of
    @f$\Gamma_h@f$, with the surface quadrature weight and the normal there. */
struct ExtensionBoundaryPoint
{
   const Vector &y;                  ///< the physical point, on @f$\Gamma@f$
   const Vector &nu;                 ///< the OUTWARD UNIT normal of @f$\Gamma@f$
   const IntegrationPoint &face_ip;  ///< where its path leaves @f$\Gamma_h@f$
   /** @brief Surface weight, the Jacobian included, and **SIGNED**.

       Negative where the path family's foot map reverses along @f$\Gamma@f$,
       which for a staircase @f$\Gamma_h@f$ it does on some faces. Summing the
       weights integrates over @f$\Gamma@f$; summing their absolute values
       integrates the length the map traverses, which is a different quantity
       and is larger. See ExtensionBoundaryQuadrature(). */
   real_t weight;
};


/** @brief Quadrature over the piece of @f$\Gamma@f$ reached by the paths
    issuing from one face of @f$\Gamma_h@f$: the @f$t=1@f$ face of the map
    ExtensionRegionQuadrature() sweeps.

    That map is @f$y(\xi,t) = x(\xi) + t\,(a(x(\xi)) - x(\xi))@f$, so at
    @f$t=1@f$ it is @f$\xi \mapsto a(x(\xi))@f$ and its image is the piece of
    @f$\Gamma@f$ the face is responsible for. The columns of that map's
    Jacobian are @f$\partial a/\partial\xi_i@f$ alone -- the @f$(1-t)@f$ that
    multiplies the face's own Jacobian has gone -- and they are taken by central
    differences, because a path family is a map and is not required to supply a
    derivative. CalcOrtho() turns them into the normal, whose magnitude IS the
    surface Jacobian.

    **The orientation is fixed by the paths, not by the face.** The normal is
    signed so that @f$\nu\cdot(a(x)-x) > 0@f$: the paths run outward from
    @f$D_h@f$, so a normal agreeing with them is the outward normal of
    @f$\Gamma@f$. Taking the face's own normal instead would be wrong wherever
    @f$\Gamma@f$ and @f$\Gamma_h@f$ are not parallel, which is everywhere that
    matters.

    **The weight is SIGNED, and that is what makes this a quadrature over
    @f$\Gamma@f$ rather than over the path the foot map traces.** The map is
    not required to be monotone along @f$\Gamma@f$. (This note used to give a
    staircase @f$\Gamma_h@f$ as the case: a "pinch face" on a circle cut from
    a diagonally split Cartesian mesh, traversing 0.0163 of arc to cover
    0.0060. That was VertexConePath's defect at a vertex lying on
    @f$\Gamma@f$, which the circle passed through; with it fixed, nothing on
    that circle backtracks. The unit test now makes a family that does.)
    Where the map reverses, so does
    @f$\partial a/\partial\xi@f$, so CalcOrtho()'s normal turns inward, the
    orientation test fires and the segment is subtracted. What survives is the
    net multiplicity, which is one.

    Summed over the faces this is therefore a quadrature over @f$\Gamma@f$
    **provided the images cover it**, which is a property of the path family and
    not of this routine -- the same hypothesis ExtensionRegionQuadrature() needs,
    minus the monotonicity the signing removes. **It is checkable and should be
    checked**: the weights must sum to @f$|\Gamma|@f$, exactly as the volume
    weights must give @f$|\Omega| - |D_h|@f$.

    **Why a visitor and not an integrator.** The caller decides what to build:
    a boundary integral of an extended flux against a spectral basis is a row
    vector rather than a form, so an Integrator subclass would be the wrong
    object for it and would narrow this to one use. This is the shape
    ExtensionRegionQuadrature() already has, for the same reason.

    **A face whose image is a point is skipped, not refused.** A closest-point
    family collapses a face lying along a ray to a single foot --- for
    ClosestPointPath::Sphere, any radial face of a staircase @f$\Gamma_h@f$,
    which near the poles is every second one. It covers zero of @f$\Gamma@f$,
    its endpoint feet coincide, and its neighbours cover @f$\Gamma@f$ between
    them, so the right contribution is none. @a visit is then not called for
    that point.

    The test is relative rather than against zero, and it has to be: a
    Jacobian annihilates the degenerate direction exactly only in exact
    arithmetic, and in floating point a radial face yields a residue pointing in
    no particular direction. Its weight is of the same order, so a quadrature
    sum is unharmed and the fault is invisible there --- but a caller reading
    @a nu alone gets a normal measured at 90 degrees from the true one.

    **The floor matches how the Jacobian was obtained.** A closed form from
    EndpointJacobian() is accurate to rounding, so the floor is
    @f$10^{-12}|\partial x/\partial\xi|@f$. A central difference divides by
    @a fd_step and so has a noise floor of @f$\epsilon|a|/\texttt{fd_step}@f$
    --- about @f$10^{-10}@f$ at the default step, six orders above
    @f$\epsilon@f$ --- and gets a floor a hundred times that instead. A
    legitimate face measures five orders clear of the looser of them.

    @param FTr       a boundary face of @f$\Gamma_h@f$.
    @param path      the transferring paths.
    @param face_ir   a rule on the face.
    @param visit     called once per quadrature point, EXCEPT where the image
                     is degenerate; see above.
    @param fd_step   the central-difference step for @f$\partial a/\partial
                     \xi@f$, as in ExtensionRegionQuadrature(). */
void ExtensionBoundaryQuadrature(
   FaceElementTransformations &FTr, const TransferPath &path,
   const IntegrationRule &face_ir,
   const std::function<void(const ExtensionBoundaryPoint &)> &visit,
   real_t fd_step = ExtensionFDStep());


/** @brief Selection of the polyhedral subdomain @f$D_h@f$ of a background mesh.

    @f$D_h@f$ is the set of elements lying entirely inside @f$\Omega@f$. For a
    convex @f$\Omega@f$ this is decided by the vertices alone -- a convex
    element is inside if and only if all its vertices are -- and that is what
    @a extra_refine = 0 tests. For a domain that is not convex an element can
    have every vertex inside and still be cut by @f$\Gamma@f$, and additional
    points have to be tested; @a extra_refine >= 1 additionally samples the
    lattice of the reference element refined that many times.

    @param mesh         the background mesh.
    @param phi          the level set: negative inside @f$\Omega@f$.
    @param offset       an element is taken only if @f$\phi \le -@f$@a offset
                        at every point tested, which pushes @f$\Gamma_h@f$
                        further from @f$\Gamma@f$. Meaningful as a distance
                        only where @a phi is a signed distance. Used to set
                        @f$\operatorname{dist}(\Gamma_h,\Gamma)@f$ deliberately,
                        as the robustness studies of the papers do.
    @param marker       set to 1 for the elements of @f$D_h@f$ and 0 otherwise.
    @param extra_refine how many times to refine the sampling lattice; 0 tests
                        the vertices only, which is exact for a convex
                        @f$\Omega@f$.
    @param phi_thread_safe the caller's PROMISE that @a phi may be called from
                        several threads at once. With it, and with OpenMP and
                        no Device configured, the element loop runs threaded:
                        every element is decided alone, so the marker and the
                        count are those of the serial loop. On a mesh with
                        nodes the lattice is mapped through
                        Mesh::GetElementTransformation(), which writes the
                        nodal space's shared DofTransformation, so there the
                        loop stays serial whenever @a extra_refine >= 1.
    @returns the number of elements selected. */
int MarkLevelSetSubdomain(const Mesh &mesh, const PositionFunction &phi,
                          real_t offset, Array<int> &marker,
                          int extra_refine = 0, bool phi_thread_safe = false);

/** @brief The selection above, followed by the removal of every element that
    would give @f$\Gamma_h@f$ a face whose paths run too close to its plane.
    **Opt-in.**

    The selection is made as by the other overload. Then, for every face
    between a selected element and an unselected one, the paths of @a path
    are evaluated at the face's quadrature points, with the normal pointing
    out of the selected element. If any has a cosine to that normal below
    @a min_cos, the selected element is removed. Removing elements creates
    new faces, so this repeats until none is found or @a max_sweeps is
    reached.

    Elements are only ever REMOVED, so @f$D_h@f$ stays inside @f$\Omega@f$
    and keeps whatever the first selection guaranteed. The price is a
    @f$\Gamma_h@f$ locally further from @f$\Gamma@f$, and so locally longer
    paths. Unlike TiltedPath it leaves the path family, and with it the exact
    tiling of @f$D_h^c@f$, untouched. See TiltedPath for why the condition
    matters and what it costs to ignore it.

    @a path is evaluated through its point-and-normal Endpoint(), on the
    background mesh, before @f$D_h@f$ exists. ClosestPointPath,
    LevelSetPath and TiltedPath wrapping either of them can be used.
    VertexConePath cannot, being defined face by face on @f$D_h@f$ itself.
    In one dimension there is no such face, and this is the other overload.
    @a phi_thread_safe reaches the first selection only; the sweeps evaluate
    the paths through the mesh's shared face transformation and stay serial.

    @returns the number of elements selected. Check the outcome with
    CheckPathAngles(): if @a max_sweeps ran out, some faces may remain. */
int MarkLevelSetSubdomain(const Mesh &mesh, const PositionFunction &phi,
                          real_t offset, Array<int> &marker,
                          const TransferPath &path, real_t min_cos,
                          int extra_refine = 0, int max_sweeps = 20,
                          bool phi_thread_safe = false);

/// What CheckPathAngles() found.
struct PathAngleReport
{
   real_t min_cos = 1.0;  ///< smallest cosine of a path to its face normal
   int points = 0;        ///< quadrature points examined
   int below = 0;         ///< of them, those below the threshold
   int faces_below = 0;   ///< faces of @f$\Gamma_h@f$ with at least one
   int worst_bdr = -1;    ///< the boundary element where min_cos was found
};

/** @brief How steeply the paths leave @f$\Gamma_h@f$: the cosine between
    each path and the outward normal of its face, over the quadrature points
    of every face of @f$\Gamma_h@f$.

    The analysis needs that cosine bounded below by a positive constant, and
    TiltedPath explains what a family that gets close to zero costs. On the
    extension miniapp's ball, faces at cosines of 0 and 0.07 carried most of
    an excess flux error, while 0.14 and above cost nothing measurable. Hence
    the default threshold of 0.15. Paths of zero length, from points already
    on @f$\Gamma@f$, are skipped.

    @param D_h           the subdomain mesh.
    @param gamma_h_attr  the boundary attribute of @f$\Gamma_h@f$.
    @param path          the family to examine.
    @param threshold     a point below this cosine counts in
                         PathAngleReport::below.
    @param order         the face rule's order. */
PathAngleReport CheckPathAngles(Mesh &D_h, int gamma_h_attr,
                                const TransferPath &path,
                                real_t threshold = 0.15, int order = 4);

/// What CheckPathIncidence() found.
struct PathIncidenceReport
{
   real_t min_cos = 1.0;  ///< smallest cosine of a path to @f$\Gamma@f$'s normal
   int points = 0;        ///< quadrature points examined
   int below = 0;         ///< of them, those below the threshold
   int faces_below = 0;   ///< faces of @f$\Gamma_h@f$ with at least one
   int worst_bdr = -1;    ///< the boundary element where min_cos was found
};

/** @brief How squarely the paths meet @f$\Gamma@f$: the cosine between each
    path and the level set's gradient at its endpoint, @f$\nabla\phi\cdot
    t/|\nabla\phi|@f$, over the quadrature points of every face of
    @f$\Gamma_h@f$.

    CheckPathAngles() asks how a path LEAVES @f$\Gamma_h@f$; this asks how
    it ARRIVES. The endpoint's derivative along the face carries
    @f$1/(\nabla\phi\cdot t)@f$, so a cosine @f$c@f$ amplifies the
    sensitivity of the endpoint to the face point by @f$1/c@f$, and at zero
    the path grazes @f$\Gamma@f$ and has no derivative: the level-set
    families abort there when given a gradient
    (LevelSetPath::SetLevelSetGradient()). Run this first to find such paths
    before a solve does. A path into @f$\Gamma@f$ from outside reads
    negative; one whose gradient vanishes or is not finite reads 0. Paths of
    zero length, from points already on @f$\Gamma@f$, are skipped.

    The default threshold, 0.1, bounds the amplification at ten. It is a
    choice, not a measurement.

    @param D_h           the subdomain mesh.
    @param gamma_h_attr  the boundary attribute of @f$\Gamma_h@f$.
    @param path          the family to examine.
    @param grad_phi      the level set's gradient, at a point of
                         @f$\Gamma@f$.
    @param threshold     a point below this cosine counts in
                         PathIncidenceReport::below.
    @param order         the face rule's order. */
PathIncidenceReport CheckPathIncidence(Mesh &D_h, int gamma_h_attr,
                                       const TransferPath &path,
                                       const VectorPositionFunction &grad_phi,
                                       real_t threshold = 0.1, int order = 4);

} // namespace mfem

#endif
