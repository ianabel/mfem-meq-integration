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

#ifndef MFEM_BILININTEG_HDG
#define MFEM_BILININTEG_HDG

#include "../bilininteg.hpp"
#include "../lininteg.hpp"

#include <vector>

namespace mfem { class ParMesh; }

namespace mfem
{

/** Integrator for the DG form:
    $$
      \alpha \langle (u \cdot n) \{v\},[w] \rangle,
    $$
    where $v$ and $w$ are the trial and test variables, respectively, and $\rho$/$u$ are
    given scalar/vector coefficients. $\{v\}$ represents the average value of $v$ on
    the face and $[v]$ is the jump such that $\{v\}=(v_++v_-)/2$ and $[v]=(v_+-v_-)$ for the
    face with $+$ and $-$ sides. For boundary elements, $v_-=0$. The vector coefficient,
    $u$, is assumed to be continuous across the faces and when given the scalar coefficient.

    The corresponding HDG stabilization is then
    $$\begin{align}
        \langle \{\tau\} v_\pm, w_\pm \rangle, & -\langle \tau_\mp \lambda,          w_\pm \rangle,\\
        \langle \{\tau\} v_\pm, \mu   \rangle, & -\langle (\tau_+ + \tau_-) \lambda, \mu   \rangle,
    \end{align}$$
    where $\tau_\pm = |\alpha (u \cdot n)| \pm \alpha (u \cdot n)$
    and $\lambda$, $\mu$ are the trial and test trace functions, respectively.

    On a boundary face, where only the $+$ side exists and $n$ is the outward
    normal, the trace-trace term is halved to $\langle \{\tau\} \lambda, \mu
    \rangle$. The element row is then the consistent one-sided flux,
    $\alpha (u \cdot n) \lambda + |\alpha (u \cdot n)| (u - \lambda)$, but the
    trace row is $|\alpha (u \cdot n)| (u - \lambda)$, with no convective flux
    in it on either sign of $u \cdot n$. A flux datum imposed through the trace
    row (a linear form on the trace space, as in miniapps/hdg/convdiff)
    therefore converges to the DIFFUSIVE normal flux on every boundary face,
    inflow and outflow alike. HDGConvectionUpwindedIntegrator differs here: what
    a datum means there depends on the sign of $u \cdot n$.
    */
class HDGConvectionCenteredIntegrator : public DGTraceIntegrator
{
#ifndef MFEM_THREAD_SAFE
   Vector tr_shape, shape1, shape2;
#endif

public:
   HDGConvectionCenteredIntegrator(VectorCoefficient &u_, real_t a = 1.)
      : DGTraceIntegrator(u_, a, 0.) { }

   /** @brief The rule AssembleHDGFaceMatrix() uses on this interior face.

       ONE SOURCE OF TRUTH, and the reason it exists: the batched face
       assembly has to integrate at exactly the rule the per-face loop does or
       the two build different operators, and a rule copied into the batch
       driver would diverge silently the day this one changed.
       AssembleHDGFaceMatrix() calls this, so it cannot. */
   const IntegrationRule &GetHDGFaceIntRule(
      const FiniteElement &trace_el, const FiniteElement &el1,
      const FiniteElement &el2, FaceElementTransformations &Trans) const;

   /** @brief The rule the SINGLE-SIDE AssembleHDGFaceMatrix() integrates
       @a side of this face at.

       A separate overload for the same reason HDGDiffusionIntegrator has one:
       the one-sided routine takes the max over ONE element and the trace, and
       its OrderW() from that side's element alone, so on a mesh whose two
       neighbours differ in order the two rules are genuinely different. The
       side matters here where it does not for the diffusion form, OrderW()
       being the one term that is a property of an element rather than of an
       order.

       ONE SOURCE OF TRUTH, and this one arrived late: the expression lived
       inline in the one-sided AssembleHDGFaceMatrix() while the two-sided
       twin had already been lifted, so the batched SHARED-face pass -- which
       is the one-sided route on a partition boundary -- had nothing to ask
       and would have had to copy it. */
   const IntegrationRule &GetHDGFaceIntRule(
      int side, const FiniteElement &trace_el, const FiniteElement &el,
      FaceElementTransformations &Trans) const;

   void AssembleHDGFaceMatrix(const FiniteElement &trace_el,
                              const FiniteElement &el1,
                              const FiniteElement &el2,
                              FaceElementTransformations &Trans,
                              DenseMatrix &elmat) override;

   void AssembleHDGFaceMatrix(int side, const FiniteElement &trace_el,
                              const FiniteElement &el,
                              FaceElementTransformations &Trans,
                              DenseMatrix &elmat) override;

   void AssembleHDGFaceVector(int type,
                              const FiniteElement &trace_face_fe,
                              const FiniteElement &fe,
                              FaceElementTransformations &Tr,
                              const Vector &trfun, const Vector &elfun,
                              Vector &elvect) override;
};

/** Integrator for the DG form:
    $$
      \alpha \langle (u \cdot n) \{v\},[w] \rangle + \beta \langle |u \cdot n| [v],[w] \rangle,
    $$
    where $v$ and $w$ are the trial and test variables, respectively, and $\rho$/$u$ are
    given scalar/vector coefficients. $\{v\}$ represents the average value of $v$ on
    the face and $[v]$ is the jump such that $\{v\}=(v_++v_-)/2$ and $[v]=(v_+-v_-)$ for the
    face with $+$ and $-$ sides. For boundary elements, $v_-=0$. The vector coefficient,
    $u$, is assumed to be continuous across the faces and when given the scalar coefficient.

    The corresponding HDG stabilization is then
    $$\begin{align}
        \langle \tau_\pm v_\pm, w_\pm \rangle, & -\langle \tau_\mp \lambda,          w_\pm \rangle,\\
        \langle \tau_\pm v_\pm, \mu   \rangle, & -\langle (\tau_+ + \tau_-) \lambda, \mu   \rangle,
    \end{align}$$
    where $\tau_\pm = (\beta |u \cdot n| \pm 1/2 \alpha (u \cdot n))$
    and $\lambda$, $\mu$ are the trial and test trace functions, respectively.

    On a boundary face, where only the $+$ side exists and $n$ is the outward
    normal, the element row is the consistent one-sided flux,
    $\tau_+ u - \tau_- \lambda = \alpha (u \cdot n) \lambda + \tau_+ (u -
    \lambda)$, but the trace row is $\tau_+ u - (\tau_+ + \tau_-) \lambda =
    \tau_+ (u - \lambda) - \tau_- \lambda$, short of the consistent row by
    $\tau_+ \lambda$. A flux datum imposed through the trace row (a linear form
    on the trace space, as in miniapps/hdg/convdiff) therefore means different
    things on the two kinds of face. With the default $\beta = \alpha/2$,
    $\tau_+ = 0$ on inflow ($u \cdot n < 0$) and $\tau_- = 0$ on outflow, so as
    $u - \lambda \to 0$ the datum converges to
    - on an INFLOW face, the TOTAL (diffusive plus convective) normal flux,
      since the trace row's convective part is $\alpha (u \cdot n) \lambda$;
    - on an OUTFLOW face, the DIFFUSIVE normal flux alone (Danckwerts' outflow
      condition), since the trace row's convective part vanishes.

    With $\beta \neq \alpha/2$ the trace row's convective part is
    $-\tau_- \lambda$ on both kinds of face, which is neither. A homogeneous
    natural condition on an outflow face is thus a zero diffusive flux, which is
    what convdiff's "Neumann BC must be applied only for the heat flux" refers
    to. A total-flux datum on an outflow face is not imposed. On convdiff's
    problem 2 with the solution shifted by a constant, it gives an $O(1)$ error
    that does not decrease under refinement. Adding $\langle \tau_+ \lambda, \mu
    \rangle$ to those faces' trace-trace block (a boundary face integrator
    beside this one, with the same $\alpha$ and $\beta$) imposes it, and
    recovers the diffusive-datum arm's errors to three digits at orders 1 and 2.
    That condition is exponentially ill-conditioned in the Peclet number,
    though: in 1-D, $-\kappa T'' + c T' = 0$, $T(0) = 0$, with total flux $g$
    at the outflow end, gives $T = -(g/c)(e^{c x/\kappa} - 1)$. At
    $\kappa = 0.01$ on that problem, no mesh up to 256 x 256 converged.
    */
class HDGConvectionUpwindedIntegrator : public DGTraceIntegrator
{
#ifndef MFEM_THREAD_SAFE
   Vector tr_shape, shape1, shape2;
#endif

public:
   /// Construct integrator with $\beta = \alpha/2$.
   HDGConvectionUpwindedIntegrator(VectorCoefficient &u_, real_t a = 1.)
      : DGTraceIntegrator(u_, a) { }

   HDGConvectionUpwindedIntegrator(VectorCoefficient &u_, real_t a, real_t b)
      : DGTraceIntegrator(u_, a, b) { }

   /** @brief The rule AssembleHDGFaceMatrix() uses on this interior face.

       ONE SOURCE OF TRUTH, and the reason it exists: the batched face
       assembly has to integrate at exactly the rule the per-face loop does or
       the two build different operators, and a rule copied into the batch
       driver would diverge silently the day this one changed.
       AssembleHDGFaceMatrix() calls this, so it cannot. */
   const IntegrationRule &GetHDGFaceIntRule(
      const FiniteElement &trace_el, const FiniteElement &el1,
      const FiniteElement &el2, FaceElementTransformations &Trans) const;

   /** @brief The rule the SINGLE-SIDE AssembleHDGFaceMatrix() integrates
       @a side of this face at.

       A separate overload for the same reason HDGDiffusionIntegrator has one:
       the one-sided routine takes the max over ONE element and the trace, and
       its OrderW() from that side's element alone, so on a mesh whose two
       neighbours differ in order the two rules are genuinely different. The
       side matters here where it does not for the diffusion form, OrderW()
       being the one term that is a property of an element rather than of an
       order.

       ONE SOURCE OF TRUTH, and this one arrived late: the expression lived
       inline in the one-sided AssembleHDGFaceMatrix() while the two-sided
       twin had already been lifted, so the batched SHARED-face pass -- which
       is the one-sided route on a partition boundary -- had nothing to ask
       and would have had to copy it. */
   const IntegrationRule &GetHDGFaceIntRule(
      int side, const FiniteElement &trace_el, const FiniteElement &el,
      FaceElementTransformations &Trans) const;

   void AssembleHDGFaceMatrix(const FiniteElement &trace_el,
                              const FiniteElement &el1,
                              const FiniteElement &el2,
                              FaceElementTransformations &Trans,
                              DenseMatrix &elmat) override;

   void AssembleHDGFaceMatrix(int side, const FiniteElement &trace_el,
                              const FiniteElement &el,
                              FaceElementTransformations &Trans,
                              DenseMatrix &elmat) override;

   void AssembleHDGFaceVector(int type,
                              const FiniteElement &trace_face_fe,
                              const FiniteElement &fe,
                              FaceElementTransformations &Tr,
                              const Vector &trfun, const Vector &elfun,
                              Vector &elvect) override;
};

/** @brief A stabilization written as data, so that a kernel can evaluate it:
    $s(x) = c_0$, or $s(x) = c_0 / \max(x_k, x_f)$ at the physical point $x$.

    The second form is the diffusion-scaled $\tau = \tau_0 / \ell$ of Nguyen,
    Peraire and Cockburn with $\kappa = 1/x_k$, floored: in an axisymmetric
    problem posed in $(R, z)$ the flux equation carries $R$ and the
    stabilization that matches it is $\tau_0/R$, kept finite at the axis by the
    floor. These are the two stabilizations meq runs, and the reason this
    exists: an HDGStabilization is a host virtual taking an
    ElementTransformation, so a batched route that has to call it evaluates
    every face weight on the host. One that describes itself through
    HDGStabilization::GetRule() is evaluated on the device instead, at the
    physical point the element's map gives (see HDGFaceWeightsOnDevice()). */
struct HDGStabilizationRule
{
   real_t c0 = 0.0;     ///< the constant, or the numerator
   int coord = -1;      ///< k, the coordinate; negative for the constant c0
   real_t floor = 0.0;  ///< x_f, the floor on x_k

   /// $s$ at the physical point @a x, which has at least coord + 1 entries.
   MFEM_HOST_DEVICE real_t Eval(const real_t *x) const
   {
      if (coord < 0) { return c0; }
      // std::max(x_k, x_f) exactly, NaN included.
      const real_t r = (x[coord] < floor) ? floor : x[coord];
      return c0 / r;
   }
};

/** @brief Stabilization function of the HDG numerical flux,
    $$
       \hat q_h + \hat F_h = q_h + F(\hat u_h)
                            + s(u_h, \hat u_h)(u_h - \hat u_h) n .
    $$

    This is Eq. (5) of Nguyen, Peraire and Cockburn, J. Comput. Phys. 228
    (2009) 8841-8855, in which $s$ is a function of the potential and of its
    own trace rather than a coefficient. Their section 2.4 splits it as
    $s = s_{diff} + s_{conv}(u_h, \hat u_h)$ with $s_{diff} = \kappa/\ell$
    constant, and for a linear flux the positivity bound of their Eq. (7)
    reduces $s_{conv}$ to a constant as well -- which is exactly the
    stabilization HDGDiffusionIntegrator and HDGConvectionUpwindedIntegrator
    already apply. The constant case is therefore not a special case bolted on
    here; it is the specialization those integrators implement, and it keeps
    its own assembly path.

    A derived class that leaves IsConstant() true costs nothing at run time:
    the integrators query it once per face, never per quadrature point.

    **What this hook cannot express, which is worth knowing before writing a
    subclass.** Eval() returns ONE number per quadrature point and the
    integrator applies it to both sides of the face, so every stabilization
    reachable through here is *symmetric* in the sign of $u \cdot n$.
    Upwinding is not: the upwinded face flux carries $\tau_\pm =
    \beta|u\cdot n| \pm \tfrac12\alpha(u\cdot n)$, whose antisymmetric half
    is the whole point of it. So a convection-dominated face cannot be served
    by any $\tau$ from this interface, however it is scaled -- measured, and
    the numbers are on HDGConvectiveFloorStabilization. Convective upwinding
    belongs on the convection integrator; this hook is for shaping the
    symmetric part.

    A class that returns false makes the face term nonlinear in the unknowns
    even for a linear equation, so it is only meaningful on the residual and
    gradient assembly, which are the only paths that see the state. The
    bilinear-form path refuses it rather than silently dropping the
    dependence. */
class HDGStabilization
{
public:
   virtual ~HDGStabilization() { }

   /** @brief True when $s$ depends on neither the potential nor its trace.
       The default is the constant case.

       A statement about the STATE, not about position: an $s$ that varies
       from point to point and not with the solution is constant here. */
   virtual bool IsConstant() const { return true; }

   /** @brief Describe Eval() as an HDGStabilizationRule, or decline.

       Return true only when Eval() IS @a rule: for every argument it returns
       `rule.Eval(x)` at the physical point `x` of the transformation it is
       handed, and reads neither `s_diff`, `un`, `u` nor `uhat`. The batched
       face routes then weigh their points on the device and never call
       Eval(). Nothing checks the claim, so a class that makes it should
       evaluate Eval() through HDGStabilizationRule::Eval() as
       HDGRuleStabilization does. The default declines, and the batched routes
       call Eval() on the host as before. Consulted only when IsConstant() is
       true. */
   virtual bool GetRule(HDGStabilizationRule &rule) const
   { MFEM_CONTRACT_VAR(rule); return false; }

   /** @brief The stabilization at one quadrature point.
       @param s_diff the value the integrator forms on its own, that is the
                     constant part built from the diffusion coefficient and the
                     local element size, with any quadrature weight removed
       @param un     the normal component of the convective velocity, unscaled
       @param u      the potential at the point
       @param uhat   the trace of the potential at the point
       @param Tr     element transformation, with the integration point set */
   virtual real_t Eval(real_t s_diff, real_t un, real_t u, real_t uhat,
                       ElementTransformation &Tr) const
   { return s_diff; }

   /** @brief The derivatives of $s$ with respect to the potential and to its
       trace, written $\partial_1 s$ and $\partial_2 s$ in Eq. (15) of the
       reference. Called only when IsConstant() is false.

       These are not optional refinements. In a hybridized method the Jacobian
       is never assembled globally, so omitting them gives no wrong answer,
       only slow Newton convergence -- a failure that survives a passing
       regression suite. */
   virtual void EvalGrad(real_t s_diff, real_t un, real_t u, real_t uhat,
                         ElementTransformation &Tr,
                         real_t &d1s, real_t &d2s) const
   { d1s = 0.; d2s = 0.; }
};

/** @brief A stabilization that is not allowed below a floor.

    `HDGDiffusionIntegrator` builds its own value as
    $\tau = \beta\,(\hat n \cdot Q \hat n)/h$, which vanishes wherever the
    diffusion does *in the direction of the face normal*. Two separate findings
    on this branch run into that, and both are recorded in section 3 of
    `doc/HDG-ROADMAP.md`:

    * a coefficient that **degenerates** on part of the boundary, where
      $Q \to 0$ and the potential loses order -- 2.18 against a clean 2.99 at
      $k = 2$;
    * ~~a coefficient that is **anisotropic**~~ -- **this second claim is
      withdrawn; see below.** It read "the *flux* loses order -- 1.49 against
      2.00 at $k = 1$ and $\kappa_\perp/\kappa_\parallel = 10^{-2}$", and a
      floor does not buy that back.

    For the degeneracy the misbehaviour to fear is $\tau \to 0$ and the remedy
    is to refuse the small values and keep the large ones. That is what this
    does. It is a floor and not a replacement, so on faces where the built-in
    value is already big enough -- the ones aligned with the strong direction
    -- nothing changes.

    **What the anisotropic claim got wrong, measured over a longer sequence
    than it was.** The original ran $n = 8$ to $64$, and on this problem every
    mesh in that window is pre-asymptotic. Taken to $n = 256$ on
    `anisodiff -p 11` at $\kappa_\perp/\kappa_\parallel = 10^{-2}$, `-tf 1`
    against `-tf 0` differs by 27% at $n = 8$ and is **identical to every
    printed digit from $n = 128$ onwards**, at $k = 1$ and $k = 2$ alike; the
    flux rate converges to $1.00$ and $2.02$ -- that is $k$ -- with the floor
    and without it. At $k = 2$ the pre-asymptotic rates read 3.28 and 3.25
    before falling to 2.02, so a sweep stopping at $n = 32$ reports the design
    order and is wrong.

    The reason is structural rather than particular to that problem: this floor
    binds only where $\hat n \cdot Q \hat n < \tau_{min} h / \beta$, and the
    built-in value grows like $1/h$, so the binding set shrinks under
    refinement -- on a sheared field it contracts onto the isolated points
    where the field meets a mesh direction. **A floor cannot change the
    scaling of $\tau$; it can only lift faces where the coefficient
    collapses.** Where the collapse is on a fixed set, as in the degeneracy
    above, it lifts a fixed set and the order really is recovered -- which is
    why the degenerate half stands and is pinned to $n = 128$ by
    "HDG: a tau floor recovers the order a degeneracy costs".

    What the anisotropy actually costs, and what recovers what, is the rate
    table on HDGDiffusionIntegrator: an $O(1)$ $\tau$ buys the flux order the
    $O(1/h)$ default gives up, and half an order remains that belongs to the
    anisotropy itself and that no $\tau$ here repairs.

    The floor is an absolute stabilization, so it is the $\eta_d = \kappa/\ell$
    of Nguyen, Peraire and Cockburn section 3.6.3 with $\ell$ a fixed problem
    length scale, which is the scaling that holds $\tau$ constant under
    refinement. It is therefore a number of the size of
    $\kappa_\parallel/\ell$, not of $\kappa_\perp$.

    Constant, so it costs nothing per quadrature point and the bilinear
    assembly path accepts it. */
class HDGFloorStabilization : public HDGStabilization
{
   real_t tau_min;

public:
   /// @param tau_min_ the smallest stabilization any face may be given.
   HDGFloorStabilization(real_t tau_min_) : tau_min(tau_min_) { }

   real_t Eval(real_t s_diff, real_t, real_t, real_t,
               ElementTransformation &) const override
   { return (s_diff > tau_min) ? s_diff : tau_min; }
};


/** @brief A stabilization that is never below the CONVECTIVE scale either.

    `HDGFloorStabilization` above answers $\tau \to 0$ arriving from the
    diffusion. This answers the other way a face can be under-stabilized: the
    problem is convection-dominated there, and a $\tau$ built from the
    diffusion alone knows nothing about it.

    $$\tau = \max\left(s_{diff},\ \beta_c |u \cdot n|\right),$$
    optionally floored again by an absolute @a tau_min.

    **It does not do what it looks like it should, and the measurement is the
    reason to keep reading.** The obvious use -- replacing the upwinded face
    flux with a stabilization that carries the same convective magnitude --
    does not work, and not because the magnitude is wrong. On
    `anisodiff -p 11`, order 1, $\kappa_\perp/\kappa_\parallel = 10^{-2}$,
    $c = 100$, $64\times64$, relative $L^2$ flux error:

        beta_c        0.125    0.25     0.5      1        2        4        8       16
        this class    3.555e-3 3.558e-3 3.566e-3 3.580e-3 3.609e-3 3.665e-3 3.772e-3 3.977e-3
        -vs alone     3.551e-3     (the same tau without the convective floor)
        no velocity   3.993e-3     (the plain symmetric built-in)
        upwinded flux 1.467e-3     (HDGConvectionUpwindedIntegrator)

    A 128x sweep of $\beta_c$ moves the error by 12%, *monotonically the wrong
    way*, and interpolates between the two symmetric endpoints rather than
    approaching the upwinded one. At $\beta_c = 16$ the floor binds on every
    face with $|u\cdot n| > 0.02\,|c|$, so it is certainly firing.

    **The obstruction is symmetry, not size.** `Eval` returns one number per
    quadrature point, which the integrator applies to both sides of the face.
    The upwinded flux applies $\tau_\pm = \beta|u\cdot n| \pm
    \tfrac12\alpha(u\cdot n)$, which is *antisymmetric* in the sign of
    $u\cdot n$ -- that is what upwinding is. No scalar returned from this hook
    can express it, whatever it is scaled by.

    Worse, it can destroy an asymmetry that is already there. Constructed as
    `HDGDiffusionIntegrator(v, Q, a)` the built-in carries $\alpha = a$ and
    $\beta = a/2$, so the weight is $\beta \pm \alpha/2$ -- $a$ on the upwind
    side and **exactly zero on the downwind side**. A maximum lifts that
    deliberate zero and symmetrizes the face. Combining this class with
    `HDGConvectionUpwindedIntegrator` therefore diverges rather than
    reinforcing: the same case gives a relative flux error of 7.6e+04.

    So this is a floor for a $\tau$ that is *meant* to be symmetric -- the
    convective analogue of HDGFloorStabilization, useful where the diffusive
    scale collapses and some stabilization is wanted -- and it is **not** a
    route to an upwinded method. For that, put the upwinded integrator on the
    potential mass form.

    Constant, so the bilinear assembly path accepts it. */
class HDGConvectiveFloorStabilization : public HDGStabilization
{
   real_t beta_c, tau_min;

public:
   /** @param beta_c_ the multiple of $|u \cdot n|$ a face must carry; $1/2$
                      matches `HDGConvectionUpwindedIntegrator`'s default.
       @param tau_min_ an absolute floor applied after the maximum, as in
                       HDGFloorStabilization. Zero disables it. */
   HDGConvectiveFloorStabilization(real_t beta_c_ = 0.5, real_t tau_min_ = 0.)
      : beta_c(beta_c_), tau_min(tau_min_) { }

   real_t Eval(real_t s_diff, real_t un, real_t, real_t,
               ElementTransformation &) const override
   {
      const real_t s_conv = beta_c * std::abs(un);
      const real_t s = (s_diff > s_conv) ? s_diff : s_conv;
      return (s > tau_min) ? s : tau_min;
   }
};

/** @brief The HDGStabilization an HDGStabilizationRule describes: Eval() is
    the rule at the transformation's physical point, and GetRule() says so.

    Thread safe: the point is held on the stack. */
class HDGRuleStabilization : public HDGStabilization
{
   HDGStabilizationRule rule;

public:
   explicit HDGRuleStabilization(const HDGStabilizationRule &r) : rule(r) { }

   real_t Eval(real_t s_diff, real_t un, real_t u, real_t uhat,
               ElementTransformation &Tr) const override;

   bool GetRule(HDGStabilizationRule &r) const override { r = rule; return true; }
};

/** @brief Batched, device-capable assembly of HDGDiffusionIntegrator's
    INTERIOR-face matrices: every face's matrix in one mfem::forall instead of
    one host call per face.

    The first of the five device kernels this directory grew, and the one the
    other four were shaped after. The face term reduces to two scalars per
    quadrature point -- the
    stabilization weight on each side, which absorbs the coefficient, the
    HDGStabilization hook and the geometry -- and a set of outer products in
    the trace and element shapes. The outer products are O(NQ*SZ^2) against
    O(NQ*ND) for the shapes, so they are the part worth moving.

    @param tr_fes   the trace (constraint) space, a DG_Interface space.
    @param el_fes   the potential space.
    @param Q        the diffusion coefficient, or NULL for 1.
    @param beta     the integrator's beta.
    @param stab     an HDGStabilization hook, or NULL for the built-in.
    @param elmats   filled with one (SZ x SZ) matrix per INTERIOR face, in mesh
                    face order, SZ = 2*ND + TRD.

    @note Uniform order and one element geometry, which CanBatch() checks.
          Interior faces only; boundary faces stay on the per-face path.
    @note The per-quadrature-point weights and shapes are precomputed on the
          HOST, so this is element-assembly rather than partial assembly: it
          moves the outer products to the device and leaves the geometry where
          it was. Removing that precompute needs the ELEMENT Jacobian
          determinant at FACE quadrature points, which FaceGeometricFactors
          does not carry, and is the next piece. */
void HDGDiffusionFaceMatricesBatched(const FiniteElementSpace &tr_fes,
                                     const FiniteElementSpace &el_fes,
                                     Coefficient *Q, real_t beta,
                                     const HDGStabilization *stab,
                                     DenseTensor &elmats);

/** @brief The same face term, SCATTERED STRAIGHT INTO the hybridization's
    storage instead of into dense per-face matrices.

    This is the version that serves a full-device path. The dense form above
    has to be read back so host code can split it into E, G, H and D -- 193 MB
    and 183 ms against an 85 ms kernel at n=96 order 3 -- and that transfer is
    larger than the arithmetic it was meant to accelerate. Writing the blocks
    where they belong removes it: nothing comes back.

    @param face_list   the INTERIOR faces, in mesh face order.
    @param E_offsets   per-face offsets into @a E_data and @a G_data; the side
                       1 block is (ND x c_dof) at E_offsets[f] and side 2
                       follows it at + c_dof*ND.
    @param H_offsets   per-face offsets into @a H_data, (c_dof x c_dof).
    @param Df_offsets  per-ELEMENT offsets into @a Df_data, (ND x ND).

    @note E, G and H are written per face and race with nothing. D ACCUMULATES
          per element, and two faces of the same element collide, so it goes
          through AtomicAdd -- which is why this needs no colouring where the
          host loop does. **And why D is not bit-reproducible run to run on a
          device**: measured 3.5e-16 to 4.7e-16 relative between runs on an
          RTX 2070 SUPER, the only batched route that varies at all; see
          DarcyHybridization::SetAssemblyMode().
    @note @a Df_data is added to and must be zeroed by the caller;
          @a E_data, @a G_data and @a H_data are overwritten.
    @note The per-point weights of an HDGDiffusionIntegrator are computed on
          the device wherever HDGFaceWeightsOnDevice() admits it, and the
          element shapes are read from one table per (local face,
          orientation) slot, so on that route nothing runs per face on the
          host but a gather of the face and element point matrices. The
          convection forms read a velocity coefficient and stay on the host.

    @note MEASURED, and the atomics are why this is a DEVICE path and not a
          host one. Correct either way -- E, G and H come out bit-exact
          against the per-face integrator and D to 8.7e-17, which is the
          accumulation order. But on the host AtomicAdd is a real atomic where
          the per-face loop does a plain +=, and it costs: 2-D quads, n=64
          order 3, per-face 38.0 ms against 60.5 ms scattered. On the device
          the same call is 43.5 ms and, unlike the dense form, NOTHING COMES
          BACK. A host build wanting this shape should take the colouring
          DarcyHybridization already builds rather than the atomics. */
/** @brief Whether HDGFaceScatterBatched() can take @a integs on @a face_list.

    Every integrator has to resolve into one SCALAR integrator per equation,
    each of which the batched kernel implements -- an HDGDiffusionIntegrator or
    either HDGConvection*Integrator, in any combination. At vdim == 1 that is
    the integrator itself. At vdim > 1 it has to be a
    VectorBlockDiagonalIntegrator of exactly vdim blocks, in either of its
    forms: one integrator per equation, or one replicated. A null block is
    refused, because VectorBlockDiagonalIntegrator::AssembleMat() SHRINKS its
    element matrix rather than zeroing that block, so the per-face route is
    already reading a matrix a block short against offsets sized for vdim of
    them (and MFEM_ASSERT is compiled out in a release build) -- a caller
    wanting a zero block passes a zero-valued integrator.

    Every one of those refusals was PROBED rather than read off the code, and
    every one fires: a bare scalar integrator at vdim = 3, a wrapper of two
    blocks against vdim = 3, a wrapper with one block left null, a wrapper
    whose block is itself a SumIntegrator, and a list mixing a wrapper with a
    bare integrator. Both admitting shapes -- one integrator per equation, and
    one replicated -- are taken. This branch has shipped a gate that could
    never fire for any caller once already.

    Each resolved block has to want ONE integration rule across the whole face
    list, since the kernel samples every face at the same points. A
    non-uniform ElementTransformation::OrderW() defeats that, which is why it
    is asked of the faces rather than of the mesh. It is asked PER BLOCK
    because that is how the per-face route calls them:
    VectorBlockDiagonalIntegrator invokes each block's
    AssembleHDGFaceMatrix() in turn and each picks its own rule.

    False means the caller keeps its per-face loop, not that anything is
    wrong.

    **The vdim > 1 case was TWO refusals and the plan recorded one.**
    HDGFaceSpacesCanBatch()'s predecessor rejected `vdim != 1` outright, and
    the plan that asked for this filed that as the whole of the item.
    Measured:
    relaxing the vdim test alone leaves the kernel refused on every system,
    because at vdim > 1 the only shape any caller installs is a
    VectorBlockDiagonalIntegrator and the kind test reported that Unsupported.
    Both had to go together, and a probe on the three configurations said so
    before either was written.

    **And the caller the plan named is not reached by this.**
    miniapps/hdg/navierstokes.cpp puts a HyperbolicFormIntegrator on the same
    NonlinearForm as its HDGDiffusionIntegrator stabilization, so
    DarcyForm::EnableHybridization()'s FaceIntegratorsAreLinear() is false and
    the whole constraint goes to `c_nlfi_p`, where there is no batched kernel
    at all -- measured as `NumPotFaceConstraintIntegrators() == 0`, not
    inferred. What this reaches is the purely bilinear system constraint:
    tests/unit/fem/test_darcy_system.cpp, examples/hdg/ex17.cpp and
    examples/hdg/ex21.cpp (HDG linear elasticity, one repeated
    HDGDiffusionIntegrator at vdim = dim). Those three additionally need
    DarcyHybridization::EnableNPC(), since the kernel writes H into `H_data`;
    that is a separate item of the plan and is not lifted here. */
bool HDGFaceScatterCanBatch(const FiniteElementSpace &tr_fes,
                            const FiniteElementSpace &el_fes,
                            const Array<BilinearFormIntegrator*> &integs,
                            const Array<int> &face_list);

/** @brief The interior-face HDG constraint of every integrator in @a integs,
    for every face in @a face_list, scattered straight into E, G, H and D.

    The generalisation of HDGDiffusionFaceScatterBatched(), which covered a
    lone pure-diffusion term. What every supported integrator has in common is
    its SHAPE: at each quadrature point it contributes

        D1 += wd1 s1 s1^T,   E1 -= we1 s1 tr^T,   G1 += wg1 tr s1^T,
        D2 += wd2 s2 s2^T,   E2 -= we2 s2 tr^T,   G2 += wg2 tr s2^T,
        H  -= wh  tr tr^T,

    and they differ only in the seven weights. Diffusion is the degenerate
    case wd1 = we1 = wg1 and wd2 = we2 = wg2; the convection forms are not --
    the centred one has E and G on different weights, and the upwinded one
    crosses them, side 1's E carrying side 2's weight. In all three
    wh = wd1 + wd2, which is checked rather than assumed.

    Each integrator is applied in its own pass at its own rule, accumulating,
    after one pass that zeroes E, G and H. D is not zeroed here: it accumulates
    across the faces of an element and its zeroing belongs to the caller.

    @note VDIM > 1, and the whole of it is the scatter INDEXING -- the seven
          weights and the shapes are the scalar kernel's, untouched. The stored
          blocks carry the space's vector dimension, because
          DarcyHybridization's offsets are built from `GetDof() * GetVDim()`:
          `Df_offsets[el]` holds (ND*vdim)^2, `E_offsets[f]` holds
          (TRD*vdim)*(ND*vdim) per side, `H_offsets[f]` holds (TRD*vdim)^2.
          Within a block the layout is GROUP outermost (side 1, side 2, trace),
          then EQUATION, then dof -- equation k's element dofs occupy the
          contiguous run [k*ND, (k+1)*ND) -- because that is what
          VectorBlockDiagonalIntegrator::AssembleMat() produces and what
          ComputeAndAssemblePotFaceMatrix() CopyMN()s out of. Ordering does not
          enter: neither route consults it, so both agree on a byVDIM space as
          well as a byNODES one.

    @note ONE PASS PER (integrator, equation) PAIR, and that is a decision
          rather than a detail. Correctness first: each equation's integrator
          picks its own integration rule, exactly as it does on the per-face
          route, so a pass is the largest unit that can share a rule. And it
          keeps the kernel at one thread per FACE with plain inner loops --
          adding an equation component to a flat index decomposition is the
          shape that turned out to be 0.37 s of a 0.95 s loop in
          ComputeElementsHBatched().

    @note The zeroing pass clears the WHOLE block, every equation, including
          the off-diagonal ones no pass ever writes. That is what makes
          zero-then-accumulate reproduce the per-face route, which CopyMN()s
          the whole sub-block out of an element matrix whose off-diagonal
          equation entries are exactly zero.

    @note MEASURED. Entrywise against the per-face route's assembled NPC trace
          gradient over 60 (face term, order, vdim, wrapper form) combinations
          in 2-D: worst 1.78e-15 against a matrix norm of 4.14, i.e. 4.3e-16
          relative -- round-off from the association, not from the indexing,
          the same level the scalar case measures. Five deliberate mis-indexings
          are counted in
          tests/unit/fem/test_darcy_batched_face.cpp; the one worth repeating
          here is that interleaving the equation into the dof index
          (`i*vdim + e` for `e*ND + i`) fails 40 of the 60 and the 20 it does
          not fail are EXACTLY the order-0 ones, where one dof per equation
          makes the two indexings the same integer. An order-0 case cannot
          discriminate this kernel.

    @note AND IT IS SLOWER ON THE HOST, as the scalar version is and for the
          same reason: D goes through AtomicAdd where the per-face loop does a
          plain +=. DarcyForm::Assemble() at vdim = 3, interleaved A/B, three
          alternating pairs per size in one process, one thread -- (n = 32,
          order 2): 0.0570/0.0584, 0.0548/0.0584, 0.0582/0.0582 s serial
          against batched; (48, 2): 0.1251/0.1298, 0.1309/0.1325,
          0.1279/0.1311; (64, 1): 0.0699/0.0719, 0.0652/0.0671, 0.0717/0.0733.
          So 2-6% slower, and the sign is the same in all nine pairs even
          though the magnitude is inside the run-to-run spread. What the
          vdim > 1 path buys is that a system's face constraint is
          EXPRESSIBLE on a device at all. **That is the standing gate on
          every device step in this directory and it is worth restating
          wherever one is measured: nothing here pays until the whole chain
          is device-resident**, because a kernel whose neighbours are on the
          host pays more in transfer than it saves. Measured end to end by
          miniapps/hdg/hdgdevice, the crossover is between 1024 and 4096
          elements -- below it every device setting is a loss. */
void HDGFaceScatterBatched(const FiniteElementSpace &tr_fes,
                           const FiniteElementSpace &el_fes,
                           const Array<BilinearFormIntegrator*> &integs,
                           const Array<int> &face_list,
                           const Array<int> &E_offsets,
                           const Array<int> &H_offsets,
                           const Array<int> &Df_offsets,
                           Vector &E_data, Vector &G_data,
                           Vector &H_data, Vector &Df_data);

/** @brief Every element's Jacobians and determinants at the points of one
    rule, laid out as GeometricFactors lays them out, WITHOUT adding nodes to
    the mesh.

    Mesh::GetGeometricFactors() calls Mesh::EnsureNodes() on a mesh that has
    none. A batched kernel that took its geometry from it therefore changed
    the CALLER'S mesh partway through an assembly that the per-element route
    performs without touching it, and changed every later computation that
    asks whether the mesh has nodes. That later computation is where it shows,
    not in the geometry. Measured on meq's mirror mesh (1,175 triangles), every
    element's Jacobian, weight and mapped point, and every face's normal,
    weight and element points, are bit-identical with and without the nodes.
    Mesh::FindPoints() is not identical: its reference coordinates move by up
    to 1.8e-15. In meq's free-boundary mirror solve, adding the nodes and
    nothing else, ahead of the per-element divergence route, moved psi_h by
    9.06e-10 of its largest value. That run's batched-against-serial
    comparison had read 6.8e-13. Its fixed-boundary tokamak, with no
    extension, did not move.

    On a mesh with nodes this is Mesh::GetGeometricFactors(), cache and all.
    On a mesh without them, each element's vertices are mapped through its
    geometry's linear transformation element, which is what
    Mesh::GetElementTransformation() does there. The arithmetic is in the same
    order: J = PointMat * DShape summed over the vertices in order, and the
    determinant as DenseMatrix::Weight() takes it. One element geometry only. */
class HDGElementGeometry
{
public:
   /// @a flags as GeometricFactors takes them: JACOBIANS, DETERMINANTS or both.
   HDGElementGeometry(Mesh &mesh, const IntegrationRule &ir, int flags);

   /// NQ x SDIM x DIM x NE, as GeometricFactors::J.
   const Vector &J() const { return gf ? gf->J : J_; }
   /// NQ x NE, as GeometricFactors::detJ.
   const Vector &DetJ() const { return gf ? gf->detJ : detJ_; }

private:
   const GeometricFactors *gf = nullptr;
   Vector J_, detJ_;
};

/** @brief Whether HDGElementPointsAndWeights() can take @a mesh: one element
    geometry, not NURBS, and no nodes or nodes of one order. */
bool HDGElementPointsOnDevice(const Mesh &mesh);

/** @brief Every element's physical points and quadrature weights at the
    points of @a ir, computed on the device from the elements' point matrices:
    @a X is (SDIM, NQ, NE) and @a W is (NQ, NE), ip.weight times the
    Jacobian's weight.

    This is what a host loop over Mesh::GetElementTransformation() gets from
    Transform() and Weight(), summed in IsoparametricTransformation's order,
    so on the host the two agree bit for bit. No ElementTransformation is
    built, and no nodes are added to the mesh. The point matrices are
    gathered on the host, one copy per entry. Requires
    HDGElementPointsOnDevice(). */
void HDGElementPointsAndWeights(const Mesh &mesh, const IntegrationRule &ir,
                                Vector &X, Vector &W);

class HDGDiffusionIntegrator;

/** @brief Whether a coefficient can be evaluated on the device at any points
    of the elements of @a mesh, with no host loop and no ElementTransformation.

    True for a ConstantCoefficient, and for a GridFunctionCoefficient whose
    GridFunction lives on @a mesh itself, in a space of one order on a mesh of
    one element geometry, with VALUE-mapped scalar elements -- H1 and L2 at
    any vector dimension, its component taken as the coefficient asks. The
    function is gathered by the space's ElementRestriction, so it is READ ON
    THE DEVICE, and evaluated against one shape table per set of reference
    points.

    That last condition is what makes an axisymmetric radius cheap: on an
    affine mesh $R$ is linear on every element, so a GridFunctionCoefficient
    of the degree-1 coordinate field, component 1, is $R$ exactly at any
    point. Exactly in exact arithmetic, that is, and not bit for bit: a
    degree-1 H1_FECollection triangle is H1_TriangleElement, which goes
    through an inverse Vandermonde matrix where the mesh's own transformation
    uses Linear2DFiniteElement's 1 - x - y. meq saw the two differ at the ulp
    in single precision.

    This is not CoefficientVector, which would evaluate a
    GridFunctionCoefficient through QuadratureInterpolator on the device for
    elements, and it is worth saying why: GridFunctionCoefficient::Project()
    projects EVERY component of its GridFunction and ignores the one it was
    built for, so a coordinate component came back as the whole interleaved
    coordinate field. Faces would also have needed a FaceQuadratureSpace in
    its own face order. One shape table per (local face, orientation) slot
    serves faces here in the order the face kernels already hold. */
bool HDGCoefficientOnDevice(Coefficient *Q, const Mesh &mesh);

/** @brief Whether the batched face routes -- HDGFaceScatterBatched(),
    HDGBdrFaceScatterBatched() and HDGFaceSideBatched() -- weigh @a dif's face
    term on the DEVICE.

    The weight of a point is the integrator's StabValue(), and it needs the
    face normal, the element's Jacobian determinant at the point, and either
    the coefficient there (the built-in stabilization) or the HDGStabilization
    (a user's). All of it is taken on the device from point matrices: the
    normal from the face's own vertices (or node dofs), the determinant and
    the physical point from the element's at the face's reference points in
    that element, each summed in the order IsoparametricTransformation sums
    it. So nothing builds a transformation per face, and on the host the
    weights are the per-face loop's own bits. That needs:

    - no velocity and no matrix coefficient on @a dif;
    - either no HDGStabilization, and a coefficient HDGCoefficientOnDevice()
      takes (or none), or a state-independent one that describes itself
      through HDGStabilization::GetRule(). With a rule the coefficient is not
      read at all, since the rule does not read `s_diff`;
    - a mesh of dimension 2 or 3 equal to its space dimension, of one element
      geometry other than a prism or a pyramid (one face geometry), without
      nodes or with continuous nodes of one order -- the meshes on which
      Mesh::GetFaceTransformation() reads a face's own points.

    Otherwise the weights are taken on the host as before, one
    ElementTransformation per face. */
bool HDGFaceWeightsOnDevice(const HDGDiffusionIntegrator &dif,
                            const FiniteElementSpace &el_fes);

/** @brief Whether HDGElementMassBatched() weighs @a bfi's points on the
    device: no coefficient, or a scalar one HDGCoefficientOnDevice() takes,
    with the determinants from HDGElementGeometry. A vector or matrix
    coefficient, or any other scalar one, is evaluated on the host. */
bool HDGElementMassOnDevice(const BilinearFormIntegrator &bfi,
                            const FiniteElementSpace &fes);

/** @brief Whether HDGElementMassBatched() can take @a integs on @a fes.

    Every integrator has to be a MassIntegrator or a VectorMassIntegrator --
    the two the Darcy mass forms carry -- and they all have to want one
    integration rule across the mesh, since the kernel samples every element
    at the same points. A VectorMassIntegrator's vdim has to match the space's;
    it is a settable field and DarcyOperator does set it. */
bool HDGElementMassCanBatch(const FiniteElementSpace &fes,
                            const Array<BilinearFormIntegrator*> &integs);

/** @brief Every element's mass matrix, for every integrator in @a integs, as
    one device-resident Vector of NE blocks of (ndof*vdim)^2.

    Element assembly rather than partial assembly, which is what the caller
    wants: DarcyHybridization stores an element matrix per element and
    condenses it, so a PA operator would have to be un-applied to get back
    what it already has. The blocks come out in NATIVE dof order and
    field-outermost, which is VectorMassIntegrator::AssembleElementMatrix()'s
    own layout and Ordering::byNODES's -- so a caller's element vdofs index
    them directly, with no lexicographic reordering of the kind
    EABilinearFormExtension::GetElementMatrices() has to do.

    @a emat is overwritten, not accumulated.

    NOT the same route as MFEM's AssemblyLevel::ELEMENT, deliberately. That
    one has no notion of vdim -- EABilinearFormExtension sizes ea_data as
    ne*ndof*ndof with a scalar ndof -- and for a DG space it folds the form's
    FACE terms into the element matrices, which is exactly the work
    DarcyForm::Assemble() routes itself. Either would be wrong here. */
void HDGElementMassBatched(const FiniteElementSpace &fes,
                           const Array<BilinearFormIntegrator*> &integs,
                           Vector &emat);

/** @brief Whether HDGElementDivBatched() can take @a integs.

    Every integrator has to be a VectorDivergenceIntegrator -- the one the
    Darcy divergence form carries on a discontinuous flux -- and they all have
    to want one integration rule across the mesh. The flux space's vdim has to
    be the mesh's space dimension, which is what makes its element matrix
    (test dofs) x (sdim * trial dofs) and lets the hat-dof mask index it. */
bool HDGElementDivCanBatch(const FiniteElementSpace &trial_fes,
                           const FiniteElementSpace &test_fes,
                           const Array<BilinearFormIntegrator*> &integs);

/** @brief Every element's divergence block, as one device-resident Vector of
    NE blocks of (test dofs) x (sdim * trial dofs), column-major.

    The mixed counterpart of HDGElementMassBatched(), and it exists for a
    sharper reason than that one: MFEM has NO mixed element assembly at all.
    MixedBilinearForm::SetAssemblyLevel aborts on AssemblyLevel::ELEMENT
    ("stay tuned"), the two-space AssembleEA virtual is commented out in
    fem/bilininteg.hpp, and VectorDivergenceIntegrator has no EA. So there is
    nothing upstream to route through even in principle, on any element shape.

    Columns are field-outermost -- column `k*trial_dofs + a` is component k of
    trial dof a -- which is DenseMatrix::GradToDiv()'s own layout and
    Ordering::byNODES's, so the caller's element vdofs index it directly. */
void HDGElementDivBatched(const FiniteElementSpace &trial_fes,
                          const FiniteElementSpace &test_fes,
                          const Array<BilinearFormIntegrator*> &integs,
                          Vector &emat);

/** @brief Whether HDGMixedConductionResidualBatched() can take @a integ on
    @a fes.

    @a integ has to be a MixedConductionNLFIntegrator whose constitutive law
    is a LinearDiffusionFlux. That is the whole of the state-independence
    requirement and it is a guarantee rather than a guess: the law's
    ComputeDualFlux() takes its state argument UNNAMED, so the dual flux is a
    fixed linear map of the flux at each point, and ComputeDualFluxJacobian()
    hands that map over directly. A FunctionDiffusionFlux is refused -- its
    conductivity is a caller-supplied `std::function<real_t(const Vector &x,
    real_t u)>` evaluated at the CURRENT potential, so there is neither a
    device-callable form of it nor a per-point weight that can be computed
    ahead of the kernel. Lifting that needs a different caller contract, not a
    guard.

    One equation only. MixedConductionNLFIntegrator::AssembleElementGrad()
    asserts a (neq*dim)-square dual-flux Jacobian and LinearDiffusionFlux
    returns a dim-square one, so neq > 1 does not work through this law in the
    per-element route either.

    A scalar-range flux basis, so one shape value serves every spatial
    component and the reference shape table is the same for every element. An
    H(div) element needs CalcVShape(), which is per element, and lays its dofs
    out per equation rather than per (equation, direction).

    And one element geometry, one dof count and one integration rule across
    the mesh, which is what lets the kernel sample every element at the same
    points -- the same three conditions HDGElementMassCanBatch() asks.

    Those three are checked over EVERY element, so this is O(NE) with an
    ElementTransformation built per element, and the kernel's own
    MFEM_VERIFY runs it a second time. That is the house style here and the
    cost is inside the timing on
    DarcyHybridization::CanBatchLocalResidual() rather than hidden from it;
    but note it is paid per residual EVALUATION, not once per assembly the
    way HDGElementMassCanBatch()'s is. */
bool HDGMixedConductionResidualCanBatch(const FiniteElementSpace &fes,
                                        BlockNonlinearFormIntegrator *integ);

/** @brief The flux row of a MixedConductionNLFIntegrator's ELEMENT term,
    every element at once, matrix free.

    @a u_all is the local flux, element-blocked: element e at offset
    (ndof*vdim)*e, component c of dof i at c*ndof + i, which is
    FiniteElementSpace::GetElementVDofs()'s own Ordering::byNODES layout and
    therefore DarcyHybridization::el_u_dofs's. @a ru_all is overwritten with
    the same layout.

    Matrix free rather than assemble-then-apply, and that is the cheaper way
    round here as well as the one that needs no storage: the residual costs
    O(nq * ndof * dim) where forming the block first costs O(nq * ndof^2 *
    dim). The per-point conductivity is read out of the law once per point per
    element -- exactly as many coefficient evaluations as the per-element
    route makes -- and the apply is one mfem::forall over ELEMENTS with the
    dof and quadrature loops inside it, which is the shape that keeps the
    index arithmetic out of the kernel.

    Round-off against MixedConductionNLFIntegrator::AssembleElementVector(),
    and the association is deliberately its association: the accumulation is
    written `w * shape(i) * mF[c]` in that order, and the dual flux is formed
    by the same k-order contraction the law's own MultABt() uses, so on the
    cases measured it comes out BITWISE. That is a measurement, not a
    guarantee -- Vector::operator*() reduces through mfem::reduce and a
    device backend need not associate a dot product the way the host does. */
void HDGMixedConductionResidualBatched(const FiniteElementSpace &fes,
                                       BlockNonlinearFormIntegrator *integ,
                                       const Vector &u_all, Vector &ru_all);

/** @brief Whether HDGBdrFaceScatterBatched() can take @a integs.

    The same question HDGFaceScatterCanBatch() asks, on boundary faces --
    including the vdim > 1 resolution into one scalar integrator per equation.
    The face lists are per integrator because a boundary integrator carries an
    attribute marker and two of them need not apply to the same faces. */
bool HDGBdrFaceScatterCanBatch(const FiniteElementSpace &tr_fes,
                               const FiniteElementSpace &el_fes,
                               const Array<BilinearFormIntegrator*> &integs,
                               const std::vector<Array<int>> &face_lists);

/** @brief The BOUNDARY-face HDG constraint of every integrator in @a integs,
    scattered straight into E, G, H and D.

    One-sided throughout, and that is the whole difference from the interior
    form: a boundary face's E slot is (element dofs) x (trace dofs) rather
    than twice that, H takes one side's weight, and there is no side 2. The
    weight identity wh = wd1 + wd2 that holds on an interior face does NOT
    hold here for the upwinded form -- its H term keeps 2*beta*|u.n| at a
    boundary deliberately, "for stability reasons", where its D takes
    beta|u.n| + alpha(u.n)/2. That is why the weights are carried separately
    rather than derived from one another.

    @a face_lists gives, per integrator, the faces its attribute marker
    admits; @a all_faces is their union and is what gets zeroed first. E, G
    and H are assigned by the per-face route (CopyMN), so zero-then-accumulate
    reproduces it; D accumulates there and here.

    The weights are on the device as HDGFaceScatterBatched()'s are, under
    HDGFaceWeightsOnDevice(), with the shapes from one table per slot when
    the elements are VALUE-mapped. */
void HDGBdrFaceScatterBatched(const FiniteElementSpace &tr_fes,
                              const FiniteElementSpace &el_fes,
                              const Array<BilinearFormIntegrator*> &integs,
                              const std::vector<Array<int>> &face_lists,
                              const Array<int> &all_faces,
                              const Array<int> &E_offsets,
                              const Array<int> &H_offsets,
                              const Array<int> &Df_offsets,
                              Vector &E_data, Vector &G_data,
                              Vector &H_data, Vector &Df_data);

#ifdef MFEM_USE_MPI

/** @brief Whether HDGSharedFaceScatterBatched() can take @a integs on
    @a face_list, the local indices of this rank's shared faces.

    The same three questions HDGFaceScatterCanBatch() asks -- the spaces, a
    constant stabilization, one rule across the list -- with the rule taken
    from the ONE-SIDED accessor, which is what the routine being replaced
    calls. */
bool HDGSharedFaceScatterCanBatch(const FiniteElementSpace &tr_fes,
                                  const FiniteElementSpace &el_fes,
                                  ParMesh &pmesh,
                                  const Array<BilinearFormIntegrator*> &integs,
                                  const Array<int> &face_list);

/** @brief The HDG constraint on this rank's SHARED faces, scattered straight
    into E, G, H and D.

    The third of the three face kernels, and the one a refusal on
    DarcyHybridization::CanBatchPotFaceAssembly() predicted would need
    machinery neither of the other two has. It needs none: a shared face is
    one-sided from this rank, its E and G slots are sized for one element by
    AllocEG(), and the per-face route it replaces --
    ComputeAndAssemblePotFaceMatrix()'s shared branch, driven by
    ParDarcyForm::AssemblePotHDGSharedFaces() -- reaches it through the
    single-side AssembleHDGFaceMatrix(int side, ...). So the SCATTER is the
    boundary kernel's, byte for byte, and only the weights and the
    transformation source differ.

    **The weights are the interior form's, halved, and NOT the boundary
    form's**, which is the one place reading the boundary twin would have
    misled. A shared FaceElementTransformations carries Elem2No >= 0 -- the
    neighbour's shifted index -- so the one-sided routines take their interior
    branch: the convection forms' trace weight is (b - a) here where a genuine
    boundary takes b (centred) or 2b (upwinded). The other rank's own pass
    sees the opposite normal and contributes (b + a), and the global assembly
    sums the two into the interior kernel's whole-face weight.

    **What that sum makes observable, and what it hides, was measured rather
    than assumed, and the answer differs between the two forms.** Substituting
    the boundary expression here fails the UPWINDED arm of
    "The batched HDG shared-face kernel assembles the partition boundary" at
    every order and mesh, by 0.27 to 0.71 against a tolerance of 1e-11 -- 2b
    from each rank is 4b where the face wants 2b. It does NOT fail the CENTRED
    arm, and that is arithmetic rather than a gap in the case: b from each rank
    is 2b, which is exactly (b - a) + (b + a). For the centred form the SPLIT of
    a face's trace weight between the two ranks is invisible to anything that
    reads the assembled operator, and only the sum is real. The weight written
    here is still the per-face route's, because agreeing with the routine being
    replaced block by block is cheaper than depending on that cancellation --
    and the case pins it through a substitution that does change the sum
    (halving it fails 6 of 6 centred sections).

    @a face_list holds LOCAL face indices, so E_offsets, H_offsets and
    Df_offsets are indexed as they are everywhere else. */
void HDGSharedFaceScatterBatched(const FiniteElementSpace &tr_fes,
                                 const FiniteElementSpace &el_fes,
                                 ParMesh &pmesh,
                                 const Array<BilinearFormIntegrator*> &integs,
                                 const Array<int> &face_list,
                                 const Array<int> &E_offsets,
                                 const Array<int> &H_offsets,
                                 const Array<int> &Df_offsets,
                                 Vector &E_data, Vector &G_data,
                                 Vector &H_data, Vector &Df_data);

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
                                    Vector &H_data, Vector &Df_data);

/** @brief Whether the batched HDG face routines' SPACE requirements hold: a
    conforming mesh of one element geometry and one dof count, a DG_Interface
    trace collection, and one vector dimension shared by the two spaces.

    The vector dimension may be greater than one, and that is the only part
    worth explaining. A system's face constraint is block diagonal in the
    equation index -- equation k's trace multiplies equation k's field alone --
    so what the kernels need is that the trace space and the field space agree
    on how many equations there are. A trace space of a DIFFERENT vector
    dimension is a different operator shape rather than a harder case, and no
    caller in the tree has one.

    Ordering does not enter, and that is MEASURED rather than reasoned from
    the code. The blocks are laid out the way
    VectorBlockDiagonalIntegrator's element matrix is -- group outermost
    (side 1, side 2, trace), then equation, then dof -- because that is what
    DarcyHybridization::ComputeAndAssemblePotFaceMatrix() CopyMN()s out of,
    so neither route consults the space's Ordering. The check: the assembled
    NPC trace gradient on the same problem built byNODES and byVDIM comes back
    with the same nnz and BIT-IDENTICAL entries at vdim = 2 and 3, on 2-D
    quads and 3-D hexes, at orders 1 and 2 -- and the batched route agrees
    with the per-face one to 1.9e-16 to 6.7e-16 relative in every one. Nothing
    here is 2-D: triangles and hexes were checked alongside quads. */
bool HDGFaceSpacesCanBatch(const FiniteElementSpace &tr_fes,
                           const FiniteElementSpace &el_fes);

/** @brief Whether HDGDiffusionFaceMatricesBatched() can run on these spaces.

    HDGFaceSpacesCanBatch() and, additionally, vdim == 1. That extra condition
    is not a gap: this predicate gates the two routines that take a bare
    Coefficient and a beta rather than an integrator
    (HDGDiffusionFaceMatricesBatched(), HDGDiffusionFaceScatterBatched()), so
    there is no way to tell them what each equation's coefficient is. The
    vdim > 1 generalisation is on HDGFaceScatterCanBatch(), which takes the
    integrators. */
bool HDGDiffusionFaceMatricesCanBatch(const FiniteElementSpace &tr_fes,
                                      const FiniteElementSpace &el_fes);

/** @brief Whether HDGNormalTraceJumpBatched() can take @a bfi on these spaces.

    @a bfi has to be a NormalTraceJumpIntegrator itself (not a wrapper of one)
    on a discontinuous SCALAR-range flux space of vdim == dim, VALUE-mapped,
    against a scalar DG_Interface trace. The mesh has to be conforming, of one
    element geometry and one order, with nodes (if any) of one order -- which
    is what makes one integration rule and one shape table per (local face,
    orientation) slot serve every face. All of it is asked of the spaces and
    the mesh, not face by face, so it costs no transformation. */
bool HDGNormalTraceJumpCanBatch(const FiniteElementSpace &tr_fes,
                                const FiniteElementSpace &el_fes,
                                const BilinearFormIntegrator *bfi);

/** @brief NormalTraceJumpIntegrator's face block, ONE SIDE AT A TIME, for
    every (face, side) pair in (@a faces, @a sides), in one mfem::forall.

    Item k's block is the rows AssembleFaceMatrix() gives side @a sides[k] of
    face @a faces[k]: (ND*dim x TRD), column-major, row `i + d*ND` for
    component d of element dof i, side 2's rows already carrying the
    integrator's minus sign. So an interior face's two items, stacked, are the
    integrator's whole element matrix, and a boundary face (side 0 only) is
    what it returns with Elem2No < 0.

    The arithmetic is the integrator's own, in its order -- the trace shape
    scaled by `ip.weight * sign` first, then `shape * face_shape * normal` --
    so on the host the blocks agree with it to the last bit wherever the
    compiler does not contract to an FMA.

    The element shapes at the face points come from one table per (local
    face, orientation) slot, which needs VALUE-mapped elements. The normals
    are taken on the DEVICE from each face's own point matrix, in
    EvalJacobian()'s order, wherever the mesh's faces are its vertices or its
    continuous nodes (see HDGFaceWeightsOnDevice() for the conditions), so
    the bits are the face transformation's. Otherwise -- discontinuous nodes,
    a prism or pyramid mesh -- they are taken on the host, one face
    transformation per face. Not Mesh::GetFaceGeometricFactors(), which
    segfaults on a simplex mesh.

    One-sided rather than two-sided because both callers want sides
    separately: DarcyHybridization::ConstructC() stores the two sides of a
    face in separate compacted blocks, and the rich reconstruction's local
    problem takes one element's side of each of its faces. */
void HDGNormalTraceJumpBatched(const FiniteElementSpace &tr_fes,
                               const FiniteElementSpace &el_fes,
                               const BilinearFormIntegrator &bfi,
                               const Array<int> &faces,
                               const Array<int> &sides,
                               Vector &blocks);

/** @brief Whether HDGFaceSideBatched() can take @a integs.

    Every integrator has to be an HDGDiffusionIntegrator with a constant (or
    no) HDGStabilization, on scalar spaces (vdim == 1), and the spaces have to
    pass the same uniformity tests as HDGNormalTraceJumpCanBatch(). The
    convection forms are refused: their one-sided matrix is the base class's
    halving of the two-sided one, which is not what this kernel computes. */
bool HDGFaceSideCanBatch(const FiniteElementSpace &tr_fes,
                         const FiniteElementSpace &el_fes,
                         const Array<BilinearFormIntegrator*> &integs);

/** @brief The ONE-SIDED HDG face matrix of every integrator in @a integs,
    summed, for every (face, side) item, in one mfem::forall.

    Item k's block is what `AssembleHDGFaceMatrix(sides[k], tr_fe, el_fe,
    Trans, elmat)` returns for face @a faces[k] -- (ND + TRD) square,
    column-major, the element block first:

        [ D  E ]     D =  w s s^T,  E = -w s t^T,
        [ G  H ]     G =  w t s^T,  H = -w t t^T,

    with one weight w per quadrature point, which is the integrator's
    StabValue() at that side's own coefficient and that side's outward
    normal. D and H are accumulated over their lower triangle and mirrored,
    as the integrator does, so the products are associated the same way.

    The weights are evaluated on the DEVICE wherever HDGFaceWeightsOnDevice()
    admits the integrator, and otherwise on the host, through
    Coefficient::Eval() and the HDGStabilization hook, which are host virtuals
    taking an ElementTransformation. The outer products, which are the
    O(NQ (ND+TRD)^2) part, are the kernel's either way. */
void HDGFaceSideBatched(const FiniteElementSpace &tr_fes,
                        const FiniteElementSpace &el_fes,
                        const Array<BilinearFormIntegrator*> &integs,
                        const Array<int> &faces,
                        const Array<int> &sides,
                        Vector &blocks);

/** Integrator for the H/LDG diffusion stabilization term
    The LDG stabilization takes the form
    $$
        1/2 \beta \langle \{h^{-1} Q\} [v], [w] \rangle
    $$
    where $Q$ is a scalar or matrix diffusion coefficient and $v$, $w$ are the trial
    and test functions, respectively.

    The corresponding HDG stabilization is then
    $$\begin{align}
        \langle \tau_\pm v_\pm, w_\pm \rangle, & -\langle \tau_\pm \lambda,          w_\pm \rangle,\\
        \langle \tau_\pm v_\pm, \mu   \rangle, & -\langle (\tau_+ + \tau_-) \lambda, \mu   \rangle,
    \end{align}$$
    where $\tau_\pm = (\beta \pm 1/2 \alpha (u \cdot n) / |u \cdot n|) \{h^{-1} Q\}$
    and $\lambda$, $\mu$ are the trial and test trace functions, respectively. The vector
    coefficient $u$ is assumed continuous across the faces.

    ### Which $\tau$ recovers which rate, and what this one costs

    **The value built here is $O(1/h)$, and that is a trade rather than a
    defect.** $\tau = \beta (\hat n \cdot Q \hat n)/h$ grows without bound
    under refinement, which is the LDG-H regime that buys a superconvergent
    potential at the price of a suboptimal flux. Measured on
    `anisodiff -p 11` with no flow, quadrilaterals, equal-order $L^2$ flux,
    potential and `DG_Interface` trace all of degree $k$, over $n = 8$ to
    $256$ -- and taken to $256$ because everything below $64$ is still
    pre-asymptotic here:

        tau            aniso        k=1 flux  k=1 pot   k=2 flux  k=2 pot
        beta Q/h       kperp/kpar=1     1.01     ~2.9       1.96     ~3.6
        (this, O(1/h)) kperp/kpar=1e-2  1.00      2.86      2.02      3.98
        O(1)           kperp/kpar=1     2.02      2.03      2.99      2.99
        (beta ~ h)     kperp/kpar=1e-2  1.43      1.92      2.59      2.89

    Read the two blocks against $k+1$, the best a degree-$k$ space can do:

    * **$O(1/h)$ gives the flux $k$ and the potential $k+2$.** One order below
      optimal in the flux, one *above* it in the potential -- the potential is
      superconverging without any postprocessing. If the potential is the
      quantity of interest this is the better default, and it is the default.
    * **$O(1)$ gives $k+1$ in both, and that is the theoretical best.** It is
      reached exactly: 2.02 and 2.99 in the flux, 2.03 and 2.99 in the
      potential, flat over the last three refinements. Get it by scaling
      $\beta \propto h$ -- `anisodiff -td` divided by the cell count is what
      produced the rows above -- not by HDGFloorStabilization, which cannot
      (see below).
    * **Anisotropy costs a further half order in the flux and nothing in the
      potential**, and it is the one loss no $\tau$ scaling here repairs:
      $1.43$ and $2.59$ against $2.02$ and $2.99$, settled at $n = 256$. That
      is $k+\tfrac12$, and it is a separate phenomenon from the scaling above.

    **A floor cannot deliver the $O(1)$ row.** `HDGFloorStabilization` raises
    $\tau$ where the built-in value is small, but the built-in value grows like
    $1/h$, so any fixed floor is overtaken everywhere as the mesh refines. On
    the anisotropic case the floor binds only where
    $\hat n \cdot Q \hat n < \tau_{min} h/\beta$, a set that shrinks with
    $h$; measured, `-tf 1` against `-tf 0` differs by 27% at $n=8$ and is
    **identical to every printed digit at $n = 128$ and $n = 256$**, at both
    orders. It is a repair for faces where the coefficient collapses, not a
    route to a different scaling. */
class HDGDiffusionIntegrator : public BilinearFormIntegrator
{
protected:
   VectorCoefficient *v;
   Coefficient *Q;
   MatrixCoefficient *MQ;
   real_t alpha, beta;
   const HDGStabilization *stab{};

   /** @brief The weighted stabilization at one quadrature point.

       With no user object this is the built-in expression, unchanged and with
       no call. With one, the quadrature weight is divided out so that the
       object sees s itself, and put back afterwards. @a u and @a uhat are only
       meaningful where the state is available; the bilinear paths pass zero,
       which is why they insist the object be constant. */
public:
   /** @brief Whether the face term is the pure diffusion one: no velocity and
       a scalar (or absent) coefficient.

       The batched assembly covers that case and nothing else, so it asks
       before taking over. */
   bool IsPureDiffusion() const { return v == nullptr && MQ == nullptr; }

   /// The scalar coefficient, or null.
   Coefficient *GetCoefficient() const { return Q; }
   /// The matrix coefficient, or null.
   MatrixCoefficient *GetMatrixCoefficient() const { return MQ; }
   /// The velocity coefficient, or null.
   VectorCoefficient *GetVelocity() const { return v; }
   /// The alpha the constructor took.
   real_t GetAlpha() const { return alpha; }
   /// The beta the constructor took.
   real_t GetBeta() const { return beta; }

   /** @brief StabValue() made public, so a batched assembly weighs a point
       the way the per-face loop does instead of copying the expression.

       It was copied once already, into HDGFaceScatterBatched()'s ancestor,
       and a copy of a two-line expression is exactly the kind of thing that
       survives a change to the original. */
   real_t EvalStabilization(real_t wq, real_t ba, real_t un, real_t face_w,
                            real_t u, real_t uhat,
                            ElementTransformation &Tr) const
   { return StabValue(wq, ba, un, face_w, u, uhat, Tr); }

   /** @brief The rule AssembleHDGFaceMatrix() uses on this interior face.

       ONE SOURCE OF TRUTH, and the reason it exists: the batched face
       assembly has to integrate at exactly the rule the per-face loop does or
       the two build different operators, and a rule copied into the batch
       driver would diverge silently the day this one changed.
       AssembleHDGFaceMatrix() calls this, so it cannot. */
   const IntegrationRule &GetHDGFaceIntRule(
      const FiniteElement &trace_el, const FiniteElement &el1,
      const FiniteElement &el2, FaceElementTransformations &Trans) const;

   /** @brief The rule the ONE-SIDED routines integrate at.

       A separate overload because the one-sided routines take the max over
       ONE element and the trace, not over two elements -- so on a mesh whose
       neighbours differ in order the two rules are genuinely different and
       collapsing them would change the operator. Same one-source-of-truth
       argument as the two-sided one: HDGNLFaceGradScatterBatched() asks this
       rather than reconstructing the expression. */
   const IntegrationRule &GetHDGFaceIntRule(
      const FiniteElement &trace_el, const FiniteElement &el,
      FaceElementTransformations &Trans) const;


protected:
   inline real_t StabValue(real_t wq, real_t ba, real_t un, real_t face_w,
                           real_t u, real_t uhat,
                           ElementTransformation &Tr) const
   {
      if (!stab) { return wq * ba; }
      const real_t s_diff = (face_w != 0.) ? (wq * ba / face_w) : 0.;
      return face_w * stab->Eval(s_diff, un, u, uhat, Tr);
   }

   /** @brief Per-point scratch, and a MEMBER only in a build that cannot
       thread.

       This used to read "these are not thread-safe!", which was true and is
       now handled the way MFEM handles it everywhere else: members when
       MFEM_THREAD_SAFE is off, method-local declarations when it is on -- see
       FluxFunction::ComputeFluxDotN() for the same pattern. Each method sizes
       what it uses, so the local declarations are bare.

       **And this IS on the element-local hot path, which is not obvious.**
       A first version of this comment said the opposite -- that a
       BilinearFormIntegrator is evaluated when the forms are assembled and not
       once per residual. It is wrong twice over: BilinearFormIntegrator
       derives from NonlinearFormIntegrator, so one can be registered on the
       nonlinear potential form, and DarcyForm::Assemble() collects that form's
       interior face integrators into a SumNLFIntegrator and hands it to
       SetConstraintIntegrators() as c_nlfi_p. DarcyHybridization's
       ConstructGrad() and LocalResidual() then call AssembleHDGFaceGrad() and
       AssembleHDGFaceVector() on it once per element per evaluation. That is
       exactly how every nonlinear HDG diffusion problem in this tree is posed
       -- the unit tests' pedestal included.

       **And every method was SHADOWING these with bare locals**, so the
       members were dead and the allocation happened anyway -- the comment
       above described a treatment the code was not applying. The locals are
       now under `#ifdef MFEM_THREAD_SAFE`, which is the other half of the
       pattern and what the two HDGConvection*Integrator classes above already
       do. Measured with DHAT on `convdiff -p 6 -nl -dg -hb -npc` at 256
       elements: 5,526 malloc/free pairs and 4.3 MB, the largest remaining
       site once the hybridization's own temporaries were hoisted.

       **A doxygen block is not evidence that its own advice was taken.** */
#ifndef MFEM_THREAD_SAFE
   Vector tr_shape, shape1, shape2, vu, nor, nh, ni;
   Vector nor_Jt, nor_Ji, ni_Jt, ni_Ji;
   DenseMatrix mq;
#endif

public:
   /// Construct integrator with $\alpha = 0$ and $\beta = a$.
   HDGDiffusionIntegrator(const real_t a = 0.5)
      : v(NULL), Q(NULL), MQ(NULL), alpha(0.), beta(a) { }

   /// Construct integrator with $\alpha = 0$ and $\beta = a$.
   HDGDiffusionIntegrator(Coefficient &q, const real_t a = 0.5)
      : v(NULL), Q(&q), MQ(NULL), alpha(0.), beta(a) { }

   /// Construct integrator with $\alpha = 0$ and $\beta = a$.
   HDGDiffusionIntegrator(MatrixCoefficient &q, const real_t a = 0.5)
      : v(NULL), Q(NULL), MQ(&q), alpha(0.), beta(a) { }

   /// Construct integrator with $\alpha = a$ and $\beta = a/2$.
   HDGDiffusionIntegrator(VectorCoefficient &v_, const real_t a = 0.5)
      : v(&v_), Q(NULL), MQ(NULL), alpha(a), beta(0.5*a) { }

   /// Construct integrator with $\alpha = a$ and $\beta = a/2$.
   HDGDiffusionIntegrator(VectorCoefficient &v_, Coefficient &q,
                          const real_t a = 0.5)
      : v(&v_), Q(&q), MQ(NULL), alpha(a), beta(0.5*a) { }

   /// Construct integrator with $\alpha = a$ and $\beta = a/2$.
   HDGDiffusionIntegrator(VectorCoefficient &v_, MatrixCoefficient &q,
                          const real_t a = 0.5)
      : v(&v_), Q(NULL), MQ(&q), alpha(a), beta(0.5*a) { }

   /** @brief Replace the stabilization with a user supplied one.

       The object is referenced, not owned, and must outlive the integrator.
       Leaving it unset keeps the built-in constant stabilization and the
       assembly path that goes with it. */
   void SetStabilization(const HDGStabilization &s) { stab = &s; }

   const HDGStabilization *GetStabilization() const { return stab; }

   using BilinearFormIntegrator::AssembleFaceMatrix;
   void AssembleFaceMatrix(const FiniteElement &el1,
                           const FiniteElement &el2,
                           FaceElementTransformations &Trans,
                           DenseMatrix &elmat) override;

   void AssembleHDGFaceMatrix(const FiniteElement &trace_el,
                              const FiniteElement &el1,
                              const FiniteElement &el2,
                              FaceElementTransformations &Trans,
                              DenseMatrix &elmat) override;

   void AssembleHDGFaceMatrix(int side, const FiniteElement &trace_el,
                              const FiniteElement &el,
                              FaceElementTransformations &Trans,
                              DenseMatrix &elmat) override;

   void AssembleHDGFaceVector(int type,
                              const FiniteElement &trace_face_fe,
                              const FiniteElement &fe,
                              FaceElementTransformations &Tr,
                              const Vector &trfun, const Vector &elfun,
                              Vector &elvect) override;

   /** @brief The gradient of the face residual.

       Overridden rather than inherited because the base class builds it from
       AssembleHDGFaceMatrix(), which cannot see the state and so cannot carry
       the derivatives of a solution dependent stabilization. */
   void AssembleHDGFaceGrad(int type,
                            const FiniteElement &trace_face_fe,
                            const FiniteElement &fe,
                            FaceElementTransformations &Tr,
                            const Vector &trfun, const Vector &elfun,
                            DenseMatrix &elmat) override;

   real_t ComputeHDGFaceEnergy(int side,
                               const FiniteElement &trace_face_fe,
                               const FiniteElement &fe,
                               FaceElementTransformations &Tr,
                               const Vector &trfun, const Vector &elfun,
                               Vector *d_energy = NULL) override;
};

/** @brief The flux components a restricted integrator carries, or NULL when
    @a bfi is not one of them.

    ONE place that knows which classes below are "restricted", so that
    DarcyForm and DarcyHybridization ask the same question rather than each
    keeping its own list of types to try. */
const Array<int> *GetRestrictedFluxComponents(const BilinearFormIntegrator
                                              *bfi);

/** @brief Check a flux component list against a space dimension: non-empty or
    empty, strictly increasing, and every entry in [0, @a sdim).

    Strictly increasing rather than merely distinct, because the component
    order IS the layout of the flux space: component @a c of the space is the
    physical direction `comps[c]`, and two lists that are permutations of each
    other describe different spaces. Aborts with a message naming @a what. */
void CheckFluxComponents(const Array<int> &comps, int sdim, const char *what);

/** @brief The divergence of a flux that carries only SOME of the Cartesian
    directions,
    $$ \Big(Q \sum_{d \in S} \partial_d u_d,\ v\Big), $$
    for a scalar-range flux space of `vdim = |S|` whose component @a c is the
    physical direction `S[c]`.

    **What this is for, and why a small coefficient is not the answer.**
    DarcyForm's flux mass is a VectorMassIntegrator$(1/\kappa)$, so a direction
    in which the medium does not diffuse asks for an infinite coefficient. A
    kinetic equation whose collision operator diffuses in the velocity
    coordinates only is exactly that: along the field line the operator is pure
    advection, so the diffusion tensor carries a structural zero eigenvalue at
    every point and permanently, not in some limit. Floored, the answer is
    reported insensitive to $\kappa$ over eight decades and then breaks -- there
    is a window in which it is right and a cliff below it, and a floored
    stabilization moves the cliff and turns an astronomical failure into a
    plausible-looking one. The fix is a flux unknown that does not exist in that
    direction, which is what this and RestrictedNormalTraceJumpIntegrator build.

    **It is a SLICE of the full block, and that is arithmetic rather than a
    claim.** MFEM lays a scalar-range vector space out component-outermost, so
    column block @a d of VectorDivergenceIntegrator's element matrix is exactly
    $(Q\,\partial_d u_d, v)$. Measured against DerivativeIntegrator(1, d) forced
    onto the same rule: `max|slice - DerivativeIntegrator(d)| = 0` in both
    components on a 2-D quad. So this DELEGATES and slices rather than
    reproducing the loop -- the base's rule, its adjugate and its `dim != sdim`
    scaling cannot drift from a copy that does not exist, which is the
    one-source-of-truth argument the batched kernels in this file make about
    quadrature rules.

    The cost of delegating is the $|S|/\mathrm{sdim}$ of the block that is
    thrown away. B is assembled once, so that is a one-off; a native loop is
    available the day it is not.

    @note `|S| == sdim` is the identity and is supported deliberately, so that
          a test can run the wrapper against the stock integrator on the same
          problem and require them equal. Measured: bit-identical.
    @note A VECTOR-range (RT or broken-RT) flux is REFUSED. Its components are
          intrinsic to the element and there is no column block to drop; the
          whole idea only makes sense for a scalar-range space carrying one
          basis per direction. */
class RestrictedVectorDivergenceIntegrator : public BilinearFormIntegrator
{
   Array<int> comps;
   VectorDivergenceIntegrator base;
#ifndef MFEM_THREAD_SAFE
   DenseMatrix full;
#endif

public:
   RestrictedVectorDivergenceIntegrator(const Array<int> &comps_)
   { comps_.Copy(comps); }
   RestrictedVectorDivergenceIntegrator(const Array<int> &comps_, Coefficient &q)
      : base(q) { comps_.Copy(comps); }

   /// The physical directions the flux carries, in the space's own order.
   const Array<int> &GetFluxComponents() const { return comps; }

   /// The scalar coefficient handed to the base, or null.
   Coefficient *GetCoefficient() const { return base.GetCoefficient(); }

   /// Forwarded, so a caller-set rule reaches the integrator that uses it.
   void SetIntRule(const IntegrationRule *ir) override
   { BilinearFormIntegrator::SetIntRule(ir); base.SetIntRule(ir); }

   void AssembleElementMatrix2(const FiniteElement &trial_fe,
                               const FiniteElement &test_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat) override;
};

/** @brief The normal trace of a flux that carries only SOME of the Cartesian
    directions,
    $$ \Big\langle \lambda,\ \sum_{d \in S} u_d n_d \Big\rangle, $$
    the constraint matching RestrictedVectorDivergenceIntegrator.

    A ROW slice of NormalTraceJumpIntegrator, by the same argument and with the
    same delegation: its scalar-range branch accumulates
    `elmat(i + d*ndof1, j) += shape1(i) face_shape(j) normal(d)` and the
    element-2 block the same at an offset of `dim*ndof1`, so the rows belonging
    to direction @a d are exactly $\langle \lambda, u_d n_d \rangle$.

    **The two offsets are the whole content of the class**, and getting them
    wrong is silent: DarcyHybridization::AssembleCtFaceMatrix() places
    element 2 at `hat_size_1`, which is `ndof1*|S|`, while the base placed it
    at `ndof1*sdim`. Handed the unrestricted block at `|S| < sdim` the
    hybridization therefore reads element 2's rows out of element 1's SECOND
    COMPONENT -- measured on a problem whose exact answer is in the discrete
    space, it returns `err_p = 6.8e-01` against a correct `1.5e-15`, finite,
    with no abort anywhere. That is why the shape guards in
    AssembleDivMatrix() and AssembleCtFaceMatrix() exist regardless of this
    class.

    @note The same object serves the interior and the boundary faces:
          DarcyForm::Assemble() installs the flux constraint integrator it was
          given as the boundary constraint on the attributes carrying a face
          integrator on B. A boundary face has `ndof2 == 0` and the base sizes
          for it, so nothing here is conditional on which it is.
    @note A VECTOR-range flux is REFUSED, for the reason given on
          RestrictedVectorDivergenceIntegrator. */
class RestrictedNormalTraceJumpIntegrator : public BilinearFormIntegrator
{
   Array<int> comps;
   NormalTraceJumpIntegrator base;
#ifndef MFEM_THREAD_SAFE
   DenseMatrix full;
#endif

public:
   RestrictedNormalTraceJumpIntegrator(const Array<int> &comps_,
                                       real_t sign_ = 1.)
      : base(sign_) { comps_.Copy(comps); }

   /// The physical directions the flux carries, in the space's own order.
   const Array<int> &GetFluxComponents() const { return comps; }

   /// Forwarded, so a caller-set rule reaches the integrator that uses it.
   void SetIntRule(const IntegrationRule *ir) override
   { BilinearFormIntegrator::SetIntRule(ir); base.SetIntRule(ir); }

   using BilinearFormIntegrator::AssembleFaceMatrix;
   void AssembleFaceMatrix(const FiniteElement &trial_face_fe,
                           const FiniteElement &test_fe1,
                           const FiniteElement &test_fe2,
                           FaceElementTransformations &Trans,
                           DenseMatrix &elmat) override;
};

/** @brief The boundary term $\langle g,\ \sum_{d \in S} v_d n_d \rangle$ of the
    FLUX load, for a flux space carrying only the directions in @a S.

    The third member of the restricted family, and the one that decides whether
    a restricted flux can take its Dirichlet datum WEAKLY. The other two make
    the operator right on a non-essential trace; this makes the load right, and
    without it the two cannot be used together at all.

    **It is a row slice, by the same delegation argument the other two make.**
    VectorBoundaryFluxLFIntegrator's scalar-coefficient branch accumulates
    `elvect(dof*k + j) += w nor(k) shape(j)`, so the entries belonging to
    direction @a d are exactly $\langle g, v_d n_d \rangle$ and the slice is
    contiguous.

    **Why it has to exist, measured rather than argued.** That integrator sizes
    its element vector `dim*dof` from the ELEMENT dimension while the flux space
    owns `|S|*dof` vdofs, and LinearForm::Assemble() hands the two to
    Vector::AddElementVector(), which reads the first `|S|*dof` entries and
    returns. At `|S| < dim` the load therefore lands in the wrong direction --
    the k = 0 block, i.e. $n_x$ -- with no abort and no NaN. On a problem whose
    discrete answer is exact (p = xy + y^2 on the unit square, diffusion in y
    only, weak Dirichlet on every attribute, order 2 on 4x4 quads) it returns
    `err_p = 9.3e-01` where this class returns `2.1e-15`. That is the same
    `dim` versus `vdim` confusion RestrictedNormalTraceJumpIntegrator's
    doxygen records for the constraint, in a third place, and it is silent for
    the same reason.

    **And it is what makes a genuine outflow work.** With no flux component
    along the outflow normal there is nothing for the one-sided constraint row
    to impose -- the row is the upwinded convective one alone, which reads
    $\hat u = u_h$ -- so the outflow needs no datum, no constraint and no
    special treatment. Measured on the problem above with the outflow at
    x = 1 given, in turn, the Dirichlet datum, nothing at all, and a flux
    constraint with nothing at all: **2.1139e-15 in all three, bit-identical**.
    A flux that DOES carry the outflow normal has no such luck and needs the
    prescribed numerical flux on DarcyForm::GetTraceRHS(); see there.

    @note A VECTOR-valued datum is REFUSED. The base interleaves it as
          `(v*dim + k)*dof + j`, so a vector datum on a restricted space is a
          different expression rather than a slice, and nothing asks for one.
    @note A VECTOR-range (RT or broken-RT) flux is REFUSED, for the reason
          given on RestrictedVectorDivergenceIntegrator. */
class RestrictedVectorBoundaryFluxLFIntegrator : public LinearFormIntegrator
{
   Array<int> comps;
   VectorBoundaryFluxLFIntegrator base;
#ifndef MFEM_THREAD_SAFE
   Vector full;
#endif

public:
   RestrictedVectorBoundaryFluxLFIntegrator(const Array<int> &comps_,
                                            Coefficient &f, real_t s = 1.0)
      : base(f, s) { comps_.Copy(comps); }

   /// The physical directions the flux carries, in the space's own order.
   const Array<int> &GetFluxComponents() const { return comps; }

   /// Forwarded, so a caller-set rule reaches the integrator that uses it.
   void SetIntRule(const IntegrationRule *ir) override
   { LinearFormIntegrator::SetIntRule(ir); base.SetIntRule(ir); }

   /** @brief Refused: the datum belongs on the faces, where the normal is.

       LinearForm::Assemble() reaches this overload from
       AddBoundaryIntegrator() and the face one from AddBdrFaceIntegrator();
       a flux space here is L2, so it is the second that a caller wants and
       the first would silently assemble nothing useful. */
   void AssembleRHSElementVect(const FiniteElement &el,
                               ElementTransformation &Tr,
                               Vector &elvect) override;

   void AssembleRHSElementVect(const FiniteElement &el,
                               FaceElementTransformations &Tr,
                               Vector &elvect) override;

   using LinearFormIntegrator::AssembleRHSElementVect;
};

class LinearForm;

/** @brief Refuse a flux load that is not restriction-aware, naming the repair.

    Called by DarcyForm::Assemble() when the flux space carries fewer
    components than the mesh has dimensions. Precise rather than general on
    purpose: VectorBoundaryFluxLFIntegrator is the one class in the tree that
    is wrong in this configuration and is the one every caller reaches for,
    `miniapps/hdg/convdiff.cpp` included, so naming it is worth more than a
    predicate over everything a caller might write. */
void CheckRestrictedFluxLoad(LinearForm *b_u, int vdim, int sdim);

/** @brief Whether HDGNLFaceGradScatterBatched() can take these integrators.

    False means the caller keeps its per-element-face loop, not that anything
    is wrong. What it asks:

    * every integrator is one the kernel weighs -- an HDGDiffusionIntegrator
      with a CONSTANT stabilization, a MixedConductionNLFIntegrator, or a
      HyperbolicFormIntegrator;
    * one element geometry, one element dof count and one trace dof count, so
      the reference shape tables and the block strides are single numbers;
    * a conforming serial mesh and a DG_Interface trace space;
    * one integration rule across every (element, face) pair, asked of the
      integrator on each pair rather than derived from the mesh -- a
      non-uniform ElementTransformation::OrderW() or a neighbour of a
      different order defeats it;
    * @a neq == 1 wherever an HDGDiffusionIntegrator is present, since that
      integrator is scalar and a system wraps it in a
      VectorBlockDiagonalIntegrator, which the kernel does not look inside. */
/** The optional @a pmesh says the list is this rank's SHARED faces: one pair
    per face rather than two, side 0 only, and the transformations from the
    ParMesh. NULL -- the default -- is the interior list and the serial path,
    which is then provably the code it always was. */
bool HDGNLFaceGradCanBatch(const FiniteElementSpace &tr_fes,
                           const FiniteElementSpace &el_fes,
                           const FiniteElementSpace *fl_fes,
                           const Array<NonlinearFormIntegrator*> &integs,
                           const Array<BlockNonlinearFormIntegrator*> &bintegs,
                           const Array<int> &face_list,
                           ParMesh *pmesh = NULL);

/** @brief The STATE-CARRYING HDG face constraint GRADIENT, for every
    (element, face) pair at once, scattered straight into D, E, G and H.

    The counterpart of HDGFaceScatterBatched() for the terms that are not
    bilinear forms: c_nlfi_p / c_nlfi rather than c_bfi_p. Those are evaluated
    once per element per Newton step inside DarcyHybridization::ConstructGrad(),
    not once at assembly, so this is a per-step kernel and it is ONE SIDED --
    a pair is (element, face, side) and each interior face is visited from
    both of its elements, exactly as the element loop visits it.

    **Two weight matrices per point, not seven, and that is measured across
    the three families rather than assumed.** At each quadrature point every
    one of them contributes

        D += W_el(di,dj) s_el(i) s_el(j),   G += W_el(di,dj) s_tr(i) s_el(j),
        E += W_tr(di,dj) s_el(i) s_tr(j),   H += W_tr(di,dj) s_tr(i) s_tr(j),

    with W_el the derivative of the face residual with respect to the ELEMENT
    state and W_tr the derivative with respect to the TRACE state. D and G
    share a weight because they are the same residual row tested against the
    two spaces, and so do E and H. Checked against all three:
    HDGDiffusionIntegrator's one-sided matrix gives W_el = +w and W_tr = -w
    (its G comes from `elmat(el+j,i) = -elmat(i,el+j)`, so +w and not -w -- the
    sign that a reading of the lower-triangular fill gets wrong);
    MixedConductionNLFIntegrator the same with its own w; and
    HyperbolicFormIntegrator W_el = -weight*sign*J(2).n and
    W_tr = -weight*sign*J(1).n, which is where the neq x neq matrices come
    from.

    The state enters ONLY through those weights, and only on the host: the
    flux Jacobian, the numerical flux's AverageGrad and the element
    transformations carry no MFEM_HOST_DEVICE. So the precompute is a host
    loop writing flat weight and shape arrays, exactly as
    HDGFaceScatterBatched()'s is, and the device part is the scatter.

    Measured shares of one AssembleHDGGrad() call on the reachable case
    (`convdiff -p 6 -o k -dg -hb -nl -npc`, 128x128, HyperbolicFormIntegrator,
    three interleaved A/B pairs with the accumulation loops ablated):

    | order | integrator | of which accumulation | + DenseMatrix scaffolding |
    |---|---|---|---|
    | 2 | 1544 ns | 671 ns (43%) | 530 ns -> 58% of the call is batchable |
    | 3 | 2800 ns | 1432 ns (51%) | 955 ns -> 64% |
    | 5 | 8764 ns | 6289 ns (72%) | 2620 ns -> 78% |

    The remaining 42/36/22% is the host precompute, which stays on the host in
    either route. So this moves 58-78% of the per-pair gradient work off the
    host critical path and cannot remove more than that.

    The FLUX state is not an argument, and that is a property of the three
    integrators rather than an omission. A c_nlfi_p integrator is handed the
    potential alone; MixedConductionNLFIntegrator, the only
    BlockNonlinearFormIntegrator admitted, builds its own zero flux
    (`DenseMatrix u(1, dim); u = 0.;`) and never reads elfun[0]. An integrator
    whose face term depends on the flux would have to be added to the gate,
    not merely passed more data.

    @param el_state one pair's neq*ND potential dofs, field-outermost,
                    pair-major; the state the weights are evaluated at.
    @param tr_state the same for the pair's face trace dofs.
    @param D_off    per pair, the offset of its element's D block.
    @param E_off    per pair, the offset of its (face, side) E block.
    @param G_off    the same for G.
    @param H_off    per pair, the offset of its face's H block.

    E and G are zeroed here, over the pairs this call covers, and then every
    integrator accumulates -- which reproduces ConstructGrad()'s
    `eg_written` rule (clear on the first writer of a face and side) without
    the flag. D and H are NOT zeroed: D carries the element mass Jacobian and
    H may carry boundary faces, and both zeroings belong to the caller.

    @note D needs AtomicAdd -- the faces of one element collide on it. E, G
          and H do not: one thread per FACE handles both sides, which makes H
          exclusive as well. That is one atomic block per pair against the
          three the obvious pair-per-thread decomposition would need. */
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
   ParMesh *pmesh = NULL);

/** @brief The RESIDUAL counterpart of HDGNLFaceGradScatterBatched(): every
    interior face's nonlinear constraint contribution to the potential row and
    to the trace row, for the whole face list at once.

    **The same pass one tensor rank lower**, and it is worth reading the two
    together. The gradient carries two `neq x neq` weight MATRICES per point
    and contracts each against two shape tables; the residual carries two
    `neq`-VECTORS per point -- the numerical flux evaluated AT the state
    rather than differentiated -- and the contraction is rank one:

        r_el(i, k) += sum_q w_el(q, k) * s_el(q, i)     the ELEM|TRACE row
        r_tr(j, k) += sum_q w_tr(q, k) * s_tr(q, j)     the CONSTR|FACE row

    so this kernel is strictly simpler than the one that exists. It takes the
    same NP = 2*NF pair indexing, the same @a el_state and @a tr_state, and
    the same refusals -- HDGNLFaceGradCanBatch() answers for both, because the
    integrator families and the geometry conditions are identical.

    @a r_el and @a r_tr come back PER PAIR, (NP x LDD) and (NP x LDC), and the
    caller scatters them: the potential row per element is disjoint, the trace
    row has two contributors per face and wants the caller's own accumulation.
    Both are zeroed here. */
/** @brief Whether HDGNLFaceResidualBatched() can take these integrators.
    Everything HDGNLFaceGradCanBatch() asks, plus a refusal of
    MixedConductionNLFIntegrator, which the residual pass deliberately does
    not implement -- see there. */
bool HDGNLFaceResidualCanBatch(
   const FiniteElementSpace &tr_fes, const FiniteElementSpace &el_fes,
   const FiniteElementSpace *fl_fes,
   const Array<NonlinearFormIntegrator*> &integs,
   const Array<BlockNonlinearFormIntegrator*> &bintegs,
   const Array<int> &face_list, ParMesh *pmesh = NULL);

void HDGNLFaceResidualBatched(
   const FiniteElementSpace &tr_fes, const FiniteElementSpace &el_fes,
   const FiniteElementSpace *fl_fes,
   const Array<NonlinearFormIntegrator*> &integs,
   const Array<BlockNonlinearFormIntegrator*> &bintegs,
   const Array<int> &face_list,
   const Vector &el_state, const Vector &tr_state,
   Vector &r_el, Vector &r_tr, ParMesh *pmesh = NULL);

}

#endif //MFEM_BILININTEG_HDG
