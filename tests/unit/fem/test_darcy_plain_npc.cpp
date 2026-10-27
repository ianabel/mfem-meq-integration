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
#include "fem/darcy_plain_npc.hpp"

using namespace mfem;

/** The NO-DEVICE half of the pair. Its twin, over the same helper, is
    "The plain NPC path solves an exact polynomial under a device" in
    tests/unit/miniapps/test_debug_device.cpp, whose binary configures
    Device("debug"). The two differ in the Device and in nothing else, which
    is what makes a disagreement between them attributable.

    See fem/darcy_plain_npc.hpp for why the answer is exact and why the
    configuration is meq's. */
TEST_CASE("The plain NPC path solves an exact polynomial",
          "[DarcyHybridization][NPC][GPU]")
{
   const int order = GENERATE(1, 2);
   const int n = GENERATE(2, 4);
   // Triangles because that is what meq runs; quadrilaterals because that is
   // what every other Darcy device case in this tree runs.
   const Element::Type geom =
      GENERATE(Element::QUADRILATERAL, Element::TRIANGLE);
   // With mass = 0 the domain term on the potential-mass form is
   // arithmetically inert, and an inert term cannot distinguish a device that
   // evaluates it from one that drops it -- which is the fault this case
   // exists to catch. mass = 3 puts a non-zero source on the same integrator
   // with a load to match, so BOTH halves move err_p off round-off if either
   // is lost. See the header.
   const real_t mass = GENERATE(0.0, 3.0);
   CAPTURE(order, n, geom == Element::TRIANGLE, mass);

   const darcy_plain_npc::Result r = darcy_plain_npc::Solve(order, n, geom, mass);
   CAPTURE(r.err_u, r.err_p, r.r0_fields, r.r0_trace, r.step, r.datum,
           r.nonfinite, r.trace_size, r.load, r.batched_residual);


   // The route, pinned. This case earns its keep by running what meq runs,
   // and PotNL is what makes that true; see the header. A silent move onto
   // the batched linear residual would leave every assertion below passing
   // while the case stopped testing meq's configuration.
   REQUIRE_FALSE(r.batched_residual);
   // FIRST, because every norm below is meaningless otherwise: Norml2()
   // cannot see a NaN, and an all-NaN vector reports a norm of zero.
   REQUIRE(r.nonfinite == 0);

   // The problem is driven by the datum alone, so a zero datum would make
   // u = p = 0 the answer and every error below would pass on nothing.
   REQUIRE(r.datum > 0.1);

   // There was something to solve. This is the assertion that meq's symptom
   // trips: a Newton that takes zero steps has been told its residual is
   // already converged.
   REQUIRE(r.r0_fields > 1e-3);
   REQUIRE(r.r0_trace > 1e-3);
   REQUIRE(r.step > 1e-3);

   // And the answer, which is arithmetic rather than a second run.

   // The non-zero arm's source is live. Without this the arm could silently
   // degrade to the inert one and every assertion below would still pass.
   // 1e-3 and not something larger: the load is an ELEMENT integral, so it
   // scales with the element measure -- 5.6e-02 at order 2 on 4x4 triangles,
   // and smaller again on any finer mesh. The question it settles is
   // live-or-zero, not how big.
   if (mass != 0.0) { REQUIRE(r.load > 1e-3); }
   else { REQUIRE(r.load == 0.0); }
   REQUIRE(r.err_u < 1e-10);
   REQUIRE(r.err_p < 1e-10);
}

/** @brief The same problem driven through DarcyNPCOperator and
    DarcyNPCSolver under a NewtonSolver -- the layer meq runs.

    **This exists because the case above could not see a defect that was
    there.** It calls NPCResidual() directly, and meq's M-80 names
    DarcyNPCOperator::Mult as the `residual` leg their Newton evaluates. That
    routine writes its answer into three BLOCK ALIASES of y and told neither
    the BlockVector nor y, so under a Device the caller read a stale y. A
    NewtonSolver handed a zero residual reports convergence at iteration zero
    and leaves the fields at the initial iterate -- whose potential block is
    zero under NPC.

    So the two assertions that matter are r0_operator, taken through the
    operator before any solve, and its >= 1. Both are meq's M-79 symptom
    written down as a requirement. */
TEST_CASE("The NPC operator drives a Newton to the exact polynomial",
          "[DarcyHybridization][NPC][GPU]")
{
   const int order = GENERATE(1, 2);
   const int n = GENERATE(2, 4);
   const Element::Type geom =
      GENERATE(Element::QUADRILATERAL, Element::TRIANGLE);
   const real_t mass = GENERATE(0.0, 3.0);
   CAPTURE(order, n, geom == Element::TRIANGLE, mass);

   const darcy_plain_npc::Result r =
      darcy_plain_npc::Solve(order, n, geom, mass, true);
   CAPTURE(r.err_u, r.err_p, r.r0_operator, r.its, r.converged, r.datum,
           r.nonfinite, r.r_dev_valid);

   REQUIRE(r.nonfinite == 0);
   REQUIRE(r.datum > 0.1);

   // meq's symptom, both halves. A residual that reads zero through the
   // operator is what makes the Newton below take no steps at all.
   REQUIRE(r.r0_operator > 1e-3);
   // **`its` is the assertion that catches it, and r0_operator is NOT** --
   // which is worth knowing because the reverse is what you would guess.
   // Measured with the sync gated off: r0_operator stays > 1e-3 and `its`
   // comes back 0. The reason is that the measurement above ends in
   // R.HostRead(), and host-reading the residual IS the missing sync, so a
   // probe that reads it correctly cannot see the fault. Newton's own r is a
   // UseDevice(true) work vector that nothing host-reads, so Newton sees the
   // stale zero and reports convergence at iteration 0. A correct measurement
   // of a quantity can hide the defect in how everyone else obtains it.
   REQUIRE(r.its >= 1);
   REQUIRE(r.converged);

   REQUIRE(r.err_u < 1e-10);
   REQUIRE(r.err_p < 1e-10);
   // **Where the data IS, not what it says.** A check that ends in a host
   // read cannot distinguish flags that were repaired from data that was
   // migrated to the host, because SyncAlias() will bring device-valid blocks
   // down and unprotect them. This samples residency directly instead.
   //
   // What it pins is that the residual leaves the operator device-resident,
   // with no d2h round trip per Newton evaluation -- a property the device
   // offload plan needs and nothing else asserts. Measured: it does NOT
   // separate the one-hop sync from the two-hop one, which stay
   // device-resident alike.
   if (Device::Allows(Backend::DEVICE_MASK)) { REQUIRE(r.r_dev_valid); }
}

/** The nonlinear two-arm agreement of fem/darcy_plain_npc.hpp, with no
    Device: the operator and the legs are the same arithmetic here, so this
    is the control that the comparison can pass. Its [DebugDevice] twin is
    the one that can fail. */
TEST_CASE("The NPC operator and the NPC legs take the same Newton steps",
          "[DarcyForm][NonlinearDarcy][NPC]")
{
   const int order = GENERATE(1, 2);
   const Element::Type geom =
      GENERATE(Element::QUADRILATERAL, Element::TRIANGLE);
   CAPTURE(order, geom == Element::TRIANGLE);

   const darcy_plain_npc::NewtonAgreement r =
      darcy_plain_npc::NonlinearNewton(order, 4, geom);
   CAPTURE(r.its_operator, r.its_legs, r.first_step_operator,
           r.first_step_legs, r.diff, r.scale);

   REQUIRE(r.nonfinite == 0);
   REQUIRE(r.converged_legs);
   REQUIRE(r.converged_operator);
   // Nonlinear for real: more than the one step an affine problem takes.
   REQUIRE(r.its_legs >= 3);
   REQUIRE(r.its_operator == r.its_legs);
   REQUIRE(r.first_step_operator ==
           MFEM_Approx(r.first_step_legs, 1e-10, 1e-10));
   REQUIRE(r.diff <= 1e-10 * r.scale);
}


/** The caller-integrand gradient re-entered Newton-style, with no Device:
    the control for its [DebugDevice] twin, which is the case that faulted
    before ResidualRowsBatched() carried its view flags back. */
TEST_CASE("The caller-integrand NPC gradient re-entered after a residual",
          "[DarcyHybridization][NPC][Batched][GPU]")
{
   const int order = GENERATE(1, 2);
   const bool batched_factor = GENERATE(false, true);
   const int calls = GENERATE(1, 3);
   CAPTURE(order, batched_factor, calls);
   const darcy_plain_npc::CallerGradientOutcome r =
      darcy_plain_npc::CallerGradient(order, batched_factor, calls);
   CAPTURE(r.worst, r.scale);
   REQUIRE(r.took_route);
   REQUIRE(r.nonfinite == 0);
   REQUIRE(r.scale > 0.0);
   REQUIRE(r.worst < 1e-10 * r.scale);
}

/** meq's whole route, several Newton-shaped steps: with no Device, the control
    for its [DebugDevice] twin -- on a host build the copies are one and this
    can only fail on arithmetic, which makes it the arbiter when the twin fails
    -- and, tagged [GPU], the same check on real device memory in
    gpu_unit_tests, which is where the defect class it guards lives. */
TEST_CASE("The caller-integrand NPC route solves what the element loop solves",
          "[DarcyHybridization][NPC][Batched][GPU]")
{
   const int order = GENERATE(1, 2);
   CAPTURE(order);
   const darcy_plain_npc::CallerRouteOutcome r =
      darcy_plain_npc::CallerRoute(order, 3, 4);
   CAPTURE(r.worst_r, r.scale_r, r.worst_y, r.scale_y);
   REQUIRE(r.took_residual);
   REQUIRE(r.took_gradient);
   REQUIRE(r.took_traversal);
   REQUIRE(r.nonfinite == 0);
   REQUIRE(r.scale_r > 0.0);
   REQUIRE(r.scale_y > 0.0);
   REQUIRE(r.worst_r < 1e-10 * r.scale_r);
   REQUIRE(r.worst_y < 1e-10 * r.scale_y);
}

/** AddNextGradientCorrection(), asked for by meq for the border fold of their
    bordered Newton. Their fold was a row added inside their integrator's
    AssembleElementGrad(), so the arbiter is exactly that integrator. The
    queued correction must solve what it solves on the element loop and on
    the caller-integrand batched route, through the whole of
    DarcyNPCSolver::ArrayMult(). Then the step with nothing queued must
    solve what a hybridization that never queued anything solves: to the
    bit on the element loop, because nothing in that step differs. */
TEST_CASE("A queued gradient correction solves what the integrator's own fold "
          "solves", "[DarcyHybridization][NPC][Batched][GPU]")
{
   const int order = GENERATE(1, 2);
   const bool batched_factor = GENERATE(false, true);
   CAPTURE(order, batched_factor);
   const darcy_plain_npc::CorrectedGradientOutcome r =
      darcy_plain_npc::CorrectedGradient(order, batched_factor, 3, 3);
   CAPTURE(r.worst_loop, r.worst_bat, r.scale, r.moved, r.last_loop,
           r.last_bat, r.last_scale);
   REQUIRE(r.took_gradient);
   if (batched_factor) { REQUIRE(r.took_traversal); }
   REQUIRE(r.left_queued == 0);
   REQUIRE(r.nonfinite == 0);
   REQUIRE(r.scale > 0.0);
   REQUIRE(r.worst_loop < 1e-10 * r.scale);
   REQUIRE(r.worst_bat < 1e-10 * r.scale);
   // The discriminator: without the correction the answer is far away, so
   // a correction that never reached the factors cannot pass the two above.
   REQUIRE(r.moved > 1e-4 * r.scale);
   REQUIRE(r.last_scale > 0.0);
   REQUIRE(r.last_loop == 0.0);
   REQUIRE(r.last_bat < 1e-10 * r.last_scale);
}
