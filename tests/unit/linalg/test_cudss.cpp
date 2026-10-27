// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "unit_tests.hpp"
#include "mfem.hpp"

#if defined(MFEM_USE_CUDSS) && defined(MFEM_USE_SUITESPARSE)

#include <cstring>

using namespace mfem;

namespace cudss_config
{

// cuDSS reaches the device directly -- CuMemAlloc, CuMemcpyDtoD, and a CSR it
// takes from SparseMatrix::ReadI()/ReadData(), which hand back HOST pointers
// when no Device is configured. The serial unit_tests main configures none and
// filters on {"~[Parallel]"} only, so a [GPU]-tagged case runs there too and
// has to decline rather than assume. gpu_unit_tests is where these do their
// work.
bool NoDevice()
{
   if (Device::Allows(Backend::CUDA_MASK)) { return false; }
   WARN("CuDSSSolver needs a CUDA device; this case only runs under "
        "gpu_unit_tests.");
   return true;
}

// A five-point operator on an n x n grid. `sym` gives it a symmetric,
// diagonally dominant (hence positive definite) stencil; otherwise the two
// horizontal weights differ and it is genuinely nonsymmetric. `lower_only`
// stores nothing above the diagonal, which is what a caller passing
// CuDSSSolver::LOWER would hand it.
SparseMatrix *FivePoint(int n, bool sym, bool lower_only = false)
{
   const int N = n * n;
   SparseMatrix *A = new SparseMatrix(N, N);
   const real_t wl = sym ? -1.0 : -0.7, wr = sym ? -1.0 : -1.3;
   for (int j = 0; j < n; j++)
   {
      for (int i = 0; i < n; i++)
      {
         const int r = j * n + i;
         if (j > 0) { A->Add(r, r - n, -1.0); }
         if (i > 0) { A->Add(r, r - 1, wl); }
         A->Add(r, r, 4.0 + 0.01 * r);
         if (!lower_only)
         {
            if (i < n - 1) { A->Add(r, r + 1, wr); }
            if (j < n - 1) { A->Add(r, r + n, -1.0); }
         }
      }
   }
   A->Finalize();
   return A;
}

void FillRHS(Vector &b, int col)
{
   real_t *h = b.HostWrite();
   for (int i = 0; i < b.Size(); i++)
   { h[i] = std::sin(0.3 * i + 1.7 * col) + 0.5 * col + 1.0; }
}

real_t MaxDiff(const Vector &a, const Vector &b)
{
   const real_t *x = a.HostRead(), *y = b.HostRead();
   real_t m = 0.0;
   for (int i = 0; i < a.Size(); i++) { m = std::max(m, std::abs(x[i] - y[i])); }
   return m;
}

// Bitwise, because a tolerance cannot tell reproducible from nearly so --
// which is the whole content of deterministic mode.
bool BitIdentical(const Vector &a, const Vector &b)
{
   const real_t *x = a.HostRead(), *y = b.HostRead();
   return std::memcmp(x, y, a.Size() * sizeof(real_t)) == 0;
}

} // namespace cudss_config

using namespace cudss_config;

TEST_CASE("CuDSSSolver takes its matrix view and its symmetry in either order",
          "[CuDSS][GPU]")
{
   if (NoDevice()) { return; }

   // SetMatrixViewType() used to READ mat_type and silently reduce a LOWER
   // request to FULL whenever SetMatrixSymType() had not been called yet. The
   // matrix here holds only its lower triangle, so a dropped view is not a
   // missed optimisation -- it is a different matrix, and the answer is wrong
   // with nothing said. Both orders must now agree with UMFPACK on the full
   // matrix, which is the same operator.
   const int n = 20;
   SparseMatrix *L = FivePoint(n, true, true);
   SparseMatrix *F = FivePoint(n, true, false);
   const int N = L->Height();

   Vector b(N);
   b.UseDevice(true);
   FillRHS(b, 0);

   Vector truth(N);
   truth = 0.0;
   {
      UMFPackSolver umf;
      umf.SetOperator(*F);
      umf.Mult(b, truth);
   }
   REQUIRE(truth.Norml2() > 1.0);            // the arbiter is not a zero vector

   for (int sym_first = 0; sym_first < 2; sym_first++)
   {
      CAPTURE(sym_first);
      Vector y(N);
      y.UseDevice(true);
      y = 0.0;

      CuDSSSolver cudss;
      if (sym_first)
      {
         cudss.SetMatrixSymType(CuDSSSolver::SYMMETRIC_POSITIVE_DEFINITE);
         cudss.SetMatrixViewType(CuDSSSolver::LOWER);
      }
      else
      {
         cudss.SetMatrixViewType(CuDSSSolver::LOWER);
         cudss.SetMatrixSymType(CuDSSSolver::SYMMETRIC_POSITIVE_DEFINITE);
      }
      cudss.SetOperator(*L);
      cudss.Mult(b, y);

      REQUIRE(MaxDiff(y, truth) == MFEM_Approx(0.0, 1e-10, 1e-10));
   }

   delete F;
   delete L;
}

TEST_CASE("CuDSSSolver reaches cuDSS's iterative refinement", "[CuDSS][GPU]")
{
   if (NoDevice()) { return; }

   // The wrapper never called cudssConfigSet(), so CUDSS_CONFIG_IR_N_STEPS
   // stayed at cuDSS's default of 0 -- no refinement of any kind -- and no
   // caller could change it. This asks only that the setting is accepted and
   // that the answer stays right; how much refinement buys is a property of
   // the matrix and is not something to assert.
   const int n = 20;
   SparseMatrix *A = FivePoint(n, false);
   const int N = A->Height();

   Vector b(N);
   b.UseDevice(true);
   FillRHS(b, 0);

   Vector truth(N);
   truth = 0.0;
   {
      UMFPackSolver umf;
      umf.SetOperator(*A);
      umf.Mult(b, truth);
   }

   for (int steps : {0, 1, 3})
   {
      CAPTURE(steps);
      Vector y(N);
      y.UseDevice(true);
      y = 0.0;

      CuDSSSolver cudss;
      cudss.SetRefinementSteps(steps);
      cudss.SetOperator(*A);
      cudss.Mult(b, y);

      REQUIRE(MaxDiff(y, truth) == MFEM_Approx(0.0, 1e-10, 1e-10));
   }

   delete A;
}

TEST_CASE("CuDSSSolver can be asked for a reproducible solve", "[CuDSS][GPU]")
{
   if (NoDevice()) { return; }

   // cuDSS's default is not reproducible: the same matrix and right-hand side,
   // solved from a fresh solver, give answers that differ in the last digit or
   // two. That is reassociation rather than a fault, but a caller asserting
   // bit-exactness between two routes has, until now, had no way to opt out.
   //
   // The second section is what keeps the first honest: if the default ever
   // becomes reproducible on some machine, it says so rather than letting the
   // deterministic arm pass for free.
   const int n = 24;
   SparseMatrix *A = FivePoint(n, false);
   const int N = A->Height();

   Vector b(N);
   b.UseDevice(true);
   FillRHS(b, 0);

   auto solve = [&](bool deterministic, Vector &y)
   {
      y.SetSize(N);
      y.UseDevice(true);
      y = 0.0;
      CuDSSSolver cudss;
      cudss.SetDeterministic(deterministic);
      cudss.SetOperator(*A);
      cudss.Mult(b, y);
   };

   SECTION("deterministic mode is bit-reproducible over repeated solves")
   {
      Vector first, again;
      solve(true, first);
      REQUIRE(first.Norml2() > 1.0);
      for (int rep = 0; rep < 4; rep++)
      {
         CAPTURE(rep);
         solve(true, again);
         REQUIRE(BitIdentical(first, again));
      }
   }

   SECTION("and it is not free: the default here is not reproducible")
   {
      // Reported rather than required. cuDSS makes no promise either way with
      // the switch off, and a machine on which the default happens to be
      // stable is not a failure -- but it would make the section above
      // vacuous, and this is where that would show up.
      Vector first, again;
      solve(false, first);
      int differing = 0;
      for (int rep = 0; rep < 4; rep++)
      {
         solve(false, again);
         if (!BitIdentical(first, again)) { differing++; }
      }
      INFO("repeats differing from the first, with deterministic mode OFF: "
           << differing << " of 4");
      REQUIRE(first.Norml2() > 1.0);
   }

   delete A;
}

TEST_CASE("CuDSSSolver refuses the deterministic case cuDSS gets wrong",
          "[CuDSS][GPU]")
{
   if (NoDevice()) { return; }

   // MEASURED on cuDSS 0.8.0: deterministic mode with CUDSS_MTYPE_SYMMETRIC
   // reports success from every phase and writes an exactly zero solution.
   // SPD is fine, so the refusal is on the one type that is silently wrong and
   // not on deterministic mode generally. If a later cuDSS fixes this, THIS is
   // the case that says so -- it starts failing, and the refusal can go.
   const int n = 16;
   SparseMatrix *A = FivePoint(n, true);
   const int N = A->Height();

   Vector b(N), y(N);
   b.UseDevice(true);
   y.UseDevice(true);
   FillRHS(b, 0);
   y = 0.0;

#ifdef MFEM_USE_EXCEPTIONS
   // Without exceptions MFEM_VERIFY aborts, and an aborting REQUIRE_THROWS
   // takes the whole run with it -- the convention test_operator.cpp follows.
   {
      CuDSSSolver cudss;
      cudss.SetDeterministic(true);
      cudss.SetMatrixSymType(CuDSSSolver::SYMMETRIC_INDEFINITE);
      REQUIRE_THROWS(cudss.SetOperator(*A));
   }
#endif

   // The same matrix, the same switch, declared positive definite: allowed,
   // and right. This is the control that convicts the matrix TYPE rather than
   // the switch.
   Vector truth(N);
   truth = 0.0;
   {
      UMFPackSolver umf;
      umf.SetOperator(*A);
      umf.Mult(b, truth);
   }

   CuDSSSolver spd;
   spd.SetDeterministic(true);
   spd.SetMatrixSymType(CuDSSSolver::SYMMETRIC_POSITIVE_DEFINITE);
   spd.SetOperator(*A);
   spd.Mult(b, y);
   REQUIRE(MaxDiff(y, truth) == MFEM_Approx(0.0, 1e-10, 1e-10));

   delete A;
}

TEST_CASE("CuDSSSolver leaves a device-resident pattern on the device",
          "[CuDSS][GPU]")
{
   if (NoDevice()) { return; }

   // SetMatrix() took the entry count from NumNonZeroElems(), which reads
   // I[height] through HostReadI(). For a matrix whose row pointers are valid
   // only on the device -- an HDG gradient's pattern is built there -- that
   // downloaded the whole of I on every SetOperator(), only for cuDSS to read
   // it on the device a line later. Found by MEQ under nsys: 12 downloads of
   // I per run. The count is now read alone, and I must stay device-only.
   //
   // Twice, with the reordering reused, so the second call also takes the
   // pattern check that compares the count against the first.
   const int n = 20;
   SparseMatrix *A = FivePoint(n, false);
   const int N = A->Height();

   Vector b(N);
   b.UseDevice(true);
   FillRHS(b, 0);

   Vector truth(N);
   truth = 0.0;
   {
      UMFPackSolver umf;
      umf.SetOperator(*A);
      umf.Mult(b, truth);
   }

   A->ReadWriteI();                  // the row pointers now live on the device
   REQUIRE_FALSE(A->GetMemoryI().HostIsValid());

   CuDSSSolver cudss;
   cudss.SetReorderingReuse(true);
   for (int call = 0; call < 2; call++)
   {
      CAPTURE(call);
      cudss.SetOperator(*A);
      REQUIRE_FALSE(A->GetMemoryI().HostIsValid());

      Vector y(N);
      y.UseDevice(true);
      y = 0.0;
      cudss.Mult(b, y);
      REQUIRE(MaxDiff(y, truth) == MFEM_Approx(0.0, 1e-10, 1e-10));
   }

   delete A;
}

#endif // MFEM_USE_CUDSS && MFEM_USE_SUITESPARSE
