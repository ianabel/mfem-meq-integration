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

#ifndef MFEM_PARDISO
#define MFEM_PARDISO

#include "../config/config.hpp"

#ifdef MFEM_USE_MKL_PARDISO

#include "mkl_pardiso.h"
#include "operator.hpp"
#include "sparsitypattern.hpp"

namespace mfem
{
/**
 * @brief MKL Parallel Direct Sparse Solver PARDISO
 *
 * Interface to MKL PARDISO: the direct sparse solver based on PARDISO
 */
class PardisoSolver : public mfem::Solver
{
public:
   enum MatType
   {
      REAL_STRUCTURE_SYMMETRIC = 1,
      REAL_SYMMETRIC_POSITIVE_DEFINITE = 2,
      REAL_SYMMETRIC_INDEFINITE = -2,
      REAL_NONSYMMETRIC = 11
   };

   /**
    * @brief Construct a new PardisoSolver object
    *
    */
   PardisoSolver();

   /**
    * @brief Set the Operator object and perform factorization
    *
    * @a op needs to be of type SparseMatrix.
    *
    * @param op Operator to use in factorization and solve
    */
   void SetOperator(const Operator &op) override;

   /**
    * @brief Solve
    *
    * @param b RHS vector
    * @param x Solution vector
    */
   void Mult(const Vector &b, Vector &x) const override;

   /**
    * @brief Solve for several right-hand sides in one pass
    *
    * The factors are walked once rather than once per column: PARDISO's
    * solve phase is called a single time with @a nrhs set to `X.Size()`,
    * which is one triangular sweep over the factors instead of `X.Size()`
    * of them. The arithmetic performed on any one column is unchanged, so
    * the saving is in memory traffic and call overhead, not in flops.
    *
    * `X.Size() == 1` forwards to Mult() and allocates nothing, following
    * STRUMPACKSolver::ArrayMult().
    *
    * @param X RHS vectors, all of length Width()
    * @param Y Solution vectors, all of length Height()
    */
   void ArrayMult(const Array<const Vector *> &X,
                  Array<Vector *> &Y) const override;

   /**
    * @brief Set the print level for MKL Pardiso
    *
    * Prints statistics after the factorization and after each solve.
    *
    * @param print_lvl Print level
    */
   void SetPrintLevel(int print_lvl);

   /**
    * @brief Set PARDISO's iterative refinement control: `iparm[7]` in MKL's
    * 0-based numbering. The constructor sets 2.
    *
    * The solve phase reads it, so it takes effect at the next Mult() without
    * refactorizing. As MKL documents it:
    * - a positive value is a cap: up to that many refinement steps on every
    *   solve, stopping early once the backward error is small enough;
    * - 0 refines, two steps, only after a numerical factorization that
    *   perturbed a pivot (see GetNumPerturbedPivots());
    * - a negative value is the cap with the residual accumulated in extended
    *   precision.
    *
    * Each step is a product with the matrix and another solve, so with
    * several right-hand sides it is paid once per column. With the default
    * of 2 it is paid even when no pivot was perturbed: one caller, factoring
    * an HDG trace system once and solving it for twenty right-hand sides,
    * saw two steps on every solve and none perturbed. MKL's level-1 report
    * put the additional calculations at 4.5 to 6 times the direct solve.
    * Whether a smaller cap keeps the accuracy a caller needs is the caller's
    * to check; GetNumRefinementSteps() says what was done.
    *
    * @param steps Value for `iparm[7]`
    */
   void SetMaxRefinementSteps(int steps) { iparm[7] = steps; }

   /**
    * @brief The number of iterative refinement steps the last solve phase
    * performed (`iparm[6]`, an MKL output). 0 before the first solve.
    */
   int GetNumRefinementSteps() const { return iparm[6]; }

   /**
    * @brief The number of pivots the last numerical factorization perturbed
    * (`iparm[13]`, an MKL output). 0 before the first factorization.
    *
    * Pivots below `10^-iparm[9]` relative are perturbed rather than taken.
    * The constructor sets `iparm[9]` to 13.
    */
   int GetNumPerturbedPivots() const { return iparm[13]; }

   /**
    * @brief Set the matrix type
    *
    * The matrix type supported is either real and symmetric or real and
    * non-symmetric.
    *
    * @param mat_type Matrix type
    */
   void SetMatrixType(MatType mat_type);

   /**
    * @brief Retain the analysis phase across SetOperator() calls and reuse it
    * whenever the sparsity pattern is unchanged. Off by default.
    *
    * PARDISO's phase 11, the reordering and symbolic factorization, depends
    * only on the sparsity pattern. A caller that refactorizes a matrix whose
    * pattern does not change -- a Newton iteration, an implicit time step, a
    * continuation loop, a parameter sweep -- otherwise repeats it on every
    * SetOperator() call and uses it once. With reuse, such a call is phase 22
    * alone.
    *
    * The pattern is checked, not assumed, by the exact comparison described in
    * UMFPackSolver::SetReuseSymbolic(): when it has changed, the factorization
    * is released and phase 11 is run again.
    *
    * @param reuse Whether to reuse the analysis
    */
   void SetReuseSymbolic(bool reuse = true);

   /// Whether analysis reuse is enabled; see SetReuseSymbolic().
   bool GetReuseSymbolic() const { return reuse_symbolic; }

   /**
    * @brief The number of analyses actually performed, that is, of PARDISO
    * phase 11 calls. Without reuse this is the number of SetOperator() calls;
    * with reuse it is the number of times the pattern changed.
    */
   long GetNumSymbolicFactorizations() const { return num_symbolic; }

   /**
    * @brief The number of numeric factorizations actually performed, that is,
    * of PARDISO phase 22 calls. Reuse never skips one of these.
    */
   long GetNumNumericFactorizations() const { return num_numeric; }

   ~PardisoSolver();

private:
   // Global number of rows
   int m;

   // Number of nonzero entries
   int nnz;

   // CSR data structure for the copy data of the local CSR matrix
   int *csr_rowptr = nullptr;
   real_t *reordered_csr_nzval = nullptr;
   int *reordered_csr_colind = nullptr;

   // Internal solver memory pointer pt,
   // 32-bit: int pt[64]
   // 64-bit: long int pt[64] or void *pt[64] should be OK on both architectures
   mutable void *pt[64] = {0};

   // Solver control parameters, detailed description can be found in the
   // constructor.
   mutable int iparm[64] = {0};
   mutable int maxfct, mnum, msglvl, phase, error;
   int mtype;
   // Number of right-hand sides of the next solve phase. Mutable because
   // ArrayMult() is const and sets it from the number of columns it is
   // given; CuDSSSolver does the same. PARDISO reads it only in the solve
   // phase, so changing it after the factorization is legal.
   mutable int nrhs;

   // Column-major packing of the right-hand sides and the solutions, sized
   // nrhs*m and reused across calls, as in STRUMPACKSolver. Empty until a
   // multi-column ArrayMult() asks for them.
   mutable Vector rhs_buf, sol_buf;

   // Dummy variables
   mutable int idum;
   mutable real_t ddum;

   // Whether the analysis may be retained and reused; see SetReuseSymbolic()
   bool reuse_symbolic = false;

   // The pattern the retained analysis was made for; empty unless reusing
   RetainedSparsityPattern pattern;

   // Whether pt holds a factorization, and so has memory to release
   bool factored = false;

   // The number of phase 11 and phase 22 calls actually made
   long num_symbolic = 0;
   long num_numeric = 0;

   // Release the memory held by pt, if there is any
   void ReleaseFactorization();
};
} // namespace mfem

#endif // MFEM_USE_MKL_PARDISO

#endif
