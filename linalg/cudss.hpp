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

#ifndef MFEM_CUDSS
#define MFEM_CUDSS

#include "../config/config.hpp"

#ifdef MFEM_USE_CUDSS

#include "cudss.h"
#include <memory>

#ifdef MFEM_USE_MPI
#include <mpi.h>
#include "hypre.hpp"
#else
#include "operator.hpp"
#include "sparsemat.hpp"
#endif

namespace mfem
{
/**
 * @brief cuDSS: A high-performance CUDA Library for Direct Sparse Solvers
 *
 * Interface for the distributed cuDSS solver
 */
class CuDSSSolver : public Solver
{
public:
   /// Specify the type of matrix we are applying the solver to
   enum MatType
   {
      /// CUDSS_MTYPE_GENERAL: General matrix [default].
      NONSYMMETRIC = 0,
      /// CUDSS_MTYPE_SYMMETRIC: Real symmetric matrix.
      SYMMETRIC_INDEFINITE = 1,
      /// CUDSS_MTYPE_SPD: Symmetric positive-definite matrix.
      SYMMETRIC_POSITIVE_DEFINITE = 2,
   };

   /// Specify the view type of matrix we are applying the solver to
   enum MatViewType
   {
      /// CUDSS_MVIEW_FULL: Full matrix [default]
      FULL = 0,
      /// CUDSS_MVIEW_LOWER: Lower-triangular matrix (including the diagonal).
      LOWER = 1,
      /// CUDSS_MVIEW_UPPER: Upper-triangular matrix (including the diagonal).
      UPPER = 2,
   };

   /**
    * @brief Constructor.
    */
   CuDSSSolver();

#ifdef MFEM_USE_MPI
   /**
    * @brief Constructor with MPI_Comm parameter.
    */
   CuDSSSolver(MPI_Comm comm);
#endif

   // Note: CuDSSSolver disables the move copy constructor and move assignment
   // operator
   CuDSSSolver(CuDSSSolver &&) = delete;
   CuDSSSolver &operator=(CuDSSSolver &&) = delete;

   /**
    * @brief Set the matrix type
    *
    * Supported matrix types:
    *          CuDSSSolver::NONSYMMETRIC,
    *          CuDSSSolver::SYMMETRIC_INDEFINITE,
    *      and CuDSSSolver::SYMMETRIC_POSITIVE_DEFINITE
    *
    * @param mtype_ Matrix type
    *
    * @note This method has to be called before SetOperator
    */
   void SetMatrixSymType(MatType mtype_);

   /**
    * @brief Set the matrix view type
    *
    * Supported matrix types:
    *          CuDSSSolver::FULL,
    *          CuDSSSolver::LOWER,
    *      and CuDSSSolver::UPPER
    *
    * @param mvtype Matrix view type
    *
    * @note This method has to be called before SetOperator, and unlike an
    * earlier version of this class it may be called in either order with
    * respect to SetMatrixSymType(). It used to read the matrix type and
    * silently reduce a LOWER or UPPER request to FULL when that type was
    * still at its NONSYMMETRIC default, so a caller who stated the view first
    * had the view dropped and -- having passed only one triangle -- got a
    * wrong answer with no warning. Measured on a 400x400 SPD Laplacian stored
    * as its lower triangle: correct to 1.3e-15 one way round and out by 2.1
    * on a solution of norm 21 the other. The request is now kept as given and
    * reconciled with the matrix type when the matrix is created.
    */
   void SetMatrixViewType(MatViewType mvtype);

   /**
    * @brief Set the flag controlling reuse of the symbolic factorization
    * for multiple operators
    *
    * @param reuse Flag to reuse symbolic factorization
    *
    * @note This method has to be called before repeated calls to SetOperator
    */
   void SetReorderingReuse(bool reuse);

   /**
    * @brief Set the number of iterative-refinement steps cuDSS takes after
    * the solve, CUDSS_CONFIG_IR_N_STEPS
    *
    * @param steps Number of refinement steps; 0 is cuDSS's own default and
    * means no refinement at all
    *
    * @note This method has to be called before SetOperator
    *
    * @note The default was MEASURED rather than read: cudssConfigCreate()
    * leaves CUDSS_CONFIG_IR_N_STEPS at 0 and CUDSS_CONFIG_IR_TOL at 0 in
    * cuDSS 0.8.0, so a solver that never calls cudssConfigSet() does no
    * refinement. Steps of 1 to 4 were checked to be honored -- cuDSS reports
    * them back through CUDSS_DATA_IR_N_STEPS -- at one and at four
    * right-hand sides.
    *
    * @note This is refinement WITHIN one precision and is not the
    * factor-low-solve-high scheme STRUMPACKMixedPrecisionSolver offers.
    * cudssMatrixCreateCsr() and cudssMatrixCreateDn() each take a single
    * value type and this class passes real_t to all of them, so there is one
    * precision per solve.
    */
   void SetRefinementSteps(int steps);

   /**
    * @brief Ask cuDSS for a reproducible solve, CUDSS_CONFIG_DETERMINISTIC_MODE
    *
    * @param deterministic_ Whether to request deterministic mode
    *
    * @note This method has to be called before SetOperator
    *
    * @note Off is cuDSS's default, and the consequence was measured rather
    * than assumed: on cuDSS 0.8.0 the same matrix and right-hand side, solved
    * from a fresh handle five times over, gives five DIFFERENT answers --
    * every repeat differs, by 6.7e-16 at one right-hand side and up to
    * 7.1e-15 at sixteen, for a general and for a symmetric matrix alike. That
    * is arithmetic reassociation rather than a fault, and it is the same class
    * of thing as a threaded BLAS-3, but a caller asserting bit-exactness
    * across two routes cannot currently opt out of it. With this on, five
    * repeats are bit-identical.
    *
    * @note cuDSS 0.8.0 supports it only for ONE right-hand side: the solve
    * phase refuses 4 and 16 columns with CUDSS_STATUS_NOT_SUPPORTED, so a
    * caller applying one factorization to several right-hand sides through
    * ArrayMult() cannot have both. This class warns once and lets cuDSS
    * refuse, rather than refusing on cuDSS's behalf.
    *
    * @note It is REFUSED here with SYMMETRIC_INDEFINITE, because on 0.8.0
    * that combination returns success and an exactly zero solution. See the
    * comment on the refusal in ApplyConfig().
    */
   void SetDeterministic(bool deterministic_);

   void SetOperator(const Operator &op) override;

   /**
    * @brief Solve $ y = Op^{-1} x $
    *
    * @param x RHS vector
    * @param y Solution vector
    */
   void Mult(const Vector &x, Vector &y) const override;

   /**
    * @brief Solve $ Y_i = Op^{-1} X_i $
    *
    * @param X Array of RHS vectors
    * @param Y Array of Solution vectors
    */
   void ArrayMult(const Array<const Vector *> &X,
                  Array<Vector *> &Y) const override;

   ~CuDSSSolver();

private:
#ifdef MFEM_USE_MPI
   // MPI_Comm
   MPI_Comm mpi_comm = MPI_COMM_NULL;

   int row_start = 0;  // the first row index in CSR matrix operator
   int row_end = 0;    // the end row index in CSR matrix operator
#endif

   // Parameter controlling whether or not to reuse the symbolic factorization
   // for multiple calls to SetOperator
   bool reorder_reuse = false;

   // Number of iterative refinement steps; 0 is cuDSS's own default.
   int ir_n_steps = 0;

   // Whether to ask cuDSS for a reproducible solve.
   bool deterministic = false;

   // So the warning about deterministic mode with several right-hand sides is
   // issued once per solver rather than once per Newton step.
   mutable bool warned_det_nrhs = false;

   // Parameter controlling the matrix type
   cudssMatrixType_t mat_type = CUDSS_MTYPE_GENERAL;

   int n_global = 0;      // global number of rows
   int n_loc = 0;         // the number of the rows in CSR matrix operator

   mutable int nrhs = 0;  // the number of the RHSs
   int nnz = 0;        // the number of non zeros

   // copy and keep the I and J arrays in device memory
   void *csr_offsets_d = NULL; // copy and keep I in device
   void *csr_columns_d = NULL; // copy and keep J in device
   void *csr_values_d = NULL;  // copy and keep csr data in device

   // The matrix view the caller ASKED for. What is handed to cuDSS is
   // ResolvedMatrixView(), which reconciles it with mat_type.
   cudssMatrixViewType_t mview_req = CUDSS_MVIEW_FULL;

   // cuDSS objects storage for sparse matrix Ac, RHS yc and solution xc
   std::unique_ptr<cudssMatrix_t> Ac;
   mutable cudssMatrix_t xc, yc;

   // common for all cuDSS solver instances.
   // cuDSS object holds the cuDSS library context
   cudssHandle_t handle;

   // cuDSS object stores configuration settings for the solver
   mutable cudssConfig_t solverConfig;
   // cuDSS object holds internal data
   mutable cudssData_t solverData;

   /// Method for configuring storage for distributed/centralized RHS and
   /// solution
   void SetNumRHS(int nrhs_) const;

#ifdef MFEM_USE_MPI
   /**
    * @brief Set the HypreParMatrix object
    *
    * @param op HypreParMatrix object
    *
    * @note This method is called inside SetOperator
    */
   void SetMatrix(const HypreParMatrix &op);
#endif

   /**
    * @brief Set the SparseMatrix object
    *
    * @param op SparseMatrix object
    *
    * @note This method is called inside SetOperator
    */
   void SetMatrix(const SparseMatrix &op);

   /**
    * @brief Set the matrix values for cuDSS
    *
    * @param csr_offsets Row offsets of the CSR matrix
    * @param csr_columns Column indices of the CSR matrix
    * @param csr_values Non-zero values of the CSR matrix
    *
    * @note This method is called inside SetMatrix.
   */
   void SetMatrixCuDSS(int* csr_offsets, int* csr_columns, real_t* csr_values);

   /// Push the settings above into the cuDSS config object, and refuse the
   /// one combination that is silently wrong. Called from SetMatrixCuDSS(),
   /// where the matrix type is known whatever order the setters were used in.
   void ApplyConfig() const;

   /// The matrix view actually passed to cuDSS: the caller's request, except
   /// that a general matrix has no triangular view and is always FULL.
   cudssMatrixViewType_t ResolvedMatrixView() const;

   /// Method for initializing the cuDSS library and creating the cuDSS handle
   void InitCuDSS();
};

} // namespace mfem

#endif // MFEM_USE_CUDSS
#endif // MFEM_CUDSS
