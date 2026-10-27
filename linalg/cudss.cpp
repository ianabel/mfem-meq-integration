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

#include "cudss.hpp"
#include "../general/communication.hpp"
#include "../general/cuda.hpp"
#include <string>

#ifdef MFEM_USE_CUDSS

#if CUDSS_VERSION >= 800
#ifdef MFEM_USE_SINGLE
#define CUDSS_REAL_T CUDSS_R_32F
#else
#define CUDSS_REAL_T CUDSS_R_64F
#endif
#define CUDSS_INT_T CUDSS_R_32I
#else
#ifdef MFEM_USE_SINGLE
#define CUDSS_REAL_T CUDA_R_32F
#else
#define CUDSS_REAL_T CUDA_R_64F
#endif
#define CUDSS_INT_T CUDA_R_32I
#endif

// Define a cuDSS error check macro, MFEM_CUDSS_CHECK(x), where x returns/is of
// type 'cudssStatus_t'. This macro evaluates 'x' and raises an error if the
// result is not CUDSS_STATUS_SUCCESS.
#define MFEM_CUDSS_CHECK(x)                                          \
  do {                                                               \
    cudssStatus_t mfem_err_internal_var_name = (x);                  \
    if (mfem_err_internal_var_name != CUDSS_STATUS_SUCCESS) {        \
      ::mfem::mfem_cudss_error(mfem_err_internal_var_name, #x,       \
                               _MFEM_FUNC_NAME, __FILE__, __LINE__); \
    }                                                                \
  } while (0)

namespace mfem
{
// Function used by the macro MFEM_CUDSS_CHECK.
void mfem_cudss_error(cudssStatus_t status, const char *expr, const char *func,
                      const char *file, int line)
{
   mfem::err << "\n\nCUDSS error: (" << expr << ") failed with error:\n --> "
             << "CUDSS call ended unsuccessfully"
             << " [code: " << static_cast<int>(status) << ']'
             << "\n ... in function: " << func << "\n ... in file: " << file
             << ':' << line << '\n';
   mfem_error();
}

CuDSSSolver::CuDSSSolver() { InitCuDSS(); }

#ifdef MFEM_USE_MPI
CuDSSSolver::CuDSSSolver(MPI_Comm comm_) : mpi_comm(comm_)
{
   InitCuDSS();

   // NOTE: Set the communication layer to NULL so that cuDSS picks it
   // from the environment variable "CUDSS_COMM_LIB"
   const char* comm_lib = GetEnv("CUDSS_COMM_LIB");
#ifdef MFEM_CUDSS_COMM_LIB
   if (comm_lib == nullptr)
   {
      comm_lib = MFEM_CUDSS_COMM_LIB;
   }
#endif
   MFEM_CUDSS_CHECK(cudssSetCommLayer(handle, comm_lib));

#if CUDSS_VERSION >= 800
   MFEM_CUDSS_CHECK(cudssDataSet(handle, solverData, CUDSS_DATA_COMM_HOST,
                                 &mpi_comm, sizeof(MPI_Comm *)));
#else
   MFEM_CUDSS_CHECK(cudssDataSet(handle, solverData, CUDSS_DATA_COMM,
                                 &mpi_comm, sizeof(MPI_Comm *)));
#endif
}
#endif // MFEM_USE_MPI

CuDSSSolver::~CuDSSSolver()
{
   // Sync the stream to make sure any pending asynchronous operations have
   // completed.
   MFEM_STREAM_SYNC;
   // Destroy the system Matrix, RHS vector and solution vector
   if (Ac)
   {
      MFEM_CUDSS_CHECK(cudssMatrixDestroy(*Ac));
      MFEM_CUDSS_CHECK(cudssMatrixDestroy(xc));
      MFEM_CUDSS_CHECK(cudssMatrixDestroy(yc));
   }

   // Destroy the cuDSS handle, solver config and solver data
   MFEM_CUDSS_CHECK(cudssDataDestroy(handle, solverData));
   MFEM_CUDSS_CHECK(cudssConfigDestroy(solverConfig));

   MFEM_CUDSS_CHECK(cudssDestroy(handle));
   handle = nullptr;


   if (csr_offsets_d != NULL)
   {
      CuMemFree(csr_offsets_d);
   }

   if (csr_columns_d != NULL)
   {
      CuMemFree(csr_columns_d);
   }

   if (csr_values_d != NULL)
   {
      CuMemFree(csr_values_d);
   }
}

void CuDSSSolver::InitCuDSS()
{
   // Create the cuDSS handle
   MFEM_CUDSS_CHECK(cudssCreate(&handle));

   // Set CuDSS to use MFEM's default stream of 0.
   MFEM_CUDSS_CHECK(cudssSetStream(handle, 0));

#ifdef MFEM_USE_OPENMP
   // NOTE: Set the threading layer library name to NULL so that cuDSS picks
   // it from the environment variable "CUDSS_THREADING_LIB"
   const char* threading_lib = GetEnv("CUDSS_THREADING_LIB");
#ifdef MFEM_CUDSS_THREADING_LIB
   if (threading_lib == nullptr)
   {
      threading_lib = MFEM_CUDSS_THREADING_LIB;
   }
#endif
   MFEM_CUDSS_CHECK(cudssSetThreadingLayer(handle, threading_lib));
#endif  // MFEM_USE_OPENMP

   // Create the solver configuration and data objects
   MFEM_CUDSS_CHECK(cudssConfigCreate(&solverConfig));
   MFEM_CUDSS_CHECK(cudssDataCreate(handle, &solverData));
}

void CuDSSSolver::SetMatrixSymType(MatType mtype_)
{
   switch (mtype_)
   {
      case MatType::SYMMETRIC_INDEFINITE:
         mat_type = CUDSS_MTYPE_SYMMETRIC;
         break;
      case MatType::SYMMETRIC_POSITIVE_DEFINITE:
         mat_type = CUDSS_MTYPE_SPD;
         break;
      default:
         mat_type = CUDSS_MTYPE_GENERAL;
         break;
   }
}

void CuDSSSolver::SetMatrixViewType(MatViewType mvtype_)
{
   // NOTE: what is stored here is the REQUEST, and whether it can be honored
   // is decided in ResolvedMatrixView() once the matrix type is also known.
   // Deciding it here would make the answer depend on whether the caller
   // reached SetMatrixSymType() first -- see that routine.
   switch (mvtype_)
   {
      case MatViewType::LOWER:
         mview_req = CUDSS_MVIEW_LOWER;
         break;
      case MatViewType::UPPER:
         mview_req = CUDSS_MVIEW_UPPER;
         break;
      default:
         mview_req = CUDSS_MVIEW_FULL;
         break;
   }
}

cudssMatrixViewType_t CuDSSSolver::ResolvedMatrixView() const
{
   // A general matrix has no triangular view: cuDSS reads the whole pattern,
   // so FULL is the only meaningful answer whatever was asked for.
   if (mat_type == CUDSS_MTYPE_GENERAL) { return CUDSS_MVIEW_FULL; }
   return mview_req;
}

void CuDSSSolver::SetReorderingReuse(bool reuse)
{
   MFEM_VERIFY(Ac == nullptr,
               "Set reordering reuse before setting the operator!");
   reorder_reuse = reuse;
}

void CuDSSSolver::SetRefinementSteps(int steps)
{
   MFEM_VERIFY(steps >= 0, "The number of refinement steps cannot be negative!");
   ir_n_steps = steps;
}

void CuDSSSolver::SetDeterministic(bool deterministic_)
{
   deterministic = deterministic_;
}

void CuDSSSolver::ApplyConfig() const
{
   // cuDSS reads the configuration at execute time, so this runs once per
   // SetOperator rather than from the setters -- which is also what lets the
   // refusal below see the matrix type, whichever order the caller set them in.
   if (ir_n_steps != 0)
   {
      MFEM_CUDSS_CHECK(cudssConfigSet(solverConfig, CUDSS_CONFIG_IR_N_STEPS,
                                      &ir_n_steps, sizeof(int)));
   }
   if (deterministic)
   {
      // MEASURED on cuDSS 0.8.0 with an RTX 2070 SUPER (sm_75): with
      // CUDSS_MTYPE_SYMMETRIC, every phase returns CUDSS_STATUS_SUCCESS and
      // the solve writes an EXACTLY ZERO solution -- residual 2.0 against a
      // right-hand side of that scale, reproduced three times, while the same
      // matrix with the switch off solves to 3.6e-15. CUDSS_MTYPE_GENERAL and
      // CUDSS_MTYPE_SPD are both correct. A silent wrong answer is not
      // something to pass on to a caller, so it is refused here; if a later
      // cuDSS fixes it, this is the one line to relax, and the re-measurement
      // is the [CuDSS] unit case.
      MFEM_VERIFY(mat_type != CUDSS_MTYPE_SYMMETRIC,
                  "cuDSS returns an exactly zero solution, and reports "
                  "success, for a SYMMETRIC matrix in deterministic mode. "
                  "Use CuDSSSolver::SYMMETRIC_POSITIVE_DEFINITE if the matrix "
                  "is definite, or leave SetDeterministic() off.");
      const int on = 1;
      MFEM_CUDSS_CHECK(cudssConfigSet(solverConfig,
                                      CUDSS_CONFIG_DETERMINISTIC_MODE,
                                      &on, sizeof(int)));
   }
}

#ifdef MFEM_USE_MPI
void CuDSSSolver::SetMatrix(const HypreParMatrix &op)
{
   bool cuDSSObjectInitialized = (Ac != nullptr);

   hypre_ParCSRMatrix *parcsr_op = op;
   op.HypreRead();
   hypre_CSRMatrix *csr_op = hypre_MergeDiagAndOffd(parcsr_op);
   op.HypreRead();
#if MFEM_HYPRE_VERSION >= 21600
   hypre_CSRMatrixBigJtoJ(csr_op);
#endif

   // Parameters of the Operator
   n_loc = height;  // Equal to the csr_op->num_rows
   n_global = internal::to_int(parcsr_op->global_num_rows);
   row_start = parcsr_op->first_row_index;
   row_end = row_start + n_loc - 1;
   MFEM_VERIFY(!cuDSSObjectInitialized || !reorder_reuse ||
               (reorder_reuse && (nnz == csr_op->num_nonzeros)),
               "Inconsistent new matrix pattern!");
   nnz = csr_op->num_nonzeros;

   SetMatrixCuDSS(csr_op->i, csr_op->j, csr_op->data);
   hypre_CSRMatrixDestroy(csr_op);
}
#endif // MFEM_USE_MPI

// The number of stored entries, I[height], read without bringing I to the
// host. SparseMatrix::NumNonZeroElems() reads it through HostReadI(), and when
// I is valid only on the device -- a matrix whose pattern was built there --
// that downloads the whole row-pointer array to read its last entry, once per
// SetOperator(), just before cuDSS takes I back on the device.
static int DeviceCSRNumNonZeros(const SparseMatrix &A)
{
   if (A.GetMemoryI().HostIsValid()) { return A.HostReadI()[A.Height()]; }
   int count;
   CuMemcpyDtoH(&count, A.ReadI() + A.Height(), sizeof(int));
   return count;
}

void CuDSSSolver::SetMatrix(const SparseMatrix &op)
{
   bool cuDSSObjectInitialized = (Ac != nullptr);

   MFEM_VERIFY(op.Finalized(), "CuDSSSolver needs a finalized SparseMatrix.");
   const int op_nnz = DeviceCSRNumNonZeros(op);

   // Parameters of the Operator
   MFEM_VERIFY(!cuDSSObjectInitialized || !reorder_reuse ||
               (reorder_reuse && (nnz == op_nnz)),
               "Inconsistent new matrix pattern!");

   SparseMatrix *A = const_cast<SparseMatrix *>(&op);

   nnz = op_nnz;
   n_global = height;  // Equal to the height in serial
   n_loc = height;     // Equal to the height in serial

   int *csr_offsets = const_cast<int *>(A->ReadI());
   int *csr_columns = const_cast<int *>(A->ReadJ());
   real_t *csr_values = const_cast<real_t *>(A->ReadData());

   SetMatrixCuDSS(csr_offsets, csr_columns, csr_values);
}

void CuDSSSolver::SetMatrixCuDSS(int *csr_offsets, int *csr_columns,
                                 real_t *csr_values)
{
   bool cuDSSObjectInitialized = (Ac != nullptr);

   ApplyConfig();

   // Initial the cudssMatrix objects
   if (!cuDSSObjectInitialized)
   {
      // Set the cudssMatrix object of csr operator
      Ac = std::make_unique<cudssMatrix_t>();
      // Create empty RHS and solution vectors
      SetNumRHS(1);
   }

   if (cuDSSObjectInitialized && !reorder_reuse)
   {
      MFEM_STREAM_SYNC;
      MFEM_CUDSS_CHECK(cudssMatrixDestroy(*Ac));
   }

   // Allocate device memory for csr values. Unless reuse is specified, the
   // nnz may be different, so we will free and reallocate.
   if (csr_values_d == NULL || !reorder_reuse)
   {
      if (csr_values_d != NULL) { CuMemFree(csr_values_d); }
      CuMemAlloc(&csr_values_d, nnz * sizeof(real_t));
   }
   CuMemcpyDtoD(csr_values_d, csr_values, nnz * sizeof(real_t));

   // We copy and store the I and J arrays, since the CuDSS matrix object
   // technically needs these to be valid, so we protect against the caller
   // destroying the original matrix.
   if (!cuDSSObjectInitialized || !reorder_reuse)
   {
      if (csr_offsets_d != NULL) { CuMemFree(csr_offsets_d); }
      CuMemAlloc(&csr_offsets_d, (n_loc + 1) * sizeof(int));
      if (csr_columns_d != NULL) { CuMemFree(csr_columns_d); }
      CuMemAlloc(&csr_columns_d, nnz * sizeof(int));
      CuMemcpyDtoD(csr_offsets_d, csr_offsets, (n_loc + 1) * sizeof(int));
      CuMemcpyDtoD(csr_columns_d, csr_columns, nnz * sizeof(int));
   }

   // New cuDSS CSR matrix object and analysis or reuse the one from a previous
   // matrix
   if (!cuDSSObjectInitialized || !reorder_reuse)
   {
      if (reorder_reuse)  // !cuDSSObjectInitialized && reorder_reuse
      {
#if CUDSS_VERSION >= 800
         MFEM_CUDSS_CHECK(
            cudssMatrixCreateCsr(
               Ac.get(), n_global, n_global, nnz, csr_offsets_d, NULL,
               csr_columns_d, csr_values_d, CUDSS_INT_T, CUDSS_INT_T, CUDSS_REAL_T,
               mat_type, ResolvedMatrixView(), CUDSS_BASE_ZERO));
#else
         MFEM_CUDSS_CHECK(
            cudssMatrixCreateCsr(
               Ac.get(), n_global, n_global, nnz, csr_offsets_d, NULL,
               csr_columns_d, csr_values_d, CUDSS_INT_T, CUDSS_REAL_T,
               mat_type, ResolvedMatrixView(), CUDSS_BASE_ZERO));
#endif
      }
      else    // !reorder_reuse
      {
#if CUDSS_VERSION >= 800
         MFEM_CUDSS_CHECK(
            cudssMatrixCreateCsr(
               Ac.get(), n_global, n_global, nnz, csr_offsets_d, NULL,
               csr_columns_d, csr_values_d, CUDSS_INT_T, CUDSS_INT_T, CUDSS_REAL_T,
               mat_type, ResolvedMatrixView(), CUDSS_BASE_ZERO));
#else
         MFEM_CUDSS_CHECK(
            cudssMatrixCreateCsr(
               Ac.get(), n_global, n_global, nnz, csr_offsets_d, NULL,
               csr_columns_d, csr_values_d, CUDSS_INT_T, CUDSS_REAL_T,
               mat_type, ResolvedMatrixView(), CUDSS_BASE_ZERO));
#endif
      }
#ifdef MFEM_USE_MPI
      if (Mpi::IsInitialized())
      {
         MFEM_CUDSS_CHECK(cudssMatrixSetDistributionRow1d(*Ac, row_start, row_end));
      }
#endif
      // Analysis
      MFEM_CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solverConfig,
                                    solverData, *Ac, yc, xc));
   }
   else    // cuDSSObjectInitialized && reorder_reuse
   {
      // NOTE: When reusing analysis result, we only update the Data array,
      // without changing the I and J arrays.
      MFEM_CUDSS_CHECK(cudssMatrixSetValues(*Ac, csr_values_d));
   }

   // Factorization
   MFEM_CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solverConfig,
                                 solverData, *Ac, yc, xc));

   // In serial, the factorization can execute asynchronously.
   MFEM_STREAM_SYNC;
}

void CuDSSSolver::SetOperator(const Operator &op)
{
   bool cuDSSObjectInitialized = (Ac != nullptr);
   MFEM_VERIFY(
      !cuDSSObjectInitialized || (height == op.Height() && width == op.Width()),
      "Inconsistent new matrix size!");
   height = op.Height();
   width = op.Width();
   if (const SparseMatrix *A = dynamic_cast<const SparseMatrix *>(&op))
   {
      SetMatrix(*A);
   }
#ifdef MFEM_USE_MPI
   else if (const HypreParMatrix *A =
               dynamic_cast<const HypreParMatrix *>(&op))
   {
      SetMatrix(*A);
   }
#endif // MFEM_USE_MPI
   else
   {
      MFEM_ABORT("Unsupported Operator Type \n");
   }
}

void CuDSSSolver::SetNumRHS(int nrhs_) const
{
   if (deterministic && nrhs_ > 1 && !warned_det_nrhs)
   {
      // Not a refusal: cuDSS declines this itself, loudly, and a later version
      // may not. What it does not do is say why, and the status alone
      // (CUDSS_STATUS_NOT_SUPPORTED) names neither of the two settings that
      // produced it. Measured on 0.8.0: the SOLVE phase refuses at K = 4 and
      // K = 16 and succeeds, bit-reproducibly, at K = 1.
      MFEM_WARNING("cuDSS 0.8.0 does not support deterministic mode with more "
                   "than one right-hand side; the solve phase will refuse "
                   << nrhs_ << " of them with CUDSS_STATUS_NOT_SUPPORTED.");
      warned_det_nrhs = true;
   }
   if (nrhs != nrhs_)
   {
      if (nrhs > 0)
      {
         // Destroy the previous RHS vector and solution vector
         MFEM_STREAM_SYNC;
         MFEM_CUDSS_CHECK(cudssMatrixDestroy(xc));
         MFEM_CUDSS_CHECK(cudssMatrixDestroy(yc));
      }
      // Create empty RHS and solution vectors
      MFEM_CUDSS_CHECK(cudssMatrixCreateDn(&xc, n_global, nrhs_, n_global, NULL,
                                           CUDSS_REAL_T, CUDSS_LAYOUT_COL_MAJOR));

      MFEM_CUDSS_CHECK(cudssMatrixCreateDn(&yc, n_global, nrhs_, n_global, NULL,
                                           CUDSS_REAL_T, CUDSS_LAYOUT_COL_MAJOR));

#ifdef MFEM_USE_MPI
      MFEM_CUDSS_CHECK(cudssMatrixSetDistributionRow1d(xc, row_start, row_end));
      MFEM_CUDSS_CHECK(cudssMatrixSetDistributionRow1d(yc, row_start, row_end));
#endif // MFEM_USE_MPI
   }
   nrhs = nrhs_;
}

void CuDSSSolver::Mult(const Vector &x, Vector &y) const
{
   Array<const Vector *> X(1);
   Array<Vector *> Y(1);
   X[0] = &x;
   Y[0] = &y;
   ArrayMult(X, Y);
}

void CuDSSSolver::ArrayMult(const Array<const Vector *> &X,
                            Array<Vector *> &Y) const
{
   SetNumRHS(X.Size());

   Vector RHS, SOL;

   if (nrhs == 1)
   {
      RHS.MakeRef(*(const_cast<Vector *>(X[0])), 0, X[0]->Size());
      SOL.MakeRef(*Y[0], 0, Y[0]->Size());
   }
   else
   {
      // NOTE: RHS must have **global** num_rows and nrhs columns
      RHS.SetSize(nrhs * n_global, *X[0]);
      for (int i = 0; i < nrhs; i++)
      {
         Vector s(RHS, i * n_global, n_loc);
         s = *X[i];
      }

      // NOTE: SOL must have **global** num_rows and nrhs columns
      SOL.SetSize(nrhs * n_global, *Y[0]);
   }

   MFEM_CUDSS_CHECK(cudssMatrixSetValues(xc, const_cast<real_t *>(RHS.Read())));
   MFEM_CUDSS_CHECK(cudssMatrixSetValues(yc, SOL.Write()));

   // Solve
   MFEM_CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, solverConfig,
                                 solverData, *Ac, yc, xc));

   if (nrhs == 1)
   {
      SOL.SyncAliasMemory(*Y[0]);
   }

   if (nrhs > 1)
   {
      // Get solution for each right-hand side
      for (int i = 0; i < nrhs; i++)
      {
         Vector s(SOL, i * n_global, n_loc);
         *Y[i] = s;
      }
   }
}

} // namespace mfem
#endif // MFEM_USE_CUDSS
