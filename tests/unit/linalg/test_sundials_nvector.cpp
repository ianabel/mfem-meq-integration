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

using namespace mfem;

#if defined(MFEM_USE_SUNDIALS) && defined(MFEM_USE_CUDA)

#include <nvector/nvector_cuda.h>

// SundialsNVector(N_Vector) wraps the N_Vector's host and device pointers,
// which registers the host pointer with the memory manager even when another
// Memory already has. Every SUNDIALS callback (KINSOL's residual, CVODE's
// right-hand side) builds such a view. Each view must give back the count it
// took, or the registration outlives the owner's buffer. A later allocation
// at the same address then inherits the old, smaller device buffer.
TEST_CASE("A SundialsNVector view returns the registration it takes",
          "[SUNDIALS][GPU]")
{
   Sundials::Init();
   const int n = 37;

   SECTION("a view of an address another Memory owns")
   {
      const void *h = nullptr;
      {
         SundialsNVector owner;
         owner.SetSize(n);   // registers h with its device copy
         owner = 1.0;        // on the device
         h = owner.GetData();
         REQUIRE(mm.IsKnown(h));

         for (int k = 0; k < 5; k++)
         {
            SundialsNVector view(static_cast<N_Vector>(owner));
            REQUIRE(view.Size() == n);
            view *= 2.0;
            REQUIRE(mm.IsKnown(h));
         }
         // The views wrote through to the owner's buffer.
         REQUIRE(owner.HostRead()[n - 1] == MFEM_Approx(32.0));
      }
      // The owner's Delete() is the last holder: the entry must be gone.
      REQUIRE_FALSE(mm.IsKnown(h));
   }

   SECTION("a view of an address no Memory owns")
   {
      // SUNDIALS allocates through SundialsMemHelper, which leaves the
      // memory manager without an entry, so the view's is the only one.
      N_Vector nv = N_VNewWithMemHelp_Cuda(n, SUNFALSE,
                                           Sundials::GetMemHelper(),
                                           Sundials::GetContext());
      REQUIRE(nv != nullptr);
      const void *h = N_VGetHostArrayPointer_Cuda(nv);
      REQUIRE_FALSE(mm.IsKnown(h));
      {
         SundialsNVector view(nv);
         view = 3.0;
         REQUIRE(mm.IsKnown(h));
      }
      REQUIRE_FALSE(mm.IsKnown(h));
      N_VDestroy(nv);
   }
}

#endif
