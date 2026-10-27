# mfem-meq-integration — a flattened snapshot, not a fork

This tree is **MFEM 4.10.1 as MEQ builds against it**: the source of branch
`meq-integration` at `1a15a28363a4ccc820208cc23ea06bf4e36adfaf`
(2026-10-26), exported with `git archive` and committed as one commit, with no
MFEM history.

`meq-integration` is a local merge, made only to be built against, of MFEM
`master` at `45799ebea8` (2026-09-19) and the following branches. Most of
them are from the HDG development line (`mfem-hdg-dev`). The count is how many
merges the branch's history names:

```
     57 gf-hdg-dev
     28 gf-hdg-linearise-first
     19 gf-hdg-subdomains-dev
     18 gf-interp-hdg-dev
     10 master
      8 gf-hdg-p-adaptivity
      4 sundials-ida-integration
      2 sundials-real-t
      2 cudss-solver-configuration
      1 transformback-single-tolerance
      1 pardiso-refinement-control
      1 pardiso-multi-rhs
      1 pardiso-device-host-sync
      1 memory-manager-registration-refcount
      1 intrules-triangle-high-order
      1 direct-solver-symbolic-reuse
      1 densematrixsvd-ldvt
      1 darcy-hrt-dev
      1 darcy-hdg-dev
      1 cmake-find-package-component-cache
      1 batched-linalg-device-fixes
```

**Do not send this upstream.** It is a custom blend; each branch goes upstream on
its own. When they are all merged into MFEM, this repository is retired.

Built for MEQ with CMake, at minimum:

    -DMFEM_USE_CUDA=YES -DCUDA_ARCH=sm_80 (A100; sm_75 on the RTX 2070 SUPER)
    -DMFEM_USE_CUDSS=YES -DMFEM_USE_SUNDIALS=YES (SUNDIALS built with CUDA)
    -DMFEM_USE_MKL_PARDISO=YES and/or -DMFEM_USE_SUITESPARSE=YES
    -DMFEM_USE_OPENMP=YES -DMFEM_THREAD_SAFE=YES -DMFEM_USE_LAPACK=YES
    -DMFEM_USE_EXCEPTIONS=YES -DMFEM_USE_MPI=NO

MEQ's `CLAUDE_MFEM.md` is the full record, and `meq-a100-kit/build.sh` the
recipe used off this machine.
