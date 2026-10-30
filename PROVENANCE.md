# mfem-meq-integration — a flattened snapshot, not a fork

This tree is **MFEM 4.10.1 as MEQ builds against it**: the source of branch
`meq-integration` at `9d57974336b09ecf2a3282ff2d59ecca960d1106`
(2026-10-28), exported with `git archive`. Each commit here is one such snapshot,
whole, with no MFEM history; a clone fast-forwards from one to the next.
`.github/` is left out, so that GitHub does not run MFEM's CI here.

`meq-integration` is a local merge, made only to be built against, of MFEM
`master` at `45799ebea8` and the following branches of the HDG development line
(`mfem-hdg-dev`), each at the tip it is in at:

```
    batched-linalg-device-fixes            72a035fa92
    batched-linalg-persistent-pointers     b41a35bea5
    cmake-find-package-component-cache     5d74ce4ffa
    cudss-solver-configuration             45117b684b
    darcy-hdg-dev                          e047e5813f
    densematrixsvd-ldvt                    cee6a378af
    direct-solver-symbolic-reuse           9eca432683
    extension-thread-safe-scratch          26fccc9f3d
    gf-hdg-dev                             52acf6f97f
    gf-hdg-linearise-first                 20f0956ba2
    gf-hdg-p-adaptivity                    30908e57c3
    gf-hdg-subdomains-dev                  2bed225310
    gf-interp-hdg-dev                      0c25527ba6
    intrules-triangle-high-order           c6693d5f09
    memory-manager-registration-refcount   19e59e5cea
    pardiso-device-host-sync               9a4ccc4945
    pardiso-multi-rhs                      3f788f9068
    pardiso-refinement-control             14ec0e2fd9
    sundials-ida-integration               fc5b94c9aa
    sundials-real-t                        b6a018e5f0
    transformback-single-tolerance         d10e8c48e7
```

**Do not send this upstream.** It is a custom blend; each branch goes upstream on
its own. When they are all merged into MFEM, this repository is retired.

Built for MEQ with CMake, at minimum:

    -DMFEM_USE_CUDA=YES -DCUDA_ARCH=sm_80 (A100; sm_75 on the RTX 2070 SUPER)
    -DMFEM_USE_CUDSS=YES -DMFEM_USE_SUNDIALS=YES (SUNDIALS built with CUDA)
    -DMFEM_USE_MKL_PARDISO=YES and/or -DMFEM_USE_SUITESPARSE=YES
    -DMFEM_USE_OPENMP=YES -DMFEM_THREAD_SAFE=YES -DMFEM_USE_LAPACK=YES
    -DMFEM_USE_EXCEPTIONS=YES -DMFEM_USE_MPI=NO

MEQ's `CLAUDE_MFEM.md` is the full record, and MEQ's
`tools/device-timing/build.sh` the recipe used off the development machine.
