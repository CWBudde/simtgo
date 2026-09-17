# From proof of concept to a production tool

This plan turns the PoC into something other people can depend on. It is
ordered by dependency, not by appeal: the early phases are the ones that
cannot be retrofitted later.

Sizes are rough — **S** ≈ days, **M** ≈ 1–3 weeks, **L** ≈ 1–2 months of
focused work.

## Where this stands

~3,300 lines. Both tracks work and are verified against CPU references on
real hardware. That verification covers **one GPU, one architecture, one CUDA
version, one OS, one Go version**.

What the PoC is not:

| | Current | Needed |
|---|---|---|
| Driver API | 18 calls, fully synchronous | streams, events, async copies, pinned memory |
| Grids | 1-D only | 1-D/2-D/3-D |
| Types | `float32`, `int32`/`int` | integers, `float64`, structs, fixed arrays |
| Kernel calls | none — a kernel cannot call a Go function | `__device__` functions |
| Errors | surface at **run time**, inside `main()` | at **build time** |
| Toolchain | hard-coded `/usr/local/cuda` (`cuda/driver_cuda.go:6`) | discovery, or no build-time CUDA at all |
| Tile ops | 7, one windowed, no reductions | reductions, 2-D, fusion planning |

## The decision that gates everything else

**Keep generating CUDA C for NVRTC, or emit PTX (or LLVM IR) directly?**

| | CUDA C + NVRTC (today) | Direct PTX/LLVM IR |
|---|---|---|
| Optimiser | NVIDIA's, for free | yours to write |
| Artifacts | readable `.cu`, easy to diff and debug | PTX only |
| Runtime dependency | `libnvrtc` (~60 MB) | none beyond the driver |
| Startup | ~25 ms compile per kernel | none, if AOT |
| Control | none over registers, scheduling, ABI | total |
| Effort to extend | low | very high |

**Recommendation: stay on CUDA C through 1.0.** The C++ round trip has not
been the limiting factor in anything measured so far, and giving up NVIDIA's
optimiser means owning register allocation and instruction scheduling — a
different project.

- [ ] **(S)** Spike: emit PTX directly for `VecAdd`, measure against the
      NVRTC path. Keeps the option open and makes the decision evidence-based
      rather than architectural taste.
- [ ] **(S)** Decide the AOT story now (Phase 1.1) — it changes the public API,
      so it cannot wait.

## Phase 0 — Fix what is known to be broken (S)

Small, verified defects. Do these first; they are cheap and they distort any
benchmark or test written on top of them.

- [x] **Parameter name collision.** `func K(ctx gpu.Ctx, x []float32, x_len int32)`
      emits `int x_len, int x_len` — a duplicate C parameter. *Verified.*
      Fix: generate lengths into a reserved namespace and reject collisions
      with a Go-level error instead of letting NVRTC report a C one.
- [x] **Symbol table keyed by name.** `transpiler.lens` (`simt/transpile.go`)
      maps identifier *strings* to length expressions. Key it on
      `types.Object` so shadowing can never resolve to the wrong symbol.
- [x] **Unvalidated launch geometry.** `ctx.SharedF32(FIRBlock + FIRMaxTaps)`
      assumes a launch with `block == FIRBlock`; launching otherwise silently
      corrupts results. Record the required block size during transpilation
      and check it in `Kernel.Launch`.
- [x] **Unvalidated shared-memory size.** Compare the total against
      `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK` at build time.
- [x] **Precedence by string inspection.** `paren()` in `simt/expr.go` guesses
      from the rendered text. Track precedence in the emitter and parenthesise
      only where the C grammar needs it.
- [x] **Emulator divergence.** `gpu.Ctx.SharedF32` matches calls across threads
      by execution order and only documents the requirement. Detect a mismatch
      and panic with a clear message instead of handing back the wrong buffer.
- [x] **No module teardown.** `cuModuleUnload` is never called; the process
      cache in `internal/jit` grows without bound.
- [x] Remove the unreachable `_` branch under `token.DEFINE` in `simt/stmt.go`.

Done. Two changes are visible in the public API: `Transpile` now returns a
`*simt.Unit` carrying `Source`, `RequiredBlock` and `SharedBytes`, and kernels
that size shared memory against a fixed block declare it with
`ctx.AssumeBlockDim(n)` — `kernels.FIR` does. `Kernel.Launch` refuses a
mismatched block, and `Build` refuses a kernel whose shared memory exceeds the
device limit. Module lifetime moved onto `cuda.Context`, which unloads
everything it owns on `Close`.

Two defects not in the original list were found and fixed along the way:

- The JIT module cache was a package-level map keyed only on (source, arch),
  so a cache hit could return a module belonging to an already-released
  context. It is now per-context, which is the same fix as the teardown item.
- `jit.Load` reported empty `PTX`/`Log` on a cache hit, because the assignment
  sat inside the miss branch.

## Phase 1 — Foundations that cannot be retrofitted

### 1.1 Build-time errors, not run-time (M) — *the single biggest gap*

Today a kernel that cannot be transpiled compiles fine and fails when
`main()` runs. No Go developer will accept that.

- [ ] `gocuda vet`, a `golang.org/x/tools/go/analysis` Analyzer that flags
      unsupported constructs. Runs in CI, in editors, and under
      `go vet -vettool`.
- [ ] AOT pipeline: `go generate` → transpile → NVRTC → `go:embed` the PTX,
      with the existing JIT path kept as the fast development loop. One
      transpiler feeds both.
- [ ] **Exit criterion:** a kernel that cannot be lowered fails `go build`.

### 1.2 Toolchain discovery and portability (M)

- [ ] Replace the hard-coded paths in `cuda/driver_cuda.go` with `CUDA_PATH` /
      `pkg-config` / a documented override.
- [ ] Load `libcuda` and `libnvrtc` at run time (`dlopen`, or `purego` to drop
      cgo entirely) so a binary **builds and ships without CUDA installed** and
      degrades gracefully on a machine with no GPU. This is what makes the
      library distributable.
- [ ] Windows support; document macOS as unsupported (no CUDA).
- [ ] Keep the `cuda` build tag working as the no-GPU fallback.

### 1.3 Error model and API review (S)

- [ ] Typed errors with `errors.Is`/`As`, preserving `CUresult` codes.
- [ ] Source positions on every transpiler diagnostic (mostly done — audit for
      the paths that report without one).
- [ ] One deliberate pass over the exported surface: naming, `context.Context`
      support, what stays internal. Cheaper now than after users arrive.

### 1.4 CI on real hardware (M)

Nothing below is verifiable without this.

- [ ] GPU runner (self-hosted or cloud), matrix over CUDA 12.x/13.x ×
      `sm_75`/`86`/`89`/`90` × Linux/Windows × supported Go versions.
- [ ] Non-GPU job proving the no-tag build, `go vet`, and the race detector.

## Phase 2 — Language coverage (L)

Each item means: a spec entry, a golden test, and a CPU/GPU parity test.

- [ ] **Device functions.** A kernel cannot call another Go function. This
      blocks most real kernels; emit `__device__` functions or inline them.
- [ ] **Multi-dimensional indexing.** `ctx.GlobalID2()`, `ctx.ThreadIdx3()`;
      the host side already has `Dim3`.
- [ ] **Types.** `int32`/`uint32`/`int64` slices, opt-in `float64`, `bool`,
      structs of scalars → CUDA structs, fixed-size arrays.
- [ ] **Shared memory.** Typed (`SharedI32`, …) and dynamically sized at launch.
- [ ] **Atomics.** `atomicAdd`/`Min`/`Max`/`CAS`, with a Go-side vocabulary
      that the emulator implements faithfully.
- [ ] **Warp-level primitives.** Shuffle, ballot, `__activemask`, warp
      reductions — the basis of every fast reduction.
- [ ] **Missing statements.** `switch`, labelled `break`/`continue`,
      `for i, v := range`.
- [ ] **`const` / `__restrict__`.** Mark non-aliased read-only slice
      parameters; it is both a correctness contract and a real speed-up.
- [ ] **Opt-in fast math** and `#pragma unroll` hints for tap-style loops.

## Phase 3 — Correctness at scale (L)

A transpiler is trusted through evidence, not review.

- [ ] **Differential fuzzing.** Generate random programs in the supported
      subset, run them on the CPU emulator and the GPU, compare. This is the
      backbone of the whole approach and should run continuously.
- [ ] **`compute-sanitizer`** (`memcheck`, `racecheck`, `initcheck`,
      `synccheck`) over every kernel in CI.
- [ ] **Barrier-divergence analysis.** A `__syncthreads()` that only some
      threads of a block reach is undefined behaviour. Reject it statically.
- [ ] **Aliasing.** Detect, or explicitly document, two slice parameters bound
      to the same device buffer.
- [ ] **Debug mode.** Emit bounds checks, device `printf` and a trap, behind a
      flag — the closest thing to a panic the device can offer.
- [ ] **Numerical policy.** Write down `float32` semantics, FMA contraction and
      denormal handling, and set the tolerance policy tests use.
- [ ] **A written spec of the supported subset** — grammar and semantics, not
      prose in a README. This is the contract.

## Phase 4 — Host runtime for real workloads (M)

The measurement that should drive this: the FIR example spends **1.0 ms in the
kernel and 9.5 ms moving data**. Production speed lives here, not in kernel
micro-optimisation.

- [ ] Streams, events, async `HtoD`/`DtoH`; overlap copy with compute.
- [ ] Pinned (page-locked) host memory; optionally managed memory.
- [ ] **Device-resident values.** `tile.Materialize` always downloads; results
      must be able to stay on the GPU between operations.
- [ ] Block-size selection via `cuOccupancyMaxPotentialBlockSize` instead of a
      hard-coded 256.
- [ ] **Persistent on-disk kernel cache** keyed by (source, arch, NVRTC
      version) — removes the 25 ms compile from every process start.
- [ ] Multi-GPU; explicit context and stream ownership; goroutine-safety
      documented *and tested*. **The per-call `cuCtxSetCurrent` is broken, and
      this is now reproduced rather than suspected.** `Context.bind` makes the
      context current on the calling OS thread, but a goroutine may migrate to
      another thread between that cgo call and the next one, which then sees a
      thread with no current context. 64 goroutines each doing
      `Upload`/`Download` 40 times against one context fail with
      `CUDA_ERROR_INVALID_CONTEXT` within half a second, on `3d31a2e` and after
      Phase 0 alike — it is pre-existing, not a regression, and it also shows up
      as a rare spurious failure of `examples/tilefir`. Fixing it means either
      `runtime.LockOSThread` around every driver call, a
      `cuCtxPushCurrent`/`cuCtxPopCurrent` pair, or binding once per thread and
      proving it stays bound; all three change ownership semantics, so it
      belongs here rather than in Phase 0.
- [ ] Buffer pooling and leak detection; `cuModuleUnload` on teardown.
- [ ] CUDA graphs for launch-bound workloads (later, measure first).

## Phase 5 — Performance parity (M)

The bar: **within a stated percentage of hand-written CUDA C** for the same
algorithm. Until that is measured, "Go kernels" is a claim, not a result.

- [ ] Benchmark suite pairing every kernel with a hand-written `.cu` baseline
      compiled by `nvcc`.
- [ ] `__restrict__`, vectorised loads (`float4`), unrolling, shared-memory
      bank-conflict checks.
- [ ] Nsight Compute profiles per benchmark; emit NVTX ranges from the runtime
      so profiles are legible.
- [ ] A fast path for the CPU emulator — a goroutine per thread is right for
      tests, wrong for a CPU fallback.

## Phase 6 — Tile track maturity (L)

Seven operations, 1-D, `float32`, one windowed op, no reductions.

- [ ] **Reductions** (sum/min/max/argmax) with the shared-memory + warp pattern.
- [ ] **2-D tiles and matmul** — the canonical proof the model generalises, and
      the example `cutile` leads with.
- [ ] **Chained windowed operations** (multi-stage halos), currently rejected
      by design.
- [ ] Broadcasting, slicing, strided and gather access.
- [ ] Graph optimisation: common-subexpression elimination, constant folding,
      dead-node elimination, buffer reuse and memory planning.
- [ ] Device-resident tensors with explicit materialisation points.
- [ ] Generic element types; richer shape diagnostics.

## Phase 7 — Release engineering (S–M)

- [ ] LICENSE; confirm the redistribution position (`libnvrtc` is dynamically
      linked, not shipped — keep it that way).
- [ ] Semantic versioning, a v1 API freeze and a deprecation policy.
- [ ] `golangci-lint` + `gofumpt` as a CI gate.
- [ ] Godoc with runnable `Example` functions; the subset spec as the reference.
- [ ] CHANGELOG and release automation.
- [ ] A published support matrix: CUDA versions, architectures, OSes, Go
      versions.
- [ ] Repository scaffolding that is simply missing: there is no LICENSE, no
      `.golangci.yml` (Trunk runs `golangci-lint2` on defaults), and no CI at
      all. `.trunk/trunk.yaml` also pins `go@1.21.0` and `gofmt@1.20.4` against
      a `go 1.26` module, so the linter's Go runtime cannot parse the sources
      it is checking.

## What 1.0 means

- [ ] Builds, tests and ships on a machine with no CUDA installed.
- [ ] A kernel that cannot be lowered fails `go build`, never `main()`.
- [ ] GPU CI green across at least three architectures and two CUDA versions.
- [ ] Differential fuzzer clean over a sustained run.
- [ ] `compute-sanitizer` clean on every kernel.
- [ ] Every benchmark within a stated margin of its hand-written CUDA baseline.
- [ ] Public API frozen; subset spec published.

## Non-goals

Naming these keeps the scope honest:

- Full Go on the device — garbage collection, interfaces, goroutines, panics,
  slices of slices, maps, strings.
- Replacing cuBLAS, cuFFT or cuDNN. Call them instead.
- Non-NVIDIA backends. If portability becomes the goal, Cogent Core's `gosl`
  already translates Go to WGSL and that is the better starting point.
- A deep-learning framework.

## Sequencing

| Phase | Size | Depends on | Main risk |
|---|---|---|---|
| 0 Known bugs | S | — | none |
| 1.1 Build-time errors | M | — | changes the public API, so do it early |
| 1.2 Portability | M | — | `dlopen`/`purego` rework of all cgo |
| 1.3 Error model & API | S | 1.1 | cheap now, expensive after users |
| 1.4 GPU CI | M | — | hardware access and cost |
| 2 Language coverage | L | 1.1, 1.4 | device functions touch the whole emitter |
| 3 Correctness | L | 1.4, 2 | fuzzer findings may force emitter redesign |
| 4 Host runtime | M | 1.3 | streams change ownership semantics |
| 5 Performance | M | 2, 4 | may expose NVRTC as the ceiling → revisit the PTX decision |
| 6 Tile maturity | L | 2, 4 | reductions and 2-D tiling are a rewrite of the code generator |
| 7 Release | S–M | all | — |

The critical path is **1.1 → 1.4 → 2 → 3**. Phases 4 and 6 can run in
parallel once the foundations hold.
