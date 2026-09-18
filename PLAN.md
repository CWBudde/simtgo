# Roadmap

This plan turns a working library into one other people can depend on. It is
ordered by dependency, not by appeal: the early phases are the ones that cannot
be retrofitted later.

Sizes are rough — **S** ≈ days, **M** ≈ 1–3 weeks, **L** ≈ 1–2 months of focused
work.

**What belongs here:** open work, and a short note per phase saying what the
done work changed. **What does not:** the findings themselves. Those are in
[`docs/`](docs/) — the defect post-mortems in
[`docs/emitter-defects.md`](docs/emitter-defects.md), the measurements in
[`docs/toolchain.md`](docs/toolchain.md) and
[`docs/verification.md`](docs/verification.md), the reasoning in
[`docs/decisions.md`](docs/decisions.md).

## Where this stands

Both tracks work and are verified against CPU references on real hardware.
Roughly 11,500 lines of library and tool, 7,100 of fuzzing infrastructure and
11,400 of tests; twelve kernels, each with a golden file, an NVRTC compile test
and a CPU/GPU parity test.

| Verified                                                     | Missing                                         |
| ------------------------------------------------------------ | ----------------------------------------------- |
| Both tracks, against independent Go references               | on **one** GPU: T550, `sm_75`, CUDA 12.8, Linux |
| The subset, as a contract checked in both directions         | —                                               |
| A differential fuzzer over three oracles, two daily in CI    | the device oracle; a GPU runner to host it      |
| `compute-sanitizer` clean on all four tools                  | it runs by hand, not in CI                      |
| A kernel that cannot be lowered fails `go build`             | —                                               |
| Builds and ships with no CUDA installed, no cgo anywhere     | Windows                                         |
| 1-D/2-D/3-D grids, structs, arrays, atomics, warp vocabulary | fast math, tuple returns                        |
| Driver API: 18 calls, fully synchronous                      | streams, events, async copies, pinned memory    |
| Tile track: 7 ops, 1-D `float32`, one windowed               | reductions, 2-D, fusion planning                |

The single largest gap is **a GPU in CI** (Phase 1.4). It blocks the fuzzer's
device leg, the sanitizer workflow, every parity test written since 2026-09-18,
and half of what 1.0 means.

## The decision that gates everything else

**Stay on CUDA C through 1.0** rather than emitting PTX or LLVM IR directly.
The reasoning and the comparison table are in
[`docs/decisions.md`](docs/decisions.md#cuda-c-through-nvrtc-not-ptx-or-llvm-ir).
What is left is to keep the option honest rather than architectural taste:

- [ ] **(S)** Spike: emit PTX directly for `VecAdd`, measure against the NVRTC
      path. Evidence, not conviction.
- [ ] **(S)** Decide the AOT story's public API shape before Phase 5 can force
      the question.

## Phase 0 — Known defects (S) — done 2026-09-17

Eight verified defects, all fixed, plus two found along the way. Two changes
reached the public API: `Transpile` returns a `*simt.Unit` carrying `Source`,
`RequiredBlock` and `SharedBytes`, and a kernel that sizes shared memory against
a fixed block declares it with `ctx.AssumeBlockDim(n)`. Module lifetime moved
onto `cuda.Context`, which unloads everything it owns on `Close` — the same fix
as the per-context JIT cache, since a module dies with its context.

Post-mortems: [`docs/emitter-defects.md`](docs/emitter-defects.md#the-phase-0-eight).

## Phase 1 — Foundations that cannot be retrofitted

### 1.1 Build-time errors, not run-time (M) — done 2026-09-18

The single biggest gap at the time: a kernel that could not be transpiled
compiled fine and failed when `main()` ran.

The lowering moved to `internal/lower` so that the vet tool and the generator
need no CUDA toolchain, `Transpile` reports every refused construct rather than
the first, and `simt.Build` consults a registry of prebuilt PTX keyed on the
hash of the **generated CUDA C** — which makes a stale artifact impossible to
use rather than merely detectable. It cost 28.7 ms of start-up, which is now
0.9 ms.

- [x] `gocuda vet`, a `go/analysis` Analyzer running the same lowering the
      transpiler does. _Not_ in editors: `gopls` runs a fixed, compiled-in set
      of analyzers, so the editor story is a `golangci-lint` module plugin or an
      on-save task.
- [x] AOT pipeline: `go generate` → transpile → NVRTC → embedded PTX, with the
      JIT path kept as the fast development loop.
- [x] **Exit criterion: a kernel that cannot be lowered fails `go build`.**

That criterion needs three mechanisms, not one, and this is the part somebody
must not undo — `go build` cannot be extended with custom vet checks, and
`go test`'s implicit vet runs a fixed subset that ignores `-vettool`:

1. `go generate` drops the constant for a kernel it could not lower, and the
   hand-written `kernels/prebuilt/gate.go` referencing it stops compiling:
   `gate.go:22:2: undefined: Scale`.
2. That only catches a kernel that _was_ regenerated. One edited and never
   regenerated keeps its constant and builds green, so `simt.VerifyPrebuilt` in
   an ordinary `go test ./...` catches the mismatch. No GPU needed; the cheapest
   and strongest of the three.
3. `gocuda vet` catches it without regenerating at all.

### 1.2 Toolchain discovery and portability (M) — 4 of 5

The question moved from build time to run time, so the answer moved with it.
There is no cgo left anywhere: `libcuda` and `libnvrtc` are `dlopen`ed through
purego, the two load independently — which is what lets a prebuilt kernel run
with no toolkit present — and `CGO_ENABLED=0 go build -tags cuda ./...` passes.
The search policy lives in untagged, loading-free Go so it is testable on a
machine with no CUDA.

- [x] Run-time loading via purego, replacing the hard-coded paths. (2026-09-17)
- [x] `cuda/library.go` decides which file to open; `pkg-config` deliberately
      not used. (2026-09-17)
- [x] The `cuda` build tag kept as the no-GPU fallback, and now enforced by
      `cuda/surface_test.go`. (2026-09-17)
- [x] Collapse `cmd/gocuda-nvrtc` into `cmd/gocuda`. (2026-09-18)

Findings: the `_v2` symbol trap and the search policy are in
[`docs/toolchain.md`](docs/toolchain.md).

- [ ] **Windows support; document macOS as unsupported.** Unblocked — there is
      no C toolchain in the way any more — but it cannot be _claimed_ without a
      Windows host or 1.4's CI, so it is written as work rather than as a
      footnote.
  - [ ] `cuda/library.go`: `nvcuda.dll` for the driver, `nvrtc64_*.dll` for
        NVRTC, and the Windows search order (`System32`, then `%CUDA_PATH%\bin`).
        The file is untagged and loading-free by design, so this is testable on
        Linux exactly as `cuda/library_test.go` already tests the POSIX order.
  - [ ] Confirm purego on Windows: it wraps `LoadLibrary`, so check whether the
        `Dlopen` spelling in `cuda/loader_cuda.go` carries over or needs a
        `_windows.go` half.
  - [ ] Nothing in the emitter changes — the 64-bit types are already
        `long long` and never `long`, which is 4 bytes there — but add a Windows
        leg to `TestStructLayoutRoundTrip` to hold that rather than assume it.
  - [ ] A `windows-latest` CI leg that needs no GPU: `go build`,
        `go test ./...`, `CGO_ENABLED=0 go build -tags cuda ./...`.
  - [ ] `README.md`: state macOS as unsupported, since NVIDIA ships no CUDA for
        it.

### 1.3 Error model and API review (S) — done 2026-09-18

- [x] Typed errors with `errors.Is`/`As`, preserving `CUresult` codes. The name
      table is pure Go, so a code still prints itself in a build without the
      `cuda` tag.
- [x] Source positions on every transpiler diagnostic. The audit found the 49
      refusal sites were all fine and the gaps were all in `Transpile` itself:
      `go/types` stopped at the first error unless given an `Error` callback,
      and the import refusal had no position of its own.
- [x] One deliberate pass over the exported surface. `context.Context` is
      deliberately **not** added anywhere until Phase 4 —
      [why](docs/decisions.md#the-driver-api-is-synchronous-so-there-is-no-contextcontext).

### 1.4 CI on real hardware (M) — 1 of 2

**Nothing below this line is verifiable without it.** This is the critical
blocker in the whole plan.

- [x] Non-GPU job proving the no-tag build, `go vet` under both tags, the race
      detector, `CGO_ENABLED=0 -tags cuda`, `gocuda generate -check`,
      `golangci-lint` and `treefmt --ci`. (2026-09-18) —
      `.github/workflows/ci.yml`.
- [ ] **A GPU runner.** It gates the sanitizer workflow, the fuzzer's device
      leg, every parity test written since 2026-09-18, and half of _What 1.0
      means_.
  - [ ] Decide self-hosted (the T550 host) versus a cloud GPU runner, and write
        the decision down. Cost and who holds the token are the deciding
        factors, not capability.
  - [ ] `.github/workflows/gpu.yml`: `go test -tags cuda ./...` with
        `GOCUDA_REQUIRE_DEVICE=1`, so a runner that lost its device fails
        instead of going green having launched nothing.
  - [ ] Flip `.github/workflows/sanitizer.yml` from `workflow_dispatch`-only to
        a schedule. The comment at its head already says this is what it waits
        for.
  - [ ] Add the fuzzer's device leg to `.github/workflows/fuzz.yml` (Phase 3).
  - [ ] **Then** the matrix: CUDA 12.x/13.x × `sm_75`/`86`/`89`/`90` ×
        Linux/Windows × supported Go versions. One green runner first — a matrix
        over a runner that does not exist is a queue.

## Phase 2 — Language coverage (L) — 14 of 16

The subset went from `float32`-and-`int32` scalars to structs, arrays, six
shared-tile types, a dynamic tile, atomics, the warp vocabulary, device
functions, three-axis grids and `const`/`__restrict__` with the aliasing promise
checked rather than assumed. Narrow integers are storage.

**Three of the items began as defects rather than features** — a labelled
`break` that left the wrong loop, an unnamed parameter that shifted every
argument after it, `[]int` read at the wrong stride — which is why Phase 3's
fuzzer exists. Each item means a spec entry, a golden test and a CPU/GPU parity
test.

- [x] **Device functions** — emitted, not inlined; recursion refused. (2026-09-17)
- [x] **Multi-dimensional indexing** — per-axis accessors, not tuples;
      `gpu.Dim`, `RunCPUDim`, `Kernel.LaunchDim`. (2026-09-17)
- [x] **Missing statements** — `switch` with two lowerings, labelled
      `break`/`continue` via `goto`, `for i, v := range`. (2026-09-17)
- [x] **Types** — `int32`/`uint32`/`int64`/`uint64`, opt-in `float64`, `bool`,
      structs, fixed-size arrays. (2026-09-18)
- [x] **Double-precision `gpu` math** — `Sqrt64` and the rest, gated on
      `//gocuda:float64` at the **call**. (2026-09-18)
- [x] **Narrow integer storage** — `[]uint8` and friends, accepted in
      `ctypeElem` and nowhere else. (2026-09-18)
- [x] **Array-typed struct fields**, with every hole declared as a
      `gocuda_padN` member. (2026-09-18)
- [x] **Shared memory** — six typed constructors plus one dynamic tile.
      (2026-09-18)
- [x] **Atomics** — six, taking a buffer and an index. (2026-09-18)
- [x] **Warp primitives** — shuffles, ballot, `__activemask`, as `Ctx` methods
      with no mask argument. (2026-09-18)
- [x] **Multiple assignment**, parallel form, through one temporary per value.
      (2026-09-18)
- [x] **`const` / `__restrict__`**, proved across the call graph and checked at
      launch. (2026-09-18)
- [x] **A `//gocuda:device` marker**, refused on a function taking no `gpu.Ctx`
      so it cannot become decoration. (2026-09-18)
- [x] **Shared memory in a device function** — block-scoped storage, accounted
      onto the kernel that reaches it. (2026-09-18)

The shapes of those decisions are in
[`docs/decisions.md`](docs/decisions.md); the defects they turned up are in
[`docs/emitter-defects.md`](docs/emitter-defects.md).

Two remain.

- [ ] **Opt-in fast math, and `#pragma unroll` hints for tap-style loops.**
  - [ ] `//gocuda:fastmath`, read exactly as `//gocuda:float64` is:
        translation-unit scoped, refused on a device function for the same
        reason.
  - [ ] `cuda.Compile` gains the option. Today it passes only
        `--gpu-architecture`, which is why every numerical setting is a default
        nobody chose — see
        [`docs/toolchain.md`](docs/toolchain.md#the-compile-options-are-one-option).
  - [ ] **`Unit.SourceHash` must cover the flag.** Otherwise a prebuilt compiled
        without it is found and used for a kernel that asked for it, and the
        whole staleness story depends on that hash meaning "these exact bytes,
        compiled this exact way".
  - [ ] `NUMERICS.md` gains a section. `--use_fast_math` implies `--ftz=true`,
        `--prec-div=false`, `--prec-sqrt=false` and `--fmad=true` — four of the
        measured defaults reversed at once — so the tolerance policy has to say
        what a fast-math kernel may still be asserted to.
  - [ ] `#pragma unroll` separately and smaller: a directive on a tap-style
        loop. Measure before claiming; NVRTC already unrolls these.
- [ ] **Tuple assignment from a call** — `a, b := f()`. The correction worth
      keeping: this is neither necessary nor sufficient for `ctx.GlobalID2()`
      ([why](docs/decisions.md#per-axis-accessors-not-globalid2)), so it splits
      in two.
  - [ ] Out-parameters in the generated C for a multi-result device function.
        This is the ABI work and the whole cost of the item.
  - [ ] Separately, a destructuring special case in `assign`, if a
        pair-returning intrinsic is ever wanted. `assign` already sees the
        statement before `expr` runs, which is where `ctx.SharedF32(n)` is
        special-cased.

## Phase 3 — Correctness at scale (L) — 4 of 8

A transpiler is trusted through evidence, not review. This phase built the
evidence: a written contract checked against the implementation in both
directions, a numerical policy where every claim is labelled measured, cited or
unverified, a barrier-divergence analysis, and a differential fuzzer that has
found more defects than every other layer combined — including one, the
`min`/`max` NaN disagreement, that had been recorded as untested for weeks and
that it reached in **six seconds**.

The machinery is described in
[`docs/verification.md`](docs/verification.md).

- [x] **A written spec of the supported subset.** (2026-09-18) — `SPEC.md`, with
      `simt/spec_test.go` failing in both directions.
- [x] **Numerical policy.** (2026-09-18) — `NUMERICS.md`, plus one tolerance
      rule in `internal/tolerance` where there were three conventions in three
      packages.
- [x] **Barrier-divergence analysis.** (2026-09-18) —
      `internal/lower/diverge.go`, interprocedural, three rules. It found two
      live bugs, and its one honest limit — the rules do not see into a
      condition, so a short-circuited `&&` can still break the warp
      participation promise — is written down at the site.
- [x] **Aliasing.** (2026-09-19) — the SIMT track refuses an overlap at launch
      and the intra-kernel case at lowering; the tile track now makes the same
      promise and checks it in `Tensor.run`. The emulator half is closed by the
      item's own "or": `RunCPU`'s doc comment states that it never sees the
      caller's slices and that this is deliberate, because there a kernel is
      ordinary Go, where aliasing is defined.

- [ ] **Differential fuzzing.** The generator, the corpus and three oracles
      exist; `.github/workflows/fuzz.yml` runs daily with one matrix leg per
      target. The box stays open because the device oracle — the one the item's
      own definition names — is the one that is missing.
  - [ ] The device leg: run the generated program on the GPU and compare against
        the emulator. Blocked on 1.4.
  - [ ] The NVRTC oracle in CI, which needs a toolkit on the runner rather than
        a device, and would catch every struct-layout disagreement on every push.
  - [ ] Raise the daily budget above 20m, or add a weekly leg that runs longer.
        Every defect so far was found in minutes, so the question the budget
        answers is whether a deeper one exists — and a 20m search cannot say.
  - [ ] Raise warp-primitive coverage from 7.5%, the lowest figure in the
        table and the least-exercised corner of the subset.

- [ ] **Signed zero, host against emulator.** The one open fuzzer finding.
      `fuzz.Generate(620)` with inputs `77` writes `+0` on the host where the
      emulator writes `-0`, and it reproduces on the commit before the
      signed-overflow fix, so that is not the cause.
  - [ ] Reproduce and minimise the case.
  - [ ] Establish which side is right, and whether it is a constant-folding
        difference — Go folds a float constant expression at arbitrary precision
        and rounds once — or something in the translation.
  - [ ] Decide whether `internal/tolerance` keeps treating the two zeros as
        different results. It deliberately does today: a relative bound cannot
        tell them apart, their difference being zero while their bits are not.
  - [ ] Record the answer in `NUMERICS.md` and commit the corpus entry, which is
        withheld today because an entry is a test and this one fails.

- [ ] **`compute-sanitizer` in CI.** The sweep exists, covers all twelve kernels
      plus the tile track's fused ones, and is clean on `memcheck`, `racecheck`,
      `initcheck` and `synccheck`; it was held to a planted fault before the zero
      was believed. The criterion says **in CI**, and CI has no device.
  - [ ] Flip `sanitizer.yml` to a schedule once 1.4 lands. That is the whole
        remaining item.

- [ ] **Debug mode** — the closest thing to a panic the device can offer, behind
      a flag.
  - [ ] Bounds checks on every slice and shared-tile index, behind a
        `simt.Build` option so the released path pays nothing.
  - [ ] Device `printf` in the `gpu` vocabulary. First answer the question the
        item rests on: does NVRTC declare it with no header? Measure it the way
        every other built-in in
        [`docs/toolchain.md`](docs/toolchain.md#what-nvrtc-declares-with-no-header-included)
        was measured.
  - [ ] `__trap()`, and what the host sees afterwards — the context is unusable,
        so say so in the error rather than letting the next call fail obscurely.
  - [ ] The interaction with `SourceHash` and the prebuilt registry, the same
        problem fast math has: a debug build must not find a release artifact.

## Phase 4 — Host runtime for real workloads (M)

The measurement that should drive this: the FIR example spends **1.0 ms in the
kernel and 9.5 ms moving data**. Speed lives here, not in kernel
micro-optimisation — and the `runtime.Pinner` work already confirmed that the
launch path is not where the time goes.

- [ ] Streams, events, async `HtoD`/`DtoH`; overlap copy with compute. This is
      also what makes `context.Context` meaningful, since `cuStreamQuery` can
      genuinely `select` on `ctx.Done()`.
- [ ] Pinned (page-locked) host memory; optionally managed memory.
- [ ] **Device-resident values.** `tile.Materialize` always downloads; results
      must be able to stay on the GPU between operations. This is also what
      first makes the tile track's aliasing check able to fail — see
      [`docs/tile.md`](docs/tile.md#aliasing-the-promise-is-made-then-checked).
- [ ] Block-size selection via `cuOccupancyMaxPotentialBlockSize` instead of a
      hard-coded 256.
- [ ] **Persistent on-disk kernel cache** keyed by (source, arch, NVRTC
      version) — removes the compile from every process start for kernels with
      no prebuilt artifact.
- [ ] **Multi-GPU; explicit context and stream ownership; goroutine-safety
      documented _and tested_.** The per-call `cuCtxSetCurrent` is broken, and
      this is reproduced rather than suspected: `Context.bind` makes the context
      current on the calling OS thread, but a goroutine may migrate to another
      thread between that call and the next, which then sees a thread with no
      current context.
  - [ ] Choose the fix: `runtime.LockOSThread` around every driver call, a
        `cuCtxPushCurrent`/`cuCtxPopCurrent` pair, or binding once per thread and
        proving it stays bound. All three change ownership semantics, which is
        why this belongs here and not in Phase 0.
  - [ ] A regression test that fails today. 64 goroutines each doing
        `Upload`/`Download` 40 times against one context fail with
        `CUDA_ERROR_INVALID_CONTEXT` within half a second, on `3d31a2e` and
        after Phase 0 alike — pre-existing, not a regression — and the same bug
        shows up as a rare spurious failure of `examples/tilefir`.
  - [ ] Document the resulting guarantee on `cuda.Context`: what may be shared
        between goroutines and what may not.

- [ ] Buffer pooling and leak detection.
- [ ] CUDA graphs for launch-bound workloads (later, measure first).

## Phase 5 — Performance parity (M)

The bar: **within a stated percentage of hand-written CUDA C** for the same
algorithm. Until that is measured, "Go kernels" is a claim, not a result.

- [ ] Benchmark suite pairing every kernel with a hand-written `.cu` baseline
      compiled by `nvcc`.
- [ ] `__restrict__` (done), vectorised loads (`float4`), unrolling,
      shared-memory bank-conflict checks.
- [ ] Nsight Compute profiles per benchmark; emit NVTX ranges from the runtime
      so profiles are legible.
- [ ] A fast path for the CPU emulator — a goroutine per thread is right for
      tests and wrong for a CPU fallback.

## Phase 6 — Tile track maturity (L)

Seven operations, 1-D, `float32`, one windowed op, no reductions. What the
generator does today and where it stops is in [`docs/tile.md`](docs/tile.md);
reductions and 2-D tiling are a rewrite of it rather than an extension.

- [ ] **Reductions** (sum/min/max/argmax) with the shared-memory + warp pattern.
      The pieces exist: `kernels.WarpReduceSum` is a kernel, and the warp
      vocabulary it needs landed in Phase 2.
- [ ] **2-D tiles and matmul** — the canonical proof the model generalises, and
      the example `cutile` leads with.
- [ ] **Chained windowed operations** (multi-stage halos), currently rejected by
      design.
- [ ] Broadcasting, slicing, strided and gather access.
- [ ] Graph optimisation: common-subexpression elimination, constant folding,
      dead-node elimination, buffer reuse and memory planning.
- [ ] Device-resident tensors with explicit materialisation points (Phase 4).
- [ ] Generic element types; richer shape diagnostics.

## Phase 7 — Release engineering (S–M) — 4 of 9

- [x] LICENSE and the redistribution position. (2026-09-18) — MIT; `libnvrtc` is
      `dlopen`'d and never shipped.
      [Why MIT](docs/decisions.md#mit-and-libnvrtc-is-never-shipped).
- [x] Repository scaffolding: `.golangci.yml`, `treefmt.toml`, CI. (2026-09-18)
      — [what was left off and why](docs/decisions.md#linting-is-golangci-lint-and-treefmt-and-the-seven-deferred-linters-are-on).
- [x] The seven linters left off rather than suppressed — `errorlint`,
      `perfsprint`, `predeclared`, `gocritic`, `intrange`, `wastedassign`,
      `revive`. (2026-09-19) — all seven on, one commit each; 71 findings across
      both build configurations, of which one was a latent panic in
      `simt/transpile_test.go` and one a fuzz loop that re-rolled its own bound.
      Two excluded rather than fixed, each argued in `.golangci.yml`:
      `ifElseChain` in `kernels/` (the rewrite would move every `SourceHash`)
      and `unused-parameter` in `cuda/stub.go` (the names are that build's
      godoc).
      [What the seven found](docs/decisions.md#linting-is-golangci-lint-and-treefmt-and-the-seven-deferred-linters-are-on).
- [x] A `justfile` for the local checks. (2026-09-19) — `just check` is the
      non-GPU CI sequence in CI's order; `just lint-cuda` is the tagged half
      nothing else lints. It mirrors the workflow rather than being called by
      it, so no runner needs `just`. `sanitize` ships unexercised: no device
      here.
- [ ] Lint the `cuda`-tagged half in CI. Three of the seven linters' findings
      were visible only under `--build-tags cuda`, and two `errcheck` findings
      in `cuda/driver_cuda.go` are open there now. A second `golangci-lint` job
      with the tag would close the gap the config's own header admits.
- [ ] Semantic versioning, a v1 API freeze and a deprecation policy.
- [ ] Godoc with runnable `Example` functions; `SPEC.md` as the reference.
- [ ] CHANGELOG and release automation.
- [ ] A published support matrix: CUDA versions, architectures, OSes, Go
      versions. Blocked on 1.4 — a matrix nothing tested is a promise.

## What 1.0 means

- [x] Builds, tests and ships on a machine with no CUDA installed.
- [x] A kernel that cannot be lowered fails `go build`, never `main()`.
- [ ] GPU CI green across at least three architectures and two CUDA versions.
- [ ] Differential fuzzer clean over a sustained run, on the device as well as
      the host.
- [ ] `compute-sanitizer` clean on every kernel, in CI rather than by hand.
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

| Phase                 | Size | State    | Depends on | Main risk                                                     |
| --------------------- | ---- | -------- | ---------- | ------------------------------------------------------------- |
| 0 Known bugs          | S    | done     | —          | —                                                             |
| 1.1 Build-time errors | M    | done     | —          | —                                                             |
| 1.2 Portability       | M    | 4 of 5   | —          | Windows cannot be claimed without a host to claim it on       |
| 1.3 Error model & API | S    | done     | 1.1        | —                                                             |
| 1.4 GPU CI            | M    | 1 of 2   | —          | hardware access and cost                                      |
| 2 Language coverage   | L    | 14 of 16 | 1.1, 1.4   | fast math changes what a prebuilt artifact means              |
| 3 Correctness         | L    | 4 of 8   | 1.4, 2     | the open oracles all need a device                            |
| 4 Host runtime        | M    | 0 of 8   | 1.3        | streams change ownership semantics; the context bug is live   |
| 5 Performance         | M    | 0 of 4   | 2, 4       | may expose NVRTC as the ceiling → revisit the PTX decision    |
| 6 Tile maturity       | L    | 0 of 7   | 2, 4       | reductions and 2-D tiling are a rewrite of the code generator |
| 7 Release             | S–M  | 4 of 9   | all        | —                                                             |

The critical path is **1.4 → 3 → 5**. Phases 4 and 6 can run in parallel; 1.2's
Windows leg is independent of everything else.
