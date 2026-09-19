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

| Verified                                                                 | Missing                                         |
| ------------------------------------------------------------------------ | ----------------------------------------------- |
| Both tracks, against independent Go references                           | on **one** GPU: T550, `sm_75`, CUDA 12.8, Linux |
| The subset, as a contract checked in both directions                     | —                                               |
| A differential fuzzer: the host and NVRTC oracles in CI daily, 4h weekly | the device oracle; a GPU runner to host it      |
| `compute-sanitizer` clean on all four tools                              | it runs by hand, not in CI                      |
| A kernel that cannot be lowered fails `go build`                         | —                                               |
| Builds and ships with no CUDA installed, no cgo anywhere                 | Windows                                         |
| 1-D/2-D/3-D grids, structs, arrays, atomics, warp vocabulary             | tuple returns                                   |
| Opt-in fast math, with the hash split that keeps it honest               | a timing harness to say it is faster            |
| Driver API: 31 calls, streams, events, async copies, page-locked memory  | managed memory, buffer pooling, multi-GPU       |
| Tile track: 7 ops, 1-D `float32`, one windowed                           | reductions, 2-D, fusion planning                |

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
- [x] One deliberate pass over the exported surface. `context.Context` was
      deliberately **not** added anywhere until Phase 4, and arrived there on
      `cuda.Stream.Wait` —
      [why, and what a cancelled wait does not do](docs/decisions.md#contextcontext-cancels-the-wait-and-says-so).

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
  - [x] `//gocuda:fastmath`, read exactly as `//gocuda:float64` is:
        translation-unit scoped, refused on a device function for the same
        reason. (2026-09-19) — and refused on a helper for a second reason of
        its own: NVRTC takes the option for a compilation, not for a function.
  - [x] `cuda.Compile` gains the option. (2026-09-19) — variadic
        `CompileOption`s, built in `nvrtcOptions` in the untagged half so the
        command line is pinned by a test on a machine with no CUDA. See
        [`docs/toolchain.md`](docs/toolchain.md#the-compile-options-are-one-option-plus-the-one-a-kernel-asks-for).
  - [x] **`Unit.SourceHash` must cover the flag.** (2026-09-19) — by emitting a
        `// gocuda: fastmath` marker into the generated C rather than salting
        the hash, so `SourceHash` still means "these exact bytes" and
        `internal/jit`'s cache key inherits the split for free.
  - [x] `NUMERICS.md` gains a section. (2026-09-19) — _Fast math, when it is
        asked for_: the one-variable PTX experiment, the measured 1 ulp on a
        T550, and the tolerance a fast-math kernel may be held to. It also
        corrected three claims the new artifact falsified.
  - [ ] `#pragma unroll` separately and smaller: a directive on a tap-style
        loop. Measure before claiming; NVRTC already unrolls these.
        (2026-09-19) — partial: measured, and the answer is that the emitter
        should **not** write one. The pragma leaves the PTX alone but adds a
        `.pragma "nounroll"` that stops `ptxas` unrolling, taking `FIR`'s tap
        loop from 29 `FFMA` in SASS to 5.
        [What was measured](docs/toolchain.md#pragma-unroll-on-the-fir-tap-loop-makes-it-slower).
        Closing the box is a decision, not a measurement: it needs either
        agreement that "no directive" is the answer, or Phase 5's timing
        harness to say something instruction counts cannot.
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

## Phase 3 — Correctness at scale (L) — 5 of 8

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

- [x] **The NVRTC oracle's noise filter.** (2026-09-19) — it discarded a
      finding whenever an allowlisted warning preceded it, and its allowlist is
      keyed on the diagnostic number, which is the same for a dropped use and
      for generated filler. The first is fixed, the second cannot be fixed
      where it lives, and an opt-in second look recovers part of the class:
      [`docs/verification.md`](docs/verification.md#10-nvrtc-warning-triage-opt-in),
      [`docs/emitter-defects.md`](docs/emitter-defects.md#nvrtcs-remark-swallowed-the-warning-behind-it),
      [`docs/decisions.md`](docs/decisions.md#a-judgment-may-add-a-finding-and-may-never-remove-one).

- [ ] **Differential fuzzing.** The generator, the corpus and three oracles
      exist, and two of them — the host and NVRTC oracles — now run in
      `.github/workflows/fuzz.yml`. The box stays open because the device
      oracle, the one the item's own definition names, is the third.
  - [ ] The device leg: run the generated program on the GPU and compare against
        the emulator. Blocked on 1.4.
  - [x] The NVRTC oracle in CI, which needs a toolkit on the runner rather than
        a device, and would catch every struct-layout disagreement on every push.
        (2026-09-19) — a job of its own installing the `nvidia-cuda-nvrtc-cu12`
        wheel and pointing `GOCUDA_LIBNVRTC` at it; no device, no `nvcc`. It
        sets the new `GOCUDA_REQUIRE_NVRTC`, because the skip it otherwise takes
        is indistinguishable from a clean search — measured, a leg with the
        library missing passes in **three milliseconds** having compiled nothing.
  - [ ] Raise the daily budget above 20m, or add a weekly leg that runs longer.
        Every defect so far was found in minutes, so the question the budget
        answers is whether a deeper one exists — and a 20m search cannot say.
        (2026-09-19) — partial: the weekly leg is written and `actionlint`-clean
        — a second cron at `17 2 * * 0` and a `FUZZTIME` of `4h` — but no
        command run here proves a schedule fires. The first weekly run is the
        evidence, and this box is what is waiting for it.
  - [x] Raise warp-primitive coverage from 7.5%, the lowest figure in the
        table and the least-exercised corner of the subset. (2026-09-19) —
        **10.7% → 19.8%**, and measured rather than asserted for the first time:
        `TestFeatureCoverage` prints the whole table and holds each row to a
        floor. None of the seven gates on a warp expression moved — each guards
        real undefined behaviour — so the draws upstream of them did: a
        statement slot of warp's own, a second slot in each expression switch,
        and ten of sixteen whole-warp block widths. Yield still 1.0.
        [The ceiling above it](docs/verification.md#warp-primitives-and-the-ceiling-over-them),
        and why raising `g.sync` is the wrong next lever.

- [x] **Signed zero, host against emulator.** (2026-09-19) — closed by
      `53ab673` on 2026-09-18, and ticked here only now: the box outlived the
      fix, and this plan and `docs/emitter-defects.md` both went on describing
      an open finding for a day after it was answered. The answer is that
      neither backend was wrong. `fmin`/`fmax` of two zeros of opposite signs
      are unspecified in C and IEEE 754 alike, and the host does not answer it
      stably — −0 at `-O1`, +0 at `-O0` — so `hostrun`'s shim pins the four
      functions on the two zeros to the device's answer, which is not open.
      `TestFminFmaxAgreeOnTheZeros` compares the bits and pins the direction.
      [The post-mortem](docs/emitter-defects.md#signed-zero-host-against-emulator--and-it-was-neither-backends-fault).
  - [x] Reproduce and minimise the case.
  - [x] Establish which side is right, and whether it is a constant-folding
        difference — Go folds a float constant expression at arbitrary precision
        and rounds once — or something in the translation. It was neither: the
        operation is unspecified, and the host compiler picks by optimisation
        level.
  - [x] Decide whether `internal/tolerance` keeps treating the two zeros as
        different results. It does, unchanged: a relative bound cannot tell them
        apart, their difference being zero while their bits are not, and the
        finding was a reason to pin the oracle rather than to loosen the rule.
  - [x] Record the answer in `NUMERICS.md` and commit the corpus entry. Both
        done in `53ab673`. The entry is a **seed**, though, not a program, so it
        tracks the generator rather than the case — raising the warp coverage
        above moved it from skipping to passing without either being about the
        zeros. [Why that is not what holds the case down](docs/verification.md#what-a-corpus-entry-does-not-pin).

- [ ] **`compute-sanitizer` in CI.** The sweep exists, covers all twelve kernels
      plus the tile track's fused ones, and is clean on `memcheck`, `racecheck`,
      `initcheck` and `synccheck`; it was held to a planted fault before the zero
      was believed. The criterion says **in CI**, and CI has no device.
  - [ ] Flip `sanitizer.yml` to a schedule once 1.4 lands. That is the whole
        remaining item.

- [ ] **Debug mode** — the closest thing to a panic the device can offer, behind
      a flag. Three of four; `simt.WithBoundsChecks()` is the flag, and the
      release path is byte for byte what it was.
      [Why a build option and not a directive](docs/decisions.md#bounds-checks-are-a-build-option-and-the-checks-are-their-own-marker),
      [what the layer sees and cannot see](docs/verification.md#9-bounds-checks-on-the-device-behind-a-flag).
  - [x] Bounds checks on every slice and shared-tile index, behind a
        `simt.Build` option so the released path pays nothing. (2026-09-19) —
        every subscript whose bound the emitter can name, through one
        `__device__` helper taking a `long long` so all four index types
        promote into it without a second overload. Two omissions are argued at
        the site: a `range` loop's induction variable is bounded by the same
        length the check would use, and a constant index into a fixed-size
        array has already been refused by `go/types`. A constant index into a
        slice is checked, a shared tile included — `tile[300]` on a
        64-element tile compiles in Go. The emulator half needed no code: a
        kernel under `RunCPU` is ordinary Go, and `gc`'s own check names the
        index, the length and the thread.
  - [ ] Device `printf` in the `gpu` vocabulary. (2026-09-19) — partial: the
        question the item rests on is answered, and the answer is **yes**.
        NVRTC declares `printf` with no header, the variadic form resolves,
        and `%f` with a runtime argument compiles
        ([the measurement](docs/toolchain.md#what-nvrtc-declares-with-no-header-included)).
        So this is no longer a capability question and is now a design one,
        which is where it stops: a fixed-arity `Ctx.PrintF32`/`PrintI32` with
        the tag folded into the format string at lowering would be a subset
        extension — a `gpu` entry, its hand-written twin in `gpupkg.go`, a new
        refusal for a non-constant tag, and a `SPEC.md` line — and whether a
        per-thread printf is wanted at all, given 256 interleaved lines a
        block and output that only appears at a synchronisation point, is not
        a measurement. Still unmeasured: whether a `printf` before a
        `__trap()` reaches stdout at all.
  - [x] `__trap()`, and what the host sees afterwards — the context is unusable,
        so say so in the error rather than letting the next call fail obscurely.
        (2026-09-19) — `cuda.ContextPoisonedError` from every later call, and
        `simt.TrapError` naming the kernel and pointing at `gpu.RunCPU`, which
        is the only place the index can still be had. The measurement was
        worse than the documentation implies and changed what the error says:
        a fresh context cannot be retained either, so the **process** is
        finished with CUDA, not just the context.
        [What was measured](docs/toolchain.md#what-the-host-sees-after-a-__trap).
  - [x] The interaction with `SourceHash` and the prebuilt registry, the same
        problem fast math had, and solved the same way: a debug build must not
        find a release artifact, so the flag belongs in the generated source
        where `SourceHash` already sees it. (2026-09-19) — the checks are in
        those bytes already; the marker line covers the kernel that indexes
        nothing, whose two builds would otherwise be identical.

## Phase 4 — Host runtime for real workloads (M) — 3 of 8

The measurement that should drive this: the FIR example spends **1.0 ms in the
kernel and 9.5 ms moving data**. Speed lives here, not in kernel
micro-optimisation — and the `runtime.Pinner` work already confirmed that the
launch path is not where the time goes. Streams are the first tool aimed at
it: they do not make a copy faster, they stop the copies being a queue.

- [x] Streams, events, async `HtoD`/`DtoH`; overlap copy with compute. This is
      also what makes `context.Context` meaningful, since `cuStreamQuery` can
      genuinely `select` on `ctx.Done()`. (2026-09-19) — `cuda.Stream` and
      `cuda.Event`, `Slice.UploadAsync`/`DownloadAsync`, `Function.Launch` and
      `simt.Kernel.LaunchOn`. 64 MiB up and back in sixteen chunks: 52.3 ms
      serial, 31.8 ms over four streams, **1.65×**. `Stream.Wait(ctx)` is the
      one entry point taking a `context.Context`; cancelling it stops the wait
      and not the device, so the stream is marked and `Close` then refuses
      with a `*BusyStreamError` rather than letting a `Free` land on memory
      the device is still reading.
      [What was measured](docs/toolchain.md#streams-overlap-the-copies-with-the-compute),
      [what a cancelled wait does not do](docs/decisions.md#contextcontext-cancels-the-wait-and-says-so).
- [x] Pinned (page-locked) host memory; optionally managed memory.
      (2026-09-19) — `cuda.HostSlice`, and the expectation it was taken on was
      wrong: page-locking is worth about **5%** on a synchronous copy here,
      not the textbook factor of two. It earns its place anyway, because an
      asynchronous copy is not available without it at all — `cuMemcpyHtoDAsync`
      falls back to a synchronous transfer for pageable memory, and Go's
      `runtime.Pinner` cannot hold a slice still past the end of the call. So
      `UploadAsync`/`DownloadAsync` take a `*HostSlice` and refuse a `[]T`.
      Managed memory, the item's optional half, was **not** done: nothing here
      needs it and unified memory is a separate ownership story.
      [The 5%](docs/toolchain.md#page-locked-host-memory-buys-little-bandwidth-here-and-is-still-required),
      [why a `[]T` is refused](docs/decisions.md#an-asynchronous-copy-does-not-take-a-go-slice).
- [ ] **Device-resident values.** `tile.Materialize` always downloads; results
      must be able to stay on the GPU between operations. This is also what
      first makes the tile track's aliasing check able to fail — see
      [`docs/tile.md`](docs/tile.md#aliasing-the-promise-is-made-then-checked).
- [ ] Block-size selection via `cuOccupancyMaxPotentialBlockSize` instead of a
      hard-coded 256.
- [x] **Persistent on-disk kernel cache** keyed by (source, arch, NVRTC
      version) — removes the compile from every process start for kernels with
      no prebuilt artifact. (2026-09-19) — `internal/jit/diskcache.go`, on by
      default, under the user cache directory; `simt.WithoutDiskCache()` turns
      it off and `GOCUDA_PTX_CACHE` moves it. 111.5 ms to 11.6 ms against
      10.4 ms for a prebuilt artifact, so a kernel with no artifact and a
      kernel with one now cost the same to within the noise. The tile track
      gains the most and this item did not say so: its fused kernels have no
      prebuilt path at all.
      [What was measured](docs/toolchain.md#the-on-disk-ptx-cache-brings-a-compiled-kernel-level-with-a-prebuilt-one),
      [why on by default and why only the location is an environment variable](docs/decisions.md#the-ptx-cache-is-on-by-default-and-only-its-location-is-an-environment-variable).
- [ ] **Multi-GPU; explicit context and stream ownership; goroutine-safety
      documented _and tested_.** (2026-09-19) — partial: goroutine safety is
      fixed, tested and written down; multi-GPU and stream ownership are not.
      Multi-GPU needs a second device, which this machine does not have, and
      stream ownership needs streams.
      [The finding](docs/decisions.md#a-driver-call-holds-its-os-thread).
  - [x] Choose the fix: `runtime.LockOSThread` around every driver call, a
        `cuCtxPushCurrent`/`cuCtxPopCurrent` pair, or binding once per thread and
        proving it stays bound. (2026-09-19) — the first, through one
        `Context.call` that locks the thread around the bind and the driver
        call together and takes the entry point as a closure, so a method that
        forgets to lock also forgets to bind and does not compile. Push/pop
        does not fix migration on its own and only restores a context nothing
        here ever takes; per-thread binding has no hook to hang a "once" on.
        `Module.Function` turned out to have no bind at all and now goes
        through the same place.
  - [x] A regression test that fails today. (2026-09-19) — and the shape this
        item described is not one that fails: 64 goroutines doing 40
        `Upload`/`Download` round trips pass every time, because
        `cuCtxSetCurrent` is sticky and in a steady thread pool every
        migration lands on a thread already bound. What is needed is a thread
        that never bound, so `TestContextIsGoroutineSafe` shreds threads — a
        goroutine exiting while holding `runtime.LockOSThread` takes its OS
        thread with it. Unfixed, 15 runs of 15 fail in about 0.4 s; fixed, 15
        of 15 pass, under `-race` and all four `compute-sanitizer` tools too.
  - [x] Document the resulting guarantee on `cuda.Context`: what may be shared
        between goroutines and what may not. (2026-09-19) — a `*cuda.Context` is safe
        to share; ordering is not promised, and a device buffer two goroutines
        reach needs the same care as any other shared memory. `README.md`'s
        "not safe to share between goroutines" went with it.

- [ ] Refuse a pointer-bearing element type on `cuda.Slice` too.
      `NewHostSlice` does, because the collector does not scan page-locked
      memory; `Slice` has the same hazard by a different route, since
      `Download` fills a Go-heap `[]T` with device bytes and a pointer-shaped
      element would hand the collector addresses to follow. `checkElem` in
      `cuda/elemtype.go` is already the check —
      [the rule](docs/decisions.md#a-buffer-outside-the-go-heap-holds-no-pointers).
- [ ] An asynchronous `LaunchShared`. `simt.Kernel.LaunchOn` refuses a kernel
      that declares a dynamic shared tile: a dynamic tile and a stream are two
      new things at once and nothing has needed both yet. Noted rather than
      done, so the gap is a decision instead of an omission.
- [ ] Buffer pooling and leak detection. The stream work made the lifetime
      question concrete: a buffer an asynchronous copy is reading must outlive
      the copy, and today the only thing that says so is a doc comment plus
      `Stream.Close` refusing after a cancelled `Wait`.
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

## Phase 7 — Release engineering (S–M) — 5 of 9

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
- [x] Lint the `cuda`-tagged half in CI. (2026-09-19) — a second
      `golangci-lint` job with `--build-tags cuda`, rather than a
      `run.build-tags` key, so the untagged run stays exactly what it was and
      the two configurations fail separately and say which one did. The two
      `errcheck` findings that blocked it were both deliberate discards and
      became explicit ones with the reason at the site.
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
| 3 Correctness         | L    | 5 of 8   | 1.4, 2     | the one oracle still open needs a device                      |
| 4 Host runtime        | M    | 3 of 8   | 1.3        | streams changed ownership semantics; buffer lifetime is next  |
| 5 Performance         | M    | 0 of 4   | 2, 4       | may expose NVRTC as the ceiling → revisit the PTX decision    |
| 6 Tile maturity       | L    | 0 of 7   | 2, 4       | reductions and 2-D tiling are a rewrite of the code generator |
| 7 Release             | S–M  | 5 of 9   | all        | —                                                             |

The critical path is **1.4 → 3 → 5**. Phases 4 and 6 can run in parallel; 1.2's
Windows leg is independent of everything else.
