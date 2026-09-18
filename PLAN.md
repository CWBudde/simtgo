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

|              | Current                                                    | Needed                                       |
| ------------ | ---------------------------------------------------------- | -------------------------------------------- |
| Driver API   | 18 calls, fully synchronous                                | streams, events, async copies, pinned memory |
| Grids        | ~~1-D only~~ 1-D/2-D/3-D                                   | —                                            |
| Types        | ~~`float32`, `int32`/`int`~~ scalars, structs, arrays      | narrow integers, `float64` math              |
| Kernel calls | ~~none~~ `__device__` functions, recursion refused         | —                                            |
| Errors       | surface at **run time**, inside `main()`                   | at **build time**                            |
| Toolchain    | ~~hard-coded `/usr/local/cuda`~~ run-time `dlopen`, no cgo | Windows                                      |
| Tile ops     | 7, one windowed, no reductions                             | reductions, 2-D, fusion planning             |

## The decision that gates everything else

**Keep generating CUDA C for NVRTC, or emit PTX (or LLVM IR) directly?**

|                    | CUDA C + NVRTC (today)                 | Direct PTX/LLVM IR     |
| ------------------ | -------------------------------------- | ---------------------- |
| Optimiser          | NVIDIA's, for free                     | yours to write         |
| Artifacts          | readable `.cu`, easy to diff and debug | PTX only               |
| Runtime dependency | `libnvrtc` (~60 MB)                    | none beyond the driver |
| Startup            | ~25 ms compile per kernel              | none, if AOT           |
| Control            | none over registers, scheduling, ABI   | total                  |
| Effort to extend   | low                                    | very high              |

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
      emits `int x_len, int x_len` — a duplicate C parameter. _Verified._
      Fix: generate lengths into a reserved namespace and reject collisions
      with a Go-level error instead of letting NVRTC report a C one.
- [x] **Symbol table keyed by name.** `transpiler.lens` (`simt/transpile.go`)
      maps identifier _strings_ to length expressions. Key it on
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

### 1.1 Build-time errors, not run-time (M) — _the single biggest gap_

Today a kernel that cannot be transpiled compiles fine and fails when
`main()` runs. No Go developer will accept that.

- [x] `gocuda vet`, a `golang.org/x/tools/go/analysis` Analyzer that flags
      unsupported constructs. Runs in CI and under `go vet -vettool`. _Not_ in
      editors: `gopls` has no plugin mechanism for third-party analyzers and
      runs a fixed, compiled-in set, so the editor story is a `golangci-lint`
      module plugin or an on-save task. Correcting that here rather than
      leaving the claim standing.
- [x] AOT pipeline: `go generate` → transpile → NVRTC → `go:embed` the PTX,
      with the existing JIT path kept as the fast development loop. One
      transpiler feeds both.
- [x] **Exit criterion:** a kernel that cannot be lowered fails `go build`.
      This needs three things, not one, and the first bullet alone does not
      achieve it — `go build` cannot be extended with custom vet checks, and
      `go test`'s implicit vet runs a fixed subset that ignores `-vettool`:

      1. `go generate` drops the constant for a kernel it could not lower, and
         a hand-written `gate.go` referencing it stops compiling:
         `kernels/prebuilt/gate.go:22:2: undefined: Scale`.
      2. That only catches a kernel that *was* regenerated. One edited and
         never regenerated keeps its constant and builds green, so
         `simt.VerifyPrebuilt` in an ordinary `go test ./...` catches the
         mismatch. No GPU needed; this is the cheapest and strongest of the
         three.
      3. `gocuda vet` catches it without regenerating at all.

Done. `Transpile` reports every refused construct rather than the first, as
typed `simt.Diagnostic`s behind a `simt.UnsupportedError` that `errors.As` can
unpack. The lowering moved to `internal/lower` so that the vet tool and the
generator do not need a CUDA toolchain to build — `simt` imports `cuda`, which
is cgo. `simt.Build` gained functional options and consults a registry of
prebuilt PTX keyed on the hash of the generated CUDA C, which makes a stale
artifact impossible to use rather than merely detectable. `simt.SetCacheDir` is
replaced by `simt.WithCacheDir`.

Three latent defects were found and fixed on the way, each an instance of the
failure this phase exists to remove:

- The emitter copied any identifier verbatim, so a kernel using a package-level
  variable lowered cleanly and failed inside NVRTC as `identifier is undefined`
  — a C error about code the author never wrote. The blank identifier was
  emitted as a C identifier for the same reason.
- `//go:embed kernels/*.go` matches `*_test.go`, which then joined the
  type-check and failed on `import "testing"`: a kernel package could not have
  tests at all.
- `sharedSize` returned the same answer for "not a shared buffer" and "a shared
  buffer I have already complained about", which under multi-diagnostic
  reporting produced a second, wrong diagnostic for one mistake.

Measured on the T550: the FIR example's `transpile + NVRTC compile` line falls
from **28.7 ms to 0.9 ms** when the PTX is prebuilt.

### 1.2 Toolchain discovery and portability (M)

- [x] Replace the hard-coded paths in `cuda/driver_cuda.go` with `CUDA_PATH` /
      `pkg-config` / a documented override. (2026-09-17) — the question moved
      from build time to run time, so the answer did too: `cuda/library.go`
      decides which _file_ to open, honouring `GOCUDA_LIBCUDA` /
      `GOCUDA_LIBNVRTC` outright and, for NVRTC alone, `CUDA_PATH` /
      `CUDA_HOME` ahead of the system library — the driver is not part of the
      toolkit, so a toolkit root says nothing about where it is. `pkg-config` is deliberately **not** used: it configures
      a link, and there is no link left to configure. The policy is untagged,
      loading-free Go, so `cuda/library_test.go` checks the search order on a
      machine with no CUDA — the machine where a path bug actually bites.
- [x] Load `libcuda` and `libnvrtc` at run time (`dlopen`, or `purego` to drop
      cgo entirely) so a binary **builds and ships without CUDA installed** and
      degrades gracefully on a machine with no GPU. This is what makes the
      library distributable. (2026-09-17) — purego; the module has no cgo left
      at all and `CGO_ENABLED=0 go build -tags cuda ./...` passes. The two
      libraries load independently, which is what lets a kernel with prebuilt
      PTX run with **no toolkit present**: with `GOCUDA_LIBNVRTC` pointed at a
      file that does not exist, `examples/fir` still computes on the GPU. A
      missing driver is a `*LibraryError` naming every candidate tried, not a
      link failure the caller could never have recovered from.
- [ ] Windows support; document macOS as unsupported (no CUDA).
      Unblocked by the above — there is no C toolchain in the way any more —
      but it cannot be _claimed_ without a Windows host or Phase 1.4's CI, so
      it is left open rather than written blind.
- [x] Keep the `cuda` build tag working as the no-GPU fallback. (2026-09-17) —
      kept as it was, and now enforced: `cuda/surface_test.go` type-checks the
      package under both tags and fails if the exported surfaces drift. That
      test was impossible before, because loading the tagged half needed a
      toolkit — so the machine that most needed the check was the one that
      could not run it.
- [ ] **Collapse `cmd/gocuda-nvrtc` into `cmd/gocuda`.** The child process
      exists only because `simt` used to be cgo while `cmd/gocuda` had to build
      without a toolkit (`cmd/gocuda-nvrtc/main.go` says so itself). Nothing is
      cgo now, so `generate` could call NVRTC directly and drop the JSON
      protocol, the `go run` of a second binary, and the build-tag split.

One finding worth writing down, because it is silent when wrong: the driver's
exported symbols are not the names in `cuda.h`. The header `#define`s
`cuMemAlloc` to `cuMemAlloc_v2`, and the same for `cuMemFree`, `cuMemcpyHtoD`,
`cuMemcpyDtoH` and `cuDevicePrimaryCtxRelease`. `libcuda` exports **both**, and
the unsuffixed ones are the pre-CUDA-3.2 API taking 32-bit sizes, so a `dlsym`
port that trusts the header names links successfully, passes every small test,
and truncates any allocation or copy above 4 GiB. The bindings name the `_v2`
symbols explicitly.

Kernel parameters no longer go through C `malloc`/`free` per launch: the driver
is handed pointers into Go memory held still by a `runtime.Pinner`, which is
what that type is for. This is a wash rather than a win — two C allocations
become two Go ones — and the FIR example measures the same as before
(1.1 ms kernel, 9.4 ms transfers), which is the point: the launch path was
never where its 10 ms went.

### 1.3 Error model and API review (S)

- [x] Typed errors with `errors.Is`/`As`, preserving `CUresult` codes.
      `cuda.Result` is a `CUresult` that is itself an error, and `*cuda.Error`
      carries the failing entry point. The name table is pure Go, so a code
      still prints itself in a build without the `cuda` tag. Note for Phase 4:
      loading garbage through `cuModuleLoadData` gives
      `CUDA_ERROR_INVALID_IMAGE`, not `INVALID_PTX` — only something the driver
      parses as PTX and then fails to JIT is `INVALID_PTX`.
- [x] Source positions on every transpiler diagnostic. The audit found the 49
      refusal sites were all fine and the gaps were all in `Transpile` itself:
      `go/types` stopped at the first error unless given an `Error` callback,
      `scanner.ErrorList` reported only its first, and the import refusal had
      no position of its own because it came from the importer.
- [x] One deliberate pass over the exported surface. `context.Context` is
      deliberately **not** added anywhere: every driver call is synchronous and
      uncancellable — `cuCtxSynchronize`, `cuMemcpyHtoD` and `cuLaunchKernel`
      cannot be interrupted — so accepting one would advertise semantics that
      do not exist. It belongs in Phase 4, where `cuStreamQuery` can genuinely
      `select` on `ctx.Done()`.

### 1.4 CI on real hardware (M)

Nothing below is verifiable without this.

- [ ] GPU runner (self-hosted or cloud), matrix over CUDA 12.x/13.x ×
      `sm_75`/`86`/`89`/`90` × Linux/Windows × supported Go versions.
- [ ] Non-GPU job proving the no-tag build, `go vet`, and the race detector.

## Phase 2 — Language coverage (L)

Each item means: a spec entry, a golden test, and a CPU/GPU parity test. Until
Phase 3 writes the subset spec, "a spec entry" is the README's _The supported
subset_ section — the plan files the real grammar-and-semantics document under
Phase 3, so pulling it forward would be doing that item, not this one.

- [x] **Device functions.** A kernel cannot call another Go function. This
      blocks most real kernels; emit `__device__` functions or inline them.
      (2026-09-17) — emitted, not inlined: every function the kernel reaches
      becomes a `__device__` function in the same translation unit, prototypes
      first so the order of the Go declarations does not matter. The CPU side
      needed nothing at all — a device function is ordinary Go, so `RunCPU`
      runs the very code the device compiles, and parity came for free.
      Recursion is refused (direct and mutual): there is no device stack depth
      to spend on it. `lower.Kernel` now takes the package's files, because
      resolving a call needs the declarations; both callers already had them,
      so the analyzer gained device functions in the same commit rather than
      afterwards.
- [x] **Multi-dimensional indexing.** `ctx.GlobalID2()`, `ctx.ThreadIdx3()`;
      the host side already has `Dim3`. (2026-09-17) — landed as **per-axis
      accessors**, not the tuple-returning spelling this line proposed:
      `ctx.ThreadIdxY()`, `ctx.BlockIdxZ()`, `ctx.GlobalIDY()` and the rest.
      `x, y := ctx.GlobalID2()` needs multiple assignment, which is a pinned
      refusal, so the spelling would have dragged a second and larger feature
      in with it; one CUDA built-in per accessor is what the emitter's table
      can express today. The unsuffixed names stay the x axis, so no existing
      kernel changed. `gpu` grew its own `Dim` — a kernel may import nothing
      but `gpu`, so the geometry it can talk about cannot be `cuda.Dim3` —
      plus `RunCPUDim`; `simt` grew `Kernel.LaunchDim`. `AssumeBlockDim` now
      counts threads per block across all three axes on both sides.
      `kernels.Transpose` is the proof, verified exactly against an
      independent reference.
- [x] **Types.** `int32`/`uint32`/`int64` slices, opt-in `float64`, `bool`,
      structs of scalars → CUDA structs, fixed-size arrays. (2026-09-18) —
      `uint32` and `bool` turned out to lower already, with nothing testing
      that they did and a refusal still reading "float32/int32 only"; they now
      have both. `int64`/`uint64` are `long long`, never `long`, which is 4
      bytes on Windows. `float64` is opt-in through `//gocuda:float64` on the
      kernel or its file, read exactly as `//gocuda:ignore` is; the permission
      covers the whole translation unit and a device function may not carry its
      own, for the reason `SharedF32` may not either. A struct's layout is
      stated by `go/types` and checked by NVRTC — `static_assert` on `sizeof`
      and `alignof`, the emitter never modelling the C ABI itself. Arrays are
      storage: declared, indexed, `len`, `range`, and nothing that moves one as
      a value.

      Three defects were found rather than designed, and the first was live.

      **`[]int` was corrupting.** `ctype` mapped Go's `int` to C's, a narrowing
      that is fine for a value passed on its own and is a different *stride*
      for an element: `cuda.Upload` copies 8 bytes each and the kernel read 4.
      On the T550, doubling `[1 2 3 4 5 6 7 8]` returned `[2 4 6 8 0 0 0 0]`.
      It lowered, compiled, launched and answered wrongly, which is exactly
      what "refuse, never mistranslate" exists to rule out — and it had been
      there since the narrowing was written. `int` in any position with a
      layout is refused now.

      **`LaunchSync` gave every parameter a fixed 8-byte slot.** Latent, since
      nothing wider existed; a 16-byte struct by value would have had its tail
      dropped. Reverting the fix makes `TestStructLayoutRoundTrip` report
      `Bias` as 0 and `TestBandGainParity` miss by exactly `Bias*Count`, so
      both are regression tests for it rather than tests that happen to pass.

      **`constant()` rendered every float at 32 bits** with an `f` suffix,
      which would have rounded every double silently and turned anything past a
      float's range into `+Inf`, and refused a legal `uint64` above `MaxInt64`
      as "does not fit in an int64".

      What the measurements settled, rather than the reasoning: NVRTC has no
      `offsetof` and no `__builtin_offsetof`, because it compiles a string with
      no include path — `nvcc -ptx` accepts the former and would have given the
      wrong answer. So the assertions reach `sizeof` and `alignof`, and the
      offsets are pinned by reading every field back through the device, which
      tests the `cuda.Upload` copy path rather than the compiler's opinion of
      it. `min`/`max` on `double` do resolve under NVRTC; `expr.go` claimed so
      in a comment and now there is a test.

      `gpu` needed no change at all: global memory in the emulator is an
      ordinary Go slice, so `[]int64`, `[]float64` and `[]Band` ran under
      `RunCPU` before any of this. The seven existing kernels' generated C is
      byte-identical, so the regeneration added two `.cu`, two `.ptx` and two
      constants and rewrote nothing.

      Left behind, as items rather than as prose:

      - [ ] **Double-precision `gpu` math** — `Sqrt64`, `Hypot64` and the rest,
            mapping to CUDA's unsuffixed `sqrt`/`hypot` and legal only under
            `//gocuda:float64`. A `float64` kernel currently gets arithmetic,
            comparison and conversion and nothing from `gpu`, which `go/types`
            reports as an ordinary type error rather than as a missing feature.
      - [ ] **Narrow integer storage** — `[]uint8` image buffers and the like.
            The widths already agree; the arithmetic does not, because C
            promotes to `int` and Go does not. The honest design is a
            storage-only type that forces a conversion to `int32` before any
            operator, which is a feature rather than a line in a switch.
      - [ ] **Array-typed struct fields.** Refused today because `sizeof` and
            `alignof` cannot say where inside the struct the array begins.
            Lifting it belongs with offset checking, not with the field rule.

- [ ] **Shared memory.** Typed (`SharedI32`, …) and dynamically sized at launch.
- [ ] **Atomics.** `atomicAdd`/`Min`/`Max`/`CAS`, with a Go-side vocabulary
      that the emulator implements faithfully.
- [ ] **Warp-level primitives.** Shuffle, ballot, `__activemask`, warp
      reductions — the basis of every fast reduction.
- [x] **Missing statements.** `switch`, labelled `break`/`continue`,
      `for i, v := range`. (2026-09-17) — `switch` has two lowerings, because
      C's switch and Go's are not the same statement: all-constant cases over
      an integral tag become a C `switch` with an explicit `break` per clause
      (and `fallthrough` honoured by leaving it out), everything else becomes
      the `if`/`else` chain Go's semantics actually describe. A bare `break`
      inside that chain is refused, because in C it would leave the enclosing
      loop. Labelled branches lower to a `goto`. `for i, v := range` binds the
      value as the copy Go makes it.
- [ ] **`const` / `__restrict__`.** Mark non-aliased read-only slice
      parameters; it is both a correctness contract and a real speed-up.
- [ ] **Opt-in fast math** and `#pragma unroll` hints for tap-style loops.

Three items done, and one of them started as a bug rather than a feature.
`*ast.BranchStmt` was lowered as `t.line("%s;", s.Tok)` and never looked at
`s.Label`, so `break outer` inside a nested loop emitted a bare `break;` and
left the **inner** loop: the kernel compiled and computed something else. That
is the failure "refuse, never mistranslate" exists to rule out, and it had
been there since the first commit. A smaller one went with it: `params()`
contributed nothing for an unnamed parameter, silently shifting every argument
after it. `AssumeBlockDim` counting the whole block rather than `blockDim.x`
is not in that list — it was not wrong while blocks were one-dimensional, it
was a question that only arose once they were not.

Review found four more, and they are worth naming because three share a shape
the golden and parity tests cannot see: valid Go lowering to C++ that does not
compile. A labelled `continue` jumped into the scope of a later declaration,
two `case` clauses declaring one name collided in the switch's scope, and a
device function with a named result referred to a local nobody emitted. The
fourth is the one to remember: the if/else chain re-evaluated the switch tag
in every arm, which was harmless when it was written — nothing in the subset
had side effects — and stopped being harmless one commit later, when device
functions arrived and a tag could write through a slice. Two features, each
correct alone. `simt/nvrtc_cuda_test.go` now puts the emitter's sharp edges
through NVRTC, which is the question the other tests never asked: is the
generated C something a compiler accepts?

What these three left behind, as items rather than as prose:

- [ ] **Multiple assignment**, and with it the tuple-returning
      `ctx.GlobalID2()` / `ctx.ThreadIdx3()` this phase chose not to build.
      It is one feature, not two: the emitter needs `a, b := f()` before any
      intrinsic can return a pair.
- [ ] **A `//gocuda:device` marker.** A helper that takes a `gpu.Ctx` is a
      kernel by the signature rule, so calling one needs `//gocuda:ignore` —
      which says what it is not, rather than what it is.
- [ ] **Shared memory in a device function.** Refused today, because the size
      and the block size it implies are accounted on the kernel and enforced
      at its launch. Lifting that means propagating the accounting through the
      call graph.

One caveat on the evidence, since Phase 1.4 is still open: all of the above is
verified on **one** GPU — a T550, `sm_75`, CUDA 12.8. The parity tests do not
need CI to run, but they need CI to have run anywhere else.

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
      documented _and tested_. **The per-call `cuCtxSetCurrent` is broken, and
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

| Phase                 | Size | Depends on | Main risk                                                     |
| --------------------- | ---- | ---------- | ------------------------------------------------------------- |
| 0 Known bugs          | S    | —          | none                                                          |
| 1.1 Build-time errors | M    | —          | changes the public API, so do it early                        |
| 1.2 Portability       | M    | —          | `dlopen`/`purego` rework of all cgo                           |
| 1.3 Error model & API | S    | 1.1        | cheap now, expensive after users                              |
| 1.4 GPU CI            | M    | —          | hardware access and cost                                      |
| 2 Language coverage   | L    | 1.1, 1.4   | device functions touch the whole emitter                      |
| 3 Correctness         | L    | 1.4, 2     | fuzzer findings may force emitter redesign                    |
| 4 Host runtime        | M    | 1.3        | streams change ownership semantics                            |
| 5 Performance         | M    | 2, 4       | may expose NVRTC as the ceiling → revisit the PTX decision    |
| 6 Tile maturity       | L    | 2, 4       | reductions and 2-D tiling are a rewrite of the code generator |
| 7 Release             | S–M  | all        | —                                                             |

The critical path is **1.1 → 1.4 → 2 → 3**. Phases 4 and 6 can run in
parallel once the foundations hold.
