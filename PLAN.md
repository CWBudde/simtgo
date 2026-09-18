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

|              | Current                                                                    | Needed                                       |
| ------------ | -------------------------------------------------------------------------- | -------------------------------------------- |
| Driver API   | 18 calls, fully synchronous                                                | streams, events, async copies, pinned memory |
| Grids        | ~~1-D only~~ 1-D/2-D/3-D                                                   | —                                            |
| Types        | ~~`float32`, `int32`/`int`~~ scalars, structs, arrays, ~~narrow integers~~ | —                                            |
| Kernel calls | ~~none~~ `__device__` functions, recursion refused                         | —                                            |
| Errors       | surface at **run time**, inside `main()`                                   | at **build time**                            |
| Toolchain    | ~~hard-coded `/usr/local/cuda`~~ run-time `dlopen`, no cgo                 | Windows                                      |
| Tile ops     | 7, one windowed, no reductions                                             | reductions, 2-D, fusion planning             |

Phase 2 is closed but for fast math and the tuple ABI: shared memory is typed
and dynamically sized, the warp vocabulary is in, pointers carry `const` and
`__restrict__` with the aliasing promise checked rather than assumed, and the
narrow integers are storage. Thirteen kernels now, up from nine.

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
- [x] **Collapse `cmd/gocuda-nvrtc` into `cmd/gocuda`.** (2026-09-18) — done
      as described: `generate` calls `cuda.Compile` in process, and the JSON
      protocol, the `go run` of a second binary and the build-tag split are
      gone. `package cuda` already carried the split internally, so the tool
      still builds and runs untagged, where `Compile` is the stub's and answers
      `ErrNoCUDA`. One behavioural change came with it, and it is the kind that
      would otherwise be found by a silently empty artifact directory: the
      `//go:generate` line now carries `-tags cuda`, because untagged the tool
      reaches that stub and produces no PTX at all.

      `parseArchLocal` went too, and it was the worse of the two
      implementations rather than merely the redundant one: any trailing `a` or
      `f` was read as an arch-conditional target without looking at what
      preceded it, so `compute_a` was refused with advice naming a string the
      function itself rejects. Every case the test already pinned behaves
      identically under `cuda.ParseArch`.

      Nothing had ever exercised the toolkit-missing path, because every test
      in `generate_test.go` sets `-no-ptx`. One does now.

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
- [x] Non-GPU job proving the no-tag build, `go vet`, and the race detector.
      (2026-09-18) — `.github/workflows/ci.yml`, ubuntu-latest, three jobs:
      `go build`, `go vet` under both tags, `go test ./...`,
      `go test -race ./gpu/`, `CGO_ENABLED=0 go build -tags cuda ./...`,
      `gocuda generate -check`, and `go test -tags cuda ./...`; plus
      `golangci-lint` and `treefmt --ci`. The workflow itself is the one thing
      here that cannot be verified before it is pushed; every command in it was
      run by hand first.

      The tagged test run took two fixes to become possible at all.
      `TestGeneratedCCompiles` and `cuda`'s `TestNVRTCVersion` both needed only
      the toolkit and neither skipped without one, so `go test -tags cuda ./...`
      **failed** rather than skipped on a machine with neither toolkit nor
      device — the machine where the emitter's own tests are most worth
      running, and the only kind of machine CI has. Both skip now, which is
      what turns that command into a gate rather than a known-red step.

      `golangci-lint` has to be built with this module's own Go. The version
      check compares the toolchain that built the linter against `go.mod`, so a
      release binary built with an older Go refuses the module outright — hence
      the action's `install-mode: goinstall`. `treefmt` cannot be `go install`ed
      at any version: the module zip is rejected by the proxy itself over a
      non-ASCII path in its own test data, and its GitHub releases are drafts,
      so CI builds it from a pinned tag.

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

      - [x] **Double-precision `gpu` math** — `Sqrt64`, `Hypot64` and the rest,
            mapping to CUDA's unsuffixed `sqrt`/`hypot` and legal only under
            `//gocuda:float64`. (2026-09-18) — and the opt-in turned out to
            need enforcing somewhere new. Every other `float64` refusal lives
            in `ctype`, which sees a type somebody wrote down, and
            `y[i] = float32(gpu.Sqrt64(2))` writes none: the argument is an
            untyped constant and the result is converted away, so the kernel
            would have run a double it never named. The permission is about
            what the device is asked to execute, so the check belongs on the
            call. Membership of `gpuFuncs64` is what requires the directive,
            which keeps a name added later from quietly escaping it.
      - [x] **Narrow integer storage** — `[]uint8` image buffers and the like.
            (2026-09-18) — built as the line above describes: the four types
            are accepted in `ctypeElem` and nowhere else, so they may be a
            slice element, an array element or a struct field and never a
            variable, a parameter or a result, and every operator on one is
            refused. `kernels.Gray` is the motivating case made real, 8-bit RGB
            to 8-bit luma with the arithmetic at `int32`.

            Some of those operators do agree — compound assignment and `++`
            truncate at the store in both languages, as do comparisons, `/` and
            `%`. They are refused anyway, and the messages say which ones agree
            rather than claiming a disagreement a reader could not reproduce.
            A rule that holds for every operator is one somebody can keep in
            their head, and relaxing a refusal later costs a line where
            retracting an acceptance costs a release.
      - [x] **Array-typed struct fields.** (2026-09-18) — and the field rule
            was indeed not the work: offset checking was. Every hole Go leaves
            is now declared as a `gocuda_padN` member, the trailing one
            included, which is what makes the existing `sizeof` assertion
            _imply_ the offsets instead of merely agreeing with them. With
            every hole spelled out the members account for exactly Go's size,
            and C++ lays each member at or after the end of the one before it,
            so the size can only match if nothing further was inserted. The
            trailing member is load-bearing rather than tidy: without it a byte
            inserted earlier could hide in the end slack.

            What the measurement settled, against expectation: this fixes no
            disagreement anybody could observe. NVRTC inserts exactly the holes
            Go does, so the assertions passed before the padding existed, and a
            struct that would fail without it does not exist on a conforming
            C++ ABI. The mechanism is shown to have teeth the other way round —
            one byte too _much_ padding is rejected. `offsetof`,
            `__builtin_offsetof` and `#include <cstddef>` were re-measured
            against 12.9 and are all still unavailable, so the repository's
            standing claim about NVRTC holds.

- [x] **Shared memory.** Typed (`SharedI32`, …) and dynamically sized at
      launch. (2026-09-18) — six constructors, one per scalar the device has
      except `bool`, driven by one table and sized through `types.Sizes` so no
      width is written down twice; `SharedF64` inherits the `//gocuda:float64`
      permission structurally, through `ctypeElem`, rather than by a second
      list somebody would have to remember. A dynamic tile is spelled without a
      size, lowers to `extern __shared__`, and carries its length as a
      generated parameter the launch fills — the same convention a slice
      already uses. `Kernel.LaunchShared` takes an element count, because bytes
      are the emitter's business; `gpu.RunCPUShared` is the emulator's half.
      `kernels.Histogram` is the demonstration, and it is also the proof that
      atomics work on a tile, which fell out of `t.lens` membership rather than
      needing anything new.

      One measurement changed a design: **NVRTC accepts a second
      `extern __shared__` declaration without a word**, of a different element
      type as readily as of the same one, and every one of them names the same
      bytes. Two names that silently alias is exactly what this emitter exists
      to refuse, so "at most one dynamic tile" is enforced here, with a
      position — and the comment says NVRTC is not the backstop, because it
      will not object.

- [x] **Atomics.** `atomicAdd`/`Min`/`Max`/`CAS`, with a Go-side vocabulary
      that the emulator implements faithfully. (2026-09-18) — `AtomicAddF32`,
      `AtomicAddI32`, `AtomicMinI32`, `AtomicMaxI32`, `AtomicExchI32` and
      `AtomicCASI32`, each returning the old value as the built-in does.

      They take a **buffer and an index** rather than a pointer, which is
      forced rather than chosen: the subset has no address-of, so `&s[i]` is
      something the emitter writes and a kernel can never say. The shape pays
      for itself twice, because it also gives the first argument something to
      be checked against — it must be a slice parameter or a shared tile, which
      are exactly the objects `t.lens` holds and exactly the ones that lower to
      something with a device address. A tile therefore works, and keeps
      working when it is passed on to a device function, where it is an
      ordinary pointer parameter.

      The vocabulary stops where CUDA's overloads stop at `compute_75` with no
      header, which is all NVRTC has. Those gaps are named in `gpu`'s package
      comment rather than left to be rediscovered: `atomicMin`, `atomicMax` and
      `atomicCAS` have no float form, and `atomicAdd` has no `long long` form,
      only `unsigned long long`, so 64-bit variants would each need a different
      reinterpret cast and the one-Go-name-to-one-built-in table would stop
      holding.

      The emulator serialises every read-modify-write on one package-level
      mutex, because global memory there is the caller's own Go slice and there
      is nowhere per-buffer to hang a lock. What the lock does *not* do matters
      more: it creates a happens-before edge only between goroutines that take
      it, so a plain write to the same element is still reported by `-race`,
      which is right, because on the device that is a race too. It unlocks
      through `defer`, and that is load-bearing rather than stylistic — an
      out-of-range index panics with the lock held, `RunCPU`'s per-thread
      recover turns that into a diagnosis rather than a crash, and without the
      `defer` the next launch to use an atomic would block forever. Removing it
      makes `TestAtomicPanicDoesNotStrandTheLock` hang until the test timeout,
      which is the only way that bug ever announces itself.

- [x] **Warp-level primitives.** Shuffle, ballot, `__activemask`.
      (2026-09-18) — the shuffles in four directions for `float32` and `int32`,
      `Ballot`, `Any`, `All`, `ActiveMask`, `SyncWarp`, `LaneID`, and
      `gpu.WarpSize`. **NVRTC declares every one of them with no header
      included**, measured one spelling at a time rather than assumed, so the
      vocabulary did not have to shrink. "Warp reductions" in the library sense
      is _not_ done: `kernels.WarpReduceSum` is a kernel, and the Phase 6
      reductions item now has the pieces it needs.

      They are `Ctx` methods rather than package functions, and that is forced
      rather than stylistic: a warp primitive is defined by *which* thread
      calls it, and a package-level Go function cannot learn which goroutine is
      calling. The atomics can be package-level precisely because they are
      handed the memory they work on.

      No mask argument, for the reason the atomics take a buffer and an index:
      it keeps one Go name mapping to one built-in and gives the contract
      something checkable. The cost is stated rather than hidden — in a block
      that is not a multiple of 32 the emitted `0xffffffff` names lanes that do
      not exist, which CUDA leaves undefined, so a kernel that depends on whole
      warps should say so with `AssumeBlockDim`.

      The emulator has no warps to borrow and builds one: a per-warp rendezvous
      between goroutines that releases when every live participant has either
      arrived or returned, with a deadline behind it so a thread that never
      arrives is a diagnosis rather than a hang. It reports two things the
      device cannot, and its `ActiveMask` is an arrival mask, which is not what
      the device would answer. Both are named in `gpu`'s package comment.

- [x] **Missing statements.** `switch`, labelled `break`/`continue`,
      `for i, v := range`. (2026-09-17) — `switch` has two lowerings, because
      C's switch and Go's are not the same statement: all-constant cases over
      an integral tag become a C `switch` with an explicit `break` per clause
      (and `fallthrough` honoured by leaving it out), everything else becomes
      the `if`/`else` chain Go's semantics actually describe. A bare `break`
      inside that chain is refused, because in C it would leave the enclosing
      loop. Labelled branches lower to a `goto`. `for i, v := range` binds the
      value as the copy Go makes it.
- [x] **`const` / `__restrict__`.** (2026-09-18) — every pointer is
      `__restrict__` and one the kernel never writes through is `const` too,
      proved across the call graph and conservatively: anything unreadable — an
      unresolvable call, a cycle, an unnamed parameter — counts as a write,
      because marking a written parameter `const` is a compile error at best
      and a wrong answer at worst.

      The contract is **checked rather than asserted**, which is the half that
      makes the qualifier honest. `Kernel.Launch` compares the device ranges it
      was handed and refuses an overlap with an `*AliasError`; and the case a
      launch cannot see — `blend(y, y)`, one Go call becoming two aliased
      restrict pointers in C — is refused where it is lowered. Two _read-only_
      parameters may share a buffer deliberately, since `restrict` forbids
      reaching a *modified* object through another pointer, so `dot(x, x)` is
      sound and refusing it would cost something for nothing. The CPU emulator
      cannot check either: it never sees the caller's slices, which arrive
      through a closure, and there a kernel is ordinary Go where aliasing is
      defined. That gap is named in `RunCPU`'s doc comment.

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

- [x] **Multiple assignment**, in its parallel form. (2026-09-18) —
      `a, b = b, a` and `a, b := x, y` lower through one temporary per value,
      unconditionally: proving the two orders coincide would mean proving no
      right-hand side reads what an earlier target writes, through calls that
      may write slices, and C++ deletes a temporary nobody needed while nothing
      recovers a swap that quietly became a copy. In a `for` clause it stays
      refused — `simple` yields one C expression, and the temporaries the
      semantics require are declarations, which C's comma operator cannot
      carry.

      **The claim this line used to make was wrong in both directions**, and
      the correction is the useful part. Multiple assignment is *not necessary*
      for `ctx.GlobalID2()`: `assign` sees the statement before `expr` is ever
      called, so a pair-returning intrinsic could be destructured there, in the
      same place `ctx.SharedF32(n)` is already special-cased. And it is *not
      sufficient*: what a **user** function returning two values needs is
      out-parameters, which is ABI work this touches nowhere. Implementing the
      parallel form brought `GlobalID2` no closer, and the two remain separate
      items rather than one.

      What the work found on its own is the better finding: Go's first phase
      also evaluates **the index expressions on the left**. `i, y[i] = 2, 7`
      stores into the old `i`'s element, and assigning in order stores into the
      new one's — code that compiles and computes something else. An index the
      statement itself writes is lifted into its own temporary; a stable one is
      not, so `y[i], y[j] = y[j], y[i]` stays free of them.

- [ ] **Tuple assignment from a call** — `a, b := f()`, which needs
      out-parameters in the generated C, and separately a destructuring special
      case if a pair-returning intrinsic is ever wanted.
- [x] **A `//gocuda:device` marker.** (2026-09-18) — read exactly as the other
      two directives are, and consulted in `IsKernelDecl` beside `Ignored`, so
      every caller that decides what a kernel is agrees. `//gocuda:ignore`
      still works and still means "not a kernel". The marker carries a check
      the negative spelling could not: on a function taking no `gpu.Ctx` it is
      refused, so it cannot become decoration. A marked function nothing calls
      draws no diagnostic, deliberately — an unreached helper is never lowered
      at all, and the package compiles for the host too.
- [x] **Shared memory in a device function.** (2026-09-18) — and the
      accounting needed no propagation after all: `deviceFunc` memoises on the
      function, so each one is emitted once and its tiles counted once.
      `__shared__` inside a `__device__` function is block-scoped storage that
      CUDA allocates per function, confirmed against NVRTC rather than
      reasoned. `AssumeBlockDim` stays refused there, because it is a promise
      about a launch; so does the _dynamic_ tile, whose length arrives as a
      parameter of the kernel that a helper has no way to be handed.

This round found two defects of its own, and the first is the kind this phase
exists to remove.

**The emitter wrote generated names into the kernel's own namespace and checked
only one of them.** A slice lowers to a pointer plus an `x_len`, and a C++
keyword is spelled with a trailing underscore; neither name appears in the Go
source. Only the first was checked, and only against the _other parameters_, so
a local reached neither check. A kernel with a slice `y` and a local `y_len`
compiled and read 3 wherever it said `len(y)`; one with a parameter named `int`
and a local named `int_` compiled and read the local twice. Both were confirmed
through NVRTC before being fixed, and the fix is the Phase 0 parameter-collision
check widened from the parameter list to every variable the function declares.
It asks `types.Info` for variables that are not fields rather than reusing
`collectNames`, whose bluntness is right for choosing a fresh generated name and
would here refuse a kernel over a struct field that collides with nothing.

**A new emulator test raced on its own shared tile**, having all eight threads
of a block write `b[0]`, and `go test -race` reported it in about one run in
ten. The emulator was right: it serialises atomics on one lock precisely so
that a plain write racing another is still reported, because on the device that
is a race too. A test whose subject is the barrier must not smuggle one in to
prove it.

One caveat on the evidence, and it is now two caveats. Everything through the
type work is verified on **one** GPU — a T550, `sm_75`, CUDA 12.8; the parity
tests do not need CI to run, but they need CI to have run anywhere else.

The atomics and the `float64` helpers were weaker than that: they were written
where there was no device **and no toolkit**, so the generated C had been read
and not compiled, and whether NVRTC declares `atomicAdd` and the unsuffixed
`sqrt` with no headers included was left as a measurement nothing in that round
could make.

**It has now been made, and it holds.** `libnvrtc` needs neither a driver nor a
device, and `cuda/library.go` already honours `GOCUDA_LIBNVRTC` outright — a
Phase 1.2 decision that paid off somewhere it was not designed for. With it in
place `TestGeneratedCCompiles` ran for the first time anywhere and passed on
every case, so `atomicAdd`, the unsuffixed `sqrt`, and since this round every
warp built-in and `extern __shared__` are all declared with no header included.
The whole committed kernel set compiles as well as lowers.

What that does **not** settle is anything about execution. NVRTC compiles; it
does not run. So the 2026-09-18 round carries the same shape of caveat one step
further on: its lowering, refusals, drift test, analyzer, emulator under
`-race` and generated C are all verified, and **every parity test it adds is
written and unrun**, along with the atomics' and the `float64` helpers'. A
device is still what settles those, and there has not been one since the type
work.

One artefact of working without one is worth recording, because it would
otherwise look like carelessness: the only NVRTC available here was 12.9, while
the committed PTX was built by 12.8. Regenerating with it would have emitted
`.version 8.8` images that a 12.8 driver refuses — recoverable, since
`internal/jit` falls back to NVRTC on a rejected image, but it would silently
have cost the ahead-of-time saving on the one machine this project's
measurements come from. So 12.8 was fetched and used, and all twelve artifacts
stay at `.version 8.7`.

A second one is a gap in the gate rather than in the artifacts.
`gocuda generate -check` compares the lowered CUDA C and **not** the compiled
PTX: a deliberately corrupted `.ptx` passes it, while a tampered `.cu` fails.
That is the right check for source staleness and a weaker claim than "are the
committed artifacts current?" — `CLAUDE.md` said the stronger thing and now
says the true one.

## Phase 3 — Correctness at scale (L)

A transpiler is trusted through evidence, not review.

- [ ] **Differential fuzzing.** Generate random programs in the supported
      subset, run them on the CPU emulator and the GPU, compare. This is the
      backbone of the whole approach and should run continuously.

      (2026-09-18) — the generator, both oracles and the corpus exist; the GPU
      comparison does not, and cannot here. The definition above turned out to
      be narrower than the item deserves: it names two backends, and there is a
      **third that needs no device at all**. `internal/fuzz/hostrun` compiles
      the emitter's generated CUDA C with an ordinary host C++ compiler behind
      a small shim and runs it, so "compiles and computes something else" is
      answerable on a machine with no GPU — which is what CI is. What still
      needs hardware is only what the device does differently from any C++
      implementation: the transcendentals, the warp primitives, `fminf`'s
      treatment of a NaN, and what happens to a subnormal.

      `internal/fuzz` is the generator: one IR rendered twice, as Go source for
      `simt.Transpile` and as a `func(gpu.Ctx)` closure for `gpu.RunCPU`. The
      closure is not an interpreter of Go semantics — it *is* Go, running the
      same operations on the same types — which is what keeps the oracle from
      being a second implementation somebody has to trust.

      **Measured, on seeds disjoint from the ones the tests use.** The yield is
      20000/20000: every generated program is accepted by the subset, so a fuzz
      run tests the emitter rather than the refusal path. Feature coverage over
      3,000 programs: `switch` 80%, `range` 70%, narrow element types 66%,
      `SyncThreads` 45%, arrays 45%, shared memory 38%, atomics 32%, `float64`
      24%, device functions 21%, warp primitives 7.5% — and **structs 0%**.
      There is no struct node in the IR, which leaves the padding members, the
      `sizeof` assertion standing in for offsets NVRTC cannot assert, and the
      `ctype`/`ctypeElem` split untested by this route. That is the next thing
      the generator wants.

      41.1% of generated programs are in the host oracle's scope — a barrier, a
      warp primitive, an atomic or a shared tile means something a sequential
      run cannot reproduce — and 34.8% are also free of a transcendental, which
      the differential excludes because `sinf` and Go's `math.Sin` are two
      implementations of a function neither language requires to be correctly
      rounded. The NVRTC oracle takes all of it.

      **It found two defects in its first minutes**, both in what the catalogue
      ranks first and neither visible to a golden or a parity test. `-(-c)` was
      emitted as `--c`, which C++ lexes as predecrement: on a modifiable lvalue
      that compiles, decrements the variable and yields the decremented value.
      And `a := a + 1` was emitted as `int a = a + 1;`, where C++ starts the
      new name at its declarator and Go starts it at the end of the
      declaration — so the C read the variable being declared instead of the
      one being shadowed. Both are fixed with tests that fail without the fix.

      **The third finding was in the oracle**, which is the outcome the
      target's failure message is written to allow for: `tolerance.Agree`
      settled a NaN for a `float32` and let a `float64` fall through to
      `reflect.DeepEqual`, so two NaNs in a `[]float64` were reported as
      disagreeing — "b[2] = NaN, want NaN". A mismatch is a finding about one
      of the two backends and not a verdict about which, and the emulator has
      been the wrong one before. The rule is now asked once for both widths,
      and `internal/tolerance` — which three callers depend on and which had no
      tests at all — has them.

      (2026-09-18, later) — **it runs continuously now**:
      `.github/workflows/fuzz.yml`, daily and on demand, one matrix leg per
      untagged target so that one finding cannot hide the other. The NVRTC
      oracle stays out of it for the same reason the parity tests are out of
      CI: no runner has a toolkit. A case it finds is uploaded as an artifact
      and committed by hand after it has been read, because a corpus entry is
      a test every future run pays for and an automated commit would add them
      faster than anybody diagnoses them.

      Two more findings from the sustained runs, both of them in the
      *generator* rather than the emitter, which is the outcome the failure
      message is written to allow for. It clamped every float to ±1000 before
      converting it to an integer — right in principle, since such a
      conversion is implementation-dependent in Go and undefined in C — but
      to a *signed* range, so a negative float reached a `uint32` and the two
      backends disagreed on every element. And it wrote `o << (o & 31)` on a
      Go `int`, which is the narrowing above. `NUMERICS.md` gained the
      conversion rule, because it is a trap for a kernel author and not only
      for a generator.

      What is committed: three targets in `simt/`, two of them running in CI on
      every push. Go runs a fuzz target's seeds as ordinary tests under plain
      `go test`, so the whole host differential costs CI 2.3s and no flag. The
      third, `FuzzOutsideTheSubsetIsRefused`, is the only oracle there is for a
      rule whose content is that something is refused: it puts each broken rule
      inside a few hundred lines of generated control flow, where a rule that
      reads the wrong scope stops firing and `simt/errors_test.go`'s three-line
      kernels would never notice. Continuous fuzzing still wants a scheduled
      workflow; it is a `-fuzztime` line, not a design.

- [ ] **`compute-sanitizer`** (`memcheck`, `racecheck`, `initcheck`,
      `synccheck`) over every kernel in CI.
- [x] **Barrier-divergence analysis.** (2026-09-18) — `internal/lower/diverge.go`,
      three rules: a barrier under a thread-varying condition, a barrier inside
      a loop whose trip count is thread-varying, and a barrier preceded by a
      return under a thread-varying condition. The middle one is the one a
      lexical rule misses and the one that matters here, because it is the
      shape every staging loop in the repo has. It follows `readonly.go` —
      memoised per declaration, a separate cycle guard, pessimistic about
      anything it cannot read — and is interprocedural, since a barrier inside
      a helper is a barrier at the call site.

      Two facts made it tractable and both were checked rather than assumed:
      Go's own `goto` is refused, so every `goto` in the output is the
      emitter's and a structured analysis over the Go AST is sound; and a
      barrier can only be an `*ast.ExprStmt`, since it returns nothing and
      `simple()` would not take it in a `for` clause.

      **It found two live bugs, and one of them was a fix from earlier the same
      day.** `TypedProbe`'s early return before a barrier had been turned into
      a guard, which moved the problem rather than removing it: the guarded
      branch called a helper that barriers inside, so the same thread missed
      the same rendezvous one level down. And `WarpSum` in the analyzer
      fixtures wrote `ctx.LaneID() == 0 && ctx.Any(v != 0)`, where Go's `&&`
      short-circuits and so does the C — `__any_sync` ran on lane 0 alone while
      the emitter passed a full-warp mask, the participation promise broken by
      the spelling itself.

      That second one is also the honest limit: the rules work over statements
      and do not see into a condition. Extending them there would refuse
      `ctx.Any(p) && ctx.All(q)`, which is legitimate, because `Any` and `All`
      are warp-uniform and the pass holds the warp primitives to the stricter
      block lattice. Closing it properly needs a warp level in the lattice.
      Both limits are written down at the site rather than left to be
      rediscovered.

- [ ] **Aliasing.** Detect, or explicitly document, two slice parameters bound
      to the same device buffer. Largely done for the SIMT track by the
      `const`/`__restrict__` work in Phase 2: `Kernel.Launch` compares device
      ranges and refuses an overlap where the kernel writes through one of
      them, and the intra-kernel case is refused at lowering. What is still
      uncovered is the **tile track**, which has its own code generator, emits
      no qualifiers and does no launch-time check, and the **CPU emulator**,
      which never sees the caller's slices.
- [ ] **Debug mode.** Emit bounds checks, device `printf` and a trap, behind a
      flag — the closest thing to a panic the device can offer.
- [x] **Numerical policy.** (2026-09-18) — `NUMERICS.md`, organised so that
      every claim is labelled **measured**, **cited** or **unverified**, which
      is the only honest way to write one where there is no device.

      The measurements went further than reading the artifacts. `cuda.Compile`
      passes exactly one NVRTC option, `--gpu-architecture`, so every numerical
      setting is a default nobody chose; recompiling `FIR.cu`, `Quantize.cu`
      and `Magnitude.cu` with only that option reproduces the committed PTX
      **byte for byte**, and flipping each flag in turn shows what it would
      change (`--fmad=false` removes the `fma`, `--ftz=true` adds `.ftz`,
      `--prec-div=false` gives `div.full.f32`, `--prec-sqrt=false` gives
      `sqrt.approx.f32`). So the defaults are established rather than assumed.

      Three findings the plan did not anticipate. `--ftz=false` is not the
      whole story: a kernel calling `expf` emits an `ex2.approx.ftz.f32` inside
      `expf`'s own implementation, which the flag does not reach. `Magnitude`'s
      `fma` survives `--fmad=false`, so it is inside `hypotf` and is not
      evidence of source contraction — FIR and Quantize are. And Go's builtin
      `min`/`max` disagree with CUDA's on NaN exactly as `math.Min` did, which
      nothing tests and no committed kernel reaches.

      The tolerance rule is now one rule in `internal/tolerance`, where there
      were three conventions in three packages, and the exactness rule — when a
      test may demand equality rather than closeness — is promoted from
      scattered comments to something stated. Five of the nine float32 helpers
      gained parity tests, and six of them went through NVRTC for the first
      time.

- [x] **A written spec of the supported subset** — grammar and semantics, not
      prose in a README. This is the contract. (2026-09-18) — `SPEC.md`, and
      the half that makes it a contract is `simt/spec_test.go`, which fails in
      both directions: a refusal the transpiler enforces and the document
      omits, or a rule the document claims and no test pins. Both were watched
      failing before the commit, because a cross-check nobody has seen fail is
      not a cross-check. It keys on the diagnostic's own wording rather than on
      rule numbers invented for the purpose, so the document quotes what a
      reader will actually see in their terminal.

      Writing it was worth it for what the inventory turned up quite apart from
      the document: the README's generated-CUDA sample still showed unqualified
      pointers, its `GlobalID()` lowering was missing the cast that makes the
      axis accessors return `int`, its type table marked the narrow integers
      position-dependent and said nothing of the kind about `int`, and the
      refusal list named seven rules out of seventy-two while reading as
      closed.

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

- [x] LICENSE; confirm the redistribution position (`libnvrtc` is dynamically
      linked, not shipped — keep it that way). (2026-09-18) — MIT. The
      redistribution position is unchanged and now written down where it
      matters: `libnvrtc` is `dlopen`'d at run time and never shipped, so the
      license covers only what the repository contains. MIT rather than
      Apache-2.0 because there is nothing here to attach a patent grant to and
      no NOTICE to propagate.
- [ ] Semantic versioning, a v1 API freeze and a deprecation policy.
- [ ] `golangci-lint` + `gofumpt` as a CI gate.
- [ ] Godoc with runnable `Example` functions; the subset spec as the reference.
- [ ] CHANGELOG and release automation.
- [ ] A published support matrix: CUDA versions, architectures, OSes, Go
      versions.
- [x] Repository scaffolding that is simply missing. `.golangci.yml`,
      `treefmt.toml` and CI landed on 2026-09-18, and the LICENSE that blocked
      everything else here landed the same day.

      The earlier version of this bullet described a `.trunk/trunk.yaml`
      pinning `go@1.21.0` and `gofmt@1.20.4` against a `go 1.26` module. That
      file has never existed in the repository, on any branch, so there was
      nothing stale to fix and nothing to migrate: Trunk is simply dropped, and
      linting is `golangci-lint` with formatting by `treefmt`. Worth recording
      as a correction rather than a silent edit, because a plan that describes
      files it cannot see is the same failure as a README that describes code
      it does not have.

      Seven linters are enabled beyond the standard set, each measured clean
      before it was turned on. Seven more are left off rather than suppressed —
      `errorlint`, `perfsprint`, `predeclared`, `gocritic`, `intrange`,
      `wastedassign`, `revive` — because their findings are real and fixing
      them is a change to the emitter and the driver that belongs in its own
      commit. A config that excluded them would say the code is clean when it
      is not.

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
