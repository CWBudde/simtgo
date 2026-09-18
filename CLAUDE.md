# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A library for writing CUDA kernels in Go, answering NVIDIA's _CUDA Rust_ post
with two Go analogues: a **SIMT track** (`simt`) that lowers a Go subset to
CUDA C at the source level, and a **tile track** (`tile`) that records a graph
of ops and fuses it into one generated kernel.

Where things are written down, because a claim in the wrong file drifts:

- `README.md` — what this is, the two tracks, the subset, how to run it.
- `SPEC.md` — the contract: what the subset accepts and refuses, checked
  against `simt/errors_test.go` by `simt/spec_test.go`, which fails in both
  directions.
- `NUMERICS.md` — what the device does to a `float32` and what a test may
  therefore assert.
- `PLAN.md` — the roadmap: **open work only**, plus a short note per phase.
  Findings do not go here.
- `docs/` — the engineering record, filed by subject:
  [`decisions.md`](docs/decisions.md) (settled questions and why),
  [`emitter-defects.md`](docs/emitter-defects.md) (every mistranslation found
  and what catches it now), [`verification.md`](docs/verification.md) (the
  layers of checking and what each cannot see),
  [`toolchain.md`](docs/toolchain.md) (measured driver/NVRTC/PTX behaviour),
  [`tile.md`](docs/tile.md).

**When a piece of work turns up a durable finding — a measurement, a defect
post-mortem, a reason a design had to be that way — it belongs in `docs/`, and
`PLAN.md` gets one line and a link.** A closed roadmap item should shrink the
plan, not grow it.

## Commands

```sh
go build ./...                       # must stay green; the prebuilt gate can break it
go test ./...                        # transpiler, golden files, CPU emulator — no GPU needed
go test -race ./gpu/                 # the emulator must be race-clean
go test -tags cuda ./...             # CPU/GPU parity on a real device
CGO_ENABLED=0 go build -tags cuda ./...   # the driver must build without cgo or a toolkit
go test -run TestGolden ./simt/      # a single test
GOCUDA_UPDATE=1 go test -run TestGolden ./simt/   # refresh simt/testdata/*.cu goldens

go run ./cmd/gocuda vet ./kernels    # refuse kernels that cannot be lowered
go vet -vettool=$(which gocuda) ./...
go run ./cmd/gocuda generate -check  # is the committed CUDA C current? (CI check)
go generate ./...                    # regenerate kernels/prebuilt (needs NVRTC)
go run ./cmd/gocuda generate -pkg ./kernels -out ./kernels/prebuilt -no-ptx   # no toolkit

go run -tags cuda ./examples/fir     # also: vecadd, magnitude, tilefir

# compute-sanitizer over every kernel, one tool at a time. Needs a device.
GOCUDA_REQUIRE_DEVICE=1 go test -tags cuda -count=1 -timeout 0 \
  -exec "compute-sanitizer --tool=memcheck --error-exitcode 1 --report-api-errors no --target-processes application-only" \
  ./simt/ ./tile/ ./cuda/   # also: racecheck, initcheck, synccheck
```

The sanitizer sweep is three flags and an environment variable, and each of
them is load-bearing. `--error-exitcode` is what makes it a gate:
`compute-sanitizer` exits 0 on findings otherwise, so without it a clean run
and a dirty one are the same result. `--report-api-errors no` drops a class
this repository tests better than the sanitizer does — `cuda`'s error tests
hand `cuModuleLoadData` deliberate garbage and assert the `CUresult` that comes
back, which the default counts as errors. `GOCUDA_REQUIRE_DEVICE` turns the
test helpers' "no CUDA device available" skip into a failure, because a sweep
that launched nothing is green and means nothing. And note that a clean run
prints **nothing**: `go test` discards a passing binary's stdout, so the
`ERROR SUMMARY` lines show up only on a failure, or under `-v`.

There is a `justfile` carrying all of the above under shorter names, and
`just check` is the whole non-GPU sequence in CI's order. It mirrors the
workflow rather than being called by it, so the workflow stays authoritative:
if the two disagree, the justfile is the stale one. `just --list` is the index.

Linting is `golangci-lint run ./...` (`.golangci.yml`) and formatting is
`treefmt` (`treefmt.toml`: gofmt for Go, prettier for Markdown, YAML and JSON);
`treefmt --ci` checks without writing. Both run in
`.github/workflows/ci.yml`, together with the checks above — that workflow is
the **non-GPU** half of Phase 1.4, so nothing in CI runs on a device and the
parity tests are unverified there.

`golangci-lint` must be built with this module's own Go: the version check
compares the toolchain that built the linter against `go.mod`, and a release
binary built with an older Go refuses the module outright. CI uses the action's
`install-mode: goinstall` for that reason.

## The `cuda` build tag

There is no cgo anywhere. The driver and NVRTC are `dlopen`ed at run time via
purego (`cuda/loader_cuda.go`), so everything builds with `CGO_ENABLED=0` and
with no toolkit installed — check that with
`CGO_ENABLED=0 go build -tags cuda ./...`.

The tag still selects the implementation: the driver lives behind
`//go:build cuda`, and `cuda/stub.go` mirrors the API for `!cuda` builds where
every call returns `ErrNoCUDA`, so the transpiler, the analyzer and their tests
run without a GPU. **Anything added to the driver needs a matching stub**, and
`cuda/surface_test.go` fails if the two surfaces drift.

Which library file is opened is decided in `cuda/library.go`, which is
deliberately untagged and free of any loading so the path policy is testable
with no CUDA present (`cuda/library_test.go`). `GOCUDA_LIBCUDA` and
`GOCUDA_LIBNVRTC` replace the respective search outright. `CUDA_PATH`/
`CUDA_HOME` apply to **NVRTC only**, ahead of the system library: the driver
ships with the driver, not the toolkit, so a toolkit root says nothing about
where it lives.

`cmd/gocuda` depends on `internal/lower` and never on `simt`. It calls
`cuda.Compile` directly, in process; the `cmd/gocuda-nvrtc` child that used to
speak JSON on stdin/stdout is gone, because the reason for it — `simt` was cgo
and `cmd/gocuda` had to build without a toolkit — went with cgo. `package
cuda` carries the tag split internally, so the tool still builds and runs
untagged, where `Compile` is `cuda/stub.go`'s and returns `ErrNoCUDA`. That is
why the `//go:generate` line in `kernels.go` carries `-tags cuda`: untagged it
would reach the stub and produce no PTX at all. Without a toolkit, `-no-ptx`
refreshes the gate and never calls NVRTC.

## Architecture

**`internal/lower` is the single definition of the supported Go subset.** Two
callers share it so they can never disagree: `simt.Transpile` (type-checks
embedded kernel sources at run time) and `analysis/simtcheck` (handed a real
package by go/analysis). Never re-implement a subset rule in `simt` or in the
analyzer — put it in `lower` and both get it.

- `lower.LoadPackage` parses and type-checks a kernel package; `Package.Kernel(name)`
  lowers one function to a `*Unit` (`Source`, `Name`, `RequiredBlock`,
  `SharedBytes`, `DynSharedWidth`, `Params`, `SourceHash`). Everything after
  `Source` is a **launch contract** discovered while lowering: what the source
  demands of the launch that will run it, which is why it travels with the unit
  rather than being restated at every launch site. A kernel that produces any
  `Diagnostic` yields `(nil, diags)` — half-lowered CUDA must never escape.
- `lower.GPUPackage()` builds `types.Package` for `gpu` **by hand**, because
  run-time type-checking has no module graph. `internal/lower/gpupkg_drift_test.go`
  compares it against the real package: **adding anything to `gpu` means
  adding it to `gpupkg.go` too**, or kernels can call it in Go and fail to
  transpile.
- A kernel is any function whose first parameter is `gpu.Ctx`. `//gocuda:device`
  in a doc comment says it is a helper rather than a kernel, and is refused on a
  function taking no `gpu.Ctx`; `//gocuda:ignore`, in a doc comment or a file's
  package comment, opts out entirely.
- `Unit.SourceHash` hashes the _generated CUDA C_, not the Go source. That is
  the whole staleness story: a prebuilt is filed under it, so an artifact built
  from anything else is simply not found and `Build` falls back to NVRTC.

**Ahead-of-time pipeline.** `go generate` (line in `kernels.go`) runs
`gocuda generate`, which writes `kernels/prebuilt/`: the `.cu`, the
`.compute_75.ptx`, and `prebuilt_gen.go` with one `Lowered` constant per kernel
that lowered plus an `init` calling `simt.RegisterPrebuilt`. Hand-written
`kernels/prebuilt/gate.go` lists those constants, so a kernel that stops
lowering fails `go build` with `undefined: <Name>`. The case the gate cannot
catch — edited but never regenerated — is covered by `TestPrebuiltIsCurrent`
(`simt.VerifyPrebuilt`), which needs no GPU. Artifacts must not land in the
kernel package itself (it is type-checked as one unit and may import only `gpu`);
`generate` refuses `-out == -pkg`.

**Build/launch path.** `simt.Build` transpiles _even when a prebuilt exists_ —
lowering is what produces the hash, and it keeps the launch contracts
(`RequiredBlock`, `SharedBytes`, `DynSharedWidth`, `Params`) derived from the
source in hand rather than trusted from an artifact. `internal/jit` then either loads registered
PTX or calls NVRTC, caching modules per `cuda.Context` (never package-global —
a module dies with its context). `.gocuda-cache/` receives the `.cu`/`.ptx`
that were actually used, for inspection; it is gitignored, and per-`Build`
via `simt.WithCacheDir`.

**Tile track.** `tile.Graph` records nodes as ordinary Go calls; nothing
touches the device until `Materialize`. `tile/codegen.go` folds elementwise ops
into one C expression at the thread index; a windowed op (FIR) stages a shared
tile including its halo through the same fused expression.
`MaterializeStepwise` runs one kernel per op, so the cost of not fusing is
measurable rather than asserted.

**`gpu` is the kernel vocabulary and the CPU emulator.** The same kernel source
runs under `gpu.RunCPU(grid, block, fn)` and on the device; parity tests
(`simt/parity_test.go`, `-tags cuda`) compare both against an independent Go
reference, so a shared misunderstanding cannot pass as agreement.

## Invariants worth preserving

- **Refuse, never mistranslate.** Anything outside the subset gets a
  `Diagnostic` with a position. `simt/errors_test.go` pins that boundary —
  extend it when the subset moves, and `SPEC.md` with it, or `simt/spec_test.go`
  fails. It checks both ways: a refusal the code enforces and the contract
  omits, and a rule the contract claims that nothing pins.
- **A barrier is on the block's common path.** `ctx.SyncThreads()` and the
  `_sync` warp built-ins are refused under a thread-varying condition, inside a
  loop whose trip count varies between threads, or after a thread-varying
  `return` (`internal/lower/diverge.go`, interprocedural, following
  `readonly.go`'s shape). Block-uniform — and so fine to branch on — are
  `BlockIdx*`, `BlockDim*`, `GridDim*`, scalar parameters, `len()` and the
  constants. The rules do not see inside a condition, so a short-circuited
  `&&` can still break the warp participation promise; that is stated at the
  site.
- **One comparison rule.** `internal/tolerance` is the single definition of
  what "close enough" means, and `NUMERICS.md` says when a test may demand
  exact equality instead.
- **Go `int` narrows to C `int`** (32-bit). This is the one deliberate
  infidelity; indices are bounded by the grid.
- `ctx.AssumeBlockDim(n)` emits no code; it records a launch requirement that
  both `Kernel.Launch` and `RunCPU` enforce. What it counts is threads per
  block across all three axes, so a 16×16 block satisfies `AssumeBlockDim(256)`.
  `Build` also refuses a kernel whose shared memory exceeds the device limit.
- A kernel may call other functions in its package; each one it reaches is
  emitted into the same translation unit as a `__device__` function. That is
  why `lower.Kernel` takes the package's files and not just the entry point.
  Recursion is refused, and so is calling a kernel — a function taking a
  `gpu.Ctx` is one, unless it carries `//gocuda:device` or `//gocuda:ignore`.
- Kernel packages may import **only** `github.com/CWBudde/gocuda/gpu`. The
  device types are `float32`, `float64` (opt-in), `int32`, `int64`, `uint32`,
  `uint64` and `bool`, plus named structs and fixed-size arrays of those.
  `int8`/`int16`/`uint8`/`uint16` are **storage only** — legal as a slice
  element, an array element or a struct field, and accepted by no operator,
  because Go's 8- and 16-bit arithmetic wraps where C's promotes to `int`. Go's
  `int` is refused anywhere a layout is involved: as a value it narrows, as an
  element it is a different stride.
- A struct's holes are emitted as `gocuda_padN` members, the trailing one
  included. That is what makes the `sizeof` assertion imply the field offsets
  rather than merely agree with them — NVRTC has no `offsetof` to assert them
  directly.
- Shared memory is one constructor per element type, plus a dynamic tile
  (`ctx.SharedDynF32()`) whose length is a generated kernel parameter the
  launch fills. A kernel gets **at most one** dynamic tile: CUDA has a single
  dynamic `__shared__` block, and NVRTC accepts a second `extern __shared__`
  declaration silently, aliasing the same bytes. A device function may declare
  a static tile; it may not declare a dynamic one, and it may not call
  `AssumeBlockDim`.
- `cuda.Result` sentinels and `errors.Is` work in `!cuda` builds too — error
  handling code compiles without a toolkit.

## Adding a kernel

1. Write it in `kernels/` (first param `gpu.Ctx`, `gpu` the only import).
2. Add its name to `Gate` in `kernels/prebuilt/gate.go`.
3. Add it to the list in `kernels/prebuilt/prebuilt_test.go`'s
   `TestGateListsEveryKernel`, which is what catches a kernel missing from the
   gate.
4. `go generate ./...` (or `gocuda generate -no-ptx` without a toolkit).
5. Add it to the list in `simt/transpile_test.go`'s `TestGolden` and run with
   `GOCUDA_UPDATE=1` to create `simt/testdata/<Name>.cu`.
6. Add it to the list in `simt/nvrtc_cuda_test.go`'s `TestGeneratedCCompiles`.
7. Add a CPU/GPU parity test in `simt/parity_test.go` with an independent Go
   reference.

The kernel's name is spelled in **four** places (the gate plus steps 3, 5 and 6) and they are the usual thing to get wrong — and the usual conflict when two
branches each add a kernel.

Changing the emitter changes every `SourceHash`, so goldens _and_
`kernels/prebuilt/` both need regenerating.

## Writing

Technical content is English. Comments here explain _why_ a thing is the way it
is — including honest limits — rather than restating the code; match that
register. `README.md`, `PLAN.md` and `docs/` state measured numbers and
explicit non-goals; do not let a claim drift from what the code does, and say
where a number was measured — one machine is one machine.
