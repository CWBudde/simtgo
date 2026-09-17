# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A proof of concept for writing CUDA kernels in Go, answering NVIDIA's *CUDA
Rust* post with two Go analogues: a **SIMT track** (`simt`) that lowers a Go
subset to CUDA C at the source level, and a **tile track** (`tile`) that
records a graph of ops and fuses it into one generated kernel. `README.md` is
the full design rationale; `PLAN.md` is the roadmap from PoC to 1.0 and tracks
which phases are done.

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
go run ./cmd/gocuda generate -check  # are the committed artifacts current? (CI check)
go generate ./...                    # regenerate kernels/prebuilt (needs NVRTC)
go run ./cmd/gocuda generate -pkg ./kernels -out ./kernels/prebuilt -no-ptx   # no toolkit

go run -tags cuda ./examples/fir     # also: vecadd, magnitude, tilefir
```

Linting is Trunk (`.trunk/trunk.yaml`: gofmt, golangci-lint2, markdownlint,
prettier) — `trunk check`. There is no CI workflow in the repo yet.

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

`cmd/gocuda` depends on `internal/lower` and never on `simt`. It reaches NVRTC
by shelling out to `cmd/gocuda-nvrtc` (built with `-tags cuda`), which speaks
JSON on stdin/stdout, one batch per invocation. That split existed because
`simt` was cgo and `cmd/gocuda` had to build without a toolkit; now that
nothing is cgo, the child process is no longer necessary — collapsing it is
tracked in `PLAN.md` under Phase 1.2.

## Architecture

**`internal/lower` is the single definition of the supported Go subset.** Two
callers share it so they can never disagree: `simt.Transpile` (type-checks
embedded kernel sources at run time) and `analysis/simtcheck` (handed a real
package by go/analysis). Never re-implement a subset rule in `simt` or in the
analyzer — put it in `lower` and both get it.

- `lower.LoadPackage` parses and type-checks a kernel package; `Package.Kernel(name)`
  lowers one function to a `*Unit` (`Source`, `RequiredBlock`, `SharedBytes`,
  `SourceHash`). A kernel that produces any `Diagnostic` yields `(nil, diags)` —
  half-lowered CUDA must never escape.
- `lower.GPUPackage()` builds `types.Package` for `gpu` **by hand**, because
  run-time type-checking has no module graph. `internal/lower/gpupkg_drift_test.go`
  compares it against the real package: **adding anything to `gpu` means
  adding it to `gpupkg.go` too**, or kernels can call it in Go and fail to
  transpile.
- A kernel is any function whose first parameter is `gpu.Ctx`. `//gocuda:ignore`
  in a doc comment (or a file's package comment) opts out.
- `Unit.SourceHash` hashes the *generated CUDA C*, not the Go source. That is
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

**Build/launch path.** `simt.Build` transpiles *even when a prebuilt exists* —
lowering is what produces the hash, and it keeps `RequiredBlock`/`SharedBytes`
derived from the source in hand. `internal/jit` then either loads registered
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
  extend it when the subset moves.
- **Go `int` narrows to C `int`** (32-bit). This is the one deliberate
  infidelity; indices are bounded by the grid.
- `ctx.AssumeBlockDim(n)` emits no code; it records a launch requirement that
  both `Kernel.Launch` and `RunCPU` enforce. `Build` also refuses a kernel
  whose shared memory exceeds the device limit.
- Kernel packages may import **only** `github.com/CWBudde/gocuda/gpu`, and only
  `float32`/`int32`-shaped types exist on the device.
- `cuda.Result` sentinels and `errors.Is` work in `!cuda` builds too — error
  handling code compiles without a toolkit.

## Adding a kernel

1. Write it in `kernels/` (first param `gpu.Ctx`, `gpu` the only import).
2. Add its name to `Gate` in `kernels/prebuilt/gate.go`.
3. `go generate ./...` (or `gocuda generate -no-ptx` without a toolkit).
4. Add it to the list in `simt/transpile_test.go`'s `TestGolden` and run with
   `GOCUDA_UPDATE=1` to create `simt/testdata/<Name>.cu`.
5. Add a CPU/GPU parity test in `simt/parity_test.go` with an independent Go
   reference.

Changing the emitter changes every `SourceHash`, so goldens *and*
`kernels/prebuilt/` both need regenerating.

## Writing

Technical content is English. Comments here explain *why* a thing is the way it
is — including honest limits — rather than restating the code; match that
register. `README.md` and `PLAN.md` state measured numbers and explicit
non-goals; do not let a claim drift from what the code does.
