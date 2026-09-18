# gocuda — writing CUDA kernels in Go

A proof of concept prompted by NVIDIA's [_Introducing CUDA Rust: Two Tracks for
Writing GPU Kernels_](https://developer.nvidia.com/blog/introducing-cuda-rust-two-tracks-for-writing-gpu-kernels/).
The question it answers: **can Go do this too?**

Short answer: not by the same means, but the result is closer than expected.
Both of NVIDIA's tracks have a working Go analogue here, and they run real
kernels on real hardware.

## The two tracks

|                         | CUDA Rust                                                                                             | This repository                                                                                                      |
| ----------------------- | ----------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **SIMT track**          | A custom `rustc` codegen backend lowers `#[kernel]` functions through MIR and LLVM IR to PTX          | `go/ast` + `go/types` lower a Go subset to CUDA C, which NVRTC compiles to PTX at run time (package `simt`)          |
| **Tile track**          | A `#[cutile::module]` proc macro embeds the kernel AST in the host binary and JITs it through Tile IR | The graph is recorded at run time by ordinary Go calls, then fused into one generated kernel (package `tile`)        |
| **Safety**              | `DisjointSlice<T>`, launch contracts, const generics                                                  | Runtime shape checks; the CPU emulator plus `go test -race`                                                          |
| **When errors surface** | `rustc` rejects the kernel                                                                            | `gocuda vet` rejects it, and `go generate` makes an unlowerable kernel fail `go build`                               |
| **Toolchain**           | Pinned nightly Rust, custom LLVM                                                                      | Plain `go1.26`, cgo, NVRTC. The library has no third-party dependencies; the `gocuda` tool uses `golang.org/x/tools` |

### Why Go cannot take Rust's route

- **`gc` has no pluggable codegen backend.** There is no Go equivalent of a
  custom `rustc` backend, so compiling the real language to PTX is not on the
  table.
- **TinyGo is the only Go toolchain on LLVM**, and its LLVM does register the
  `nvptx64` target — but 0.37 accepts only Go 1.19–1.24, and its runtime and
  garbage collector assume a host, not a device.
- **Go has no macros**, so `cutile`'s trick of embedding a kernel AST at
  compile time has no direct counterpart.

What Go does have, and Rust does not, is **a parser and a type checker in its
standard library**. Translating a well-defined subset at the source level gets
most of the way there — and it keeps kernels runnable as plain Go.

## Track 1 — SIMT

A kernel is an ordinary Go function:

```go
func FIR(ctx gpu.Ctx, y, x, h []float32) {
	ctx.AssumeBlockDim(FIRBlock)
	tile := ctx.SharedF32(FIRBlock + FIRMaxTaps)
	taps := len(h)
	base := ctx.BlockIdx() * ctx.BlockDim()
	t := ctx.ThreadIdx()

	for k := t; k < ctx.BlockDim()+taps-1; k += ctx.BlockDim() {
		src := base + k - (taps - 1)
		v := float32(0)
		if src >= 0 && src < len(x) {
			v = x[src]
		}
		tile[k] = v
	}
	ctx.SyncThreads()
	// ... tap loop
}
```

It compiles with the rest of the module, runs on the CPU through
`gpu.RunCPU`, and is lowered to this:

```cuda
extern "C" __global__ void FIR(float* y, int y_len, float* x, int x_len, float* h, int h_len)
{
	__shared__ float tile[320];
	int taps = h_len;
	int base = (int)blockIdx.x * (int)blockDim.x;
	int t = (int)threadIdx.x;
	for (int k = t; k < (int)blockDim.x + taps - 1; k += (int)blockDim.x)
	{
		int src = base + k - (taps - 1);
		float v = 0.0f;
		if (src >= 0 && src < x_len)
		{
			v = x[src];
		}
		tile[k] = v;
	}
	__syncthreads();
	// ...
}
```

Launching mirrors the Go signature, minus the `gpu.Ctx`:

```go
k, _ := simt.Build(ctx, gocuda.Kernels(), "FIR")
k.LaunchN(n, kernels.FIRBlock, dy, dx, dh)
```

**The same source runs on both backends.** That is the part Rust does not
offer: every kernel here is covered by a parity test that runs it on the CPU
emulator and on the device and compares, with an independent Go reference to
keep a shared misunderstanding from passing as agreement.

### The supported subset

Slices lower to a pointer plus a length, so `len()` works. `:=`, `=`, compound
assignment, `if`/`else`, three-clause `for`, `for i := range`,
`for i, v := range`, `switch`, `break`, `continue`, their labelled forms,
arithmetic, comparisons, indexing and conversions all translate.
`gpu.Sqrt`, `gpu.Hypot` and friends become `sqrtf`, `hypotf`; `ctx.GlobalID()`
becomes `blockIdx.x * blockDim.x + threadIdx.x`; `ctx.SharedF32(n)` becomes a
`__shared__` array. Struct literals and field access translate too.

Grids and blocks have three axes. The unsuffixed accessors are `x`, which is
CUDA's own spelling, and `ThreadIdxY`, `BlockIdxZ`, `GlobalIDY` and the rest
are the other two; `GlobalID` also answers to `GlobalIDX`, because next to
`GlobalIDY` the bare name reads like an oversight. Each is one built-in rather
than a tuple — `x, y := ctx.GlobalID2()` would need multiple assignment, which
the subset does not have. The host side matches: `Kernel.LaunchDim` and
`gpu.RunCPUDim` take a three-axis extent, `Launch` and `RunCPU` stay the
one-dimensional spelling, and `AssumeBlockDim` counts threads per block across
all three axes, so a 16×16 block satisfies `AssumeBlockDim(256)`.

Two of those need a word, because Go and C disagree about what the spelling
means. A `switch` whose cases are all constant becomes a C `switch`, with an
explicit `break` closing each clause because Go does not fall through;
`fallthrough` is honoured by leaving that `break` out. Any other switch — a
tagless one, or one testing a variable — becomes the `if`/`else` chain that Go
actually describes, and a bare `break` inside _that_ is refused rather than
emitted, because in C it would leave the enclosing loop. A labelled `break` or
`continue` becomes a `goto` to a target after the loop or at the end of its
body; before that was implemented the label was dropped, which is the kind of
silent mistranslation the rest of this section exists to rule out.

A kernel that sizes shared memory against a fixed block size says so with
`ctx.AssumeBlockDim(n)`. It emits no code; it records the requirement, so that
`Kernel.Launch` and the CPU emulator both refuse a mismatched launch instead of
quietly reading the wrong stretch of memory. `Build` likewise refuses a kernel
whose shared memory exceeds what the device offers per block.

A kernel may call another function in its package, which is emitted as a
`__device__` function alongside it: a prototype for each one the kernel
reaches, then the definitions, then the entry point. Slice parameters split
into a pointer and a length there too, so the call passes both. Recursion is
refused — there is no stack depth on the device to spend on it — and so are
methods, generics, variadics, more than one result, and a _named_ result,
which would be a local the body assigns to and a bare return that carries it.
A function taking a `gpu.Ctx` **is** a kernel by the rule above, so calling
one is refused unless it carries `//gocuda:ignore`, which already means
"not a kernel"; the `Ctx` then vanishes from the C signature as the kernel's
own does, and shared memory and `AssumeBlockDim` stay refused inside it,
because both are promises about a launch. The CPU side needs nothing at all
for any of this: a device function is ordinary Go, so `RunCPU` runs the very
code the device compiles.

#### Types

| Go                          | CUDA                 |
| --------------------------- | -------------------- |
| `float32`                   | `float`              |
| `float64`                   | `double`, opt-in     |
| `int`, `int32`              | `int`                |
| `int64`                     | `long long`          |
| `uint32`                    | `unsigned int`       |
| `uint64`                    | `unsigned long long` |
| `bool`                      | `bool`               |
| a named struct of the above | a CUDA `struct`      |
| `[N]T` of the above         | `T name[N]`          |

The 64-bit types are `long long` and never `long`, which is 8 bytes on Linux
and 4 on Windows.

`float64` needs `//gocuda:float64` on the kernel, or on its file's package
comment. The cost is invisible in the source — the device runs a double at a
fraction of the float32 rate, so a kernel that acquired one by accident would
be correct and far slower — and the opt-in makes that something somebody wrote
down. It covers the kernel's whole translation unit, device functions included;
a helper may not carry its own, for the reason `SharedF32` and `AssumeBlockDim`
may not either. There is no `float64` vocabulary in `gpu`: `gpu.Sqrt` and
friends are `float32`, so a double kernel gets arithmetic, comparison and
conversion and nothing else.

**Go's `int` is 64-bit and CUDA's is 32-bit.** That narrowing is the one
deliberate infidelity, and it holds only for a value passed on its own, where
an index is bounded by the grid anyway. As a slice element, an array element or
a struct field it is not a lost high word but a different stride, and
`cuda.Upload` copies Go's layout regardless — so `[]int` is refused, along with
`int` as a struct field. It used to lower cleanly and return the wrong numbers.

`int8`, `int16`, `uint8` and `uint16` are refused too, and not for their width,
which matches: Go computes `int8 * int8` in 8 bits and wraps, while C promotes
both to `int` and truncates only at the assignment, so
`a, b := int8(100), int8(3); a*b/2` is 22 in Go and −106 in C.

**A struct is laid out by Go and checked by CUDA.** The generated C carries
what `go/types` says about the type:

```c
struct Shape { float Floor; int Count; double Bias; };
static_assert(sizeof(Shape) == 16, "gocuda: Shape is a different size in CUDA than in Go");
static_assert(alignof(Shape) == 8, "gocuda: Shape is differently aligned in CUDA than in Go");
```

The emitter never models the C ABI. It states Go's numbers and lets the C++
compiler refuse them, so the guarantee holds wherever the kernel is built
rather than where it was written. Offsets are missing because NVRTC compiles a
string with no include path and so has no `offsetof`; `TestStructLayoutRoundTrip`
pins those by reading each field back through the device instead, which
exercises the `cuda.Upload` copy path a `static_assert` could only have had an
opinion about. Struct literals lower positionally, because designated
initialisers are C++20 and NVRTC defaults to C++17.

**An array is storage, not a value.** It can be declared, indexed, measured
with `len` and ranged over. It cannot be a parameter — Go passes a copy and C
decays the parameter to a pointer, so a write inside the function would reach
the caller's array — nor a result, which C cannot return at all, nor assigned
whole, which C++ will not do. Wrap it in a struct if it has to travel; both
languages copy that. `==` on a struct or an array is refused for the same kind
of reason: Go compares field by field and C++ gives a plain aggregate no
operator at all.

Everything else is **refused with a file and line**, never mistranslated:
allocation, interfaces, goroutines, multiple assignment, methods, embedded
fields, and any import other than package `gpu`. `simt/errors_test.go` pins
that boundary.

### Errors before `main()`

A kernel is ordinary Go, so the compiler has no opinion about whether it can
run on a device. Three things give it one:

```sh
go install github.com/CWBudde/gocuda/cmd/gocuda@latest
gocuda vet ./kernels                   # or: go vet -vettool=$(which gocuda) ./...
```

```text
kernels/bad.go:4:2: kernels may not import math (only github.com/CWBudde/gocuda/gpu is available on the device)
kernels/bad.go:10:23: float64 needs //gocuda:float64 on kernel Bad, or on its file's package comment: the device runs double at a fraction of the float32 rate, so it is opt-in
kernels/bad.go:10:38: []int cannot cross to the device: Go's int is 8 bytes and CUDA's int is 4, so the elements would not line up; use int32 or int64
kernels/bad.go:11:2: multiple assignment is not supported in kernels
```

The analyzer runs the **same lowering** the transpiler does, rather than a
second opinion about it, so what it accepts and what `simt.Build` accepts
cannot drift apart. A function whose first parameter is a `gpu.Ctx` is a
kernel; `//gocuda:ignore` in its doc comment opts one out.

`go generate` writes `kernels/prebuilt/`: the generated CUDA C, its PTX, and one
constant per kernel that lowered. A hand-written `gate.go` lists the constants
that must exist, so a kernel that cannot be lowered fails the build itself:

```console
$ go build ./...
kernels/prebuilt/gate.go:22:2: undefined: Scale
```

That covers a kernel which was regenerated and turned out not to lower. The
case it cannot cover — edited and _never_ regenerated — is caught by a test
that needs no GPU:

```go
func TestPrebuiltIsCurrent(t *testing.T) {
	if err := simt.VerifyPrebuilt(gocuda.Kernels(), prebuilt.Names()...); err != nil {
		t.Error(err)
	}
}
```

The embedded PTX is also what removes the compile from start-up. `Build` still
transpiles — that is what produces the hash the prebuilt is filed under — but
NVRTC is skipped when one matches:

|              | transpile + NVRTC |
| ------------ | ----------------: |
| JIT          |           28.7 ms |
| prebuilt PTX |        **0.9 ms** |

PTX is forward compatible, so one `compute_75` artifact serves every newer
device; an older one falls back to NVRTC. A machine with no CUDA toolkit builds
and runs from the committed PTX, and `gocuda generate -no-ptx` refreshes
everything but the PTX there.

## Track 2 — tile

The pipeline is built by running Go code; nothing touches the device until
`Materialize`:

```go
g := tile.New(ctx)
out := tile.Hypot(tile.FIR(tile.Scale(g.Input(re), 0.5), h), g.Input(im))
vals, err := out.Materialize()
```

Elementwise operations never become temporaries — they fold into one C
expression evaluated at the thread's index. The windowed operation cannot, so
it stages a shared tile, _including its halo computed through the same fused
expression_, and leaves its result in a local:

```cuda
// generated by github.com/CWBudde/gocuda/tile: 6 operations fused into one kernel
extern "C" __global__ void fused(float* out, int out_len, float* p0, int p0_len, ...)
{
	int base = (int)(blockIdx.x * blockDim.x);
	int t = (int)threadIdx.x;
	int i = base + t;

	int taps3 = p1_len;
	__shared__ float tile3[320];
	for (int k = t; k < (int)blockDim.x + taps3 - 1; k += (int)blockDim.x)
	{
		int src = base + k - (taps3 - 1);
		tile3[k] = (src >= 0 && src < out_len) ? (0.5f * p0[src]) : 0.0f;
	}
	__syncthreads();
	float v3 = 0.0f;
	for (int k = 0; k < taps3; k++)
	{
		v3 += p1[k] * tile3[t + taps3 - 1 - k];
	}

	if (i < out_len)
	{
		out[i] = hypotf(v3, p2[i]);
	}
}
```

`MaterializeStepwise` runs the same graph one kernel per operation, so the cost
of _not_ fusing is measurable rather than asserted.

## Measured

NVIDIA T550 Laptop (`sm_75`, 4 GB), CUDA 12.8 NVRTC, driver 580, go1.26.8,
11-core CPU.

FIR filter, 4.19M samples, 33 taps (`go run -tags cuda ./examples/fir`), across
several runs:

|                     |         time | vs one core |
| ------------------- | -----------: | ----------: |
| CPU, one core       |      87.5 ms |        1.0× |
| CPU, 11 cores       |      21.1 ms |        4.1× |
| GPU kernel only     | 1.03–1.07 ms | **~82–90×** |
| GPU incl. transfers |      10.5 ms |       ~8.5× |

Transpiling and compiling the kernel costs 28.7 ms, once — or 0.9 ms when the
PTX was generated ahead of time.

Tile pipeline, 4.19M samples (`go run -tags cuda ./examples/tilefir`):

|          | kernels |                                                   time |
| -------- | ------: | -----------------------------------------------------: |
| fused    |       1 |                                           14.3–14.9 ms |
| stepwise |       3 | 16.3–20.7 ms (1.15–1.39× slower, plus two temporaries) |

## What Go still cannot do

Honest limits, not papered over:

- **No ownership safety.** Rust's `DisjointSlice<T>` and `partition` prove at
  compile time that threads do not alias. Go has no borrow checker and no
  const generics; this repository substitutes runtime shape checks and a
  race-detectable CPU emulator. That is weaker, and knowingly so.
- **No recursion.** A kernel may call another Go function, but not one that
  reaches itself. That is a deliberate refusal rather than a gap: device
  stack depth is a launch-configuration problem, not a language one.
- **No multiple assignment**, so no tuple-returning intrinsic: the axes are
  read one accessor at a time. It is a limit of the emitter, not of the
  device.
- **No chained windowed operations** in the tile track: the halo of the outer
  window would need values the inner one does not have at those indices.
- **Source-level, not IR-level.** Without a real backend there is no
  optimisation, no register-pressure model and no PTX-level control. NVRTC
  does the optimising.

## What already exists in Go

| Project                                                                            | What it does                                            | Relation to this                                                                                       |
| ---------------------------------------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| [`gorgonia.org/cu`](https://pkg.go.dev/gorgonia.org/cu)                            | Idiomatic bindings to the CUDA driver API               | The host half, done properly. Kernels are still written in CUDA C                                      |
| [mumax3's `cuda2go`](https://github.com/mumax/3)                                   | Generates Go _wrappers_ from hand-written `.cu` kernels | The opposite direction: CUDA is the source of truth                                                    |
| [`gosl`](https://www.cogentcore.org/lab/gosl/) (Cogent Core, formerly `emer/gosl`) | Translates Go to WGSL compute shaders for WebGPU        | The closest existing work — Go as a shader language, portable across vendors rather than CUDA-specific |
| TinyGo                                                                             | Go on LLVM                                              | Registers `nvptx64`, but its Go version support and host-oriented runtime rule it out today            |

This repository's distinguishing bet is the **single source**: the kernel is a
Go function that both backends run, so correctness is testable without a GPU.

## Layout

```text
cuda/          CUDA driver API + NVRTC, loaded at run time (build tag "cuda")
gpu/           kernel vocabulary + CPU grid emulator
simt/          transpile, build and launch             (track 1)
tile/          lazy graph -> one fused kernel          (track 2)
internal/lower/   Go AST -> CUDA C; the one definition of the subset
analysis/simtcheck/  the go/analysis Analyzer behind "gocuda vet"
cmd/gocuda/    vet and generate                        (no driver needed)
cmd/gocuda-nvrtc/  the NVRTC child process             (build tag "cuda")
kernels/       the example kernels, embedded as source
kernels/prebuilt/  generated: CUDA C, PTX, and the build gate
internal/jit/  compile, cache and load, shared by both tracks
examples/      vecadd, fir, magnitude, tilefir
```

There is no cgo. `libcuda` and `libnvrtc` are opened with `dlopen` at run
time (through [purego](https://github.com/ebitengine/purego)), so the whole
module — driver bindings included — compiles with `CGO_ENABLED=0` on a machine
that has never had a CUDA toolkit installed. The `cuda` build tag still selects
between the driver and `cuda/stub.go`, whose calls all return `ErrNoCUDA`, so
the transpiler and its tests need neither a tag nor a GPU.

## Running it

```sh
go test ./...                 # transpiler, golden files, CPU emulator: no GPU needed
go test -race ./gpu/          # the emulator must be race-clean
go test -tags cuda ./...      # CPU/GPU parity on a real device
go run -tags cuda ./examples/fir

go run ./cmd/gocuda vet ./kernels        # refuse kernels that cannot be lowered
go run ./cmd/gocuda generate -check      # are the committed artifacts current?
go generate ./...                        # regenerate them (needs NVRTC)
```

Generated `.cu` and `.ptx` land in `.gocuda-cache/` for inspection. Pass
`simt.WithCacheDir("elsewhere")` to `simt.Build` to point it somewhere else, or
`simt.WithCacheDir("")` to turn it off.

Nothing is needed at build time: no headers, no libraries, no toolkit. At run
time a `-tags cuda` binary wants an NVIDIA driver (`libcuda.so.1`, installed
with the driver) and, only if it has to compile a kernel, `libnvrtc` from the
toolkit — a kernel whose PTX is prebuilt runs with no toolkit present at all.

The two are searched for differently, because they come from different
places. The **driver** is installed by the driver package and lands in the
dynamic linker's default path, so only `libcuda.so.1` and `libcuda.so` are
tried — a toolkit root is not consulted, and the toolkit's own
`lib64/stubs/libcuda.so` is deliberately never a candidate, since it exists to
satisfy a linker and fails every call. **NVRTC** is part of the toolkit, so
`$CUDA_PATH` and `$CUDA_HOME` are tried first (an explicitly configured
installation wins over the system one), then the linker's default path, then
`/usr/local/cuda` and `/opt/cuda`.

`GOCUDA_LIBCUDA` and `GOCUDA_LIBNVRTC` each name a file outright and replace
that search rather than heading it. When nothing is found, the error lists
every path tried.
