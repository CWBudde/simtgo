# gocuda — writing CUDA kernels in Go

A Go answer to NVIDIA's [_Introducing CUDA Rust: Two Tracks for Writing GPU
Kernels_](https://developer.nvidia.com/blog/introducing-cuda-rust-two-tracks-for-writing-gpu-kernels/).
Not by the same means — Go has no pluggable codegen backend and no macros — but
both of NVIDIA's tracks have a working analogue here, and both run real kernels
on real hardware.

A kernel is an ordinary Go function. It runs unchanged on a CPU emulator and on
the device, and what the subset accepts is [a written contract](SPEC.md) checked
against the implementation rather than whatever the emitter happens to do.

The project is under active development and the API is not frozen. What is
verified, and on what, is stated below.

## The two tracks

|                         | CUDA Rust                                                                                             | This repository                                                                                                            |
| ----------------------- | ----------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **SIMT track**          | A custom `rustc` codegen backend lowers `#[kernel]` functions through MIR and LLVM IR to PTX          | `go/ast` + `go/types` lower a Go subset to CUDA C, which NVRTC compiles to PTX at run time (package `simt`)                |
| **Tile track**          | A `#[cutile::module]` proc macro embeds the kernel AST in the host binary and JITs it through Tile IR | The graph is recorded at run time by ordinary Go calls, then fused into one generated kernel (package `tile`)              |
| **Safety**              | `DisjointSlice<T>`, launch contracts, const generics                                                  | Runtime shape checks; the CPU emulator plus `go test -race`                                                                |
| **When errors surface** | `rustc` rejects the kernel                                                                            | `gocuda vet` rejects it, and `go generate` makes an unlowerable kernel fail `go build`                                     |
| **Toolchain**           | Pinned nightly Rust, custom LLVM                                                                      | Plain `go1.26`, no cgo, NVRTC loaded at run time. One dependency, purego; the `gocuda` tool also uses `golang.org/x/tools` |

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

## Status

**In place.** Both tracks, verified against independent Go references on real
hardware. The supported subset is written down in [`SPEC.md`](SPEC.md) and
checked against the implementation in both directions — a refusal the code
enforces and the document omits fails the build, and so does a rule the document
claims that nothing pins. Twelve kernels, each with a golden file, an NVRTC
compile test and a CPU/GPU parity test. A differential fuzzer with three oracles,
all three running in CI daily and a longer search weekly. A kernel that cannot be lowered fails
`go build` rather than `main()`. No cgo anywhere, so the module builds and ships
on a machine that has never had a CUDA toolkit. `compute-sanitizer` clean on
`memcheck`, `racecheck`, `initcheck` and `synccheck`.

**Not yet.** Linux only; Windows is planned and macOS is not possible, since
NVIDIA ships no CUDA for it. The driver API is synchronous — no streams, no
events, no async copies — and `cuda.Context` is not safe to share between
goroutines, which is a known and reproduced bug rather than an untested claim.
The tile track is 1-D `float32` with seven operations and no reductions. One
architecture has been measured, `sm_75`, and CI has no GPU, so the parity tests
are unverified anywhere but the machine they were written on.

The roadmap is [`PLAN.md`](PLAN.md); the engineering record — what was measured,
what broke, and why things are the way they are — is [`docs/`](docs/).

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
extern "C" __global__ void FIR(float* __restrict__ y, int y_len, const float* __restrict__ x, int x_len, const float* __restrict__ h, int h_len)
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
assignment, parallel assignment (`a, b = b, a`), `if`/`else`, three-clause
`for`, `for i := range`,
`for i, v := range`, `switch`, `break`, `continue`, their labelled forms,
arithmetic, comparisons, indexing and conversions all translate.
`gpu.Sqrt`, `gpu.Hypot` and friends become `sqrtf`, `hypotf`; `ctx.GlobalID()`
becomes `(int)(blockIdx.x * blockDim.x + threadIdx.x)` — the cast is
load-bearing, because CUDA's built-ins are unsigned and mixing them into Go's
`int` comparisons would change answers; `ctx.SharedF32(n)` becomes a
`__shared__` array. Struct literals and field access translate too.

Grids and blocks have three axes. The unsuffixed accessors are `x`, which is
CUDA's own spelling, and `ThreadIdxY`, `BlockIdxZ`, `GlobalIDY` and the rest
are the other two; `GlobalID` also answers to `GlobalIDX`, because next to
`GlobalIDY` the bare name reads like an oversight. Each is one built-in rather
than a tuple. `x, y := ctx.GlobalID2()` is the _tuple_ form, which is still
refused: a device function lowers to one C return type, and returning two
would need out-parameters. Parallel assignment, which the subset now has, is a
different feature and does not bring it closer — an intrinsic returning a pair
would be destructured before any of that machinery ran. The host side matches: `Kernel.LaunchDim` and
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

Shared memory comes in one tile per element type — `ctx.SharedF32(n)`,
`SharedF64`, `SharedI32`, `SharedI64`, `SharedU32`, `SharedU64` — each becoming
a `__shared__` array of that type. `SharedF64` needs `//gocuda:float64` like
any other double. There is no `SharedBool`: nothing wanted one, and a
vocabulary is easier to widen than to narrow.

A tile whose size is only known at launch is spelled without one:
`ctx.SharedDynF32()` lowers to `extern __shared__ float s[];`, and its length
travels as a generated parameter the launch fills, exactly as a slice's does.
CUDA has a single dynamic `__shared__` block per launch, so a second such tile
is refused — NVRTC accepts a second `extern __shared__` declaration without a
word, of a different element type as readily as of the same one, and every one
of them names the same bytes. Two names that silently alias is the
mistranslation this emitter exists to refuse, and it is refused here rather
than left to a compiler that will not object. The host side is
`Kernel.LaunchShared`, which takes an element count; the emulator's is
`gpu.RunCPUShared`. A dynamic tile is also what lets a kernel stop needing
`AssumeBlockDim`, since the tile can be sized to the block instead of the block
to the tile.

Every pointer the emitter writes is `__restrict__`, and one the kernel never
writes through is `const` as well — proved across the call graph, so a slice
handed to a helper that writes it is not marked. `__restrict__` promises
something Go cannot: `VecAdd(ctx, c, a, b)` may be given one buffer three
times. So the promise is checked rather than assumed. `Kernel.Launch` compares
the device ranges it was given and refuses an overlap with an `*AliasError`,
and passing one slice twice to a helper that writes it is refused where it is
lowered. Two _read-only_ parameters may share a buffer, deliberately: what
`__restrict__` forbids is reaching a modified object through another pointer,
so `dot(x, x)` is sound. The CPU emulator cannot check any of this — it never
sees the caller's slices, because they arrive through a closure — and there a
kernel is ordinary Go, where aliasing is defined.

A kernel that sizes shared memory against a fixed block size says so with
`ctx.AssumeBlockDim(n)`. It emits no code; it records the requirement, so that
`Kernel.Launch` and the CPU emulator both refuse a mismatched launch instead of
quietly reading the wrong stretch of memory. `Build` likewise refuses a kernel
whose shared memory exceeds what the device offers per block.

`gpu.AtomicAddF32`, `AtomicAddI32`, `AtomicMinI32`, `AtomicMaxI32`,
`AtomicExchI32` and `AtomicCASI32` become `atomicAdd`, `atomicMin` and the
rest, and each returns the value the element held before, as the CUDA built-in
does. They take a **buffer and an index** rather than a pointer, because the
subset has no address-of: `&s[i]` is something the emitter writes and a kernel
can never say. That shape is also what makes the first argument checkable — it
must be a slice parameter or a shared tile, the only two things a kernel has
whose address means anything on the device, and anything else is refused with a
position rather than left to NVRTC.

The vocabulary stops where CUDA's overloads do at `compute_75` with no header,
which is all NVRTC has: `atomicMin`, `atomicMax` and `atomicCAS` have no float
form, so there is no `AtomicMinF32`, and `atomicAdd` has no `long long` form,
only `unsigned long long`, so 64-bit variants would each need a different
reinterpret cast. The CPU emulator serialises all of them on one lock, which
is atomic enough to be faithful and deliberately not enough to hide a plain
write racing an atomic — `go test -race` still reports that, because on the
device it is a race too.

`ctx.ShuffleF32`, `ShuffleXorI32`, `ShuffleDownF32` and the rest become
`__shfl_sync`, `__shfl_xor_sync`, `__shfl_down_sync`; `ctx.Ballot`,
`ctx.Any`, `ctx.All`, `ctx.ActiveMask`, `ctx.SyncWarp` and `ctx.LaneID`
complete the warp vocabulary, and `gpu.WarpSize` is 32. NVRTC declares every
one of them with no header included, which was measured rather than assumed.

**None of them takes a participation mask.** CUDA's `_sync` forms do; the
emitter writes `0xffffffff` and the Go-level contract is that every thread of
the warp reaches the call — the same trade the atomics made by taking a buffer
and an index instead of a pointer. The cost is real and worth stating: in a
block that is not a multiple of 32 that mask names lanes which do not exist,
which CUDA leaves undefined. Launch whole warps, and say so with
`AssumeBlockDim` if the kernel depends on it.

The emulator has no warps to borrow, so it builds one: a per-warp rendezvous
between goroutines, with the same report-rather-than-panic discipline the
shared tiles use. It diagnoses two things the device cannot — a call some
threads never reach, and lanes meeting in different warp calls — and its
`ActiveMask` is the arrival mask, which is not what the device would answer.
A thread that has returned is not diagnosed, because CUDA permits exactly that.

A kernel may call another function in its package, which is emitted as a
`__device__` function alongside it: a prototype for each one the kernel
reaches, then the definitions, then the entry point. Slice parameters split
into a pointer and a length there too, so the call passes both. Recursion is
refused — there is no stack depth on the device to spend on it — and so are
methods, generics, variadics, more than one result, and a _named_ result,
which would be a local the body assigns to and a bare return that carries it.
A function taking a `gpu.Ctx` **is** a kernel by the rule above, so calling one
is refused unless it says otherwise. `//gocuda:device` is the spelling to
reach for: it says what the helper _is_, and it is checked — on a function
taking no `gpu.Ctx` it is refused, so it cannot become decoration.
`//gocuda:ignore` still works and still means "not a kernel"; the `Ctx` then vanishes from the C signature as the kernel's
own does. A device function may declare a shared tile of its own — that is
block-scoped storage, which CUDA allocates once per function, and the bytes are
accounted onto the kernel that reaches it. `AssumeBlockDim` stays refused
there, and so does the dynamic tile: the first is a promise about a launch, and
the second reads a length that arrives as a kernel parameter a helper cannot
see. The CPU side needs nothing at all
for any of this: a device function is ordinary Go, so `RunCPU` runs the very
code the device compiles.

#### Types

| Go                          | CUDA                                            |
| --------------------------- | ----------------------------------------------- |
| `float32`                   | `float`                                         |
| `float64`                   | `double`, opt-in                                |
| `int32`                     | `int`                                           |
| `int`                       | `int`, **by value only**                        |
| `int64`                     | `long long`                                     |
| `uint32`                    | `unsigned int`                                  |
| `uint64`                    | `unsigned long long`                            |
| `bool`                      | `bool`                                          |
| `int8`, `int16`             | `signed char`, `short`, storage only            |
| `uint8`, `uint16`           | `unsigned char`, `unsigned short`, storage only |
| a named struct of the above | a CUDA `struct`                                 |
| `[N]T` of the above         | `T name[N]`                                     |

The 64-bit types are `long long` and never `long`, which is 8 bytes on Linux
and 4 on Windows.

`float64` needs `//gocuda:float64` on the kernel, or on its file's package
comment. The cost is invisible in the source — the device runs a double at a
fraction of the float32 rate, so a kernel that acquired one by accident would
be correct and far slower — and the opt-in makes that something somebody wrote
down. It covers the kernel's whole translation unit, device functions included;
a helper may not carry its own, for the reason `AssumeBlockDim` may not
either. `gpu.Sqrt` and friends stay `float32`; the double-precision
half is spelled `gpu.Sqrt64`, `gpu.Hypot64`, `gpu.Fmax64` and the rest, which
become CUDA's unsuffixed `sqrt`, `hypot`, `fmax`. Reaching one of those without
the directive is refused by name, and the refusal has to sit on the _call_
rather than on a type: `y[i] = float32(gpu.Sqrt64(2))` writes no `float64`
anywhere, so a check that waits for one to be declared never sees it.

**Go's `int` is 64-bit and CUDA's is 32-bit.** That narrowing is the one
deliberate infidelity, and it holds only for a value passed on its own, where
an index is bounded by the grid anyway. As a slice element, an array element or
a struct field it is not a lost high word but a different stride, and
`cuda.Upload` copies Go's layout regardless — so `[]int` is refused, along with
`int` as a struct field. It used to lower cleanly and return the wrong numbers.

`int8`, `int16`, `uint8` and `uint16` are **storage, not arithmetic**. They may
be a slice element, an array element or a struct field — `[]uint8` is what an
image buffer is — but never a variable, a parameter or a result, and no
operator accepts one. The widths match; the arithmetic does not. Go computes
`int8 * int8` in 8 bits and wraps, while C promotes both to `int` and truncates
only at the assignment, so `a, b := int8(100), int8(3); a*b/2` is 22 in Go and
−106 in C. Convert to `int32`, compute there, and convert back:

```go
v := int32(rgb[3*i])*77 + int32(rgb[3*i+1])*150 + int32(rgb[3*i+2])*29
out[i] = uint8(v / 256)
```

Some of those operators would in fact agree — a compound assignment and `++`
truncate at the store in both languages, and so do comparisons, `/` and `%`.
They are refused anyway, because a rule that holds for every operator is one a
reader can keep in their head, and because relaxing a refusal later costs
nothing while retracting an acceptance costs a release.

**A struct is laid out by Go and checked by CUDA.** The generated C carries
what `go/types` says about the type:

```c
struct Shape
{
	float Floor;
	int Count;
	double Bias;
};
static_assert(sizeof(Shape) == 16, "gocuda: Shape is a different size in CUDA than in Go");
static_assert(alignof(Shape) == 8, "gocuda: Shape is differently aligned in CUDA than in Go");
```

The emitter never models the C ABI. It states Go's numbers and lets the C++
compiler refuse them, so the guarantee holds wherever the kernel is built
rather than where it was written.

**Every hole Go leaves is declared**, as an `unsigned char gocuda_padN[k]`
member, the trailing one included. That is what makes the size assertion
enough to pin the offsets, and the argument is short: with every hole spelled
out the members already account for exactly Go's size, and C++ lays each member
at or after the end of the one before it, so `sizeof` can only match if nothing
further was inserted — and then every field sits where Go put it. The trailing
member is load-bearing rather than tidy: without it, a byte inserted earlier
could hide in the end slack and the size would still agree.

This is a belt on top of braces rather than a fix for a disagreement anybody
observed. NVRTC inserts exactly the holes Go does, so the assertions passed
before the padding existed too; what changed is that they now _imply_ the
offsets rather than merely being consistent with them. Offsets could not be
asserted directly because NVRTC compiles a bare string with no include path:
`offsetof`, `__builtin_offsetof` and `#include <cstddef>` are all unavailable,
re-measured against 12.9 rather than taken on trust.

Struct literals lower positionally and step over the padding, because
designated initialisers are C++20 and NVRTC defaults to C++17.

**An array is storage, not a value.** It can be declared, indexed, measured
with `len`, ranged over, and held as a struct field. It cannot be a parameter — Go passes a copy and C
decays the parameter to a pointer, so a write inside the function would reach
the caller's array — nor a result, which C cannot return at all, nor assigned
whole, which C++ will not do. Wrap it in a struct if it has to travel; both
languages copy that. `==` on a struct or an array is refused for the same kind
of reason: Go compares field by field and C++ gives a plain aggregate no
operator at all.

Everything else is **refused with a file and line**, never mistranslated:
allocation, interfaces, goroutines, tuple assignment from a call, methods,
embedded fields, and any import other than package `gpu`. That list is
illustrative rather than closed — the emitter carries some forty distinct
refusal rules, and a few are worth knowing because nothing about the Go source
suggests them:

- A variable spelled like a name the emitter generates. A slice `y` brings an
  `y_len` with it, and a C++ keyword is emitted with a trailing underscore, so
  a local called `y_len`, or one called `int_` beside a parameter called `int`,
  would be the same C variable as the generated one. Both compiled and returned
  wrong numbers until they were refused.
- `&^=`, though `&^` itself lowers — Go's only operator with no C spelling
  becomes `a & ~b`, and the compound form has no such rewriting.
- Parallel assignment in a `for` clause. It works as a statement; in an
  init or post clause it does not, because the temporaries the semantics
  require are declarations and C's comma operator carries only expressions.
- A negative constant lane offset to a shuffle: CUDA reads those as unsigned,
  so `-1` is lane 4294967295 rather than the neighbour the minus sign implies.

`simt/errors_test.go` pins
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
kernels/bad.go:11:2: a, b := f() is not supported in kernels: a device function lowers to one C return type
```

The analyzer runs the **same lowering** the transpiler does, rather than a
second opinion about it, so what it accepts and what `simt.Build` accepts
cannot drift apart. A function whose first parameter is a `gpu.Ctx` is a
kernel; `//gocuda:device` in its doc comment says it is a helper instead, and
`//gocuda:ignore` opts it out entirely.

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
extern "C" __global__ void fused(float* __restrict__ out, int out_len, const float* __restrict__ p0, int p0_len, ...)
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

The pointers carry `const` and `__restrict__` for the same reason the SIMT
track's do, and the promise is checked the same way rather than asserted:
`Materialize` refuses a launch whose output buffer overlaps one of the inputs.
Nothing can fail that check today — the output is a fresh allocation on every
call, so it cannot be a buffer the graph already holds — and it is there for
the step that changes that, device-resident values, where the caller supplies
the buffers. Two _inputs_ sharing one buffer is accepted, because `restrict`
forbids reaching a modified object through another pointer and an input is
never modified; `MaterializeStepwise` produces exactly that whenever one node
feeds both operands of an operation.

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
- **No tuple assignment from a call**, so no tuple-returning intrinsic: the
  axes are read one accessor at a time. Parallel assignment works; what is
  missing is the C ABI for a function with two results, which would be
  out-parameters. It is a limit of the emitter, not of the device.
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

## Documentation

| Document                                             | What it answers                                                       |
| ---------------------------------------------------- | --------------------------------------------------------------------- |
| [`SPEC.md`](SPEC.md)                                 | what the subset accepts and refuses — the contract, checked by a test |
| [`NUMERICS.md`](NUMERICS.md)                         | what the device does to a `float32`, and what a test may assert       |
| [`PLAN.md`](PLAN.md)                                 | what is still to be done, and in what order                           |
| [`docs/decisions.md`](docs/decisions.md)             | settled questions and the reasoning that settled them                 |
| [`docs/emitter-defects.md`](docs/emitter-defects.md) | every mistranslation found, and what catches it now                   |
| [`docs/verification.md`](docs/verification.md)       | the layers of checking, and what each one cannot see                  |
| [`docs/toolchain.md`](docs/toolchain.md)             | measured behaviour of the driver, NVRTC and PTX                       |
| [`docs/tile.md`](docs/tile.md)                       | the tile track's code generator, and where it stops                   |

## Layout

```text
cuda/          CUDA driver API + NVRTC, loaded at run time (build tag "cuda")
gpu/           kernel vocabulary + CPU grid emulator
simt/          transpile, build and launch             (track 1)
tile/          lazy graph -> one fused kernel          (track 2)
internal/lower/   Go AST -> CUDA C; the one definition of the subset
analysis/simtcheck/  the go/analysis Analyzer behind "gocuda vet"
cmd/gocuda/    vet and generate; calls NVRTC in process, no driver needed
kernels/       the example kernels, embedded as source
kernels/prebuilt/  generated: CUDA C, PTX, and the build gate
internal/jit/  compile, cache and load, shared by both tracks
internal/tolerance/ what "close enough" means, shared by every parity test
internal/fuzz/ the differential fuzzer: one IR rendered as Go and as a closure
internal/fuzz/hostrun/  the generated CUDA C through a host C++ compiler
examples/      vecadd, fir, magnitude, tilefir
docs/          the engineering record: decisions, defects, measurements
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

There is a `justfile` for the same commands under shorter names — `just check`
runs everything CI decides without a device, in CI's order, and `just fix`
applies what the linters and formatters can apply themselves. `just --list`
has the rest. It mirrors `.github/workflows/ci.yml`; the workflow is not
written in terms of it, so that a CI runner needs no `just` and a red job names
the step that failed.

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
