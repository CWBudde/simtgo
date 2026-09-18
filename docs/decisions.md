# Decisions

Questions that were settled, and what settled them. A decision here is one that
would be expensive to revisit — it shaped an API, a refusal rule, or what the
emitter is allowed to assume — so the reasoning is recorded rather than left to
be reconstructed from the code.

The roadmap is [`../PLAN.md`](../PLAN.md); this page is the part of it that
stopped being a plan.

## Refuse, never mistranslate

The rule everything else here is downstream of: anything outside the supported
subset gets a `Diagnostic` with a file and a line, and nothing is translated
approximately. A refusal costs a user five minutes; a mistranslation costs them
a wrong answer they have no reason to suspect.

It is not a slogan. Every entry in
[`emitter-defects.md`](emitter-defects.md) is a case where the emitter did
translate approximately, and in every one of them the kernel compiled, launched
and computed something else. `simt/errors_test.go` pins the boundary and
`simt/spec_test.go` checks it against [`../SPEC.md`](../SPEC.md) in both
directions — a refusal the code enforces and the contract omits, and a rule the
contract claims that nothing pins.

The rule has a corollary that shows up repeatedly below: **relaxing a refusal
later costs a line, retracting an acceptance costs a release.** Where the two
languages agree for some cases of a construct and disagree for others, the
whole construct is refused.

## CUDA C through NVRTC, not PTX or LLVM IR

|                    | CUDA C + NVRTC (today)                 | Direct PTX/LLVM IR     |
| ------------------ | -------------------------------------- | ---------------------- |
| Optimiser          | NVIDIA's, for free                     | yours to write         |
| Artifacts          | readable `.cu`, easy to diff and debug | PTX only               |
| Runtime dependency | `libnvrtc` (~60 MB)                    | none beyond the driver |
| Startup            | ~25 ms compile per kernel              | none, if AOT           |
| Control            | none over registers, scheduling, ABI   | total                  |
| Effort to extend   | low                                    | very high              |

**Stay on CUDA C through 1.0.** The C++ round trip has not been the limiting
factor in anything measured so far, and giving up NVIDIA's optimiser means
owning register allocation and instruction scheduling — a different project.

The ahead-of-time pipeline removed most of the startup argument on its own:
transpile-plus-compile falls from 28.7 ms to 0.9 ms when the PTX is prebuilt,
so the remaining case for emitting PTX directly is control, not latency. The
decision stays open as a measurement rather than a conviction — PLAN.md carries
a spike to emit PTX for `VecAdd` and compare — and Phase 5 is where it would be
revisited, if NVRTC turns out to be the performance ceiling.

## The driver API is synchronous, so there is no `context.Context`

Every driver call this repository makes is synchronous and uncancellable:
`cuCtxSynchronize`, `cuMemcpyHtoD` and `cuLaunchKernel` cannot be interrupted
once issued. Accepting a `context.Context` would advertise semantics that do not
exist, and a cancellation that does nothing is worse than no cancellation at
all, because callers write recovery code against it.

It belongs in Phase 4, where `cuStreamQuery` can genuinely `select` on
`ctx.Done()`. Adding it before then would be an API that has to change meaning
later.

## Atomics take a buffer and an index, not a pointer

`gpu.AtomicAddF32(s, i, v)` rather than something spelled like `&s[i]`.

This is forced rather than chosen: the subset has no address-of. `&s[i]` is
something the emitter writes and a kernel can never say, so a pointer-shaped
signature would have no legal argument.

The shape pays for itself twice, because it also gives the first argument
something to be **checked** against. It must be a slice parameter or a shared
tile — exactly the objects the transpiler's length table holds, and exactly the
ones that lower to something with a device address. Anything else is refused
with a position rather than left to NVRTC. A tile therefore works, and keeps
working when it is passed on to a device function, where it is an ordinary
pointer parameter.

The vocabulary stops where CUDA's overloads stop at `compute_75` with no header,
which is all NVRTC has: `atomicMin`, `atomicMax` and `atomicCAS` have no float
form, and `atomicAdd` has no `long long` form, only `unsigned long long`. A
64-bit signed variant would need a reinterpret cast of its own and the
one-Go-name-to-one-built-in table would stop holding, so those gaps are named in
`gpu`'s package comment rather than left to be rediscovered.

## Warp primitives are `Ctx` methods and take no mask

They are methods rather than package functions, and that is forced rather than
stylistic: a warp primitive is defined by _which_ thread calls it, and a
package-level Go function cannot learn which goroutine is calling. The atomics
can be package-level precisely because they are handed the memory they work on.

**None of them takes a participation mask**, for the reason the atomics take a
buffer and an index: it keeps one Go name mapping to one built-in and gives the
contract something checkable. The emitter writes `0xffffffff` and the Go-level
contract is that every thread of the warp reaches the call.

The cost is stated rather than hidden. In a block that is not a multiple of 32
that mask names lanes which do not exist, which CUDA leaves undefined, so a
kernel depending on whole warps should say so with `ctx.AssumeBlockDim`.

## Per-axis accessors, not `GlobalID2()`

Multi-dimensional indexing landed as `ctx.ThreadIdxY()`, `ctx.BlockIdxZ()`,
`ctx.GlobalIDY()` and the rest — one CUDA built-in per accessor — rather than as
the tuple-returning `x, y := ctx.GlobalID2()` originally proposed. The
unsuffixed names stay the x axis, which is CUDA's own spelling, so no existing
kernel changed.

The tuple spelling needs multiple assignment, which was a pinned refusal at the
time, so it would have dragged a second and larger feature in with it.

**When multiple assignment did land, the reasoning behind that turned out to be
wrong in both directions, and the correction is the useful part.** Parallel
assignment is _not necessary_ for a pair-returning intrinsic: `assign` sees the
statement before `expr` is ever called, so a pair could be destructured there,
in the same place `ctx.SharedF32(n)` is already special-cased. And it is _not
sufficient_: what a **user** function returning two values needs is
out-parameters in the generated C, which is ABI work that parallel assignment
touches nowhere. The two remain separate items.

## Narrow integers are storage, and every operator on one is refused

`int8`, `int16`, `uint8` and `uint16` may be a slice element, an array element
or a struct field — `[]uint8` is what an image buffer is — and never a variable,
a parameter or a result. The widths match; the arithmetic does not. Go computes
`int8 * int8` in 8 bits and wraps, C promotes both to `int` and truncates only
at the assignment, so `a, b := int8(100), int8(3); a*b/2` is 22 in Go and −106
in C.

Some of those operators do in fact agree — compound assignment and `++` truncate
at the store in both languages, as do comparisons, `/` and `%`. **They are
refused anyway**, and the diagnostics say which ones agree rather than claiming
a disagreement a reader could not reproduce. A rule that holds for every
operator is one somebody can keep in their head, and this is the corollary of
"refuse, never mistranslate" above: relaxing this later costs a line.

Go's `int` is refused wherever a layout is involved for a different reason —
see [`emitter-defects.md`](emitter-defects.md#int-read-at-a-different-stride),
where it is a live defect rather than a design choice.

## `//gocuda:float64` is checked on the call, not on a declared type

Double precision is opt-in because the cost is invisible in the source: the
device runs a double at a fraction of the `float32` rate, so a kernel that
acquired one by accident would be correct and far slower. The directive makes
that something somebody wrote down. It covers the whole translation unit, and a
device function may not carry its own, for the reason `AssumeBlockDim` may not
either — it is a statement about the kernel being built.

The enforcement point is the part worth recording. Every other `float64`
refusal lives in `ctype`, which sees a type somebody wrote down, and
`y[i] = float32(gpu.Sqrt64(2))` writes none: the argument is an untyped constant
and the result is converted away, so the kernel would have run a double it never
named. The permission is about what the device is asked to **execute**, so the
check belongs on the call. Membership of the `float64` helper table is what
requires the directive, which keeps a name added later from quietly escaping it.

## At most one dynamic shared tile

CUDA has a single dynamic `__shared__` block per launch. **NVRTC accepts a
second `extern __shared__` declaration without a word** — of a different element
type as readily as of the same one — and every one of them names the same bytes.

Two names that silently alias is exactly the mistranslation this emitter exists
to refuse, so the rule is enforced here, with a position, and the comment at the
site says NVRTC is not the backstop because it will not object. See
[`toolchain.md`](toolchain.md#what-nvrtc-accepts-that-it-should-not).

## Signed arithmetic goes through the unsigned type of the same width

Go defines signed integer overflow as wrapping; C leaves it **undefined**, for
`+`, `-`, `*`, unary `-` and `<<` alike. The generated code therefore had no
defined meaning on exactly the values Go does define, and the difference is
invisible: it compiles, and what it does depends on the optimiser.

The decision is that **the generated C should mean what the Go means**. Those
five operators on a signed integer are emitted through the unsigned type of the
same width. The conversion is once per _region_ of arithmetic rather than once
per operator, or the source would vanish under casts: `a + b*c` is one cast back
and three operands converted in, not a nest four deep. `/`, `%` and `>>` end a
region, meaning something different unsigned; `&`, `|`, `^` are left outside
too, so the rule has no exception.

It is not free in readability and that was the accepted cost — `h[i+1]` becomes
`h[(int)((unsigned int)(i) + 1u)]`. It is very nearly free in everything else;
the PTX measurements are in
[`emitter-defects.md`](emitter-defects.md#signed-overflow-defined-in-go-undefined-in-c).

Two limits remain, and [`../SPEC.md`](../SPEC.md) states both: this makes the C
_defined_, not _equal to Go_, for Go's `int`, which wraps at 64 bits where the
device wraps at 32; and `MinInt / -1` is still undefined on the device, which
routing through unsigned cannot fix.

## `pkg-config` is deliberately not used

The obvious answer to "where is CUDA installed?" is `pkg-config`, and it is the
wrong one here. `pkg-config` configures a **link**, and there is no link left to
configure: `libcuda` and `libnvrtc` are `dlopen`ed at run time through purego
and the module has no cgo at all.

So the question moved from build time to run time and the answer moved with it.
`cuda/library.go` decides which _file_ to open; it is deliberately untagged and
free of any loading, so `cuda/library_test.go` can check the search order on a
machine with no CUDA — the machine where a path bug actually bites. The policy
itself is in [`toolchain.md`](toolchain.md#which-file-gets-opened).

## MIT, and `libnvrtc` is never shipped

MIT rather than Apache-2.0 because there is nothing here to attach a patent
grant to and no NOTICE to propagate.

The redistribution position it rests on: `libnvrtc` is `dlopen`ed at run time
and never shipped, so the license covers only what the repository contains.
Keep it that way — vendoring a toolkit library would make the license question
somebody else's problem to answer.

## Linting is `golangci-lint` and `treefmt`, with seven linters left off

Trunk was dropped rather than migrated. An earlier version of the plan described
a `.trunk/trunk.yaml` pinning `go@1.21.0` against a `go 1.26` module; that file
has never existed in the repository, on any branch, so there was nothing stale
to fix. Recorded as a correction rather than a silent edit, because a plan that
describes files it cannot see is the same failure as a README that describes
code it does not have.

Seven linters are enabled beyond the standard set, each measured clean before it
was turned on. Seven more are left **off rather than suppressed** —
`errorlint`, `perfsprint`, `predeclared`, `gocritic`, `intrange`,
`wastedassign`, `revive` — because their findings are real and fixing them is a
change to the emitter and the driver that belongs in its own commit. A config
that excluded them would say the code is clean when it is not.

`golangci-lint` must be built with this module's own Go: the version check
compares the toolchain that built the linter against `go.mod`, and a release
binary built with an older Go refuses the module outright. CI uses the action's
`install-mode: goinstall` for that reason. `treefmt` cannot be `go install`ed at
any version — the module zip is rejected by the proxy itself over a non-ASCII
path in its own test data, and its GitHub releases are drafts — so CI builds it
from a pinned tag.
