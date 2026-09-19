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

## `context.Context` cancels the wait, and says so

This section used to read _"the driver API is synchronous, so there is no
`context.Context`"_, and the reasoning it gave was right: every call was
synchronous and uncancellable, so accepting a context could only have meant
ignoring it, and a cancellation that does nothing is worse than none at all
because callers write recovery code against it. It parked the question in
Phase 4, where `cuStreamQuery` would make polling possible.

Phase 4 arrived. `cuda.Stream.Wait(ctx)` is the one entry point in the package
that takes a context, and the premise that changed is that there is now
something to poll.

**What it does not do is stop the device.** There is no call in the driver API
that abandons queued work — no `cuStreamCancel`, no interruptible
`cuStreamSynchronize`. So cancelling `Wait` returns control to the caller and
leaves the kernel running, the copies in flight, and every buffer the stream
touches in use. The dangerous next line is not the cancellation itself; it is
the `Free()` that usually follows a wait returning, which would hand the
allocator memory the device is still reading. The host cannot see that go
wrong, because the memory is the device's.

That is why the honesty is enforced and not only documented. A `Stream` whose
`Wait` returned early is marked, and `Close` on it refuses with a
`*BusyStreamError` naming the situation; `Sync` clears the mark, because at
that point the work genuinely has finished; and `CloseAbandoned` is there for
a caller who means it. The refusal is the only moment between a cancelled wait
and a use-after-free at which anything can be said, so something is said.

`Wait` polls rather than blocking, which looks like the worse implementation
until the alternative is written out. A goroutine parked in
`cuStreamSynchronize` cannot be released by a closed channel — there is
nothing to select on — and parking a locked OS thread there so it can close a
channel when it returns makes the cancellation a lie in the other direction,
since the thread stays until the device is done either way. The poll backs off
from 10 µs to a 500 µs ceiling, so work already finished costs one query.

The convention the package comment promised holds: `context.Context` first and
named `ctx`, the CUDA context second and named `dev`.

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

## Linting is `golangci-lint` and `treefmt`, and the seven deferred linters are on

Trunk was dropped rather than migrated. An earlier version of the plan described
a `.trunk/trunk.yaml` pinning `go@1.21.0` against a `go 1.26` module; that file
has never existed in the repository, on any branch, so there was nothing stale
to fix. Recorded as a correction rather than a silent edit, because a plan that
describes files it cannot see is the same failure as a README that describes
code it does not have.

Seven linters were enabled with the scaffolding, each measured clean first.
Seven more were left **off rather than suppressed** — `errorlint`,
`perfsprint`, `predeclared`, `gocritic`, `intrange`, `wastedassign`, `revive` —
on the grounds that their findings were real and that fixing them belonged in
its own commit rather than in an exclusion list. Those commits are in
(2026-09-19); all fourteen are on, and what the seven actually found is below,
because the count is the interesting part of the answer either way.

**Seventy-one findings across the two build configurations, one of them a
latent defect.** `simt/transpile_test.go` sliced a string at
`strings.Index`'s result without testing it for −1, so a transpile that emitted
no entry point would have panicked in the slice instead of failing with the
source attached — in the one test that most needs to report what it got. The
same shape sat twelve lines below, where the author had already written
`decl < 0` into the assertion but the slice ran first. `gocritic`'s `offBy1`
flags the first spelling and not the second.

One more was wrong rather than merely untidy: `internal/fuzz`'s binary-chain
generator wrote its bound as `i < 2+g.r.IntN(3)` and so re-rolled it on every
iteration. The chain length was still 2 to 4, but not for the reason the code
appeared to give, and each re-roll consumed a draw. Seeds map to different
programs since; nothing pins that mapping, and the test that does pin
determinism — same seed, same program — still holds.

The seven `redefines-builtin-id` sites were not bugs and are worth naming
anyway: `cmd/gocuda`'s `renderGen` declared `any := len(compiled) > 0` and used
it twice, thirty lines apart, in a repository whose entire subject is which
type a value has.

The rest were what the deferral predicted, and the shape of the count is worth
recording:

| linter         |  n  | what it mostly was                                             |
| -------------- | :-: | -------------------------------------------------------------- |
| `perfsprint`   | 30  | `fmt.Sprintf` formatting one number, in the fuzz source writer |
| `revive`       | 29  | 13 `unused-parameter`, 8 `exported`, 7 `redefines-builtin-id`  |
| `intrange`     |  5  | `for i := 0; i < n; i++`                                       |
| `gocritic`     |  3  | the `offBy1` above, an `appendAssign`, an `ifElseChain`        |
| `errorlint`    |  3  | a type assertion where `errors.As` belongs                     |
| `wastedassign` |  1  | a value overwritten before anything read it                    |

`predeclared` reports nothing in that table and not because it found nothing:
it flags the same seven positions as `revive`'s `redefines-builtin-id`, and
`golangci-lint` deduplicates findings that share a position. Enabling it is
still worth doing — it is the rule that survives if `revive` is ever narrowed —
but it adds no coverage while `revive` is on.

**Two findings were excluded rather than fixed, and the reasons are in
`.golangci.yml` next to each.** `gocritic`'s `ifElseChain` is scoped out of
`kernels/`: those files are input to the emitter, so rewriting an if/else chain
as a tagless `switch` — which `SPEC.md` §3 says lowers back to an if/else chain
anyway — would change the generated CUDA C, every `SourceHash` with it, and so
every file under `simt/testdata/` and `kernels/prebuilt/`. Regenerating the
whole artifact set for a cosmetic rule is the wrong trade. `revive`'s
`unused-parameter` is scoped out of `cuda/stub.go`, where every parameter is
unused by construction and the names are the whole of the godoc a machine with
no CUDA gets: `Alloc(n int)` documents itself and `Alloc(_ int)` does not.
`cuda/surface_test.go` compares exported _names_ and would have passed either
spelling, which is exactly why the choice had to be argued rather than left to
the linter.

**A lint run sees one build configuration, and it is not the tagged one.**
Three of the seventy-one were invisible to `golangci-lint run ./...` and were
found only by passing `--build-tags cuda` by hand: a shadowed `min` in the
NVRTC version call, a hand-counted loop and a `vote[i] += 1` in the parity
tests. `just lint-cuda` is that run. It is not in CI, and while it is not, the
driver's own half is linted by nobody unless somebody remembers — which is why
running it before a release is written down here rather than assumed. That run
also reports two `errcheck` findings in `cuda/driver_cuda.go` that predate all
of this and are open work, not a decision.

`golangci-lint` must be built with this module's own Go: the version check
compares the toolchain that built the linter against `go.mod`, and a release
binary built with an older Go refuses the module outright. CI uses the action's
`install-mode: goinstall` for that reason. `treefmt` cannot be `go install`ed at
any version — the module zip is rejected by the proxy itself over a non-ASCII
path in its own test data, and its GitHub releases are drafts — so CI builds it
from a pinned tag.

## Bounds checks are a build option, and the checks are their own marker

`simt.WithBoundsChecks()` emits a range check at every subscript whose bound
the emitter can name. Four alternatives were considered, and each is rejected
for a reason this repository has already had to learn once.

**A `//gocuda:debug` source directive**, read the way `//gocuda:float64` and
`//gocuda:fastmath` are. Rejected because it puts the choice in the kernel,
where it gets committed, generated into `kernels/prebuilt/` and shipped.
Those two directives are properties of _what the kernel computes_ — a
precision, a set of compiler licences. Bounds checking is a property of _this
build_, and the roadmap's own wording is the distinction: "so the released
path pays nothing". For the same reason `gocuda generate` has no debug mode
at all; a bounds-checked artifact in `kernels/prebuilt/` would be the release
path.

**An environment variable.** Rejected for the reason `WithCacheDir` was: a
process-wide switch expressed in one corner of a program silently changes
what an unrelated library builds.

**Salting `SourceHash`.** Rejected because the hash has to keep meaning
_these exact bytes_, which is what lets a committed `.cu` explain its own
identity. A salt also has to be threaded into `internal/jit`'s cache key
separately, since that one hashes the source — two places to keep in step
instead of none.

**A separate cache namespace, or a `Debug` field on `jit.Request`.** Fixes
the module cache and not the registry: a prebuilt is found by what the kernel
lowered to, and there is no second key to consult.

So the checks go into the generated C, which is what `SourceHash` is taken
over, and the registry and the JIT cache inherit the split for free.

**A marker line is emitted anyway**, and it is worth saying why, because it is
redundant in every kernel that indexes anything. A kernel with no subscript at
all generates identical C either way, and without the marker its debug build
would find and load the release prebuilt. That load would in fact be
_correct_ — same bytes, same compilation — so this is not a correctness fix.
It is what lets "a debug build never finds a release artifact" be true with no
case split, and what lets someone reading a dumped `.cu` see which mode
produced it.

### What the two backends can each tell you

Asymmetric, and the errors say so rather than papering over it.

The emulator needed no code at all: a kernel under `RunCPU` is ordinary Go, so
`gc`'s bounds check is already there, already unconditional, and already names
the index, the length and the thread. The device can say only _that_ it
trapped — `__trap()` carries no payload, and the fault takes the process with
it. `simt.TrapError` therefore points the reader at `RunCPU`, which is where
the question "which index?" can still be answered.

### The alternative that would name the index, and its price

The helper could instead record `(site, index, length)` into a device buffer,
clamp the index and let the launch finish, so the host could print the exact
fault and keep the context alive. It is strictly more informative. It also
costs a generated kernel parameter that exists only in debug mode — so the
launch path, and not just the source, differs between the two — and it lets a
kernel carry on computing with wrong data, which is the opposite of what a
panic is for. The trap ships first, as the roadmap asked; this is recorded so
that adding it later is a decision with its trade already written down.

### The `gocuda_` prefix is reserved, in every build

The emitter writes two names of its own into the generated C: `gocuda_padN`,
the explicit padding a struct's holes become, and `gocuda_bounds`, the range
check `WithBoundsChecks` emits. Both are fixed spellings at file scope, and
both are perfectly legal Go identifiers — so a kernel package can declare
them, and then the emitter's definition and the author's are one C symbol. A
local named `gocuda_bounds` shadows the helper and NVRTC rejects the call it
cannot resolve; a device function of that name redefines it.

Neither is a mistranslation, which is the only reason this is a refusal rather
than an emitter defect: both fail loudly at compile time. What they fail with
is a message about code the author never wrote.

Two ways out were available. **Pick an unspellable helper name** — there is
none: every C identifier is a legal Go identifier, so any fixed spelling is
reachable. **Refuse the spelling** — which is what the field check for
`gocuda_pad` already did, generalised to the prefix.

It is refused **unconditionally**, not only when the checks are on, and that is
the part worth recording. A build option must not move the subset: a kernel
that `gocuda vet` accepts and `simt.Build(WithBoundsChecks())` then refuses
would make the analyzer wrong about what builds, and the analyzer has no build
options to be told about. The cost is that a name nobody would choose is
refused in release builds too, where nothing would have collided.

### A sticky fault belongs to the primary context, not to the `*Context`

`cuda.NewContext(0)` twice retains the same primary context and hands back two
Go wrappers around one `CUcontext`. The poison marker first lived on the
struct, so a trap observed through one wrapper left the other handing out bare
719s — which is the obscurity `ContextPoisonedError` exists to replace. The
marker is now a package-level cell keyed by **device ordinal**, which is what
identifies a primary context, and every wrapper of that device shares it.

Not keyed process-wide, although the measurement in
[`toolchain.md`](toolchain.md#what-the-host-sees-after-a-trap) found the whole
process finished with CUDA. That machine has one GPU; saying a fault on device
0 implies anything about device 1 would be a claim nothing has measured.

Two consequences. The entry **outlives `Close`**, which is correct: the same
measurement says a replacement context for that device cannot be retained, so
one made anyway is unusable before it is used. And every call that can be the
one to notice now records the fault — `cuMemAlloc`, `cuMemFree` and both
copies, not just `Sync` and the launch. A host program that allocates after
launching is the ordinary shape, and a call that recognised the code without
recording it left the _next_ one repeating the bare number.

### `TrapError` is raised for two result codes, not for every failed launch

The first version wrapped whatever `LaunchSync` returned whenever the kernel
carried bounds checks, and that was wrong in two directions. A launch refused
before it started — the wrong argument count, a block the device cannot fit,
`CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES` — is not a fault at all, so "its launch
faulted, which an index outside a slice would do" sends the reader hunting an
index that was never read. And once the context is poisoned every later launch
fails in `Context.bind` before dispatch: naming _that_ kernel accuses one that
did not run, and buries the `ContextPoisonedError` that says which one did.

So `Kernel.diagnose` passes a poisoned context straight through, and otherwise
wraps only `CUDA_ERROR_LAUNCH_FAILED` — what a `__trap()` was measured to
surface as — and `CUDA_ERROR_ILLEGAL_ADDRESS`, which is where an overrun the
checks did _not_ cover lands. Under a bounds-checked build those two are the
same mistake wearing different codes, which is what the error already says out
loud. Everything else is returned untouched.

## A driver call holds its OS thread

`cuCtxSetCurrent` binds a context to the **calling OS thread**, and Go moves
goroutines between threads at any preemption point. `Context.bind` and the
driver call after it were two statements, so a goroutine preempted between
them could finish its call on a thread the context had never been made current
on, and the driver answered `CUDA_ERROR_INVALID_CONTEXT`. This is not a data
race and the race detector cannot see it: no Go memory is touched
concurrently, only a per-thread state the driver keeps and Go does not know
about.

The fix is one choke point, `Context.call`, which locks the thread around the
bind and the call together. There is nowhere left to write the bug: a method
that forgets to lock also forgets to bind, and does not compile against a
`call` that takes the entry point as a closure.

Two other candidates were considered.

|                                          | what it fixes             | why not                                                                                                                                         |
| ---------------------------------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `LockOSThread` + `cuCtxSetCurrent`       | the migration window      | — chosen                                                                                                                                        |
| `cuCtxPushCurrent`/`cuCtxPopCurrent`     | nothing, on its own       | the goroutine can still migrate between push and call; it needs the lock too, and then only adds restoring a context somebody else made current |
| bind once per thread, and prove it stays | the repeated `SetCurrent` | Go offers no hook on thread creation, so there is no "once" to hang it on — an optimisation over the first row, not an alternative to it        |

Push/pop is the polite form: it gives a thread back whatever context was
current on it before. Nothing needs that today, because Go's threads are Go's
and no host application shares them, and it costs a second driver call on
every operation. If gocuda is ever called on a thread it did not create, this
is the row to revisit.

### What it costs, and what the measurement actually was

One OS thread pinned for the duration of one driver call. Those calls are
microseconds to milliseconds and block in the driver anyway, so the scheduler
loses nothing it could have run there; `LockOSThread` itself is a couple of
pointer writes. `go test -tags cuda ./...` takes the same time it did.

The interesting part is the reproduction, because the roadmap described a
shape that **does not fail**. 64 goroutines doing 40 `Upload`/`Download` round
trips each against one context passes every time on the unfixed code, measured
here. The reason is that `cuCtxSetCurrent` is sticky: once a thread has been
made current it stays current, so in a steady thread pool every migration
lands somewhere already bound, and the bug is invisible.

What is needed is a thread that has **never** bound — one the runtime created
after the work started. `TestContextIsGoroutineSafe` manufactures those: a
goroutine that exits while still holding `runtime.LockOSThread` takes its OS
thread down with it, so a loop of those retires threads continuously and
obliges the scheduler to make fresh ones for the workers. On the unfixed code
that fails in about 0.4 s, 15 runs out of 15, with three to six of 32 workers
reporting `CUDA_ERROR_INVALID_CONTEXT`; with the fix, 15 of 15 pass, under the
race detector too.

That is worth stating plainly: a concurrency bug that a thread pool hides is a
bug that reaches production and not CI. Measured on the T550 — one machine is
one machine, but the mechanism is in the driver's contract rather than in this
one's behaviour.

### What is still not promised

Ordering. Every call is synchronous and has finished before it returns, but
two goroutines allocating, copying and launching against one context interleave
however the scheduler likes, and nothing serialises them. A device buffer
shared between goroutines needs the same care as any other shared memory. That
is now written on `cuda.Context` rather than left to be inferred.

Multi-GPU is untested, not unsupported: `NewContext` takes a device ordinal and
the poison table is already keyed by one, but this machine has one GPU, so
nothing here has exercised two contexts on two devices.

## The PTX cache is on by default, and only its location is an environment variable

The persistent cache in `internal/jit/diskcache.go` is on unless a caller says
`simt.WithoutDiskCache()`. An opt-in cache does not do the thing the roadmap
asked for — taking NVRTC out of process start — because the programs that
would most benefit are the ones that never learn the option exists.

`GOCUDA_PTX_CACHE` moves the directory, and that is not the environment
variable this file
[rejected for bounds checks](#bounds-checks-are-a-build-option-and-the-checks-are-their-own-marker).
The objection there was that a process-wide switch expressed in one corner of a
program silently changes what an unrelated library _builds_: different bytes,
different behaviour, and nothing at the call site saying so. A location says
where bytes are kept and never what is compiled, so no program's output changes
because another part of it set the variable. `internal/fuzz/hostrun` already
does the same thing one layer down with `GOCUDA_HOSTRUN_CACHE`.

Whether the cache is consulted at all stays a build option, for exactly the
rejected reason: that one _is_ about whether a compiler runs.

### Not the same thing as `WithCacheDir`

Two caches, deliberately separate, and conflating them was the first design.

|           | `WithCacheDir` / `.gocuda-cache`  | the PTX cache                       |
| --------- | --------------------------------- | ----------------------------------- |
| For       | a human to read                   | the next process                    |
| Named     | after the kernel                  | after a hash of what produced it    |
| Holds     | the `.cu` and the `.ptx` that ran | PTX only                            |
| Read back | never                             | that is the point                   |
| Keyed on  | nothing — it overwrites           | source, architecture, NVRTC version |

A dump named `FIR-a1b2c3d4e5f6.cu` is the right thing for reading and the
wrong thing to load from: the 12 hex digits in that name are a convenience,
not collision resistance, which is stated where they are produced. Reusing it
would have meant hanging a module lookup on a truncated hash.

### The key is length-prefixed, and a test says why

`sha256("a\0b" + "\0" + "c" + "\0" + "d")` and `sha256("a" + "\0" + "b\0c" +
"\0" + "d")` are the same digest for two different triples. A NUL is not
reachable in any of the three terms today — the source is generated CUDA C,
the other two are short machine-written strings — but "not reachable today" is
how a joined key becomes ambiguous later, and a length prefix is one line.
That was found by the test rather than by reading the code.

## A buffer outside the Go heap holds no pointers

`cuda.HostSlice` is generic over `any`, which is more than it can honour, so
`NewHostSlice` refuses an element type carrying a pointer — a pointer, string,
slice, map, channel, function, interface, or a struct or array containing one.

The memory comes from `cuMemHostAlloc` and the garbage collector does not scan
it. Store the only reference to a Go object in a `[]T` over that memory and
nothing keeps the object alive: the collector cannot see the reference, frees
the target, and a later read through the slice returns a value that is simply
wrong. No panic, no race-detector report, nothing to grep for — which is what
makes it worth a refusal rather than a warning in a doc comment.

`runtime.Pinner` does not rescue it. A `Pinner` holds one object for as long
as the `Pinner` itself lives, and a buffer the caller fills whenever they like
has no such moment to hang it on. That is the same reason the asynchronous
copies refuse a `[]T`, one section down, and the two restrictions come from
one fact about this memory rather than from two rules.

Zero-sized element types are refused by the same check, for a duller reason:
there is no allocation for the driver to make, so a `Len()` of a thousand
would have no memory behind it and `Slice()` could not agree with it.

The check is `reflect`, once, at allocation, before the context is touched —
so it costs nothing against a driver call, and it is tested without a GPU,
which is why the rule lives in an untagged file.

**`cuda.Slice` is not yet guarded the same way**, and should be: its device
memory is not scanned either, but `Download` builds a `[]T` on the Go heap and
fills it with device bytes, so a pointer-bearing `T` would hand the collector
addresses to follow. It is a pre-existing hazard rather than one this rule
introduced, and `PLAN.md` carries it.

## An asynchronous copy does not take a Go slice

`Slice.UploadAsync` and `Slice.DownloadAsync` take a `*cuda.HostSlice` — a
buffer from `cuMemHostAlloc` — and refuse a plain `[]T`. Every other copy in
the package takes a `[]T`, so this is the one asymmetry in the API and it is
worth the paragraph.

Two independent reasons, and either alone would settle it.

**The driver would not have made it asynchronous.** `cuMemcpyHtoDAsync` out of
pageable memory is documented as falling back to a synchronous transfer: the
copy engine needs a physical address that will not move, and ordinary host
memory does not have one, so the driver stages through a page-locked bounce
buffer of its own and blocks while it does. A `[]T` overload would therefore
have compiled, run, returned an error-free `nil`, and quietly not overlapped
anything — the worst available outcome for a feature whose entire purpose is
overlap.

**`runtime.Pinner` cannot hold Go memory still for long enough.** This package
shows Go memory to the driver in three other places — the launch parameter
block, the PTX image, the device-name buffer — and each pins it for the
duration of the call, which is exactly as long as the driver needs it. An
asynchronous copy breaks that: the call returns and the transfer has not
started. Pinning until the stream drained would mean the `Pinner` outliving
the function that created it, owned by the stream, released by whatever
eventually synchronised — an ownership problem invented purely to keep a
convenience overload. Memory from `cuMemHostAlloc` is not Go memory, is not
the collector's business, and has none of this.

So `HostSlice` is not merely the faster option for asynchronous work. It is
what makes the operation expressible. That is also why it landed in the same
batch: the "overlap copy with compute" item could not have been demonstrated
without it, only claimed.

The synchronous `CopyFrom`/`CopyTo` keep taking a `[]T`, and for the symmetric
reason — they finish before they return, so there is nothing to outlive. A
`HostSlice` works there too, through `Slice()`, and is simply faster.
