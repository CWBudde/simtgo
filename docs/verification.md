# How this is verified

A transpiler is trusted through evidence, not review. This page says what
checks what, and — the part that matters more — **what each layer cannot see**,
because every defect in [`emitter-defects.md`](emitter-defects.md) slipped
past at least one of them.

The layers are ordered by cost. **The first six need no GPU** — of those, only
layer 4 needs a CUDA toolkit and layer 5 a host C++ compiler, and the rest need
nothing but Go. Only layers 7 and 8 need a device, which is what CI does not
have yet.

## 1. Golden files — the exact bytes the emitter produced

`simt/testdata/*.cu`, one per kernel, compared by `TestGolden` and refreshed
with `GOCUDA_UPDATE=1`.

Cheap, and the right check for "did this change the output, and where". It is
also the weakest claim in the set: a golden file says the emitter is
_consistent_, not that it is _correct_. Changing the emitter changes every
`SourceHash`, so goldens and `kernels/prebuilt/` both need regenerating
together.

**Cannot see:** anything about whether the C is valid, or whether it computes
the right thing.

## 2. The contract, checked in both directions

`simt/spec_test.go` compares [`../SPEC.md`](../SPEC.md) against
`simt/errors_test.go` and fails **in both directions**: a refusal the
transpiler enforces and the document omits, or a rule the document claims and
no test pins.

Both failure modes were watched failing before the check was committed, because
a cross-check nobody has seen fail is not a cross-check. It keys on the
diagnostic's own wording rather than on rule numbers invented for the purpose,
so the document quotes what a reader will actually see in their terminal.

Writing the spec was worth it for what the inventory turned up quite apart from
the document: the README's generated-CUDA sample still showed unqualified
pointers, its `GlobalID()` lowering was missing the cast that makes the axis
accessors return `int`, its type table marked the narrow integers
position-dependent and said nothing of the kind about `int`, and the refusal
list named seven rules out of seventy-two while reading as closed.

## 3. The two hand-maintained surfaces that must not drift

Both of these exist because something is written down twice and a compiler
cannot see that it is.

- `internal/lower/gpupkg_drift_test.go` compares `lower.GPUPackage()` — a
  `types.Package` for `gpu` built **by hand**, because run-time type-checking
  has no module graph — against the real package. Adding anything to `gpu`
  means adding it to `gpupkg.go` too, or kernels can call it in Go and fail to
  transpile.
- `cuda/surface_test.go` type-checks `package cuda` under both build tags and
  fails if the exported surfaces drift. Anything added to the driver needs a
  matching stub in `cuda/stub.go`.

The second was **impossible before the cgo removal**, because loading the
tagged half needed a toolkit — so the machine that most needed the check was
the one that could not run it.

## 4. NVRTC as an oracle, without executing anything

`simt/nvrtc_cuda_test.go` puts the emitter's sharp edges through NVRTC. This is
the question the golden and parity tests never ask: **is the generated C
something a compiler accepts?** Three of the defects in the catalogue are valid
Go lowering to C++ that does not compile, and this is the only layer that sees
them.

It is stronger than "it compiles" for one case. The emitter writes a
`static_assert` on every struct's `sizeof` and `alignof`, so **a wrong layout
fails to compile** — no execution required. A generated struct reaching
`cuda.Compile` is the whole test, and 250 fuzzer-generated ones did.

Needs `libnvrtc` but no driver and no device, which is what made it runnable on
a machine that had neither. Skips without a toolkit rather than failing — see
[below](#what-turns-a-skip-into-a-gate).

## 5. A host C++ compiler — the third backend

`internal/fuzz/hostrun` compiles the emitter's generated CUDA C with an
ordinary host C++ compiler behind a small shim and runs it. This is the layer
the original plan missed: the differential fuzzing item named two backends, and
there is a **third that needs no device at all**, which is what CI is.

"Compiles and computes something else" is therefore answerable on a machine
with no GPU. What still needs hardware is only what the device does differently
from any C++ implementation: the transcendentals, the warp primitives,
`fminf`'s treatment of a NaN, and what happens to a subnormal.

`-ffp-contract=off` is the load-bearing compiler flag and is not cosmetic;
[`../NUMERICS.md`](../NUMERICS.md) says why.

## 6. The CPU emulator, under the race detector

`gpu.RunCPU(grid, block, fn)` runs the kernel as goroutines, one per thread. A
device function is ordinary Go, so `RunCPU` runs the very code the device
compiles — the oracle is not a second implementation somebody has to trust.

`go test -race ./gpu/` must stay clean, and the emulator is built so that the
detector keeps its teeth. Atomics serialise on one package-level mutex, because
global memory there is the caller's own Go slice and there is nowhere
per-buffer to hang a lock. **What the lock does not do matters more:** it
creates a happens-before edge only between goroutines that take it, so a plain
write to the same element is still reported by `-race`, which is right, because
on the device that is a race too.

The emulator reports two things the device cannot — a warp call some threads
never reach, and lanes meeting in different warp calls — and its `ActiveMask`
is an arrival mask, which is not what the device would answer. Both are named
in `gpu`'s package comment.

**Cannot see:** aliasing. The kernel's slices arrive through a closure and
never pass through `RunCPU`, so there is nothing to compare — and there a
kernel is ordinary Go, where aliasing is defined. That is stated in `RunCPU`'s
doc comment rather than left as an oversight.

## 7. CPU/GPU parity, against an independent reference

`simt/parity_test.go`, under `-tags cuda`. Every kernel runs on the emulator
and on the device, and both are compared against **an independent Go
reference**, so a shared misunderstanding cannot pass as agreement.

What "agree" means is one rule in `internal/tolerance`, where there were three
conventions in three packages. [`../NUMERICS.md`](../NUMERICS.md) says when a
test may demand exact equality instead.

## 8. `compute-sanitizer`, four tools

The sweep wraps the tagged test binaries of `simt`, `tile` and `cuda` with
`go test -exec`. That is the whole surface: those are the only packages that
reach a device, and between them they launch all twelve kernels in
`kernels/prebuilt/gate.go`, the inline probes the parity tests build, the
hand-written kernel in `cuda`'s raw-API test, and the **tile track's fused
kernels** — a second code generator that a `simt`-only sweep would have missed
entirely.

```sh
GOCUDA_REQUIRE_DEVICE=1 go test -tags cuda -count=1 -timeout 0 \
  -exec "compute-sanitizer --tool=memcheck --error-exitcode 1 --report-api-errors no --target-processes application-only" \
  ./simt/ ./tile/ ./cuda/
```

**Measured on the T550**, all four tools, every package: `ERROR SUMMARY: 0
errors`, and `RACECHECK SUMMARY: 0 hazards displayed`.

The cost is not what the folklore says. `memcheck` takes `simt` from 8.6 s to
14.2 s, and `racecheck`, `initcheck` and `synccheck` are within noise of an
uninstrumented run. These kernels are small and there are not many launches;
the 10–100× figure is about neither.

### A clean result is an absence, so it was held to a planted fault

Before the zero was believed, a kernel with no bounds guard was launched with
256 threads over 16 elements. It passes `go test` and is **224
`Invalid __global__ write` errors** under `memcheck`. That is what makes the
zero a claim rather than a hope.

### Each flag is load-bearing

The default behaviour of every one of them is the wrong answer here.

- `--error-exitcode 1` — `compute-sanitizer` exits 0 on findings, so without it
  a clean run and a dirty one are the same result and the job is decoration.
- `--report-api-errors no` — the default counts a CUDA API call returning an
  error as an error, and `cuda`'s error tests hand `cuModuleLoadData`
  deliberate garbage on purpose. That is two "findings" the repository already
  tests more precisely than the sanitizer can, by asserting the `CUresult`.
- `--target-processes application-only` — the test binary is what launches, not
  a child.

### What turns a skip into a gate

`GOCUDA_REQUIRE_DEVICE=1` turns the test helpers' "no CUDA device available"
**skip into a failure**. The worst outcome available to a sanitizer job is to
go green having launched nothing, and a silent skip is exactly how that
happens. The GPU CI job will want the same switch for the same reason.

The mirror image of that argument applies to skips that should stay skips.
`TestGeneratedCCompiles` and `cuda`'s `TestNVRTCVersion` both needed only the
toolkit and neither skipped without one, so `go test -tags cuda ./...`
**failed** rather than skipped on a machine with neither toolkit nor device —
the machine where the emitter's own tests are most worth running, and the only
kind of machine CI has. Both skip now, which is what turns that command into a
gate rather than a known-red step.

### A green run prints nothing

`go test` discards a passing binary's stdout, so the `ERROR SUMMARY` lines
appear only on a failure — or under `-v`, which is how the numbers above were
read.

`.github/workflows/sanitizer.yml` is written against a runner that does not
exist yet: `runs-on: [self-hosted, gpu]`, one matrix leg per tool so one
finding cannot hide another, and `workflow_dispatch` **only** — a `schedule`
with no matching runner queues a job forever and reports nothing, which is
worse than not running.

## 9. Bounds checks on the device, behind a flag

`simt.WithBoundsChecks()` wraps every slice, array and shared-tile subscript
whose bound the emitter can name in a helper that calls `__trap()`. It is the
only layer here that needs no external tool and no second backend: the check
is in the generated C, so it runs wherever the kernel does.

**What it sees.** Every subscript the emitter wrote, on any device, including
the ones a parity test would never reach because the reference agrees with the
kernel about which indices are in range. Layer 8 sees more — a stray pointer,
an uninitialised read — but needs `compute-sanitizer` and therefore a machine
with one installed and matching the driver.

**What it cannot see.** A subscript whose bound the emitter cannot name is
emitted unchecked, because a build option must never change which programs are
accepted. A `range` loop's induction variable is skipped as provably dead —
the emitted loop is bounded by the same length — and a constant index into a
fixed-size array is skipped because `go/types` has already refused it if it is
out of range. A constant index into a slice, a shared tile included, is
checked.

**What the fault tells you: less than the emulator's.** `__trap()` carries no
payload, so the device can report only that it trapped. `gpu.RunCPU` on the
same kernel is Go, and Go's own bounds check names the index, the length and
the thread — so `simt.TrapError` says to go there. The asymmetry is pinned
from both ends: `gpu.TestOutOfRangeIndexIsDiagnosedWithTheIndex` holds the
emulator's message, and `internal/faulttest` holds the device's.

**`internal/faulttest` is deliberately outside layer 8's sweep.** A trap is a
real fault and the sanitizer reports it as one, so `--error-exitcode 1` would
turn a passing test into a failed sweep. It is also a package of its own
because a trap ends the whole process's use of CUDA — not merely the
context's, which was
[measured](toolchain.md#what-the-host-sees-after-a-__trap) — and `go test`
giving each package its own binary is the only isolation strong enough.

## The differential fuzzer

`internal/fuzz` generates random programs in the supported subset and compares
backends. It is the backbone of the whole approach, and it has found more
defects than every other layer combined.

### One IR, rendered twice

The generator builds one IR and renders it two ways: as **Go source** for
`simt.Transpile`, and as a **`func(gpu.Ctx)` closure** for `gpu.RunCPU`. The
closure is not an interpreter of Go semantics — it _is_ Go, running the same
operations on the same types — which is what keeps the oracle from being a
second implementation somebody has to trust.

### Measured, on seeds disjoint from the ones the tests use

The yield is **20000/20000**: every generated program is accepted by the
subset, so a fuzz run tests the emitter rather than the refusal path.

Feature coverage over 3,000 programs. This table used to be prose — measured
once, written down, and then unable to move when the generator did. It is
`TestFeatureCoverage` (`internal/fuzz/coverage_test.go`) now, which prints it
and holds each row to a floor, so a construct the generator quietly stops
reaching is a failing test rather than a stale sentence:

| Feature                 | Coverage |
| ----------------------- | -------: |
| array locals            |    97.8% |
| `switch`                |    80.4% |
| device funcs declared   |    74.9% |
| `range`                 |    70.8% |
| narrow element types    |    66.6% |
| `SyncThreads`           |    42.9% |
| shared memory           |    37.1% |
| structs                 |    32.0% |
| atomics                 |    32.0% |
| `float64`               |    25.8% |
| `LaneID`                |    24.3% |
| warp `_sync` vocabulary |    19.8% |

A row is "programs that reach this at least once", matched against the
generated Go source. Several figures differ from the hand-measured table this
replaced because the **definitions** differ, not because the generator does:
"array locals" counts a `var x [n]T` declaration and "device funcs declared" a
helper in the source, both of which the earlier figures evidently read more
narrowly. Treat this table as the baseline and the earlier one as superseded.

Structs were **0%** before the generator learned them, and are 32% now at an
unchanged yield. 250 struct-bearing programs went through NVRTC with none
refused — which is the test that matters, because the emitter writes a `sizeof`
and an `alignof` assertion for every struct and those fail at compile time when
a layout is wrong.

`LaneID` is counted apart from the rest of the warp vocabulary because it is
reached two ways — through `warpExpr` behind all seven gates, and as an
ordinary thread-varying position behind only `warpOK` — and a single figure
covering both would move when either did.

### Warp primitives, and the ceiling over them

The warp `_sync` vocabulary was the lowest row in the table at **10.7%**, and
the cause was structural rather than incidental: a warp expression sits behind
a conjunction of **seven** conditions while a barrier sits behind three and
owns a whole statement slot, which is the whole of why `SyncThreads` was at 43%
and warp primitives at a quarter of that.

None of the seven moved. Each guards real undefined behaviour — the emitter
writes `0xffffffff` into every `_sync` built-in, so a short warp names lanes
that do not exist; and `__any_sync` on the right of a short-circuiting `&&` is
undefined on the device and a deadlock on the emulator. What changed is the
draws **upstream** of them: a statement slot of warp's own, a second slot in
each of the three expression switches, and more whole-warp block widths (ten of
sixteen, from seven of thirteen). That took it to **19.8%** at a yield still
measured at 1.0, and a 120-second search on each of the host and NVRTC oracles
— 62,800 and 14,288 executions — found nothing.

There is a ceiling not far above. A warp primitive needs `g.sync`, a coin flip,
and `g.warpOK`, about 0.64: no more than a third of programs can carry one at
all. Raising `g.sync` is the obvious next lever and it is the wrong one — a
program with a barrier or a warp primitive is outside the host oracle's scope,
so that trade buys warp coverage directly out of the differential's reach.

### Struct shapes are chosen for their holes

A shape with no hole exercises none of the padding, so the catalogue is four
shapes picked for what they pad:

| Shape   | Hole                                   | What it tests                            |
| ------- | -------------------------------------- | ---------------------------------------- |
| `SPair` | none                                   | the control                              |
| `SHole` | three bytes after an `int8`            | an interior hole                         |
| `SWide` | four bytes to reach an `int64`'s align | alignment-driven padding                 |
| `STail` | six bytes at the **end**               | moves no field; only `sizeof` can see it |

`TestShapesLowerWithTheirPadding` pins each one, naming the hole it is there
for.

Two constraints shaped the design and the obvious approach does not survive
them. The shapes are **declared as real Go types**, not invented at run time,
because the closure renderer has to hold real values and there is no way to
make a Go type at run time. And the IR node is **field access, not a struct
value**: a `Field` node whose `kind()` is the _field's_ kind, which is what
keeps the change confined — every existing per-kind compiler in `closure.go`
handles the result unchanged, and the struct never has to become a kind of its
own at the ~200 `case K…` sites those compilers are made of. The cost is
per-(shape, field) typed accessors so the closure stays unboxed, which is the
property the frame exists to have.

### Which oracle takes which program

**41.1%** of generated programs are in the host oracle's scope — a barrier, a
warp primitive, an atomic or a shared tile means something a sequential run
cannot reproduce — and **34.8%** are also free of a transcendental, which the
differential excludes because `sinf` and Go's `math.Sin` are two
implementations of a function neither language requires to be correctly
rounded.

The NVRTC oracle takes all of it, because it never executes anything.

### Running continuously

`.github/workflows/fuzz.yml`, daily and on demand, **one matrix leg per
untagged target** so that one finding cannot hide the other.

The NVRTC oracle runs there too, in a job of its own, and getting it there cost
nothing but noticing that it needs a **toolkit and not a device**: NVRTC
compiles to PTX and nothing is launched, so the `nvidia-cuda-nvrtc-cu12` wheel
is the whole dependency and `GOCUDA_LIBNVRTC` points the loader at it. That is
the same trick [`toolchain.md`](toolchain.md) records for running
`TestGeneratedCCompiles` on a machine with no GPU. The version is pinned,
because the target whitelists four NVRTC diagnostic numbers as generator noise
and which numbers those are is a property of a particular NVRTC.

**That job sets `GOCUDA_REQUIRE_NVRTC=1`, and the reason is the whole point of
the job.** Without the library the target skips, and a skip is indistinguishable
from a clean search: with `GOCUDA_LIBNVRTC` pointed at a file that does not
exist, a twenty-minute leg passes **in three milliseconds** having compiled
nothing. The variable turns that skip into a failure, exactly as
`GOCUDA_REQUIRE_DEVICE` does for the sanitizer sweep.

Two schedules, because they answer different questions. The daily run is the
regression cadence — every defect the fuzzer has found turned up in minutes, so
twenty of them is generous for "did the last day's commits break something".
Whether a _deeper_ case exists is a question twenty minutes cannot answer at
all, and that is the weekly leg: four hours, once.

Three targets are committed in `simt/`, two of them running on every push. Go
runs a fuzz target's seeds as ordinary tests under plain `go test`, so the whole
host differential costs CI 2.3 s and no flag. The third,
`FuzzOutsideTheSubsetIsRefused`, is the only oracle there is for a rule whose
content is that something is **refused**: it puts each broken rule inside a few
hundred lines of generated control flow, where a rule that reads the wrong
scope stops firing and `simt/errors_test.go`'s three-line kernels would never
notice.

**A case the fuzzer finds is committed by hand**, after it has been read,
because a corpus entry is a test every future run pays for and an automated
commit would add them faster than anybody diagnoses them.

### What a corpus entry does not pin

An entry here is a **seed**, not a program: `int64(620)` and `int64(77)` are
what `fuzz.Generate` and `Program.Inputs` are handed. So a change to the
generator's draws changes the program that seed produces, and the entry goes on
running without going on testing the case it was named for. Raising the warp
coverage above made `signed-zero-620` stop skipping on the transcendental
filter and start passing, and made `signed-overflow-279` start skipping — two
entries, neither of which now reaches what it was committed for, and no test
went red to say so.

This is not an argument against seed entries; they cost almost nothing and they
do catch a case that comes back while the generator is still the one that found
it. It is an argument for what actually holds a diagnosed case down, which is a
hand-written regression test next to the fix. `TestFminFmaxAgreeOnTheZeros` is
the one for the signed zeros, and it is why that case is closed whatever seed
620 renders next.

### Barrier divergence

`internal/lower/diverge.go` is not a fuzzer target but belongs in the same
argument, because what it checks is a promise no backend can verify at run
time. Three rules: a barrier under a thread-varying condition, a barrier inside
a loop whose trip count is thread-varying, and a barrier preceded by a return
under a thread-varying condition. The middle one is what a lexical rule misses
and the one that matters here, because it is the shape every staging loop in
the repository has.

It is interprocedural — a barrier inside a helper is a barrier at the call site
— and follows `readonly.go`: memoised per declaration, a separate cycle guard,
pessimistic about anything it cannot read.

Two facts made it tractable and both were checked rather than assumed: Go's own
`goto` is refused, so every `goto` in the output is the emitter's and a
structured analysis over the Go AST is sound; and a barrier can only be an
`*ast.ExprStmt`, since it returns nothing and `simple()` would not take it in a
`for` clause.

**It found two live bugs**, and one of them was a fix from earlier the same
day. `TypedProbe`'s early return before a barrier had been turned into a guard,
which moved the problem rather than removing it: the guarded branch called a
helper that barriers inside, so the same thread missed the same rendezvous one
level down. And a fixture wrote `ctx.LaneID() == 0 && ctx.Any(v != 0)`, where
Go's `&&` short-circuits and so does the C — `__any_sync` ran on lane 0 alone
while the emitter passed a full-warp mask, the participation promise broken by
the spelling itself.

That second one is also **the honest limit**: the rules work over statements
and do not see into a condition. Extending them there would refuse
`ctx.Any(p) && ctx.All(q)`, which is legitimate, because `Any` and `All` are
warp-uniform and the pass holds the warp primitives to the stricter block
lattice. Closing it properly needs a warp level in the lattice. Both limits are
written down at the site.

## What is not verified

Stated plainly, because the rest of this page is a list of things that are.

- **One machine.** Everything measured here is an NVIDIA T550 Laptop (`sm_75`,
  4 GB), CUDA 12.8, driver 580, go1.26.8, Linux. That is one GPU, one
  architecture, one CUDA version, one OS, one Go version. The parity tests do
  not need CI to run, but they need CI to have run anywhere else.
- **Parity tests written and unrun.** The work of 2026-09-18 — atomics, the
  `float64` helpers, warp primitives, shared memory, structs — was written
  where there was no device. Its lowering, refusals, drift test, analyzer,
  emulator under `-race` and generated C are all verified; **every parity test
  it adds is written and unrun.** NVRTC compiles; it does not run.
- **The generate gate checks source, not artifacts.** `gocuda generate -check`
  compares the lowered CUDA C and **not** the compiled PTX: a deliberately
  corrupted `.ptx` passes it, while a tampered `.cu` fails. That is the right
  check for source staleness and a weaker claim than "are the committed
  artifacts current?".
- **Multi-GPU.** `NewContext` takes a device ordinal and the poison table is
  keyed by one, but this machine has one GPU, so two contexts on two devices
  have never been exercised. The goroutine-safety guarantee on `cuda.Context`
  is tested ([the finding](decisions.md#a-driver-call-holds-its-os-thread));
  the multi-device half of that roadmap item is not.
- **Overlap, on one copy-engine configuration.** The streams work is tested
  for correctness — an asynchronous round trip, a cancelled wait, event
  timing, the launch parameters surviving a collection — and those tests hold
  on any device. The _overlap_ claim does not: `BenchmarkOverlap` shows copies
  and compute running at once because this T550 reports
  `ASYNC_ENGINE_COUNT = 3` and `CONCURRENT_KERNELS = 1`. A device with one
  copy engine would overlap strictly less, and nothing here has run on one.
  What is measured is in
  [`toolchain.md`](toolchain.md#streams-overlap-the-copies-with-the-compute).
- **No GPU in CI.** `.github/workflows/ci.yml` is the non-GPU half. Nothing in
  it runs on a device, so the parity tests are unverified there, the sanitizer
  workflow has no runner, and the fuzzer's device leg does not exist. This is
  the single largest gap and it is Phase 1.4 in [`../PLAN.md`](../PLAN.md).
