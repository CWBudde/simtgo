# Emitter defects, and what catches them now

The record of what this emitter has got wrong, in four groups.

**Live defects** come first, and they are the ones the page exists for: valid Go
lowered to CUDA C that **compiled, launched and computed something else**. That
is precisely the failure
["refuse, never mistranslate"](decisions.md#refuse-never-mistranslate) exists to
rule out. After them, **latent defects**, correct only because nothing reached
them yet — including generated C++ that would not have compiled at all;
**infrastructure defects** in the repository's own tooling and tests; and
defects in the **oracles**, kept separate because a differential that reports a
mismatch has named a finding about one of its two backends without yet saying
which.

The useful part of each entry is not the embarrassment but the last line: what
would catch it today, and whether anything would have caught it then.

The pattern across the list is worth naming up front. A golden test pins the
bytes the emitter produced; a parity test pins that two backends agree. Neither
can see **valid Go lowering to C++ that does not compile**, and neither can see
a case that no committed kernel reaches. Most of what follows was found by
something else — NVRTC, a host C++ compiler, a differential fuzzer, or a device
returning the wrong numbers — which is why
[`verification.md`](verification.md) is organised as layers.

## Live defects — wrong numbers on real hardware

### `[]int` read at a different stride

`ctype` mapped Go's `int` to C's, a narrowing that is fine for a value passed on
its own and is a different **stride** for an element: `cuda.Upload` copies 8
bytes each and the kernel read 4.

On the T550, doubling `[1 2 3 4 5 6 7 8]` returned `[2 4 6 8 0 0 0 0]`. It
lowered, compiled, launched and answered wrongly, and it had been there since
the narrowing was written.

`int` in any position with a layout is refused now — as a slice element, an
array element or a struct field. As a value passed on its own the narrowing
stands, and it is the one deliberate infidelity
[`../SPEC.md`](../SPEC.md) records; an index is bounded by the grid anyway.

### A labelled `break` left the wrong loop

`*ast.BranchStmt` was lowered as `t.line("%s;", s.Tok)` and never looked at
`s.Label`, so `break outer` inside a nested loop emitted a bare `break;` and
left the **inner** loop.

Present since the first commit. Labelled branches lower to a `goto` now.

### An unnamed parameter shifted every argument after it

`params()` contributed nothing for `func f(ctx gpu.Ctx, _ []float32, y []int32)`,
so the C signature was one parameter short and every argument after the gap went
to the wrong place.

### Generated names collided with the kernel's own

A slice `y` lowers to a pointer plus an `y_len`, and a C++ keyword is spelled
with a trailing underscore. **Neither name appears in the Go source**, and only
the first was checked — and only against the _other parameters_, so a local
reached neither check.

A kernel with a slice `y` and a local `y_len` compiled and read 3 wherever it
said `len(y)`. One with a parameter named `int` and a local named `int_`
compiled and read the local twice. Both were confirmed through NVRTC before
being fixed.

The fix is the Phase 0 parameter-collision check widened from the parameter list
to **every variable the function declares**. It asks `types.Info` for variables
that are not fields rather than reusing `collectNames`, whose bluntness is right
for choosing a fresh generated name and would here refuse a kernel over a struct
field that collides with nothing.

### A `range` index the body assigns to became the loop counter

`for p := range x { … p-- … }` emitted `p` _as_ the C loop variable, so `p--` in
the body decremented the loop against its own `p++` and **the kernel never
terminated**.

Go's range variable is per-iteration. The fix is what the range _value_ had
always done: bind a copy. Found by the differential fuzzer, and it also found a
hole in the fuzzer's own oracle — see
[below](#the-oracle-had-its-own-bugs).

### `-(-c)` emitted as `--c`

C++ lexes `--` as predecrement. On a modifiable lvalue that compiles,
_decrements the variable_ and yields the decremented value.

Found by the differential fuzzer in its first minutes. Invisible to a golden
test (nothing committed writes `-(-x)`) and to a parity test (the CPU emulator
runs the Go, which is correct).

### `a := a + 1` emitted as `int a = a + 1;`

C++ starts the new name at its **declarator**; Go starts it at the end of the
declaration. So the C read the variable being declared instead of the one being
shadowed.

Found by the fuzzer alongside the one above. NVRTC's "used before its value is
set" is the signal.

### `shadowRename` reserved the escaped name

The fix for the previous entry had a defect of its own: it reserved the
_escaped_ name, so `double := double + 1` reserved `double_`, found nothing
called that, and handed back the name the outer variable already had. **The
rename renamed nothing and the defect it exists to fix was still there.**

Caught by the same NVRTC diagnostic that found the original.

### The `if`/`else` switch lowering re-evaluated the tag in every arm

A `switch` that is not all-constant becomes the `if`/`else` chain Go's semantics
describe, and each arm compared against a freshly emitted copy of the tag
expression.

This was **harmless when it was written** — nothing in the subset had side
effects — and stopped being harmless one commit later, when device functions
arrived and a tag could write through a slice. Two features, each correct alone.
It is the entry to remember: the bug was created by a change to a different
part of the emitter.

### Signed overflow: defined in Go, undefined in C

Go defines signed integer overflow as wrapping; C leaves it undefined for `+`,
`-`, `*`, unary `-` and `<<`. The generated code had no defined meaning on
exactly the values Go does define, and the difference is invisible: it compiles,
and what it does depends on the optimiser.

The fuzzer hit it through `11 - (x << 63)`, where `x << 63` is `MinInt64` and the
subtraction overflows. `fuzz.Generate(-279)` with inputs `205` reproduces it,
and the host and the emulator disagreed by whole powers of two on every element.

Fixed by routing those operators through the unsigned type of the same width —
the reasoning is in [`decisions.md`](decisions.md#signed-arithmetic-goes-through-the-unsigned-type-of-the-same-width).
**What it cost, measured:** six of the twelve kernels changed, and of their PTX
three are byte-identical (FIR, Quantize, Transpose), Gray is the same size,
BandGain grows 28 bytes, and **Classify shrinks by 483**. Removing an assumption
the optimiser was entitled to make did not cost instructions; in one case it
saved them.

### Go's `min`/`max` disagree with CUDA's about NaN

Go's builtins propagate a NaN operand; CUDA's `fminf` and `fmaxf` ignore one. So
`min(0.0/0.0, x)` is a NaN in Go and `x` on the device.

This had been recorded since the numerics work, in the same sentence that said
nothing tested it and no committed kernel reached it. **The fuzzer reached it in
six seconds**, which is the whole argument for having one. It is refused now,
pointing at `gpu.Fmin`/`gpu.Fmax`, which already mean the device's answer on
both backends — the same resolution the `int` narrowing got, for the same
reason. The integer overloads are untouched.

## Latent defects — correct only because nothing reached them

### `LaunchSync` gave every parameter a fixed 8-byte slot

Harmless while nothing wider than 8 bytes existed; a 16-byte struct passed by
value would have had its tail dropped.

Reverting the fix makes `TestStructLayoutRoundTrip` report `Bias` as 0 and
`TestBandGainParity` miss by exactly `Bias*Count`, so both are regression tests
for it rather than tests that happen to pass.

### `constant()` rendered every float at 32 bits

With an `f` suffix, which would have rounded every `double` silently and turned
anything past a float's range into `+Inf`. It also refused a legal `uint64`
above `MaxInt64` as "does not fit in an int64".

### Identifiers copied verbatim

The emitter copied any identifier through, so a kernel using a package-level
variable lowered cleanly and failed inside NVRTC as `identifier is undefined` —
a C error about code the author never wrote. The blank identifier was emitted as
a C identifier for the same reason.

### Valid Go lowering to C++ that does not compile

Three of these, found by review and none visible to a golden or a parity test:

- A labelled `continue` jumped into the scope of a later declaration.
- Two `case` clauses declaring one name collided in the switch's scope.
- A device function with a named result referred to a local nobody emitted.

`simt/nvrtc_cuda_test.go` now puts the emitter's sharp edges through NVRTC,
which is the question the other tests never asked: **is the generated C
something a compiler accepts?**

## The Phase 0 eight

Small, verified defects found by reading rather than by running. All fixed.

| Defect                                                            | Why it mattered                                                 |
| ----------------------------------------------------------------- | --------------------------------------------------------------- |
| Parameter name collision: `func K(ctx, x []float32, x_len int32)` | emitted `int x_len, int x_len` — a duplicate C parameter        |
| Symbol table keyed by identifier _string_                         | shadowing could resolve to the wrong symbol                     |
| Unvalidated launch geometry                                       | `ctx.SharedF32(FIRBlock + FIRMaxTaps)` assumed a block size     |
| Unvalidated shared-memory size                                    | no comparison against the device's per-block limit              |
| Precedence guessed by inspecting rendered text in `paren()`       | parenthesisation not derived from the C grammar                 |
| Emulator divergence: `SharedF32` matched calls by execution order | a mismatch handed back the wrong buffer, and only a doc said so |
| No module teardown: `cuModuleUnload` never called                 | the process-wide JIT cache grew without bound                   |
| Unreachable `_` branch under `token.DEFINE`                       | dead code                                                       |

Two changes to the public API came out of it: `Transpile` returns a `*simt.Unit`
carrying `Source`, `RequiredBlock` and `SharedBytes`, and a kernel that sizes
shared memory against a fixed block declares it with `ctx.AssumeBlockDim(n)`.
Module lifetime moved onto `cuda.Context`, which unloads everything it owns on
`Close`.

Two more were found along the way and are the same defect twice:

- The JIT module cache was package-level and keyed only on (source, arch), so a
  cache hit could return a module belonging to an **already-released context**.
  It is per-context now, which is the same fix as the teardown item.
- `jit.Load` reported empty `PTX`/`Log` on a cache hit, because the assignment
  sat inside the miss branch.

## Infrastructure defects

### NVRTC's remark swallowed the warning behind it

The fuzzer treats a clean compile that still warned as a finding, after
filtering out the four diagnostic numbers the generator's own filler produces.
The filter finished by asking whether what was left began with NVRTC's
`Remark: The warnings can be suppressed`, and read yes as "the log was
entirely noise".

But the remark is printed after the **first** warning, not at the end. So a
log whose first warning was allowlisted and whose second was a finding
filtered down to a residue starting with the remark, and the finding went out
with it. The diagnostic that arrives second is as likely as any to be `#549-D`
— the one that caught both halves of the shadowed-initialiser defect above.

Found by reading the filter against real `nvcc` output rather than against the
idealised log the code assumed. The remark is boilerplate and is now dropped
wherever it appears, which makes an empty residue mean what it says. Pinned by
`TestRemarkDoesNotSwallowALaterFinding` in `simt/triagefilter_cuda_test.go`.

### The noise allowlist is keyed on something that does not carry the distinction

Still open, and not fixable where it lives. `generatorNoise` decides that a
warning says nothing about the translation by looking up its **diagnostic
number**, and the number does not separate the two cases. An emitter that
declares a variable the Go source uses and then drops the use produces
`#177-D: declared but never referenced` — the same number, and very nearly the
same sentence, as the generator's own unused filler. `#550-D`, `#186-D` and
`#128-D` each have such a twin.

The allowlist therefore suppresses that whole class silently, and no amount of
work on the list fixes it: the input it keys on lacks the information.
`TestAllowlistCannotTellFillerFromADroppedUse` records the blind spot rather
than closing it. What closes part of it is the triage in
[`verification.md`](verification.md#10-nvrtc-warning-triage-opt-in), which
re-reads the suppressed warnings against the Go source they came from.

### `//go:embed kernels/*.go` matched `*_test.go`

Which then joined the type-check and failed on `import "testing"`: **a kernel
package could not have tests at all.**

### `sharedSize` gave one answer for two questions

It returned the same value for "not a shared buffer" and "a shared buffer I have
already complained about", which under multi-diagnostic reporting produced a
second, wrong diagnostic for one mistake.

### `parseArchLocal` was the worse of two implementations

`cmd/gocuda` carried its own arch parser. Any trailing `a` or `f` was read as an
arch-conditional target without looking at what preceded it, so `compute_a` was
refused with advice naming a string the function itself rejects. It is gone;
every case its test pinned behaves identically under `cuda.ParseArch`.

### Nothing had exercised the toolkit-missing path

Every test in `generate_test.go` sets `-no-ptx`, so the branch that calls NVRTC
and finds no toolkit was never run. One test does now.

### A new emulator test raced on its own shared tile

It had all eight threads of a block write `b[0]`, and `go test -race` reported it
in about one run in ten. **The emulator was right**: it serialises atomics on one
lock precisely so that a plain write racing another is still reported, because on
the device that is a race too. A test whose subject is the barrier must not
smuggle one in to prove it.

## The oracle had its own bugs

Worth its own heading, because it is the reason the differential fuzzer's
failure message says a mismatch is **a finding about one of the two backends and
not a verdict about which**.

The ratio is what to take from it. One burst of searching, roughly half an hour,
produced seven defects — five in the emitter and two in the oracle — and an
oracle this young goes on having its own. The four below were collected across
the whole fuzzing effort rather than in that half hour, and two of them are in
the **harness** rather than in a comparison, which is a third thing again: a
generator that produces an input neither backend was asked about, and a runner
that never comes back.

- **`tolerance.Agree` settled a NaN for a `float32` and let a `float64` fall
  through to `reflect.DeepEqual`**, which compares them with `==`, so two NaNs in
  a `[]float64` were reported as disagreeing — in a message that printed them
  identically: "b[2] = NaN, want NaN". The rule is asked once for both widths
  now, and `internal/tolerance`, which three callers depend on and which had no
  tests at all, has them.
- **`Program.Compare` fell through to `reflect.DeepEqual` for a struct**, the
  same bug one level up, so two buffers holding the same NaN were reported as
  differing. It compares field by field through `tolerance.Agree` now.
- **The generator clamped every float to a _signed_ range** before converting it
  to an integer. Clamping is right in principle, since such a conversion is
  implementation-dependent in Go and undefined in C — but a negative float
  reached a `uint32` and the two backends disagreed on every element.
  [`../NUMERICS.md`](../NUMERICS.md) gained the conversion rule, because it is a
  trap for a kernel author and not only for a generator.
- **`hostrun.Run` did not bound the child.** The non-terminating `range` kernel
  above took a whole fuzz worker with it — the driver sat at 100% of a core for
  four and a half minutes with no diagnosis and no failing case recorded. It has
  a timeout now, and a `TimeoutError` distinct from a compile failure and from a
  non-zero exit, because "does not finish" may be either the generator's fault or
  the emitter's and the caller has to be able to say which.

## Signed zero, host against emulator — and it was neither backend's fault

The last open fuzzer finding, and the answer turned out not to be a defect at
all, which is why it is filed here rather than above: it cost the same
diagnosis and it is the same kind of thing to know.

`fuzz.Generate(620)` with inputs `77` wrote `+0` on the host where the emulator
wrote `-0`, and the differential reported it because `internal/tolerance`
deliberately treats the two zeros as different results rather than a rounding —
a relative bound cannot tell them apart, their difference being zero while their
bits are not.

Neither side was wrong. `fmin` and `fmax` of two zeros of opposite signs are
**unspecified**, in C and in IEEE 754 alike — `minNum` "returns either one" —
and the host does not even answer it stably: the same expression gives −0
compiled at `-O1` and +0 at `-O0`, the compiler having picked a different
instruction. `hostrun` builds at `-O1`.

The device's answer is _not_ open, so that is the one both sides were pinned to.
`gpu.Fmin` returns −0 and `gpu.Fmax` returns +0, which `NUMERICS.md` records and
`TestFminFmaxParity` asserts against real hardware, and
[`internal/fuzz/hostrun`](../internal/fuzz/hostrun)'s shim now pins `fmin`,
`fmax`, `fminf` and `fmaxf` on the two zeros while leaving everything else — the
NaN rule included — to the real functions. The alternative, excluding `fmin` and
`fmax` from the differential the way the transcendentals are excluded, was
measured before being rejected: it would have cut the host oracle's reach from
34.8% of generated programs to 17.9%, for a case the shim settles in twenty
lines.

`TestFminFmaxAgreeOnTheZeros` compares the **bits**, because `+0 == -0` is true
in both languages and an `==` would have passed whatever the shim did, and it
pins the direction too, so a shim that made the two agree on the wrong answer
would still fail. That test, and not the corpus entry, is what holds this down:
`signed-zero-620` is committed, but an entry is a seed rather than a program, so
what seed 620 renders moves whenever the generator does. See
[`verification.md`](verification.md#what-a-corpus-entry-does-not-pin).

Fixed in `53ab673`.
