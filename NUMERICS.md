# Numerics

What a `float32` means in this repository, on each of the two backends, and how
far apart a test is allowed to let them drift.

It exists because the SIMT track's whole claim is that one Go function runs on
the CPU emulator and on the device and computes the same thing. "The same
thing" is not a property of the source; it is a property of two very different
arithmetics, and a parity test is only as good as the bound it applies. Three
places in this repository compare a `float32` result, and they had drifted into
three different rules before anybody wrote one down.

## How to read a claim here

This file was written on a machine with NVRTC and no GPU: the compiler could be
asked questions and the hardware could not. A numerics document written under
that condition and full of unattributed assertions about hardware would be
exactly the drift the rest of the repository works to avoid, so every claim
below carries its provenance:

- **[measured]** — established from an artifact in this repository or from a
  command anyone can rerun here. The command or the file is named.
- **[cited]** — taken from somebody else's documentation, which is named. It is
  not a measurement of this code and is not treated as one.
- **[unverified]** — believed, wanted, or implied by the two above, and not
  established. These are the interesting ones.

## What the compiler is told

**[measured]** `cuda.Compile` (`cuda/driver_cuda.go`) builds its option list in
`nvrtcOptions` (`cuda/error.go`). `--gpu-architecture=<arch>` is always there
and is the only one a caller cannot influence; `--use_fast_math` is added when
the caller passes `cuda.WithFastMath()`, which `simt` does for a kernel
carrying `//gocuda:fastmath` and for no other reason. There is no third option
and no other place in this repository where a numerical flag is set.

Every numerical property below is therefore a default that nobody chose —
**except** in a fast-math kernel, which is the one place somebody did. That
distinction is the whole point of the directive being opt-in, and the section
[Fast math, when it is asked for](#fast-math-when-it-is-asked-for) is what it
buys and costs.

**[cited]** NVRTC's defaults for the four options that matter are
`--fmad=true`, `--ftz=false`, `--prec-div=true` and `--prec-sqrt=true`;
`--use_fast_math` turns that into `--ftz=true`, `--prec-div=false`,
`--prec-sqrt=false` with `--fmad` still true. — _NVRTC User Guide_ (12.8),
"Supported Compile Options".

**[measured]** All four are visible in the committed PTX, which is what makes
this a measurement rather than a reading of the manual. Recompiling
`kernels/prebuilt/FIR.cu`, `Quantize.cu` and `Magnitude.cu` with NVRTC 12.8 and
only `--gpu-architecture=compute_75` reproduces the committed
`.compute_75.ptx` **byte for byte**, and flipping one option at a time changes
exactly one thing each:

| option              | default PTX                               | with the option                      |
| ------------------- | ----------------------------------------- | ------------------------------------ |
| `--fmad=false`      | `fma.rn.f32` (FIR ×5, Quantize)           | the multiply and the add, separately |
| `--ftz=true`        | `fma.rn.f32`, `div.rn.f32`, `sqrt.rn.f32` | the same with `.ftz`                 |
| `--prec-div=false`  | `div.rn.f32`                              | `div.full.f32`                       |
| `--prec-sqrt=false` | `sqrt.rn.f32`                             | `sqrt.approx.f32`                    |

So the arithmetic a kernel that asked for nothing gets is: multiply-add
contracted, denormals kept, division and square root in their precise forms
rather than their approximate ones.

**[measured]** The committed artifacts agree. Across the twelve
`kernels/prebuilt/*.ptx` built without a directive: `fma.rn.f32` appears five
times in `FIR` and once each in `Classify`, `Magnitude` and `Quantize`,
`fma.rn.f64` once in `BandGain`; `div.rn.f32` appears in `Quantize` and
`Softclip` and `sqrt.rn.f32` in `Magnitude`; and `grep '\.ftz\.'` and
`grep approx` over those twelve return nothing.

`MagnitudeFast.compute_75.ptx` is the thirteenth and is deliberately the
exception — it carries `//gocuda:fastmath`, and every one of those greps finds
it. That is the point of it existing.

**[unverified]** All of this is **PTX, and PTX is not what runs**. The driver
JITs it to SASS for the device it finds, and nobody here has looked at that
layer or has a device to look at it with. Every statement above is a statement
about what NVRTC emitted.

## FMA contraction

**[measured]** The device contracts `a*b + c` written in the kernel source.
`FIR`'s tap loop is `v += h[k] * tile[...]` and the PTX is five `fma.rn.f32`,
none of which survives `--fmad=false`. `Quantize`'s is subtler and shows the
same thing at one remove: the source says
`(float)((int)(h % 1024u)) / 1024.0f - 0.5f`, and the PTX is one
`fma.rn.f32 %f3, %f2, 0f3A800000, 0fBF000000` — the division by a power of two
became a multiply by 2⁻¹⁰, which was then contracted with the subtraction.

**[measured]** Not every `fma` in the PTX comes from the source. `Magnitude`'s
survives `--fmad=false`, so it is inside `hypotf`'s own implementation and not
a contraction of anything the author wrote. Compiling a kernel that calls
`sqrtf`, `logf`, `expf`, `sinf` and `cosf` gives **30** `fma.rn.f32` for five
calls — those are the library's polynomials, and the flag does not reach them.

`Classify`'s `gain * v + bias * 0.001f` and `BandGain`'s
`sum / 4.0 * gain + Bias` are the same source shape as `FIR`'s and are
presumably contracted for the same reason; they were not probed individually.

**[measured]** The CPU emulator does **not** contract, on this machine.
`func fma(a, b, c float32) float32 { return a*b + c }` compiles under go1.26.0
to `MULSS; ADDSS` on `GOARCH=amd64` and to a single `FMADDS` on
`GOARCH=arm64` (`go build -gcflags=-S`), and a witness value confirms the
amd64 half numerically: with `a = b = 1 + 2⁻¹²` and `c = −1` the function
returns 2⁻¹¹, while the unrounded product would give 2⁻¹¹ + 2⁻²⁴.

**[cited]** That is a fact about the toolchain and not about the language. The
Go specification explicitly permits an implementation to fuse a floating-point
multiply and add and to round only once. — _The Go Programming Language
Specification_, "Floating-point operators".

So: **the bounds below were set on a host where the emulator rounds twice and
the device rounds once.** This machine and CI's `ubuntu-latest` are both
`amd64`; the environment README.md reports its measurements from does not say.
On an `arm64` host the same tests would be comparing two contracted
arithmetics, which is a different comparison with the same numbers on it, and
whether they hold there is **[unverified]** — CI runs no parity test at all.

## Denormals

**[measured]** Nothing is flushed on either side. No committed PTX contains a
`.ftz` instruction, and `--ftz=true` is what would put one there. On the Go
side, 2⁻¹²⁶ × 0.5 evaluates to `0x00400000` — a subnormal, not a zero — and the
smallest subnormal doubles to `0x00000002`, so the emulator does not run with
flush-to-zero either.

**[measured]** One exception, and it is the sort of thing this document exists
to surface: the default PTX for a kernel calling `expf` contains exactly one
flushing instruction, `ex2.approx.ftz.f32`, inside `expf`'s own
implementation. `--ftz` has nothing to do with it. A subnormal argument or
result of `expf` is therefore not covered by the sentence above; nothing else
measured here is affected.

**[unverified]** No subnormal has ever been through a **device** from this
repository, and that part is unchanged: the parity tests use normally
distributed inputs, so what the hardware does with one is still an open
question rather than an established property.

**[measured]** Subnormals have now been through the _generated C_, which is a
weaker statement and worth keeping separate from the one above.
`internal/fuzz/hostrun` compiles the emitter's output with a host C++ compiler
and runs it, and the fuzz generator's input distribution puts the smallest
subnormal, both zeros, both infinities and a NaN into every buffer it makes.
The host and `gpu.RunCPU` agree on all of them, bit for bit. That says the
translation preserves them on x86-64; it says nothing about sm_75, where the
question above is still open.

## Division, square root and the library functions

**[measured]** `div.rn.f32` and `sqrt.rn.f32` — the round-to-nearest forms, in
every committed kernel but the one that asked otherwise.

**[cited]** NVIDIA's accuracy figures for the single-precision library, from
the _CUDA C++ Programming Guide_, appendix "Mathematical Functions", table
_Single-Precision Mathematical Standard Library Functions with Maximum ULP
Error_: addition and multiplication are IEEE-compliant; `sqrtf` is correctly
rounded under `--prec-sqrt=true`; `logf` is within 1 ulp; `sinf`, `cosf` and
`expf` within 2; `hypotf` within 3.

**[unverified]** Those are NVIDIA's numbers for their library. No result from
this repository has been compared against a correctly rounded reference on a
device, so they are an expectation the parity tests are built around rather
than something the repository has confirmed.

## Fast math, when it is asked for

A kernel carrying `//gocuda:fastmath` is compiled with `--use_fast_math`.
Nothing else in the repository sets a numerical flag, and no kernel acquires
this one by being called from one that has it — the directive is refused on a
device function, because NVRTC takes the option for a compilation and not for a
function.

**[measured]** The flag reaches NVRTC, and the experiment is one variable.
Compiling the committed `kernels/prebuilt/MagnitudeFast.cu` twice at
`--gpu-architecture=compute_75`, once plain and once with `--use_fast_math`,
changes exactly the four things the option is documented to change and nothing
else:

| plain         | with `--use_fast_math` |
| ------------- | ---------------------- |
| `sqrt.rn.f32` | `sqrt.approx.ftz.f32`  |
| `div.rn.f32`  | `div.approx.ftz.f32`   |
| `fma.rn.f32`  | `fma.rn.ftz.f32`       |
| `mul.f32`     | `mul.ftz.f32`          |

The second column is byte-for-byte what `go generate` committed as
`MagnitudeFast.compute_75.ptx`, apart from the `.version` line — NVRTC 12.8
targets PTX ISA 8.7 and a `nvcc -ptx` reproduction defaults to 8.0. The
instruction stream is identical, which is what makes this a measurement of the
pipeline rather than of a hand-run compiler.

**[measured]** It changes the answer on the device, and by about one ulp.
`TestFastMathChangesTheAnswer` (`simt/parity_test.go`, `-tags cuda`) builds one
source twice, differing only in the directive, and compares what came back from
a T550 (`sm_75`, CUDA 12.8, driver 580.178.04): **6605 of 16384 outputs differ,
40.3% of them, and the worst relative gap is 1.19·10⁻⁷** — one ulp of a
`float32` at the top of its binade. `TestMagnitudeFastParity` measures the same
kernel against a `float64` reference and gets the same worst figure.

Both halves of that matter. That 40% differ is why the test is a real negative
control: a regression that dropped the option would show two identical outputs.
That the worst gap is one ulp is **this kernel, this input, this device** —
`sqrtf` and division are the two operations here, both of which
`--use_fast_math` demotes to instructions NVIDIA documents at roughly 2 ulp, so
a kernel leaning on `expf`, `sinf` or a long accumulation has no claim on this
number.

**[unverified]** What fast math is _for_ — that it is faster. Nothing here has
timed it. The repository has no kernel-execution benchmarks at all (that is
Phase 5), so `--use_fast_math` is currently a documented change in the
instruction stream and an opt-in to a looser bound, with the speed it is named
for taken on trust. A kernel should carry the directive because its author
measured something, not because this file says the flag exists.

**What a test may assert about a fast-math kernel.** The tolerance policy below
is unchanged; what changes is which number goes in it. `1e-6`, the band every
other parity test uses, is on the order of ten ulp and is fine here on the
measured evidence — but it is fine by a factor of ten on one device, against
one input, for the two cheapest operations the flag touches. `1e-5` is what
`simt/parity_test.go` actually asserts for `MagnitudeFast`, and it is headroom
over NVIDIA's documented 2 ulp rather than a number the measurement demanded.
A fast-math kernel may **not** be held to the exact-equality rule below, in any
of its three cases: the approximate instructions are not correctly rounded, so
"every value is exactly representable" stops being a property of the source and
becomes a property of the compiler's mood.

## What `float32` means in the emulator

**[measured]** Every `float32` math helper in `gpu/math.go` is
`float32(math.F(float64(x)))`. The emulator therefore does **not** compute what
the device computes: it computes the binary64 result and rounds it once.

For `Sqrt` that is harmless. **[cited]** Rounding a correctly rounded binary64
result again to binary32 gives the correctly rounded binary32 result, because
53 ≥ 2·24 + 2 — the classical double-rounding condition (Figueroa, _When is
double rounding innocuous?_, 1995). Together with `sqrtf`'s cited 0 ulp it
follows that the two sides should agree bit for bit; whether they do is
**[unverified]**, and `TestMathHelperParity` deliberately does not demand it.

For `Hypot`, `Sin`, `Cos`, `Exp` and `Log` it is a real difference. The
emulator is within half an ulp of the exact value and the device is within the
cited one to three, so the two disagree in the last bits by construction, and
the parity tolerances have been absorbing that silently since the first one was
written.

`Abs` is exact through binary64, so it costs nothing.

### `Fmin` and `Fmax`

**[measured]** `gpu.Fmin` and `gpu.Fmax` follow IEEE 754 `minNum`/`maxNum`: a
NaN operand is ignored and the other one is returned, both NaN gives a NaN,
`Fmin` of the two zeros is −0 and `Fmax` is +0. They are written out rather
than wrapping `math.Min`/`math.Max`, which propagate NaN — Go documents that
— and the reason is in `gpu/math.go` and in commit `3115f2f`. This document
records the rule; it does not reopen it.

**[cited]** The NaN half is what CUDA documents for `fminf` and `fmaxf`: a NaN
argument is treated as missing data and the numeric argument is chosen. —
_CUDA Math API_, single-precision functions.

**[unverified]** The signed-zero half is **not** documented by CUDA.
`TestFminFmaxParity` asserts it against the device anyway, because a test that
declined to ask would leave the emulator's rule resting on nothing. If it ever
fails, it is `gpu.Fmin` that should be revisited and this section that should
record what the device did.

`Fmin64` and `Fmax64` were wrapping `math.Min` and `math.Max` — the very bug
the float32 pair had been corrected for — and now take the NaN cases out first.
The rest is still `math.Min`'s, which unlike the float32 case exists and
documents the remaining special values; **[cited]** Go specifies
`Min(-0, ±0) = -0`, which is what `fmin` does too.

**[measured]** The **host oracle** had to be told this. `fmin` and `fmax` of
two zeros of opposite signs are unspecified in C and in IEEE 754 alike —
`minNum` "returns either one" — and the host does not answer it stably: the
same expression gives −0 compiled at `-O1` and +0 at `-O0`, the compiler having
picked a different instruction. The differential fuzzer duly reported the
difference as a mismatch (`fuzz.Generate(620)`, inputs `77`).

Since the device's answer is _not_ open, `internal/fuzz/hostrun`'s shim pins
`fmin`, `fmax`, `fminf` and `fmaxf` on the two zeros and leaves everything
else, the NaN rule included, to the real functions. That is what a shim is for:
it supplies the device's C where the host's is free to differ, so a
disagreement in a run means a mistranslation rather than a tie-break nobody
promised. `TestFminFmaxAgreeOnTheZeros` holds it there.

### Go's `min` and `max` are a different function

**[cited]** In Go they are not the same function as CUDA's: the specification
says a builtin `min` with a NaN operand returns NaN, and CUDA's `fminf` follows
IEEE `minNum`, which ignores a NaN operand and returns the number.

**[measured]** So a kernel written with `min(a, b)` agreed with itself on both
backends for ordinary numbers and disagreed the moment a NaN reached it. This
section used to end "nothing currently tests this"; the differential fuzzer
tested it, in six seconds, by putting NaN in the input distribution and
generating `min(0.0/0.0, x)`.

**The float builtins are now refused**, pointing at `gpu.Fmin`/`gpu.Fmax`,
which mean the device's answer on both backends — see `SPEC.md`. The _integer_
overloads still lower to CUDA's `min`/`max` and are untouched, no integer being
a NaN.

**[measured]** For the record, since it is what made the disagreement invisible:
NVRTC compiles `min(a, b)` and `fminf(a, b)` to the **same** PTX instruction. A
probe containing both emits one `min.f32` and stores its result twice, so the
generated code gave no sign that the two languages disagreed about it.

## Converting a float to an integer it does not fit

**[cited]** Neither language promises anything. Go's specification says that in
a non-constant conversion, "if the result type cannot represent the value the
conversion succeeds but the result value is implementation-dependent"; C leaves
the same conversion undefined. So a float outside the destination's range is
one of the few places where the emulator and the device may legitimately
disagree and neither is wrong.

**[measured]** The differential fuzzer walked into it, and the case is worth
recording because it is not the obvious one. Its generator already clamped
every float to ±1000 before converting, which is in range for a signed 32-bit
integer — and not for an _unsigned_ one, where a negative value fits nothing.
Converting `-4.0f` to a `uint32` gave one answer on the host and another in the
emulator, on every element of the buffer. The clamp now takes the
destination's signedness into account.

Nothing refuses this at lowering, and nothing can: the value is not known until
the kernel runs. Clamp before converting, and clamp to a floor the destination
can hold.

## The tolerance policy

There is one rule, it lives in `internal/tolerance`, and the three places that
compare a `float32` result all call it: `simt/parity_test.go`,
`tile/tile_test.go` and `simt/prebuilt_cuda_test.go`. It is a package rather
than a helper per file because those are three Go packages, two of them
external test packages, and a rule spelled three times had already become
three rules — two relative comparisons with different signatures, and one bare
absolute comparison following neither.

**The rule.** `got` agrees with `want` when

```text
|got − want|  ≤  tol × max(|want|, 1)
```

with the difference taken in `float64`, so that the subtraction itself cannot
round.

**What it does not bound.** All of this is worth stating because the number
looks tighter than it is:

- It is **not** in units in the last place. `tol` is a relative error against
  the expected value, and one ulp of a `float32` is between 6·10⁻⁸ and 1.2·10⁻⁷
  of it depending on where in its binade the value sits — so `1e-6` is on the
  order of ten ulp and `1e-5` on the order of a hundred.
- The floor of 1 makes it an **absolute** bound wherever `|want| < 1`, which is
  most of a signal-processing test. At `|want| = 10⁻⁶` a `tol` of `10⁻⁶`
  permits an answer of zero. That is deliberate — a relative bound near a zero
  crossing is a bound on nothing — but it is not what the number looks like.
- A NaN on either side always fails, including NaN against NaN. No tolerance
  relates a NaN to anything; a test that expects one asks `IsNaN`, as
  `TestFminFmaxParity` does.
- It says nothing about the distribution of the error. The comparison stops at
  the first element outside the bound, so "passed" means no element exceeded
  it, not that the error is small on average.
- It is a bound on the **test**, not a bound derived from the arithmetic.
  Nothing in this repository proves any kernel meets it.

**The values in use, and where they came from.**

| bound  | where                                                                                      | why                                                                                                      |
| ------ | ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| exact  | `Transpose`, `Quantize`, `Gray`, the atomics, the warp primitives, the layout round trips  | see the next section                                                                                     |
| `1e-6` | `VecAdd`, `Scale`, `Magnitude`, `Classify`, `Softclip`, `BandGain`, tile stepwise-vs-fused | one or two operations per element                                                                        |
| `1e-5` | `FIR` (33 taps), the tile pipeline (17 taps), the prebuilt round trip                      | an accumulation, where the device's contracted tap loop and the emulator's uncontracted one part company |
| `1e-5` | `TestMathHelperParity`                                                                     | chosen here; see below                                                                                   |

The `1e-6` and `1e-5` bounds predate this document and were chosen against a
real device, which nobody working on this file can reach. **No tolerance value
was changed by it.** Two comparisons were re-spelled and neither moved:
`TestTransposeParity` asked for a tolerance of zero and now asks for equality,
which asserts the same thing without reading as though there were something to
round; and `runFIR` in `simt/prebuilt_cuda_test.go` used a bare absolute
`1e-5`, which the rule reproduces exactly here because that signal is a 17-tap
moving average of sines and every expected value has magnitude at most 1, so
the scale is its floor throughout.

`TestMathHelperParity`'s `1e-5` is new and is deliberately loose. The cited ulp
bounds plus the reference's own half ulp put the worst case at 2.5 ulp, which
is at most 3·10⁻⁷ relative; the bound is more than thirty times that, because
nobody has run it and because what was measured here is PTX rather than what
the device executes. Tightening it is work for the first person with a device,
and it is the most useful thing that could be done to this file.

## When a test may demand exact equality

The reasoning already existed, in comments spread across `gpu/atomic.go`,
`gpu/gpu_test.go`, `simt/parity_test.go` and `kernels/gray.go`. As a rule:

> A test compares floating-point results **exactly** when every value on both
> sides is exactly representable in its type **and** the result does not depend
> on the order the threads arrived in. Integers and bools are compared exactly
> always. Everything else uses the tolerance rule above.

Both halves are load-bearing, and the existing sites show why:

- **Exactly representable.** `kernels/gray.go` scales the Rec. 601 luma weights
  by 256 and does the arithmetic in `int32` precisely so that its parity test
  can assert equality; a `float32` pipeline would need a tolerance, and a
  tolerance is where a one-bit disagreement about how the two languages
  truncated a byte would hide.
- **Order-independent.** `gpu/atomic.go` spells out that `float32` addition is
  not associative and that the order goroutines take the emulator's lock is not
  the order the device's threads arrive in. `TestAtomicAddF32Histogram`
  (`gpu/gpu_test.go`) and `TestAtomicSharedTileParity` (`simt/parity_test.go`)
  may compare exactly only because every addend is 1 and every partial sum
  stays far below 2²⁴, where `float32` addition of integers is exact and the
  order stops mattering. Both comments say so, and both stop being true the
  moment the addends are real data.
- **Neither applies, but nothing was computed.** `TestTransposeParity` moves
  values without arithmetic and `TestNarrowStorageRoundTrip` and the struct and
  array layout probes read values back unchanged. A rounding cannot have
  happened, so a tolerance would only be somewhere for a wrong answer to hide.
- **A function that returns one of its operands.** `fminf` and `fmaxf` compute
  nothing, so `TestFminFmaxParity` compares exactly — asking `IsNaN` and the
  sign bit for the two cases `==` cannot decide, which are both operands NaN
  and the pair of signed zeros.

`internal/tolerance` provides both halves, `AssertClose` and `AssertEqual`, so
that choosing between them is a visible decision at the call site rather than a
tolerance of zero.

## What is still unknown

In rough order of how much it would be worth learning:

1. **Everything below PTX.** The driver's JIT is where these instructions
   become the ones that run, and this repository has never looked at SASS.
2. **Whether `TestMathHelperParity` passes, and by how much.** It skips without
   a device. The margin it reports on real hardware is what would let the bound
   come down from `1e-5` to something derived rather than guessed.
3. **Whether the device's `fminf` prefers −0.** Asserted by
   `TestFminFmaxParity`, documented by nobody.
4. **Subnormals.** Not flushed, per the PTX; exercised through the generated
   C on a host compiler, never through a device.
5. **`arm64`.** The emulator contracts there and the tolerances were not set
   for it.
6. **`float64`.** `BandGain` is the only kernel that uses it, its
   `fma.rn.f64` is measured, and the same contraction asymmetry applies one
   width up. Nothing else here is specific to double precision.
