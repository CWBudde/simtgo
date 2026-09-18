# The toolchain, measured

What the driver, NVRTC and the PTX between them actually do — as opposed to
what their headers and documentation say. Everything on this page was measured
on the machine described at the end, and the entries are here because each one
was a surprise that would have been silent if it had gone the other way.

There is no cgo anywhere in this module. `libcuda` and `libnvrtc` are `dlopen`ed
at run time through [purego](https://github.com/ebitengine/purego), so the whole
module compiles with `CGO_ENABLED=0` on a machine that has never had a CUDA
toolkit installed. Why there is no `pkg-config` in that story is in
[`decisions.md`](decisions.md#pkg-config-is-deliberately-not-used).

## The driver does not export the names in `cuda.h`

**The one on this page to remember**, because it is silent when wrong.

The header `#define`s `cuMemAlloc` to `cuMemAlloc_v2`, and does the same for
`cuMemFree`, `cuMemcpyHtoD`, `cuMemcpyDtoH` and `cuDevicePrimaryCtxRelease`.
`libcuda` exports **both** names. The unsuffixed ones are the pre-CUDA-3.2 API
taking **32-bit sizes**.

So a `dlsym` port that trusts the header names:

- links successfully,
- passes every small test,
- and **truncates any allocation or copy above 4 GiB**.

The bindings name the `_v2` symbols explicitly. Anything added to the driver
should be checked against `nm -D` rather than against the header.

## Which file gets opened

`cuda/library.go` decides. It is deliberately untagged and free of any loading,
so `cuda/library_test.go` checks the search order on a machine with no CUDA —
the machine where a path bug actually bites.

The two libraries are searched **differently, because they come from different
places**, and conflating them is the mistake the policy exists to prevent.

| Library    | Ships with  | Search order                                                                                                 |
| ---------- | ----------- | ------------------------------------------------------------------------------------------------------------ |
| `libcuda`  | the driver  | `GOCUDA_LIBCUDA`, else `libcuda.so.1`, `libcuda.so` from the linker path                                     |
| `libnvrtc` | the toolkit | `GOCUDA_LIBNVRTC`, else `$CUDA_PATH`/`$CUDA_HOME`, then the linker path, then `/usr/local/cuda`, `/opt/cuda` |

`CUDA_PATH` and `CUDA_HOME` apply to **NVRTC only**. The driver ships with the
driver, not the toolkit, so a toolkit root says nothing about where it lives.
The toolkit's own `lib64/stubs/libcuda.so` is deliberately never a candidate: it
exists to satisfy a linker and fails every call.

`GOCUDA_LIBCUDA` and `GOCUDA_LIBNVRTC` each name a file outright and **replace**
the respective search rather than heading it. When nothing is found, the error
is a `*LibraryError` naming every candidate tried — not a link failure the
caller could never have recovered from.

The two libraries load **independently**, and that is what lets a kernel with
prebuilt PTX run with **no toolkit present at all**: with `GOCUDA_LIBNVRTC`
pointed at a file that does not exist, `examples/fir` still computes on the GPU.

That independence paid off somewhere it was not designed for. `libnvrtc` needs
neither a driver nor a device, so pointing `GOCUDA_LIBNVRTC` at a downloaded
toolkit made `TestGeneratedCCompiles` runnable for the first time on a machine
with no GPU — which is how most of the next section got measured.

## What NVRTC declares with no header included

NVRTC compiles a bare string. There is no include path, so anything the
generated code uses has to be declared by the compiler itself. Each of these was
measured **one spelling at a time**, not assumed:

| Declared with no header                                                   | Status    |
| ------------------------------------------------------------------------- | --------- |
| `atomicAdd`, `atomicMin`, `atomicMax`, `atomicExch`, `atomicCAS`          | declared  |
| `__shfl_sync`, `__shfl_up/down/xor_sync`                                  | declared  |
| `__ballot_sync`, `__any_sync`, `__all_sync`, `__activemask`, `__syncwarp` | declared  |
| `sqrtf`, `hypotf`, `fminf`, `fmaxf`, `expf` and the `float32` set         | declared  |
| unsuffixed `sqrt`, `hypot`, `fmax` (the `double` set)                     | declared  |
| `min` / `max` on `double`                                                 | resolve   |
| `extern __shared__`                                                       | accepted  |
| `static_assert`, `sizeof`, `alignof`                                      | available |

The whole committed kernel set compiles as well as lowers, so the vocabulary did
not have to shrink to fit.

## What NVRTC does not have

| Missing              | Consequence                                      |
| -------------------- | ------------------------------------------------ |
| `offsetof`           | struct field offsets cannot be asserted directly |
| `__builtin_offsetof` | same                                             |
| `#include <cstddef>` | no include path at all                           |
| C++20                | defaults to C++17, so no designated initialisers |

Both offset spellings were **re-measured against 12.9** rather than taken on
trust, so the claim is current. `nvcc -ptx` accepts `offsetof` and would have
given the wrong answer about what NVRTC can do — the two are not
interchangeable for questions like this.

This is why the emitter asserts `sizeof` and `alignof` and declares **every hole
Go leaves** as a `gocuda_padN` member, the trailing one included. With every
hole spelled out the members account for exactly Go's size, and C++ lays each
member at or after the end of the one before it, so `sizeof` can only match if
nothing further was inserted — and then every field sits where Go put it. The
trailing member is load-bearing rather than tidy: without it a byte inserted
earlier could hide in the end slack.

**Against expectation, this fixes no disagreement anybody can observe.** NVRTC
inserts exactly the holes Go does, so the assertions passed before the padding
existed, and a struct that would fail without it does not exist on a conforming
C++ ABI. The mechanism was shown to have teeth the other way round: one byte too
_much_ padding is rejected. The offsets themselves are pinned by reading every
field back through the device, which tests the `cuda.Upload` copy path rather
than the compiler's opinion of it.

C++17 is why struct literals lower **positionally** and step over the padding:
designated initialisers are C++20.

## What NVRTC accepts that it should not

**A second `extern __shared__` declaration, without a word.** Of a different
element type as readily as of the same one — and every one of them names the
same bytes.

CUDA has one dynamic `__shared__` block per launch, so two names silently
aliasing is exactly the mistranslation this emitter exists to refuse. "At most
one dynamic tile" is therefore enforced in `internal/lower`, with a position,
and the comment at the site says NVRTC is not the backstop because it will not
object.

## PTX versions and forward compatibility

PTX is forward compatible, so one `compute_75` artifact serves every newer
device; an older device falls back to NVRTC.

**Backward is where it bites.** NVRTC 12.9 emits `.version 8.8` images, which a
12.8 driver refuses. It is recoverable — `internal/jit` falls back to NVRTC on a
rejected image — but it would silently cost the ahead-of-time saving on exactly
the machine this project's measurements come from.

So the committed artifacts are all built by 12.8 and stay at `.version 8.7`,
even when the only NVRTC on the machine doing the work is newer. Worth knowing
before regenerating: check what `nvrtcVersion` reports, not what is on `PATH`.

## The compile options are one option

`cuda.Compile` passes exactly one NVRTC option, `--gpu-architecture`. **Every
numerical setting is therefore a default nobody chose**, which is a different
claim from "we chose the defaults" and is the reason
[`../NUMERICS.md`](../NUMERICS.md) exists.

Those defaults were established rather than assumed: recompiling `FIR.cu`,
`Quantize.cu` and `Magnitude.cu` with only that option reproduces the committed
PTX **byte for byte**, and flipping each flag in turn shows what it would
change.

| Flag                | What flipping it does   |
| ------------------- | ----------------------- |
| `--fmad=false`      | removes the `fma`       |
| `--ftz=true`        | adds `.ftz`             |
| `--prec-div=false`  | gives `div.full.f32`    |
| `--prec-sqrt=false` | gives `sqrt.approx.f32` |

Two findings that the reasoning did not anticipate, both recorded in
[`../NUMERICS.md`](../NUMERICS.md) in full:

- **`--ftz=false` is not the whole story.** A kernel calling `expf` emits an
  `ex2.approx.ftz.f32` inside `expf`'s own implementation, which the flag does
  not reach.
- **`Magnitude`'s `fma` survives `--fmad=false`**, so it is inside `hypotf` and
  is not evidence of source contraction. FIR and Quantize are.

## Launch parameters go through `runtime.Pinner`

Kernel parameters are not marshalled through C `malloc`/`free` per launch: the
driver is handed pointers into Go memory held still by a `runtime.Pinner`, which
is what that type is for.

Recorded here because the measurement is the useful part and it is a **wash
rather than a win** — two C allocations become two Go ones, and the FIR example
measures the same as before (1.1 ms kernel, 9.4 ms transfers). That is the
point: the launch path was never where its 10 ms went. The transfers are, which
is what Phase 4 in [`../PLAN.md`](../PLAN.md) is aimed at.

## Where these numbers come from

NVIDIA T550 Laptop (`sm_75`, 4 GB), CUDA 12.8 NVRTC, driver 580, go1.26.8,
Linux, 11-core CPU. The NVRTC-only measurements were made with a 12.9 toolkit on
a machine with no device, where noted.

One machine is one machine — see
[`verification.md`](verification.md#what-is-not-verified).
