//go:build cuda

package simt_test

import (
	"errors"
	"strings"
	"testing"
	"testing/fstest"

	"github.com/CWBudde/gocuda"
	"github.com/CWBudde/gocuda/cuda"
	"github.com/CWBudde/gocuda/simt"
)

// TestGeneratedCCompiles puts the emitter's sharp edges through NVRTC.
//
// The golden tests pin what is emitted and the parity tests pin what it
// computes, but neither asks the question in between: is the generated C
// something a compiler accepts? Four constructs turned out to be valid Go
// that lowered to C++ NVRTC rejects -- a jump entering a variable's scope, a
// name declared in two case clauses of one switch, and a device function
// whose result was never declared. Every one of them is an error message
// about code the author never wrote, which is the failure mode the SIMT
// track exists to remove.
//
// No device is needed, only the toolkit: NVRTC compiles to PTX, and nothing
// here launches anything. When the toolkit is missing there is no question to
// answer, so the test skips rather than fails -- an absent libnvrtc says
// nothing about the emitter, and reporting it as a failure would train readers
// to ignore the one signal this test exists to give.
func TestGeneratedCCompiles(t *testing.T) {
	// The cheapest probe that loads libnvrtc and nothing else. *LibraryError
	// reports ErrNoCUDA as well as ErrLibraryNotFound, so this single
	// comparison covers both "no toolkit installed" and "the library is
	// somewhere the search does not look".
	if _, _, err := cuda.NVRTCVersion(); errors.Is(err, cuda.ErrNoCUDA) {
		t.Skipf("no CUDA toolkit available: %v", err)
	}

	const arch = "compute_75"
	cases := []struct{ name, body string }{{
		// Whether the built-ins the emitter reaches have double overloads
		// NVRTC can see with no headers included. fmin/fmax are what a double
		// min or max lowers to now -- the builtins are refused on floats,
		// because Go's propagate a NaN and CUDA's ignore one -- and a comment
		// claiming CUDA provides the double overloads is not a measurement.
		name: "float64 arithmetic, fmin/fmax on doubles, and 64-bit literals",
		body: "//gocuda:float64\n" +
			"func K(ctx gpu.Ctx, out, a, b []float64) {\n" +
			"\ti := ctx.GlobalID()\n" +
			"\tif i < len(out) {\n" +
			"\t\tlo := gpu.Fmin64(a[i], b[i])\n" +
			"\t\thi := gpu.Fmax64(a[i], b[i])\n" +
			"\t\tvar n int64 = 9007199254740993\n" +
			"\t\tout[i] = lo*2.5 + hi + float64(n%7)\n" +
			"\t}\n}",
	}, {
		// The integer overloads, which are untouched: no integer is a NaN, so
		// there is nothing for the two to disagree about, and these are still
		// what expr.go maps min and max directly onto.
		name: "min and max on integers",
		body: "func K(ctx gpu.Ctx, out []int64, a []int32, b int32) {\n" +
			"\ti := ctx.GlobalID()\n" +
			"\tif i < len(out) {\n" +
			"\t\tout[i] = int64(min(a[i], b)) + int64(max(a[i], b))\n" +
			"\t}\n}",
	}, {
		// NVRTC accepted this happily while the counter was an int, which is
		// why it needed a golden assertion rather than only a compile check --
		// the wrong answer was in the arithmetic, not in the syntax.
		name: "ranging over a 64-bit bound",
		body: "func K(ctx gpu.Ctx, out []int64, n int64) {\n" +
			"\tfor i := range n {\n\t\tout[0] = i * i\n\t}\n}",
	}, {
		name: "the minimum int64 constant",
		body: "func K(ctx gpu.Ctx, out []int64) {\n" +
			"\tvar n int64 = -9223372036854775808\n\tout[0] = n\n}",
	}, {
		name: "unsigned and 64-bit integer arithmetic",
		body: "func K(ctx gpu.Ctx, out []int64, x []int32, seed uint32) {\n" +
			"\ti := ctx.GlobalID()\n" +
			"\tif i < len(out) {\n" +
			"\t\th := seed ^ uint32(x[i])\n" +
			"\t\th = h*2654435761 + 1\n" +
			"\t\tv := int64(x[i])\n" +
			"\t\tout[i] = v*v + int64(h%16)\n" +
			"\t}\n}",
	}, {
		// The struct emission is the C++ furthest from anything the author
		// wrote, and the static_asserts in it are the layout guarantee: they
		// state what Go believes and let NVRTC refuse it. Shape is Floor at 0,
		// Count at 4, Bias at 8 and 16 bytes in all -- a mixed-width layout
		// with, as it happens, no hole anywhere in it, so nothing here is
		// padded and the emitter must add nothing.
		name: "a mixed-width struct, by value and as a slice element",
		body: "type Shape struct {\n\tFloor float32\n\tCount int32\n\tBias  float64\n}\n\n" +
			"type Band struct{ Upper, Gain float32 }\n\n" +
			"//gocuda:float64\n" +
			"func K(ctx gpu.Ctx, y, x []float32, bands []Band, cfg Shape) {\n" +
			"\ti := ctx.GlobalID()\n" +
			"\tif i >= len(y) {\n\t\treturn\n\t}\n" +
			"\tbest := Band{1, cfg.Floor}\n" +
			"\tfor _, b := range bands {\n" +
			"\t\tif x[i] <= b.Upper {\n\t\t\tbest = b\n\t\t\tbreak\n\t\t}\n\t}\n" +
			"\ty[i] = float32(float64(x[i])*float64(best.Gain)+cfg.Bias) + float32(cfg.Count)\n}",
	}, {
		// A struct with a hole before a field and slack after the last one,
		// which is where the padding is emitted. What is being asked is only
		// whether NVRTC accepts what comes out; that the padding is measured
		// rather than ignored is TestEmittedPaddingIsMeasured below, and it
		// has to be asked with hand-written C because the emitter cannot
		// produce a wrong answer to ask it with.
		name: "a struct with a hole before a field and slack after the last",
		body: "type S struct {\n\tA int32\n\tB float64\n\tC uint8\n}\n\n" +
			"//gocuda:float64\n" +
			"func K(ctx gpu.Ctx, y []float32, ss []S, one S) {\n" +
			"\ts := S{1, 2, 3}\n" +
			"\ty[0] = float32(s.A) + float32(ss[0].B) + float32(int32(one.C))\n}",
	}, {
		// An array field, which is the shape that was refused until the
		// offsets could be pinned, together with a narrow one so that the
		// padding around both is what NVRTC is asked about.
		name: "a struct with an array field and a narrow field",
		body: "type Taps struct {\n\tN    uint8\n\tW    [3]float32\n\tGain float64\n}\n\n" +
			"//gocuda:float64\n" +
			"func K(ctx gpu.Ctx, y []float32, ts []Taps) {\n" +
			"\tsum := float32(0)\n" +
			"\tfor _, w := range ts[0].W {\n\t\tsum += w\n\t}\n" +
			"\ty[0] = sum * float32(ts[0].Gain) * float32(int32(ts[0].N))\n}",
	}, {
		// The narrow types as storage: all four widths as slice elements, an
		// array of them as a local, arithmetic done in int32 and the result
		// converted back. Whether "signed char" and "unsigned short" are even
		// spellings NVRTC accepts as pointer element types with no header
		// included is a measurement, not a claim.
		name: "narrow integer storage with the arithmetic done in int32",
		body: "func K(ctx gpu.Ctx, out []uint8, a []int8, b []uint16, c []int16) {\n" +
			"\ti := ctx.GlobalID()\n" +
			"\tif i >= len(out) {\n\t\treturn\n\t}\n" +
			"\tvar buf [4]uint8\n" +
			"\tbuf[0] = out[i]\n" +
			"\tv := int32(a[i]) + int32(b[i]) + int32(c[i]) + int32(buf[0])\n" +
			"\tout[i] = uint8(v & 255)\n}",
	}, {
		name: "a struct literal with keyed fields, some of them left out",
		body: "type P struct{ X, Y, Z float32 }\n\n" +
			"func K(ctx gpu.Ctx, y []float32) {\n" +
			"\tp := P{Y: 2}\n" +
			"\ty[0] = p.X + p.Y + p.Z\n}",
	}, {
		name: "a fixed-size array declared, indexed and ranged over",
		body: "func K(ctx gpu.Ctx, y []float32) {\n" +
			"\tvar taps [4]float32\n" +
			"\ttaps[1] = 2\n" +
			"\tsum := float32(0)\n" +
			"\tfor _, v := range taps {\n\t\tsum += v\n\t}\n" +
			"\ty[0] = sum\n}",
	}, {
		name: "a labelled continue past a later declaration",
		body: "func K(ctx gpu.Ctx, y []float32, n int32) {\n" +
			"outer:\n" +
			"\tfor i := 0; i < int(n); i++ {\n" +
			"\t\tif i == 1 {\n\t\t\tcontinue outer\n\t\t}\n" +
			"\t\tv := float32(1)\n\t\ty[i] = v\n" +
			"\t}\n}",
	}, {
		name: "a labelled break and continue with declarations around both",
		body: "func K(ctx gpu.Ctx, y []float32, n int32) {\n" +
			"outer:\n" +
			"\tfor i := 0; i < int(n); i++ {\n" +
			"\t\tfor j := 0; j < int(n); j++ {\n" +
			"\t\t\tif i == j {\n\t\t\t\tbreak outer\n\t\t\t}\n" +
			"\t\t\tif i < j {\n\t\t\t\tcontinue outer\n\t\t\t}\n" +
			"\t\t\tw := float32(j)\n\t\t\ty[i] = w\n" +
			"\t\t}\n" +
			"\t\tz := float32(i)\n\t\ty[0] = z\n" +
			"\t}\n" +
			"\tq := float32(1)\n\ty[1] = q\n}",
	}, {
		name: "the same name declared in two case clauses",
		body: "func K(ctx gpu.Ctx, y []float32, a int32) { switch a {\n" +
			"case 1:\n\tv := float32(1)\n\ty[0] = v\n" +
			"case 2:\n\tv := float32(2)\n\ty[0] = v\n} }",
	}, {
		name: "a declaration in a clause of a chained switch",
		body: "func K(ctx gpu.Ctx, y []float32, a, b int32) { switch a {\n" +
			"case b:\n\tv := float32(1)\n\ty[0] = v\n" +
			"default:\n\tv := float32(2)\n\ty[0] = v\n} }",
	}, {
		name: "a fallthrough between clauses that both declare",
		body: "func K(ctx gpu.Ctx, y []float32, a int32) { switch a {\n" +
			"case 1:\n\tv := float32(1)\n\ty[0] = v\n\tfallthrough\n" +
			"case 2:\n\tv := float32(2)\n\ty[1] = v\n} }",
	}, {
		name: "a device function called from another device function",
		body: "func inner(x float32) float32 { return x + 1 }\n\n" +
			"func outer(x float32) float32 { return inner(x) * 2 }\n\n" +
			"func K(ctx gpu.Ctx, y, x []float32) { y[0] = outer(x[0]) + inner(x[1]) }",
	}, {
		name: "a device function taking a slice, and gpu.Ctx",
		body: "//gocuda:device\nfunc where(ctx gpu.Ctx) int { return ctx.GlobalID() }\n\n" +
			"func total(xs []float32) float32 { s := float32(0)\n\tfor _, v := range xs { s += v }\n\treturn s }\n\n" +
			"func K(ctx gpu.Ctx, y, x []float32) { i := where(ctx); if i < len(y) { y[i] = total(x) } }",
	}, {
		// A parallel assignment is the one construct that emits declarations
		// the author did not write, in the middle of a scope they did. Two of
		// them in one block must not declare the same temporary twice, and the
		// index lifted out of `i, y[i] = ...` has to be declared before it is
		// used -- both of which are questions for the compiler rather than for
		// a golden file.
		name: "parallel assignment, twice in one scope and with a lifted index",
		body: "type P struct{ X, Y float32 }\n\n" +
			"func K(ctx gpu.Ctx, y []float32, ps []P) {\n" +
			"\ta, b := y[0], y[1]\n" +
			"\ta, b = b, a\n" +
			"\ta, b = b, a\n" +
			"\ti := 0\n" +
			"\ti, y[i] = 1, a+b\n" +
			"\tps[0].X, ps[0].Y = ps[0].Y, ps[0].X\n" +
			"\tj := len(y) - 1\n" +
			"\tfor i < j {\n\t\ty[i], y[j] = y[j], y[i]\n\t\ti, j = i+1, j-1\n\t}\n}",
	}, {
		// The pointer qualifiers, which every signature now carries. const is
		// only sound if nothing writes through the parameter, and a __device__
		// function taking `const T* __restrict__` has to be callable with the
		// kernel's own pointer -- both of which are questions for the compiler
		// and for nobody else.
		name: "const and __restrict__ across a device function and an atomic",
		body: "func total(xs []float32) float32 { s := float32(0)\n\tfor _, v := range xs { s += v }\n\treturn s }\n\n" +
			"func fill(ys []float32, v float32) { for i := range ys { ys[i] = v } }\n\n" +
			"func K(ctx gpu.Ctx, y []float32, x []float32, h []int32) {\n" +
			"\tfill(y, total(x))\n" +
			"\tgpu.AtomicAddI32(h, 0, 1)\n}",
	}, {
		name: "every axis of the grid",
		body: "func K(ctx gpu.Ctx, y []float32) {\n" +
			"\ti := ctx.GlobalIDX() + ctx.GlobalIDY() + ctx.GlobalIDZ()\n" +
			"\tj := ctx.ThreadIdxY() + ctx.ThreadIdxZ() + ctx.BlockIdxY() + ctx.BlockIdxZ()\n" +
			"\tk := ctx.BlockDimY() + ctx.BlockDimZ() + ctx.GridDimY() + ctx.GridDimZ()\n" +
			"\tif i+j+k < len(y) { y[0] = 1 }\n}",
	}, {
		// The double-precision vocabulary added in Phase 2. The question is
		// the same one the case above asks and it is worth asking separately:
		// these are the unsuffixed names, and an unsuffixed name that NVRTC
		// resolved to the float overload instead would compile silently and
		// halve the precision the kernel asked for.
		// The float32 half of the same question, and it was the half with a
		// hole in it. fabsf and hypotf reach NVRTC through the committed
		// kernels and sqrtf through a golden, but sinf, cosf, expf, logf,
		// fminf and fmaxf sat in the name table in internal/lower with no
		// compiler between them and a claim.
		name: "the float32 gpu helpers with no headers included",
		body: "func K(ctx gpu.Ctx, out, a, b []float32) {\n" +
			"\ti := ctx.GlobalID()\n" +
			"\tif i < len(out) {\n" +
			"\t\tout[i] = gpu.Fmax(gpu.Hypot(gpu.Sqrt(gpu.Abs(a[i])), b[i]),\n" +
			"\t\t\tgpu.Fmin(gpu.Log(gpu.Exp(a[i])), gpu.Sin(a[i])+gpu.Cos(b[i])))\n" +
			"\t}\n}",
	}, {
		name: "the float64 gpu helpers with no headers included",
		body: "//gocuda:float64\n" +
			"func K(ctx gpu.Ctx, out, a, b []float64) {\n" +
			"\ti := ctx.GlobalID()\n" +
			"\tif i < len(out) {\n" +
			"\t\tout[i] = gpu.Fmax64(gpu.Hypot64(gpu.Sqrt64(gpu.Abs64(a[i])), b[i]),\n" +
			"\t\t\tgpu.Fmin64(gpu.Log64(gpu.Exp64(a[i])), gpu.Sin64(a[i])+gpu.Cos64(b[i])))\n" +
			"\t}\n}",
	}, {
		// The atomic vocabulary. NVRTC compiles a bare string with no
		// #include, so whether atomicAdd and friends are even declared is a
		// measurement rather than a claim -- and this is the only test that
		// can make it. Every overload the emitter can reach is here: the float
		// and int adds, min, max, exch and CAS.
		name: "the atomic vocabulary with no headers included",
		body: "func K(ctx gpu.Ctx, y []float32, h []int32) {\n" +
			"\ti := ctx.GlobalID()\n" +
			"\tgpu.AtomicAddF32(y, i%4, 1)\n" +
			"\told := gpu.AtomicAddI32(h, 0, 1)\n" +
			"\tgpu.AtomicMinI32(h, 1, old)\n" +
			"\tgpu.AtomicMaxI32(h, 2, old)\n" +
			"\tgpu.AtomicExchI32(h, 3, old)\n" +
			"\tif gpu.AtomicCASI32(h, 4, 0, old) == 0 {\n\t\ty[0] = 1\n\t}\n}",
	}, {
		// An atomic on a __shared__ tile takes the address of an array rather
		// than of a pointer's target, so the operand is a generic pointer and
		// not a global one. Whether NVRTC accepts that overload at all is the
		// question; the hardware resolving it back to a shared atomic is what
		// the parity test measures.
		// One tile per element type, in one kernel. The spellings are derived
		// rather than written down, so whether "unsigned long long s[8]" is
		// something CUDA has at all is a measurement and not a claim.
		name: "a shared tile of every element type",
		body: "//gocuda:float64\n" +
			"func K(ctx gpu.Ctx, y []float64) {\n" +
			"\ta := ctx.SharedF32(8)\n" +
			"\tb := ctx.SharedF64(8)\n" +
			"\tc := ctx.SharedI32(8)\n" +
			"\td := ctx.SharedI64(8)\n" +
			"\te := ctx.SharedU32(8)\n" +
			"\tf := ctx.SharedU64(8)\n" +
			"\ti := ctx.ThreadIdx() % 8\n" +
			"\ta[i] = 1\n\tb[i] = 1\n\tc[i] = 1\n\td[i] = 1\n\te[i] = 1\n\tf[i] = 1\n" +
			"\tctx.SyncThreads()\n" +
			"\ty[0] = float64(a[0]) + b[0] + float64(c[0]) + float64(d[0]) + float64(e[0]) + float64(f[0])\n}",
	}, {
		// atomicAdd on an int32 tile is the pair the Histogram kernel is
		// built out of, and the overload NVRTC has to resolve with no headers.
		name: "an atomic on an int32 shared tile",
		body: "func K(ctx gpu.Ctx, bins, x []int32) {\n" +
			"\ttile := ctx.SharedI32(256)\n" +
			"\tfor k := ctx.ThreadIdx(); k < 256; k += ctx.BlockDim() {\n\t\ttile[k] = 0\n\t}\n" +
			"\tctx.SyncThreads()\n" +
			"\tgpu.AtomicAddI32(tile, ctx.GlobalID()%256, 1)\n" +
			"\tctx.SyncThreads()\n" +
			"\tgpu.AtomicAddI32(bins, 0, tile[0])\n" +
			"\tx[0] = 1\n}",
	}, {
		// The dynamic block: `extern __shared__` inside the kernel, with the
		// length arriving as the generated trailing parameter.
		name: "a dynamically sized shared tile, beside a static one",
		body: "func K(ctx gpu.Ctx, y []float32) {\n" +
			"\tfixed := ctx.SharedF32(32)\n" +
			"\ts := ctx.SharedDynF32()\n" +
			"\tfor i := ctx.ThreadIdx(); i < len(s); i += ctx.BlockDim() {\n\t\ts[i] = 1\n\t}\n" +
			"\tfixed[ctx.ThreadIdx()%32] = 1\n" +
			"\tctx.SyncThreads()\n" +
			"\ty[0] = s[0] + fixed[0]\n}",
	}, {
		name: "a dynamically sized tile of a 64-bit element",
		body: "func K(ctx gpu.Ctx, y []int64) {\n" +
			"\ts := ctx.SharedDynI64()\n" +
			"\tif ctx.ThreadIdx() < len(s) {\n\t\ts[ctx.ThreadIdx()] = int64(ctx.GlobalID())\n\t}\n" +
			"\tctx.SyncThreads()\n" +
			"\ty[0] = s[0]\n}",
	}, {
		// __shared__ inside a __device__ function. It is block-scoped storage
		// that CUDA allocates once for the function, and the only thing that
		// settles whether NVRTC accepts it is NVRTC.
		name: "a shared tile inside a device function",
		body: "//gocuda:ignore\n" +
			"func stage(ctx gpu.Ctx, x []float32) float32 {\n" +
			"\ttile := ctx.SharedF32(256)\n" +
			"\ttile[ctx.ThreadIdx()%256] = x[0]\n" +
			"\tctx.SyncThreads()\n" +
			"\treturn tile[0]\n}\n\n" +
			"func K(ctx gpu.Ctx, y, x []float32) { y[0] = stage(ctx, x) + stage(ctx, x) }",
	}, {
		name: "an atomic on a shared tile, and one across a device function",
		body: "//gocuda:ignore\n" +
			"func bump(h []int32, i int) { gpu.AtomicAddI32(h, i, 1) }\n\n" +
			"func K(ctx gpu.Ctx, y []float32, h []int32) {\n" +
			"\ts := ctx.SharedF32(256)\n" +
			"\tgpu.AtomicAddF32(s, ctx.ThreadIdx(), 1)\n" +
			"\tctx.SyncThreads()\n" +
			"\tbump(h, ctx.GlobalID()%4)\n" +
			"\ty[0] = s[0]\n}",
	}, {
		// The warp vocabulary, and the case this whole test exists for: NVRTC
		// compiles a bare string with no #include, so whether the _sync
		// built-ins are declared at all was an open question until this
		// compiled. Every entry in the emitter's table is reached here, both
		// element types of every shuffle, because they are separate overloads.
		name: "the warp-level vocabulary with no headers included",
		body: "func K(ctx gpu.Ctx, y []float32, h []int32) {\n" +
			"\ti := ctx.GlobalID()\n" +
			"\tlane := ctx.LaneID()\n" +
			"\tv := y[i]\n" +
			"\tv += ctx.ShuffleF32(v, 0)\n" +
			"\tv += ctx.ShuffleXorF32(v, gpu.WarpSize/2)\n" +
			"\tv += ctx.ShuffleUpF32(v, 1)\n" +
			"\tv += ctx.ShuffleDownF32(v, 2)\n" +
			"\tn := h[i]\n" +
			"\tn += ctx.ShuffleI32(n, lane)\n" +
			"\tn += ctx.ShuffleXorI32(n, 8)\n" +
			"\tn += ctx.ShuffleUpI32(n, 4)\n" +
			"\tn += ctx.ShuffleDownI32(n, 4)\n" +
			"\tm := ctx.Ballot(v > 0) | ctx.ActiveMask()\n" +
			"\tif ctx.Any(n > 0) && ctx.All(lane < gpu.WarpSize) {\n\t\tm += 1\n\t}\n" +
			"\tctx.SyncWarp()\n" +
			"\th[i] = n + int32(m%2)\n" +
			"\ty[i] = v + float32(lane)\n}",
	}, {
		// A warp primitive reached through a device function, which is where
		// the Ctx vanishes from the C signature: the built-ins are globals, so
		// nothing has to be passed for them to work.
		name: "a warp primitive inside a device function",
		body: "//gocuda:ignore\n" +
			"func warpSum(ctx gpu.Ctx, v float32) float32 {\n" +
			"\tfor off := gpu.WarpSize / 2; off > 0; off /= 2 {\n" +
			"\t\tv += ctx.ShuffleDownF32(v, off)\n\t}\n" +
			"\treturn v\n}\n\n" +
			"func K(ctx gpu.Ctx, y, x []float32) {\n" +
			"\ti := ctx.GlobalID()\n" +
			"\ts := warpSum(ctx, x[i])\n" +
			"\tif ctx.LaneID() == 0 {\n\t\ty[i/gpu.WarpSize] = s\n\t}\n}",
	}}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			src := "package kernels\n\nimport \"github.com/CWBudde/gocuda/gpu\"\n\n" + tc.body + "\n"
			u, err := simt.Transpile(fstest.MapFS{"k.go": &fstest.MapFile{Data: []byte(src)}}, "K")
			if err != nil {
				t.Fatalf("Transpile: %v", err)
			}
			if _, err := cuda.Compile(u.Source, "K.cu", arch); err != nil {
				t.Fatalf("NVRTC refused the generated CUDA: %v\n%s", err, u.Source)
			}
		})
	}

	// The committed kernels go through the same gate, so a kernel that stops
	// compiling is caught here and not at some later launch.
	for _, name := range []string{"VecAdd", "Magnitude", "MagnitudeFast", "Scale", "FIR", "Classify", "Softclip", "Transpose", "Quantize", "BandGain", "Gray", "Histogram", "WarpReduceSum"} {
		t.Run(name, func(t *testing.T) {
			u, err := simt.Transpile(gocuda.Kernels(), name)
			if err != nil {
				t.Fatalf("Transpile: %v", err)
			}
			if _, err := cuda.Compile(u.Source, name+".cu", arch); err != nil {
				t.Fatalf("NVRTC refused %s: %v", name, err)
			}
			if strings.TrimSpace(u.Source) == "" {
				t.Fatalf("%s lowered to nothing", name)
			}
		})
	}
}

// TestEmittedPaddingIsMeasured is the negative half of the offset check, and
// it is written in C by hand because the emitter cannot produce a wrong answer
// to ask the question with.
//
// The padding exists to make sizeof say something it could not say on its own.
// C++ turns out to insert exactly the holes Go does -- the first case here is
// the struct with no padding members at all, asserting Go's size, and NVRTC
// accepts it -- so the old assertion was never failing on any real ABI. What it
// could not do was rule out a compiler that put a field somewhere else and made
// the size come out right anyway, because a hole it did not declare is a hole
// sizeof cannot see. With every hole declared, the members add up to Go's size
// by construction, and C++ places each at or after the end of the one before,
// so sizeof can only equal Go's size if nothing was inserted and every field
// therefore sits at Go's offset. That argument needs the padding to be part of
// what sizeof measures, which is what the second case asks: one byte too much
// and NVRTC refuses the struct.
//
// The honest limit is that the argument also assumes each field's size in C is
// its size in Go. For a scalar that is the type map, and for a nested struct it
// is that struct's own assertion, so it is assumed nowhere that it is not also
// checked -- but it is an assumption, not a consequence.
func TestEmittedPaddingIsMeasured(t *testing.T) {
	if _, _, err := cuda.NVRTCVersion(); errors.Is(err, cuda.ErrNoCUDA) {
		t.Skipf("no CUDA toolkit available: %v", err)
	}
	const arch = "compute_75"
	const entry = "\nextern \"C\" __global__ void K(float* y) { y[0] = 1; }\n"

	// Go's layout for struct{A int32; B float64}: A at 0, B at 8, 16 bytes.
	undeclared := "struct S { int A; double B; };\nstatic_assert(sizeof(S) == 16, \"size\");" + entry
	if _, err := cuda.Compile(undeclared, "S.cu", arch); err != nil {
		t.Errorf("NVRTC refused a struct whose hole it inserts itself, so the premise of this test is wrong: %v", err)
	}

	declared := "struct S { int A; unsigned char gocuda_pad0[4]; double B; };\nstatic_assert(sizeof(S) == 16, \"size\");" + entry
	if _, err := cuda.Compile(declared, "S.cu", arch); err != nil {
		t.Errorf("NVRTC refused the struct the emitter would write: %v", err)
	}

	wrong := "struct S { int A; unsigned char gocuda_pad0[5]; double B; };\nstatic_assert(sizeof(S) == 16, \"size\");" + entry
	if _, err := cuda.Compile(wrong, "S.cu", arch); err == nil {
		t.Error("NVRTC accepted a struct with one byte too much padding, so sizeof is not measuring the padding and the offsets are not pinned by it")
	}
}
