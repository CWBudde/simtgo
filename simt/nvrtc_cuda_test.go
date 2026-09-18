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
		// NVRTC can see with no headers included. min/max are the ones expr.go
		// maps directly, and a comment claiming CUDA provides them is not a
		// measurement.
		name: "float64 arithmetic, min/max on doubles, and 64-bit literals",
		body: "//gocuda:float64\n" +
			"func K(ctx gpu.Ctx, out, a, b []float64) {\n" +
			"\ti := ctx.GlobalID()\n" +
			"\tif i < len(out) {\n" +
			"\t\tlo := min(a[i], b[i])\n" +
			"\t\thi := max(a[i], b[i])\n" +
			"\t\tvar n int64 = 9007199254740993\n" +
			"\t\tout[i] = lo*2.5 + hi + float64(n%7)\n" +
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
		// state what Go believes and let NVRTC refuse it. A padded struct is
		// the interesting case -- Count at 4, four bytes of padding, Bias at 8,
		// size 16, align 8 -- because that is where the two could disagree.
		name: "a padded struct, by value and as a slice element",
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
		name: "an atomic on a shared tile, and one across a device function",
		body: "//gocuda:ignore\n" +
			"func bump(h []int32, i int) { gpu.AtomicAddI32(h, i, 1) }\n\n" +
			"func K(ctx gpu.Ctx, y []float32, h []int32) {\n" +
			"\ts := ctx.SharedF32(256)\n" +
			"\tgpu.AtomicAddF32(s, ctx.ThreadIdx(), 1)\n" +
			"\tctx.SyncThreads()\n" +
			"\tbump(h, ctx.GlobalID()%4)\n" +
			"\ty[0] = s[0]\n}",
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
	for _, name := range []string{"VecAdd", "Magnitude", "Scale", "FIR", "Classify", "Softclip", "Transpose", "Quantize", "BandGain"} {
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
