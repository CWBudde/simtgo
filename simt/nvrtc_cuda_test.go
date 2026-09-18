//go:build cuda

package simt_test

import (
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
// here launches anything.
func TestGeneratedCCompiles(t *testing.T) {
	const arch = "compute_75"
	cases := []struct{ name, body string }{{
		// The float64 vocabulary is arithmetic and nothing else: gpu.Sqrt and
		// friends are float32-only, so the interesting question is whether the
		// builtins the emitter does reach have double overloads NVRTC can see
		// with no headers included. min/max are the ones expr.go maps directly,
		// and a comment claiming CUDA provides them is not a measurement.
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
		body: "//gocuda:ignore\nfunc where(ctx gpu.Ctx) int { return ctx.GlobalID() }\n\n" +
			"func total(xs []float32) float32 { s := float32(0)\n\tfor _, v := range xs { s += v }\n\treturn s }\n\n" +
			"func K(ctx gpu.Ctx, y, x []float32) { i := where(ctx); if i < len(y) { y[i] = total(x) } }",
	}, {
		name: "every axis of the grid",
		body: "func K(ctx gpu.Ctx, y []float32) {\n" +
			"\ti := ctx.GlobalIDX() + ctx.GlobalIDY() + ctx.GlobalIDZ()\n" +
			"\tj := ctx.ThreadIdxY() + ctx.ThreadIdxZ() + ctx.BlockIdxY() + ctx.BlockIdxZ()\n" +
			"\tk := ctx.BlockDimY() + ctx.BlockDimZ() + ctx.GridDimY() + ctx.GridDimZ()\n" +
			"\tif i+j+k < len(y) { y[0] = 1 }\n}",
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
	for _, name := range []string{"VecAdd", "Magnitude", "Scale", "FIR", "Classify", "Softclip", "Transpose"} {
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
