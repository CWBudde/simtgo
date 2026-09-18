package simt_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"testing/fstest"

	"github.com/CWBudde/gocuda"
	"github.com/CWBudde/gocuda/simt"
)

// TestGolden pins the generated CUDA for every example kernel. Run with
// GOCUDA_UPDATE=1 to refresh the golden files after an intentional change.
func TestGolden(t *testing.T) {
	for _, name := range []string{"VecAdd", "Magnitude", "Scale", "FIR", "Classify", "Softclip", "Transpose"} {
		t.Run(name, func(t *testing.T) {
			u, err := simt.Transpile(gocuda.Kernels(), name)
			if err != nil {
				t.Fatalf("Transpile: %v", err)
			}
			got := u.Source
			path := filepath.Join("testdata", name+".cu")
			if os.Getenv("GOCUDA_UPDATE") == "1" {
				if err := os.WriteFile(path, []byte(got), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			want, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("read golden (set GOCUDA_UPDATE=1 to create): %v", err)
			}
			if got != string(want) {
				t.Errorf("generated CUDA differs from %s:\n--- got ---\n%s", path, got)
			}
		})
	}
}

// transpile lowers a single kernel named K, so that a behavioural test is one
// readable source string rather than a fixture.
func transpile(t *testing.T, body string) *simt.Unit {
	t.Helper()
	src := "package kernels\n\nimport \"github.com/CWBudde/gocuda/gpu\"\n\n" + body + "\n"
	u, err := simt.Transpile(fstest.MapFS{"k.go": &fstest.MapFile{Data: []byte(src)}}, "K")
	if err != nil {
		t.Fatalf("Transpile: %v", err)
	}
	return u
}

// TestCPrecedence pins the parenthesisation of the generated expressions.
//
// The emitter renders a Go parse tree against C's precedence table, and the
// two languages disagree: Go reads `a & b == c` as `(a & b) == c`, C as
// `a & (b == c)`, and shifts against addition go the same way. Getting this
// wrong is silent -- the generated code compiles and computes something else
// -- so both disagreements are pinned here, together with the cases that must
// stay free of parentheses.
func TestCPrecedence(t *testing.T) {
	const decl = "func K(ctx gpu.Ctx, y []float32, a, b, c int32) "
	cases := []struct{ name, body, want string }{{
		name: "bitwise and binds looser than equality in C",
		body: decl + "{ if a&b == c { y[0] = 1 } }",
		want: "if ((a & b) == c)",
	}, {
		name: "shift binds looser than addition in C",
		body: decl + "{ y[0] = float32(a<<b + c) }",
		want: "(float)((a << b) + c)",
	}, {
		name: "a left-associative chain needs no grouping",
		body: decl + "{ y[0] = float32(a - b - c) }",
		want: "(float)(a - b - c)",
	}, {
		name: "the right operand of an equal level does",
		body: decl + "{ y[0] = float32(a - (b - c)) }",
		want: "(float)(a - (b - c))",
	}, {
		name: "a condition is not wrapped twice",
		body: "func K(ctx gpu.Ctx, y []float32) { i := ctx.GlobalID(); if i < len(y) { y[i] = 1 } }",
		want: "if (i < y_len)",
	}, {
		name: "and-not lowers to a masked complement",
		body: decl + "{ y[0] = float32(a&^b + c) }",
		want: "(float)((a & ~b) + c)",
	}}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := transpile(t, tc.body).Source
			if !strings.Contains(got, tc.want) {
				t.Errorf("generated CUDA does not contain %q:\n%s", tc.want, got)
			}
		})
	}
}

// TestShadowedLen pins how len() is resolved. The symbol table is keyed on the
// object go/types resolved, not on the identifier's text, so a declaration
// that shadows a slice parameter is a distinct symbol and the parameter's
// length stays reachable -- and correct -- outside the shadow.
func TestShadowedLen(t *testing.T) {
	u := transpile(t, "func K(ctx gpu.Ctx, y, x []float32) {\n"+
		"\ti := ctx.GlobalID()\n"+
		"\tif i < len(x) {\n"+
		"\t\tx := i + 1\n"+ // an int shadowing the slice parameter x
		"\t\ty[i] = float32(x)\n"+ // the local x, not the parameter
		"\t}\n"+
		"\ty[0] = float32(len(x))\n"+ // the parameter again, once the shadow is gone
		"}")
	for _, want := range []string{"if (i < x_len)", "int x = i + 1;", "y[0] = (float)(x_len);"} {
		if !strings.Contains(u.Source, want) {
			t.Errorf("generated CUDA does not contain %q:\n%s", want, u.Source)
		}
	}
}

// TestUnitRequirements checks that what a kernel demands of its launch is read
// out of its source, so that Build and Launch can enforce it without the call
// site having to repeat it.
func TestUnitRequirements(t *testing.T) {
	u, err := simt.Transpile(gocuda.Kernels(), "FIR")
	if err != nil {
		t.Fatalf("Transpile: %v", err)
	}
	// FIR stages FIRBlock samples plus a halo of FIRMaxTaps, one float32 each.
	// The numbers are spelled out rather than imported from package kernels,
	// which would tie these tests to that package building.
	if u.RequiredBlock != 256 {
		t.Errorf("RequiredBlock = %d, want %d", u.RequiredBlock, 256)
	}
	if want := 4 * (256 + 64); u.SharedBytes != want {
		t.Errorf("SharedBytes = %d, want %d", u.SharedBytes, want)
	}
	if strings.Contains(u.Source, "AssumeBlockDim") {
		t.Errorf("AssumeBlockDim leaked into the generated CUDA:\n%s", u.Source)
	}

	// A kernel that says nothing about its geometry must not acquire a
	// requirement out of thin air.
	plain := transpile(t, "func K(ctx gpu.Ctx, y []float32) { y[ctx.GlobalID()] = 1 }")
	if plain.RequiredBlock != 0 || plain.SharedBytes != 0 {
		t.Errorf("RequiredBlock = %d, SharedBytes = %d, want 0 and 0", plain.RequiredBlock, plain.SharedBytes)
	}
}

// TestSwitch pins both lowerings of a Go switch. A switch whose cases are all
// constant becomes a C switch, which needs an explicit break per clause
// because Go does not fall through; anything else becomes an if/else chain,
// because C's switch cannot test a non-constant case at all.
func TestSwitch(t *testing.T) {
	const decl = "func K(ctx gpu.Ctx, y []float32, a, b int32) "
	cases := []struct {
		name, body string
		want       []string
		absent     []string
	}{{
		name: "constant cases become a C switch",
		body: decl + "{ switch a { case 1: y[0] = 1\ncase 2, 3: y[0] = 2\ndefault: y[0] = 3 } }",
		want: []string{"switch (a)", "case 1:", "case 2:", "case 3:", "default:", "break;"},
	}, {
		name: "fallthrough drops the break that ends a clause",
		body: decl + "{ switch a { case 1: y[0] = 1\nfallthrough\ncase 2: y[0] = 2 } }",
		// One clause ends in fallthrough, the other in break: exactly one break.
		want:   []string{"switch (a)", "case 1:", "case 2:"},
		absent: []string{"fallthrough"},
	}, {
		name: "a tagless switch becomes an if/else chain",
		body: decl + "{ switch { case a < 1: y[0] = 1\ncase a < 2: y[0] = 2\ndefault: y[0] = 3 } }",
		want: []string{"if (a < 1)", "else", "if (a < 2)"},
		// C's switch cannot test a condition, so none may be emitted.
		absent: []string{"switch ("},
	}, {
		name: "a non-constant case becomes an if/else chain against the tag",
		body: decl + "{ switch a { case b: y[0] = 1\ndefault: y[0] = 2 } }",
		want: []string{"int switch_tag = a;", "if (switch_tag == b)", "else"},
		// A C switch case label must be a constant expression.
		absent: []string{"switch ("},
	}, {
		// Parenthesised only where C needs it: == binds tighter than ||, so
		// the comparisons stand as they are.
		name: "a case with several values tests each of them",
		body: decl + "{ switch a { case b, b + 1: y[0] = 1 } }",
		want: []string{"if (switch_tag == b || switch_tag == b + 1)"},
	}, {
		name: "an initialiser gets its own scope",
		body: decl + "{ switch c := a + b; c { case 1: y[0] = 1 } }",
		want: []string{"int c = a + b;", "switch (c)"},
	}}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := transpile(t, tc.body).Source
			for _, w := range tc.want {
				if !strings.Contains(got, w) {
					t.Errorf("generated CUDA does not contain %q:\n%s", w, got)
				}
			}
			for _, a := range tc.absent {
				if strings.Contains(got, a) {
					t.Errorf("generated CUDA contains %q, which it must not:\n%s", a, got)
				}
			}
		})
	}
}

// TestFallthroughEmitsOneBreak pins the count rather than the text: a clause
// that falls through must not be closed, and the one that does not must be.
func TestFallthroughEmitsOneBreak(t *testing.T) {
	u := transpile(t, "func K(ctx gpu.Ctx, y []float32, a int32) "+
		"{ switch a { case 1: y[0] = 1\nfallthrough\ncase 2: y[0] = 2 } }")
	if n := strings.Count(u.Source, "break;"); n != 1 {
		t.Errorf("generated CUDA has %d break statements, want 1:\n%s", n, u.Source)
	}
}

// TestLabelledBranch pins labelled break and continue.
//
// Before this was implemented the label was dropped and `break outer` emitted
// a bare `break;`, which leaves the inner loop -- a mistranslation, not a
// refusal. C has no labelled break, so both lower to a goto: the break target
// sits after the loop, the continue target as the last statement of its body,
// where falling off the end still runs the for-clause's post statement, which
// is what Go's labelled continue does.
func TestLabelledBranch(t *testing.T) {
	src := "func K(ctx gpu.Ctx, y []float32, n int32) {\n" +
		"outer:\n" +
		"\tfor i := 0; i < int(n); i++ {\n" +
		"\t\tfor j := 0; j < int(n); j++ {\n" +
		"\t\t\tif i == j {\n\t\t\t\tbreak outer\n\t\t\t}\n" +
		"\t\t\tif i < j {\n\t\t\t\tcontinue outer\n\t\t\t}\n" +
		"\t\t\ty[i] = 1\n" +
		"\t\t}\n" +
		"\t}\n" +
		"}"
	got := transpile(t, src).Source
	for _, want := range []string{
		"goto outer_break;",
		"goto outer_continue;",
		"outer_continue: ;",
		"outer_break: ;",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("generated CUDA does not contain %q:\n%s", want, got)
		}
	}
	// The continue target must be inside the labelled loop and the break
	// target after it, or the two jumps mean the wrong thing.
	if strings.Index(got, "outer_continue: ;") > strings.Index(got, "outer_break: ;") {
		t.Errorf("the continue target must precede the break target:\n%s", got)
	}
}

// TestUnusedLabelEmitsNothing keeps a label that nothing jumps to out of the
// generated source: an unused label is a warning in C, and the Go compiler
// would have refused the kernel anyway.
func TestUnusedLabelEmitsNothing(t *testing.T) {
	got := transpile(t, "func K(ctx gpu.Ctx, y []float32, n int32) {\n"+
		"outer:\n"+
		"\tfor i := 0; i < int(n); i++ {\n"+
		"\t\tif i == 0 {\n\t\t\tbreak outer\n\t\t}\n"+
		"\t}\n"+
		"}").Source
	if strings.Contains(got, "outer_continue") {
		t.Errorf("an unused continue target was emitted:\n%s", got)
	}
	if !strings.Contains(got, "outer_break: ;") {
		t.Errorf("the used break target is missing:\n%s", got)
	}
}

// TestRangeValue pins `for i, v := range x`. The value is a copy in Go, so it
// is a local in C too -- assigning to it must not write through to the slice.
func TestRangeValue(t *testing.T) {
	cases := []struct{ name, body, want string }{{
		name: "named index",
		body: "func K(ctx gpu.Ctx, y, x []float32) { for i, v := range x { y[i] = v } }",
		want: "float v = x[i];",
	}, {
		name: "blank index gets a synthesised one",
		body: "func K(ctx gpu.Ctx, y, x []float32) { s := float32(0)\nfor _, v := range x { s += v }\ny[0] = s }",
		want: "float v = x[v_i];",
	}, {
		name: "the synthesised index steps around a name already in use",
		body: "func K(ctx gpu.Ctx, y, x []float32, v_i int32) { s := float32(0)\nfor _, v := range x { s += v + float32(v_i) }\ny[0] = s }",
		want: "float v = x[v_i2];",
	}, {
		name: "ranging over a shared tile",
		body: "func K(ctx gpu.Ctx, y []float32) { t := ctx.SharedF32(4)\nctx.SyncThreads()\ns := float32(0)\nfor _, v := range t { s += v }\ny[0] = s }",
		want: "float v = t[v_i];",
	}}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := transpile(t, tc.body).Source
			if !strings.Contains(got, tc.want) {
				t.Errorf("generated CUDA does not contain %q:\n%s", tc.want, got)
			}
		})
	}
}

// TestDeviceFunctions pins the shape of a translation unit that contains more
// than the entry point: a prototype for every device function, then the
// definitions, then the __global__ kernel. Prototypes first is what makes the
// order of the Go declarations irrelevant.
func TestDeviceFunctions(t *testing.T) {
	cases := []struct {
		name, body string
		want       []string
	}{{
		name: "a scalar helper",
		body: "func scale(v, k float32) float32 { return v * k }\n\n" +
			"func K(ctx gpu.Ctx, y, x []float32) { i := ctx.GlobalID(); if i < len(y) { y[i] = scale(x[i], 2) } }",
		want: []string{
			"__device__ float scale(float v, float k);",
			"__device__ float scale(float v, float k)",
			"return v * k;",
			"y[i] = scale(x[i], 2.0f);",
		},
	}, {
		// A slice parameter lowers to a pointer plus a length here exactly as
		// it does on a kernel, so the call site has to pass both.
		name: "a slice parameter travels with its length",
		body: "func total(xs []float32) float32 { s := float32(0)\nfor _, v := range xs { s += v }\nreturn s }\n\n" +
			"func K(ctx gpu.Ctx, y, x []float32) { y[0] = total(x) }",
		want: []string{
			"__device__ float total(float* xs, int xs_len);",
			"y[0] = total(x, x_len);",
		},
	}, {
		name: "a helper with no result",
		body: "func fill(xs []float32, v float32) { for i := range xs { xs[i] = v } }\n\n" +
			"func K(ctx gpu.Ctx, y []float32) { fill(y, 1) }",
		want: []string{"__device__ void fill(float* xs, int xs_len, float v);", "fill(y, y_len, 1.0f);"},
	}, {
		name: "a helper reached only through another helper",
		body: "func inner(x float32) float32 { return x + 1 }\n\n" +
			"func outer(x float32) float32 { return inner(x) * 2 }\n\n" +
			"func K(ctx gpu.Ctx, y, x []float32) { y[0] = outer(x[0]) }",
		want: []string{
			"__device__ float inner(float x);",
			"__device__ float outer(float x);",
			"return inner(x) * 2.0f;",
		},
	}, {
		// gpu.Ctx has no device representation, so it is dropped from the
		// signature and from the call, the way a kernel's own is. The opt-out
		// is what stops the helper being taken for a kernel in its own right.
		name: "a helper taking gpu.Ctx, opted out of being a kernel",
		body: "//gocuda:ignore\nfunc where(ctx gpu.Ctx) int { return ctx.GlobalID() }\n\n" +
			"func K(ctx gpu.Ctx, y []float32) { i := where(ctx); if i < len(y) { y[i] = 1 } }",
		want: []string{"__device__ int where();", "int i = where();"},
	}}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := transpile(t, tc.body).Source
			for _, w := range tc.want {
				if !strings.Contains(got, w) {
					t.Errorf("generated CUDA does not contain %q:\n%s", w, got)
				}
			}
			// The entry point stays last, so every prototype precedes it.
			if i := strings.Index(got, "__global__"); i >= 0 && strings.Contains(got[i:], "__device__") {
				t.Errorf("a device function was emitted after the entry point:\n%s", got)
			}
			// A device function is discovered partway through another
			// function's body, so its own indent has to start again.
			for _, line := range strings.Split(got, "\n") {
				if strings.Contains(line, "__device__") && strings.HasPrefix(line, "\t") {
					t.Errorf("a device function was emitted indented:\n%s", got)
					break
				}
			}
		})
	}
}

// TestDeviceFunctionEmittedOnce keeps a helper reached by two paths from being
// defined twice, which C would refuse.
func TestDeviceFunctionEmittedOnce(t *testing.T) {
	u := transpile(t, "func inner(x float32) float32 { return x + 1 }\n\n"+
		"func outer(x float32) float32 { return inner(x) * 2 }\n\n"+
		"func K(ctx gpu.Ctx, y, x []float32) { y[0] = inner(x[0]) + outer(x[1]) }")
	if n := strings.Count(u.Source, "__device__ float inner(float x)\n"); n != 1 {
		t.Errorf("inner is defined %d times, want 1:\n%s", n, u.Source)
	}
}

// TestUncalledFunctionIsNotEmitted keeps the translation unit to what the
// kernel actually reaches.
func TestUncalledFunctionIsNotEmitted(t *testing.T) {
	u := transpile(t, "func unused(x float32) float32 { return x }\n\n"+
		"func K(ctx gpu.Ctx, y []float32) { y[0] = 1 }")
	if strings.Contains(u.Source, "unused") {
		t.Errorf("an uncalled function was emitted:\n%s", u.Source)
	}
}

// TestAxisBuiltins pins the per-axis accessors against the CUDA built-ins
// they stand for. Getting an axis wrong is silent: the kernel compiles and
// reads the wrong neighbour.
func TestAxisBuiltins(t *testing.T) {
	cases := []struct{ call, want string }{
		{"ThreadIdxY()", "(int)threadIdx.y"},
		{"ThreadIdxZ()", "(int)threadIdx.z"},
		{"BlockIdxY()", "(int)blockIdx.y"},
		{"BlockIdxZ()", "(int)blockIdx.z"},
		{"BlockDimY()", "(int)blockDim.y"},
		{"BlockDimZ()", "(int)blockDim.z"},
		{"GridDimY()", "(int)gridDim.y"},
		{"GridDimZ()", "(int)gridDim.z"},
		{"GlobalIDX()", "(int)(blockIdx.x * blockDim.x + threadIdx.x)"},
		{"GlobalIDY()", "(int)(blockIdx.y * blockDim.y + threadIdx.y)"},
		{"GlobalIDZ()", "(int)(blockIdx.z * blockDim.z + threadIdx.z)"},
	}
	for _, tc := range cases {
		t.Run(tc.call, func(t *testing.T) {
			got := transpile(t, "func K(ctx gpu.Ctx, y []float32) { y[0] = float32(ctx."+tc.call+") }").Source
			if !strings.Contains(got, tc.want) {
				t.Errorf("generated CUDA does not contain %q:\n%s", tc.want, got)
			}
		})
	}
}

// TestChainSwitchBindsTheTagOnce pins what Go guarantees: a switch evaluates
// its tag exactly once. The if/else chain a non-constant switch lowers to
// would otherwise re-evaluate the expression in every arm, and a tag calling
// a device function that writes through a slice would then do that write once
// per arm and could select a different branch than Go does.
func TestChainSwitchBindsTheTagOnce(t *testing.T) {
	u := transpile(t, "func bump(xs []float32) int32 { xs[0] = xs[0] + 1\n\treturn int32(xs[0]) }\n\n"+
		"func K(ctx gpu.Ctx, y, x []float32) { switch bump(x) {\n"+
		"case int32(len(y)):\n\ty[0] = 1\n"+
		"case int32(len(x)):\n\ty[0] = 2\n"+
		"default:\n\ty[0] = 3\n} }")
	entry := u.Source[strings.Index(u.Source, "__global__"):]
	if n := strings.Count(entry, "bump("); n != 1 {
		t.Errorf("the tag is evaluated %d times in the entry point, want 1:\n%s", n, u.Source)
	}
	if !strings.Contains(entry, "int switch_tag = bump(x, x_len);") {
		t.Errorf("the tag was not bound to a local:\n%s", u.Source)
	}
}

// TestGeneratedCScopes covers the two places the emitter has to open a scope
// it has no Go counterpart for. Both are invisible in Go and fatal in C++:
// a jump may not enter the scope of an initialised variable, and two case
// clauses may not declare the same name in one switch scope.
func TestGeneratedCScopes(t *testing.T) {
	t.Run("the continue target is not jumped to across a declaration", func(t *testing.T) {
		got := transpile(t, "func K(ctx gpu.Ctx, y []float32, n int32) {\n"+
			"outer:\n"+
			"\tfor i := 0; i < int(n); i++ {\n"+
			"\t\tif i == 1 {\n\t\t\tcontinue outer\n\t\t}\n"+
			"\t\tv := float32(1)\n\t\ty[i] = v\n"+
			"\t}\n}").Source
		// The declaration has to sit inside a scope the goto leaves, so the
		// target must follow that scope's closing brace.
		decl := strings.Index(got, "float v = 1.0f;")
		closing := strings.Index(got[decl:], "}")
		target := strings.Index(got, "outer_continue: ;")
		if decl < 0 || target < 0 || target < decl+closing {
			t.Errorf("the continue target does not follow the body's own scope:\n%s", got)
		}
	})

	t.Run("each case clause gets its own scope", func(t *testing.T) {
		got := transpile(t, "func K(ctx gpu.Ctx, y []float32, a int32) { switch a {\n"+
			"case 1:\n\tv := float32(1)\n\ty[0] = v\n"+
			"case 2:\n\tv := float32(2)\n\ty[0] = v\n} }").Source
		for _, want := range []string{"case 1:\n\t\t{", "case 2:\n\t\t{"} {
			if !strings.Contains(got, want) {
				t.Errorf("generated CUDA does not brace the clause after %q:\n%s", strings.TrimSuffix(want, "\n\t\t{"), got)
			}
		}
	})
}

// TestFixedSizeArrays covers the three things an array is allowed to be:
// declared, indexed, and walked. Its extent lands after the name, which is the
// whole reason declarations go through cdecl rather than a type string.
func TestFixedSizeArrays(t *testing.T) {
	u := transpile(t, "func K(ctx gpu.Ctx, y []float32) {\n"+
		"\tvar taps [4]float32\n"+
		"\ttaps[0] = 1\n"+
		"\tsum := float32(0)\n"+
		"\tfor _, v := range taps {\n\t\tsum += v\n\t}\n"+
		"\ty[0] = sum / float32(len(taps))\n}")
	for _, want := range []string{
		"float taps[4] = {};",
		"taps[0] = 1.0f;",
		"for (int v_i = 0; v_i < 4; v_i++)",
		"float v = taps[v_i];",
		// len() on an array is a Go constant, so float32(len(taps)) folds to a
		// float32 constant before the emitter is asked. There is no length
		// parameter to consult and no conversion left to render.
		"y[0] = sum / 4.0f;",
	} {
		if !strings.Contains(u.Source, want) {
			t.Errorf("missing %q in:\n%s", want, u.Source)
		}
	}
}

// TestZeroValueOfAnAggregate pins the one place `var` could not keep emitting
// `= 0`: C++ has no such initialiser for an array, and a bool deserved better
// than an int anyway.
func TestZeroValueOfAnAggregate(t *testing.T) {
	u := transpile(t, "func K(ctx gpu.Ctx, y []float32) {\n"+
		"\tvar a [2]float32\n\tvar n int32\n"+
		"\ta[0] = float32(n)\n\ty[0] = a[0]\n}")
	if !strings.Contains(u.Source, "float a[2] = {};") {
		t.Errorf("array not value-initialised:\n%s", u.Source)
	}
	if !strings.Contains(u.Source, "int n = 0;") {
		t.Errorf("scalar zero value changed, which would rewrite every golden:\n%s", u.Source)
	}
}

// TestWideScalars pins the widened type map, including the two kinds that
// lowered all along with nothing to say they did.
func TestWideScalars(t *testing.T) {
	u := transpile(t, "func K(ctx gpu.Ctx, a []int64, b []uint32, c []uint64, d []bool, n int64, m uint32) {\n"+
		"\ta[0] = n\n\tb[0] = m\n\tc[0] = 7\n\td[0] = n > 0\n}")
	for _, want := range []string{
		"long long* a", "unsigned int* b", "unsigned long long* c", "bool* d",
		"long long n", "unsigned int m",
		// The suffix is what keeps the literal's type the one Go gave it,
		// rather than the first C++ type it happens to fit in.
		"c[0] = 7ull;",
	} {
		if !strings.Contains(u.Source, want) {
			t.Errorf("missing %q in:\n%s", want, u.Source)
		}
	}
}

// TestFloat64Directive covers both spellings of the opt-in, and that a double
// constant keeps its precision rather than being rounded to a float.
func TestFloat64Directive(t *testing.T) {
	onFunc := transpile(t, "//gocuda:float64\nfunc K(ctx gpu.Ctx, y []float64) { y[0] = 0.1 }")
	if !strings.Contains(onFunc.Source, "double* y") {
		t.Errorf("directive on the function had no effect:\n%s", onFunc.Source)
	}
	if !strings.Contains(onFunc.Source, "y[0] = 0.1;") {
		t.Errorf("double constant not rendered at full width:\n%s", onFunc.Source)
	}

	// Directly above the package clause, with no blank line: that is what makes
	// it the file's doc comment rather than a detached comment near the top.
	src := "//gocuda:float64\npackage kernels\n\nimport \"github.com/CWBudde/gocuda/gpu\"\n\n" +
		"func K(ctx gpu.Ctx, y []float64) { y[0] = 1 }\n"
	u, err := simt.Transpile(fstest.MapFS{"k.go": &fstest.MapFile{Data: []byte(src)}}, "K")
	if err != nil {
		t.Fatalf("directive in the package comment was not honoured: %v", err)
	}
	if !strings.Contains(u.Source, "double* y") {
		t.Errorf("file-wide directive had no effect:\n%s", u.Source)
	}
}
