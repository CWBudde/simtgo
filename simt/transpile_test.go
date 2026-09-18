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
	for _, name := range []string{"VecAdd", "Magnitude", "Scale", "FIR", "Classify", "Softclip", "Transpose", "Quantize", "BandGain"} {
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
			"__device__ float total(const float* __restrict__ xs, int xs_len);",
			"y[0] = total(x, x_len);",
		},
	}, {
		name: "a helper with no result",
		body: "func fill(xs []float32, v float32) { for i := range xs { xs[i] = v } }\n\n" +
			"func K(ctx gpu.Ctx, y []float32) { fill(y, 1) }",
		want: []string{"__device__ void fill(float* __restrict__ xs, int xs_len, float v);", "fill(y, y_len, 1.0f);"},
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
		// signature and from the call, the way a kernel's own is. The marker
		// is what stops the helper being taken for a kernel in its own right.
		name: "a helper taking gpu.Ctx, marked as a device function",
		body: "//gocuda:device\nfunc where(ctx gpu.Ctx) int { return ctx.GlobalID() }\n\n" +
			"func K(ctx gpu.Ctx, y []float32) { i := where(ctx); if i < len(y) { y[i] = 1 } }",
		want: []string{"__device__ int where();", "int i = where();"},
	}, {
		// The same helper with a slice beside the Ctx, which is the shape that
		// proves deviceParams and deviceArgs drop the Ctx in step: the
		// signature loses it and so does the call, and the slice that follows
		// still splits into a pointer and a length on both sides. An emitter
		// that dropped it on one side only would emit a call C rejects.
		name: "a marked helper whose Ctx is followed by a slice",
		body: "//gocuda:device\nfunc mine(ctx gpu.Ctx, xs []float32) float32 { return xs[ctx.GlobalID()%len(xs)] }\n\n" +
			"func K(ctx gpu.Ctx, y, x []float32) { y[0] = mine(ctx, x) }",
		want: []string{
			"__device__ float mine(const float* __restrict__ xs, int xs_len);",
			"y[0] = mine(x, x_len);",
		},
	}, {
		// //gocuda:ignore keeps meaning what it always meant: not a kernel.
		// A helper that carries it is still lowered when a kernel calls it.
		name: "//gocuda:ignore still opts a Ctx helper out of being a kernel",
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

// TestParallelAssignment pins the two-phase lowering that `a, b = c, d` needs.
//
// Go evaluates every value before it stores any of them, so a swap is a swap.
// C has no such statement, and the emitter writes the phases out: a temporary
// per value, then the assignments. Emitting the assignments directly would
// compile and turn `a, b = b, a` into two copies of b, which is the failure
// this whole package is arranged to rule out.
func TestParallelAssignment(t *testing.T) {
	cases := []struct {
		name, body string
		want       []string
		absent     []string
	}{{
		name: "a swap goes through temporaries",
		body: "func K(ctx gpu.Ctx, y []float32) { a, b := y[0], y[1]; a, b = b, a; y[0] = a + b }",
		want: []string{"float a_tmp2 = b;", "float b_tmp2 = a;", "a = a_tmp2;", "b = b_tmp2;"},
	}, {
		name: "a declaration declares both, after both values are evaluated",
		body: "func K(ctx gpu.Ctx, y []float32) { a, b := y[0], y[1]; y[0] = a - b }",
		want: []string{"float a_tmp = y[0];", "float b_tmp = y[1];", "float a = a_tmp;", "float b = b_tmp;"},
	}, {
		// A mixed `:=`, where a exists and b does not: a is assigned and b is
		// declared, and the value bound to b is the *old* a.
		name: "a mixed short declaration",
		body: "func K(ctx gpu.Ctx, y []float32) { a := float32(1); a, b := y[0], a; y[0] = a + b }",
		want: []string{"float b_tmp = a;", "a = a_tmp;", "float b = b_tmp;"},
	}, {
		// The idiom this exists for. Neither subscript is written by the
		// statement, so neither needs a temporary of its own and the generated
		// C still reads like the Go.
		name:   "swapping two elements needs no index temporaries",
		body:   "func K(ctx gpu.Ctx, y []float32, n int32) { i, j := 0, int(n); y[i], y[j] = y[j], y[i]; y[0] = float32(i + j) }",
		want:   []string{"float y_tmp = y[j];", "float y_tmp2 = y[i];", "y[i] = y_tmp;", "y[j] = y_tmp2;"},
		absent: []string{"_idx"},
	}, {
		// And the case that forces one: Go evaluates the subscript against the
		// old i, C would index with the new one. The emitted temporary is what
		// keeps the two the same statement.
		name: "an index that the statement itself writes is lifted out",
		body: "func K(ctx gpu.Ctx, y []float32) { i := 0; i, y[i] = 1, 2; y[0] = float32(i) }",
		want: []string{"int y_idx = i;", "i = i_tmp;", "y[y_idx] = y_tmp;"},
	}, {
		name: "two structs swap as values",
		body: "type P struct{ X, Y float32 }\n\n" +
			"func K(ctx gpu.Ctx, y []float32) { p, q := P{X: 1}, P{Y: 2}; p, q = q, p; y[0] = p.X + q.Y }",
		want: []string{"P p_tmp2 = q;", "P q_tmp2 = p;"},
	}, {
		// Two swaps in one C scope must not declare the same temporary twice,
		// which is why the generated names are never handed back.
		name: "a second parallel assignment gets its own names",
		body: "func K(ctx gpu.Ctx, y []float32) { a, b := y[0], y[1]; a, b = b, a; a, b = b, a; y[0] = a + b }",
		want: []string{"float a_tmp2 = b;", "float a_tmp3 = b;"},
	}, {
		name: "struct fields swap",
		body: "type P struct{ X, Y float32 }\n\n" +
			"func K(ctx gpu.Ctx, y []float32, ps []P) { ps[0].X, ps[0].Y = ps[0].Y, ps[0].X; y[0] = 1 }",
		want: []string{"float X_tmp = ps[0].Y;", "ps[0].X = X_tmp;", "ps[0].Y = Y_tmp;"},
	}, {
		// The loop step the for-clause refusal sends here, so that what the
		// refusal advises is known to work.
		name: "a two-variable loop step in the body",
		body: "func K(ctx gpu.Ctx, y []float32, n int32) {\n" +
			"\ti := 0\n\tj := int(n) - 1\n" +
			"\tfor i < j {\n\t\ty[i], y[j] = y[j], y[i]\n\t\ti, j = i+1, j-1\n\t}\n}",
		want: []string{"int i_tmp = i + 1;", "int j_tmp = j - 1;", "i = i_tmp;", "j = j_tmp;"},
	}}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := transpile(t, tc.body).Source
			for _, w := range tc.want {
				if !strings.Contains(got, w) {
					t.Errorf("generated CUDA does not contain %q:\n%s", w, got)
				}
			}
			for _, w := range tc.absent {
				if strings.Contains(got, w) {
					t.Errorf("generated CUDA contains %q, which it should not:\n%s", w, got)
				}
			}
		})
	}
}

// TestReadOnlyParameters pins the const half of the pointer qualifiers.
//
// A parameter is const only when nothing writes through it on any path the
// kernel reaches, which makes the analysis interprocedural: a slice is
// forwarded to a device function as a bare pointer, and nothing at the call
// site says what happens to it. Marking a written parameter const is a compile
// error at best, so every case here is really asking whether the analysis
// stayed pessimistic where it could not see.
func TestReadOnlyParameters(t *testing.T) {
	cases := []struct {
		name, body string
		want       []string
	}{{
		name: "what the kernel writes is not const, what it reads is",
		body: "func K(ctx gpu.Ctx, y, x []float32) { i := ctx.GlobalID(); if i < len(y) { y[i] = x[i] } }",
		want: []string{"float* __restrict__ y, int y_len, const float* __restrict__ x, int x_len"},
	}, {
		// The case the analysis exists for: nothing in K's own body writes y.
		name: "a write inside a device function reaches the caller's parameter",
		body: "func fill(xs []float32, v float32) { for i := range xs { xs[i] = v } }\n\n" +
			"func K(ctx gpu.Ctx, y []float32) { fill(y, 1) }",
		want: []string{
			"__device__ void fill(float* __restrict__ xs, int xs_len, float v);",
			"__global__ void K(float* __restrict__ y, int y_len)",
		},
	}, {
		name: "a helper that only reads leaves both parameters const",
		body: "func total(xs []float32) float32 { s := float32(0)\nfor _, v := range xs { s += v }\nreturn s }\n\n" +
			"func K(ctx gpu.Ctx, y, x []float32) { y[0] = total(x) }",
		want: []string{
			"__device__ float total(const float* __restrict__ xs, int xs_len);",
			"__global__ void K(float* __restrict__ y, int y_len, const float* __restrict__ x, int x_len)",
		},
	}, {
		name: "a write two calls deep still reaches the parameter",
		body: "func inner(xs []float32) { xs[0] = 1 }\n\n" +
			"func outer(xs []float32) { inner(xs) }\n\n" +
			"func K(ctx gpu.Ctx, y, x []float32) { outer(y); y[1] = x[0] }",
		want: []string{
			"__device__ void inner(float* __restrict__ xs, int xs_len);",
			"__device__ void outer(float* __restrict__ xs, int xs_len);",
		},
	}, {
		// An atomic is the one write that is not an assignment.
		name: "an atomic counts as a write",
		body: "func K(ctx gpu.Ctx, h []int32, x []int32) { gpu.AtomicAddI32(h, 0, x[0]) }",
		want: []string{"int* __restrict__ h, int h_len, const int* __restrict__ x, int x_len"},
	}, {
		name: "an atomic inside a device function counts too",
		body: "func bump(h []int32, i int) { gpu.AtomicAddI32(h, i, 1) }\n\n" +
			"func K(ctx gpu.Ctx, h []int32) { bump(h, ctx.GlobalID()) }",
		want: []string{"__global__ void K(int* __restrict__ h, int h_len)"},
	}, {
		name: "a write to a struct field of an element",
		body: "type P struct{ X, Y float32 }\n\n" +
			"func K(ctx gpu.Ctx, ps []P, qs []P) { ps[0].X = qs[0].Y }",
		want: []string{"P* __restrict__ ps, int ps_len, const P* __restrict__ qs, int qs_len"},
	}, {
		name: "an increment is a write",
		body: "func K(ctx gpu.Ctx, n []int32, x []int32) { n[0]++; n[1] = x[0] }",
		want: []string{"int* __restrict__ n, int n_len, const int* __restrict__ x, int x_len"},
	}, {
		// A shared tile is not a parameter, so writing one says nothing about
		// the parameter it was filled from.
		name: "staging into a shared tile leaves the source const",
		body: "func K(ctx gpu.Ctx, y, x []float32) {\n" +
			"\tctx.AssumeBlockDim(64)\n\ts := ctx.SharedF32(64)\n" +
			"\ts[ctx.ThreadIdx()] = x[ctx.GlobalID()]\n\tctx.SyncThreads()\n" +
			"\ty[ctx.GlobalID()] = s[0]\n}",
		want: []string{"float* __restrict__ y, int y_len, const float* __restrict__ x, int x_len"},
	}, {
		// Passing one buffer to two parameters is only a problem when the
		// callee writes through one of them; two readers may share memory,
		// because restrict is a promise about what is modified.
		name: "a buffer passed twice to a helper that only reads",
		body: "func dot(a, b []float32) float32 { s := float32(0)\nfor i := range a { s += a[i] * b[i] }\nreturn s }\n\n" +
			"func K(ctx gpu.Ctx, y, x []float32) { y[0] = dot(x, x) }",
		want: []string{"y[0] = dot(x, x_len, x, x_len);"},
	}}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := transpile(t, tc.body).Source
			for _, w := range tc.want {
				if !strings.Contains(got, w) {
					t.Errorf("generated CUDA does not contain %q:\n%s", w, got)
				}
			}
		})
	}
}

// TestUnitParams checks that what a launch needs to know about a parameter
// travels with the unit: which arguments are buffers, and which of those the
// kernel writes. The aliasing check at launch is exactly this list plus the
// pointers the caller bound.
func TestUnitParams(t *testing.T) {
	u := transpile(t, "func K(ctx gpu.Ctx, y, x []float32, k float32) { y[0] = x[0] * k }")
	want := []simt.Param{
		{Name: "y", Slice: true},
		{Name: "x", Slice: true, ReadOnly: true},
		{Name: "k"},
	}
	if len(u.Params) != len(want) {
		t.Fatalf("got %d parameters, want %d: %+v", len(u.Params), len(want), u.Params)
	}
	for i, p := range u.Params {
		if p != want[i] {
			t.Errorf("parameter %d is %+v, want %+v", i, p, want[i])
		}
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
		"long long* __restrict__ a", "unsigned int* __restrict__ b",
		"unsigned long long* __restrict__ c", "bool* __restrict__ d",
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
	if !strings.Contains(onFunc.Source, "double* __restrict__ y") {
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
	if !strings.Contains(u.Source, "double* __restrict__ y") {
		t.Errorf("file-wide directive had no effect:\n%s", u.Source)
	}
}

// TestStructLayoutIsAsserted is the layout guarantee in the only form NVRTC
// can carry it.
//
// The emitter never models the C ABI. It states what Go believes about the type
// and lets the C++ compiler check that against what it believes, so the promise
// holds on whatever architecture and CUDA version the kernel is built for
// rather than on the one it was written on. Offsets are missing from the
// assertions because NVRTC compiles a string with no include path and so has no
// offsetof; TestStructRoundTrip in the parity tests pins those instead.
func TestStructLayoutIsAsserted(t *testing.T) {
	u := transpile(t, "type Shape struct {\n\tFloor float32\n\tCount int32\n\tBias  float64\n}\n\n"+
		"//gocuda:float64\nfunc K(ctx gpu.Ctx, y []float32, cfg Shape) { y[0] = cfg.Floor }")
	for _, want := range []string{
		"struct Shape\n{\n\tfloat Floor;\n\tint Count;\n\tdouble Bias;\n};",
		// 4 + 4 of Count, four bytes of padding, then Bias at 8: the padded
		// case is the one where Go and C could part company.
		`static_assert(sizeof(Shape) == 16,`,
		`static_assert(alignof(Shape) == 8,`,
		"y[0] = cfg.Floor;",
	} {
		if !strings.Contains(u.Source, want) {
			t.Errorf("missing %q in:\n%s", want, u.Source)
		}
	}
	// The definition has to precede the entry point that names it.
	if strings.Index(u.Source, "struct Shape") > strings.Index(u.Source, "__global__") {
		t.Errorf("struct defined after the kernel that uses it:\n%s", u.Source)
	}
}

// TestStructLiteralIsPositional pins the spelling rather than the values.
//
// Designated initialisers are C++20 and NVRTC defaults to C++17, so a keyed Go
// literal has to come out positionally -- with the fields left off written as
// their zero value rather than left to C++ to fill in.
func TestStructLiteralIsPositional(t *testing.T) {
	u := transpile(t, "type P struct{ X, Y, Z float32 }\n\n"+
		"func K(ctx gpu.Ctx, y []float32) { p := P{Y: 2}; y[0] = p.X + p.Y + p.Z }")
	if !strings.Contains(u.Source, "P p = P{0, 2.0f, 0};") {
		t.Errorf("literal not rendered positionally:\n%s", u.Source)
	}
	if strings.Contains(u.Source, ".Y =") {
		t.Errorf("designated initialiser emitted, which NVRTC's dialect has no answer for:\n%s", u.Source)
	}
}

// TestStructDefinedOnce covers a type reached by two paths -- a slice element
// and a by-value parameter -- which must not be defined twice.
func TestStructDefinedOnce(t *testing.T) {
	u := transpile(t, "type P struct{ X, Y float32 }\n\n"+
		"func K(ctx gpu.Ctx, y []float32, ps []P, one P) { y[0] = ps[0].X + one.Y }")
	if n := strings.Count(u.Source, "struct P\n"); n != 1 {
		t.Errorf("struct P defined %d times:\n%s", n, u.Source)
	}
}

// TestRangeOverAWideIntegerKeepsItsType is a regression test for a silent
// mistranslation that arrived with int64.
//
// Go gives `i` in `for i := range n` the type of n, so an int64 bound makes the
// index an int64. Emitting `for (int i = 0; ...)` computed every expression
// using it in 32 bits instead: i*i at i = 50000 is 2500000000 in Go and
// -1794967296 in C, and a bound above MaxInt32 would never terminate, since the
// counter wraps before it reaches one. NVRTC compiled it without a murmur.
func TestRangeOverAWideIntegerKeepsItsType(t *testing.T) {
	u := transpile(t, "func K(ctx gpu.Ctx, out []int64, n int64) {\n"+
		"\tfor i := range n {\n\t\tout[0] = i * i\n\t}\n}")
	if !strings.Contains(u.Source, "for (long long i = 0; i < n; i++)") {
		t.Errorf("range counter did not keep the bound's type:\n%s", u.Source)
	}

	// A slice still counts in int, because len() is an int and widening it
	// would change every committed golden to say the same thing.
	v := transpile(t, "func K(ctx gpu.Ctx, y []float32) {\n"+
		"\tfor i := range y {\n\t\ty[i] = 1\n\t}\n}")
	if !strings.Contains(v.Source, "for (int i = 0; i < y_len; i++)") {
		t.Errorf("slice range counter changed:\n%s", v.Source)
	}
}

// TestMinInt64Literal covers the one integer constant C++ cannot spell
// directly: it tokenises the positive magnitude first and applies unary minus
// afterwards, and 9223372036854775808 fits no signed type. NVRTC and nvcc both
// accept the naive spelling, but the generated .cu is a committed artifact and
// other compilers are entitled to refuse it.
func TestMinInt64Literal(t *testing.T) {
	u := transpile(t, "func K(ctx gpu.Ctx, out []int64) { var n int64 = -9223372036854775808; out[0] = n }")
	if !strings.Contains(u.Source, "(-9223372036854775807ll - 1)") {
		t.Errorf("minimum int64 not rendered representably:\n%s", u.Source)
	}
}

// TestAtomics pins what the atomic vocabulary lowers to.
//
// The interesting part is the first two arguments, which become one C operand:
// the Go side takes a buffer and an index because the subset has no
// address-of, and "&s[i]" is the only "&" this emitter ever writes.
func TestAtomics(t *testing.T) {
	const decl = "func K(ctx gpu.Ctx, y []float32, h []int32) "
	cases := []struct{ name, body, want string }{{
		name: "an int add takes the address of the element",
		body: decl + "{ i := ctx.GlobalID(); gpu.AtomicAddI32(h, i, 1) }",
		want: "atomicAdd(&h[i], 1);",
	}, {
		name: "a float add renders its value as a float",
		body: decl + "{ i := ctx.GlobalID(); gpu.AtomicAddF32(y, i, 2) }",
		want: "atomicAdd(&y[i], 2.0f);",
	}, {
		name: "the old value is an ordinary result",
		body: decl + "{ old := gpu.AtomicAddI32(h, 0, 1); y[0] = float32(old) }",
		want: "int old = atomicAdd(&h[0], 1);",
	}, {
		name: "min",
		body: decl + "{ gpu.AtomicMinI32(h, 0, 3) }",
		want: "atomicMin(&h[0], 3);",
	}, {
		name: "max",
		body: decl + "{ gpu.AtomicMaxI32(h, 0, 3) }",
		want: "atomicMax(&h[0], 3);",
	}, {
		name: "exchange",
		body: decl + "{ gpu.AtomicExchI32(h, 0, 3) }",
		want: "atomicExch(&h[0], 3);",
	}, {
		name: "compare-and-swap carries both operands",
		body: decl + "{ gpu.AtomicCASI32(h, 0, 0, 7) }",
		want: "atomicCAS(&h[0], 0, 7);",
	}, {
		name: "the index may be an expression",
		body: decl + "{ i := ctx.GlobalID(); gpu.AtomicAddI32(h, i+1, 1) }",
		want: "atomicAdd(&h[i + 1], 1);",
	}, {
		// A shared tile is a __shared__ array rather than a pointer parameter,
		// so its address is generic; the hardware resolves that back to a
		// shared atomic. Nothing in the emitted C says which it is.
		name: "a shared tile is addressable too",
		body: "func K(ctx gpu.Ctx, y []float32) { s := ctx.SharedF32(256); gpu.AtomicAddF32(s, ctx.ThreadIdx(), 1); ctx.SyncThreads(); y[0] = s[0] }",
		want: "atomicAdd(&s[(int)threadIdx.x], 1.0f);",
	}, {
		// The buffer is rendered through the ordinary identifier path, so a
		// name that collides with a C++ keyword is escaped exactly once and in
		// one place. Spelling it a second time here is how the generated C
		// would come to name something that was never declared.
		name: "a buffer whose name is a C++ keyword keeps its escape",
		body: "func K(ctx gpu.Ctx, float []int32) { gpu.AtomicAddI32(float, 0, 1) }",
		want: "atomicAdd(&float_[0], 1);",
	}, {
		name: "the result composes into a larger expression",
		body: decl + "{ y[0] = float32(gpu.AtomicAddI32(h, 0, 1)) * 2 }",
		want: "(float)(atomicAdd(&h[0], 1)) * 2.0f",
	}, {
		// A slice parameter of a device function is a pointer like any other,
		// so an atomic works across the call boundary with nothing special.
		name: "inside a device function",
		body: "//gocuda:ignore\nfunc bump(h []int32, i int) { gpu.AtomicAddI32(h, i, 1) }\n\n" +
			"func K(ctx gpu.Ctx, h []int32) { bump(h, ctx.GlobalID()) }",
		want: "atomicAdd(&h[i], 1);",
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

// TestFloat64Math pins the double-precision vocabulary onto CUDA's unsuffixed
// built-ins. sqrt rather than sqrtf is the whole point: reaching the float32
// table by accident would halve the precision the kernel asked for, silently.
func TestFloat64Math(t *testing.T) {
	const decl = "//gocuda:float64\nfunc K(ctx gpu.Ctx, y []float64) "
	cases := []struct{ name, body, want string }{{
		name: "sqrt is the double built-in, not sqrtf",
		body: decl + "{ y[0] = gpu.Sqrt64(y[1]) }",
		want: "y[0] = sqrt(y[1]);",
	}, {
		name: "abs is fabs",
		body: decl + "{ y[0] = gpu.Abs64(y[1]) }",
		want: "y[0] = fabs(y[1]);",
	}, {
		name: "hypot takes two",
		body: decl + "{ y[0] = gpu.Hypot64(y[1], y[2]) }",
		want: "y[0] = hypot(y[1], y[2]);",
	}, {
		name: "fmin and fmax nest",
		body: decl + "{ y[0] = gpu.Fmax64(gpu.Fmin64(y[1], y[2]), y[3]) }",
		want: "fmax(fmin(y[1], y[2]), y[3]);",
	}, {
		name: "log and exp",
		body: decl + "{ y[0] = gpu.Log64(gpu.Exp64(y[1])) }",
		want: "log(exp(y[1]));",
	}, {
		// The float32 table is unchanged by any of this: the suffixed names
		// still reach the suffixed built-ins.
		name: "the float32 helpers are untouched",
		body: "func K(ctx gpu.Ctx, y []float32) { y[0] = gpu.Sqrt(y[1]) }",
		want: "y[0] = sqrtf(y[1]);",
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
