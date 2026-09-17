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
	for _, name := range []string{"VecAdd", "Magnitude", "Scale", "FIR"} {
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
