package simt_test

import (
	"errors"
	"strings"
	"testing"
	"testing/fstest"

	"github.com/CWBudde/gocuda/simt"
)

const diagPrelude = "package kernels\n\nimport \"github.com/CWBudde/gocuda/gpu\"\n\n"

// refuse lowers a kernel expected to fail and returns its diagnostics.
func refuse(t *testing.T, body string) []simt.Diagnostic {
	t.Helper()
	src := diagPrelude + body + "\n"
	_, err := simt.Transpile(fstest.MapFS{"k.go": &fstest.MapFile{Data: []byte(src)}}, "K")
	if err == nil {
		t.Fatalf("expected the kernel to be refused, but it lowered")
	}
	var ue *simt.UnsupportedError
	if !errors.As(err, &ue) {
		t.Fatalf("error %v is not an *UnsupportedError", err)
	}
	return ue.Diags
}

// TestSeveralDiagnostics is the point of collecting rather than latching: a
// kernel with three unrelated problems must report three, not the first.
func TestSeveralDiagnostics(t *testing.T) {
	diags := refuse(t, `func K(ctx gpu.Ctx, a []float32) {
	i, j := 0, 1
	go func() {}()
	var m map[int]int
	a[i] = float32(j) + float32(len(m))
}`)
	if len(diags) != 3 {
		t.Fatalf("got %d diagnostics, want 3:\n%s", len(diags), render(diags))
	}
	for _, want := range []string{"multiple assignment", "unsupported statement", "map[int]int"} {
		if !strings.Contains(render(diags), want) {
			t.Errorf("diagnostics do not mention %q:\n%s", want, render(diags))
		}
	}
}

// TestDiagnosticsResumeAfterAGoodStatement is the behaviour the per-statement
// window buys, and the one a two-problem test would not catch: a refusal must
// not silence the rest of the kernel, not even across a statement that lowers
// perfectly well.
func TestDiagnosticsResumeAfterAGoodStatement(t *testing.T) {
	diags := refuse(t, `func K(ctx gpu.Ctx, a []float32) {
	i, j := 0, 1
	a[i] = 1
	go func() {}()
	a[j] = 2
}`)
	if len(diags) != 2 {
		t.Fatalf("got %d diagnostics, want 2:\n%s", len(diags), render(diags))
	}
	if diags[0].Pos.Line >= diags[1].Pos.Line {
		t.Errorf("diagnostics are not in source order:\n%s", render(diags))
	}
}

// TestNoCascade guards the poison set. A shared buffer whose size is not
// constant is one mistake, and must read as one: without the tri-state in
// sharedSize the declaration would also be rejected for being a []float32, and
// without the poison set every later len() of it would complain again.
func TestNoCascade(t *testing.T) {
	diags := refuse(t, `func K(ctx gpu.Ctx, a []float32) {
	s := ctx.SharedF32(len(a))
	s[0] = float32(len(s))
}`)
	if len(diags) != 1 {
		t.Fatalf("got %d diagnostics, want 1:\n%s", len(diags), render(diags))
	}
	if !strings.Contains(diags[0].Msg, "constant size") {
		t.Errorf("unexpected diagnostic: %v", diags[0])
	}
}

// TestSignatureRefusalIsAlone keeps a kernel whose signature is wrong from
// reporting the consequences of that in its body: the parameters are the frame
// the body is read in, so there is nothing useful to say about it yet.
func TestSignatureRefusalIsAlone(t *testing.T) {
	diags := refuse(t, `func K(a []float32) {
	i, j := 0, 1
	a[i] = gpu.Sqrt(float32(j))
}`)
	if len(diags) != 1 {
		t.Fatalf("got %d diagnostics, want 1:\n%s", len(diags), render(diags))
	}
	if !strings.Contains(diags[0].Msg, "first parameter must be gpu.Ctx") {
		t.Errorf("unexpected diagnostic: %v", diags[0])
	}
}

// TestSingleDiagnosticTextIsUnchanged pins the rendering that every existing
// test and every log line already depends on.
func TestSingleDiagnosticTextIsUnchanged(t *testing.T) {
	// The subject is the rendering -- "simt: file:line:col: msg" -- not this
	// particular refusal, so the fixture only has to be something still
	// refused. It was []float64 until //gocuda:float64 made that a question
	// about a directive rather than about a type, and []int8 until narrow
	// integer storage made that one a question about position.
	src := diagPrelude + "func K(ctx gpu.Ctx, a []int) { a[0] = 1 }\n"
	_, err := simt.Transpile(fstest.MapFS{"k.go": &fstest.MapFile{Data: []byte(src)}}, "K")
	if err == nil {
		t.Fatal("expected an error")
	}
	const want = "simt: k.go:5:21: int cannot cross to the device: Go's int is 8 bytes and CUDA's int is 4, so the elements would not line up; use int32 or int64"
	if err.Error() != want {
		t.Errorf("got  %q\nwant %q", err.Error(), want)
	}
}

func render(diags []simt.Diagnostic) string {
	var b strings.Builder
	for _, d := range diags {
		b.WriteString("  " + d.Error() + "\n")
	}
	return b.String()
}

// TestPackageLevelIdentRefused covers a construct that used to be silently
// mistranslated: a package-level variable was emitted into the CUDA verbatim
// and only rejected by NVRTC, at run time, as "identifier is undefined" -- a C
// error about code nobody wrote.
func TestPackageLevelIdentRefused(t *testing.T) {
	src := diagPrelude + "var gain float32 = 2\n\nfunc K(ctx gpu.Ctx, a []float32) { a[0] = gain }\n"
	_, err := simt.Transpile(fstest.MapFS{"k.go": &fstest.MapFile{Data: []byte(src)}}, "K")
	if err == nil {
		t.Fatal("expected a package-level variable to be refused")
	}
	if !strings.Contains(err.Error(), "declared outside the kernel") {
		t.Errorf("unexpected error: %v", err)
	}
}

// TestPackageLevelConstStillWorks is the other half of that check: constants
// are folded by go/types before the emitter sees them, which is what lets
// kernels.FIR size its tile with FIRBlock + FIRMaxTaps.
func TestPackageLevelConstStillWorks(t *testing.T) {
	src := diagPrelude + "const gain = 2.0\n\nfunc K(ctx gpu.Ctx, a []float32) { a[0] = gain }\n"
	u, err := simt.Transpile(fstest.MapFS{"k.go": &fstest.MapFile{Data: []byte(src)}}, "K")
	if err != nil {
		t.Fatalf("Transpile: %v", err)
	}
	if !strings.Contains(u.Source, "a[0] = 2.0f") {
		t.Errorf("constant was not folded:\n%s", u.Source)
	}
}

// TestGenericKernelRefused reports the actual problem rather than complaining
// about the type parameter's interface underlying type.
func TestGenericKernelRefused(t *testing.T) {
	src := diagPrelude + "func K[T float32](ctx gpu.Ctx, a []T) { a[0] = 1 }\n"
	_, err := simt.Transpile(fstest.MapFS{"k.go": &fstest.MapFile{Data: []byte(src)}}, "K")
	if err == nil {
		t.Fatal("expected a generic kernel to be refused")
	}
	if !strings.Contains(err.Error(), "must not be generic") {
		t.Errorf("unexpected error: %v", err)
	}
}

// TestImportRefusedWithPosition covers the rule that used to be enforced by
// the importer alone. It now lives in the walk, because an analyzer is handed a
// package the real compiler built, where importing "math" is legal Go -- and
// reporting it directly gets the import's own position rather than burying it
// in the type checker's complaint about an unresolvable package.
func TestImportRefusedWithPosition(t *testing.T) {
	src := "package kernels\n\nimport (\n\t\"math\"\n\n\t\"github.com/CWBudde/gocuda/gpu\"\n)\n\n" +
		"func K(ctx gpu.Ctx, a []float32) { a[0] = float32(math.Pi) }\n"
	_, err := simt.Transpile(fstest.MapFS{"k.go": &fstest.MapFile{Data: []byte(src)}}, "K")
	if err == nil {
		t.Fatal("expected the import to be refused")
	}
	const want = "simt: k.go:4:2: kernels may not import math"
	if !strings.HasPrefix(err.Error(), want) {
		t.Errorf("got  %q\nwant prefix %q", err.Error(), want)
	}
}

// TestSeveralTypeErrors: go/types stops at the first error unless asked not to,
// so a kernel with three mistakes used to report one.
func TestSeveralTypeErrors(t *testing.T) {
	diags := refuse(t, `func K(ctx gpu.Ctx, a []float32) {
	a[0] = undefinedOne
	a[1] = undefinedTwo
	a[2] = undefinedThree
}`)
	if len(diags) != 3 {
		t.Fatalf("got %d diagnostics, want 3:\n%s", len(diags), render(diags))
	}
}

// TestTestFilesAreNotKernelSources: go:embed's kernels/*.go matches a _test.go
// too, and it would then join the type-check and fail on "testing" -- so a
// kernel package could not have tests at all.
func TestTestFilesAreNotKernelSources(t *testing.T) {
	fsys := fstest.MapFS{
		"k.go":      &fstest.MapFile{Data: []byte(diagPrelude + "func K(ctx gpu.Ctx, a []float32) { a[0] = 1 }\n")},
		"k_test.go": &fstest.MapFile{Data: []byte("package kernels\n\nimport \"testing\"\n\nfunc TestNothing(t *testing.T) {}\n")},
	}
	if _, err := simt.Transpile(fsys, "K"); err != nil {
		t.Fatalf("a _test.go file must not be treated as kernel source: %v", err)
	}
}

// TestUnknownKernelNamesTheOnesThatExist turns a typo from a bare "not found"
// into something actionable.
func TestUnknownKernelNamesTheOnesThatExist(t *testing.T) {
	src := diagPrelude + "func K(ctx gpu.Ctx, a []float32) { a[0] = 1 }\n" +
		"func Other(ctx gpu.Ctx, a []float32) { a[0] = 2 }\n"
	_, err := simt.Transpile(fstest.MapFS{"k.go": &fstest.MapFile{Data: []byte(src)}}, "Kay")
	if err == nil {
		t.Fatal("expected an error")
	}
	for _, want := range []string{`no kernel named "Kay"`, "K, Other"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q", err, want)
		}
	}
}
