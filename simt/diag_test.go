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
	a[i] = float32(j)
	var m map[int]int
	_ = m
}`)
	if len(diags) != 3 {
		t.Fatalf("got %d diagnostics, want 3:\n%s", len(diags), render(diags))
	}
	for _, want := range []string{"multiple assignment", "unsupported statement", "unsupported"} {
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
	src := diagPrelude + "func K(ctx gpu.Ctx, a []float64) { a[0] = 1 }\n"
	_, err := simt.Transpile(fstest.MapFS{"k.go": &fstest.MapFile{Data: []byte(src)}}, "K")
	if err == nil {
		t.Fatal("expected an error")
	}
	const want = "simt: k.go:5:21: unsupported type float64 on the device (kernels are float32/int32 only)"
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
