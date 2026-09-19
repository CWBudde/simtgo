//go:build cuda

package simt_test

import (
	"strings"
	"testing"
)

// The logs below are nvcc 12.0's own output, kept verbatim rather than
// idealised, because the shape of it is what the filter has to survive.

// TestRemarkDoesNotSwallowALaterFinding pins the defect that made the filter
// lose the one diagnostic it exists to catch.
//
// nvcc prints its "the warnings can be suppressed" remark after the *first*
// warning, not at the end. The filter used to ask whether the residue began
// with that remark and read yes as "the log was entirely noise", so a log
// whose first warning was noise and whose second was #549-D filtered down to
// a residue starting with the remark and threw the finding away with it. That
// is the diagnostic docs/emitter-defects.md credits with catching both halves
// of the shadowed-initialiser defect.
func TestRemarkDoesNotSwallowALaterFinding(t *testing.T) {
	log := "k.cu(5): warning #186-D: pointless comparison of unsigned integer with zero\n" +
		"\n" +
		"Remark: The warnings can be suppressed with \"-diag-suppress <warning-number>\"\n" +
		"\n" +
		"k.cu(8): warning #549-D: variable \"a\" is used before its value is set\n"

	got := remainingWarnings(log)
	if got == "" {
		t.Fatal("the #549-D finding was swallowed: a noise warning ahead of it left the log's " +
			"residue starting with NVRTC's remark")
	}
	if !strings.Contains(got, "#549-D") {
		t.Errorf("residue does not name the finding:\n%s", got)
	}
	if strings.Contains(got, "#186-D") {
		t.Errorf("residue kept the allowlisted noise:\n%s", got)
	}
}

// TestAllNoiseStillFiltersToNothing is the other direction: the remark is
// boilerplate, so a log carrying nothing but noise and the remark is empty.
func TestAllNoiseStillFiltersToNothing(t *testing.T) {
	log := "k.cu(5): warning #186-D: pointless comparison of unsigned integer with zero\n" +
		"      if (n >= 0u)\n" +
		"          ^\n" +
		"\n" +
		"Remark: The warnings can be suppressed with \"-diag-suppress <warning-number>\"\n"

	if got := remainingWarnings(log); got != "" {
		t.Errorf("a log of nothing but noise survived the filter:\n%s", got)
	}
}

// TestSplitWarningsKeepsTheCaretWithItsDiagnostic pins the reason blocks exist
// at all: NVRTC follows a diagnostic with the quoted source and a caret, and a
// filter that dropped only the numbered line would report the caret.
func TestSplitWarningsKeepsTheCaretWithItsDiagnostic(t *testing.T) {
	log := "k.cu(6): warning #177-D: variable \"t3\" was declared but never referenced\n" +
		"      int t3 = 0;\n" +
		"          ^\n" +
		"k.cu(9): warning #549-D: variable \"a\" is used before its value is set\n"

	blocks := splitWarnings(log)
	var noise *warningBlock
	for i := range blocks {
		if blocks[i].num == "#177-D" {
			noise = &blocks[i]
		}
	}
	if noise == nil {
		t.Fatalf("no #177-D block in %d blocks", len(blocks))
	}
	if len(noise.lines) != 3 {
		t.Errorf("block took %d lines, want the diagnostic plus its source and caret:\n%q", len(noise.lines), noise.lines)
	}
	if got := noise.ident(); got != "t3" {
		t.Errorf("quoted identifier = %q, want %q", got, "t3")
	}
}

// droppedUseLog is what NVRTC says when the emitter declares a variable the Go
// source uses and never emits the use. It is indistinguishable, as text, from
// the generator's own unused filler -- same number, same sentence.
const droppedUseLog = "k.cu(4): warning #177-D: variable \"scale\" was declared but never referenced\n" +
	"      float scale = 2.0f;\n" +
	"            ^\n"

// TestAllowlistCannotTellFillerFromADroppedUse records the blind spot rather
// than fixing it, because it cannot be fixed at this layer: generatorNoise is
// keyed on the diagnostic number, and the number is the same for both. This
// test is the reason triagedWarnings exists, and it should keep passing --
// the day it fails, the allowlist has grown a distinction it did not have.
func TestAllowlistCannotTellFillerFromADroppedUse(t *testing.T) {
	filler := "k.cu(6): warning #177-D: variable \"t3\" was declared but never referenced\n" +
		"      int t3 = 0;\n" +
		"          ^\n"

	for name, log := range map[string]string{"generator filler": filler, "a dropped use": droppedUseLog} {
		if got := remainingWarnings(log); got != "" {
			t.Errorf("%s: expected the allowlist to suppress it, got:\n%s", name, got)
		}
		if len(suppressedWarnings(log)) != 1 {
			t.Errorf("%s: expected exactly one suppressed block", name)
		}
	}
}

// The three tests below pin what the review of this change found: the slicing
// that decides what a judgment is shown was wrong in two ways that are silent
// rather than loud, because both produce a plausible-looking window.

// TestMentionsMatchesWholeIdentifiers pins the match against the generator's
// own name pool. Half of `namePool` in internal/fuzz is a single letter, so a
// substring test holds on nearly every line of any program: the window would
// become the whole source, and the filtering the measurement rests on would
// buy nothing while appearing to work.
func TestMentionsMatchesWholeIdentifiers(t *testing.T) {
	src := "auto class_ = 1;\nfloat scale = 2.0f;\nint a = 5;\nint y_len = 8;"

	got := mentions(src, "a", 0)
	if !strings.Contains(got, "int a = 5;") {
		t.Errorf("the declaration of a was not found:\n%s", got)
	}
	for _, unwanted := range []string{"auto", "scale", "y_len"} {
		if strings.Contains(got, unwanted) {
			t.Errorf("matching %q dragged in the line containing %q:\n%s", "a", unwanted, got)
		}
	}
	// "_" is a word character, so neither of these reaches into the other.
	if got := mentions(src, "class", 0); got != "" {
		t.Errorf("class matched inside class_:\n%s", got)
	}
	if got := mentions(src, "len", 0); got != "" {
		t.Errorf("len matched inside y_len:\n%s", got)
	}
}

// TestSliceFindsTheGoNameBehindAnEscapedCName pins the translation back.
//
// A Go name that is a C++ keyword is emitted with a trailing underscore, and
// the generator picks one from keywordPool one time in five, so the name
// NVRTC quotes is `class_` where the Go says `class`. Slicing the Go by the C
// spelling finds nothing, and a judgment shown no Go at all has every reason
// to call faithful filler a mistranslation -- which would fail a nightly fuzz
// run over nothing at all.
func TestSliceFindsTheGoNameBehindAnEscapedCName(t *testing.T) {
	goSrc := "func K(ctx gpu.Ctx, y []float32) {\n\tclass := 3\n\tclass = class\n\ty[0] = 1\n}"
	cSrc := "extern \"C\" __global__ void K(float* __restrict__ y, int y_len)\n{\n\tint class_ = 3;\n\tclass_ = class_;\n\ty[0] = 1.0f;\n}"
	block := splitWarnings("k.cu(3): warning #550-D: variable \"class_\" was set but never used\n")[0]

	goPart, cPart, err := slice(block, goSrc, cSrc)
	if err != nil {
		t.Fatalf("slice: %v", err)
	}
	if !strings.Contains(goPart, "class := 3") {
		t.Errorf("the Go slice does not carry the construct the warning is about:\n%s", goPart)
	}
	if !strings.Contains(cPart, "int class_ = 3;") {
		t.Errorf("the C slice does not carry the declaration:\n%s", cPart)
	}
}

// TestSliceNeverHandsOverAnEmptyGoSource is the general form of the previous
// one. `y_len` is a name the emitter invents and the Go never had, and there
// are others. Whenever the quoted name cannot be found in the Go, the whole
// capped source goes rather than an empty string, because an empty Go source
// is the input most likely to turn generator noise into a finding.
func TestSliceNeverHandsOverAnEmptyGoSource(t *testing.T) {
	goSrc := "func K(ctx gpu.Ctx, y []float32) {\n\ty[0] = 1\n}"
	cSrc := "extern \"C\" __global__ void K(float* __restrict__ y, int y_len)\n{\n\ty[0] = 1.0f;\n}"
	block := splitWarnings("k.cu(1): warning #177-D: variable \"y_len\" was declared but never referenced\n")[0]

	goPart, _, err := slice(block, goSrc, cSrc)
	if err != nil {
		t.Fatalf("slice: %v", err)
	}
	if goPart == "" {
		t.Fatal("the Go slice is empty, which is the input most likely to make noise look like a finding")
	}
	if !strings.Contains(goPart, "y[0] = 1") {
		t.Errorf("the fallback did not carry the Go source:\n%s", goPart)
	}
}
