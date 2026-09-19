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
