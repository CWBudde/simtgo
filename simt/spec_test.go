package simt_test

import (
	"os"
	"regexp"
	"sort"
	"strings"
	"testing"
)

// SPEC.md is the contract, and a contract nothing checks is a description of
// what the code used to do. These two tests are what keep it honest, in both
// directions: every refusal the transpiler enforces must appear in the
// document, and every refusal the document claims must be pinned by a test.
//
// The key is the diagnostic's own wording rather than a rule number invented
// for the purpose. That costs a little precision -- two rules that happen to
// share a phrase are one entry -- and buys the property that matters: the
// document quotes what a reader will actually see in their terminal, so the
// spec and the message cannot drift apart without one of these failing.
const specPath = "../SPEC.md"

// specDiagnostic matches the quoted phrase in a Refusals entry, written as
// "diagnostic: `phrase`". Several diagnostics quote an operator and so contain
// a backtick themselves -- "`+=` is refused anyway" -- and those are delimited
// by a doubled backtick with one space of padding, which is Markdown's own
// answer to the same problem. The doubled form is tried first, because a
// single-backtick pattern would otherwise match only as far as the phrase's
// own first backtick and silently truncate it.
var specDiagnostic = regexp.MustCompile("diagnostic: (?:``(.+?)``|`([^`]+)`)")

// diagnosticOf returns whichever alternative matched.
func diagnosticOf(m []string) string {
	if m[1] != "" {
		return strings.TrimSpace(m[1])
	}
	return m[2]
}

func readSpec(t *testing.T) string {
	t.Helper()
	b, err := os.ReadFile(specPath)
	if err != nil {
		t.Fatalf("read %s: %v", specPath, err)
	}
	return string(b)
}

// TestSpecCoversEveryRefusal fails when the subset gains a refusal the
// contract does not mention.
func TestSpecCoversEveryRefusal(t *testing.T) {
	spec := readSpec(t)

	missing := map[string]string{} // want -> the case that needs it
	for _, r := range refusals {
		if !strings.Contains(spec, r.want) {
			if _, seen := missing[r.want]; !seen {
				missing[r.want] = r.name
			}
		}
	}
	if len(missing) == 0 {
		return
	}
	keys := make([]string, 0, len(missing))
	for k := range missing {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		t.Errorf("%s does not carry the diagnostic for %q:\n\tdiagnostic: `%s`", specPath, missing[k], k)
	}
	t.Logf("%d refusal(s) enforced by the transpiler and absent from the contract", len(missing))
}

// TestEverySpecRefusalIsPinned fails the other way: a rule the document
// claims, that no test holds the transpiler to.
func TestEverySpecRefusalIsPinned(t *testing.T) {
	spec := readSpec(t)

	pinned := make(map[string]bool, len(refusals))
	for _, r := range refusals {
		pinned[r.want] = true
	}
	for _, m := range specDiagnostic.FindAllStringSubmatch(spec, -1) {
		if d := diagnosticOf(m); !pinned[d] {
			t.Errorf("%s claims a refusal no case in refusals pins:\n\tdiagnostic: `%s`", specPath, d)
		}
	}
}
