package simt_test

import (
	"context"
	"os"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/CWBudde/gocuda/internal/typesafe"
)

// DocReviewEnv turns the advisory document reviews on. They are off by default
// because they reach a network service, and ci.yml must keep running on a
// machine with nothing installed.
const DocReviewEnv = "GOCUDA_DOC_REVIEW"

// What spec_test.go checks is that SPEC.md and simt/errors_test.go quote the
// same diagnostic phrase, in both directions. What nothing checks is the
// English to the left of the em-dash:
//
//	- a narrow local — diagnostic: `not a variable, a parameter or a result`
//	  ^^^^^^^^^^^^^^ this
//
// That half is what a reader actually reads, and it can drift from the rule
// without either direction of the exact-phrase check noticing: the phrase is
// pinned, the sentence around it is not. Comparing a sentence to a rule is not
// something a regular expression does, so this asks.
//
// It is advisory and it is opt-in, for a measured reason. Swapping each
// bullet's description with a neighbouring rule's from the same section --
// the hardest negatives available -- it agreed with the document on 72 of 78
// real bullets and caught 63 of 78 swaps at a threshold of 0.5. Six false
// flags in 78 is useful to a person editing SPEC.md and unusable as a gate,
// so it is a thing you run, not a thing that runs on you.

// specBullet is one Refusals entry: the sentence, and the phrase that joins it
// to the test which pins it.
type specBullet struct {
	section string
	desc    string
	phrase  string
}

// parseSpecBullets pulls the Refusals entries out of SPEC.md. A bullet whose
// diagnostic did not fit on one line continues on the next, indented, so the
// continuation is joined back on before the phrase is looked for.
func parseSpecBullets(spec string) []specBullet {
	lines := strings.Split(spec, "\n")
	var out []specBullet
	section := ""
	for i := 0; i < len(lines); i++ {
		if after, ok := strings.CutPrefix(lines[i], "### "); ok {
			section = after
			continue
		}
		if !strings.HasPrefix(lines[i], "- ") {
			continue
		}
		joined := lines[i]
		for !specDiagnostic.MatchString(joined) && i+1 < len(lines) &&
			strings.HasPrefix(lines[i+1], "  ") && strings.TrimSpace(lines[i+1]) != "" {
			i++
			joined += " " + strings.TrimSpace(lines[i])
		}
		m := specDiagnostic.FindStringSubmatch(joined)
		if m == nil {
			continue
		}
		desc := joined
		if k := strings.Index(joined, "— diagnostic:"); k >= 0 {
			desc = joined[:k]
		}
		desc = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(desc), "- "))
		out = append(out, specBullet{section: section, desc: desc, phrase: diagnosticOf(m)})
	}
	return out
}

// TestSpecProseDescribesTheRule reads every Refusals bullet next to the source
// that triggers it and reports the ones that no longer read like each other.
func TestSpecProseDescribesTheRule(t *testing.T) {
	if os.Getenv(DocReviewEnv) == "" {
		t.Skipf("%s is not set", DocReviewEnv)
	}
	if !typesafe.Available() {
		t.Skipf("%s is not set", typesafe.KeyEnv)
	}

	byWant := make(map[string]refusal, len(refusals))
	for _, r := range refusals {
		if _, seen := byWant[r.want]; !seen {
			byWant[r.want] = r
		}
	}
	bullets := parseSpecBullets(readSpec(t))
	if len(bullets) == 0 {
		t.Fatal("no Refusals bullets found in SPEC.md; the document's shape has changed")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	type finding struct {
		bullet specBullet
		name   string
		score  float64
	}
	var flagged, worth []finding
	asked := 0
	for _, b := range bullets {
		r, ok := byWant[b.phrase]
		if !ok {
			// spec_test.go owns this failure; here it only means there is
			// nothing to compare the sentence against.
			continue
		}
		score, err := judgeSpecBullet(ctx, b, refusalPrelude+r.body)
		if err != nil {
			t.Logf("%-50s unreachable: %v", r.name, err)
			continue
		}
		asked++
		switch {
		case score < 0.3:
			flagged = append(flagged, finding{b, r.name, score})
		case score < 0.5:
			worth = append(worth, finding{b, r.name, score})
		}
	}
	if asked == 0 {
		t.Skip("no bullet could be asked about")
	}

	sort.Slice(worth, func(i, j int) bool { return worth[i].score < worth[j].score })
	for _, f := range worth {
		t.Logf("worth a look (%.2f) [%s] %q\n\tpinned by %q", f.score, f.bullet.section, f.bullet.desc, f.name)
	}

	sort.Slice(flagged, func(i, j int) bool { return flagged[i].score < flagged[j].score })
	for _, f := range flagged {
		t.Logf("does not read like its rule (%.2f)\n"+
			"\tsection:    %s\n\tsays:       %s\n\tdiagnostic: %s\n\tpinned by:  %s",
			f.score, f.bullet.section, f.bullet.desc, f.bullet.phrase, f.name)
	}

	// This reports and does not fail, and the reason is the measurement rather
	// than timidity. Six of 78 correct bullets came back below the same
	// threshold these are below, so a red result here would mean "somebody
	// look", not "something is wrong" -- and a test that says the first while
	// looking like the second is one that gets wired into CI by mistake and
	// then ignored. Read the list; spec_test.go is still the gate.
	t.Logf("read %d of %d Refusals bullets: %d do not read like their rule, %d are worth a look. "+
		"Advisory -- roughly one in twenty correct bullets lands here.",
		asked, len(bullets), len(flagged), len(worth))
}

// judgeSpecBullet asks whether one sentence describes the construct in the
// source that the rule refuses. The source is the test's own body, so what is
// being compared is the document against the thing the code actually does.
func judgeSpecBullet(ctx context.Context, b specBullet, src string) (float64, error) {
	res, err := typesafe.Ask(ctx,
		map[string]any{
			"spec_description":  b.desc,
			"kernel_source":     src,
			"diagnostic_phrase": b.phrase,
		},
		map[string]typesafe.Question{
			"describes": typesafe.Noul(
				map[string]any{
					"question": "A compiler refuses to translate `kernel_source` and reports a message containing " +
						"`diagnostic_phrase`. A specification document describes the rejected construct as " +
						"`spec_description`. Does `spec_description` describe the construct in `kernel_source` " +
						"that was rejected?",
					"focus": "Judge whether the description names the construct actually present in the source. " +
						"Ignore wording style; judge the construct.",
				},
				"The description names the construct that is in the source. A reader given the description "+
					"would write this source.",
				"The description names a different construct. The source does not contain what the description "+
					"says, so the document and the rule have drifted apart."),
		})
	if err != nil {
		return 0, err
	}
	return res.Answers["describes"].Noul, nil
}
