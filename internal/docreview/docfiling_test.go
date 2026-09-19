package docreview_test

import (
	"context"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/CWBudde/simtgo/internal/typesafe"
)

// DocReviewEnv turns this on. It is off by default: it reaches a network
// service, and ci.yml runs on a machine with nothing installed.
const DocReviewEnv = "SIMTGO_DOC_REVIEW"

// This asks, for each section of the engineering record, which page owns its
// subject, and reports the ones it would have filed elsewhere.
//
// What it is worth, measured over all 72 sections of docs/, SPEC.md and
// NUMERICS.md, against docs/README.md's own table as the rubric: it agreed
// with where the section actually is 55 times out of 72.
//
// The disagreements are not evenly useful, and one of them is worth knowing
// before reading any output. docs/tile.md's "What the sanitizer sweep covers
// here" was assigned to verification.md at confidence 1.00. That section is
// filed correctly -- it is a tile-track fact that already links to
// verification.md -- and the highest confidence available did not help. The
// reason is the rubric rather than the sections: this routes by topic, and
// the repository files by subject ownership, so a tile-specific fact about
// verification belongs to tile and reads like verification.
//
// So: confidence here does not mean correctness, three quarters is the
// agreement rate to expect, and this reports rather than decides. It exists
// because the table had nothing checking it at all, and a list a person reads
// once after moving text around is worth more than nothing.

// owners is docs/README.md's table, written as a rubric. Keep it in step with
// that table: it is the document's claim about itself, and this only checks
// the record against it.
var owners = map[string]any{
	"docs/decisions.md": map[string]any{
		"what":    "A settled question and the reasoning that settled it: why the project chose one approach over another, and what the rejected alternative was.",
		"not_for": "A measurement, a bug post-mortem, or a description of the test layers.",
	},
	"docs/emitter-defects.md": map[string]any{
		"what":    "A specific mistranslation the transpiler was found to produce: the wrong output, its mechanism, and which test catches it now.",
		"not_for": "A design decision, or a general description of how checking works.",
	},
	"docs/verification.md": map[string]any{
		"what":    "A layer of checking -- a kind of test or oracle -- and what that layer cannot see. The structure of the testing strategy.",
		"not_for": "An individual bug, or a measured hardware number.",
	},
	"docs/toolchain.md": map[string]any{
		"what":    "Measured, observed behaviour of the CUDA driver, NVRTC or PTX on real hardware: what was run, and the numbers or yes/no answers that came back.",
		"not_for": "Project design choices, or the behaviour of the Go-side code.",
	},
	"docs/tile.md": map[string]any{
		"what":    "The tile track specifically: graph recording, kernel fusion, halo staging, aliasing between tiles, and where the tile track stops.",
		"not_for": "The SIMT track, the driver, or general verification.",
	},
	"SPEC.md": map[string]any{
		"what":    "The contract of the supported Go subset: exactly what the transpiler accepts and what it refuses, stated as a rule a user must obey.",
		"not_for": "Why a rule was chosen, or how it is tested.",
	},
	"NUMERICS.md": map[string]any{
		"what":    "Floating-point behaviour on the device and what a test is therefore allowed to assert about a float32 or float64 result.",
		"not_for": "Integer semantics, or compilation behaviour.",
	},
}

const repoRoot = "../.."

// section is one "## " heading and the prose under it.
type section struct {
	file    string
	heading string
	body    string
}

// splitSections cuts a Markdown file into its level-two sections. Fenced code
// is tracked so that a "## " inside a fence is not read as a heading.
func splitSections(path, rel string, maxBody int) ([]section, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var out []section
	var cur *section
	fenced := false
	for _, line := range strings.Split(string(raw), "\n") {
		if strings.HasPrefix(line, "```") {
			fenced = !fenced
		}
		if after, ok := strings.CutPrefix(line, "## "); ok && !fenced {
			out = append(out, section{file: rel, heading: after})
			cur = &out[len(out)-1]
			continue
		}
		if cur != nil {
			cur.body += line + "\n"
		}
	}
	for i := range out {
		out[i].body = strings.TrimSpace(out[i].body)
		if len(out[i].body) > maxBody {
			out[i].body = out[i].body[:maxBody] + "\n..."
		}
	}
	return out, nil
}

// TestDocSectionsAreFiledBySubject reports, and never fails. See the note
// above for what its output is worth.
func TestDocSectionsAreFiledBySubject(t *testing.T) {
	if os.Getenv(DocReviewEnv) == "" {
		t.Skipf("%s is not set", DocReviewEnv)
	}
	if !typesafe.Available() {
		t.Skipf("%s is not set", typesafe.KeyEnv)
	}

	var all []section
	for rel := range owners {
		secs, err := splitSections(filepath.Join(repoRoot, rel), rel, 1800)
		if err != nil {
			t.Fatalf("read %s: %v", rel, err)
		}
		for _, s := range secs {
			// A short section is a pointer or an index, not a claim with a home.
			if len(s.body) >= 200 {
				all = append(all, s)
			}
		}
	}
	sort.Slice(all, func(i, j int) bool { return all[i].file+all[i].heading < all[j].file+all[j].heading })

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	agreed, asked := 0, 0
	for _, s := range all {
		where, confidence, err := judgeOwner(ctx, s)
		if err != nil {
			t.Logf("%s / %s: unreachable: %v", s.file, s.heading, err)
			continue
		}
		asked++
		if where == s.file {
			agreed++
			continue
		}
		t.Logf("would file elsewhere (confidence %.2f)\n\tsection: %s\n\tis in:   %s\n\tsuggests: %s",
			confidence, s.heading, s.file, where)
	}
	if asked == 0 {
		t.Skip("no section could be asked about")
	}
	t.Logf("read %d sections, agreed with %d. Advisory: the measured agreement rate is about "+
		"three quarters, and a disagreement at high confidence has been wrong before.", asked, agreed)
}

func judgeOwner(ctx context.Context, s section) (string, float64, error) {
	res, err := typesafe.Ask(ctx,
		map[string]any{"heading": s.heading, "text": s.body},
		map[string]typesafe.Question{
			"owner": typesafe.Choice(
				map[string]any{
					"question": "This is one section of engineering documentation from a single repository. " +
						"Which of the repository's documents is the right home for it?",
					"focus": "Judge by what the section is about, using `heading` and `text`. Pick the " +
						"document whose subject the section belongs to.",
				},
				owners),
		})
	if err != nil {
		return "", 0, err
	}
	a := res.Answers["owner"]
	return a.Choice, a.Confidence, nil
}
