package fuzz_test

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
	"testing"

	"github.com/CWBudde/simtgo/internal/fuzz"
)

// Feature coverage: what fraction of generated programs contain each construct.
//
// This existed as a table in docs/verification.md before it existed as a test,
// which made "warp primitives are at 7.5%" a sentence somebody had measured
// once and nobody could re-measure. A number in prose drifts silently -- the
// generator changes, the figure does not -- and the whole register of this
// repository is that a claim says where it was measured. So the table is
// computed here, and the floors below are what stop it sliding back.
//
// Coverage is read off the generated **Go source**, not the IR. That is the
// artifact the oracles actually consume, it is what a reader checks by opening
// one of these programs, and it cannot silently miss a construct the way a
// hand-written IR walker can -- which is a failure this generator has had
// once already, in the walker that could not see a Field.

// coverageFeatures is the table, one regexp per construct. Each is matched
// against a whole program, so a figure is "programs that reach this" rather
// than "times this appears".
//
// "warp _sync vocabulary" is every warp primitive that lowers to a built-in
// taking a mask, and it is deliberately not the whole warp table.
// ctx.LaneID() is in internal/lower's ctxWarp with the rest, but it lowers to
// an arithmetic expression rather than a _sync built-in, and the generator
// reaches it two ways -- through warpExpr, behind all seven gates, and as an
// ordinary thread-varying position behind only warpOK. The two are identical
// in the source text, so a single figure covering both would move when either
// moved. Splitting them is what makes each number mean one thing.
var coverageFeatures = []struct {
	name string
	re   *regexp.Regexp
}{
	{"switch", regexp.MustCompile(`\bswitch\b`)},
	{"range", regexp.MustCompile(`\brange\b`)},
	{"narrow element types", regexp.MustCompile(`\[\](?:u?int8|u?int16)\b`)},
	{"SyncThreads", regexp.MustCompile(`ctx\.SyncThreads\(\)`)},
	{"array locals", regexp.MustCompile(`\bvar \w+ \[\d+\]`)},
	{"shared memory", regexp.MustCompile(`ctx\.Shared(?:Dyn)?[A-Z]`)},
	{"structs", regexp.MustCompile(`\bS(?:Pair|Hole|Wide|Tail)\b`)},
	{"atomics", regexp.MustCompile(`gpu\.Atomic[A-Z]`)},
	{"float64", regexp.MustCompile(`\bfloat64\b`)},
	{"device funcs declared", regexp.MustCompile(`(?m)^func h\d+\(`)},
	{"warp _sync vocabulary", regexp.MustCompile(
		`ctx\.(?:Shuffle(?:Xor|Up|Down)?(?:F32|I32)|Ballot|Any|All|ActiveMask|SyncWarp)\(`)},
	{"LaneID", regexp.MustCompile(`ctx\.LaneID\(\)`)},
}

// coverageFloors are the figures this generator is held to. They sit below
// what is measured, for TestYield's reason: a floor at the measured value
// fails on the next harmless change to the mix, and then somebody edits the
// floor instead of reading the number.
//
// The warp floor is the one with a story. Warp expressions sit behind a
// conjunction of seven gates while a barrier sits behind three and owns a
// whole statement slot, which is the whole of why one was at 43% and the other
// at 10.7% when this test was written. None of those seven gates is
// negotiable: each guards real undefined behaviour, so the rate was raised by
// changing the draws upstream of them -- a statement slot of warp's own, a
// second slot in each expression switch, and more whole-warp block widths --
// and it is 19.8% now.
//
// There is a ceiling above it, and it is worth knowing before anybody tries to
// push this figure further. A warp primitive needs g.sync, a coin flip, and
// g.warpOK, about 0.64 now: no more than a third of programs can carry one at
// all. Raising g.sync is the obvious next lever and it is the wrong one -- a
// program with a barrier or a warp primitive is outside the host oracle's
// scope, so that trade buys warp coverage directly out of the differential's
// reach.
var coverageFloors = map[string]float64{
	"switch":                0.70,
	"range":                 0.60,
	"narrow element types":  0.55,
	"SyncThreads":           0.35,
	"array locals":          0.35,
	"shared memory":         0.30,
	"structs":               0.25,
	"atomics":               0.25,
	"float64":               0.18,
	"device funcs declared": 0.15,
	"warp _sync vocabulary": 0.15,
	"LaneID":                0.18,
}

// TestFeatureCoverage measures what fraction of generated programs reach each
// construct, and fails when one falls below its floor.
//
// The seeds are the ones a fuzz run would start from, so this measures the
// generator as it is actually used rather than a sample chosen to flatter it.
func TestFeatureCoverage(t *testing.T) {
	const n = 3000

	hits := map[string]int{}
	for i := range n {
		src := fuzz.Generate(int64(i)).Source()
		for _, f := range coverageFeatures {
			if f.re.MatchString(src) {
				hits[f.name]++
			}
		}
	}

	var b strings.Builder
	fmt.Fprintf(&b, "feature coverage over %d programs\n", n)
	for _, f := range coverageFeatures {
		fmt.Fprintf(&b, "  %-22s %5.1f%%  (%d)\n",
			f.name, 100*float64(hits[f.name])/float64(n), hits[f.name])
	}
	t.Log("\n" + b.String())

	for _, name := range sortedFloors() {
		got := float64(hits[name]) / float64(n)
		if got < coverageFloors[name] {
			t.Errorf("%s: coverage %.3f is below the floor %.3f -- the generator has stopped reaching it as often",
				name, got, coverageFloors[name])
		}
	}
}

// TestTheTwoTablesAgree keeps coverageFeatures and coverageFloors from
// drifting apart, and it checks **both** directions because each has its own
// way of going quiet.
//
// A floor for a feature nobody measures is a check that never runs. A feature
// with no floor is worse and less obvious: TestFeatureCoverage iterates the
// floors, so such a row is printed and then not asserted, and the generator
// could stop emitting it entirely with the table still green. LaneID was
// exactly that until review caught it.
func TestTheTwoTablesAgree(t *testing.T) {
	measured := map[string]bool{}
	for _, f := range coverageFeatures {
		measured[f.name] = true
		if _, ok := coverageFloors[f.name]; !ok {
			t.Errorf("coverageFeatures measures %q, which has no floor -- it would be reported and never asserted", f.name)
		}
	}
	for name := range coverageFloors {
		if !measured[name] {
			t.Errorf("coverageFloors has %q, which coverageFeatures does not measure", name)
		}
	}
}

func sortedFloors() []string {
	out := make([]string, 0, len(coverageFloors))
	for k := range coverageFloors {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
