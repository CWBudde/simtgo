package fuzz_test

import (
	"fmt"
	"go/parser"
	"go/token"
	"reflect"
	"sort"
	"strings"
	"testing"
	"testing/fstest"

	"github.com/CWBudde/gocuda/internal/fuzz"
	"github.com/CWBudde/gocuda/simt"
)

// minYield is the fraction of generated programs simt.Transpile has to accept.
//
// It is a floor rather than a measurement. The measurement, on the seeds this
// test uses, is 1.0 -- every program of the first twenty thousand seeds was
// accepted when this was written -- and the floor is set below that so that a
// change to the subset shows up as a drop worth reading rather than as an
// immediate failure. What it is protecting against is the failure mode a
// generator like this has: if the yield falls, the fuzzer is testing the
// refusal path instead of the emitter, and every hour it runs buys nothing.
const minYield = 0.95

// TestYield generates programs and holds the generator to two things at once:
// that the subset takes them, and that what it produced was Go in the first
// place. The second is not implied by the first -- a syntactically broken
// source is refused by Transpile too, and would be counted as a subset
// refusal and hide the real fault.
func TestYield(t *testing.T) {
	const n = 2000
	accepted := 0
	reasons := map[string]int{}
	for i := range n {
		p := fuzz.Generate(int64(i))
		src := p.Source()
		if _, err := parser.ParseFile(token.NewFileSet(), "k.go", src, parser.SkipObjectResolution); err != nil {
			t.Fatalf("seed %d produced source that is not Go: %v\n%s", i, err, src)
		}
		fsys := fstest.MapFS{"k.go": &fstest.MapFile{Data: []byte(src)}}
		if _, err := simt.Transpile(fsys, p.Name()); err != nil {
			reasons[firstLine(err.Error())]++
			continue
		}
		accepted++
	}

	got := float64(accepted) / float64(n)
	t.Logf("yield %d/%d = %.4f", accepted, n, got)
	if got < minYield {
		for _, r := range sortedKeys(reasons) {
			t.Logf("%4d  %s", reasons[r], r)
		}
		t.Fatalf("yield %.4f is below %.2f: the generator is mostly producing refusals, "+
			"so a fuzz run would be testing the refusal path rather than the emitter", got, minYield)
	}
}

// TestDeterminism pins what a fuzz failure is reproducible from.
//
// One number has to be enough to get the same program back: the report of a
// mismatch carries a seed and nothing else, so a generator that consulted the
// clock, a map's iteration order or the global random source would leave every
// failure unreproducible. Generating each seed twice, in an interleaved order
// so that a shared generator would be caught, is what holds it to that.
func TestDeterminism(t *testing.T) {
	const n = 200
	first := make([]string, n)
	for i := range n {
		first[i] = fuzz.Generate(int64(i)).Source()
	}
	for i := n - 1; i >= 0; i-- {
		if got := fuzz.Generate(int64(i)).Source(); got != first[i] {
			t.Fatalf("seed %d produced two different programs\n--- first ---\n%s\n--- again ---\n%s", i, first[i], got)
		}
	}
	// The launch and the buffers have to come back the same way, since a
	// mismatch is only reproducible if what the kernel was handed is too.
	for i := range 20 {
		a, b := fuzz.Generate(int64(i)), fuzz.Generate(int64(i))
		if a.Grid != b.Grid || a.Block != b.Block || a.DynLen != b.DynLen || a.HasDyn != b.HasDyn {
			t.Fatalf("seed %d produced two different launches: %v vs %v", i, a, b)
		}
		x, y := a.Inputs(7), b.Inputs(7)
		if len(x.Vals) != len(y.Vals) {
			t.Fatalf("seed %d produced two different argument lists", i)
		}
		for j := range x.Vals {
			// %v rather than reflect.DeepEqual: a NaN is not equal to itself,
			// and the input distribution puts several in every float buffer.
			if fmt.Sprintf("%v", x.Vals[j]) != fmt.Sprintf("%v", y.Vals[j]) {
				t.Fatalf("seed %d argument %d differs between two draws", i, j)
			}
		}
	}
}

// TestRunsOnTheEmulator is the cheap half of the oracle: the closure rendering
// has to execute without panicking, terminate, and leave the buffers in a
// state somebody can compare. It is separate from the agreement test because
// it runs many more programs -- nothing here compiles anything.
//
// Three failures it is really looking for, each of which would be a generated
// program rather than a transpiler defect, and each of which the emulator
// reports rather than hides: an index outside a buffer, a loop the generator
// cannot bound, and a barrier the threads of a block disagree about. The last
// is the interesting one -- the subset accepts a barrier under a
// short-circuited && or ||, because SPEC.md §6 says the rules do not see into
// a condition, so nothing but running it catches one.
func TestRunsOnTheEmulator(t *testing.T) {
	const n = 1000
	for i := range n {
		p := fuzz.Generate(int64(i))
		args := p.Inputs(int64(i))
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("seed %d panicked on the emulator: %.600v\n%s", i, r, p.Source())
				}
			}()
			p.Run(args)
		}()
	}
}

// TestAliasingIsReadOnly holds the one thing the emulator cannot check.
//
// Two read-only parameters may deliberately be bound to one buffer, because
// what __restrict__ forbids is reaching a modified object through another
// pointer. A written one may not: the emulator never sees the caller's slices,
// so it would compute an answer where the device has undefined behaviour, and
// the difference would look like an emitter bug.
func TestAliasingIsReadOnly(t *testing.T) {
	shared := 0
	for i := range 3000 {
		p := fuzz.Generate(int64(i))
		for j, spec := range p.Params {
			if spec.AliasOf < 0 {
				continue
			}
			shared++
			other := p.Params[spec.AliasOf]
			if !spec.ReadOnly || !other.ReadOnly {
				t.Fatalf("seed %d binds %s and %s to one buffer and writes one of them", i, other.Name, spec.Name)
			}
			if spec.Kind != other.Kind || spec.Len != other.Len {
				t.Fatalf("seed %d binds parameters of different shapes to one buffer", i)
			}
			args := p.Inputs(int64(i))
			if !sameBuffer(args.Vals[j], args.Vals[spec.AliasOf]) {
				t.Fatalf("seed %d says %s aliases %s but handed them different buffers", i, spec.Name, other.Name)
			}
			clone := p.CloneArgs(args)
			if !sameBuffer(clone.Vals[j], clone.Vals[spec.AliasOf]) {
				t.Fatalf("seed %d: CloneArgs unshared %s from %s, which is a different program", i, spec.Name, other.Name)
			}
		}
	}
	if shared == 0 {
		t.Fatal("no generated program ever bound two read-only parameters to one buffer")
	}
	t.Logf("%d aliased parameter pairs over 3000 programs", shared)
}

// sameBuffer reports whether two arguments are one piece of memory, which is
// what aliasing means here -- equal contents would not be the same thing.
func sameBuffer(a, b any) bool {
	x, y := reflect.ValueOf(a), reflect.ValueOf(b)
	if x.Kind() != reflect.Slice || y.Kind() != reflect.Slice {
		return false
	}
	return x.Len() == y.Len() && x.Pointer() == y.Pointer()
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[:i]
	}
	if len(s) > 140 {
		s = s[:140]
	}
	return s
}

func sortedKeys(m map[string]int) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
