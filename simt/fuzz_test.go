package simt_test

import (
	"regexp"
	"strings"
	"testing"
	"testing/fstest"

	"github.com/CWBudde/gocuda/internal/fuzz"
	"github.com/CWBudde/gocuda/internal/fuzz/hostrun"
	"github.com/CWBudde/gocuda/simt"
)

// seeds is the committed corpus, and it is committed because Go runs a fuzz
// target's seeds as an ordinary test under plain "go test", with no -fuzz flag
// and no corpus directory. So every entry here is a regression test that CI
// runs on every push, at the cost of one g++ invocation each, and a case the
// fuzzer finds becomes one by being written to simt/testdata/fuzz/.
//
// The pairs are (program, inputs). They are separate numbers because what a
// kernel computes and what it is handed are independent choices: the same
// program against a different input distribution is a different test, and only
// the first of the two is what the transpiler is being held to.
//
// The program seeds are chosen rather than round: about two thirds of what the
// generator writes is out of the host oracle's scope -- a barrier, a warp
// primitive, an atomic or a shared tile means something a sequential run
// cannot reproduce -- so an obvious corpus of 0, 1, 2, 3 would be a corpus
// that mostly skips. Every seed here lowers, is in scope, and reaches the
// comparison, between them covering two-dimensional blocks, blocks of one
// thread, ragged grids and five-buffer signatures.
var seeds = [][2]int64{
	{3, 1}, {5, 2}, {12, 3}, {13, 4},
	{16, 5}, {21, 6}, {30, 7}, {40, 8},
}

// transcendental is why a program can be in hostrun's scope and still not be
// comparable.
//
// sinf on x86-64 and Go's float32(math.Sin(float64(x))) are two different
// implementations of a function neither language requires to be correctly
// rounded, so they disagree in the last bits for ordinary arguments and
// disagree completely for large ones -- expf overflows to an infinity at an
// argument where Go's math.Exp still returns a finite double, and a relative
// bound relates an infinity to nothing. That is a real difference between two
// runtimes and not a fault in the translation, which is what this target is
// looking for, so these programs are left to the NVRTC oracle.
//
// sqrtf stays: IEEE 754 requires it to be correctly rounded, so the two agree
// to within the double rounding Go's float64 route costs, which is what the
// tolerance below is for.
var transcendental = regexp.MustCompile(`\b(sinf?|cosf?|expf?|logf?)\b`)

// hostTol bounds what sqrtf may differ by.
//
// Go computes a float32 square root by widening to float64, taking the root
// and rounding back, which is two roundings where the host's sqrtf does one.
// It is the same bound internal/fuzz/hostrun's Magnitude case needed for the
// same reason, and internal/tolerance's rule is what applies it -- so a result
// that is bit-identical, which is almost all of them, never consults it at all.
const hostTol = 1e-6

// FuzzHostAgreesWithEmulator is the differential the whole round is for:
// generate a program, lower it, compile the generated CUDA C with a host C++
// compiler, run it, and compare element by element against the same program
// under gpu.RunCPU.
//
// The two sides are the same IR rendered twice -- as Go source for the
// transpiler and as a func(gpu.Ctx) closure for the emulator -- so a
// disagreement is a disagreement about what the Go meant, which is exactly the
// class of defect that compiles cleanly and computes something else. NVRTC
// accepting the source cannot see any of it.
//
// A mismatch is a finding, not a verdict about which side is wrong. The
// emulator has been the wrong one before: gpu.Fmin propagated a NaN where
// fminf does not, and that was the oracle disagreeing with the device rather
// than the transpiler mistranslating anything.
func FuzzHostAgreesWithEmulator(f *testing.F) {
	for _, s := range seeds {
		f.Add(s[0], s[1])
	}
	if err := hostrun.Available(); err != nil {
		f.Skipf("no host C++ compiler: %v", err)
	}

	f.Fuzz(func(t *testing.T, seed, inputSeed int64) {
		p := fuzz.Generate(seed)
		src := p.Source()
		fsys := fstest.MapFS{"k.go": &fstest.MapFile{Data: []byte(src)}}

		u, err := simt.Transpile(fsys, p.Name())
		if err != nil {
			// The generator writes inside the subset by construction, and the
			// measured yield is 1.0, so a refusal here is a finding either
			// way round: the generator has drifted outside the subset, or the
			// subset has narrowed under it.
			t.Fatalf("seed %d: the subset refused a generated program: %v\n%s", seed, err, src)
		}

		// Out of the oracle's scope rather than wrong: a barrier, a warp
		// primitive, an atomic or a shared tile cannot mean the same thing
		// when the threads are run one after another. hostrun says so itself
		// rather than being second-guessed here, and those programs are the
		// NVRTC oracle's.
		if err := hostrun.CheckScope(u); err != nil {
			t.Skip(err.Error())
		}
		if transcendental.MatchString(u.Source) {
			t.Skip("a transcendental is two implementations of one function, not a translation")
		}

		in := p.Inputs(inputSeed)
		host, cpu := p.CloneArgs(in), p.CloneArgs(in)

		if err := hostrun.Run(u, p.Grid, p.Block, host.Vals...); err != nil {
			t.Fatalf("seed %d: %v\n%s\n%s", seed, err, src, u.Source)
		}
		p.Run(cpu)

		if bad := p.Compare(host, cpu, hostTol); len(bad) > 0 {
			var b strings.Builder
			for _, m := range bad {
				b.WriteString("\n  " + m.String())
			}
			t.Fatalf("seed %d, inputs %d: the host run and the emulator disagree.\n"+
				"That is a finding about one of them and not a verdict about which:%s\n\n%s\n%s\n%s",
				seed, inputSeed, b.String(), p, src, u.Source)
		}
	})
}

// FuzzOutsideTheSubsetIsRefused is the negative target, and the only oracle
// there is for a rule whose whole content is that something is refused.
//
// A refusal that fires in a three-line kernel has not been shown to fire in a
// real one. These edits put each broken rule inside a few hundred lines of
// generated control flow, under whatever names and types the generator chose,
// which is where a rule that reads the wrong scope or stops at the first
// function quietly stops working. simt/errors_test.go pins the diagnostics;
// this pins that they still appear with a program around them.
func FuzzOutsideTheSubsetIsRefused(f *testing.F) {
	for i := range fuzz.Violations {
		f.Add(int64(i*7919), i)
	}

	f.Fuzz(func(t *testing.T, seed int64, which int) {
		p := fuzz.Generate(seed)
		src, v := fuzz.Violate(p, which)
		fsys := fstest.MapFS{"k.go": &fstest.MapFile{Data: []byte(src)}}

		_, err := simt.Transpile(fsys, p.Name())
		if err == nil {
			t.Fatalf("seed %d: the subset accepted %s\n%s", seed, v.Name, src)
		}
		if !strings.Contains(err.Error(), v.Want) {
			// Refused, but for something else -- which would leave the rule
			// under test unexercised while the target stayed green.
			t.Fatalf("seed %d: %s was refused with the wrong diagnostic:\ngot  %v\nwant it to mention %q\n%s",
				seed, v.Name, err, v.Want, src)
		}
	})
}
