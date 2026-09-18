package hostrun_test

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	gocuda "github.com/CWBudde/gocuda"
	"github.com/CWBudde/gocuda/gpu"
	"github.com/CWBudde/gocuda/internal/fuzz/hostrun"
	"github.com/CWBudde/gocuda/internal/lower"
	"github.com/CWBudde/gocuda/internal/tolerance"
	"github.com/CWBudde/gocuda/kernels"
	"github.com/CWBudde/gocuda/simt"
)

// TestMain points the driver cache at a directory of its own.
//
// A test that wrote into the user's cache would leave a few megabytes behind
// and, worse, would report a warm compile as a cold one the second time it
// ran. Here every run starts with nothing, so the cost this file logs is the
// cost a fuzzer pays on its first case.
func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "gocuda-hostrun-test-")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if err := os.Setenv("GOCUDA_HOSTRUN_CACHE", dir); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	code := m.Run()
	if err := os.RemoveAll(dir); err != nil {
		fmt.Fprintln(os.Stderr, err)
	}
	os.Exit(code)
}

func transpile(t *testing.T, name string) *lower.Unit {
	t.Helper()
	u, err := simt.Transpile(gocuda.Kernels(), name)
	if err != nil {
		t.Fatalf("transpiling %s: %v", name, err)
	}
	return u
}

func requireCompiler(t *testing.T) {
	t.Helper()
	if err := hostrun.Available(); err != nil {
		t.Skipf("%v", err)
	}
}

// A hostCase is one kernel, one launch and one set of inputs.
//
// args is a constructor rather than a value because it is called twice, once
// per backend: the two runs must start from bit-identical inputs that neither
// can see the other write. call applies the Go kernel to whichever set the
// emulator was given, by asserting the same types Run will derive by
// reflection -- so a case whose argument list does not match the kernel fails
// in the test rather than being silently reinterpreted.
type hostCase struct {
	name        string
	grid, block gpu.Dim
	args        func() []any
	call        func(c gpu.Ctx, a []any)

	// tol is a relative bound for the float32 outputs, per internal/tolerance's
	// rule. Zero means the comparison is bit-exact, which is what almost every
	// case here can demand: see the comment on compare.
	tol float64
}

// cases covers the eight committed kernels hostrun is in scope for.
//
// Every geometry is ragged -- n is never a multiple of the block -- because
// the tail is where a kernel's bounds guard is, and a guard is exactly the
// kind of thing a source-level translation can get subtly wrong.
func cases() []hostCase {
	return []hostCase{
		{
			name: "VecAdd", grid: gpu.D1(5), block: gpu.D1(8),
			args: func() []any {
				// Inf against -Inf gives a NaN, and the two zeros add to +0
				// either way round; both sides should say so identically.
				return []any{make([]float32, 37), floats(1, 37, true), floats(2, 37, true)}
			},
			call: func(c gpu.Ctx, a []any) {
				kernels.VecAdd(c, a[0].([]float32), a[1].([]float32), a[2].([]float32))
			},
		},
		{
			name: "Scale", grid: gpu.D1(4), block: gpu.D1(16),
			args: func() []any {
				// A negative zero for k, so that k*x carries the sign rule
				// through a multiply rather than only through a copy.
				return []any{make([]float32, 61), floats(3, 61, true), float32(math.Copysign(0, -1))}
			},
			call: func(c gpu.Ctx, a []any) {
				kernels.Scale(c, a[0].([]float32), a[1].([]float32), a[2].(float32))
			},
		},
		{
			name: "Magnitude", grid: gpu.D1(7), block: gpu.D1(8),
			args: func() []any {
				return []any{make([]float32, 53), floats(4, 53, true), floats(5, 53, true)}
			},
			call: func(c gpu.Ctx, a []any) {
				kernels.Magnitude(c, a[0].([]float32), a[1].([]float32), a[2].([]float32))
			},
			// The one case that is not held to the bits, and the reason is not
			// the emitter: gpu.Hypot is float32(math.Hypot(float64, float64))
			// and the host's is glibc's hypotf, so a last-bit difference here
			// would say nothing about the translation. NUMERICS.md's 1e-6 is
			// its bound for one or two operations per element.
			//
			// On this machine -- glibc 2.39, go1.26.0, amd64 -- the bound has
			// never been needed: every element of this case agrees bit for
			// bit, and compare() logs it if that stops being true. It is a
			// fact about two libraries rather than a guarantee, which is why
			// the bound stays.
			tol: 1e-6,
		},
		{
			name: "Softclip", grid: gpu.D1(6), block: gpu.D1(8),
			args: func() []any {
				return []any{make([]float32, 45), floats(6, 45, true), float32(0.75)}
			},
			call: func(c gpu.Ctx, a []any) {
				kernels.Softclip(c, a[0].([]float32), a[1].([]float32), a[2].(float32))
			},
		},
		{
			name: "Classify", grid: gpu.D1(5), block: gpu.D1(8),
			args: func() []any {
				// The edges stay finite. Classify sums them into bias, so one
				// infinity there would make every output NaN and the case
				// would stop discriminating anything.
				return []any{make([]float32, 39), floats(7, 39, true), []float32{0.125, 0.25, 0.5, 1, 2}}
			},
			call: func(c gpu.Ctx, a []any) {
				kernels.Classify(c, a[0].([]float32), a[1].([]float32), a[2].([]float32))
			},
		},
		{
			name: "Gray", grid: gpu.D1(4), block: gpu.D1(8),
			args: func() []any {
				const n = 29 // pixels; the buffer is three bytes each
				out := make([]uint8, n)
				rgb := make([]uint8, 3*n)
				r := rand.New(rand.NewPCG(8, 99))
				for i := range rgb {
					rgb[i] = uint8(r.UintN(256))
				}
				// The corners of the byte range, where a shift that lost a bit
				// or a promotion that wrapped would show.
				copy(rgb, []uint8{0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 255})
				return []any{out, rgb}
			},
			call: func(c gpu.Ctx, a []any) {
				kernels.Gray(c, a[0].([]uint8), a[1].([]uint8))
			},
		},
		{
			name: "Quantize", grid: gpu.D1(6), block: gpu.D1(8),
			args: func() []any {
				const n = 43
				// Finite values only, and that exclusion is a real limit
				// rather than tidiness. Quantize converts a float to an int,
				// which C++ leaves undefined when the value does not fit and
				// Go leaves implementation-specific; a NaN or an infinity here
				// would be comparing two undefined behaviours and calling the
				// difference a transpiler bug.
				return []any{
					make([]int32, n), make([]int64, n), make([]bool, n),
					floats(9, n, false), float32(0.01), uint32(0x9e3779b9),
				}
			},
			call: func(c gpu.Ctx, a []any) {
				kernels.Quantize(c, a[0].([]int32), a[1].([]int64), a[2].([]bool),
					a[3].([]float32), a[4].(float32), a[5].(uint32))
			},
		},
		{
			name: "BandGain", grid: gpu.D1(5), block: gpu.D1(8),
			args: func() []any {
				const n = 33
				bands := []kernels.Band{{Upper: 0.25, Gain: 0.5}, {Upper: 1, Gain: 1}, {Upper: 4, Gain: 2}}
				cfg := kernels.Shape{Floor: 0.125, Count: 3, Bias: 1.0 / 3}
				return []any{make([]float32, n), floats(10, n, true), bands, cfg}
			},
			call: func(c gpu.Ctx, a []any) {
				kernels.BandGain(c, a[0].([]float32), a[1].([]float32),
					a[2].([]kernels.Band), a[3].(kernels.Shape))
			},
		},
	}
}

// specials are the float32 values NUMERICS.md records as never having been
// through any of this: the two zeros, the two infinities, a NaN and the
// bottom of the subnormal range, plus the extremes of the normal one.
//
// Neither backend flushes a subnormal -- the emulator is measured not to and
// the host compiler is not asked to -- so a disagreement about one is a
// disagreement about the translation, which is the whole point of putting them
// here.
func specials() []float32 {
	return []float32{
		0,
		float32(math.Copysign(0, -1)),
		float32(math.Inf(1)),
		float32(math.Inf(-1)),
		float32(math.NaN()),
		math.SmallestNonzeroFloat32,
		-math.SmallestNonzeroFloat32 * 5,
		math.MaxFloat32,
		-math.MaxFloat32,
	}
}

// floats builds n deterministic float32 samples, optionally opening with the
// special values so that every kernel meets them at a different index.
func floats(seed uint64, n int, withSpecials bool) []float32 {
	out := make([]float32, n)
	r := rand.New(rand.NewPCG(seed, 0x5eed))
	for i := range out {
		out[i] = float32(r.NormFloat64()) * 2
	}
	if withSpecials {
		copy(out, specials())
	}
	return out
}

// TestKernelsAgreeWithEmulator is the acceptance test: every kernel in scope,
// run twice from identical inputs, once as the Go the author wrote and once as
// the CUDA C the emitter produced.
func TestKernelsAgreeWithEmulator(t *testing.T) {
	requireCompiler(t)
	for _, c := range cases() {
		t.Run(c.name, func(t *testing.T) {
			u := transpile(t, c.name)

			want := c.args()
			gpu.RunCPUDim(c.grid, c.block, func(ctx gpu.Ctx) { c.call(ctx, want) })

			got := c.args()
			cold := time.Now()
			if err := hostrun.Run(u, c.grid, c.block, got...); err != nil {
				t.Fatalf("hostrun: %v", err)
			}
			first := time.Since(cold)

			// The same call again, which the cache answers without a compiler.
			// Logging both is how the per-case cost in the report was measured,
			// and it is also a check: a second run that disagreed with the
			// first would mean the driver is not a function of its input.
			warm := time.Now()
			again := c.args()
			if err := hostrun.Run(u, c.grid, c.block, again...); err != nil {
				t.Fatalf("hostrun, second call: %v", err)
			}
			t.Logf("compile+run %v, run alone %v", first.Round(time.Millisecond), time.Since(warm).Round(time.Microsecond))

			for i, p := range u.Params {
				if !p.Slice {
					continue
				}
				if d := compare(t, p.Name, got[i], want[i], c.tol); d > 0 {
					t.Errorf("%s: %d elements disagree between the host and the emulator", p.Name, d)
				}
				if d := compare(t, p.Name+" (rerun)", again[i], got[i], c.tol); d > 0 {
					t.Errorf("%s: the second host run disagreed with the first in %d elements", p.Name, d)
				}
			}
		})
	}
}

// TestMutationIsDetected perturbs one operator in the generated C and demands
// that the oracle notice.
//
// Without it the acceptance test proves only that two things agree, which a
// pair of stopped clocks also manages. This is the evidence that the
// comparison can fail at all.
func TestMutationIsDetected(t *testing.T) {
	requireCompiler(t)
	u := transpile(t, "VecAdd")
	const n, block = 37, 8
	grid := gpu.D1((n + block - 1) / block)

	mutated := *u
	mutated.Source = strings.Replace(u.Source, "] + ", "] - ", 1)
	if mutated.Source == u.Source {
		t.Fatalf("the mutation did not apply; the generated C no longer contains the operator this test edits:\n%s", u.Source)
	}
	// SourceHash is deliberately left as the unmutated source's. Nothing in
	// hostrun reads it -- an ahead-of-time artifact is the only thing filed
	// under it -- and recomputing it here would suggest this Unit is something
	// that could be built, which it is not.

	a, b := floats(11, n, false), floats(12, n, false)
	want := make([]float32, n)
	gpu.RunCPUDim(grid, gpu.D1(block), func(ctx gpu.Ctx) { kernels.VecAdd(ctx, want, a, b) })

	got := make([]float32, n)
	if err := hostrun.Run(&mutated, grid, gpu.D1(block), got, a, b); err != nil {
		t.Fatalf("hostrun: %v", err)
	}

	diffs := 0
	for i := range got {
		if got[i] != want[i] {
			diffs++
		}
	}
	if diffs == 0 {
		t.Fatal("the mutated kernel agreed with the emulator on every element: the oracle is blind")
	}
	// Every element, because the mutation is on the only statement there is.
	// A weaker assertion would pass if the oracle noticed the ragged tail and
	// nothing else.
	if diffs != n {
		t.Errorf("%d of %d elements disagree; the mutation is on every element's only statement, so all of them should", diffs, n)
	}
}

// TestArgumentsAreChecked pins the transport's half of "refuse, never
// mistranslate".
//
// Every case here is one a fuzzer could generate by accident, and every one of
// them would otherwise be decoded as numbers: bytes are laid out by the
// argument list, so a list that does not match the kernel's parameters
// produces a plausible-looking answer to a question nobody asked. None of
// these reaches a compiler, which is why the test does not need one.
func TestArgumentsAreChecked(t *testing.T) {
	u := transpile(t, "VecAdd")
	f := []float32{0, 1, 2}
	for _, tc := range []struct {
		name string
		args []any
		want string
	}{
		{"too few", []any{f, f}, "takes 3 arguments"},
		{"too many", []any{f, f, f, f}, "takes 3 arguments"},
		{"a scalar for a slice", []any{f, f, float32(1)}, "is a slice, got float32"},
		{"a slice of int", []any{f, f, []int{1}}, "different stride"},
		{"nil", []any{f, f, nil}, "is nil"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := hostrun.Run(u, gpu.D1(1), gpu.D1(8), tc.args...)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Run = %v, want an error containing %q", err, tc.want)
			}
		})
	}
}

// TestOneBinaryServesEveryCase pins the claim the package comment makes about
// the cache: the driver is compiled from the kernel and the shape of its
// parameters, and the data and the geometry arrive on stdin, so a fuzzer
// throwing a thousand cases at one kernel pays for one compile.
//
// It counts files in the cache directory rather than timing anything, because
// a timing threshold on shared CI hardware is a flake waiting to happen. At
// most one new binary for three different lengths -- including an empty one,
// which is where the driver's allocation would otherwise ask calloc for zero
// bytes -- is the whole assertion.
func TestOneBinaryServesEveryCase(t *testing.T) {
	requireCompiler(t)
	u := transpile(t, "VecAdd")
	dir := os.Getenv("GOCUDA_HOSTRUN_CACHE")
	before := cacheEntries(t, dir)

	for _, g := range []struct{ n, block int }{{0, 8}, {1, 64}, {100, 7}} {
		grid := gpu.D1(max((g.n+g.block-1)/g.block, 1))
		a, b := floats(13, g.n, true), floats(14, g.n, true)
		want := make([]float32, g.n)
		gpu.RunCPUDim(grid, gpu.D1(g.block), func(ctx gpu.Ctx) { kernels.VecAdd(ctx, want, a, b) })

		got := make([]float32, g.n)
		if err := hostrun.Run(u, grid, gpu.D1(g.block), got, a, b); err != nil {
			t.Fatalf("n=%d block=%d: %v", g.n, g.block, err)
		}
		if d := compare(t, fmt.Sprintf("c (n=%d)", g.n), got, want, 0); d > 0 {
			t.Errorf("n=%d block=%d: %d elements disagree", g.n, g.block, d)
		}
	}

	if added := cacheEntries(t, dir) - before; added > 1 {
		t.Errorf("three shapes of one kernel produced %d new cached binaries, want at most 1", added)
	}
}

// BenchmarkWarmRun is the per-case cost once the driver is compiled: spawning
// the process, one pass over the grid, and the bytes in each direction. It is
// what a fuzzer pays per case, and it is roughly two orders of magnitude below
// the compile it amortises -- which is the argument for keeping the data out
// of the driver's source.
func BenchmarkWarmRun(b *testing.B) {
	if err := hostrun.Available(); err != nil {
		b.Skipf("%v", err)
	}
	u, err := simt.Transpile(gocuda.Kernels(), "VecAdd")
	if err != nil {
		b.Fatal(err)
	}
	const n, block = 4096, 128
	grid := gpu.D1(n / block)
	c, x, y := make([]float32, n), floats(15, n, false), floats(16, n, false)
	if err := hostrun.Run(u, grid, gpu.D1(block), c, x, y); err != nil {
		b.Fatal(err) // the compile, kept out of the measurement
	}
	b.ResetTimer()
	for b.Loop() {
		if err := hostrun.Run(u, grid, gpu.D1(block), c, x, y); err != nil {
			b.Fatal(err)
		}
	}
}

func cacheEntries(t *testing.T, dir string) int {
	t.Helper()
	ents, err := os.ReadDir(dir)
	if errors.Is(err, os.ErrNotExist) {
		return 0
	}
	if err != nil {
		t.Fatal(err)
	}
	return len(ents)
}

// TestOutOfScopeKernelsAreRefused pins the other half of the contract: the
// four committed kernels a sequential run cannot model are named, and refused
// before anything is compiled.
func TestOutOfScopeKernelsAreRefused(t *testing.T) {
	for _, tc := range []struct{ kernel, construct string }{
		{"FIR", "__shared__"},
		{"Histogram", "__shared__"},
		{"Transpose", "__shared__"},
		{"WarpReduceSum", "__shfl_down_sync"},
	} {
		t.Run(tc.kernel, func(t *testing.T) {
			u := transpile(t, tc.kernel)
			var se *hostrun.ScopeError
			if err := hostrun.CheckScope(u); !errors.As(err, &se) {
				t.Fatalf("CheckScope(%s) = %v, want a *ScopeError", tc.kernel, err)
			}
			if se.Construct != tc.construct {
				t.Errorf("refused over %s, want %s", se.Construct, tc.construct)
			}
			// Run must refuse it too, and refuse it the same way: a caller
			// that skipped CheckScope must not get a number instead.
			err := hostrun.Run(u, gpu.D1(1), gpu.D1(32), []float32{0}, []float32{0}, []float32{0})
			if !errors.As(err, &se) {
				t.Fatalf("Run(%s) = %v, want a *ScopeError", tc.kernel, err)
			}
		})
	}
}

// TestInScopeKernelsAreAccepted is the complement, and it is what would fail
// if a scope rule grew teeth it should not have: the eight kernels the package
// comment claims are in scope must all pass the same check.
func TestInScopeKernelsAreAccepted(t *testing.T) {
	for _, c := range cases() {
		if err := hostrun.CheckScope(transpile(t, c.name)); err != nil {
			t.Errorf("CheckScope(%s) = %v, want nil", c.name, err)
		}
	}
}

// compare counts the elements of two slices that differ, and reports the first
// few.
//
// The bit comparison is stronger than == in both directions that matter here:
// it separates +0 from -0, and it holds two NaNs equal so that a case carrying
// one is not failed by IEEE's rule that a NaN equals nothing. Everything but
// Magnitude can demand it outright, because +, -, * and / are correctly
// rounded on both sides and -ffp-contract=off keeps the host from fusing what
// Go did not.
//
// Where tol is non-zero the elements it rescues are counted and logged rather
// than passed over in silence: a bound that turns out never to be needed is
// worth knowing about, and so is one that is doing all the work.
func compare(t *testing.T, name string, got, want any, tol float64) int {
	t.Helper()
	g, w := reflect.ValueOf(got), reflect.ValueOf(want)
	if g.Len() != w.Len() {
		t.Errorf("%s: got %d elements, want %d", name, g.Len(), w.Len())
		return g.Len() + w.Len()
	}
	diffs, loose := 0, 0
	for i := range g.Len() {
		gi, wi := g.Index(i).Interface(), w.Index(i).Interface()
		switch {
		case same(gi, wi, 0):
		case same(gi, wi, tol):
			loose++
		default:
			diffs++
			if diffs <= 5 {
				t.Errorf("%s[%d] = %v, want %v", name, i, gi, wi)
			}
		}
	}
	if loose > 0 {
		t.Logf("%s: %d of %d elements agree only within %g, not bit for bit", name, loose, g.Len(), tol)
	}
	return diffs
}

// same decides one element, which internal/tolerance.Agree is the rule for.
//
// It is one line because the rule belongs in one place: bit equality first,
// then the values no tolerance relates -- a NaN, an infinity against a finite
// number, the two zeros -- and only a pair of ordinary finite numbers judged
// against the bound. This file used to state all of that itself, which is
// exactly how the repository came to have three comparison rules before.
func same(got, want any, tol float64) bool { return tolerance.Agree(got, want, tol) }
