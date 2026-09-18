// Package tolerance is the one comparison the tests in this repository hold a
// float32 result to, and its counterpart for results that have no tolerance to
// speak of.
//
// It is a package rather than a helper in each test file because there is no
// other way for three of them to share one rule: the parity tests are
// package simt_test, the prebuilt round trip is inside package simt, and the
// tile pipeline test is package tile_test. Spelled three times, the rule had
// already drifted into three -- two relative comparisons with different
// signatures and one bare absolute one that followed neither.
//
// NUMERICS.md states the rule, what it does not bound, and when a test may ask
// for exact equality instead. The short version of both halves:
//
//   - Close is relative error with the scale floored at 1, so below 1 it is an
//     absolute bound. It is not in units in the last place, and the floor makes
//     it very loose for small magnitudes.
//   - Equal is for integers and bools always, and for floats only where every
//     value is exactly representable and the result does not depend on the
//     order the threads arrived in.
package tolerance

import (
	"math"
	"reflect"
	"testing"
)

// Close reports whether got is within tol of want.
//
// The difference is taken in float64. Subtracting two float32 values in
// float32 is exact when they are close and rounds when they are not, which is
// a rounding inside the instrument rather than inside what is being measured;
// widening first removes it. The tolerance itself is a float64 for the same
// reason -- 1e-6 is not a float32 and writing it as one rounds the bound
// before it is used.
//
// A NaN on either side always fails, deliberately. No tolerance relates a NaN
// to anything, including to another NaN, so a test that expects one asks
// math.IsNaN instead; simt's Fmin/Fmax parity test is the example.
func Close(got, want float32, tol float64) bool {
	d := math.Abs(float64(got) - float64(want))
	return d <= tol*max(math.Abs(float64(want)), 1)
}

// AssertClose fails t at the first element of got that is not Close to want.
//
// The name says which two things are being compared. That is not decoration:
// the same helper is used for device against emulator and for emulator against
// an independent reference, and a message that named one of those for both
// would misreport half its failures.
func AssertClose(t testing.TB, name string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d, want %d", name, len(got), len(want))
	}
	for i := range got {
		if !Close(got[i], want[i], tol) {
			d := math.Abs(float64(got[i]) - float64(want[i]))
			t.Fatalf("%s element %d: got %v, want %v (|delta| %g, bound %g)",
				name, i, got[i], want[i], d, tol*max(math.Abs(float64(want[i])), 1))
		}
	}
}

// AssertEqual is AssertClose's counterpart for results that must agree
// exactly. An integer or a bool that disagrees is a bug, not a rounding, and
// so is a float32 the policy says should have come back bit for bit.
func AssertEqual[T comparable](t testing.TB, name string, got, want []T) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d, want %d", name, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("%s element %d: got %v, want %v", name, i, got[i], want[i])
		}
	}
}

// Agree is Close's front door for a value whose type is not known until run
// time: one element of a generated kernel's output, where the element type is
// whatever the generator chose.
//
// The order of the questions is the whole of it. Bit equality comes first, so
// that is what an ordinary element is judged by and a bound is never consulted
// for a result that is simply right. Then the values no tolerance relates to
// anything: a NaN, which equals nothing including another NaN, so both sides
// have to be one; an infinity against a finite number, where the relative
// error is not a number either; and the two zeros, which a relative bound
// cannot tell apart, because their difference is zero while their bits are
// not. Only a pair of ordinary finite numbers reaches Close, which is all that
// rule ever claimed to be about.
//
// Anything that is not a float32 is compared exactly. An integer or a bool
// that disagrees is a bug rather than a rounding, which is the same line
// AssertEqual draws.
func Agree(got, want any, tol float64) bool {
	g, ok := got.(float32)
	if !ok {
		return reflect.DeepEqual(got, want)
	}
	w, ok := want.(float32)
	if !ok {
		return false // two different types cannot agree about anything
	}
	gd, wd := float64(g), float64(w)
	switch {
	case math.Float32bits(g) == math.Float32bits(w):
		return true
	case math.IsNaN(gd) || math.IsNaN(wd):
		return math.IsNaN(gd) && math.IsNaN(wd)
	case math.IsInf(gd, 0) || math.IsInf(wd, 0), g == 0 && w == 0:
		return false
	case tol > 0:
		return Close(g, w, tol)
	}
	return false
}
