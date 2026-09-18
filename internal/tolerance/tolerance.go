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
	return Close64(float64(got), float64(want), tol)
}

// Close64 is Close for a pair of float64 values, and the rule itself: relative
// error with the scale floored at 1, so below 1 it is an absolute bound.
//
// Close widens to this rather than restating it, which is also why the
// widening Close's own documentation describes is not visible here -- by the
// time a float32 arrives it has already happened.
func Close64(got, want, tol float64) bool {
	d := math.Abs(got - want)
	return d <= tol*max(math.Abs(want), 1)
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
// Both float widths go through the same questions. Handling only float32 was
// the first thing the fuzzer found, and it found it in this file rather than
// in the transpiler: two NaNs in a []float64 fell through to DeepEqual, which
// compares them with ==, and were reported as a mismatch reading "NaN, want
// NaN". Anything that is not a float is compared exactly, because an integer
// or a bool that disagrees is a bug rather than a rounding -- the same line
// AssertEqual draws.
func Agree(got, want any, tol float64) bool {
	switch g := got.(type) {
	case float32:
		w, ok := want.(float32)
		return ok && agree64(float64(g), float64(w), math.Float32bits(g) == math.Float32bits(w), tol)
	case float64:
		w, ok := want.(float64)
		return ok && agree64(g, w, math.Float64bits(g) == math.Float64bits(w), tol)
	}
	return reflect.DeepEqual(got, want)
}

// agree64 is Agree's body, once, for both widths.
//
// It takes the bit comparison rather than making it, because that is the one
// step that differs between the two: a float32 has to be compared as 32 bits
// and not as the float64 it widens to, or -0 and +0 would already have been
// called equal by the time the zeros are asked about.
//
// The order is the whole of it. Bit equality first, so an ordinary element is
// judged by that and a bound is never consulted for a result that is simply
// right. Then the values no tolerance relates to anything: a NaN, which equals
// nothing including another NaN, so both sides have to be one; an infinity
// against a finite number; and the two zeros, which a relative bound cannot
// tell apart because their difference is zero while their bits are not. Only a
// pair of ordinary finite numbers reaches the bound.
func agree64(g, w float64, sameBits bool, tol float64) bool {
	switch {
	case sameBits:
		return true
	case math.IsNaN(g) || math.IsNaN(w):
		return math.IsNaN(g) && math.IsNaN(w)
	case math.IsInf(g, 0) || math.IsInf(w, 0), g == 0 && w == 0:
		return false
	case tol > 0:
		return Close64(g, w, tol)
	}
	return false
}
