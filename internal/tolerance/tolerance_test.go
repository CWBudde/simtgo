package tolerance_test

import (
	"math"
	"testing"

	"github.com/CWBudde/gocuda/internal/tolerance"
)

// TestAgreeOnTheValuesNoToleranceRelates pins the order of Agree's questions,
// which is the whole of what it is for.
//
// Every case is stated for both widths, because handling only float32 is
// exactly the bug this test exists for: a []float64 of NaNs fell through to
// reflect.DeepEqual, which compares them with ==, and the fuzzer reported a
// mismatch reading "NaN, want NaN". A rule with a hole in one width is not one
// rule.
func TestAgreeOnTheValuesNoToleranceRelates(t *testing.T) {
	const tol = 1e-6
	cases := []struct {
		name     string
		f32      func() (float32, float32)
		f64      func() (float64, float64)
		want     bool
		wantZero bool // what the same pair gives with tol == 0
	}{{
		name: "a NaN agrees with a NaN, which == does not",
		f32:  func() (float32, float32) { return f32NaN(), f32NaN() },
		f64:  func() (float64, float64) { return math.NaN(), math.NaN() },
		want: true, wantZero: true,
	}, {
		name: "a NaN agrees with nothing else, however loose the bound",
		f32:  func() (float32, float32) { return f32NaN(), 1 },
		f64:  func() (float64, float64) { return math.NaN(), 1 },
		want: false, wantZero: false,
	}, {
		name: "the two zeros are different results, not a rounding",
		f32:  func() (float32, float32) { return 0, float32(math.Copysign(0, -1)) },
		f64:  func() (float64, float64) { return 0, math.Copysign(0, -1) },
		want: false, wantZero: false,
	}, {
		name: "a zero agrees with itself, sign included",
		f32:  func() (float32, float32) { return float32(math.Copysign(0, -1)), float32(math.Copysign(0, -1)) },
		f64:  func() (float64, float64) { return math.Copysign(0, -1), math.Copysign(0, -1) },
		want: true, wantZero: true,
	}, {
		name: "an infinity is not a large finite number",
		f32:  func() (float32, float32) { return float32(math.Inf(1)), math.MaxFloat32 },
		f64:  func() (float64, float64) { return math.Inf(1), math.MaxFloat64 },
		want: false, wantZero: false,
	}, {
		name: "the same infinity agrees",
		f32:  func() (float32, float32) { return float32(math.Inf(-1)), float32(math.Inf(-1)) },
		f64:  func() (float64, float64) { return math.Inf(-1), math.Inf(-1) },
		want: true, wantZero: true,
	}, {
		// The bound is a ceiling, not a floor: a pair that is bit-identical
		// never reaches it, which is why tol == 0 is the ordinary case.
		name: "two ordinary numbers inside the bound",
		f32:  func() (float32, float32) { return 1.0000001, 1 },
		f64:  func() (float64, float64) { return 1.0000001, 1 },
		want: true, wantZero: false,
	}, {
		name: "two ordinary numbers outside it",
		f32:  func() (float32, float32) { return 1.1, 1 },
		f64:  func() (float64, float64) { return 1.1, 1 },
		want: false, wantZero: false,
	}}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g32, w32 := tc.f32()
			g64, w64 := tc.f64()
			for _, c := range []struct {
				width    string
				got, wnt any
			}{{"float32", g32, w32}, {"float64", g64, w64}} {
				if got := tolerance.Agree(c.got, c.wnt, tol); got != tc.want {
					t.Errorf("%s: Agree(%v, %v, %g) = %v, want %v", c.width, c.got, c.wnt, tol, got, tc.want)
				}
				if got := tolerance.Agree(c.got, c.wnt, 0); got != tc.wantZero {
					t.Errorf("%s: Agree(%v, %v, 0) = %v, want %v", c.width, c.got, c.wnt, got, tc.wantZero)
				}
			}
		})
	}
}

// TestAgreeComparesAnythingElseExactly is the other half: a rounding is a
// float's business, and an integer or a bool that disagrees is a defect.
func TestAgreeComparesAnythingElseExactly(t *testing.T) {
	for _, tc := range []struct {
		got, want any
		agree     bool
	}{
		{int32(3), int32(3), true},
		{int32(3), int32(4), false},
		{uint64(1 << 40), uint64(1 << 40), true},
		{true, true, true},
		{true, false, false},
		{int64(-1), int64(-1), true},
		// Two widths are two different answers, never the same one.
		{float32(1), float64(1), false},
		{int32(1), int64(1), false},
	} {
		if got := tolerance.Agree(tc.got, tc.want, 0.5); got != tc.agree {
			t.Errorf("Agree(%#v, %#v, 0.5) = %v, want %v", tc.got, tc.want, got, tc.agree)
		}
	}
}

// f32NaN is a NaN that is genuinely a float32 one. math.NaN() converted is
// still a NaN, but going through the bits says so without depending on that.
func f32NaN() float32 { return math.Float32frombits(0x7fc00001) }
