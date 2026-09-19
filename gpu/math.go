package gpu

import "math"

// Float32 math for kernels. Go's math package is float64-only, and sm_75
// runs float64 at 1/32 rate, so kernels use these instead. The transpiler
// maps each one to the corresponding CUDA single-precision built-in
// (sqrtf, fabsf, hypotf, ...).

// Sqrt returns the square root of x (sqrtf).
func Sqrt(x float32) float32 { return float32(math.Sqrt(float64(x))) }

// Abs returns the absolute value of x (fabsf).
func Abs(x float32) float32 { return float32(math.Abs(float64(x))) }

// Hypot returns sqrt(x*x + y*y) without overflow (hypotf).
func Hypot(x, y float32) float32 { return float32(math.Hypot(float64(x), float64(y))) }

// Sin returns the sine of x (sinf).
func Sin(x float32) float32 { return float32(math.Sin(float64(x))) }

// Cos returns the cosine of x (cosf).
func Cos(x float32) float32 { return float32(math.Cos(float64(x))) }

// Exp returns e**x (expf).
func Exp(x float32) float32 { return float32(math.Exp(float64(x))) }

// Log returns the natural logarithm of x (logf).
func Log(x float32) float32 { return float32(math.Log(float64(x))) }

// Fmin returns the smaller of x and y (fminf).
//
// It is not math.Min, and the difference is the whole point of writing it out.
// Go's math.Min propagates NaN -- its own documentation says Min(x, NaN) = NaN
// -- while C's fminf follows IEEE 754 minNum and returns the operand that is
// not NaN. Wrapping math.Min would therefore have made the emulator and the
// device disagree about a value no tolerance can reconcile: NaN against a
// number. A parity test exists to catch exactly that, so it cannot be built
// out of a helper that has it.
func Fmin(x, y float32) float32 {
	switch {
	case isNaN(x):
		return y
	case isNaN(y):
		return x
	case x == y:
		// Signed zeros compare equal, and fminf is specified to prefer the
		// negative one. Testing the sign bit is the only way to tell them
		// apart once == has said they are the same.
		if signbit(x) {
			return x
		}
		return y
	case x < y:
		return x
	}
	return y
}

// Fmax returns the larger of x and y (fmaxf).
//
// The mirror of Fmin, and not math.Max, for the reason given there.
func Fmax(x, y float32) float32 {
	switch {
	case isNaN(x):
		return y
	case isNaN(y):
		return x
	case x == y:
		if signbit(x) {
			return y
		}
		return x
	case x > y:
		return x
	}
	return y
}

// isNaN and signbit are spelled here rather than through math so that they
// take a float32 and stay exact: converting to float64 to ask preserves both
// answers, but the conversion is noise in a file whose subject is that
// converting to float64 is precisely what changes the result.
func isNaN(x float32) bool { return x != x }

func signbit(x float32) bool { return math.Signbit(float64(x)) }

// Float64 math, legal only in a kernel carrying //simtgo:float64. Each maps to
// the unsuffixed CUDA built-in, which is the double one -- sqrt rather than
// sqrtf -- and NVRTC provides those without a header too.
//
// These are separate functions rather than overloads because Go has none, and
// the suffix is the same one ctx.SharedF32 already uses for "this is the
// float32 variant". Reaching one of them without the directive is refused with
// the directive's name, rather than lowering to a double the author did not
// ask to pay for.

// Sqrt64 returns the square root of x (sqrt).
func Sqrt64(x float64) float64 { return math.Sqrt(x) }

// Abs64 returns the absolute value of x (fabs).
func Abs64(x float64) float64 { return math.Abs(x) }

// Hypot64 returns sqrt(x*x + y*y) without overflow (hypot).
func Hypot64(x, y float64) float64 { return math.Hypot(x, y) }

// Sin64 returns the sine of x (sin).
func Sin64(x float64) float64 { return math.Sin(x) }

// Cos64 returns the cosine of x (cos).
func Cos64(x float64) float64 { return math.Cos(x) }

// Exp64 returns e**x (exp).
func Exp64(x float64) float64 { return math.Exp(x) }

// Log64 returns the natural logarithm of x (log).
func Log64(x float64) float64 { return math.Log(x) }

// Fmin64 returns the smaller of x and y (fmin).
//
// The NaN cases are taken out before math.Min sees them, for the reason Fmin
// gives: math.Min propagates a NaN and fmin returns the operand that is not
// one. Everything else is math.Min's, which unlike the float32 case exists and
// documents the remaining special values -- -Inf wins, and Min(-0, +0) is -0,
// which is what fmin does with them too.
func Fmin64(x, y float64) float64 {
	switch {
	case math.IsNaN(x):
		return y
	case math.IsNaN(y):
		return x
	}
	return math.Min(x, y)
}

// Fmax64 returns the larger of x and y (fmax). The mirror of Fmin64.
func Fmax64(x, y float64) float64 {
	switch {
	case math.IsNaN(x):
		return y
	case math.IsNaN(y):
		return x
	}
	return math.Max(x, y)
}
