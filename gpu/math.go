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
func Fmin(x, y float32) float32 { return float32(math.Min(float64(x), float64(y))) }

// Fmax returns the larger of x and y (fmaxf).
func Fmax(x, y float32) float32 { return float32(math.Max(float64(x), float64(y))) }

// Float64 math, legal only in a kernel carrying //gocuda:float64. Each maps to
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
func Fmin64(x, y float64) float64 { return math.Min(x, y) }

// Fmax64 returns the larger of x and y (fmax).
func Fmax64(x, y float64) float64 { return math.Max(x, y) }
