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
