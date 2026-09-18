// Package prebuilt holds the ahead-of-time artifacts for the kernels beside it.
//
// It is deliberately not part of the kernel package: a kernel package may
// import nothing but gocuda/gpu, and everything here imports simt.
package prebuilt

// Lowered names a kernel that "gocuda generate" was able to lower to CUDA C.
type Lowered string

// Gate lists every kernel that must lower for this package to compile.
//
// Each name is a constant in prebuilt_gen.go, which "gocuda generate"
// writes only for the kernels it could lower. A kernel it had to drop stops
// resolving here, and the build then fails with "undefined" -- which is how a
// kernel that cannot be lowered fails "go build" instead of main().
//
// Add a name here whenever you add a kernel.
var Gate = []Lowered{
	BandGain,
	Classify,
	FIR,
	Histogram,
	VecAdd,
	Magnitude,
	Quantize,
	Scale,
	Softclip,
	Transpose,
}

// Names returns Gate as plain strings, for simt.VerifyPrebuilt.
func Names() []string {
	out := make([]string, len(Gate))
	for i, n := range Gate {
		out[i] = string(n)
	}
	return out
}
