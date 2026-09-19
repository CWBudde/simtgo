// Package faulttest holds the one test that deliberately faults the device,
// and exists as a package of its own because that test cannot share a process
// with any other.
//
// A __trap() raises a sticky CUDA error, and what that costs was measured
// here rather than assumed. Every later call in the context returns
// CUDA_ERROR_LAUNCH_FAILED -- Sync, cuMemAlloc, cuMemcpyDtoH, and the
// cuModuleUnload inside Close -- and, worse than the documentation implies,
// so does cuDevicePrimaryCtxRetain: a *replacement* context cannot be made
// either. The process is finished with CUDA. `go test` gives each package its
// own binary, which is the isolation that makes this testable at all, and it
// is why there is exactly one trapping test below rather than several.
//
// Two things must stay true of this package.
//
// It must never be added to the compute-sanitizer sweep in the justfile or in
// CLAUDE.md. A trap is a real fault and the sanitizer reports it as one, so
// --error-exitcode 1 would turn a passing test into a failed sweep.
//
// And it must not be tidied back into simt. Nothing about it is
// simt-specific; what it needs is a process nobody else is using.
package faulttest
