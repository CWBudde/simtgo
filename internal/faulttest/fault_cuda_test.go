//go:build cuda

package faulttest_test

import (
	"errors"
	"os"
	"testing"
	"testing/fstest"

	"github.com/CWBudde/simtgo/cuda"
	"github.com/CWBudde/simtgo/simt"
)

// probe indexes a slice with the thread's own id and nothing else, so a launch
// wider than the buffer puts every thread above it out of range.
const probe = `package kernels

import "github.com/CWBudde/simtgo/gpu"

func Overrun(ctx gpu.Ctx, y []float32) {
	y[ctx.GlobalID()] = 1
}
`

// device is simt's helper, duplicated rather than shared because this package
// cannot import a test file. It also requires NVRTC, which the parity tests do
// not: a bounds-checked build never matches a prebuilt, so there is always a
// compilation.
func device(t *testing.T) *cuda.Context {
	t.Helper()
	if !cuda.Available() {
		if os.Getenv("SIMTGO_REQUIRE_DEVICE") != "" {
			t.Fatal("SIMTGO_REQUIRE_DEVICE is set, but no CUDA device is available")
		}
		t.Skip("no CUDA device available")
	}
	ctx, err := cuda.NewContext(0)
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	// Deliberately no t.Cleanup(ctx.Close): after the trap, Close fails too,
	// and the process is about to end anyway.
	return ctx
}

// TestTrapPoisonsTheContextAndSaysSo is the whole of the trap half of debug
// mode, in one test because the first trap ends this process's use of CUDA.
//
// It asserts three things in sequence, and the middle one is the criterion the
// roadmap set: the context is unusable afterwards, so say so in the error
// rather than letting the next call fail obscurely.
func TestTrapPoisonsTheContextAndSaysSo(t *testing.T) {
	ctx := device(t)

	// A second wrapper around the same device, made before anything faults.
	// cuDevicePrimaryCtxRetain hands both of these the same CUcontext, so the
	// trap below happens to this one too -- and it has to say so. It used to
	// keep returning bare 719s, because the marker lived on whichever *Context
	// the faulting launch went through.
	other, err := cuda.NewContext(0)
	if err != nil {
		t.Fatalf("a second NewContext on the same device: %v", err)
	}

	src := fstest.MapFS{"k.go": &fstest.MapFile{Data: []byte(probe)}}
	k, err := simt.Build(ctx, src, "Overrun", simt.WithBoundsChecks())
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if !k.BoundsChecks {
		t.Fatal("the kernel does not know it was built with checks, so a fault would not be reported as a trap")
	}
	if k.Prebuilt {
		t.Fatal("a bounds-checked build loaded a prebuilt, so the checks are not in the PTX that ran")
	}

	dy, err := cuda.NewSlice[float32](ctx, 4)
	if err != nil {
		t.Fatalf("NewSlice: %v", err)
	}

	// 64 threads over a 4-element buffer: threads 4 and up are out of range.
	err = k.LaunchN(64, 64, dy)
	if err == nil {
		t.Fatal("an out-of-range index did not fault, so the checks are not doing anything")
	}

	var trap *simt.TrapError
	if !errors.As(err, &trap) {
		t.Fatalf("a bounds-checked launch faulted but was not reported as a trap: %v", err)
	}
	if trap.Kernel != "Overrun" {
		t.Errorf("TrapError.Kernel = %q, want %q", trap.Kernel, "Overrun")
	}
	// The driver's own code still has to be reachable: a caller matching on
	// CUDA_ERROR_LAUNCH_FAILED must not have to know about TrapError.
	if !errors.Is(err, cuda.ErrLaunchFailed) {
		t.Errorf("the trap does not unwrap to the driver code: %v", err)
	}
	t.Logf("the launch reported: %v", err)

	// The criterion. Every later call must say the context died and why,
	// instead of repeating the driver's code with no explanation.
	for _, c := range []struct {
		what string
		call func() error
	}{
		{"Sync", ctx.Sync},
		{"Download", func() error { _, e := dy.Download(); return e }},
		{"Alloc", func() error { _, e := cuda.NewSlice[float32](ctx, 4); return e }},
	} {
		err := c.call()
		var poisoned *cuda.ContextPoisonedError
		if !errors.As(err, &poisoned) {
			t.Errorf("%s after a trap returned %v, not a *cuda.ContextPoisonedError", c.what, err)
			continue
		}
		if !errors.Is(err, cuda.ErrLaunchFailed) {
			t.Errorf("%s after a trap does not unwrap to the driver code: %v", c.what, err)
		}
	}
	t.Logf("a later call reports: %v", ctx.Sync())

	// The other wrapper, which never launched anything, reports the same
	// fault -- because the fault belongs to the primary context, not to the
	// Go value in front of it.
	var second *cuda.ContextPoisonedError
	if err := other.Sync(); !errors.As(err, &second) {
		t.Errorf("Sync on a second wrapper of the same device returned %v, not a *cuda.ContextPoisonedError", err)
	}
	if _, err := cuda.NewSlice[float32](other, 4); !errors.As(err, &second) {
		t.Errorf("Alloc on a second wrapper of the same device returned %v, not a *cuda.ContextPoisonedError", err)
	}

	// Close is deliberately not in that list. It holds the context's own
	// mutex and never calls bind, because its job is to unload what it owns
	// and let go -- so it reports the driver's error as it always has. That
	// is the right answer for a teardown: the caller is already leaving.
	if err := ctx.Close(); !errors.Is(err, cuda.ErrLaunchFailed) {
		t.Errorf("Close after a trap = %v, want the driver's own code", err)
	}

	// The second wrapper closes too, and that is load-bearing rather than
	// tidy: the measurement below is about a primary context whose reference
	// count has reached zero. With one reference still outstanding
	// cuDevicePrimaryCtxRetain hands the same poisoned context straight back
	// and succeeds, which measures nothing about whether a replacement can be
	// made. The result is discarded deliberately -- it is the sticky code
	// ctx.Close just returned, and it was asserted there.
	_ = other.Close()

	// And the measurement the documentation rests on: a replacement context
	// cannot be made either, which is stronger than "the context is unusable"
	// and is why the error says the process has to exit. NewContext reports
	// the driver's own failure here rather than the poison marker, precisely
	// so this stays a measurement of cuDevicePrimaryCtxRetain.
	fresh, err := cuda.NewContext(0)
	if err == nil {
		t.Error("a fresh context was retained after a trap; docs/toolchain.md says it cannot be, and that claim is now wrong")
	} else {
		if fresh != nil {
			t.Error("NewContext returned both a context and an error after a trap")
		}
		if !errors.Is(err, cuda.ErrLaunchFailed) {
			t.Errorf("a fresh context after a trap failed with %v, which is not the sticky code", err)
		}
		t.Logf("a fresh context after a trap: %v", err)
	}
}
