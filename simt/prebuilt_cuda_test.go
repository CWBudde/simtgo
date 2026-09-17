//go:build cuda

package simt

import (
	"bytes"
	"math"
	"os"
	"testing"

	"github.com/CWBudde/gocuda/cuda"
	"github.com/CWBudde/gocuda/kernels"
)

// The kernel sources are read from disk rather than through kernelSources,
// which embeds the same files: the root package links the generated prebuilt
// package, that package imports this one, and a test inside package simt
// cannot close that loop.
var kernelSources = os.DirFS("../kernels")

// These tests are internal to the package so they can empty the registry for
// their own duration: the artifacts below are built for one device at test
// time, and leaving them registered would quietly change which path every
// later test in this binary takes.

// freshDevice returns a context nothing has been loaded into yet.
//
// A context caches the modules loaded through it, keyed on the source and the
// architecture, and nothing else -- deliberately, since a loaded module is a
// loaded module however its PTX was obtained. So a second Build of the same
// kernel in the same context reports the first one's origin, whatever it is
// asked for, and every test here that means to observe which path ran needs a
// context of its own.
func freshDevice(t testing.TB) *cuda.Context {
	t.Helper()
	if !cuda.Available() {
		t.Skip("no CUDA device available")
	}
	ctx, err := cuda.NewContext(0)
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	t.Cleanup(func() { ctx.Close() })
	return ctx
}

// compileFIR produces the PTX a generator would have written out, by the same
// route: transpile the kernel, hand the CUDA C to NVRTC.
func compileFIR(t testing.TB, arch string) (*Unit, *cuda.PTX) {
	t.Helper()
	u, err := Transpile(kernelSources, "FIR")
	if err != nil {
		t.Fatalf("Transpile: %v", err)
	}
	ptx, err := cuda.Compile(u.Source, "FIR.cu", arch)
	if err != nil {
		t.Fatalf("Compile for %s: %v", arch, err)
	}
	return u, ptx
}

func registerFIR(t testing.TB, u *Unit, ptx *cuda.PTX, arch string) {
	t.Helper()
	RegisterPrebuilt(Prebuilt{
		Name:          "FIR",
		SourceSHA256:  u.SourceHash,
		Arch:          arch,
		PTX:           ptx.Bytes,
		RequiredBlock: u.RequiredBlock,
		SharedBytes:   u.SharedBytes,
		Log:           ptx.Log,
	})
}

// runFIR launches the kernel and checks the result against a direct
// convolution, so that a kernel loaded from a prebuilt image is held to the
// same standard as one just compiled rather than merely to "it launched".
func runFIR(t *testing.T, ctx *cuda.Context, k *Kernel) {
	t.Helper()
	const n = 4096
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(math.Sin(float64(i) * 0.05))
	}
	h := make([]float32, 17)
	for i := range h {
		h[i] = float32(1) / float32(len(h))
	}

	dy, err := cuda.NewSlice[float32](ctx, n)
	if err != nil {
		t.Fatalf("NewSlice: %v", err)
	}
	defer dy.Free()
	dx, err := cuda.Upload(ctx, x)
	if err != nil {
		t.Fatalf("Upload: %v", err)
	}
	defer dx.Free()
	dh, err := cuda.Upload(ctx, h)
	if err != nil {
		t.Fatalf("Upload: %v", err)
	}
	defer dh.Free()

	if err := k.LaunchN(n, kernels.FIRBlock, dy, dx, dh); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	got, err := dy.Download()
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	for i := range got {
		var want float32
		for j := range h {
			if i-j >= 0 {
				want += h[j] * x[i-j]
			}
		}
		if math.Abs(float64(got[i]-want)) > 1e-5 {
			t.Fatalf("y[%d] = %v, want %v", i, got[i], want)
		}
	}
}

// TestBuildPrebuilt is the whole point of the ahead-of-time path: the kernel
// that runs is the registered image, NVRTC was never asked, and the numbers
// are the same either way.
func TestBuildPrebuilt(t *testing.T) {
	swapRegistry(t)
	arch := freshDevice(t).Arch()
	u, ptx := compileFIR(t, arch)
	registerFIR(t, u, ptx, arch)

	t.Run("prebuilt", func(t *testing.T) {
		dev := freshDevice(t)
		k, err := Build(dev, kernelSources, "FIR", WithCacheDir(t.TempDir()))
		if err != nil {
			t.Fatalf("Build: %v", err)
		}
		if !k.Prebuilt {
			t.Error("Build compiled with NVRTC although the current source is registered")
		}
		if k.Arch != arch {
			t.Errorf("Arch = %q, want %q: the arch the loaded PTX was built for", k.Arch, arch)
		}
		if !bytes.Equal(k.PTX, ptx.Bytes) {
			t.Errorf("PTX = %d bytes, want the %d registered", len(k.PTX), len(ptx.Bytes))
		}
		// The source is transpiled on this path too, so everything derived
		// from it is present and is derived from what is on disk now.
		if k.Source != u.Source {
			t.Error("Source is not the freshly generated CUDA C")
		}
		if k.RequiredBlock != kernels.FIRBlock {
			t.Errorf("RequiredBlock = %d, want %d", k.RequiredBlock, kernels.FIRBlock)
		}
		if err := k.Launch(1, kernels.FIRBlock+1); err == nil {
			t.Error("a launch at the wrong block size was accepted")
		}
		runFIR(t, dev, k)
	})

	// The negative control. Without it the assertion above only says that
	// Prebuilt can be true, not that it tracks anything.
	t.Run("without prebuilt", func(t *testing.T) {
		dev := freshDevice(t)
		k, err := Build(dev, kernelSources, "FIR", WithCacheDir(t.TempDir()), WithoutPrebuilt())
		if err != nil {
			t.Fatalf("Build: %v", err)
		}
		if k.Prebuilt {
			t.Error("WithoutPrebuilt still served the registered image")
		}
		if k.Arch != dev.Arch() {
			t.Errorf("Arch = %q, want the device's %q", k.Arch, dev.Arch())
		}
		if len(k.PTX) == 0 {
			t.Error("Build reported no PTX on the NVRTC path")
		}
		runFIR(t, dev, k)
	})
}

// TestBuildPrebuiltOlderArch exercises the forward compatibility the choice of
// image rests on, against the driver rather than against an assumption: an
// image built for an older baseline than the device is loaded and JITted, and
// the kernel computes the same thing.
func TestBuildPrebuiltOlderArch(t *testing.T) {
	swapRegistry(t)
	dev := freshDevice(t)
	major, minor := dev.ComputeCapability()
	if major < 7 || (major == 7 && minor == 0) {
		t.Skipf("device is compute %d.%d; nothing older is worth targeting", major, minor)
	}
	const older = "compute_70"
	u, ptx := compileFIR(t, older)
	registerFIR(t, u, ptx, older)

	// A context of its own, since compileFIR has already loaded nothing into
	// dev but the test above shows how easily that stops being true.
	target := freshDevice(t)
	k, err := Build(target, kernelSources, "FIR", WithCacheDir(t.TempDir()))
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if !k.Prebuilt {
		t.Fatalf("Build ignored a %s image on a compute %d.%d device", older, major, minor)
	}
	if k.Arch != older {
		t.Errorf("Arch = %q, want %q", k.Arch, older)
	}
	runFIR(t, target, k)
}

// TestBuildPrebuiltNewerArch is the other side of it: an image for hardware
// this device is not is never handed to the driver, since the driver could
// only refuse it.
func TestBuildPrebuiltNewerArch(t *testing.T) {
	swapRegistry(t)
	dev := freshDevice(t)
	u, err := Transpile(kernelSources, "FIR")
	if err != nil {
		t.Fatalf("Transpile: %v", err)
	}
	// Bytes no driver would accept, so that taking this path at all fails
	// loudly rather than by being slower than it should be.
	RegisterPrebuilt(Prebuilt{Name: "FIR", SourceSHA256: u.SourceHash, Arch: "compute_99", PTX: []byte("not ptx")})

	k, err := Build(dev, kernelSources, "FIR", WithCacheDir(t.TempDir()))
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if k.Prebuilt {
		t.Error("an image built for a newer architecture was loaded")
	}
}

// BenchmarkBuild measures what the ahead-of-time path actually buys.
//
// Every iteration builds in a context of its own, because the module cache is
// per context: reusing one would make every iteration after the first a map
// lookup, and both numbers would converge on the cost of nothing happening.
// The price is that the context setup is in both numbers, which is the same
// constant on both sides.
func BenchmarkBuild(b *testing.B) {
	if !cuda.Available() {
		b.Skip("no CUDA device available")
	}
	swapRegistry(b)
	// One context is held open for the whole benchmark. Releasing the last
	// reference to a device's primary context destroys it, and retaining it
	// again costs upwards of 150 ms -- a hundred times what is being measured
	// here. Keeping one alive turns the per-iteration NewContext into a
	// reference count, while each iteration still gets a module cache of its
	// own, which is the only thing a fresh context is wanted for.
	keepAlive, err := cuda.NewContext(0)
	if err != nil {
		b.Fatalf("NewContext: %v", err)
	}
	defer keepAlive.Close()
	arch := keepAlive.Arch()
	u, ptx := compileFIR(b, arch)

	run := func(b *testing.B, opts ...BuildOption) {
		b.Helper()
		// The dump is disabled on both sides: writing two files per iteration
		// would time the file system rather than the compiler.
		opts = append(opts, WithCacheDir(""))
		for b.Loop() {
			dev, err := cuda.NewContext(0)
			if err != nil {
				b.Fatalf("NewContext: %v", err)
			}
			if _, err := Build(dev, kernelSources, "FIR", opts...); err != nil {
				dev.Close()
				b.Fatalf("Build: %v", err)
			}
			dev.Close()
		}
	}

	// What the two numbers below have in common, so that the difference
	// between them can be read as the difference between the two paths rather
	// than taken on trust.
	b.Run("context only", func(b *testing.B) {
		for b.Loop() {
			dev, err := cuda.NewContext(0)
			if err != nil {
				b.Fatalf("NewContext: %v", err)
			}
			dev.Close()
		}
	})
	b.Run("nvrtc", func(b *testing.B) { run(b, WithoutPrebuilt()) })
	b.Run("prebuilt", func(b *testing.B) {
		registerFIR(b, u, ptx, arch)
		run(b)
	})
}

// TestWithoutPrebuiltOnAWarmContext pins the negative control.
//
// Build the prebuilt path first and the NVRTC path second, in one context, and
// the second call used to be answered from the module cache with the first
// call's module -- because the cache key held only the source and the device's
// architecture, not where the PTX came from. WithoutPrebuilt then silently did
// nothing, which is the one thing an option whose whole purpose is to be a
// control must never do.
func TestWithoutPrebuiltOnAWarmContext(t *testing.T) {
	swapRegistry(t)
	dev := freshDevice(t)
	arch := dev.Arch()
	u, ptx := compileFIR(t, arch)
	registerFIR(t, u, ptx, arch)

	warm, err := Build(dev, kernelSources, "FIR", WithCacheDir(""))
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if !warm.Prebuilt {
		t.Fatal("setup: the first build did not take the prebuilt path")
	}

	cold, err := Build(dev, kernelSources, "FIR", WithoutPrebuilt(), WithCacheDir(""))
	if err != nil {
		t.Fatalf("Build with WithoutPrebuilt: %v", err)
	}
	if cold.Prebuilt {
		t.Error("WithoutPrebuilt returned the prebuilt module that was already loaded")
	}
	if cold.Arch != arch {
		t.Errorf("Arch = %q, want the device's own %q on the NVRTC path", cold.Arch, arch)
	}
	// Both are real, launchable modules; the point is only where they came from.
	if cold.RequiredBlock != warm.RequiredBlock || cold.Source != warm.Source {
		t.Error("the two paths disagree about the kernel itself, not just its origin")
	}
}
