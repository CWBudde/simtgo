//go:build cuda

package cuda_test

import (
	"errors"
	"testing"

	"github.com/CWBudde/gocuda/cuda"
)

// TestComputeCapability checks the numeric form of the architecture against
// the string one, since the two are now derived from the same pair of numbers
// and a mismatch would mean one of them is invented.
func TestComputeCapability(t *testing.T) {
	requireDevice(t)
	ctx, err := cuda.NewContext(0)
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Close()

	major, minor := ctx.ComputeCapability()
	t.Logf("device %q, compute capability %d.%d", ctx.Name(), major, minor)
	if major <= 0 {
		t.Fatalf("ComputeCapability() = (%d, %d), want a real capability on a device", major, minor)
	}
	parsedMajor, parsedMinor, err := cuda.ParseArch(ctx.Arch())
	if err != nil {
		t.Fatalf("ParseArch(%q): %v", ctx.Arch(), err)
	}
	if parsedMajor != major || parsedMinor != minor {
		t.Errorf("Arch() = %q, which reads back as (%d, %d), want (%d, %d)",
			ctx.Arch(), parsedMajor, parsedMinor, major, minor)
	}
}

// TestLoadPTXError pins down the codes the driver really returns for input it
// will not load, because they are not the one the name suggests: text that has
// no PTX header at all is rejected as an image, and only something the JIT
// recognises as PTX and then chokes on comes back as CUDA_ERROR_INVALID_PTX.
// Observed on a T550 with driver 580 and CUDA 12.8.
func TestLoadPTXError(t *testing.T) {
	requireDevice(t)
	ctx, err := cuda.NewContext(0)
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Close()

	if _, err := ctx.LoadPTX([]byte("this is not ptx\n")); !errors.Is(err, cuda.ErrInvalidImage) {
		t.Errorf("LoadPTX(garbage) = %v, want CUDA_ERROR_INVALID_IMAGE", err)
	}

	// A well-formed header with a body the JIT cannot compile is the case the
	// name is actually about.
	const brokenPTX = ".version 8.0\n.target sm_75\n.address_size 64\n.visible .entry k() { bogus; ret; }\n"
	_, err = ctx.LoadPTX([]byte(brokenPTX))
	if !errors.Is(err, cuda.ErrInvalidPTX) {
		t.Errorf("LoadPTX(broken PTX) = %v, want CUDA_ERROR_INVALID_PTX", err)
	}
	var drvErr *cuda.Error
	if !errors.As(err, &drvErr) {
		t.Fatalf("errors.As(%v, *cuda.Error) = false", err)
	}
	if drvErr.Op != "cuModuleLoadData" {
		t.Errorf("recovered Op = %q, want %q", drvErr.Op, "cuModuleLoadData")
	}
	if drvErr.Desc == "" {
		t.Error("recovered Desc is empty, want the driver's description")
	}
}

// TestCompileError checks that a refused kernel arrives as a *CompileError
// with its log intact: the log is the only thing that says what the generated
// CUDA got wrong, and it used to be reachable only by reading the message.
func TestCompileError(t *testing.T) {
	requireDevice(t)
	_, err := cuda.Compile("__global__ void k(){ nope; }", "k.cu", "compute_75")
	if err == nil {
		t.Fatal("Compile of a kernel with an undefined identifier succeeded")
	}
	var compErr *cuda.CompileError
	if !errors.As(err, &compErr) {
		t.Fatalf("errors.As(%v, *cuda.CompileError) = false", err)
	}
	if compErr.Log == "" {
		t.Error("CompileError.Log is empty, want the compiler's diagnostics")
	}
	if compErr.Name != "k.cu" || compErr.Arch != "compute_75" {
		t.Errorf("CompileError{Name: %q, Arch: %q}, want {\"k.cu\", \"compute_75\"}", compErr.Name, compErr.Arch)
	}
	if !errors.Is(err, cuda.ErrNVRTCCompilation) {
		t.Errorf("errors.Is(%v, ErrNVRTCCompilation) = false", err)
	}
	t.Logf("nvrtc refused the kernel: %v", err)
}

// TestNVRTCVersion reports the compiler this build is linked against; it is
// worth having in a test log next to a kernel that only fails on one toolkit.
//
// A missing toolkit is a skip rather than a failure, for the same reason the
// device tests skip a missing device: there is no question to answer, and the
// tagged suite has to be runnable on a machine that has neither if it is ever
// to gate anything.
func TestNVRTCVersion(t *testing.T) {
	major, minor, err := cuda.NVRTCVersion()
	if errors.Is(err, cuda.ErrNoCUDA) {
		t.Skipf("no CUDA toolkit available: %v", err)
	}
	if err != nil {
		t.Fatalf("NVRTCVersion: %v", err)
	}
	if major < 11 {
		t.Errorf("NVRTCVersion() = (%d, %d), want at least CUDA 11", major, minor)
	}
	t.Logf("nvrtc %d.%d", major, minor)
}
