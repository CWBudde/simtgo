//go:build !cuda

package main

import (
	"errors"
	"strings"
	"testing"

	"github.com/CWBudde/gocuda/cuda"
)

// TestNoNVRTCIsExplained covers the path a machine without a toolkit takes.
// Every other test in this package runs with -no-ptx, so nothing else reaches
// it at all.
//
// The file is tagged rather than the test skipped, because the premise is a
// property of this build and not of the machine: without the "cuda" tag
// cuda.Compile is the stub and always reports ErrNoCUDA, which is the same
// unavailability a missing libnvrtc reports. Left in the untagged file it
// would also be compiled by "go test -tags cuda ./...", where a machine that
// *has* a toolkit compiles the kernel successfully, run returns nil, and the
// assertion below fails -- a test that passes only where it was written, which
// is the thing two other tagged tests in this repository had to be fixed for.
func TestNoNVRTCIsExplained(t *testing.T) {
	o := fixture(t, map[string]string{"k.go": goodKernel})
	o.noPTX = false
	err := run(o)
	if !errors.Is(err, cuda.ErrNoCUDA) {
		t.Fatalf("run without NVRTC: got %v, want an error matching cuda.ErrNoCUDA", err)
	}
	if !strings.Contains(err.Error(), "-no-ptx") {
		t.Errorf("the error must name the way out; got %v", err)
	}
}
