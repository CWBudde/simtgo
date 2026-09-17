package cuda

import (
	"errors"
	"slices"
	"strings"
	"testing"
)

// These tests are deliberately not gated on the "cuda" build tag: which file
// the driver is loaded from is a question of environment and paths, and the
// answer has to be checkable on a machine with no CUDA at all -- which is
// precisely the machine where getting it wrong is expensive.

// clearEnv puts the search into a known state, since the machine running the
// tests may well have CUDA_PATH set for its own reasons.
func clearEnv(t *testing.T) {
	t.Helper()
	for _, k := range []string{envLibCUDA, envLibNVRTC, "CUDA_PATH", "CUDA_HOME"} {
		t.Setenv(k, "")
	}
}

func TestDriverCandidatesDefault(t *testing.T) {
	clearEnv(t)
	got := driverCandidates()
	want := []string{"libcuda.so.1", "libcuda.so"}
	if !slices.Equal(got, want) {
		t.Fatalf("driverCandidates() = %v, want %v", got, want)
	}
	// The toolkit's stub library satisfies a linker and fails every call, so
	// it must never be somewhere the search can reach.
	for _, c := range got {
		if strings.Contains(c, "stubs") {
			t.Errorf("driver candidate %q is a link-time stub", c)
		}
	}
}

func TestEnvOverrideIsExclusive(t *testing.T) {
	// An override that fell back to the system library would send whoever set
	// it to debug a library they did not choose, so it replaces the search
	// rather than heading it.
	clearEnv(t)
	t.Setenv(envLibCUDA, "/opt/weird/libcuda.so.1")
	t.Setenv(envLibNVRTC, "/opt/weird/libnvrtc.so")

	if got := driverCandidates(); !slices.Equal(got, []string{"/opt/weird/libcuda.so.1"}) {
		t.Errorf("driverCandidates() = %v, want the override alone", got)
	}
	if got := nvrtcCandidates(); !slices.Equal(got, []string{"/opt/weird/libnvrtc.so"}) {
		t.Errorf("nvrtcCandidates() = %v, want the override alone", got)
	}
}

func TestNVRTCCandidatesPreferConfiguredRoot(t *testing.T) {
	clearEnv(t)
	t.Setenv("CUDA_PATH", "/opt/cuda-12.4")

	got := nvrtcCandidates()
	root := slices.Index(got, "/opt/cuda-12.4/lib64/libnvrtc.so.13")
	bare := slices.Index(got, "libnvrtc.so.12")
	fallback := slices.Index(got, "/usr/local/cuda/lib64/libnvrtc.so.12")

	switch {
	case root < 0 || bare < 0 || fallback < 0:
		t.Fatalf("missing candidates in %v", got)
	case root > bare:
		t.Errorf("CUDA_PATH candidate at %d comes after the bare soname at %d", root, bare)
	case bare > fallback:
		t.Errorf("bare soname at %d comes after the /usr/local/cuda fallback at %d", bare, fallback)
	}
}

func TestNVRTCCandidatesNewestMajorFirst(t *testing.T) {
	clearEnv(t)
	got := nvrtcCandidates()
	v13 := slices.Index(got, "libnvrtc.so.13")
	v12 := slices.Index(got, "libnvrtc.so.12")
	v11 := slices.Index(got, "libnvrtc.so.11")
	if v13 < 0 || v12 < 0 || v11 < 0 {
		t.Fatalf("missing sonames in %v", got)
	}
	if v13 > v12 || v12 > v11 {
		t.Errorf("sonames out of order in %v", got)
	}
}

func TestNVRTCCandidatesAreUnique(t *testing.T) {
	// CUDA_PATH pointing at a well-known root would otherwise produce the same
	// path twice, which is harmless to try but misleading in an error message.
	clearEnv(t)
	t.Setenv("CUDA_PATH", "/usr/local/cuda")
	t.Setenv("CUDA_HOME", "/usr/local/cuda")

	got := nvrtcCandidates()
	seen := make(map[string]bool, len(got))
	for _, c := range got {
		if seen[c] {
			t.Errorf("duplicate candidate %q in %v", c, got)
		}
		seen[c] = true
	}
}

func TestLibraryErrorNamesWhatWasTried(t *testing.T) {
	err := &LibraryError{
		Lib:    "libnvrtc",
		Tried:  []string{"libnvrtc.so.12", "/usr/local/cuda/lib64/libnvrtc.so"},
		EnvVar: envLibNVRTC,
	}
	msg := err.Error()
	for _, want := range []string{"libnvrtc.so.12", "/usr/local/cuda/lib64/libnvrtc.so", envLibNVRTC} {
		if !strings.Contains(msg, want) {
			t.Errorf("message %q does not mention %q", msg, want)
		}
	}

	if !errors.Is(err, ErrLibraryNotFound) {
		t.Error("errors.Is(err, ErrLibraryNotFound) = false")
	}
	// "No CUDA compiled in" and "no CUDA installed" are the same condition to
	// a caller, so code written against the stub keeps working unchanged.
	if !errors.Is(err, ErrNoCUDA) {
		t.Error("errors.Is(err, ErrNoCUDA) = false")
	}
}
