package cuda

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

// This file decides *which file* the driver and NVRTC are loaded from. It is
// deliberately free of any loading: the policy is ordinary Go, so it can be
// tested on a machine with no CUDA, no GPU and no "cuda" build tag, which is
// where a path bug is cheapest to find. loader_cuda.go does the opening.
//
// Until Phase 1.2 this question did not exist at run time: the library was
// chosen at *build* time by the #cgo flags in driver_cuda.go, which named
// /usr/local/cuda and nothing else. Nothing now links CUDA, so the question
// moved here.

// ErrLibraryNotFound reports that a shared library could not be opened from
// any candidate path. Use errors.Is to recognise "this machine has no CUDA"
// as distinct from "CUDA said no".
var ErrLibraryNotFound = errors.New("cuda: shared library not found")

// LibraryError says which library could not be loaded and everywhere that was
// tried. The list is the point: a driver that is present but unreadable, a
// toolkit installed somewhere unusual and a machine with no CUDA at all are
// three different problems, and only the candidates distinguish them.
type LibraryError struct {
	// Lib is the library as this package names it: "libcuda" or "libnvrtc".
	Lib string
	// Tried lists the candidates in the order they were attempted.
	Tried []string
	// EnvVar is the environment variable that overrides the search.
	EnvVar string
}

func (e *LibraryError) Error() string {
	return fmt.Sprintf("cuda: cannot load %s: tried %s; set %s to a full path to override",
		e.Lib, strings.Join(e.Tried, ", "), e.EnvVar)
}

// Is reports ErrLibraryNotFound so callers can match the condition without
// naming this type, and ErrNoCUDA so that code written against the
// build-tag-less stub keeps working: to a caller, "no CUDA compiled in" and
// "no CUDA installed" are the same unavailability.
func (e *LibraryError) Is(target error) bool {
	return target == ErrLibraryNotFound || target == ErrNoCUDA
}

// The environment variables that override the search outright.
const (
	envLibCUDA  = "GOCUDA_LIBCUDA"
	envLibNVRTC = "GOCUDA_LIBNVRTC"
)

// driverCandidates lists the files to try for the CUDA driver library.
//
// The driver is installed by the *driver*, not by the toolkit, and lands in
// the dynamic linker's default search path, so the bare soname is the right
// answer almost everywhere. The toolkit's own lib64/stubs/libcuda.so is
// deliberately not a candidate: it exists to satisfy a linker on a machine
// with no GPU and every call into it fails.
func driverCandidates() []string {
	if p := os.Getenv(envLibCUDA); p != "" {
		return []string{p}
	}
	return []string{"libcuda.so.1", "libcuda.so"}
}

// nvrtcCandidates lists the files to try for NVRTC.
//
// NVRTC *is* part of the toolkit, so the toolkit root matters here. An
// explicitly configured root (CUDA_PATH, then CUDA_HOME) is tried before the
// linker's default path: someone who sets it is telling us which of several
// installations to use, and silently preferring the system one would ignore
// that. The well-known roots come last, as a fallback rather than a policy.
func nvrtcCandidates() []string {
	if p := os.Getenv(envLibNVRTC); p != "" {
		return []string{p}
	}

	// Newest first: NVRTC's soname carries the CUDA major version.
	sonames := []string{"libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so.11", "libnvrtc.so"}

	var out []string
	add := func(p string) {
		if !slices.Contains(out, p) {
			out = append(out, p)
		}
	}
	fromRoot := func(root string) {
		if root == "" {
			return
		}
		for _, dir := range []string{"lib64", "lib"} {
			for _, name := range sonames {
				add(filepath.Join(root, dir, name))
			}
		}
	}

	fromRoot(os.Getenv("CUDA_PATH"))
	fromRoot(os.Getenv("CUDA_HOME"))
	for _, name := range sonames {
		add(name)
	}
	for _, root := range []string{"/usr/local/cuda", "/opt/cuda"} {
		fromRoot(root)
	}
	return out
}
