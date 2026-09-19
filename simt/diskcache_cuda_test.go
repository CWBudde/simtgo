//go:build cuda

package simt

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

// The persistent PTX cache, end to end: what internal/jit tests as a store,
// tested here as the thing that takes NVRTC out of a build.
//
// Every test below redirects SIMTGO_PTX_CACHE at a directory of its own, so
// nothing here reads or writes the developer's real cache, and each starts
// cold. t.Setenv forbids t.Parallel, which is correct: the directory is
// process-wide.

// cacheEntries lists the .ptx files in a redirected cache.
func cacheEntries(t *testing.T, dir string) []string {
	t.Helper()
	found, err := filepath.Glob(filepath.Join(dir, "*.ptx"))
	if err != nil {
		t.Fatalf("Glob: %v", err)
	}
	return found
}

// TestDiskCacheServesTheSecondBuild is the whole point of the item, and it is
// discriminating rather than merely consistent.
//
// "The second build produced the same PTX" proves nothing: so would compiling
// twice. So the first build's bytes are read back out of the cache, a comment
// is appended -- PTX takes // comments, and one changes nothing about what
// ptxas makes of it -- and written back under the same key. A second build
// that returns those bytes can only have read the file, because NVRTC would
// never emit that line.
func TestDiskCacheServesTheSecondBuild(t *testing.T) {
	swapRegistry(t)
	dir := t.TempDir()
	t.Setenv("SIMTGO_PTX_CACHE", dir)

	first := freshDevice(t)
	k1, err := Build(first, kernelSources, "FIR", WithoutPrebuilt(), WithCacheDir(""))
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if k1.Prebuilt {
		t.Fatal("setup: WithoutPrebuilt still took the prebuilt path")
	}

	entries := cacheEntries(t, dir)
	if len(entries) != 1 {
		t.Fatalf("the cache holds %d entries after one build, want 1: %v", len(entries), entries)
	}
	stored, err := os.ReadFile(entries[0])
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(stored, k1.PTX) {
		t.Errorf("the cached file is %d bytes, the build reported %d", len(stored), len(k1.PTX))
	}

	const marker = "\n// served from the on-disk cache\n"
	if err := os.WriteFile(entries[0], append(stored, marker...), 0o644); err != nil {
		t.Fatal(err)
	}

	// A context of its own: the module cache on the first one would answer
	// before anything reached the disk.
	second := freshDevice(t)
	k2, err := Build(second, kernelSources, "FIR", WithoutPrebuilt(), WithCacheDir(""))
	if err != nil {
		t.Fatalf("Build (second context): %v", err)
	}
	if !bytes.Contains(k2.PTX, []byte(marker)) {
		t.Error("the second build recompiled: its PTX is not the bytes on disk")
	}
	if k2.Prebuilt {
		t.Error("a disk-cache hit reported Prebuilt: that field names the registry, not this cache")
	}
	if k2.Arch != first.Arch() {
		t.Errorf("Arch = %q, want the device's own %q", k2.Arch, first.Arch())
	}
}

// TestWithoutDiskCacheIsANegativeControl. An option that quietly does nothing
// is worse than no option, and this is the same discipline WithoutPrebuilt
// gets in TestWithoutPrebuiltOnAWarmContext.
func TestWithoutDiskCacheIsANegativeControl(t *testing.T) {
	swapRegistry(t)
	dir := t.TempDir()
	t.Setenv("SIMTGO_PTX_CACHE", dir)

	dev := freshDevice(t)
	if _, err := Build(dev, kernelSources, "FIR",
		WithoutPrebuilt(), WithoutDiskCache(), WithCacheDir("")); err != nil {
		t.Fatalf("Build: %v", err)
	}
	if entries := cacheEntries(t, dir); len(entries) != 0 {
		t.Errorf("WithoutDiskCache wrote %d entries: %v", len(entries), entries)
	}

	// And it does not read one either. Plant PTX under what would be the key
	// and check the build does not come back with it.
	//
	// The populating build needs a context of its own: dev has the module
	// loaded already, so its cache would answer and the build callback --
	// which is where the disk is written -- would never run. That is not a
	// hypothetical; it is what the first version of this test did.
	populate := freshDevice(t)
	if _, err := Build(populate, kernelSources, "FIR", WithoutPrebuilt(), WithCacheDir("")); err != nil {
		t.Fatalf("Build (to populate the cache): %v", err)
	}
	entries := cacheEntries(t, dir)
	if len(entries) != 1 {
		t.Fatalf("the cache holds %d entries, want 1: %v", len(entries), entries)
	}
	stored, err := os.ReadFile(entries[0])
	if err != nil {
		t.Fatal(err)
	}
	const marker = "\n// planted\n"
	if err := os.WriteFile(entries[0], append(stored, marker...), 0o644); err != nil {
		t.Fatal(err)
	}
	fresh := freshDevice(t)
	k, err := Build(fresh, kernelSources, "FIR",
		WithoutPrebuilt(), WithoutDiskCache(), WithCacheDir(""))
	if err != nil {
		t.Fatalf("Build with WithoutDiskCache: %v", err)
	}
	if bytes.Contains(k.PTX, []byte(marker)) {
		t.Error("WithoutDiskCache read the cache anyway")
	}
}

// TestWithoutDiskCacheOnAWarmContext is the negative control's other half,
// and the one a fresh context cannot see.
//
// The module cache on *cuda.Context answers before jit.Load's build callback
// runs, so a context that has already loaded this kernel by way of the disk
// used to satisfy a later WithoutDiskCache() from that same module -- the
// option doing nothing, silently, which is the one thing a control must never
// do. This is TestWithoutPrebuiltOnAWarmContext's twin, one option later, and
// the cache key carries the mode for the same reason it carries the origin.
func TestWithoutDiskCacheOnAWarmContext(t *testing.T) {
	swapRegistry(t)
	dir := t.TempDir()
	t.Setenv("SIMTGO_PTX_CACHE", dir)

	// Populate the disk cache, in a context that is then dropped.
	populate := freshDevice(t)
	if _, err := Build(populate, kernelSources, "FIR", WithoutPrebuilt(), WithCacheDir("")); err != nil {
		t.Fatalf("Build (to populate the cache): %v", err)
	}
	entries := cacheEntries(t, dir)
	if len(entries) != 1 {
		t.Fatalf("the cache holds %d entries, want 1: %v", len(entries), entries)
	}
	stored, err := os.ReadFile(entries[0])
	if err != nil {
		t.Fatal(err)
	}
	const marker = "\n// served from the on-disk cache\n"
	if err := os.WriteFile(entries[0], append(stored, marker...), 0o644); err != nil {
		t.Fatal(err)
	}

	// One context, warmed from the disk and then asked for the other path.
	dev := freshDevice(t)
	warm, err := Build(dev, kernelSources, "FIR", WithoutPrebuilt(), WithCacheDir(""))
	if err != nil {
		t.Fatalf("Build (warming): %v", err)
	}
	if !bytes.Contains(warm.PTX, []byte(marker)) {
		t.Fatal("setup: the warming build did not come from the disk cache")
	}

	cold, err := Build(dev, kernelSources, "FIR",
		WithoutPrebuilt(), WithoutDiskCache(), WithCacheDir(""))
	if err != nil {
		t.Fatalf("Build with WithoutDiskCache on a warm context: %v", err)
	}
	if bytes.Contains(cold.PTX, []byte(marker)) {
		t.Error("WithoutDiskCache was answered from the module already loaded off the disk")
	}
}

// TestDiskCacheRecoversFromAnUnloadableEntry covers the one way a
// content-addressed cache can still be wrong: the file is not what NVRTC
// wrote. Truncation, tampering, an interrupted write by something else --
// garbage stands in for all of them.
//
// Not the driver-downgrade case, which this recovery does not mend: there the
// cached .version is the one the local NVRTC emits, so the recompile emits it
// again. The planted bytes below happen to come back as
// CUDA_ERROR_UNSUPPORTED_PTX_VERSION all the same, measured, which is exactly
// why the retry is not narrowed by result code -- see Load in
// internal/jit/jit.go.
//
// Both halves matter. The build must succeed, and the bad entry must be gone
// -- otherwise every process afterwards pays the failed load and the
// recompile for as long as the file survives.
func TestDiskCacheRecoversFromAnUnloadableEntry(t *testing.T) {
	swapRegistry(t)
	dir := t.TempDir()
	t.Setenv("SIMTGO_PTX_CACHE", dir)

	warm := freshDevice(t)
	good, err := Build(warm, kernelSources, "FIR", WithoutPrebuilt(), WithCacheDir(""))
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	entries := cacheEntries(t, dir)
	if len(entries) != 1 {
		t.Fatalf("the cache holds %d entries, want 1: %v", len(entries), entries)
	}
	if err := os.WriteFile(entries[0], []byte(".version 99.9\nnot ptx at all\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	dev := freshDevice(t)
	k, err := Build(dev, kernelSources, "FIR", WithoutPrebuilt(), WithCacheDir(""))
	if err != nil {
		t.Fatalf("Build over an unloadable cache entry: %v", err)
	}
	if !bytes.Equal(k.PTX, good.PTX) {
		t.Errorf("the recompile produced %d bytes, want the %d NVRTC produced before",
			len(k.PTX), len(good.PTX))
	}

	// The retry compiles with the cache off, so the entry is dropped and not
	// rewritten. Either is acceptable to a caller; only leaving the bad one
	// in place is not.
	if after := cacheEntries(t, dir); len(after) == 1 {
		left, err := os.ReadFile(after[0])
		if err != nil {
			t.Fatal(err)
		}
		if bytes.Contains(left, []byte("not ptx at all")) {
			t.Error("the unloadable entry is still in the cache")
		}
	}
}
