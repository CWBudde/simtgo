package jit

import (
	"os"
	"path/filepath"
	"sync"
	"testing"
)

// These test the store itself, with no CUDA anywhere: the keying, the
// atomicity and the failure modes are policy, and a policy that can only be
// checked on a machine with a toolkit installed is a policy nobody checks.
// What the store does inside a build is covered by simt's tagged tests.

// redirect points the cache at a directory of this test's own, by the same
// route a caller has. t.Setenv forbids parallel tests, which is correct here:
// the directory is process-wide.
func redirect(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	t.Setenv("GOCUDA_PTX_CACHE", dir)
	if got := diskCacheDir(); got != dir {
		t.Fatalf("GOCUDA_PTX_CACHE = %q but the cache resolved to %q", dir, got)
	}
	return dir
}

func TestDiskKeySeparatesWhatChangesTheBytes(t *testing.T) {
	base := diskKey("src", "compute_75", "nvrtc12.8")
	for _, tc := range []struct {
		name             string
		src, arch, nvrtc string
		want             bool // same key as base?
	}{
		{"identical", "src", "compute_75", "nvrtc12.8", true},
		{"other source", "src2", "compute_75", "nvrtc12.8", false},
		{"other arch", "src", "compute_86", "nvrtc12.8", false},
		{"other nvrtc", "src", "compute_75", "nvrtc12.0", false},
		{"no toolkit", "src", "compute_75", "none", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := diskKey(tc.src, tc.arch, tc.nvrtc) == base
			if got != tc.want {
				t.Errorf("key equal to base = %v, want %v", got, tc.want)
			}
		})
	}

	// Two triples a separator byte would run together into one digest. The
	// key is length-prefixed so that they cannot, and this is the assertion
	// that caught a first version which joined them with a NUL.
	if diskKey("a\x00b", "c", "d") == diskKey("a", "b\x00c", "d") {
		t.Error("two different triples hash alike: the terms are not separated")
	}
}

func TestDiskRoundTrip(t *testing.T) {
	dir := redirect(t)
	key := diskKey("kernel source", "compute_75", "nvrtc12.8")

	if got := readDisk(key); got != nil {
		t.Fatalf("an empty cache returned %d bytes", len(got))
	}
	writeDisk(key, []byte("the ptx"))
	if got := string(readDisk(key)); got != "the ptx" {
		t.Errorf("readDisk = %q, want %q", got, "the ptx")
	}

	// One file, named by the key: the store is content-addressed, so a second
	// write of the same bytes must not accumulate anything.
	writeDisk(key, []byte("the ptx"))
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 {
		names := make([]string, len(entries))
		for i, e := range entries {
			names[i] = e.Name()
		}
		t.Errorf("cache holds %d files %v, want 1", len(entries), names)
	}

	// A different source is a different entry, not an overwrite.
	other := diskKey("another source", "compute_75", "nvrtc12.8")
	if got := readDisk(other); got != nil {
		t.Errorf("a different source hit the cache and got %q", got)
	}

	forgetDisk(key)
	if got := readDisk(key); got != nil {
		t.Errorf("readDisk after forgetDisk = %q, want a miss", got)
	}
}

// TestDiskMissesRatherThanFails pins the rule the whole file rests on: a cache
// is an optimisation, so everything that can go wrong with it is a miss. The
// one thing that must not happen -- serving bytes from a different
// compilation -- is the key's business and is tested above.
func TestDiskMissesRatherThanFails(t *testing.T) {
	dir := redirect(t)
	key := diskKey("src", "compute_75", "nvrtc12.8")

	// An empty file. A rename cannot produce one, but a disk that filled up
	// under some other program could, and zero bytes would reach
	// cuModuleLoadData as a generic failure a long way from here.
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(diskPath(key), nil, 0o644); err != nil {
		t.Fatal(err)
	}
	if got := readDisk(key); got != nil {
		t.Errorf("an empty cache file was served as %d bytes of PTX", len(got))
	}

	// Writing nothing stores nothing, rather than leaving a file that the
	// line above then has to treat as a miss forever.
	empty := diskKey("empty", "compute_75", "nvrtc12.8")
	writeDisk(empty, nil)
	if _, err := os.Stat(diskPath(empty)); !os.IsNotExist(err) {
		t.Errorf("writeDisk(nil) left a file: Stat err = %v", err)
	}

	// Forgetting something that was never there is not an error.
	forgetDisk(diskKey("never written", "compute_75", "nvrtc12.8"))

	// An unwritable directory is a miss, not a panic and not a failed build.
	// Root ignores the mode bits, so there is nothing to exercise there.
	if os.Geteuid() == 0 {
		t.Log("running as root: the unwritable-directory case cannot be exercised")
		return
	}
	ro := filepath.Join(dir, "readonly")
	if err := os.Mkdir(ro, 0o500); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(ro, 0o700) })
	if err := os.WriteFile(filepath.Join(ro, "probe"), []byte("x"), 0o644); err == nil {
		t.Skip("the directory mode did not take on this filesystem")
	}
	t.Setenv("GOCUDA_PTX_CACHE", filepath.Join(ro, "ptx"))
	blocked := diskKey("blocked", "compute_75", "nvrtc12.8")
	writeDisk(blocked, []byte("the ptx")) // must not panic
	if got := readDisk(blocked); got != nil {
		t.Errorf("a write into an unwritable directory was read back as %q", got)
	}
	forgetDisk(blocked) // must not panic either
}

// TestDiskConcurrentWriters is the reason for the per-entry mutex and the
// rename: readers must never see a partial file, whoever else is writing.
func TestDiskConcurrentWriters(t *testing.T) {
	redirect(t)
	key := diskKey("hot", "compute_75", "nvrtc12.8")
	want := make([]byte, 64*1024)
	for i := range want {
		want[i] = byte(i)
	}

	var wg sync.WaitGroup
	for range 16 {
		wg.Go(func() { writeDisk(key, want) })
		wg.Go(func() {
			// nil (a miss) or the whole thing; never a prefix.
			if got := readDisk(key); got != nil && len(got) != len(want) {
				t.Errorf("read a partial file: %d bytes of %d", len(got), len(want))
			}
		})
	}
	wg.Wait()

	if got := readDisk(key); string(got) != string(want) {
		t.Errorf("after the race the entry is %d bytes, want %d", len(got), len(want))
	}
}

// TestNvrtcTagIsStable covers the half of the key that is not an argument. It
// must not change under a running process: an entry written early and read
// back late has to land on the same key, and a toolkit cannot be swapped out
// from under a loaded library anyway.
func TestNvrtcTagIsStable(t *testing.T) {
	first := nvrtcTag()
	if first == "" {
		t.Fatal("nvrtcTag is empty; it is a key term and must name something")
	}
	if second := nvrtcTag(); second != first {
		t.Errorf("nvrtcTag returned %q then %q; it must not change under a process", first, second)
	}
	// In an untagged build there is no NVRTC to ask, and "none" is a
	// perfectly good partition: nothing compiles on that path, so nothing is
	// ever stored under it.
	t.Logf("nvrtcTag = %q", first)
}

// TestRememberKeepingLogDoesNotBlankACompilersLog covers the one thing a disk
// hit knows less about than the compile that filled the file.
//
// artifactOf is keyed on the cache key and not on the context, so a second
// context reading the PTX off disk would otherwise replace the first
// context's entry with a blank log -- and the first context, whose module
// cache answers before any of this runs, would start reporting no log for a
// compile that had one. That is the guarantee TestJITLoadAcrossContexts pins
// within one context; this is the same guarantee across two.
func TestRememberKeepingLogDoesNotBlankACompilersLog(t *testing.T) {
	const key = "TestRememberKeepingLog"
	t.Cleanup(func() {
		artifactMu.Lock()
		defer artifactMu.Unlock()
		delete(artifactOf, key)
	})

	// A real compile: PTX, a warning, an architecture.
	remember(key, artifacts{ptx: []byte("compiled"), log: "warning: something", arch: "compute_75"})

	// A disk hit for the same key, from another context. Same bytes by
	// construction -- the disk key pins source, architecture and compiler --
	// and no log, because the file holds only the PTX.
	rememberKeepingLog(key, artifacts{ptx: []byte("compiled"), arch: "compute_75"})

	got, ok := recall(key)
	if !ok {
		t.Fatal("the entry disappeared")
	}
	if got.log != "warning: something" {
		t.Errorf("log = %q, want the compile's own %q", got.log, "warning: something")
	}
	if string(got.ptx) != "compiled" {
		t.Errorf("ptx = %q, want %q", got.ptx, "compiled")
	}

	// With nothing recorded, it is an ordinary write: no log to keep, and the
	// entry must still appear rather than being skipped.
	const fresh = "TestRememberKeepingLogFresh"
	t.Cleanup(func() {
		artifactMu.Lock()
		defer artifactMu.Unlock()
		delete(artifactOf, fresh)
	})
	rememberKeepingLog(fresh, artifacts{ptx: []byte("from disk"), arch: "compute_75"})
	got, ok = recall(fresh)
	if !ok {
		t.Fatal("a disk hit with no earlier entry recorded nothing")
	}
	if got.log != "" || string(got.ptx) != "from disk" {
		t.Errorf("got %+v, want the disk hit's own ptx and no log", got)
	}
}
