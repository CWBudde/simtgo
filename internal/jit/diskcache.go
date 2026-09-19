package jit

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/CWBudde/gocuda/cuda"
)

// The persistent PTX cache. It removes NVRTC from process start for a kernel
// with no prebuilt artifact: the module cache on *cuda.Context dies with the
// context and so with the process, while this survives it.
//
// It is not the same thing as CacheDir, and the two are deliberately separate.
// CacheDir is a dump for a human to read -- named after the kernel, overwritten
// freely, off by default in a library. This is a content-addressed store keyed
// on everything that can change the bytes, and reading it back is the point.
//
// Nothing here may sink a build. A cache is an optimisation, so every error
// below is a miss: an unwritable directory, a half-written file, a filesystem
// that has gone away. The only thing that must not happen is serving PTX that
// belongs to a different compilation, which is what the key exists for.

// diskKey identifies compiled PTX on disk.
//
// Three terms, and the third is the one the in-memory cacheKey does not need.
// A module cache lives inside one process, where NVRTC cannot change under it;
// a file outlives the toolkit that wrote it, and PTX carries an ISA version in
// its .version directive that the driver either accepts or does not. Serving a
// 12.8 artifact to a process that has loaded NVRTC 12.0 would be handing back
// something the local compiler would never have produced.
//
// The origin term of cacheKey has no counterpart here: this cache only ever
// holds what NVRTC compiled, since the prebuilt path already has its bytes and
// has nothing to look up.
//
// Each term is length-prefixed rather than joined by a separator byte. A NUL
// is not reachable in any of the three today -- the source is generated CUDA
// C, the other two are short machine-written strings -- but "not reachable
// today" is how a joined key becomes ambiguous later, and a length is one
// line. A test hashes two different triples that a separator would run
// together.
func diskKey(src, arch, nvrtc string) string {
	h := sha256.New()
	for _, term := range []string{src, arch, nvrtc} {
		fmt.Fprintf(h, "%d:", len(term))
		h.Write([]byte(term))
	}
	return hex.EncodeToString(h.Sum(nil))
}

// nvrtcTag names the compiler, for the key.
//
// Resolved once: every Build asks, and the answer cannot change under a
// running process. A toolkit that cannot be loaded gives "none", which is a
// perfectly good cache partition -- nothing will ever compile on that path, so
// nothing will ever be stored under it.
var nvrtcTag = sync.OnceValue(func() string {
	major, minor, err := cuda.NVRTCVersion()
	if err != nil {
		return "none"
	}
	return fmt.Sprintf("nvrtc%d.%d", major, minor)
})

// diskCacheDir is where compiled PTX lives.
//
// The user cache directory rather than the repository's .gocuda-cache, for the
// same reason internal/fuzz/hostrun keeps its binaries there: these are not
// artifacts anyone inspects, they accumulate one per kernel shape, and a
// library has no business writing them into whatever directory its caller
// happened to start in.
//
// GOCUDA_PTX_CACHE moves them. That is not the process-wide switch
// docs/decisions.md rejected for bounds checks: it says where bytes are kept
// and never what is compiled, so no program's output changes because another
// part of it set the variable. Whether the cache is used at all is a build
// option, not an environment variable.
//
// The variable is read on every call and only the fallback is resolved once.
// Caching the answer -- which is what internal/fuzz/hostrun does one layer
// down -- made the tests below untestable past the first one, since a
// sync.OnceValue cannot be re-resolved and each test wants a directory of its
// own. An environment lookup is nanoseconds against a file read, and a
// variable that takes effect when it is set is the less surprising of the two
// behaviours anyway.
func diskCacheDir() string {
	if dir := os.Getenv("GOCUDA_PTX_CACHE"); dir != "" {
		return dir
	}
	return defaultDiskCacheDir()
}

var defaultDiskCacheDir = sync.OnceValue(func() string {
	base, err := os.UserCacheDir()
	if err != nil {
		base = os.TempDir()
	}
	return filepath.Join(base, "gocuda", "ptx")
})

// diskLocks is one mutex per cache entry rather than one for the package: two
// goroutines wanting the same kernel should compile it once, and two wanting
// different ones have no reason to wait for each other.
var diskLocks sync.Map // path -> *sync.Mutex

func diskLock(path string) *sync.Mutex {
	l, _ := diskLocks.LoadOrStore(path, new(sync.Mutex))
	mu, ok := l.(*sync.Mutex)
	if !ok { // impossible: nothing else is ever stored here
		panic("jit: diskLocks holds a non-mutex")
	}
	return mu
}

// diskPath is where the PTX for this key would be. An empty key means the
// cache is off.
func diskPath(key string) string {
	if key == "" {
		return ""
	}
	return filepath.Join(diskCacheDir(), key+".ptx")
}

// readDisk returns previously compiled PTX, or nil if there is none.
//
// An empty file is a miss rather than a hit: a rename cannot produce one, but
// a disk that filled up mid-write under some other program could, and PTX of
// zero bytes would reach cuModuleLoadData as a generic failure a long way from
// here.
func readDisk(key string) []byte {
	path := diskPath(key)
	if path == "" {
		return nil
	}
	mu := diskLock(path)
	mu.Lock()
	defer mu.Unlock()
	b, err := os.ReadFile(path)
	if err != nil || len(b) == 0 {
		return nil
	}
	return b
}

// writeDisk stores compiled PTX under its key.
//
// Written to a temporary file in the same directory and renamed into place, so
// that a second process finding the file finds a whole one -- rename is atomic
// within a filesystem, and both ends are in the cache directory by
// construction.
func writeDisk(key string, ptx []byte) {
	path := diskPath(key)
	if path == "" || len(ptx) == 0 {
		return
	}
	mu := diskLock(path)
	mu.Lock()
	defer mu.Unlock()

	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return
	}
	tmp, err := os.CreateTemp(dir, "ptx-")
	if err != nil {
		return
	}
	name := tmp.Name()
	if _, err := tmp.Write(ptx); err != nil {
		_ = tmp.Close()
		_ = os.Remove(name)
		return
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(name)
		return
	}
	if err := os.Rename(name, path); err != nil {
		_ = os.Remove(name)
	}
}

// forgetDisk drops an entry the driver refused.
//
// A file the driver will not load is worthless whatever made it so, and
// without this every process would pay the failed load as well as the
// recompile for as long as the file survived. Which causes the recompile
// after it can and cannot mend is set out at Load, in jit.go.
func forgetDisk(key string) {
	path := diskPath(key)
	if path == "" {
		return
	}
	mu := diskLock(path)
	mu.Lock()
	defer mu.Unlock()
	_ = os.Remove(path)
}
