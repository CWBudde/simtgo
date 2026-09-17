// Package jit compiles generated CUDA C and loads it, for both the SIMT and
// the tile track.
package jit

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/CWBudde/gocuda/cuda"
)

// DefaultCacheDir is where the generated CUDA and the PTX that was actually
// loaded are written, so that what ran can be read afterwards. It is the value
// a caller starts from rather than a process-wide switch: each Request carries
// its own directory, so one track's debugging preference cannot silently
// change the other's.
const DefaultCacheDir = ".gocuda-cache"

// Request is one kernel to load.
type Request struct {
	Src, Name string

	// PTX, when non-nil, is ahead-of-time PTX for Src and is loaded instead of
	// calling NVRTC. PTXArch is the virtual architecture it was built for,
	// which is not necessarily the device's: PTX is forward compatible, so a
	// compute_75 artifact is what a compute_89 device loads and JITs.
	PTX     []byte
	PTXArch string

	// CacheDir receives the .cu and .ptx dumps. Empty disables them.
	CacheDir string
}

// Result is a compiled and loaded kernel.
type Result struct {
	Func *cuda.Function
	PTX  []byte
	Log  string

	// Prebuilt says the PTX came from Request.PTX, so NVRTC never ran. It is
	// the only honest way to tell the two paths apart: the PTX and the kernel
	// are meant to be indistinguishable otherwise.
	Prebuilt bool

	// Arch is what the loaded PTX was built for, which on the NVRTC path is
	// the device's own architecture and on the prebuilt path is whatever the
	// generator targeted.
	Arch string
}

// artifacts is what this package remembers about a module besides the module
// itself, so that the second Load of a source reports the PTX, the log and the
// origin exactly as the first one did.
//
// It is a map here rather than a value parked in the context because, unlike a
// CUmodule, none of this dies with a context: bytes and strings stay valid and
// stay true for as long as the source they were built from exists. Memory is
// bounded by the number of distinct kernel sources a process compiles.
//
// An entry describes the most recent build for the key. Two builds of the same
// source for the same architecture differ only in where the PTX came from, and
// only a caller that deliberately mixes the prebuilt and the NVRTC path for
// one kernel can observe which of the two is reported.
type artifacts struct {
	ptx      []byte
	log      string
	prebuilt bool
	arch     string
}

var (
	artifactMu sync.Mutex
	artifactOf = map[string]artifacts{}
)

// Load returns the entry point of req's kernel, compiling it with NVRTC unless
// req carries prebuilt PTX. Identical sources are loaded once per context.
//
// The module cache deliberately lives on the context rather than in a
// package-level map here. A CUmodule is only valid inside the context it was
// loaded into, while a package variable lives as long as the process: a shared
// cache therefore hands a module of a long-released context to whoever
// compiles the same source next, and every generated tile kernel is called
// "fused", so those collisions are the rule rather than the exception.
// Splitting it this way keeps the compile and dump policy here, where the file
// names and the NVRTC options are decided, and puts the module's lifetime in
// the one place that can end it: the context, which unloads what it owns on
// Close.
func Load(ctx *cuda.Context, req Request) (*Result, error) {
	// The architecture is folded into the hash even though a context has only
	// one, so that dumps of the same source built for different devices do not
	// overwrite each other. The full sum keys the cache: the 12 hex digits
	// below are a convenience for reading file names, not 48 bits of collision
	// resistance to hang a module lookup on.
	sum := sha256.Sum256([]byte(req.Src + "\x00" + ctx.Arch()))
	key := hex.EncodeToString(sum[:])
	short := key[:12]

	res, err := load(ctx, key, short, req)
	if err != nil && req.PTX != nil && rejectedPTX(err) {
		// Prebuilt PTX was produced by the NVRTC of whoever ran the generator,
		// and it records that NVRTC's ISA version in its .version directive --
		// 8.7 needs a driver of at least 570. A machine with an older driver,
		// or one whose architecture the generator never targeted, refuses the
		// image outright. Its own NVRTC matches its own driver by
		// construction, so compiling here produces something loadable, and the
		// only cost of the ahead-of-time path going wrong is the compile it
		// was meant to save. The retry is conditioned on the driver saying it
		// could not accept the image, because any other failure would be just
		// as fatal the second time round.
		//
		// A failed build leaves the module cache untouched, so this is simply
		// a second LoadPTXCached call under the same key.
		req.PTX, req.PTXArch = nil, ""
		res, err = load(ctx, key, short, req)
	}
	return res, err
}

// rejectedPTX reports whether the driver turned the image down, as opposed to
// failing for a reason a recompilation cannot mend.
func rejectedPTX(err error) bool {
	return errors.Is(err, cuda.ErrUnsupportedPTXVersion) ||
		errors.Is(err, cuda.ErrInvalidPTX) ||
		errors.Is(err, cuda.ErrInvalidImage) ||
		errors.Is(err, cuda.ErrNoBinaryForGPU)
}

// load is one attempt: at most one build, and at most one module load.
func load(ctx *cuda.Context, key, short string, req Request) (*Result, error) {
	// compileErr distinguishes a failure of ours from one of the context's, so
	// that the message says which step actually gave up. Together with built
	// it is written under the context's lock and read after LoadPTXCached has
	// returned, so the two never race.
	var compileErr error
	var built *artifacts

	mod, err := ctx.LoadPTXCached(key, func() ([]byte, error) {
		if req.PTX != nil {
			// The dump still happens on this path: the point of the directory
			// is to show what ran, and "nothing was written because nothing
			// was compiled" is the least useful answer it could give while
			// chasing a kernel that misbehaves only when prebuilt.
			dump(req.CacheDir, req.Name, short, req.Src, req.PTX)
			built = &artifacts{ptx: req.PTX, prebuilt: true, arch: req.PTXArch}
			return req.PTX, nil
		}
		p, err := cuda.Compile(req.Src, req.Name+".cu", ctx.Arch())
		if err != nil {
			// Dump the source even though there is no PTX: a kernel that does
			// not compile is exactly the one worth reading. The error is
			// wrapped rather than replaced so that a caller can still recover
			// the *cuda.CompileError and read the compiler log out of it.
			dump(req.CacheDir, req.Name, short, req.Src, nil)
			compileErr = fmt.Errorf("compiling %s: %w", req.Name, err)
			return nil, compileErr
		}
		dump(req.CacheDir, req.Name, short, req.Src, p.Bytes)
		built = &artifacts{ptx: p.Bytes, log: p.Log, arch: ctx.Arch()}
		return p.Bytes, nil
	})
	if err != nil {
		if compileErr != nil {
			return nil, compileErr
		}
		return nil, fmt.Errorf("loading %s: %w", req.Name, err)
	}

	fn, err := mod.Function(req.Name)
	if err != nil {
		return nil, err
	}

	a := artifacts{}
	if built != nil {
		a = *built
		remember(key, a)
	} else {
		// A cache hit means some earlier build in this process loaded that
		// module, and every build records what it produced, so the lookup
		// answers. A miss would leave the PTX unreported rather than wrong.
		a, _ = recall(key)
	}
	return &Result{Func: fn, PTX: a.ptx, Log: a.log, Prebuilt: a.prebuilt, Arch: a.arch}, nil
}

func remember(key string, a artifacts) {
	artifactMu.Lock()
	defer artifactMu.Unlock()
	artifactOf[key] = a
}

func recall(key string) (artifacts, bool) {
	artifactMu.Lock()
	defer artifactMu.Unlock()
	a, ok := artifactOf[key]
	return a, ok
}

// dump writes the generated sources for inspection. A failure here must not
// sink a launch, so errors are deliberately dropped.
func dump(dir, name, hash, src string, ptx []byte) {
	if dir == "" {
		return
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return
	}
	base := filepath.Join(dir, fmt.Sprintf("%s-%s", name, hash))
	_ = os.WriteFile(base+".cu", []byte(src), 0o644)
	if ptx != nil {
		_ = os.WriteFile(base+".ptx", ptx, 0o644)
	}
}
