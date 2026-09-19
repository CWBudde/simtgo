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

	"github.com/CWBudde/simtgo/cuda"
)

// DefaultCacheDir is where the generated CUDA and the PTX that was actually
// loaded are written, so that what ran can be read afterwards. It is the value
// a caller starts from rather than a process-wide switch: each Request carries
// its own directory, so one track's debugging preference cannot silently
// change the other's.
const DefaultCacheDir = ".simtgo-cache"

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

	// FastMath compiles with --use_fast_math. It needs no place in cacheKey:
	// simt sets it only for a kernel whose generated Src carries the
	// "// simtgo: fastmath" marker, so the source the key already hashes
	// differs, and the two builds cannot collide in the module cache.
	FastMath bool

	// NoDiskCache skips the persistent PTX cache, compiling with NVRTC and
	// storing nothing. See diskcache.go for what that cache is and why it is
	// not the same thing as CacheDir.
	NoDiskCache bool
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
// The key carries the origin as well as the source and the architecture, so
// the prebuilt and the NVRTC build of one kernel have entries of their own and
// cannot overwrite each other's -- which they could when the key was only
// (source, architecture), reporting one context's origin to another's caller.
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
	key := cacheKey(ctx, req)
	res, err := load(ctx, key, req)

	// Cached PTX the driver will not accept. What the recompile recovers is a
	// file that is not what NVRTC produced -- truncated, tampered with, a
	// write some other program interrupted -- and that is the likely case,
	// because the key pins the source, the architecture and the NVRTC
	// version, so a hit means the local compiler would have produced these
	// bytes.
	//
	// It does *not* recover a driver downgraded under an unchanged toolkit.
	// There the cached .version is the one this NVRTC emits, so recompiling
	// emits it again and the second load fails with the driver's own error,
	// which is the right thing for a machine that genuinely needs a different
	// toolkit.
	//
	// The result code cannot tell the two apart, and that is measured rather
	// than assumed: corrupt bytes behind a bogus .version come back as
	// CUDA_ERROR_UNSUPPORTED_PTX_VERSION, the same code a real downgrade
	// gives (see cuda.TestLoadPTXError for the codes). Narrowing the retry to
	// the codes only corruption produces would therefore drop the recovery
	// for the likelier cause, to save one compile in the rarer one.
	//
	// Dropping the entry matters either way: a file the driver refuses is
	// worthless whichever reason it refused it for.
	//
	// The retry keeps the original key, so the module it loads is the one the
	// caller's next Build asks for rather than a second copy under a key only
	// this recovery uses.
	if err != nil && req.PTX == nil && !req.NoDiskCache && rejectedPTX(err) {
		forgetDisk(diskKey(req.Src, ctx.Arch(), nvrtcTag()))
		req.NoDiskCache = true
		return load(ctx, key, req)
	}

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
		res, err = load(ctx, cacheKey(ctx, req), req)
	}
	return res, err
}

// cacheKey identifies a loaded module.
//
// The architecture is folded in even though a context has only one, so that
// dumps of the same source built for different devices do not overwrite each
// other. So is where the PTX came from: a module built from prebuilt PTX and
// one compiled here are different modules, and a caller that asked for the
// second must not be handed the first because the first happened to be loaded
// already. Leaving the origin out is what made WithoutPrebuilt unreliable.
//
// NoDiskCache partitions it for the same reason, one option later. The bytes
// a disk hit produces are the bytes NVRTC would have produced -- that is what
// the disk key guarantees -- so the two modules agree whenever the file is
// what it claims to be. But a caller who passed NoDiskCache is precisely the
// one not taking that on trust, and an option answered from a module loaded
// the other way is an option that quietly does nothing. The cost is one
// duplicate module in a context that builds the same kernel both ways, which
// is a test or a benchmark and nothing else.
func cacheKey(ctx *cuda.Context, req Request) string {
	origin := "nvrtc"
	if req.NoDiskCache {
		origin = "nvrtc-nodisk"
	}
	if req.PTX != nil {
		origin = "prebuilt\x00" + req.PTXArch
	}
	sum := sha256.Sum256([]byte(req.Src + "\x00" + ctx.Arch() + "\x00" + origin))
	return hex.EncodeToString(sum[:])
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
func load(ctx *cuda.Context, key string, req Request) (*Result, error) {
	// The 12 hex digits are a convenience for reading file names, not
	// collision resistance to hang a module lookup on.
	short := key[:12]
	// compileErr distinguishes a failure of ours from one of the context's, so
	// that the message says which step actually gave up. It is written under
	// the context's lock and read after LoadPTXCached has returned, so the two
	// never race.
	var compileErr error

	mod, err := ctx.LoadPTXCached(key, func() ([]byte, error) {
		if req.PTX != nil {
			// The dump still happens on this path: the point of the directory
			// is to show what ran, and "nothing was written because nothing
			// was compiled" is the least useful answer it could give while
			// chasing a kernel that misbehaves only when prebuilt.
			dump(req.CacheDir, req.Name, short, req.Src, req.PTX)
			remember(key, artifacts{ptx: req.PTX, prebuilt: true, arch: req.PTXArch})
			return req.PTX, nil
		}
		// The persistent cache sits here, between the artifact a generator
		// produced and the compiler: a hit is indistinguishable from what
		// NVRTC would have returned, because it is what NVRTC did return.
		// The log is not stored with it -- it is a compiler's warnings about
		// a source that compiled, and remembering them across processes would
		// mean a second run reporting warnings nothing emitted this time.
		var disk string
		if !req.NoDiskCache {
			disk = diskKey(req.Src, ctx.Arch(), nvrtcTag())
			if ptx := readDisk(disk); ptx != nil {
				dump(req.CacheDir, req.Name, short, req.Src, ptx)
				rememberKeepingLog(key, artifacts{ptx: ptx, arch: ctx.Arch()})
				return ptx, nil
			}
		}

		var opts []cuda.CompileOption
		if req.FastMath {
			opts = append(opts, cuda.WithFastMath())
		}
		p, err := cuda.Compile(req.Src, req.Name+".cu", ctx.Arch(), opts...)
		if err != nil {
			// Dump the source even though there is no PTX: a kernel that does
			// not compile is exactly the one worth reading. The error is
			// wrapped rather than replaced so that a caller can still recover
			// the *cuda.CompileError and read the compiler log out of it.
			dump(req.CacheDir, req.Name, short, req.Src, nil)
			compileErr = fmt.Errorf("compiling %s: %w", req.Name, err)
			return nil, compileErr
		}
		if disk != "" {
			writeDisk(disk, p.Bytes)
		}
		dump(req.CacheDir, req.Name, short, req.Src, p.Bytes)
		remember(key, artifacts{ptx: p.Bytes, log: p.Log, arch: ctx.Arch()})
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

	// Every build records what it produced before returning, and the context
	// publishes the module only after the build returns, so a caller that
	// arrives here by a cache hit finds the metadata already recorded. Doing
	// it the other way round -- recording once LoadPTXCached had returned, and
	// so once the lock was released -- let a concurrent hit read nothing.
	a, _ := recall(key)
	return &Result{Func: fn, PTX: a.ptx, Log: a.log, Prebuilt: a.prebuilt, Arch: a.arch}, nil
}

func remember(key string, a artifacts) {
	artifactMu.Lock()
	defer artifactMu.Unlock()
	artifactOf[key] = a
}

// rememberKeepingLog records a build that does not know the compiler log,
// without discarding one an earlier build did know.
//
// A disk hit is that build: the log is not stored with the PTX, and this map
// is keyed on the cache key rather than on the context, so a second context
// reading the file would otherwise replace the first context's entry with a
// blank -- and the first context, whose module cache answers before any of
// this runs, would start reporting no log for a compile that had one. Under
// the lock, because a read followed by a write is not one.
func rememberKeepingLog(key string, a artifacts) {
	artifactMu.Lock()
	defer artifactMu.Unlock()
	if old, ok := artifactOf[key]; ok && old.log != "" {
		a.log = old.log
	}
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
