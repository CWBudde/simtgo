// Package jit compiles generated CUDA C and loads it, for both the SIMT and
// the tile track.
package jit

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"

	"github.com/CWBudde/gocuda/cuda"
)

// CacheDir receives the generated CUDA and the PTX NVRTC produced from it, so
// that what actually ran can be read. Empty disables the dump.
var CacheDir = ".gocuda-cache"

// Result is a compiled and loaded kernel.
type Result struct {
	Func *cuda.Function
	PTX  []byte
	Log  string
}

// artifacts is what this package remembers about a module besides the module
// itself. The context stores it opaquely next to the module and returns it on
// every cache hit, so the second Load of a source reports the PTX and the
// compiler log exactly as the first one did.
type artifacts struct {
	ptx []byte
	log string
}

// Load compiles src with NVRTC and returns its entry point. Identical sources
// are compiled once per context.
//
// The cache deliberately lives on the context rather than in a package-level
// map here. A CUmodule is only valid inside the context it was loaded into,
// while a package variable lives as long as the process: a shared cache
// therefore hands a module of a long-released context to whoever compiles the
// same source next, and every generated tile kernel is called "fused", so
// those collisions are the rule rather than the exception. Splitting it this
// way keeps the compile and dump policy here, where the file names and the
// NVRTC options are decided, and puts the module's lifetime in the one place
// that can end it: the context, which unloads what it owns on Close.
func Load(ctx *cuda.Context, src, name string) (*Result, error) {
	// The architecture is folded into the hash even though a context has only
	// one, so that dumps of the same source built for different devices do not
	// overwrite each other. The full sum keys the cache: the 12 hex digits
	// below are a convenience for reading file names, not 48 bits of collision
	// resistance to hang a module lookup on.
	sum := sha256.Sum256([]byte(src + "\x00" + ctx.Arch()))
	key := hex.EncodeToString(sum[:])
	short := key[:12]

	// compileErr distinguishes a failure of ours from one of the context's, so
	// that the message says which step actually gave up. It is written under
	// the context's lock and read after LoadPTXCached has returned, so the two
	// never race.
	var compileErr error
	mod, extra, err := ctx.LoadPTXCached(key, func() ([]byte, any, error) {
		p, err := cuda.Compile(src, name+".cu", ctx.Arch())
		if err != nil {
			// Dump the source even though there is no PTX: a kernel that does
			// not compile is exactly the one worth reading. The error is
			// wrapped rather than replaced so that a caller can still recover
			// the *cuda.CompileError and read the compiler log out of it.
			dump(name, short, src, nil)
			compileErr = fmt.Errorf("compiling %s: %w", name, err)
			return nil, nil, compileErr
		}
		dump(name, short, src, p.Bytes)
		return p.Bytes, &artifacts{ptx: p.Bytes, log: p.Log}, nil
	})
	if err != nil {
		if compileErr != nil {
			return nil, compileErr
		}
		return nil, fmt.Errorf("loading %s: %w", name, err)
	}

	fn, err := mod.Function(name)
	if err != nil {
		return nil, err
	}
	res := &Result{Func: fn}
	if a, ok := extra.(*artifacts); ok {
		res.PTX, res.Log = a.ptx, a.log
	}
	return res, nil
}

// dump writes the generated sources for inspection. A failure here must not
// sink a launch, so errors are deliberately dropped.
func dump(name, hash, src string, ptx []byte) {
	if CacheDir == "" {
		return
	}
	if err := os.MkdirAll(CacheDir, 0o755); err != nil {
		return
	}
	base := filepath.Join(CacheDir, fmt.Sprintf("%s-%s", name, hash))
	_ = os.WriteFile(base+".cu", []byte(src), 0o644)
	if ptx != nil {
		_ = os.WriteFile(base+".ptx", ptx, 0o644)
	}
}
