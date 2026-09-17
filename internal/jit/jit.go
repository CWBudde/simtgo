// Package jit compiles generated CUDA C and loads it, for both the SIMT and
// the tile track.
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

// CacheDir receives the generated CUDA and the PTX NVRTC produced from it, so
// that what actually ran can be read. Empty disables the dump.
var CacheDir = ".gocuda-cache"

var modules struct {
	sync.Mutex
	byHash map[string]*cuda.Module
}

// Result is a compiled and loaded kernel.
type Result struct {
	Func *cuda.Function
	PTX  []byte
	Log  string
}

// Load compiles src with NVRTC and returns its entry point. Identical sources
// are compiled once per process.
func Load(ctx *cuda.Context, src, name string) (*Result, error) {
	sum := sha256.Sum256([]byte(src + "\x00" + ctx.Arch()))
	hash := hex.EncodeToString(sum[:])[:12]

	modules.Lock()
	defer modules.Unlock()
	if modules.byHash == nil {
		modules.byHash = map[string]*cuda.Module{}
	}

	res := &Result{}
	mod, cached := modules.byHash[hash]
	if !cached {
		ptx, log, err := cuda.Compile(src, name+".cu", ctx.Arch())
		if err != nil {
			dump(name, hash, src, nil)
			return nil, fmt.Errorf("compiling %s: %w", name, err)
		}
		res.PTX, res.Log = ptx, log
		dump(name, hash, src, ptx)
		if mod, err = ctx.LoadPTX(ptx); err != nil {
			return nil, fmt.Errorf("loading %s: %w", name, err)
		}
		modules.byHash[hash] = mod
	}

	fn, err := mod.Function(name)
	if err != nil {
		return nil, err
	}
	res.Func = fn
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
