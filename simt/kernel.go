package simt

import (
	"fmt"
	"io/fs"

	"github.com/CWBudde/gocuda/cuda"
	"github.com/CWBudde/gocuda/internal/jit"
)

// Kernel is a Go kernel that has been transpiled, compiled and loaded.
type Kernel struct {
	Name   string // the Go function's name, also the C entry point
	Source string // the generated CUDA C
	PTX    []byte // what NVRTC produced
	Log    string // NVRTC's compiler log, usually empty

	// RequiredBlock and SharedBytes are what the kernel's source demands of a
	// launch; see Unit.
	RequiredBlock int
	SharedBytes   int

	fn *cuda.Function
}

// SetCacheDir chooses where the generated .cu and .ptx files are written for
// inspection. Empty disables the dump. The default is ".gocuda-cache".
func SetCacheDir(dir string) { jit.CacheDir = dir }

// Build transpiles the named Go kernel from fsys, compiles it with NVRTC and
// loads it into ctx: the whole SIMT pipeline, Go source to CUDA C to PTX.
func Build(ctx *cuda.Context, fsys fs.FS, name string) (*Kernel, error) {
	u, err := Transpile(fsys, name)
	if err != nil {
		return nil, err
	}
	// A tile larger than the device's limit is rejected by cuModuleLoadData
	// with nothing but a generic error code, so it is worth catching here,
	// where both the total and the kernel's name are still at hand.
	if limit := ctx.MaxSharedMemPerBlock(); limit > 0 && u.SharedBytes > limit {
		return nil, fmt.Errorf("simt: %s declares %d bytes of shared memory, but this device offers %d per block",
			name, u.SharedBytes, limit)
	}
	res, err := jit.Load(ctx, u.Source, name)
	if err != nil {
		return nil, err
	}
	return &Kernel{
		Name:          name,
		Source:        u.Source,
		PTX:           res.PTX,
		Log:           res.Log,
		RequiredBlock: u.RequiredBlock,
		SharedBytes:   u.SharedBytes,
		fn:            res.Func,
	}, nil
}

// Launch runs the kernel over grid blocks of block threads. Arguments are
// given as Go values -- device slices, float32, int32 -- in the same order as
// the Go kernel's parameters, minus the gpu.Ctx.
//
// A kernel that declared its block size with gpu.Ctx.AssumeBlockDim is refused
// at any other size: its shared tiles are sized for that one geometry, so a
// different block would stage the wrong number of samples and read past them.
func (k *Kernel) Launch(grid, block int, args ...any) error {
	if k.RequiredBlock != 0 && block != k.RequiredBlock {
		return fmt.Errorf("simt: %s requires block == %d, got %d", k.Name, k.RequiredBlock, block)
	}
	flat, err := cuda.BuildArgs(args...)
	if err != nil {
		return err
	}
	return k.fn.Launch(cuda.D1(grid), cuda.D1(block), 0, flat...)
}

// LaunchN runs the kernel over enough blocks to cover n threads. Kernels guard
// against the ragged tail themselves, exactly as in CUDA C.
func (k *Kernel) LaunchN(n, block int, args ...any) error {
	return k.Launch((n+block-1)/block, block, args...)
}
