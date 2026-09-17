package simt

import (
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

	fn *cuda.Function
}

// SetCacheDir chooses where the generated .cu and .ptx files are written for
// inspection. Empty disables the dump. The default is ".gocuda-cache".
func SetCacheDir(dir string) { jit.CacheDir = dir }

// Build transpiles the named Go kernel from fsys, compiles it with NVRTC and
// loads it into ctx: the whole SIMT pipeline, Go source to CUDA C to PTX.
func Build(ctx *cuda.Context, fsys fs.FS, name string) (*Kernel, error) {
	src, err := Transpile(fsys, name)
	if err != nil {
		return nil, err
	}
	res, err := jit.Load(ctx, src, name)
	if err != nil {
		return nil, err
	}
	return &Kernel{Name: name, Source: src, PTX: res.PTX, Log: res.Log, fn: res.Func}, nil
}

// Launch runs the kernel over grid blocks of block threads. Arguments are
// given as Go values -- device slices, float32, int32 -- in the same order as
// the Go kernel's parameters, minus the gpu.Ctx.
func (k *Kernel) Launch(grid, block int, args ...any) error {
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
