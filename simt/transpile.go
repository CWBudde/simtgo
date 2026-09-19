// Package simt implements the SIMT track: kernels are written as ordinary Go
// functions and lowered to CUDA C, which NVRTC compiles to PTX at run time.
//
// Go's gc compiler has no pluggable code generation backend, so the route CUDA
// Rust takes -- a custom rustc backend emitting PTX -- is not open inside gc.
// It is open outside it, through an LLVM-based Go compiler such as llgo, at
// the price of an out-of-tree toolchain; see the README. What Go does have, and
// Rust does not, is a parser and a type checker in its standard library.
// Translating a well-defined subset of Go at the source level gets most of the
// way there, and keeps kernels runnable on the CPU.
package simt

import (
	"fmt"
	"io/fs"

	"github.com/CWBudde/simtgo/internal/lower"
)

// Unit is one transpiled kernel: the generated CUDA C together with what its
// source says about how it has to be launched.
type Unit = lower.Unit

// Param is one of a kernel's parameters, in the order a launch supplies them.
type Param = lower.Param

// GPUPkgPath is the import path of the kernel vocabulary package.
const GPUPkgPath = lower.GPUPkgPath

// Transpile type-checks every Go file in fsys as one package and lowers the
// kernel function named fn to CUDA C.
//
// Everything the kernel cannot express on a device is reported here, with a
// position, rather than left for NVRTC to complain about in generated code the
// author never wrote.
//
// It takes the same BuildOption type Build does, although only the options
// that change what is emitted mean anything here -- lowering is the first step
// of a build, and WithCacheDir, WithoutPrebuilt and WithoutDiskCache all
// concern the loading that comes after it. Sharing one option type is what
// lets a caller ask what a debug build would generate without a device or a
// toolkit anywhere in reach.
func Transpile(fsys fs.FS, fn string, opts ...BuildOption) (*Unit, error) {
	var o buildOptions
	for _, opt := range opts {
		opt(&o)
	}
	pkg, diags, err := lower.LoadPackage(fsys)
	if err != nil {
		return nil, fmt.Errorf("simt: %w", err)
	}
	if len(diags) > 0 {
		return nil, newUnsupportedError(pkg.Fset, fn, diags)
	}
	u, diags, err := pkg.Kernel(fn, o.lowerOptions()...)
	if err != nil {
		return nil, fmt.Errorf("simt: %w", err)
	}
	if len(diags) > 0 {
		return nil, newUnsupportedError(pkg.Fset, fn, diags)
	}
	return u, nil
}
