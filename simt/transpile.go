// Package simt implements the SIMT track: kernels are written as ordinary Go
// functions and lowered to CUDA C, which NVRTC compiles to PTX at run time.
//
// Go's gc compiler has no pluggable code generation backend, so the route CUDA
// Rust takes -- a custom rustc backend emitting PTX -- is not open to Go. What
// Go does have, and Rust does not, is a parser and a type checker in its
// standard library. Translating a well-defined subset of Go at the source
// level gets most of the way there, and keeps kernels runnable on the CPU.
package simt

import (
	"fmt"
	"io/fs"

	"github.com/CWBudde/gocuda/internal/lower"
)

// Unit is one transpiled kernel: the generated CUDA C together with what its
// source says about how it has to be launched.
type Unit = lower.Unit

// GPUPkgPath is the import path of the kernel vocabulary package.
const GPUPkgPath = lower.GPUPkgPath

// Transpile type-checks every Go file in fsys as one package and lowers the
// kernel function named fn to CUDA C.
//
// Everything the kernel cannot express on a device is reported here, with a
// position, rather than left for NVRTC to complain about in generated code the
// author never wrote.
func Transpile(fsys fs.FS, fn string) (*Unit, error) {
	pkg, diags, err := lower.LoadPackage(fsys)
	if err != nil {
		return nil, fmt.Errorf("simt: %w", err)
	}
	if len(diags) > 0 {
		return nil, newUnsupportedError(pkg.Fset, fn, diags)
	}
	u, diags, err := pkg.Kernel(fn)
	if err != nil {
		return nil, fmt.Errorf("simt: %w", err)
	}
	if len(diags) > 0 {
		return nil, newUnsupportedError(pkg.Fset, fn, diags)
	}
	return u, nil
}
