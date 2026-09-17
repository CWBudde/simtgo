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
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"io/fs"
	"sort"

	"github.com/CWBudde/gocuda/internal/lower"
)

// Unit is one transpiled kernel: the generated CUDA C together with what its
// source says about how it has to be launched.
type Unit = lower.Unit

// GPUPkgPath is the import path of the kernel vocabulary package.
const GPUPkgPath = lower.GPUPkgPath

// Transpile type-checks every Go file in fsys as one package and lowers the
// kernel function named fn to CUDA C.
func Transpile(fsys fs.FS, fn string) (*Unit, error) {
	fset := token.NewFileSet()
	names, err := fs.Glob(fsys, "*.go")
	if err != nil {
		return nil, err
	}
	if len(names) == 0 {
		return nil, fmt.Errorf("simt: no Go sources found")
	}
	sort.Strings(names)

	var files []*ast.File
	for _, name := range names {
		src, err := fs.ReadFile(fsys, name)
		if err != nil {
			return nil, err
		}
		f, err := parser.ParseFile(fset, name, src, parser.SkipObjectResolution)
		if err != nil {
			return nil, fmt.Errorf("simt: %w", err)
		}
		files = append(files, f)
	}

	info := &types.Info{
		Types:      make(map[ast.Expr]types.TypeAndValue),
		Defs:       make(map[*ast.Ident]types.Object),
		Uses:       make(map[*ast.Ident]types.Object),
		Selections: make(map[*ast.SelectorExpr]*types.Selection),
	}
	conf := types.Config{Importer: lower.SynthImporter{GPU: lower.GPUPackage()}}
	if _, err := conf.Check("kernels", fset, files, info); err != nil {
		return nil, fmt.Errorf("simt: type error in kernel source: %w", err)
	}

	var decl *ast.FuncDecl
	for _, f := range files {
		for _, d := range f.Decls {
			if fd, ok := d.(*ast.FuncDecl); ok && fd.Name.Name == fn {
				decl = fd
			}
		}
	}
	if decl == nil {
		return nil, fmt.Errorf("simt: no function %q in kernel source", fn)
	}

	u, diags := lower.Kernel(fset, info, decl)
	if len(diags) > 0 {
		return nil, newUnsupportedError(fset, fn, diags)
	}
	return u, nil
}
