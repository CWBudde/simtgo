// Package simtcheck provides the analyzer behind "gocuda vet": it reports Go
// constructs that a CUDA kernel cannot use.
//
// Without it a kernel that cannot be lowered compiles like any other Go
// function and fails when main() runs, which is the single largest obstacle to
// anyone else using this library. The analyzer runs the same lowering the
// transpiler does -- not a second opinion about it -- so what it accepts and
// what simt.Transpile accepts cannot drift apart.
package simtcheck

import (
	"go/ast"
	"go/types"
	"strings"

	"golang.org/x/tools/go/analysis"

	"github.com/CWBudde/gocuda/internal/lower"
)

// Doc is the analyzer's documentation, also shown by "gocuda vet -help".
const Doc = `report Go constructs a CUDA kernel cannot use

A kernel is a function whose first parameter is a gpu.Ctx. This analyzer lowers
each one exactly as the transpiler does and reports whatever the transpiler
would refuse: unsupported statements and types, calls to functions that have no
device equivalent, shared memory declared with a non-constant size, and imports
other than the kernel vocabulary package.

Without it those refusals only surface when the kernel is built at run time.

A function that takes a gpu.Ctx but is deliberately never lowered can be
excluded with a //gocuda:ignore line in its doc comment; the same line in a
file's package comment excludes the whole file. A helper that takes a gpu.Ctx
because it needs the thread's position, and is meant to be called by a kernel,
says so with //gocuda:device instead.`

// Analyzer is the gocuda vet check. It is exported so that it can be composed
// into someone else's multichecker.
var Analyzer = &analysis.Analyzer{
	Name: "simt",
	Doc:  Doc,
	URL:  "https://github.com/CWBudde/gocuda",
	Run:  run,
	// The traversal is a flat loop over each file's declarations, so the AST
	// index that passes/inspect builds would be pure cost for the many
	// packages that contain no kernel at all.
	Requires: nil,
	// Roughly half the lowering reads go/types results and relies on them
	// being there. A kernel that does not type-check is already a build
	// failure, and the compiler will have described it better.
	RunDespiteErrors: false,
}

func run(pass *analysis.Pass) (any, error) {
	// A package that does not import the kernel vocabulary cannot contain a
	// kernel. Imports() is a pre-computed slice of the direct imports, so for
	// almost every package in a module this is the whole of the work.
	if !importsGPU(pass.Pkg) {
		return nil, nil
	}

	var files []*ast.File
	var kernels []*ast.FuncDecl
	for _, f := range pass.Files {
		if isTestFile(pass, f) || lower.Ignored(f.Doc) {
			continue
		}
		files = append(files, f)
		for _, d := range f.Decls {
			// IsKernelDecl applies the opt-out itself, so that what this
			// reports and what the generator lowers cannot disagree.
			fd, ok := d.(*ast.FuncDecl)
			if ok && lower.IsKernelDecl(pass.TypesInfo, fd) {
				kernels = append(kernels, fd)
			}
		}
	}
	// The import rule is a property of the package, because the transpiler
	// type-checks every file of it together -- an offending import in a file
	// with no kernel in it still breaks every kernel beside it. It is only
	// applied to packages that actually contain a kernel, since otherwise it
	// would flag every package in the module.
	if len(kernels) == 0 {
		return nil, nil
	}
	for _, f := range files {
		report(pass, lower.CheckImports(f))
		// Also a property of the package: the function carrying a pointless
		// //gocuda:device may be one no kernel reaches, so nothing in the
		// per-kernel lowering below would ever look at it.
		report(pass, lower.CheckDeviceMarkers(pass.TypesInfo, f))
	}
	for _, fd := range kernels {
		// The Unit is discarded: this runs the real lowering rather than a
		// cheaper approximation of it, so that a construct the emitter would
		// mistranslate is a construct the analyzer sees.
		_, diags := lower.Kernel(pass.Fset, pass.TypesInfo, pass.Files, fd)
		report(pass, diags)
	}
	return nil, nil
}

func report(pass *analysis.Pass, diags []lower.Diagnostic) {
	for _, d := range diags {
		// An analysis.Diagnostic carries its own position and category, so the
		// message must supply neither.
		pass.Report(analysis.Diagnostic{Pos: d.Pos, Category: "simt", Message: d.Msg})
	}
}

func importsGPU(p *types.Package) bool {
	for _, imp := range p.Imports() {
		if imp.Path() == lower.GPUPkgPath {
			return true
		}
	}
	return false
}

// isTestFile keeps the internal test variant of a kernel package quiet. go vet
// analyzes "pkg [pkg.test]" as a unit of its own, containing the kernels and
// the test files together, so without this every kernel package that has tests
// would be told it may not import "testing".
func isTestFile(pass *analysis.Pass, f *ast.File) bool {
	return strings.HasSuffix(pass.Fset.Position(f.Pos()).Filename, "_test.go")
}
