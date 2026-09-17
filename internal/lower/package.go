package lower

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/scanner"
	"go/token"
	"go/types"
	"io/fs"
	"sort"
	"strings"
)

// A Package is a parsed and type-checked set of kernel sources.
//
// It exists because both callers need to ask what a package contains before
// lowering anything: the generator has to enumerate the kernels it will emit
// artifacts for, and the vet tool has to check all of them. Loading once also
// removes a real cost -- Transpile used to re-parse and re-type-check every
// file for each kernel it was asked about.
// Info is nil when loading produced diagnostics; Fset and Files are always
// populated, so a caller can resolve the positions of the diagnostics it got.
type Package struct {
	Fset  *token.FileSet
	Info  *types.Info
	Files []*ast.File
}

// LoadPackage parses every non-test Go file at the root of fsys and type-checks
// them as one package.
//
// Test files are excluded deliberately. A kernel package's *_test.go is matched
// by both go:embed's kernels/*.go and this glob, and would then join the
// type-check and fail on its import of "testing" -- a kernel package could not
// have tests at all.
func LoadPackage(fsys fs.FS) (*Package, []Diagnostic, error) {
	names, err := fs.Glob(fsys, "*.go")
	if err != nil {
		return nil, nil, fmt.Errorf("listing kernel sources: %w", err)
	}
	sort.Strings(names)

	fset := token.NewFileSet()
	var files []*ast.File
	var diags []Diagnostic
	for _, name := range names {
		if strings.HasSuffix(name, "_test.go") {
			continue
		}
		src, err := fs.ReadFile(fsys, name)
		if err != nil {
			return nil, nil, fmt.Errorf("reading %s: %w", name, err)
		}
		// Comments are parsed because //gocuda:ignore lives in one, and an
		// opt-out the analyzer honours but the generator does not is worse
		// than no opt-out at all: it passes the check and then fails the
		// build. The analysis framework hands over a commented AST, so this is
		// what makes the two agree.
		f, err := parser.ParseFile(fset, name, src, parser.SkipObjectResolution|parser.ParseComments)
		if err != nil {
			return &Package{Fset: fset, Files: files}, append(diags, parseDiagnostics(err)...), nil
		}
		files = append(files, f)
	}
	if len(files) == 0 {
		return nil, nil, fmt.Errorf("no Go source files in the kernel package")
	}

	// The import rule is checked before type-checking so that it is reported
	// as itself, rather than as the type checker's failure to resolve a
	// package the importer was never going to serve.
	for _, f := range files {
		diags = append(diags, CheckImports(f)...)
	}
	if len(diags) > 0 {
		return &Package{Fset: fset, Files: files}, tidy(diags), nil
	}

	info := &types.Info{
		Types:      make(map[ast.Expr]types.TypeAndValue),
		Defs:       make(map[*ast.Ident]types.Object),
		Uses:       make(map[*ast.Ident]types.Object),
		Selections: make(map[*ast.SelectorExpr]*types.Selection),
	}
	// Error collects every type error instead of stopping at the first, which
	// is the difference between telling an author about one mistake and
	// telling them about their kernel.
	conf := types.Config{
		Importer: SynthImporter{GPU: GPUPackage()},
		Error: func(err error) {
			if te, ok := err.(types.Error); ok {
				diags = append(diags, Diagnostic{Pos: te.Pos, Msg: te.Msg})
				return
			}
			diags = append(diags, Diagnostic{Msg: err.Error()})
		},
	}
	if _, err := conf.Check("kernels", fset, files, info); err != nil {
		if len(diags) == 0 {
			diags = append(diags, Diagnostic{Msg: err.Error()})
		}
		return &Package{Fset: fset, Files: files}, tidy(diags), nil
	}
	return &Package{Fset: fset, Info: info, Files: files}, nil, nil
}

// Names lists the kernels the package declares, in source order.
func (p *Package) Names() []string {
	var out []string
	for _, f := range p.Files {
		if Ignored(f.Doc) {
			continue
		}
		for _, d := range f.Decls {
			if fd, ok := d.(*ast.FuncDecl); ok && IsKernelDecl(p.Info, fd) {
				out = append(out, fd.Name.Name)
			}
		}
	}
	return out
}

// Decl returns the declaration of the named function, or nil.
func (p *Package) Decl(name string) *ast.FuncDecl {
	for _, f := range p.Files {
		for _, d := range f.Decls {
			if fd, ok := d.(*ast.FuncDecl); ok && fd.Name.Name == name {
				return fd
			}
		}
	}
	return nil
}

// Kernel lowers the named kernel.
func (p *Package) Kernel(name string) (*Unit, []Diagnostic, error) {
	fd := p.Decl(name)
	if fd == nil {
		declared := p.Names()
		if len(declared) == 0 {
			return nil, nil, fmt.Errorf("no kernel named %q; this package declares none", name)
		}
		return nil, nil, fmt.Errorf("no kernel named %q; this package declares %s",
			name, strings.Join(declared, ", "))
	}
	u, diags := Kernel(p.Fset, p.Info, fd)
	return u, diags, nil
}

// parseDiagnostics turns a scanner.ErrorList into one diagnostic per error.
// Its own Error method reports only the first, plus a count of the rest.
func parseDiagnostics(err error) []Diagnostic {
	var list scanner.ErrorList
	if !asErrorList(err, &list) {
		return []Diagnostic{{Msg: err.Error()}}
	}
	out := make([]Diagnostic, 0, len(list))
	for _, e := range list {
		out = append(out, Diagnostic{Msg: e.Pos.String() + ": " + e.Msg})
	}
	return out
}

func asErrorList(err error, out *scanner.ErrorList) bool {
	if list, ok := err.(scanner.ErrorList); ok {
		*out = list
		return true
	}
	return false
}
