package lower

import (
	"go/ast"
	"go/token"
	"go/types"
	"strconv"
	"strings"
)

// GPUPkgPath is the import path of the kernel vocabulary package.
const GPUPkgPath = "github.com/CWBudde/gocuda/gpu"

// GPUPackage builds a types.Package describing package gpu by hand.
//
// Kernel sources are type-checked at run time, where neither the module graph
// nor compiled export data is guaranteed to be available. Since the kernel
// vocabulary is small and fixed, synthesising it is more robust than asking
// the toolchain to import the real package: nothing outside the standard
// library is needed, and the transpiler's view of gpu cannot drift from the
// symbols it knows how to lower.
func GPUPackage() *types.Package {
	pkg := types.NewPackage(GPUPkgPath, "gpu")
	scope := pkg.Scope()

	intT := types.Typ[types.Int]
	f32 := types.Typ[types.Float32]
	f32Slice := types.NewSlice(f32)

	ctxName := types.NewTypeName(token.NoPos, pkg, "Ctx", nil)
	ctx := types.NewNamed(ctxName, types.NewStruct(nil, nil), nil)
	scope.Insert(ctxName)

	recv := types.NewVar(token.NoPos, pkg, "c", ctx)
	method := func(name string, params, results []*types.Var) {
		sig := types.NewSignatureType(recv, nil, nil,
			types.NewTuple(params...), types.NewTuple(results...), false)
		ctx.AddMethod(types.NewFunc(token.NoPos, pkg, name, sig))
	}
	ret := func(t types.Type) []*types.Var {
		return []*types.Var{types.NewVar(token.NoPos, pkg, "", t)}
	}

	// One accessor per axis rather than a tuple: each is a single CUDA
	// built-in, and the emitter's table maps a method to exactly one C
	// expression. The unsuffixed names are the x axis, which is CUDA's own
	// spelling and what every one-dimensional kernel already uses.
	for _, name := range []string{
		"ThreadIdx", "ThreadIdxY", "ThreadIdxZ",
		"BlockIdx", "BlockIdxY", "BlockIdxZ",
		"BlockDim", "BlockDimY", "BlockDimZ",
		"GridDim", "GridDimY", "GridDimZ",
		"GlobalID", "GlobalIDX", "GlobalIDY", "GlobalIDZ",
	} {
		method(name, nil, ret(intT))
	}
	method("SyncThreads", nil, nil)
	method("SharedF32", []*types.Var{types.NewVar(token.NoPos, pkg, "n", intT)}, ret(f32Slice))
	// AssumeBlockDim yields nothing on either backend: on the CPU it is a
	// check, on the device it lowers to no code at all. It is declared here
	// all the same, because the transpiler has to see the call to learn which
	// block size the kernel was written for.
	method("AssumeBlockDim", []*types.Var{types.NewVar(token.NoPos, pkg, "n", intT)}, nil)

	fn := func(name string, arity int) {
		params := make([]*types.Var, arity)
		for i := range params {
			params[i] = types.NewVar(token.NoPos, pkg, "x", f32)
		}
		sig := types.NewSignatureType(nil, nil, nil,
			types.NewTuple(params...), types.NewTuple(ret(f32)...), false)
		scope.Insert(types.NewFunc(token.NoPos, pkg, name, sig))
	}
	for _, name := range []string{"Sqrt", "Abs", "Sin", "Cos", "Exp", "Log"} {
		fn(name, 1)
	}
	for _, name := range []string{"Hypot", "Fmin", "Fmax"} {
		fn(name, 2)
	}

	// The double-precision helpers, whose signatures fn cannot express. They
	// are declared unconditionally: whether a kernel may reach one is decided
	// when it is lowered, not by what the type checker can see, so refusing
	// them here would report a missing symbol instead of a missing directive.
	f64 := types.Typ[types.Float64]
	fn64 := func(name string, arity int) {
		params := make([]*types.Var, arity)
		for i := range params {
			params[i] = types.NewVar(token.NoPos, pkg, "x", f64)
		}
		sig := types.NewSignatureType(nil, nil, nil,
			types.NewTuple(params...), types.NewTuple(ret(f64)...), false)
		scope.Insert(types.NewFunc(token.NoPos, pkg, name, sig))
	}
	for _, name := range []string{"Sqrt64", "Abs64", "Sin64", "Cos64", "Exp64", "Log64"} {
		fn64(name, 1)
	}
	for _, name := range []string{"Hypot64", "Fmin64", "Fmax64"} {
		fn64(name, 2)
	}

	i32 := types.Typ[types.Int32]
	i32Slice := types.NewSlice(i32)

	// atomic declares one of the read-modify-write helpers. The shape is
	// always the same: the buffer, an index into it, then one value of the
	// element type (two for compare-and-swap), returning the value the element
	// held before -- which is what the CUDA built-in returns.
	//
	// It is a second builder rather than a generalisation of fn above, because
	// fn's nine callers are the one thing in this file that must not drift,
	// and rewriting them to thread an element type through would put every
	// float32 signature at risk to describe six that are not float32 at all.
	atomic := func(name string, slice, elem types.Type, vals int) {
		params := []*types.Var{
			types.NewVar(token.NoPos, pkg, "s", slice),
			types.NewVar(token.NoPos, pkg, "i", intT),
		}
		for range vals {
			params = append(params, types.NewVar(token.NoPos, pkg, "v", elem))
		}
		sig := types.NewSignatureType(nil, nil, nil,
			types.NewTuple(params...), types.NewTuple(ret(elem)...), false)
		scope.Insert(types.NewFunc(token.NoPos, pkg, name, sig))
	}
	atomic("AtomicAddF32", f32Slice, f32, 1)
	atomic("AtomicAddI32", i32Slice, i32, 1)
	atomic("AtomicMinI32", i32Slice, i32, 1)
	atomic("AtomicMaxI32", i32Slice, i32, 1)
	atomic("AtomicExchI32", i32Slice, i32, 1)
	atomic("AtomicCASI32", i32Slice, i32, 2)

	pkg.MarkComplete()
	return pkg
}

// SynthImporter serves package gpu from memory and refuses everything else,
// which keeps the supported kernel vocabulary explicit.
type SynthImporter struct{ GPU *types.Package }

func (im SynthImporter) Import(path string) (*types.Package, error) {
	if path == GPUPkgPath {
		return im.GPU, nil
	}
	return nil, &unsupportedImportError{path: path}
}

type unsupportedImportError struct{ path string }

func (e *unsupportedImportError) Error() string {
	return importRefusal(e.path)
}

func importRefusal(path string) string {
	return "kernels may not import " + path + " (only " + GPUPkgPath + " is available on the device)"
}

// IgnoreDirective opts a declaration, or a whole file, out of being treated as
// a kernel.
const IgnoreDirective = "//gocuda:ignore"

// DeviceDirective marks a gpu.Ctx-taking function as a device function rather
// than a kernel.
//
// It exists because the signature rule cannot tell the two apart: a helper
// that wants the thread's position takes a gpu.Ctx, and taking one is what
// makes a function a kernel. Until this directive the only way to say so was
// IgnoreDirective, which says what the function is *not* -- it opts out of
// being generated, which is also what a Ctx-taking function nobody lowers
// wants. The two now mean different things, and the difference is legible at
// the declaration: //gocuda:ignore is "leave this alone", //gocuda:device is
// "this runs on the device, as a __device__ function".
const DeviceDirective = "//gocuda:device"

// Float64Directive opts a kernel into double precision.
//
// It is opt-in rather than simply allowed because the cost is invisible in the
// source: sm_75 runs float64 at 1/32 the float32 rate, so a kernel that gained
// a double by accident -- an untyped constant binding wider than intended, say
// -- would still be correct and thirty times slower. The directive makes that
// a decision somebody wrote down.
const Float64Directive = "//gocuda:float64"

// Ignored reports whether doc carries the opt-out directive.
//
// It lives here, beside the rule it opts out of, because every caller that
// decides what a kernel is has to agree: an opt-out honoured by the vet tool
// but not by the generator would pass the check and then fail generation,
// which is worse than having no opt-out at all.
//
// go vet has no //nolint equivalent -- there is no general way for a user to
// silence one of its diagnostics -- so the check has to bring its own.
func Ignored(doc *ast.CommentGroup) bool { return hasDirective(doc, IgnoreDirective) }

// DeviceMarked reports whether doc declares the function a device function.
//
// Unlike Ignored it is not honoured on a file's package comment. A file-wide
// "everything here takes a Ctx and none of it is a kernel" is what
// //gocuda:ignore already says; the point of this directive is that it names
// one declaration.
func DeviceMarked(doc *ast.CommentGroup) bool { return hasDirective(doc, DeviceDirective) }

// Float64Enabled reports whether doc carries the double-precision opt-in.
func Float64Enabled(doc *ast.CommentGroup) bool { return hasDirective(doc, Float64Directive) }

// hasDirective reports whether doc carries the line name, alone or followed by
// a space and an explanation. A directive is a whole line, so a mention of it
// inside a sentence is prose about the directive rather than a use of it.
func hasDirective(doc *ast.CommentGroup, name string) bool {
	if doc == nil {
		return false
	}
	for _, c := range doc.List {
		if c.Text == name || strings.HasPrefix(c.Text, name+" ") {
			return true
		}
	}
	return false
}

// CheckImports refuses every import a kernel source may not have.
//
// When simt type-checks kernel sources itself the rule is enforced by
// SynthImporter, which simply cannot resolve anything else. An analyzer is
// handed a package the real compiler already built, where importing "math" is
// perfectly legal Go, so the rule has to exist as a check in its own right --
// and it has to be this one, not a second opinion about it.
func CheckImports(f *ast.File) []Diagnostic {
	var diags []Diagnostic
	for _, spec := range f.Imports {
		path, err := strconv.Unquote(spec.Path.Value)
		if err != nil || path == GPUPkgPath {
			continue
		}
		diags = append(diags, Diagnostic{Pos: spec.Path.Pos(), Msg: importRefusal(path)})
	}
	return diags
}

// IsKernelDecl reports whether fd is a kernel: a plain function whose first
// parameter is a gpu.Ctx and whose doc comment claims it is something else.
//
// Taking a Ctx is what a kernel is for, and nothing else in this repository
// does it at the top level -- the CPU emulator is driven by function literals,
// which are not declarations. Making the marker the signature rather than a
// comment is deliberate: an opt-in directive that someone forgets restores
// exactly the "compiles fine, dies in main()" failure this phase removes. The
// two directives that take it back both say so at the declaration:
// //gocuda:ignore means "never lowered", //gocuda:device means "lowered, but
// as a __device__ function a kernel calls".
func IsKernelDecl(info *types.Info, fd *ast.FuncDecl) bool {
	if Ignored(fd.Doc) || DeviceMarked(fd.Doc) {
		return false
	}
	return takesCtx(info, fd)
}

// takesCtx reports whether fd has the signature a kernel has, before any
// directive has had its say.
func takesCtx(info *types.Info, fd *ast.FuncDecl) bool {
	if fd.Recv != nil || fd.Body == nil || fd.Type.Params == nil || len(fd.Type.Params.List) == 0 {
		return false
	}
	first := fd.Type.Params.List[0]
	if len(first.Names) == 0 {
		return false
	}
	obj := info.Defs[first.Names[0]]
	return obj != nil && IsCtx(obj.Type())
}

// CheckDeviceMarkers refuses //gocuda:device on a function that takes no
// gpu.Ctx.
//
// There the directive changes nothing: such a function is already an ordinary
// helper, and already lowers to a __device__ function the moment a kernel
// calls it. A marker that is sometimes load-bearing and sometimes decoration
// is read as decoration everywhere, and the one place it is load-bearing --
// keeping a Ctx-taking helper from being generated as a kernel in its own
// right -- is exactly where being ignored would hurt.
//
// It is a package-wide check, like CheckImports, rather than something the
// lowering notices: a marked function no kernel reaches is never lowered at
// all, so a check that lived in deviceFunc would pass over the case where the
// mistake is easiest to make.
func CheckDeviceMarkers(info *types.Info, f *ast.File) []Diagnostic {
	var diags []Diagnostic
	for _, d := range f.Decls {
		fd, ok := d.(*ast.FuncDecl)
		if !ok || !DeviceMarked(fd.Doc) || takesCtx(info, fd) {
			continue
		}
		diags = append(diags, Diagnostic{
			Pos: fd.Pos(),
			Msg: DeviceDirective + " does nothing on " + fd.Name.Name + ", which takes no gpu.Ctx: " +
				"it is already a device function wherever a kernel calls it, and the directive only means anything on a function the signature rule would otherwise make a kernel",
		})
	}
	return diags
}
