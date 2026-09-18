package lower

import (
	"go/ast"
	"go/constant"
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

	// The warp-level vocabulary. It is spelled on Ctx rather than at package
	// level, as the atomics are, because a warp primitive is defined by which
	// thread calls it: the CPU emulator runs a thread per goroutine and only
	// the Ctx says which one this is. See gpu/warp.go.
	i32 := types.Typ[types.Int32]
	u32 := types.Typ[types.Uint32]
	boolT := types.Typ[types.Bool]
	arg := func(name string, t types.Type) *types.Var {
		return types.NewVar(token.NoPos, pkg, name, t)
	}
	method("LaneID", nil, ret(intT))
	// The lane argument is an int and not a uint32 even though CUDA reads it
	// as unsigned, because every other index in a kernel is an int and a
	// reduction loop counts with one. A negative constant is refused when the
	// call is lowered rather than being made unspellable here, so that the
	// refusal can say why.
	for _, name := range []string{"ShuffleF32", "ShuffleXorF32", "ShuffleUpF32", "ShuffleDownF32"} {
		method(name, []*types.Var{arg("v", f32), arg("lane", intT)}, ret(f32))
	}
	for _, name := range []string{"ShuffleI32", "ShuffleXorI32", "ShuffleUpI32", "ShuffleDownI32"} {
		method(name, []*types.Var{arg("v", i32), arg("lane", intT)}, ret(i32))
	}
	method("Ballot", []*types.Var{arg("pred", boolT)}, ret(u32))
	method("Any", []*types.Var{arg("pred", boolT)}, ret(boolT))
	method("All", []*types.Var{arg("pred", boolT)}, ret(boolT))
	method("ActiveMask", nil, ret(u32))
	method("SyncWarp", nil, nil)

	// WarpSize is a constant on both sides, so go/types folds it and the
	// emitter never sees the selector at all -- which is the point: CUDA's own
	// warpSize is a variable, and a kernel that sizes a shared tile against it
	// would not compile.
	scope.Insert(types.NewConst(token.NoPos, pkg, "WarpSize",
		types.Typ[types.UntypedInt], constant.MakeInt64(32)))

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
// parameter is a gpu.Ctx.
//
// Taking a Ctx is what a kernel is for, and nothing else in this repository
// does it at the top level -- the CPU emulator is driven by function literals,
// which are not declarations. Making the marker the signature rather than a
// comment is deliberate: an opt-in directive that someone forgets restores
// exactly the "compiles fine, dies in main()" failure this phase removes.
func IsKernelDecl(info *types.Info, fd *ast.FuncDecl) bool {
	if fd.Recv != nil || fd.Body == nil || fd.Type.Params == nil || len(fd.Type.Params.List) == 0 {
		return false
	}
	if Ignored(fd.Doc) {
		return false
	}
	first := fd.Type.Params.List[0]
	if len(first.Names) == 0 {
		return false
	}
	obj := info.Defs[first.Names[0]]
	return obj != nil && IsCtx(obj.Type())
}
