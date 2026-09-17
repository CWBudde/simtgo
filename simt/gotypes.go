package simt

import (
	"go/token"
	"go/types"
)

// GPUPkgPath is the import path of the kernel vocabulary package.
const GPUPkgPath = "github.com/CWBudde/gocuda/gpu"

// gpuPackage builds a types.Package describing package gpu by hand.
//
// Kernel sources are type-checked at run time, where neither the module graph
// nor compiled export data is guaranteed to be available. Since the kernel
// vocabulary is small and fixed, synthesising it is more robust than asking
// the toolchain to import the real package: nothing outside the standard
// library is needed, and the transpiler's view of gpu cannot drift from the
// symbols it knows how to lower.
func gpuPackage() *types.Package {
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

	for _, name := range []string{"ThreadIdx", "BlockIdx", "BlockDim", "GridDim", "GlobalID"} {
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

	pkg.MarkComplete()
	return pkg
}

// synthImporter serves package gpu from memory and refuses everything else,
// which keeps the supported kernel vocabulary explicit.
type synthImporter struct{ gpu *types.Package }

func (im synthImporter) Import(path string) (*types.Package, error) {
	if path == GPUPkgPath {
		return im.gpu, nil
	}
	return nil, &unsupportedImportError{path: path}
}

type unsupportedImportError struct{ path string }

func (e *unsupportedImportError) Error() string {
	return "simt: kernels may not import " + e.path + " (only " + GPUPkgPath + " is available on the device)"
}
