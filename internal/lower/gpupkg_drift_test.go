package lower

import (
	"go/types"
	"testing"

	"golang.org/x/tools/go/packages"
)

// TestGPUPackageMatchesReal keeps the hand-written model of package gpu in step
// with the real one.
//
// Kernel sources are type-checked at run time, where neither the module graph
// nor compiled export data is guaranteed to be available, so GPUPackage builds
// gpu by hand. Nothing else makes the two agree: adding a function to gpu and
// forgetting this file would leave kernels able to call it as plain Go and
// unable to transpile, with the failure only appearing at run time.
func TestGPUPackageMatchesReal(t *testing.T) {
	if testing.Short() {
		t.Skip("loads the real gpu package with go/packages")
	}
	cfg := &packages.Config{Mode: packages.NeedTypes | packages.NeedImports | packages.NeedDeps}
	pkgs, err := packages.Load(cfg, GPUPkgPath)
	if err != nil {
		t.Fatalf("loading %s: %v", GPUPkgPath, err)
	}
	if len(pkgs) != 1 || pkgs[0].Types == nil {
		t.Fatalf("loading %s returned %d usable packages", GPUPkgPath, len(pkgs))
	}
	real, synth := pkgs[0].Types, GPUPackage()

	t.Run("PackageScope", func(t *testing.T) {
		compareNames(t, "package scope", deviceNames(real.Scope()), deviceNames(synth.Scope()))
		for _, name := range deviceNames(real.Scope()) {
			r, s := real.Scope().Lookup(name), synth.Scope().Lookup(name)
			if r == nil || s == nil {
				continue
			}
			// Ctx is a named type, and the two are necessarily distinct
			// objects from distinct packages. Its method set is what has to
			// agree, and CtxMethods below is where that is checked.
			if _, isType := r.(*types.TypeName); isType {
				continue
			}
			// types.Identical, not a string comparison: TypeString renders
			// parameter names, and the synthetic builder names every parameter
			// "x" while gpu/math.go names the second one "y".
			if !types.Identical(r.Type(), s.Type()) {
				t.Errorf("gpu.%s is %s in the real package and %s in the synthetic one",
					name, types.TypeString(r.Type(), nil), types.TypeString(s.Type(), nil))
			}
		}
	})

	t.Run("CtxMethods", func(t *testing.T) {
		realCtx, synthCtx := namedCtx(t, real), namedCtx(t, synth)
		// Only the method set is compared. The synthetic Ctx is an empty
		// struct where the real one carries six unexported fields, which is
		// deliberate: they hold the CPU emulator's state and are invisible to
		// a kernel either way.
		rm, sm := methodSigs(realCtx), methodSigs(synthCtx)
		compareNames(t, "gpu.Ctx method set", keys(rm), keys(sm))
		for name, rs := range rm {
			ss, ok := sm[name]
			if !ok {
				continue
			}
			// The receivers are stripped before comparing: the two Ctx types
			// come from different types.Package values, so signatures that
			// mention them can never be Identical.
			if !types.Identical(bare(rs), bare(ss)) {
				t.Errorf("gpu.Ctx.%s is %s in the real package and %s in the synthetic one",
					name, types.TypeString(rs, nil), types.TypeString(ss, nil))
			}
		}
	})

	// The axis that actually bites. Without it, adding gpu.Tanh produces two
	// different complaints about one mistake -- "undefined: gpu.Tanh" from the
	// run-time path, "no device equivalent" from the analyzer -- and neither
	// says that the emitter is what needs updating.
	t.Run("EmitterKnowsEverySymbol", func(t *testing.T) {
		for _, name := range deviceNames(real.Scope()) {
			obj := real.Scope().Lookup(name)
			if _, isFunc := obj.(*types.Func); !isFunc {
				continue
			}
			if _, ok := gpuAtomics[name]; ok {
				// The atomics are lowered by dedicated code rather than by the
				// float32 table, because their first two arguments become one
				// C operand: &s[i].
				continue
			}
			if _, ok := gpuFuncs64[name]; ok {
				continue
			}
			if _, ok := gpuFuncs[name]; !ok {
				t.Errorf("gpu.%s has no entry in gpuFuncs, gpuFuncs64 or gpuAtomics; the emitter cannot lower it", name)
			}
		}
		for name := range methodSigs(namedCtx(t, real)) {
			// The shared tiles and AssumeBlockDim are lowered by dedicated
			// code rather than by a table entry -- the tiles because a tile is
			// a declaration and not an expression, and AssumeBlockDim because
			// it lowers to nothing at all. Both are still table-driven, by
			// sharedElems and sharedDynElems, and a tile constructor missing
			// from those is caught by the name comparison above rather than
			// here: GPUPackage declares them straight from the same tables.
			if _, ok := sharedElems[name]; ok {
				continue
			}
			if _, ok := sharedDynElems[name]; ok {
				continue
			}
			if name == "AssumeBlockDim" {
				continue
			}
			if _, ok := ctxBuiltins[name]; !ok {
				t.Errorf("gpu.Ctx.%s has no entry in ctxBuiltins; the emitter cannot lower it", name)
			}
		}
	})
}

// hostOnly names exported members of package gpu that exist for the host and
// have no place on a device, so the synthetic model deliberately omits them.
// Anything not listed here is kernel vocabulary and must be modelled.
var hostOnly = map[string]string{
	"RunCPU":          "runs a kernel on the CPU; a kernel cannot launch itself",
	"RunCPUDim":       "runs a kernel on the CPU; a kernel cannot launch itself",
	"RunCPUShared":    "runs a kernel on the CPU; a kernel cannot launch itself",
	"RunCPUSharedDim": "runs a kernel on the CPU; a kernel cannot launch itself",
	"Dim":             "launch geometry, which a kernel reads one axis at a time instead",
	"D1":              "builds a Dim",
	"D2":              "builds a Dim",
	"D3":              "builds a Dim",
}

// deviceNames lists the exported members of s that a kernel may use.
func deviceNames(s *types.Scope) []string {
	var out []string
	for _, name := range s.Names() {
		if _, skip := hostOnly[name]; skip {
			continue
		}
		if obj := s.Lookup(name); obj != nil && obj.Exported() {
			out = append(out, name)
		}
	}
	return out
}

func namedCtx(t *testing.T, pkg *types.Package) *types.Named {
	t.Helper()
	obj := pkg.Scope().Lookup("Ctx")
	if obj == nil {
		t.Fatalf("%s declares no Ctx", pkg.Path())
	}
	named, ok := obj.Type().(*types.Named)
	if !ok {
		t.Fatalf("%s.Ctx is %T, not a named type", pkg.Path(), obj.Type())
	}
	return named
}

func methodSigs(named *types.Named) map[string]*types.Signature {
	out := map[string]*types.Signature{}
	ms := types.NewMethodSet(named)
	for i := range ms.Len() {
		sel := ms.At(i)
		if !sel.Obj().Exported() {
			continue
		}
		if sig, ok := sel.Obj().Type().(*types.Signature); ok {
			out[sel.Obj().Name()] = sig
		}
	}
	return out
}

func bare(s *types.Signature) *types.Signature {
	return types.NewSignatureType(nil, nil, nil, s.Params(), s.Results(), s.Variadic())
}

func keys(m map[string]*types.Signature) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

func compareNames(t *testing.T, what string, real, synth []string) {
	t.Helper()
	inSynth := map[string]bool{}
	for _, n := range synth {
		inSynth[n] = true
	}
	inReal := map[string]bool{}
	for _, n := range real {
		inReal[n] = true
	}
	for _, n := range real {
		if !inSynth[n] {
			t.Errorf("%s: %s exists in package gpu but not in the synthetic model", what, n)
		}
	}
	for _, n := range synth {
		if !inReal[n] {
			t.Errorf("%s: %s exists in the synthetic model but not in package gpu", what, n)
		}
	}
}
