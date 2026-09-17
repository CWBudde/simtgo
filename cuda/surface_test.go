package cuda

import (
	"go/types"
	"slices"
	"testing"

	"golang.org/x/tools/go/packages"
)

const pkgPath = "github.com/CWBudde/gocuda/cuda"

// TestBuildTagSurfacesMatch keeps stub.go in step with the real driver.
//
// The "cuda" build tag is what lets the transpiler, the analyzer and their
// tests run on a machine with no CUDA, and that only works while both halves
// present the same API: a method added to the driver and forgotten in the stub
// does not fail here, it fails in somebody else's no-tag build of a package
// that calls it.
//
// The test can exist at all only since the driver stopped being cgo. While
// -tags cuda meant "link against /usr/local/cuda", loading the tagged half
// needed a toolkit, so the one machine that most needed this check -- one with
// no CUDA -- was the one that could not run it.
func TestBuildTagSurfacesMatch(t *testing.T) {
	if testing.Short() {
		t.Skip("type-checks the package twice with go/packages")
	}
	tagged := load(t, "-tags", "cuda")
	plain := load(t)

	tn, pn := exported(tagged), exported(plain)
	if !slices.Equal(tn, pn) {
		t.Errorf("exported names differ\n  -tags cuda: %v\n  no tag:     %v", tn, pn)
		for _, name := range missing(tn, pn) {
			t.Errorf("%s is missing from the no-tag build (add it to stub.go)", name)
		}
		for _, name := range missing(pn, tn) {
			t.Errorf("%s exists only in the no-tag build", name)
		}
		return
	}

	for _, name := range tn {
		a, b := tagged.Scope().Lookup(name), plain.Scope().Lookup(name)
		if _, isType := a.(*types.TypeName); isType {
			// Only the method set is compared for a named type. Context and
			// Module hold driver handles under the tag and nothing without it,
			// and those unexported fields are precisely the difference the tag
			// exists to make.
			compare(t, name+" methods", methods(a), methods(b))
			continue
		}
		if x, y := str(a), str(b); x != y {
			t.Errorf("%s differs\n  -tags cuda: %s\n  no tag:     %s", name, x, y)
		}
	}
}

func load(t *testing.T, flags ...string) *types.Package {
	t.Helper()
	cfg := &packages.Config{
		Mode:       packages.NeedTypes | packages.NeedImports | packages.NeedDeps,
		BuildFlags: flags,
	}
	pkgs, err := packages.Load(cfg, pkgPath)
	if err != nil {
		t.Fatalf("loading %s with %v: %v", pkgPath, flags, err)
	}
	if len(pkgs) != 1 || pkgs[0].Types == nil {
		t.Fatalf("loading %s with %v returned %d usable packages", pkgPath, flags, len(pkgs))
	}
	for _, e := range pkgs[0].Errors {
		t.Fatalf("loading %s with %v: %v", pkgPath, flags, e)
	}
	return pkgs[0].Types
}

// str renders an object with the package printed by name, so that two objects
// from two separate loads of the same package compare equal.
func str(obj types.Object) string {
	return types.ObjectString(obj, func(p *types.Package) string { return p.Name() })
}

func exported(p *types.Package) []string {
	var out []string
	for _, name := range p.Scope().Names() {
		if obj := p.Scope().Lookup(name); obj.Exported() {
			out = append(out, name)
		}
	}
	slices.Sort(out)
	return out
}

// methods lists a named type's exported methods as rendered declarations.
// Declared methods rather than a types.MethodSet: Slice[T] is generic, and the
// declared set is what a stub has to mirror anyway.
func methods(obj types.Object) []string {
	named, ok := obj.Type().(*types.Named)
	if !ok {
		return nil
	}
	var out []string
	for i := range named.NumMethods() {
		if m := named.Method(i); m.Exported() {
			out = append(out, str(m))
		}
	}
	slices.Sort(out)
	return out
}

func compare(t *testing.T, what string, tagged, plain []string) {
	t.Helper()
	if slices.Equal(tagged, plain) {
		return
	}
	for _, m := range missing(tagged, plain) {
		t.Errorf("%s: %s is missing from the no-tag build", what, m)
	}
	for _, m := range missing(plain, tagged) {
		t.Errorf("%s: %s exists only in the no-tag build", what, m)
	}
}

func missing(from, in []string) []string {
	var out []string
	for _, x := range from {
		if !slices.Contains(in, x) {
			out = append(out, x)
		}
	}
	return out
}
