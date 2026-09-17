package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const goodKernel = `package kernels

import "github.com/CWBudde/gocuda/gpu"

func VecAdd(ctx gpu.Ctx, c, a, b []float32) {
	i := ctx.GlobalID()
	if i < len(c) {
		c[i] = a[i] + b[i]
	}
}
`

const badKernel = `package kernels

import "github.com/CWBudde/gocuda/gpu"

func Broken(ctx gpu.Ctx, a []float64) {
	a[0] = 1
}
`

// fixture lays out a kernel package and the artifact directory beside it.
func fixture(t *testing.T, files map[string]string) generateOptions {
	t.Helper()
	root := t.TempDir()
	pkg := filepath.Join(root, "kernels")
	if err := os.MkdirAll(pkg, 0o755); err != nil {
		t.Fatal(err)
	}
	for name, src := range files {
		if err := os.WriteFile(filepath.Join(pkg, name), []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return generateOptions{
		pkgDir: pkg,
		outDir: filepath.Join(pkg, "prebuilt"),
		arch:   "compute_75",
		noPTX:  true,
	}
}

func read(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}
	return string(b)
}

// TestGenerateWritesTheGate covers the ordinary case without needing a
// toolkit: lowering is what decides the gate, and lowering is pure Go.
func TestGenerateWritesTheGate(t *testing.T) {
	o := fixture(t, map[string]string{"k.go": goodKernel})
	if err := run(o); err != nil {
		t.Fatalf("generate: %v", err)
	}
	gen := read(t, filepath.Join(o.outDir, genFile))
	if !strings.Contains(gen, `const VecAdd Lowered = "VecAdd"`) {
		t.Errorf("generated file does not gate VecAdd:\n%s", gen)
	}
	if !strings.Contains(read(t, filepath.Join(o.outDir, "VecAdd.cu")), "__global__ void VecAdd") {
		t.Error("the generated CUDA C was not written")
	}
}

// TestGenerateDropsAnUnlowerableKernel is the build gate itself: the constant
// disappears, so the hand-written gate.go stops compiling.
func TestGenerateDropsAnUnlowerableKernel(t *testing.T) {
	o := fixture(t, map[string]string{"k.go": goodKernel, "bad.go": badKernel})
	err := run(o)
	if err == nil {
		t.Fatal("expected generate to fail on a kernel it cannot lower")
	}
	if !strings.Contains(err.Error(), "Broken") {
		t.Errorf("error does not name the offending kernel: %v", err)
	}
	// It must write what it could before failing, or a broken kernel would
	// also destroy the artifacts of the ones beside it.
	gen := read(t, filepath.Join(o.outDir, genFile))
	if !strings.Contains(gen, "VecAdd") {
		t.Errorf("the kernels that did lower were not written:\n%s", gen)
	}
	if strings.Contains(gen, "Broken") {
		t.Errorf("a kernel that cannot be lowered is still gated:\n%s", gen)
	}
}

// TestGenerateRemovesOrphans: a renamed kernel must not leave an artifact
// behind, and above all must not leave a go:embed pointing at a missing file.
func TestGenerateRemovesOrphans(t *testing.T) {
	o := fixture(t, map[string]string{"k.go": goodKernel})
	if err := run(o); err != nil {
		t.Fatal(err)
	}
	orphan := filepath.Join(o.outDir, "Gone.cu")
	if err := os.WriteFile(orphan, []byte("// stale\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := run(o); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(orphan); !os.IsNotExist(err) {
		t.Errorf("%s survived a regeneration", orphan)
	}
}

// TestCheckWritesNothing: -check is the CI gate, so it must be able to run on
// a tree it is not allowed to modify.
func TestCheckWritesNothing(t *testing.T) {
	o := fixture(t, map[string]string{"k.go": goodKernel})
	check := o
	check.check, check.noPTX = true, false
	if err := run(check); err == nil {
		t.Error("expected -check to fail when nothing has been generated yet")
	}
	if _, err := os.Stat(o.outDir); !os.IsNotExist(err) {
		t.Errorf("-check created %s", o.outDir)
	}

	if err := run(o); err != nil {
		t.Fatal(err)
	}
	if err := run(check); err != nil {
		t.Errorf("-check failed on a freshly generated tree: %v", err)
	}

	// Editing a kernel without regenerating is the case the gate alone cannot
	// catch, and the one -check exists for.
	if err := os.WriteFile(filepath.Join(o.pkgDir, "k.go"),
		[]byte(strings.Replace(goodKernel, "a[i] + b[i]", "a[i] - b[i]", 1)), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := run(check); err == nil {
		t.Error("-check passed although the kernel changed since it was generated")
	}
}

// TestOutMustNotBeThePackage guards the rule that a generated file inside the
// kernel package would be type-checked as kernel source and refused for
// importing simt.
func TestOutMustNotBeThePackage(t *testing.T) {
	o := fixture(t, map[string]string{"k.go": goodKernel})
	o.outDir = o.pkgDir
	err := run(o)
	if err == nil || !strings.Contains(err.Error(), "same directory") {
		t.Errorf("expected -out == -pkg to be refused, got %v", err)
	}
}

func TestParseArchLocal(t *testing.T) {
	for _, tc := range []struct {
		in           string
		major, minor int
		wantErr      bool
	}{
		{in: "compute_75", major: 7, minor: 5},
		{in: "compute_61", major: 6, minor: 1},
		{in: "compute_100", major: 10, minor: 0},
		{in: "compute_90a", wantErr: true},
		{in: "sm_75", wantErr: true},
		{in: "compute_7", wantErr: true},
		{in: "compute_", wantErr: true},
		{in: "", wantErr: true},
	} {
		major, minor, err := parseArchLocal(tc.in)
		if tc.wantErr {
			if err == nil {
				t.Errorf("parseArchLocal(%q) = (%d, %d), want an error", tc.in, major, minor)
			}
			continue
		}
		if err != nil {
			t.Errorf("parseArchLocal(%q): %v", tc.in, err)
		} else if major != tc.major || minor != tc.minor {
			t.Errorf("parseArchLocal(%q) = (%d, %d), want (%d, %d)", tc.in, major, minor, tc.major, tc.minor)
		}
	}
}
