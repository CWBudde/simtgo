package main

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"go/format"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/CWBudde/gocuda/cuda"
	"github.com/CWBudde/gocuda/internal/lower"
)

// The generator is split at the CUDA C boundary, and everything else about it
// follows from where that line falls:
//
//   - lowering is pure Go, and is what decides which kernels exist. It is
//     therefore what the build gate can depend on, and it works on every
//     machine.
//   - NVRTC needs a toolkit, and only produces PTX. It is an optimisation.
//
// So a machine with no CUDA can still refresh the gate, and a machine with a
// toolkit but no GPU can still produce PTX. What a run must never do is
// quietly delete committed PTX because the toolkit was missing, which is why
// dropping to the lowering-only mode has to be asked for rather than inferred.
type generateOptions struct {
	pkgDir   string
	outDir   string
	arch     string
	noPTX    bool
	check    bool
	initGate bool
}

func generate(args []string) int {
	var o generateOptions
	fs := flag.NewFlagSet("gocuda generate", flag.ExitOnError)
	fs.StringVar(&o.pkgDir, "pkg", "./kernels", "directory holding the kernel sources")
	fs.StringVar(&o.outDir, "out", "", "directory to write artifacts to (default <pkg>/prebuilt)")
	fs.StringVar(&o.arch, "arch", "compute_75", "virtual architecture to compile PTX for")
	fs.BoolVar(&o.noPTX, "no-ptx", false, "refresh the gate and the CUDA C only, leaving PTX alone")
	fs.BoolVar(&o.check, "check", false, "write nothing; fail if the generated files are not current")
	fs.BoolVar(&o.initGate, "init", false, "also write gate.go if it does not exist")
	fs.Usage = func() {
		fmt.Fprint(fs.Output(), `usage: gocuda generate [flags]

Lowers every kernel in a package, compiles each to PTX with NVRTC, and writes
the artifacts a build can embed. Run it from a //go:generate line.

Exits non-zero when a kernel cannot be lowered, having first removed that
kernel from the generated file -- which is what makes "go build" fail on it.

`)
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if o.outDir == "" {
		o.outDir = filepath.Join(o.pkgDir, "prebuilt")
	}
	if err := run(o); err != nil {
		fmt.Fprintf(os.Stderr, "gocuda generate: %v\n", err)
		return 1
	}
	return 0
}

func run(o generateOptions) error {
	if _, _, err := cuda.ParseArch(o.arch); err != nil {
		return fmt.Errorf("-arch: %w", err)
	}
	// The artifacts must not land in the kernel package itself. It is
	// type-checked as a whole and may import nothing but gpu, so a generated
	// file importing simt would break every kernel beside it -- and the
	// go:embed that feeds the transpiler would pull it in as kernel source.
	//
	// A subdirectory is fine, and is the default: neither "kernels/*.go" nor
	// the transpiler's own glob crosses a directory separator.
	if same, err := sameDir(o.outDir, o.pkgDir); err != nil {
		return err
	} else if same {
		return fmt.Errorf("-out and -pkg are the same directory; the generated files would be type-checked as kernel source (try -out %s)", filepath.Join(o.pkgDir, "prebuilt"))
	}

	pkg, diags, err := lower.LoadPackage(os.DirFS(o.pkgDir))
	if err != nil {
		return fmt.Errorf("%s: %w", o.pkgDir, err)
	}
	if len(diags) > 0 {
		printDiags(o.pkgDir, pkg, diags)
		// Nothing in the package can be lowered, so nothing may stay gated.
		// A refusal that applies to the whole package -- a forbidden import
		// is the usual one, and is perfectly valid Go -- would otherwise
		// leave yesterday's constants in place and let "go build" pass.
		if !o.check && !o.initGate {
			if err := writeAll(o, map[string][]byte{genFile: renderGen(o, nil, nil)}); err != nil {
				return err
			}
		}
		return fmt.Errorf("%s cannot be lowered", o.pkgDir)
	}

	names := pkg.Names()
	if len(names) == 0 {
		return fmt.Errorf("%s declares no kernels (a kernel's first parameter is a gpu.Ctx)", o.pkgDir)
	}

	var units []*lower.Unit
	var refused []string
	for _, name := range names {
		u, diags, err := pkg.Kernel(name)
		if err != nil {
			return err
		}
		if len(diags) > 0 {
			printDiags(o.pkgDir, pkg, diags)
			refused = append(refused, name)
			continue
		}
		units = append(units, u)
	}

	var compiled map[string]compiledPTX
	switch {
	case o.check || o.noPTX:
		// Both modes reuse whatever PTX is on disk, and keep only what still
		// matches the CUDA C it was built from. Neither runs NVRTC, so neither
		// can be made to delete a committed artifact by a missing toolkit.
		compiled = reusePTX(o, units)
	default:
		compiled, err = compilePTX(o, units)
		if err != nil {
			return err
		}
	}

	files := plan(o, units, compiled)
	if o.check {
		return checkCurrent(o, files)
	}
	if err := writeAll(o, files); err != nil {
		return err
	}
	if o.initGate {
		if err := writeGate(o, names); err != nil {
			return err
		}
	}
	if len(refused) > 0 {
		return fmt.Errorf("%s cannot be lowered; %s no longer appears in %s, so the build will fail on it",
			strings.Join(refused, ", "), plural(len(refused), "it", "they"), filepath.Join(o.outDir, genFile))
	}
	for _, u := range units {
		if _, ok := compiled[u.Name]; !ok && !o.noPTX {
			return fmt.Errorf("%s lowered but NVRTC would not compile it; it stays in the gate and will be built at run time", u.Name)
		}
	}
	return nil
}

func plural(n int, one, many string) string {
	if n == 1 {
		return one
	}
	return many
}

func printDiags(dir string, pkg *lower.Package, diags []lower.Diagnostic) {
	for _, d := range diags {
		pos := d.Msg
		if pkg != nil && pkg.Fset != nil && d.Pos.IsValid() {
			p := pkg.Fset.Position(d.Pos)
			pos = filepath.Join(dir, p.Filename) + fmt.Sprintf(":%d:%d: ", p.Line, p.Column) + d.Msg
		}
		fmt.Fprintln(os.Stderr, pos)
	}
}

func sameDir(a, b string) (bool, error) {
	pa, err := filepath.Abs(a)
	if err != nil {
		return false, err
	}
	pb, err := filepath.Abs(b)
	if err != nil {
		return false, err
	}
	return filepath.Clean(pa) == filepath.Clean(pb), nil
}

type compiledPTX struct {
	ptx  []byte
	log  string
	arch string
}

// reusePTX keeps the PTX already on disk for every kernel whose CUDA C is
// unchanged, and reports nothing for the rest. It never deletes: a run that
// could not compile must leave a tree that still builds.
func reusePTX(o generateOptions, units []*lower.Unit) map[string]compiledPTX {
	out := map[string]compiledPTX{}
	for _, u := range units {
		cu, err := os.ReadFile(filepath.Join(o.outDir, u.Name+".cu"))
		if err != nil || string(cu) != u.Source {
			continue
		}
		ptx, err := os.ReadFile(filepath.Join(o.outDir, ptxName(u.Name, o.arch)))
		if err != nil {
			continue
		}
		out[u.Name] = compiledPTX{ptx: ptx, arch: o.arch}
	}
	return out
}

// compilePTX runs NVRTC over the batch.
//
// A kernel NVRTC refuses is reported and left out, because the others are
// still worth generating and the caller decides what a refusal means. Not
// reaching NVRTC at all is the opposite case: it says nothing about any
// kernel, every one of them would fail identically, and PTX already on disk
// must not be replaced on the strength of it, so the run stops instead.
func compilePTX(o generateOptions, units []*lower.Unit) (map[string]compiledPTX, error) {
	out := map[string]compiledPTX{}
	for _, u := range units {
		ptx, err := cuda.Compile(u.Source, u.Name+".cu", o.arch)
		if err != nil {
			// ErrNoCUDA covers both halves of "there is no NVRTC here": a
			// binary built without the "cuda" tag, and a tagged one that could
			// not load libnvrtc. To someone running go generate they are the
			// same missing toolkit, and the same -no-ptx gets them moving.
			if errors.Is(err, cuda.ErrNoCUDA) {
				return nil, fmt.Errorf(`could not reach NVRTC, so no PTX was produced and nothing was written.
Is the CUDA toolkit installed, and was this built with -tags cuda? Re-run with -no-ptx to refresh the gate without it.

%w`, err)
			}
			fmt.Fprintf(os.Stderr, "%s: %v\n", u.Name, err)
			continue
		}
		out[u.Name] = compiledPTX{ptx: ptx.Bytes, log: ptx.Log, arch: o.arch}
	}
	return out, nil
}

func ptxName(kernel, arch string) string { return kernel + "." + arch + ".ptx" }

const genFile = "prebuilt_gen.go"

// writeAll replaces the whole output directory, so that a kernel that was
// renamed or removed does not leave an orphaned artifact behind -- and, more
// importantly, does not leave a go:embed directive pointing at a file that is
// gone, which would fail the build somewhere unhelpful.
func writeAll(o generateOptions, files map[string][]byte) error {
	if err := os.MkdirAll(o.outDir, 0o755); err != nil {
		return err
	}
	entries, err := os.ReadDir(o.outDir)
	if err != nil {
		return err
	}
	for _, e := range entries {
		name := e.Name()
		generated := name == genFile || strings.HasSuffix(name, ".cu") || strings.HasSuffix(name, ".ptx")
		if _, keep := files[name]; !generated || keep {
			continue
		}
		// Without a toolkit this run cannot produce PTX, so it must not
		// destroy any: the machine that cannot rebuild the artifact is
		// exactly the machine that needs the committed one. It is left
		// unregistered rather than removed, and says so.
		if o.noPTX && strings.HasSuffix(name, ".ptx") {
			fmt.Fprintf(os.Stderr, "%s no longer matches the kernel it was built from; it is kept but not registered, and a run with NVRTC will replace it\n",
				filepath.Join(o.outDir, name))
			continue
		}
		if err := os.Remove(filepath.Join(o.outDir, name)); err != nil {
			return err
		}
	}
	names := make([]string, 0, len(files))
	for name := range files {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		if err := os.WriteFile(filepath.Join(o.outDir, name), files[name], 0o644); err != nil {
			return err
		}
	}
	return nil
}

func checkCurrent(o generateOptions, files map[string][]byte) error {
	var stale []string
	for name, want := range files {
		got, err := os.ReadFile(filepath.Join(o.outDir, name))
		if err != nil || !bytes.Equal(got, want) {
			stale = append(stale, name)
		}
	}
	if len(stale) > 0 {
		sort.Strings(stale)
		return fmt.Errorf("%s is not current: %s; run \"go generate ./...\"",
			o.outDir, strings.Join(stale, ", "))
	}
	return nil
}

// plan renders every file the run would write. PTX is deliberately not part of
// what -check compares: NVRTC stamps its own version into the output, so the
// bytes change whenever the toolkit does, while the CUDA C and the generated
// Go depend only on the transpiler.
func plan(o generateOptions, units []*lower.Unit, compiled map[string]compiledPTX) map[string][]byte {
	files := map[string][]byte{}
	for _, u := range units {
		files[u.Name+".cu"] = []byte(u.Source)
		if c, ok := compiled[u.Name]; ok {
			files[ptxName(u.Name, c.arch)] = c.ptx
		}
	}
	files[genFile] = renderGen(o, units, compiled)
	return files
}

func renderGen(o generateOptions, units []*lower.Unit, compiled map[string]compiledPTX) []byte {
	pkgName := filepath.Base(o.outDir)
	var b strings.Builder
	fmt.Fprintf(&b, "// Code generated by \"gocuda generate\"; DO NOT EDIT.\n\n")
	fmt.Fprintf(&b, "package %s\n\n", pkgName)

	embedsPTX := len(compiled) > 0
	if embedsPTX {
		b.WriteString("import (\n\t_ \"embed\"\n\n\t\"github.com/CWBudde/gocuda/simt\"\n)\n\n")
	}

	for _, u := range units {
		fmt.Fprintf(&b, "// %s reports that the kernel of that name lowered to CUDA C.\n", u.Name)
		fmt.Fprintf(&b, "const %s Lowered = %q\n\n", u.Name, u.Name)
	}
	for _, u := range units {
		c, ok := compiled[u.Name]
		if !ok {
			continue
		}
		fmt.Fprintf(&b, "//go:embed %s\nvar ptx%s []byte\n\n", ptxName(u.Name, c.arch), u.Name)
	}
	if embedsPTX {
		b.WriteString("func init() {\n")
		for _, u := range units {
			c, ok := compiled[u.Name]
			if !ok {
				continue
			}
			fmt.Fprintf(&b, "\tsimt.RegisterPrebuilt(simt.Prebuilt{\n")
			fmt.Fprintf(&b, "\t\tName:          %q,\n", u.Name)
			fmt.Fprintf(&b, "\t\tSourceSHA256:  %q,\n", u.SourceHash)
			fmt.Fprintf(&b, "\t\tArch:          %q,\n", c.arch)
			fmt.Fprintf(&b, "\t\tPTX:           ptx%s,\n", u.Name)
			if u.RequiredBlock != 0 {
				fmt.Fprintf(&b, "\t\tRequiredBlock: %d,\n", u.RequiredBlock)
			}
			if u.SharedBytes != 0 {
				fmt.Fprintf(&b, "\t\tSharedBytes:   %d,\n", u.SharedBytes)
			}
			if u.DynSharedWidth != 0 {
				fmt.Fprintf(&b, "\t\tDynSharedWidth: %d,\n", u.DynSharedWidth)
			}
			if c.log != "" {
				fmt.Fprintf(&b, "\t\tLog:           %q,\n", c.log)
			}
			fmt.Fprintf(&b, "\t})\n")
		}
		b.WriteString("}\n")
	}
	src, err := format.Source([]byte(b.String()))
	if err != nil {
		// Unformattable output is a bug in this file, not in the kernels, and
		// writing it anyway makes that visible at the next build.
		return []byte(b.String())
	}
	return src
}

// writeGate scaffolds the hand-written half, which the generator then never
// touches again. It is the file that makes an unlowerable kernel a build
// failure: it names every kernel that must lower, and a name the generator
// dropped stops resolving.
func writeGate(o generateOptions, names []string) error {
	path := filepath.Join(o.outDir, "gate.go")
	if _, err := os.Stat(path); err == nil {
		return nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	pkgName := filepath.Base(o.outDir)
	var b strings.Builder
	fmt.Fprintf(&b, `// Package %s holds the ahead-of-time artifacts for the kernels beside it.
//
// It is deliberately not part of the kernel package: a kernel package may
// import nothing but gocuda/gpu, and everything here imports simt.
package %s

// Lowered names a kernel that "gocuda generate" was able to lower to CUDA C.
type Lowered string

// Gate lists every kernel that must lower for this package to compile.
//
// Each name is a constant in %s, which "gocuda generate"
// writes only for the kernels it could lower. A kernel it had to drop stops
// resolving here, and the build then fails with "undefined" -- which is how a
// kernel that cannot be lowered fails "go build" instead of main().
//
// Add a name here whenever you add a kernel.
var Gate = []Lowered{
`, pkgName, pkgName, genFile)
	for _, n := range names {
		fmt.Fprintf(&b, "\t%s,\n", n)
	}
	b.WriteString(`}

// Names returns Gate as plain strings, for simt.VerifyPrebuilt.
func Names() []string {
	out := make([]string, len(Gate))
	for i, n := range Gate {
		out[i] = string(n)
	}
	return out
}
`)
	src, err := format.Source([]byte(b.String()))
	if err != nil {
		src = []byte(b.String())
	}
	return os.WriteFile(path, src, 0o644)
}
