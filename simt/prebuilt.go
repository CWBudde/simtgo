package simt

import (
	"bytes"
	"fmt"
	"io/fs"
	"strings"
	"sync"

	"github.com/CWBudde/gocuda/cuda"
	"github.com/CWBudde/gocuda/internal/lower"
)

// Prebuilt is PTX that "gocuda generate" produced ahead of time for one
// kernel, registered from the generated file's init so that Build can load it
// instead of calling NVRTC.
type Prebuilt struct {
	Name           string // the kernel's Go function name, for diagnostics only
	SourceSHA256   string // lower.SourceHash of the generated CUDA C; the registry key
	Arch           string // the virtual arch NVRTC targeted, e.g. "compute_75"
	PTX            []byte
	RequiredBlock  int
	SharedBytes    int
	DynSharedWidth int
	Log            string // NVRTC's log at generate time; usually empty
}

// The registry is keyed on the source hash alone, and the kernel's name is
// carried only so a message can say which kernel is meant.
//
// That dissolves what would otherwise be the hard problem here: a registry is
// package-scoped and process-wide, while Build is handed an arbitrary fs.FS
// whose identity nobody can establish. Keying on what the kernel lowers to
// makes the question moot. Two packages that both export a VecAdd generate
// different CUDA C, hash differently and cannot collide; two whose VecAdd
// lowers to byte-identical CUDA C share the PTX, which is correct, because
// they are the same kernel. A user's own kernel package needs no namespacing
// and no FS identity to register into this map safely.
//
// Staleness is not detected, it is structurally impossible to hit. An entry
// built from an older kernel, or by an older emitter, hashes to something the
// current source never asks for: it is simply not found, and Build transpiles
// and JITs exactly as it does with no registry at all. Nothing can be served
// that was not built from the very bytes being compiled.
var (
	prebuiltMu sync.Mutex
	prebuilts  = map[string][]Prebuilt{}
)

// RegisterPrebuilt files p under its source hash. It is meant to be called
// from the init of a generated file, and several packages may do so at once.
//
// Registering the same PTX twice is a no-op, because a generated file may well
// be linked into a binary through more than one path. Registering *different*
// PTX for the same source and architecture panics: the two cannot both be the
// compilation of that source, so one of them is a generator bug, and serving
// whichever happened to register first would hide it behind a kernel that
// misbehaves only in some builds.
func RegisterPrebuilt(p Prebuilt) {
	prebuiltMu.Lock()
	defer prebuiltMu.Unlock()
	for _, old := range prebuilts[p.SourceSHA256] {
		if old.Arch != p.Arch {
			continue
		}
		if bytes.Equal(old.PTX, p.PTX) {
			return
		}
		panic(fmt.Sprintf("simt: two different prebuilt PTX images registered for %s (%s, source %s); the generated files disagree",
			p.Name, p.Arch, p.SourceSHA256[:12]))
	}
	prebuilts[p.SourceSHA256] = append(prebuilts[p.SourceSHA256], p)
}

// pickPrebuilt returns the registration for hash that best suits a device of
// the given compute capability.
//
// PTX is forward compatible: the driver JITs a compute_75 image for a compute
// _89 device, but not the other way round, so a registration is usable when
// the device is at or newer than the baseline it was built for. Among those,
// the newest baseline wins, since it is the one whose generator knew the most
// about the hardware.
//
// An Arch that does not parse -- a hand-edited generated file, or an
// arch-conditional target whose PTX is not forward compatible at all -- is
// skipped rather than reported. There is always a working way to produce this
// kernel, and refusing to run because an artifact meant to save time is
// unusable would be the wrong trade. A build without the "cuda" tag reports
// (0, 0) and therefore satisfies nothing, which is how the JIT path ends up
// being the only path there without a special case.
func pickPrebuilt(hash string, major, minor int) (Prebuilt, bool) {
	prebuiltMu.Lock()
	defer prebuiltMu.Unlock()

	var best Prebuilt
	var bestMajor, bestMinor int
	found := false
	for _, p := range prebuilts[hash] {
		pMajor, pMinor, err := cuda.ParseArch(p.Arch)
		if err != nil {
			continue
		}
		if pMajor > major || (pMajor == major && pMinor > minor) {
			continue
		}
		if !found || pMajor > bestMajor || (pMajor == bestMajor && pMinor > bestMinor) {
			best, bestMajor, bestMinor, found = p, pMajor, pMinor, true
		}
	}
	return best, found
}

// VerifyPrebuilt transpiles the named kernels of fsys and reports the ones no
// registration matches. With no names it checks every kernel the package
// declares.
//
// It is what a user puts in a one-line test to find out that "go generate" has
// not been run since the kernels changed. Build itself never needs this: a
// stale artifact is one it does not find, and it compiles instead. The failure
// this catches is therefore never a wrong result, only a binary that quietly
// went back to compiling at start-up, which is exactly the kind of regression
// that needs a test to be noticed at all.
//
// No device is involved: the architectures a registration was built for say
// nothing about whether it was built from the current source, which is the
// only question here.
func VerifyPrebuilt(fsys fs.FS, names ...string) error {
	pkg, diags, err := lower.LoadPackage(fsys)
	if err != nil {
		return fmt.Errorf("simt: %w", err)
	}
	if len(diags) > 0 {
		return newUnsupportedError(pkg.Fset, "", diags)
	}
	if len(names) == 0 {
		names = pkg.Names()
	}

	var stale []string
	for _, name := range names {
		u, diags, err := pkg.Kernel(name)
		if err != nil {
			return fmt.Errorf("simt: %w", err)
		}
		if len(diags) > 0 {
			return newUnsupportedError(pkg.Fset, name, diags)
		}
		if !registered(u.SourceHash) {
			stale = append(stale, name)
		}
	}
	if len(stale) == 0 {
		return nil
	}
	return fmt.Errorf("simt: %s %s no prebuilt PTX for the current source; run %q",
		strings.Join(stale, ", "), plural(len(stale), "has", "have"), "go generate ./...")
}

func registered(hash string) bool {
	prebuiltMu.Lock()
	defer prebuiltMu.Unlock()
	return len(prebuilts[hash]) > 0
}

func plural(n int, one, many string) string {
	if n == 1 {
		return one
	}
	return many
}
