// Package hostrun runs a lowered kernel's generated CUDA C on the host: an
// ordinary C++ compiler builds it behind a shim that supplies threadIdx,
// blockIdx, blockDim and gridDim, and a generated main drives it one thread at
// a time.
//
// It is an oracle, not a backend. The SIMT track's claim is that one Go
// function means the same thing in two places, and until now both sides of
// every parity test came from this repository: gpu.RunCPU runs the Go the
// author wrote and the device runs PTX, but nothing ran the CUDA C in between
// without a GPU. hostrun does, so a disagreement between it and gpu.RunCPU is
// a mistranslation by the emitter and nothing else -- which is what makes it
// the answering half of a differential fuzzer, and what makes it useful on a
// machine with no device.
//
// # What it does not model
//
// Concurrency, and the scope follows from that one sentence. The threads of a
// block run one after another, so __syncthreads() is a statement that does
// nothing rather than a barrier, __shared__ memory outlives the block it
// belongs to, an atomic is an ordinary read-modify-write with nobody to race,
// and a warp primitive reads a lane that has already finished or has not yet
// started. None of those is a small inaccuracy that shows up as a rounding
// difference; each one makes the run answer a different question from the one
// asked. A kernel containing any of them is refused with a ScopeError. An
// oracle that quietly answers the wrong question is worse than one that
// declines, so the rule here is the same as the emitter's: refuse, never
// mistranslate.
//
// Of the kernels in kernels/, that leaves BandGain, Classify, Gray, Magnitude,
// Quantize, Scale, Softclip and VecAdd in scope, and FIR, Histogram, Transpose
// and WarpReduceSum out of it.
//
// # How values cross
//
// Arguments are written to the driver's stdin as raw bytes in the order the
// kernel declares them, and the slices come back on its stdout the same way.
// Nothing is spelled as a C literal. That is not a shortcut: a value that went
// through a decimal spelling on the way in would be testing the two languages'
// formatters rather than the kernel -- Go's shortest form for 1.0 is "1", and
// "1f" is not a C++ literal, a trap the emitter's own constant() already
// guards against -- and a negative zero or a NaN payload would not survive the
// round trip at all. Bytes survive everything, including the padding inside a
// struct, whose layout the generated C already asserts with static_assert.
//
// It has a second effect worth having: the compiled driver depends on the
// kernel's source and on the shape of its parameters, but not on the data or
// the geometry, both of which arrive on stdin. So the binary cache hits across
// every case a fuzzer generates for one kernel, and compilation -- which is
// two orders of magnitude more expensive than the run -- happens once.
package hostrun

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"regexp"
	"strings"
	"sync"

	"github.com/CWBudde/gocuda/gpu"
	"github.com/CWBudde/gocuda/internal/lower"
)

// ErrNoCompiler reports that no host C++ compiler was found.
//
// It is a sentinel so that a caller can tell "this machine cannot answer" from
// "this kernel is wrong" and skip rather than fail. CI's ubuntu-latest has
// g++; a container that does not is a missing tool, not a finding.
var ErrNoCompiler = errors.New("hostrun: no host C++ compiler found (tried g++ and clang++)")

// A ScopeError is a kernel hostrun will not run, because running it one thread
// at a time would answer a different question. See the package comment.
type ScopeError struct {
	Kernel    string // the kernel's name
	Construct string // the identifier in the generated C that is out of scope
	Why       string // what one-thread-at-a-time does to it
}

func (e *ScopeError) Error() string {
	return fmt.Sprintf("hostrun: %s is out of scope: it uses %s, and %s", e.Kernel, e.Construct, e.Why)
}

// A CompileError is the host compiler refusing the generated C.
//
// It carries the compiler's own output because that output is the finding: the
// emitter produced C that a C++ compiler will not accept, and the message
// saying why is more useful than any summary of it. Source is kept too, since
// the line numbers in Output refer to it and to nothing on disk.
type CompileError struct {
	Kernel   string
	Compiler string
	Output   string
	Source   string
	Err      error
}

func (e *CompileError) Error() string {
	return fmt.Sprintf("hostrun: %s refused the generated C for %s: %v\n%s", e.Compiler, e.Kernel, e.Err, e.Output)
}

func (e *CompileError) Unwrap() error { return e.Err }

// A RunError is the compiled driver failing to run to completion.
type RunError struct {
	Kernel string
	Stderr string
	Err    error
}

func (e *RunError) Error() string {
	return fmt.Sprintf("hostrun: the driver for %s failed: %v\n%s", e.Kernel, e.Err, e.Stderr)
}

func (e *RunError) Unwrap() error { return e.Err }

// Run executes u's generated CUDA C over grid blocks of block threads, binding
// args to the kernel's parameters after the gpu.Ctx, and writes what the
// kernel wrote back into the caller's slices.
//
// Writing back in place rather than returning fresh buffers is what makes the
// call symmetrical with gpu.RunCPU(grid, block, func(c gpu.Ctx) { K(c, y, x) }):
// both hand the kernel the caller's own slices and leave the answer there, so
// a differential test is two identical sets of inputs, two calls and one
// comparison, with nothing in between to get the correspondence wrong.
//
// Every slice is copied back, read-only ones included. A const parameter the
// kernel somehow wrote through would not have compiled, so this costs nothing
// and means the caller's buffers describe the whole of what the run did.
func Run(u *lower.Unit, grid, block gpu.Dim, args ...any) error {
	if err := CheckScope(u); err != nil {
		return err
	}
	if err := checkGeometry(u, grid, block); err != nil {
		return err
	}
	slots, err := plan(u, args)
	if err != nil {
		return err
	}

	src := driverSource(u, slots)
	bin, err := build(u.Name, src)
	if err != nil {
		return err
	}

	var stdout, stderr bytes.Buffer
	cmd := exec.Command(bin)
	cmd.Stdin = bytes.NewReader(encode(grid, block, slots))
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return &RunError{Kernel: u.Name, Stderr: stderr.String(), Err: err}
	}
	return decode(u.Name, stdout.Bytes(), slots)
}

// Available reports whether this machine has a host C++ compiler, so a caller
// can skip rather than fail. It is the same lookup Run does and caches, so
// asking costs nothing after the first time.
func Available() error {
	_, err := compiler()
	return err
}

// CheckScope reports whether hostrun can run u at all, without compiling
// anything.
//
// It is exported because a fuzzer wants to ask before it goes to the trouble
// of generating inputs, and because a test that asserts which kernels are
// refused should be asking the same question Run asks rather than a second
// opinion about it.
func CheckScope(u *lower.Unit) error {
	// The check reads the generated C rather than the Go source, because the
	// generated C is what will be compiled and because the Unit does not carry
	// the Go source at all. Scanning identifiers rather than substrings keeps
	// the word "atomic" in the leading comment from refusing a kernel.
	for _, id := range cIdent.FindAllString(u.Source, -1) {
		if why := outOfScope(id); why != "" {
			return &ScopeError{Kernel: u.Name, Construct: id, Why: why}
		}
	}
	return nil
}

var cIdent = regexp.MustCompile(`[A-Za-z_][A-Za-z0-9_]*`)

// hostSafe is the set of double-underscore spellings the emitter produces that
// a sequential host run can honour.
//
// __restrict__ is kept rather than defined away, which is the interesting one:
// it is a real qualifier to g++ and clang++, so the host compiler is held to
// the same non-aliasing promise the device is, and a launch that breaks it is
// as undefined here as there. The other two are storage classes CUDA needs and
// a host does not.
var hostSafe = map[string]bool{
	"__restrict__": true,
	"__global__":   true,
	"__device__":   true,
}

// atomics is CUDA's atomic vocabulary, which is wider than SPEC.md section 4
// exposes. The extra names cost nothing and mean this does not have to be
// revisited when the subset grows one.
var atomics = map[string]bool{
	"atomicAdd": true, "atomicSub": true, "atomicExch": true,
	"atomicMin": true, "atomicMax": true, "atomicInc": true,
	"atomicDec": true, "atomicCAS": true, "atomicAnd": true,
	"atomicOr": true, "atomicXor": true,
}

// outOfScope says why an identifier in the generated C cannot be run one
// thread at a time, or "" if it can.
//
// Every double-underscore name that is not on the short allowlist is refused,
// including ones this package has never heard of. That is deliberately the
// wrong way round from an allowlist of known-bad names: a builtin added to the
// emitter later would pass a blocklist silently and be run with whatever a
// single thread makes of it, whereas being refused costs a fuzzer one skipped
// kernel and a one-line change here.
func outOfScope(id string) string {
	switch {
	case atomics[id]:
		return "an atomic read-modify-write has nobody to race with when the threads run one at a time, " +
			"so it cannot show what the ordering does to the answer"
	case !strings.HasPrefix(id, "__"), hostSafe[id]:
		return ""
	case id == "__syncthreads", id == "__syncwarp":
		return "a barrier between threads that have already finished is not a barrier"
	case strings.HasSuffix(id, "_sync"), id == "__activemask":
		return "a warp primitive reads lanes that a sequential run has either finished or not yet started"
	case id == "__shared__":
		return "shared memory is per block, and a sequential run leaves one block's tile in place for the next"
	default:
		return "it is a CUDA builtin this package has not been taught to model, and guessing is not an oracle"
	}
}

func checkGeometry(u *lower.Unit, grid, block gpu.Dim) error {
	if grid.X <= 0 || grid.Y <= 0 || grid.Z <= 0 || block.X <= 0 || block.Y <= 0 || block.Z <= 0 {
		return fmt.Errorf("hostrun: %s: grid %v and block %v must be positive on every axis", u.Name, grid, block)
	}
	// The same contract Kernel.Launch and gpu.RunCPU enforce, counted the same
	// way: threads per block across all three axes, so a 16x16 block satisfies
	// AssumeBlockDim(256). An oracle held to a weaker contract than the thing
	// it judges would report its own launch as the kernel's disagreement.
	if threads := block.X * block.Y * block.Z; u.RequiredBlock != 0 && threads != u.RequiredBlock {
		return fmt.Errorf("hostrun: %s requires block == %d, got %d", u.Name, u.RequiredBlock, threads)
	}
	return nil
}

// build compiles src, reusing an earlier binary for the same source.
//
// The cache is keyed on the source text alone, which is sound because the
// source is the whole input: the data and the geometry reach the driver on
// stdin. The compiler's identity goes into the key too, since the same source
// built by g++ and by clang++ are two different answers and a fuzzer switching
// between them should not be handed the other one's.
func build(name, src string) (string, error) {
	cxx, err := compiler()
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256([]byte(cxx + "\x00" + strings.Join(cxxFlags, " ") + "\x00" + src))
	bin := filepath.Join(cacheDir(), hex.EncodeToString(sum[:])[:32])

	// One mutex per binary rather than one for the package: a fuzzer runs
	// cases in parallel and two different kernels have no reason to wait for
	// each other, while two goroutines wanting the same one would otherwise
	// both compile it.
	lock, _ := locks.LoadOrStore(bin, new(sync.Mutex))
	mu := lock.(*sync.Mutex)
	mu.Lock()
	defer mu.Unlock()

	if _, err := os.Stat(bin); err == nil {
		return bin, nil
	}
	if err := os.MkdirAll(cacheDir(), 0o755); err != nil {
		return "", fmt.Errorf("hostrun: %w", err)
	}
	// Compiled under a name of its own and renamed into place, so that a
	// second process finding the file finds a whole one. The .cpp goes with
	// the temporary directory either way: on failure CompileError carries the
	// source, and on success nobody reads it, while a fuzzer would leave
	// thousands.
	tmp, err := os.MkdirTemp(cacheDir(), "build-")
	if err != nil {
		return "", fmt.Errorf("hostrun: %w", err)
	}
	defer func() { _ = os.RemoveAll(tmp) }()

	cpp := filepath.Join(tmp, name+".cpp")
	if err := os.WriteFile(cpp, []byte(src), 0o644); err != nil {
		return "", fmt.Errorf("hostrun: %w", err)
	}
	out := filepath.Join(tmp, "driver")
	cmd := exec.Command(cxx, append(append([]string{}, cxxFlags...), "-o", out, cpp)...)
	if log, err := cmd.CombinedOutput(); err != nil {
		return "", &CompileError{Kernel: name, Compiler: cxx, Output: string(log), Source: src, Err: err}
	}
	if err := os.Rename(out, bin); err != nil {
		return "", fmt.Errorf("hostrun: %w", err)
	}
	return bin, nil
}

var locks sync.Map // binary path -> *sync.Mutex

// cxxFlags is what the driver is built with.
//
// -ffp-contract=off is the load-bearing one and is not cosmetic. NUMERICS.md
// records that Go does not contract a multiply-add on amd64 while g++ does, so
// with contraction on, every kernel with an a*b+c in it would report a
// disagreement with gpu.RunCPU in the last bits -- which reads exactly like a
// transpiler bug and is not one. The device does contract, and that asymmetry
// is the parity tests' business; here both sides must round twice or the
// comparison means nothing.
//
// -O1 rather than -O0 because it is faster to build and run than -O2 and still
// exercises the optimiser a little; no optimisation level may change a result,
// and one that does is worth knowing about.
//
// -Wall costs nothing here and pays for itself in CompileError.Output, where a
// warning is usually what explains the error under it. A warning on its own is
// not a finding: the only one the committed kernels produce is g++ suggesting
// parentheses in Gray's `a + b >> 8`, where the emitter dropped the Go
// source's parentheses because C's `>>` binds looser than `+` and reproduces
// the same grouping -- which this package's own comparison confirms bit for
// bit.
var cxxFlags = []string{"-std=c++17", "-ffp-contract=off", "-O1", "-Wall"}

// compiler finds a host C++ compiler, preferring g++.
//
// g++ first because it is what CI has and what the reference measurements were
// taken with, and clang++ second so that a machine with only it can still run
// the oracle. The answer is resolved once: a fuzzer asks thousands of times
// and the answer cannot change under it.
var compiler = sync.OnceValues(func() (string, error) {
	for _, name := range []string{"g++", "clang++"} {
		if path, err := exec.LookPath(name); err == nil {
			return path, nil
		}
	}
	return "", ErrNoCompiler
})

// cacheDir is where compiled drivers live.
//
// The user cache directory rather than the repository's .gocuda-cache, because
// these are not artifacts of a build anyone inspects and a fuzzer produces one
// per kernel shape; GOCUDA_HOSTRUN_CACHE overrides it for a caller that wants
// them somewhere it can delete.
var cacheDir = sync.OnceValue(func() string {
	if dir := os.Getenv("GOCUDA_HOSTRUN_CACHE"); dir != "" {
		return dir
	}
	base, err := os.UserCacheDir()
	if err != nil {
		base = os.TempDir()
	}
	return filepath.Join(base, "gocuda", "hostrun")
})

// A slot is one kernel parameter, resolved against the Go value bound to it.
type slot struct {
	param  lower.Param
	ctype  string        // how the driver declares it
	elem   int           // bytes per element for a slice, or the whole width
	n      int           // elements, for a slice
	val    reflect.Value // the Go value, whose bytes cross
	narrow bool          // a Go int passed by value, which crosses as a C int
}

// plan matches the caller's arguments to the kernel's parameters.
func plan(u *lower.Unit, args []any) ([]slot, error) {
	if len(args) != len(u.Params) {
		return nil, fmt.Errorf("hostrun: %s takes %d arguments after the gpu.Ctx, got %d",
			u.Name, len(u.Params), len(args))
	}
	slots := make([]slot, len(args))
	for i, a := range args {
		p := u.Params[i]
		v := reflect.ValueOf(a)
		if !v.IsValid() {
			return nil, fmt.Errorf("hostrun: %s: argument %d (%s) is nil", u.Name, i, p.Name)
		}
		s := slot{param: p, val: v}
		var et reflect.Type // the type whose C spelling the driver declares
		switch {
		case p.Slice && v.Kind() != reflect.Slice:
			return nil, fmt.Errorf("hostrun: %s: parameter %s is a slice, got %s", u.Name, p.Name, v.Type())
		case !p.Slice && v.Kind() == reflect.Slice:
			return nil, fmt.Errorf("hostrun: %s: parameter %s is a scalar, got %s", u.Name, p.Name, v.Type())
		case p.Slice:
			et = v.Type().Elem()
			ct, err := ctypeElem(et)
			if err != nil {
				return nil, fmt.Errorf("hostrun: %s: parameter %s: %w", u.Name, p.Name, err)
			}
			s.ctype, s.elem, s.n = ct, int(et.Size()), v.Len()
		default:
			et = v.Type()
			ct, err := ctype(et)
			if err != nil {
				return nil, fmt.Errorf("hostrun: %s: parameter %s: %w", u.Name, p.Name, err)
			}
			s.ctype, s.elem = ct, int(et.Size())
			if et.Kind() == reflect.Int {
				// The one narrowing SPEC.md documents: a kernel's int
				// parameter is C's 32-bit int. cuda.BuildArgs does exactly
				// this for a real launch, so the oracle has to as well, or it
				// would be judging a call nobody can make.
				s.elem, s.narrow = 4, true
			}
		}
		// A struct crosses as its Go bytes, which is sound only if the C
		// declaration is the one the emitter wrote and asserted the size of.
		// A type the translation unit does not declare would fail to compile
		// with the driver's line number rather than the parameter's name, so
		// it is caught here while the name is still at hand.
		if et.Kind() == reflect.Struct && !declaresStruct(u.Source, s.ctype) {
			return nil, fmt.Errorf("hostrun: %s: parameter %s has Go type %s, "+
				"but the generated C declares no struct %q", u.Name, p.Name, et, s.ctype)
		}
		slots[i] = s
	}
	return slots, nil
}

// declaresStruct looks for the emitter's own spelling of a struct definition,
// which is the type's name alone on a line. It is a text search because the
// alternative is a C++ parser, and what is being checked is only that the
// driver and the kernel mean the same type by a name.
func declaresStruct(src, name string) bool {
	for line := range strings.Lines(src) {
		if strings.TrimRight(line, "\r\n") == "struct "+name {
			return true
		}
	}
	return false
}

// ctype is SPEC.md section 3's by-value map: a scalar parameter.
func ctype(t reflect.Type) (string, error) {
	switch t.Kind() {
	case reflect.Float32:
		return "float", nil
	case reflect.Float64:
		return "double", nil
	case reflect.Int32:
		return "int", nil
	case reflect.Int:
		return "int", nil // narrowed; see plan
	case reflect.Int64:
		return "long long", nil
	case reflect.Uint32:
		return "unsigned int", nil
	case reflect.Uint64:
		return "unsigned long long", nil
	case reflect.Bool:
		return "bool", nil
	case reflect.Struct:
		return t.Name(), nil
	default:
		return "", fmt.Errorf("%s cannot be passed by value to a kernel", t)
	}
}

// ctypeElem is SPEC.md section 3's layout map: a slice element. It differs
// from ctype in exactly the two places the spec says it does -- the narrow
// integers are storage and nothing else, and Go's int is a stride rather than
// a value and so is refused.
func ctypeElem(t reflect.Type) (string, error) {
	switch t.Kind() {
	case reflect.Int8:
		return "signed char", nil
	case reflect.Int16:
		return "short", nil
	case reflect.Uint8:
		return "unsigned char", nil
	case reflect.Uint16:
		return "unsigned short", nil
	case reflect.Int:
		return "", errors.New("[]int cannot cross to the device: Go's int is 8 bytes and CUDA's is 4, " +
			"so it is a different stride rather than a lost high word")
	default:
		return ctype(t)
	}
}
