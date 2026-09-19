package cuda_test

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"runtime"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/CWBudde/gocuda/cuda"
	"github.com/CWBudde/gocuda/internal/jit"
)

// requireDevice skips a test that needs a GPU, or fails it when the caller has
// promised one.
//
// GOCUDA_REQUIRE_DEVICE is what a job sets when the whole point of the run is
// that kernels executed -- a compute-sanitizer sweep, say. There a silent skip
// is the worst outcome available: the job goes green having launched nothing.
// See .github/workflows/sanitizer.yml.
// It takes a testing.TB rather than a *testing.T so that the benchmarks can
// use it too: a benchmark that silently measured nothing is the same failure
// as a test that silently checked nothing.
func requireDevice(t testing.TB) {
	t.Helper()
	if cuda.Available() {
		return
	}
	if os.Getenv("GOCUDA_REQUIRE_DEVICE") != "" {
		t.Fatal("GOCUDA_REQUIRE_DEVICE is set, but no CUDA device is available")
	}
	t.Skip("no CUDA device available")
}

const addSrc = `
extern "C" __global__ void add(float* c, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}`

// TestRawKernel exercises the whole host path with a hand-written kernel:
// NVRTC compile, module load, upload, launch, download.
func TestRawKernel(t *testing.T) {
	requireDevice(t)
	ctx, err := cuda.NewContext(0)
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Close()
	t.Logf("device %q, arch %s", ctx.Name(), ctx.Arch())

	ptx, err := cuda.Compile(addSrc, "add.cu", ctx.Arch())
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	if ptx.Log != "" {
		t.Logf("nvrtc log: %s", ptx.Log)
	}
	mod, err := ctx.LoadPTX(ptx.Bytes)
	if err != nil {
		t.Fatalf("LoadPTX: %v", err)
	}
	fn, err := mod.Function("add")
	if err != nil {
		t.Fatalf("Function: %v", err)
	}

	const n = 1024
	a := make([]float32, n)
	b := make([]float32, n)
	for i := range a {
		a[i] = float32(i)
		b[i] = float32(2 * i)
	}
	da, err := cuda.Upload(ctx, a)
	if err != nil {
		t.Fatalf("Upload a: %v", err)
	}
	defer da.Free()
	db, err := cuda.Upload(ctx, b)
	if err != nil {
		t.Fatalf("Upload b: %v", err)
	}
	defer db.Free()
	dc, err := cuda.NewSlice[float32](ctx, n)
	if err != nil {
		t.Fatalf("NewSlice: %v", err)
	}
	defer dc.Free()

	const block = 256
	grid := (n + block - 1) / block
	if err := fn.LaunchSync(cuda.D1(grid), cuda.D1(block), 0,
		dc.Arg(), da.Arg(), db.Arg(), cuda.ArgI32(n)); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	got, err := dc.Download()
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	for i := range got {
		if want := a[i] + b[i]; got[i] != want {
			t.Fatalf("c[%d] = %v, want %v", i, got[i], want)
		}
	}
}

// runAdd launches the add kernel over a small vector and checks the result, so
// that a test can assert a module is not merely loaded but actually usable in
// the context it claims to belong to.
func runAdd(t *testing.T, ctx *cuda.Context, fn *cuda.Function) {
	t.Helper()

	const n = 256
	a := make([]float32, n)
	b := make([]float32, n)
	for i := range a {
		a[i] = float32(i)
		b[i] = float32(2 * i)
	}
	da, err := cuda.Upload(ctx, a)
	if err != nil {
		t.Fatalf("Upload a: %v", err)
	}
	defer da.Free()
	db, err := cuda.Upload(ctx, b)
	if err != nil {
		t.Fatalf("Upload b: %v", err)
	}
	defer db.Free()
	dc, err := cuda.NewSlice[float32](ctx, n)
	if err != nil {
		t.Fatalf("NewSlice: %v", err)
	}
	defer dc.Free()

	const block = 64
	if err := fn.LaunchSync(cuda.D1(n/block), cuda.D1(block), 0,
		dc.Arg(), da.Arg(), db.Arg(), cuda.ArgI32(n)); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	got, err := dc.Download()
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	for i := range got {
		if want := a[i] + b[i]; got[i] != want {
			t.Fatalf("c[%d] = %v, want %v", i, got[i], want)
		}
	}
}

// TestMaxSharedMemPerBlock checks that the device's shared memory limit is
// actually queried. Zero is the documented "unknown" answer of the build
// without a device, so a real device must report more than that.
func TestMaxSharedMemPerBlock(t *testing.T) {
	requireDevice(t)
	ctx, err := cuda.NewContext(0)
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Close()

	got := ctx.MaxSharedMemPerBlock()
	t.Logf("max shared memory per block: %d bytes", got)
	if got <= 0 {
		t.Fatalf("MaxSharedMemPerBlock() = %d, want a positive limit on a real device", got)
	}
	// Every CUDA device since compute 1.0 offers at least 16 KiB per block; a
	// smaller answer means the attribute was misread rather than unsupported.
	if got < 16*1024 {
		t.Errorf("MaxSharedMemPerBlock() = %d, implausibly small", got)
	}
}

// TestModuleCacheIsPerContext pins down who owns a cached module.
//
// The cache used to be a package-level map in internal/jit keyed on the source
// hash alone, so it survived the contexts it cached for: closing a context and
// opening a new one returned the dead context's module, and looking a function
// up in it worked on a stale handle. Here the same key is asked for in two
// contexts, and the builder must run again for the second one.
func TestModuleCacheIsPerContext(t *testing.T) {
	requireDevice(t)
	first, err := cuda.NewContext(0)
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	ptx, err := cuda.Compile(addSrc, "add.cu", first.Arch())
	if err != nil {
		first.Close()
		t.Fatalf("Compile: %v", err)
	}

	builds := 0
	build := func() ([]byte, error) {
		builds++
		return ptx.Bytes, nil
	}

	modA, err := first.LoadPTXCached("add", build)
	if err != nil {
		first.Close()
		t.Fatalf("LoadPTXCached (miss): %v", err)
	}
	if builds != 1 {
		t.Fatalf("builds = %d after the first load, want 1", builds)
	}
	modB, err := first.LoadPTXCached("add", build)
	if err != nil {
		first.Close()
		t.Fatalf("LoadPTXCached (hit): %v", err)
	}
	if builds != 1 {
		t.Errorf("builds = %d after a repeated load in the same context, want 1", builds)
	}
	if modB != modA {
		t.Errorf("a cache hit returned a different module than the miss")
	}

	if err := first.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	// Close is called from t.Cleanup in other packages while the same context
	// may still be closed by a defer; the second call must not release the
	// primary context a second time.
	if err := first.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}

	second, err := cuda.NewContext(0)
	if err != nil {
		t.Fatalf("NewContext (second): %v", err)
	}
	defer second.Close()

	modC, err := second.LoadPTXCached("add", build)
	if err != nil {
		t.Fatalf("LoadPTXCached in a fresh context: %v", err)
	}
	if builds != 2 {
		t.Errorf("builds = %d after loading in a fresh context, want 2: the module of the closed context was reused", builds)
	}
	if modC == modA {
		t.Fatal("a fresh context handed back the closed context's module")
	}

	fn, err := modC.Function("add")
	if err != nil {
		t.Fatalf("Function: %v", err)
	}
	runAdd(t, second, fn)

	// An explicit Unload drops the module from the cache as well, so the next
	// request for the key rebuilds rather than serving an unloaded module.
	if err := modC.Unload(); err != nil {
		t.Fatalf("Unload: %v", err)
	}
	if err := modC.Unload(); err != nil {
		t.Fatalf("second Unload: %v", err)
	}
	if _, err := second.LoadPTXCached("add", build); err != nil {
		t.Fatalf("LoadPTXCached after Unload: %v", err)
	}
	if builds != 3 {
		t.Errorf("builds = %d after reloading an unloaded key, want 3", builds)
	}
}

// TestJITLoadAcrossContexts is the same ownership question one level up, at
// the API simt and tile actually call, and it also covers what a cache hit
// reports: PTX and the compiler log used to be returned on the first build of
// a source and silently dropped on every later one.
func TestJITLoadAcrossContexts(t *testing.T) {
	requireDevice(t)
	// A Request with no cache directory writes no dumps, which is what this
	// test wants: nothing here is worth reading afterwards.
	req := jit.Request{Src: addSrc, Name: "add"}

	first, err := cuda.NewContext(0)
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	firstRes, err := jit.Load(first, req)
	if err != nil {
		first.Close()
		t.Fatalf("Load: %v", err)
	}
	if len(firstRes.PTX) == 0 {
		first.Close()
		t.Fatal("Load returned no PTX on a compile")
	}

	hit, err := jit.Load(first, req)
	if err != nil {
		first.Close()
		t.Fatalf("Load (cache hit): %v", err)
	}
	if !bytes.Equal(hit.PTX, firstRes.PTX) {
		first.Close()
		t.Fatalf("a cache hit reported %d bytes of PTX, want the %d bytes the compile produced",
			len(hit.PTX), len(firstRes.PTX))
	}
	if hit.Log != firstRes.Log {
		t.Errorf("a cache hit reported log %q, want %q", hit.Log, firstRes.Log)
	}
	if err := first.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	second, err := cuda.NewContext(0)
	if err != nil {
		t.Fatalf("NewContext (second): %v", err)
	}
	defer second.Close()

	// With a process-wide cache this call returned the module loaded into the
	// context that has just been released, and the launch below ran on a stale
	// handle. It must compile and load into the new context instead.
	res, err := jit.Load(second, req)
	if err != nil {
		t.Fatalf("Load in a fresh context: %v", err)
	}
	if len(res.PTX) == 0 {
		t.Error("Load in a fresh context returned no PTX")
	}
	runAdd(t, second, res.Func)
}

// TestContextIsGoroutineSafe is the regression test for the one thing a
// caller is most likely to try first: sharing a *Context between goroutines.
//
// The mechanism is not a data race and the race detector cannot see it.
// cuCtxSetCurrent binds a context to the *calling OS thread*, and Go may move
// a goroutine to another thread at any preemption point; a driver call that
// lands on a thread the context was never made current on is answered with
// CUDA_ERROR_INVALID_CONTEXT. The same bug showed up as a rare spurious
// failure of examples/tilefir.
//
// Concurrency alone does not reproduce it, and that was measured rather than
// assumed: 64 goroutines doing 40 Upload/Download round trips each pass every
// time on the unfixed code. The reason is that cuCtxSetCurrent is sticky --
// once a thread has been made current it stays current -- so in a steady
// thread pool every migration lands somewhere already bound. What is needed
// is a thread that has *never* bound, which means a thread the runtime
// created after the work started.
//
// Hence the shredder. A goroutine that exits while still holding
// runtime.LockOSThread takes its OS thread down with it, so the loop below
// retires threads continuously and obliges the scheduler to make fresh ones
// for the workers. On the unfixed code every worker fails, inside about a
// second; with the fix there is no window to land in.
func TestContextIsGoroutineSafe(t *testing.T) {
	requireDevice(t)
	if n := runtime.GOMAXPROCS(0); n < 2 {
		t.Skipf("GOMAXPROCS = %d: a goroutine cannot migrate between threads", n)
	}

	ctx, err := cuda.NewContext(0)
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Close()

	var stop atomic.Bool
	var shredder sync.WaitGroup
	shredder.Add(1)
	go func() {
		defer shredder.Done()
		for !stop.Load() {
			var batch sync.WaitGroup
			for range 8 {
				batch.Add(1)
				go func() {
					defer batch.Done()
					runtime.LockOSThread() // exits locked, so the thread dies
				}()
			}
			batch.Wait()
		}
	}()

	const (
		workers = 32
		rounds  = 50
	)
	want := []float32{1, 2, 3, 4, 5, 6, 7, 8}

	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for w := range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for r := range rounds {
				s, err := cuda.Upload(ctx, want)
				if err != nil {
					errs <- fmt.Errorf("worker %d round %d: Upload: %w", w, r, err)
					return
				}
				got, err := s.Download()
				s.Free()
				if err != nil {
					errs <- fmt.Errorf("worker %d round %d: Download: %w", w, r, err)
					return
				}
				if !slices.Equal(got, want) {
					errs <- fmt.Errorf("worker %d round %d: got %v, want %v", w, r, got, want)
					return
				}
			}
		}()
	}
	wg.Wait()
	stop.Store(true)
	shredder.Wait()
	close(errs)

	// Report the first failure with its type intact and only count the rest:
	// a broken binding produces one per worker and they all say the same
	// thing.
	n := 0
	for err := range errs {
		if n == 0 {
			t.Errorf("%v", err)
			if errors.Is(err, cuda.ErrInvalidContext) {
				t.Log("CUDA_ERROR_INVALID_CONTEXT: a worker ran on a thread the context was never made current on")
			}
		}
		n++
	}
	if n > 1 {
		t.Errorf("%d of %d workers failed", n, workers)
	}
}

// TestParseArch covers the reading of a virtual architecture, including the
// two-digit assumption that used to be implicit: "compute_100" is compute
// capability 10.0, and splitting it as one major and two minor digits would
// report a device three generations older than the one in the machine.
func TestParseArch(t *testing.T) {
	ok := []struct {
		in           string
		major, minor int
	}{
		{"compute_75", 7, 5},
		{"compute_100", 10, 0},
		{"compute_61", 6, 1},
		{"compute_90", 9, 0},
		{"compute_120", 12, 0},
	}
	for _, tc := range ok {
		major, minor, err := cuda.ParseArch(tc.in)
		if err != nil {
			t.Errorf("ParseArch(%q): %v", tc.in, err)
			continue
		}
		if major != tc.major || minor != tc.minor {
			t.Errorf("ParseArch(%q) = (%d, %d), want (%d, %d)", tc.in, major, minor, tc.major, tc.minor)
		}
	}

	bad := []struct {
		in  string
		why string
	}{
		{"compute_90a", "arch-conditional targets are not forward compatible"},
		{"compute_100f", "family-conditional targets are not forward compatible"},
		{"sm_75", "a real architecture is not a virtual one"},
		{"compute_", "no compute capability at all"},
		{"compute_7x", "not all digits"},
		{"compute_7", "no minor digit"},
		{"", "empty"},
	}
	for _, tc := range bad {
		if major, minor, err := cuda.ParseArch(tc.in); err == nil {
			t.Errorf("ParseArch(%q) = (%d, %d), want an error: %s", tc.in, major, minor, tc.why)
		}
	}
}

// TestResultNames checks that a status code names itself without the driver,
// which is what lets these errors be printed in a build without the "cuda"
// tag and long after the call that produced them.
func TestResultNames(t *testing.T) {
	if got, want := cuda.ErrInvalidContext.Name(), "CUDA_ERROR_INVALID_CONTEXT"; got != want {
		t.Errorf("ErrInvalidContext.Name() = %q, want %q", got, want)
	}
	if got, want := cuda.Result(1234).Name(), "CUresult(1234)"; got != want {
		t.Errorf("Result(1234).Name() = %q, want %q", got, want)
	}
	if got := cuda.ErrInvalidContext.Error(); !strings.Contains(got, "CUDA_ERROR_INVALID_CONTEXT") {
		t.Errorf("ErrInvalidContext.Error() = %q, want it to name the code", got)
	}
	if got := cuda.Result(1234).Error(); !strings.Contains(got, "1234") {
		t.Errorf("Result(1234).Error() = %q, want it to carry the number", got)
	}
	if got, want := cuda.ErrNVRTCCompilation.Name(), "NVRTC_ERROR_COMPILATION"; got != want {
		t.Errorf("ErrNVRTCCompilation.Name() = %q, want %q", got, want)
	}
	if got, want := cuda.NVRTCResult(99).Name(), "nvrtcResult(99)"; got != want {
		t.Errorf("NVRTCResult(99).Name() = %q, want %q", got, want)
	}
}

// TestErrorIsAndAs is the point of the typed errors: a caller decides what a
// failure was from the code, not from the wording of the message, and can
// still get at the driver entry point that produced it.
func TestErrorIsAndAs(t *testing.T) {
	err := error(&cuda.Error{Op: "cuMemAlloc", Code: cuda.ErrInvalidContext, Desc: "invalid device context"})
	if !errors.Is(err, cuda.ErrInvalidContext) {
		t.Errorf("errors.Is(%v, ErrInvalidContext) = false", err)
	}
	if errors.Is(err, cuda.ErrOutOfMemory) {
		t.Errorf("errors.Is(%v, ErrOutOfMemory) = true", err)
	}

	// Wrapping is what internal/jit does to say which kernel failed, so the
	// recovery has to survive it.
	wrapped := fmt.Errorf("launching k: %w", err)
	var drvErr *cuda.Error
	if !errors.As(wrapped, &drvErr) {
		t.Fatalf("errors.As(%v, *cuda.Error) = false", wrapped)
	}
	if drvErr.Op != "cuMemAlloc" {
		t.Errorf("recovered Op = %q, want %q", drvErr.Op, "cuMemAlloc")
	}
	if !errors.Is(wrapped, cuda.ErrInvalidContext) {
		t.Errorf("errors.Is(%v, ErrInvalidContext) = false through a wrap", wrapped)
	}

	// A CompileError keeps the log in its message, since that is all a failed
	// NVRTC run has to offer.
	compile := error(&cuda.CompileError{
		Name: "add.cu", Arch: "compute_75",
		Code: cuda.ErrNVRTCCompilation,
		Log:  `add.cu(1): error: identifier "nope" is undefined`,
	})
	if !errors.Is(compile, cuda.ErrNVRTCCompilation) {
		t.Errorf("errors.Is(%v, ErrNVRTCCompilation) = false", compile)
	}
	if !strings.Contains(compile.Error(), "identifier") {
		t.Errorf("CompileError.Error() = %q, want the compiler log in it", compile.Error())
	}
}
