package cuda_test

import (
	"bytes"
	"testing"

	"github.com/CWBudde/gocuda/cuda"
	"github.com/CWBudde/gocuda/internal/jit"
)

const addSrc = `
extern "C" __global__ void add(float* c, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}`

// TestRawKernel exercises the whole host path with a hand-written kernel:
// NVRTC compile, module load, upload, launch, download.
func TestRawKernel(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no CUDA device available")
	}
	ctx, err := cuda.NewContext(0)
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Close()
	t.Logf("device %q, arch %s", ctx.Name(), ctx.Arch())

	ptx, log, err := cuda.Compile(addSrc, "add.cu", ctx.Arch())
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	if log != "" {
		t.Logf("nvrtc log: %s", log)
	}
	mod, err := ctx.LoadPTX(ptx)
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
	if err := fn.Launch(cuda.D1(grid), cuda.D1(block), 0,
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
	if err := fn.Launch(cuda.D1(n/block), cuda.D1(block), 0,
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
	if !cuda.Available() {
		t.Skip("no CUDA device available")
	}
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
	if !cuda.Available() {
		t.Skip("no CUDA device available")
	}
	first, err := cuda.NewContext(0)
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	ptx, _, err := cuda.Compile(addSrc, "add.cu", first.Arch())
	if err != nil {
		first.Close()
		t.Fatalf("Compile: %v", err)
	}

	builds := 0
	build := func() ([]byte, any, error) {
		builds++
		return ptx, builds, nil
	}

	modA, extraA, err := first.LoadPTXCached("add", build)
	if err != nil {
		first.Close()
		t.Fatalf("LoadPTXCached (miss): %v", err)
	}
	if builds != 1 {
		t.Fatalf("builds = %d after the first load, want 1", builds)
	}
	modB, extraB, err := first.LoadPTXCached("add", build)
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
	if extraB != extraA {
		t.Errorf("extra = %v on the hit, want the %v the miss stored", extraB, extraA)
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

	modC, _, err := second.LoadPTXCached("add", build)
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
	if _, _, err := second.LoadPTXCached("add", build); err != nil {
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
	if !cuda.Available() {
		t.Skip("no CUDA device available")
	}
	// Keep the debug dumps out of the repository.
	old := jit.CacheDir
	jit.CacheDir = t.TempDir()
	defer func() { jit.CacheDir = old }()

	first, err := cuda.NewContext(0)
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	firstRes, err := jit.Load(first, addSrc, "add")
	if err != nil {
		first.Close()
		t.Fatalf("Load: %v", err)
	}
	if len(firstRes.PTX) == 0 {
		first.Close()
		t.Fatal("Load returned no PTX on a compile")
	}

	hit, err := jit.Load(first, addSrc, "add")
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
	res, err := jit.Load(second, addSrc, "add")
	if err != nil {
		t.Fatalf("Load in a fresh context: %v", err)
	}
	if len(res.PTX) == 0 {
		t.Error("Load in a fresh context returned no PTX")
	}
	runAdd(t, second, res.Func)
}
