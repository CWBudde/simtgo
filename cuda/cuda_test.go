package cuda_test

import (
	"testing"

	"github.com/CWBudde/gocuda/cuda"
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
