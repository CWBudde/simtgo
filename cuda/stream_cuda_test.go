//go:build cuda

package cuda_test

import (
	"context"
	"errors"
	"runtime"
	"testing"
	"time"

	"github.com/CWBudde/simtgo/cuda"
)

// spinSrc is a kernel with a dial on it.
//
// Every test below needs work that is still running when the host looks, and
// the add kernel finishes far too fast for that. The loop carries a
// dependency through x so that neither NVRTC nor ptxas can remove it, and
// iters turns microseconds into milliseconds without changing the amount of
// memory touched -- which matters for the overlap benchmark, where the copy
// and the compute have to be separately tunable.
const spinSrc = `
extern "C" __global__ void spin(float* out, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = out[i];
    for (int k = 0; k < iters; k++) x = x * 1.0000001f + 1e-7f;
    out[i] = x;
}`

// spinKernel compiles spinSrc once for the calling test.
func spinKernel(tb testing.TB, dev *cuda.Context) *cuda.Function {
	tb.Helper()
	ptx, err := cuda.Compile(spinSrc, "spin.cu", dev.Arch())
	if err != nil {
		tb.Fatalf("Compile: %v", err)
	}
	mod, err := dev.LoadPTX(ptx.Bytes)
	if err != nil {
		tb.Fatalf("LoadPTX: %v", err)
	}
	fn, err := mod.Function("spin")
	if err != nil {
		tb.Fatalf("Function: %v", err)
	}
	return fn
}

const spinBlock = 256

func spinGrid(n int) (grid, block cuda.Dim3) {
	return cuda.D1((n + spinBlock - 1) / spinBlock), cuda.D1(spinBlock)
}

// zeroed allocates a device buffer with defined contents.
//
// spin reads out[i] before it writes it, so launching it over a fresh
// cuMemAlloc is a genuine read of uninitialised device memory -- which
// compute-sanitizer's initcheck reports, correctly, once per element. It is
// the tests that were wrong and not the tool: a kernel reading what nobody
// wrote is exactly what initcheck exists to find, and a test that trips it
// would mask a real one later.
func zeroed(tb testing.TB, dev *cuda.Context, n int) *cuda.Slice[float32] {
	tb.Helper()
	d, err := cuda.NewSlice[float32](dev, n)
	if err != nil {
		tb.Fatalf("NewSlice: %v", err)
	}
	if err := d.CopyFrom(make([]float32, n)); err != nil {
		tb.Fatalf("CopyFrom: %v", err)
	}
	tb.Cleanup(d.Free)
	return d
}

// TestStreamAsyncRoundTrip runs the whole asynchronous path end to end:
// queue a copy up, a kernel, and a copy back, all without blocking, and read
// the answer only after the stream says it is finished.
func TestStreamAsyncRoundTrip(t *testing.T) {
	dev := device(t)
	fn := spinKernel(t, dev)
	const n = 64 * 1024

	st, err := dev.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	defer st.Close()

	host, err := cuda.NewHostSlice[float32](dev, n)
	if err != nil {
		t.Fatalf("NewHostSlice: %v", err)
	}
	defer host.Free()
	for i := range host.Slice() {
		host.Slice()[i] = 1
	}

	d, err := cuda.NewSlice[float32](dev, n)
	if err != nil {
		t.Fatalf("NewSlice: %v", err)
	}
	defer d.Free()

	if err := d.UploadAsync(st, host); err != nil {
		t.Fatalf("UploadAsync: %v", err)
	}
	grid, block := spinGrid(n)
	if err := fn.Launch(st, grid, block, 0, d.Arg(), cuda.ArgI32(n), cuda.ArgI32(1000)); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	if err := d.DownloadAsync(st, host); err != nil {
		t.Fatalf("DownloadAsync: %v", err)
	}
	if err := st.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}

	// 1000 iterations of x*1.0000001 + 1e-7 starting from 1: the exact value
	// is not the point, only that the kernel ran on the uploaded data and the
	// result came back. An untouched buffer would still read 1.
	for i, got := range host.Slice() {
		if got == 1 || got < 1 || got > 2 {
			t.Fatalf("element %d = %v: the kernel did not run over the uploaded data", i, got)
		}
	}
}

// TestStreamDoneReportsOutstandingWork is what Wait is built on, so it is
// worth pinning on its own: a stream with a long kernel on it must say it is
// not finished, and must say it is once it has been synchronised.
func TestStreamDoneReportsOutstandingWork(t *testing.T) {
	dev := device(t)
	fn := spinKernel(t, dev)
	const n = 1 << 20

	st, err := dev.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	defer st.Close()
	d := zeroed(t, dev, n)

	grid, block := spinGrid(n)
	if err := fn.Launch(st, grid, block, 0, d.Arg(), cuda.ArgI32(n), cuda.ArgI32(200000)); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	// Not asserted as "must be false": a fast enough device could finish
	// between the launch and the query, and a test that fails on a fast
	// machine is worse than one that logs. What is asserted is the true, and
	// that Done never lies in a way that costs correctness.
	if done, err := st.Done(); err != nil {
		t.Fatalf("Done: %v", err)
	} else if done {
		t.Log("the kernel finished before the first query; the timing half of this test did not run")
	}

	if err := st.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	if done, err := st.Done(); err != nil {
		t.Fatalf("Done after Sync: %v", err)
	} else if !done {
		t.Error("Done reports outstanding work after Sync returned")
	}
}

// TestStreamWaitReturnsWhenTheStreamDrains is the ordinary path: a context
// that is never cancelled waits exactly as Sync would.
func TestStreamWaitReturnsWhenTheStreamDrains(t *testing.T) {
	dev := device(t)
	fn := spinKernel(t, dev)
	const n = 1 << 18

	st, err := dev.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	defer st.Close()
	d := zeroed(t, dev, n)

	grid, block := spinGrid(n)
	if err := fn.Launch(st, grid, block, 0, d.Arg(), cuda.ArgI32(n), cuda.ArgI32(20000)); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	if err := st.Wait(t.Context()); err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if done, err := st.Done(); err != nil || !done {
		t.Errorf("Wait returned but the stream is not drained: done=%v err=%v", done, err)
	}
}

// TestStreamWaitCancelledLeavesTheStreamBusy is the whole contract of the
// cancellable wait, and the reason it was worth writing down twice.
//
// Cancelling stops the waiting and not the device. The stream must say so:
// Wait returns the context's error, Close refuses because the buffers are
// still in use, Sync lets the work finish, and only then does Close succeed.
// The alternative -- Close quietly destroying a stream whose kernel is still
// reading memory the caller is about to Free -- is a use-after-free that the
// host cannot see, because the memory belongs to the device.
func TestStreamWaitCancelledLeavesTheStreamBusy(t *testing.T) {
	dev := device(t)
	fn := spinKernel(t, dev)
	const n = 1 << 20

	st, err := dev.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	d := zeroed(t, dev, n)

	grid, block := spinGrid(n)
	if err := fn.Launch(st, grid, block, 0, d.Arg(), cuda.ArgI32(n), cuda.ArgI32(400000)); err != nil {
		t.Fatalf("Launch: %v", err)
	}

	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Millisecond)
	defer cancel()
	err = st.Wait(ctx)
	if err == nil {
		// The kernel beat the deadline. Nothing below can be tested, and
		// saying so is better than tightening the timeout until the test is
		// flaky on somebody else's machine.
		t.Skip("the kernel finished inside the deadline; raise iters to exercise cancellation here")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Wait = %v, want context.DeadlineExceeded", err)
	}

	var busy *cuda.BusyStreamError
	if err := st.Close(); !errors.As(err, &busy) {
		t.Fatalf("Close after a cancelled Wait = %v, want a *cuda.BusyStreamError", err)
	}

	if err := st.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	if err := st.Close(); err != nil {
		t.Errorf("Close after Sync = %v, want nil", err)
	}
	if err := st.Close(); err != nil {
		t.Errorf("Close is not idempotent: %v", err)
	}
}

// TestStreamCloseAbandonedAcceptsOutstandingWork covers the escape hatch,
// and that it is the only way past the refusal.
func TestStreamCloseAbandonedAcceptsOutstandingWork(t *testing.T) {
	dev := device(t)
	fn := spinKernel(t, dev)
	const n = 1 << 20

	st, err := dev.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	d := zeroed(t, dev, n)

	grid, block := spinGrid(n)
	if err := fn.Launch(st, grid, block, 0, d.Arg(), cuda.ArgI32(n), cuda.ArgI32(400000)); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Millisecond)
	defer cancel()
	if err := st.Wait(ctx); err == nil {
		t.Skip("the kernel finished inside the deadline")
	}
	if err := st.CloseAbandoned(); err != nil {
		t.Errorf("CloseAbandoned = %v, want nil", err)
	}
	// The buffer is deliberately not freed while the kernel may still be
	// reading it: that is what "abandoned" means, and freeing it here would
	// be the exact use-after-free the refusal exists to prevent. The context
	// is closed by the test's cleanup, which waits for the device.
	if err := dev.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	d.Free()
}

// TestEventTimesTheDevice checks the one thing a host clock cannot do.
//
// time.Since around an asynchronous launch measures how long it took to
// queue, which on a warm context is a few microseconds whatever the kernel
// does. Two events either side measure the device.
func TestEventTimesTheDevice(t *testing.T) {
	dev := device(t)
	fn := spinKernel(t, dev)
	const n = 1 << 18

	st, err := dev.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	defer st.Close()
	d := zeroed(t, dev, n)

	start, err := dev.NewEvent()
	if err != nil {
		t.Fatalf("NewEvent: %v", err)
	}
	defer start.Close()
	end, err := dev.NewEvent()
	if err != nil {
		t.Fatalf("NewEvent: %v", err)
	}
	defer end.Close()

	grid, block := spinGrid(n)
	if err := start.Record(st); err != nil {
		t.Fatalf("Record start: %v", err)
	}
	if err := fn.Launch(st, grid, block, 0, d.Arg(), cuda.ArgI32(n), cuda.ArgI32(50000)); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	if err := end.Record(st); err != nil {
		t.Fatalf("Record end: %v", err)
	}
	if err := end.Sync(); err != nil {
		t.Fatalf("event Sync: %v", err)
	}

	d1, err := cuda.Elapsed(start, end)
	if err != nil {
		t.Fatalf("Elapsed: %v", err)
	}
	if d1 <= 0 {
		t.Errorf("Elapsed = %v, want a positive duration", d1)
	}
	t.Logf("the kernel took %v on the device", d1)

	// Reusing an event overwrites its mark rather than erroring, which is
	// what makes a pair of them usable inside a loop.
	if err := start.Record(st); err != nil {
		t.Errorf("re-recording an event: %v", err)
	}
	if err := st.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	if err := start.Close(); err != nil {
		t.Errorf("Close: %v", err)
	}
	if err := start.Close(); err != nil {
		t.Errorf("Close is not idempotent: %v", err)
	}
}

// TestLaunchAsyncSurvivesGC checks the claim the shared launch path makes in
// a comment, because it is the one a reader should not have to take on trust.
//
// LaunchSync pins the parameter block only for the duration of
// cuLaunchKernel, and Launch inherits that -- correct only if the driver has
// copied the parameters out by the time the call returns. If it had not, a
// collection between the launch and the kernel actually running would be free
// to reuse that memory and the kernel would read garbage. So: launch, collect
// hard, then check the answer.
func TestLaunchAsyncSurvivesGC(t *testing.T) {
	dev := device(t)
	fn := spinKernel(t, dev)
	const n = 1 << 16

	st, err := dev.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	defer st.Close()
	host, err := cuda.NewHostSlice[float32](dev, n)
	if err != nil {
		t.Fatalf("NewHostSlice: %v", err)
	}
	defer host.Free()
	for i := range host.Slice() {
		host.Slice()[i] = 1
	}
	d, err := cuda.NewSlice[float32](dev, n)
	if err != nil {
		t.Fatalf("NewSlice: %v", err)
	}
	defer d.Free()
	if err := d.CopyFrom(host.Slice()); err != nil {
		t.Fatalf("CopyFrom: %v", err)
	}

	grid, block := spinGrid(n)
	if err := fn.Launch(st, grid, block, 0, d.Arg(), cuda.ArgI32(n), cuda.ArgI32(100000)); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	// Churn the heap while the kernel is in flight. Anything the launch left
	// behind on the Go heap is now fair game for reuse.
	for range 5 {
		sink := make([][]byte, 64)
		for i := range sink {
			sink[i] = make([]byte, 64*1024)
		}
		runtime.GC()
	}
	if err := st.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	if err := d.CopyTo(host.Slice()); err != nil {
		t.Fatalf("CopyTo: %v", err)
	}
	for i, got := range host.Slice() {
		if got == 1 || got < 1 || got > 2 {
			t.Fatalf("element %d = %v: the launch read the wrong parameters", i, got)
		}
	}
}

func TestAsyncCopyRefusesTheWrongLength(t *testing.T) {
	dev := device(t)
	st, err := dev.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	defer st.Close()

	d, err := cuda.NewSlice[float32](dev, 16)
	if err != nil {
		t.Fatalf("NewSlice: %v", err)
	}
	defer d.Free()
	h, err := cuda.NewHostSlice[float32](dev, 8)
	if err != nil {
		t.Fatalf("NewHostSlice: %v", err)
	}
	defer h.Free()

	var e *cuda.LengthError
	if err := d.UploadAsync(st, h); !errors.As(err, &e) {
		t.Errorf("UploadAsync = %v, want a *cuda.LengthError", err)
	}
	if err := d.DownloadAsync(st, h); !errors.As(err, &e) {
		t.Errorf("DownloadAsync = %v, want a *cuda.LengthError", err)
	}
}

// BenchmarkOverlap is the measurement the whole item exists for: does queuing
// the copies and the kernel on streams actually let them happen at the same
// time?
//
// Both cases move the same bytes, run the same kernel over the same chunks,
// and use page-locked host memory. The only difference is that "serial" waits
// for each step before starting the next -- upload, compute, download, next
// chunk -- while "overlapped" hands the chunks to a small pool of streams and
// waits once at the end, so chunk 2's upload can run while chunk 1 computes
// and chunk 0 comes back.
//
// The device decides whether that is possible. This one reports
// ASYNC_ENGINE_COUNT = 3 and CONCURRENT_KERNELS = 1, so the three directions
// really can run at once; a device with one copy engine would show much less,
// and the numbers in docs/toolchain.md name the machine for that reason.
func BenchmarkOverlap(b *testing.B) {
	dev := device(b)
	fn := spinKernel(b, dev)

	const (
		chunks     = 16
		perChunk   = 1 << 20 // 4 MiB of float32 per chunk, 64 MiB in all
		iters      = 3000    // tuned so compute and copy are the same order
		streamPool = 4
	)

	host := make([]*cuda.HostSlice[float32], chunks)
	devb := make([]*cuda.Slice[float32], chunks)
	for i := range chunks {
		h, err := cuda.NewHostSlice[float32](dev, perChunk)
		if err != nil {
			b.Fatalf("NewHostSlice: %v", err)
		}
		defer h.Free()
		for j := range h.Slice() {
			h.Slice()[j] = 1
		}
		host[i] = h

		d, err := cuda.NewSlice[float32](dev, perChunk)
		if err != nil {
			b.Fatalf("NewSlice: %v", err)
		}
		defer d.Free()
		devb[i] = d
	}
	grid, block := spinGrid(perChunk)
	bytes := int64(chunks) * perChunk * 4 * 2 // up and back

	b.Run("serial", func(b *testing.B) {
		b.SetBytes(bytes)
		for b.Loop() {
			for i := range chunks {
				if err := devb[i].CopyFrom(host[i].Slice()); err != nil {
					b.Fatalf("CopyFrom: %v", err)
				}
				if err := fn.LaunchSync(grid, block, 0, devb[i].Arg(), cuda.ArgI32(perChunk), cuda.ArgI32(iters)); err != nil {
					b.Fatalf("LaunchSync: %v", err)
				}
				if err := devb[i].CopyTo(host[i].Slice()); err != nil {
					b.Fatalf("CopyTo: %v", err)
				}
			}
		}
	})

	b.Run("overlapped", func(b *testing.B) {
		streams := make([]*cuda.Stream, streamPool)
		for i := range streams {
			s, err := dev.NewStream()
			if err != nil {
				b.Fatalf("NewStream: %v", err)
			}
			defer s.Close()
			streams[i] = s
		}
		b.SetBytes(bytes)
		for b.Loop() {
			for i := range chunks {
				s := streams[i%streamPool]
				if err := devb[i].UploadAsync(s, host[i]); err != nil {
					b.Fatalf("UploadAsync: %v", err)
				}
				if err := fn.Launch(s, grid, block, 0, devb[i].Arg(), cuda.ArgI32(perChunk), cuda.ArgI32(iters)); err != nil {
					b.Fatalf("Launch: %v", err)
				}
				if err := devb[i].DownloadAsync(s, host[i]); err != nil {
					b.Fatalf("DownloadAsync: %v", err)
				}
			}
			for _, s := range streams {
				if err := s.Sync(); err != nil {
					b.Fatalf("Sync: %v", err)
				}
			}
		}
	})
}

// TestNilArgumentsAreRefusedNotPanicked is the third review finding. These
// entry points dereferenced their *Stream, *HostSlice and *Event without
// looking, so a caller who dropped an error from NewStream got a panic from
// inside the package rather than an error naming the argument -- and
// simt.Kernel.LaunchOn already refused a nil stream properly, which made the
// pair inconsistent as well as unhelpful.
func TestNilArgumentsAreRefusedNotPanicked(t *testing.T) {
	dev := device(t)
	fn := spinKernel(t, dev)
	st, err := dev.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	defer st.Close()

	d := zeroed(t, dev, 64)
	h, err := cuda.NewHostSlice[float32](dev, 64)
	if err != nil {
		t.Fatalf("NewHostSlice: %v", err)
	}
	defer h.Free()
	ev, err := dev.NewEvent()
	if err != nil {
		t.Fatalf("NewEvent: %v", err)
	}
	defer ev.Close()
	grid, block := spinGrid(64)

	for _, tc := range []struct {
		name string
		call func() error
		arg  string
	}{
		{"UploadAsync/stream", func() error { return d.UploadAsync(nil, h) }, "stream"},
		{"UploadAsync/src", func() error { return d.UploadAsync(st, nil) }, "src"},
		{"DownloadAsync/stream", func() error { return d.DownloadAsync(nil, h) }, "stream"},
		{"DownloadAsync/dst", func() error { return d.DownloadAsync(st, nil) }, "dst"},
		{"Launch/stream", func() error { return fn.Launch(nil, grid, block, 0, d.Arg(), cuda.ArgI32(64), cuda.ArgI32(1)) }, "stream"},
		{"Record/stream", func() error { return ev.Record(nil) }, "stream"},
		{"Elapsed/start", func() error { _, err := cuda.Elapsed(nil, ev); return err }, "start"},
		{"Elapsed/end", func() error { _, err := cuda.Elapsed(ev, nil); return err }, "end"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var e *cuda.NilError
			err := tc.call() // must not panic
			if !errors.As(err, &e) {
				t.Fatalf("got %v, want a *cuda.NilError", err)
			}
			if e.Arg != tc.arg {
				t.Errorf("Arg = %q, want %q", e.Arg, tc.arg)
			}
		})
	}

	// Nothing above may have left work on the stream.
	if done, err := st.Done(); err != nil || !done {
		t.Errorf("a refused call queued something: done=%v err=%v", done, err)
	}
}
