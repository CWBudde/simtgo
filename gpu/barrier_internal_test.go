package gpu

import (
	"strings"
	"testing"
	"time"
)

// TestBlockStallIsDiagnosedRatherThanDeadlocked is why the block barrier has a
// deadline, and it is an internal test for the same reason its warp twin is:
// shortening that deadline is the only way to ask the question in under five
// seconds.
//
// The shape is a thread waiting at a barrier for one that is waiting in a warp
// rendezvous it will never reach. Neither can arrive, nothing has panicked,
// and before the deadline existed RunCPU simply never returned -- so the
// assertion is not only the message but that this test finishes at all.
func TestBlockStallIsDiagnosedRatherThanDeadlocked(t *testing.T) {
	saved := blockStallTimeout
	blockStallTimeout = 50 * time.Millisecond
	defer func() { blockStallTimeout = saved }()

	savedWarp := warpStallTimeout
	warpStallTimeout = 50 * time.Millisecond
	defer func() { warpStallTimeout = savedWarp }()

	done := make(chan string, 1)
	go func() {
		defer func() {
			r := recover()
			msg, _ := r.(string)
			done <- msg
		}()
		RunCPU(1, 2, func(ctx Ctx) {
			if ctx.ThreadIdx() == 0 {
				ctx.SyncThreads()
				return
			}
			// Thread 1 waits for its whole warp, which cannot assemble while
			// thread 0 is holding still at the barrier above.
			ctx.SyncWarp()
		})
	}()

	select {
	case msg := <-done:
		if !strings.Contains(msg, "gpu: ") {
			t.Fatalf("expected a gpu diagnosis, got %q", msg)
		}
		// Either side of the standoff may report first; both name the block
		// and the thread, and either is a diagnosis rather than a hang.
		if !strings.Contains(msg, "block 0") {
			t.Errorf("diagnosis does not name the block: %q", msg)
		}
	case <-time.After(20 * time.Second):
		t.Fatal("RunCPU never returned: a stalled block barrier is deadlocking rather than reporting")
	}
}

// TestBarrierReleasedByAThreadThatReturns pins the half of the fix that is not
// about the deadline at all.
//
// A thread that leaves the kernel early must release the barrier, exactly as
// it releases its warp, because CUDA does not require an exited thread to
// reach one either. Without that, this launch waits out the full deadline and
// reports a stall -- so the assertion is that it finishes quickly and says
// nothing.
func TestBarrierReleasedByAThreadThatReturns(t *testing.T) {
	saved := blockStallTimeout
	blockStallTimeout = 30 * time.Second // long, so a release cannot be mistaken for a timeout
	defer func() { blockStallTimeout = saved }()

	done := make(chan struct{})
	go func() {
		defer close(done)
		out := make([]int32, 8)
		RunCPU(1, 8, func(ctx Ctx) {
			if ctx.ThreadIdx() >= 4 {
				return // half the block leaves before the barrier
			}
			ctx.SyncThreads()
			out[ctx.ThreadIdx()] = 1
		})
		for i := range 4 {
			if out[i] != 1 {
				panic("a thread did not get past the barrier")
			}
		}
	}()

	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("threads that returned early did not release the barrier")
	}
}
