package gpu

import (
	"strings"
	"testing"
	"time"
)

// TestWarpStallIsDiagnosedRatherThanDeadlocked is why the rendezvous has a
// deadline at all, and it is an internal test because shortening that deadline
// is the only way to ask the question in under five seconds.
//
// Divergence that ends in a return releases the warp through abandon, which
// TestDivergenceThatExitsDoesNotHang covers. What nothing can release is a
// thread blocked somewhere else while its siblings wait for it: on the device
// that is __syncthreads() against a warp primitive, and here it is a channel,
// which is the same shape and can be made deterministic. The waiters must
// therefore give up and say so, because a test suite handed a deadlock has
// nothing at all to report.
//
// It is deterministic despite the clock: nothing can free the blocked threads
// until a waiter has already timed out and closed the channel.
func TestWarpStallIsDiagnosedRatherThanDeadlocked(t *testing.T) {
	saved := warpStallTimeout
	warpStallTimeout = 100 * time.Millisecond
	t.Cleanup(func() { warpStallTimeout = saved })

	blocked := make(chan struct{})
	msg := recovered(t, func() {
		RunCPU(1, 32, func(ctx Ctx) {
			if ctx.LaneID() >= 16 {
				<-blocked
				return
			}
			ctx.SyncWarp()
			if ctx.LaneID() == 0 {
				close(blocked)
			}
		})
	})
	for _, want := range []string{"warp rendezvous timed out", "SyncWarp", "warp 0 (32 lanes)"} {
		if !strings.Contains(msg, want) {
			t.Errorf("diagnosis does not mention %q:\n%s", want, msg)
		}
	}
}

// TestDivergentActiveMaskIsDiagnosed pins the one warp primitive whose CUDA
// counterpart is not collective.
//
// __activemask() may legally be called on one side of a divergent branch, so
// this kernel is valid CUDA and the divergence checker accepts it -- it
// refuses a divergent SyncThreads, which this is not. The emulator still
// cannot run it: ActiveMask is a rendezvous, the lanes that skipped it are
// held at the block barrier, and each half waits for the other. The test is
// here to keep that a reported stall rather than a hang, and to keep the
// message pointing at the conditional, which is the only thing the author can
// act on.
func TestDivergentActiveMaskIsDiagnosed(t *testing.T) {
	saved := warpStallTimeout
	warpStallTimeout = 100 * time.Millisecond
	t.Cleanup(func() { warpStallTimeout = saved })

	// Half of one warp, so the divergence is inside a warp rather than
	// between two of them: a branch on ThreadIdx that splits on a multiple of
	// WarpSize leaves every warp uniform and is not this bug.
	msg := recovered(t, func() {
		RunCPU(1, 64, func(ctx Ctx) {
			if ctx.LaneID() < 16 {
				_ = ctx.ActiveMask()
			}
			ctx.SyncThreads()
		})
	})
	for _, want := range []string{"warp rendezvous timed out", "ActiveMask", "Hoist the call out of the conditional"} {
		if !strings.Contains(msg, want) {
			t.Errorf("diagnosis does not mention %q:\n%s", want, msg)
		}
	}
}

// recovered runs fn and returns the string it panicked with. It duplicates the
// external tests' mustPanic because the two live in different packages, and
// this file is in package gpu only so that it can reach warpStallTimeout.
func recovered(t *testing.T, fn func()) string {
	t.Helper()
	var msg string
	func() {
		defer func() {
			r := recover()
			if r == nil {
				t.Fatal("did not panic, want a diagnosis")
			}
			s, ok := r.(string)
			if !ok {
				t.Fatalf("panicked with %T (%v), want a string diagnosis", r, r)
			}
			msg = s
		}()
		fn()
	}()
	return msg
}
