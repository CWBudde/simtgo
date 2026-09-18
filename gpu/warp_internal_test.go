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
