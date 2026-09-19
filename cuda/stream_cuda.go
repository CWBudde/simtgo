//go:build cuda

package cuda

import (
	"context"
	"sync/atomic"
	"time"
)

// Streams and events: the asynchronous half of the driver API.
//
// Everything else in this package is synchronous -- the call returns when the
// device has finished. A stream is an ordered queue the device works through
// on its own: operations put on one stream happen in order relative to each
// other and in no particular order relative to another stream's, and the host
// carries on immediately. That is what lets a copy run while a kernel runs,
// which is the point, because the FIR example spends nine tenths of its time
// moving data and none of that time overlapped anything.
//
// The null stream is not exposed. Its rules are genuinely surprising -- work
// on it implicitly synchronises with every other blocking stream in the
// context -- and this package has no need of it: everything synchronous here
// already waits for itself. NewStream asks for a non-blocking stream, so the
// streams a caller creates are independent of each other and of whatever the
// null stream is doing.

// streamNonBlocking is CU_STREAM_NON_BLOCKING.
const streamNonBlocking = 1

// eventDefault is CU_EVENT_DEFAULT: timing enabled, a blocking-sync flag off
// (so cuEventSynchronize spins rather than descheduling, which is the right
// trade for the short waits this package has).
const eventDefault = 0

// errNotReady is CUDA_ERROR_NOT_READY, which cuStreamQuery returns for a
// stream with work outstanding. It is not exported as a sentinel because it
// is not an error at this level: Done turns it into a bool.
const errNotReady Result = 600

// Stream is an ordered queue of device work.
//
// A *Stream is safe to share between goroutines in the same sense a *Context
// is: each call is made with the context current on a locked OS thread, and
// none of them corrupt each other. What is not safe is the *ordering* two
// goroutines expect from interleaving their own work on one stream, which is
// theirs to arrange -- a stream orders what was put on it, in the order it
// was put on.
type Stream struct {
	s uintptr
	x *Context
	// abandoned records a Wait that returned before the stream drained,
	// which is the one state in which Close is dangerous. See Wait.
	abandoned atomic.Bool
	closed    atomic.Bool
}

// NewStream creates a stream in this context.
func (c *Context) NewStream() (*Stream, error) {
	s := &Stream{x: c}
	if err := c.call("cuStreamCreate", func() Result {
		return cuStreamCreate(&s.s, streamNonBlocking)
	}); err != nil {
		return nil, err
	}
	return s, nil
}

// Close destroys the stream.
//
// It refuses a stream whose Wait was cancelled and that has not drained since,
// with a *BusyStreamError. cuStreamDestroy itself is safe there -- the driver
// documents it as returning immediately and tearing the stream down once its
// work finishes -- but the caller's next line is almost always Free() on the
// buffers that work is still reading, and that is a use-after-free the
// sanitizer will report a long way from here. Refusing is how they find out
// in the right place. Sync first, or accept the outstanding work with
// CloseAbandoned.
//
// Close is idempotent.
func (s *Stream) Close() error {
	if s.closed.Load() {
		return nil
	}
	if s.abandoned.Load() {
		return &BusyStreamError{Op: "Close"}
	}
	return s.destroy()
}

// CloseAbandoned destroys the stream even though a cancelled Wait left work
// on it.
//
// This is the escape hatch for a caller who genuinely means it -- a process
// that is about to exit, or one that has deliberately leaked the buffers the
// outstanding work is using. It does not wait and it does not make the work
// safe; it only says that Close's refusal was understood.
func (s *Stream) CloseAbandoned() error {
	if s.closed.Load() {
		return nil
	}
	return s.destroy()
}

func (s *Stream) destroy() error {
	if !s.closed.CompareAndSwap(false, true) {
		return nil
	}
	// The handle is deliberately not zeroed. Doing so would be a write to a
	// plain field that Sync, Done and the async copies all read, so a Close
	// racing any of them would be a data race on the handle itself -- and
	// the payoff would only be turning a driver error into a nil handle.
	// Left alone, a stream used after Close is refused by the driver with
	// CUDA_ERROR_INVALID_HANDLE, which says more, and the CAS above is what
	// keeps the destroy itself exactly once.
	return s.x.call("cuStreamDestroy", func() Result { return cuStreamDestroy(s.s) })
}

// Sync blocks until every operation on the stream has completed.
//
// It cannot be interrupted -- cuStreamSynchronize has no timeout and no
// cancellation -- which is exactly why Wait exists and why Wait has to poll.
func (s *Stream) Sync() error {
	h := s.s
	if err := s.x.call("cuStreamSynchronize", func() Result { return cuStreamSynchronize(h) }); err != nil {
		return err
	}
	// Whatever a cancelled Wait left outstanding has now finished, so the
	// stream is ordinary again and Close will accept it.
	s.abandoned.Store(false)
	return nil
}

// Done reports whether the stream has finished everything put on it.
//
// A false with a nil error is the ordinary "still working" answer;
// CUDA_ERROR_NOT_READY is how the driver spells it and is not passed on as an
// error, because at this level it is not one.
func (s *Stream) Done() (bool, error) {
	h := s.s
	var r Result
	// Not through call's error conversion: NOT_READY has to be read before
	// check turns it into an *Error, and a genuine failure still has to go
	// through fault so a sticky code is recorded.
	if err := s.x.call("cuStreamQuery", func() Result {
		r = cuStreamQuery(h)
		if r == errNotReady {
			return Success
		}
		return r
	}); err != nil {
		return false, err
	}
	return r != errNotReady, nil
}

// Wait blocks until the stream drains or ctx is done, whichever comes first.
//
// # What cancelling does not do
//
// It stops the waiting. It does not stop the device. The kernels run to
// completion, the copies complete, and every buffer the stream is using stays
// in use until they do. There is no way to ask the driver to abandon queued
// work, so a cancelled Wait leaves the program in a state where Free on those
// buffers is a use-after-free -- which the host will not notice, because the
// memory is the device's.
//
// The stream remembers that this happened. Close then refuses with a
// *BusyStreamError rather than letting the mistake pass silently, and a later
// Sync clears the mark because at that point the work really has finished.
// That is the whole guarantee: a cancellation is honest about being a
// cancellation of the wait, and the one place it can be caught catches it.
//
// # Why it polls
//
// cuStreamSynchronize is uninterruptible, so a goroutine parked in it cannot
// be released by a closed channel; there is nothing to select on. The
// alternative -- parking a locked OS thread in cuStreamSynchronize and
// closing a channel from it -- would make the cancellation a lie in the other
// direction, since the thread stays there until the device is done anyway.
// So this queries, backing off from a busy first look to a 500 microsecond
// ceiling: work that is already finished costs one query, and work that takes
// a millisecond costs a handful.
func (s *Stream) Wait(ctx context.Context) error {
	const maxBackoff = 500 * time.Microsecond
	backoff := 10 * time.Microsecond
	timer := time.NewTimer(backoff)
	defer timer.Stop()
	for {
		done, err := s.Done()
		if err != nil {
			return err
		}
		if done {
			s.abandoned.Store(false)
			return nil
		}
		select {
		case <-ctx.Done():
			s.abandoned.Store(true)
			return ctx.Err()
		case <-timer.C:
		}
		if backoff < maxBackoff {
			backoff = min(backoff*2, maxBackoff)
		}
		timer.Reset(backoff)
	}
}

// UploadAsync queues a copy from page-locked host memory into this buffer.
//
// It returns as soon as the copy is queued, so src must not be written and
// dst must not be freed until the stream says the copy has finished.
//
// The source is a *HostSlice and not a []T, and that restriction is the
// reason HostSlice exists. Two things would go wrong with a Go slice: the
// driver falls back to a synchronous transfer for pageable memory, so the
// call would silently not be asynchronous at all; and the copy outlives the
// call, so runtime.Pinner -- which is how this package shows Go memory to the
// driver everywhere else -- cannot hold it still for long enough. See
// docs/decisions.md#an-asynchronous-copy-does-not-take-a-go-slice.
func (s *Slice[T]) UploadAsync(st *Stream, src *HostSlice[T]) error {
	if src.n != s.n {
		return &LengthError{Op: "UploadAsync", Want: s.n, Got: src.n}
	}
	if s.n == 0 {
		return nil
	}
	p, h, n := src.p, st.s, s.n*sizeOf[T]()
	return s.ctx.call("cuMemcpyHtoDAsync", func() Result {
		return cuMemcpyHtoDAsync(uint64(s.ptr), p, uint64(n), h)
	})
}

// DownloadAsync queues a copy from this buffer into page-locked host memory.
// See UploadAsync for why the destination cannot be a []T.
func (s *Slice[T]) DownloadAsync(st *Stream, dst *HostSlice[T]) error {
	if dst.n != s.n {
		return &LengthError{Op: "DownloadAsync", Want: s.n, Got: dst.n}
	}
	if s.n == 0 {
		return nil
	}
	p, h, n := dst.p, st.s, s.n*sizeOf[T]()
	return s.ctx.call("cuMemcpyDtoHAsync", func() Result {
		return cuMemcpyDtoHAsync(p, uint64(s.ptr), uint64(n), h)
	})
}

// Launch queues the kernel on a stream and returns without waiting.
//
// It is LaunchSync without the Sync, and the difference is where a device
// fault surfaces: LaunchSync reports one because it waits, and this cannot,
// so a fault raised by this launch is reported by the next Sync or Wait on
// the stream -- or by any other call on the context, since a sticky fault
// poisons all of it.
func (f *Function) Launch(s *Stream, grid, block Dim3, sharedBytes int, args ...Arg) error {
	return f.launch(s.s, grid, block, sharedBytes, args)
}

// Event marks a point in a stream. Recording one on each side of a region and
// asking for the time between them is how a launch is timed without a host
// clock that can only see the queueing.
type Event struct {
	e      uintptr
	x      *Context
	closed atomic.Bool
}

// NewEvent creates an event in this context.
func (c *Context) NewEvent() (*Event, error) {
	e := &Event{x: c}
	if err := c.call("cuEventCreate", func() Result {
		return cuEventCreate(&e.e, eventDefault)
	}); err != nil {
		return nil, err
	}
	return e, nil
}

// Close destroys the event. It is idempotent, and for the reason given at
// Stream.destroy the handle is left alone rather than zeroed.
func (e *Event) Close() error {
	if !e.closed.CompareAndSwap(false, true) {
		return nil
	}
	return e.x.call("cuEventDestroy", func() Result { return cuEventDestroy(e.e) })
}

// Record places the event in the stream, after everything queued so far.
// Recording the same event twice overwrites the first mark, which is what
// makes an event reusable across iterations of a loop.
func (e *Event) Record(s *Stream) error {
	h := s.s
	return e.x.call("cuEventRecord", func() Result { return cuEventRecord(e.e, h) })
}

// Sync blocks until the event has been reached.
func (e *Event) Sync() error {
	return e.x.call("cuEventSynchronize", func() Result { return cuEventSynchronize(e.e) })
}

// Elapsed reports the device time between two recorded events.
//
// This is device time, not host time, and that is the reason to use it: the
// host cannot see how long a kernel took, only how long it took to queue it.
// Both events must have been recorded and reached; the driver refuses with
// CUDA_ERROR_NOT_READY otherwise.
//
// The driver's resolution is about half a microsecond, and it reports a
// float32 of milliseconds, so the returned Duration is exact only to roughly
// that -- do not read nanoseconds off it.
func Elapsed(start, end *Event) (time.Duration, error) {
	var ms float32
	if err := start.x.call("cuEventElapsedTime", func() Result {
		return cuEventElapsedTime(&ms, start.e, end.e)
	}); err != nil {
		return 0, err
	}
	return time.Duration(float64(ms) * float64(time.Millisecond)), nil
}
