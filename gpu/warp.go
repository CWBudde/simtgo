package gpu

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// Warp-level primitives: the lanes of one warp exchanging values without
// going through memory.
//
// They are methods on Ctx rather than package-level functions, which the
// atomics beside them are. An atomic is handed the memory it works on, so it
// needs to know nothing about who is calling; a warp primitive is defined by
// *which* thread calls it, and under the emulator a thread is a goroutine with
// no identity a package-level function could recover. Ctx is that identity,
// and it is already how every other collective operation -- SyncThreads,
// SharedF32, AssumeBlockDim -- is spelled.
//
// None of them takes a participation mask, though every CUDA built-in they
// lower to does. The emitter writes 0xffffffff and the Go-level contract is
// that *every thread of the warp reaches the call*, which is the same trade
// the atomics made in taking a buffer and an index rather than a pointer: it
// gives one Go name exactly one built-in, and it gives both backends something
// they can check. Ballot and ActiveMask remain the way to read a mask.
//
// What the emulation cannot promise, since none of it is visible from the call
// site:
//
//   - A warp is threads 32w..32w+31 of a block by flat thread index, x fastest
//     -- which is how CUDA partitions a block -- so a block that is not a
//     multiple of 32 threads ends in a short warp. The emulator sizes that
//     warp at what is left. The device does not: it runs a full warp with the
//     missing lanes inactive, and 0xffffffff then names lanes that do not
//     exist, which CUDA leaves undefined. Launch warp-level kernels with a
//     multiple of 32 threads per block and the question does not arise;
//     AssumeBlockDim is how a kernel says so.
//   - Reading a lane that is not taking part returns the caller's own value
//     here. That matches what __shfl_up_sync and __shfl_down_sync do for a
//     delta that walks off the end of the warp, which is the case kernels
//     actually rely on. For an absent lane of a short warp, or one that has
//     already left the kernel, the device's value is undefined and this one is
//     merely definite: agreement between the two backends there is not
//     evidence of anything.
//   - ActiveMask returns the mask of threads that arrived at the rendezvous.
//     The device's __activemask() is the mask of threads converged at that
//     instruction, which depends on how the compiler scheduled the branches
//     around it and can differ from this for the same source. Use it for
//     diagnostics, not for correctness.
//   - There is no reconvergence model. A device warp executes both sides of a
//     divergent branch in lockstep; these threads are goroutines that reach
//     the rendezvous when they get there. A kernel whose threads take
//     different paths *to* a warp primitive is diagnosed here (see the stall
//     report below) rather than being given a plausible answer.

// WarpSize is how many threads a warp holds. It is 32 on every CUDA device to
// date, which is why it may be a constant here: a kernel that folds it into a
// shared-memory extent or a loop bound needs a number at compile time, and
// CUDA's own warpSize is an ordinary variable that no constant expression can
// use.
const WarpSize = 32

// warpStallTimeout bounds how long a thread waits for the rest of its warp.
//
// A warp primitive is a rendezvous, so the failure mode of a kernel whose
// threads disagree about reaching one is a wait for somebody who will never
// come. Threads that finish the kernel release their warp (see warpBarrier's
// abandon), which covers divergence that ends in a return; what it does not
// cover is a thread blocked somewhere else -- at __syncthreads(), typically,
// waiting for the very thread that is waiting for it. That is a genuine
// deadlock, and a deadlock is the worst thing a test suite can be handed, so
// the wait gives up and reports instead. It is a variable rather than a
// constant only so the tests can make the wait short.
var warpStallTimeout = 5 * time.Second

// LaneID is the thread's position within its warp, 0 to WarpSize-1.
//
// It is derived from the flat thread index rather than from threadIdx.x alone,
// because CUDA fills warps with consecutive flat indices (x fastest): in a
// 16x16 block, thread (0,2) is lane 0 of warp 1, not lane 0 of warp 0.
func (c Ctx) LaneID() int { return flat(c.tid, c.bdim) % WarpSize }

// ShuffleF32 returns the value v held by lane srcLane of this warp
// (__shfl_sync).
//
// srcLane is taken modulo WarpSize, as the built-in does. A lane that is not
// taking part yields the caller's own v; see the note at the top of this file
// for why that is the emulator's answer and not the device's.
func (c Ctx) ShuffleF32(v float32, srcLane int) float32 {
	return f32(c.shuffle("ShuffleF32", bits32(v), srcLane&(WarpSize-1)))
}

// ShuffleI32 is ShuffleF32 for an int32 (__shfl_sync).
func (c Ctx) ShuffleI32(v int32, srcLane int) int32 {
	return int32(c.shuffle("ShuffleI32", uint32(v), srcLane&(WarpSize-1)))
}

// ShuffleXorF32 returns the value v held by lane (LaneID ^ laneMask)
// (__shfl_xor_sync), which is the butterfly exchange a reduction is built out
// of.
func (c Ctx) ShuffleXorF32(v float32, laneMask int) float32 {
	return f32(c.shuffle("ShuffleXorF32", bits32(v), c.LaneID()^laneMask))
}

// ShuffleXorI32 is ShuffleXorF32 for an int32 (__shfl_xor_sync).
func (c Ctx) ShuffleXorI32(v int32, laneMask int) int32 {
	return int32(c.shuffle("ShuffleXorI32", uint32(v), c.LaneID()^laneMask))
}

// ShuffleUpF32 returns the value v held by the lane delta below this one
// (__shfl_up_sync). The lowest delta lanes have no such neighbour and get
// their own v back, which is what the built-in does too.
func (c Ctx) ShuffleUpF32(v float32, delta int) float32 {
	return f32(c.shuffle("ShuffleUpF32", bits32(v), c.LaneID()-delta))
}

// ShuffleUpI32 is ShuffleUpF32 for an int32 (__shfl_up_sync).
func (c Ctx) ShuffleUpI32(v int32, delta int) int32 {
	return int32(c.shuffle("ShuffleUpI32", uint32(v), c.LaneID()-delta))
}

// ShuffleDownF32 returns the value v held by the lane delta above this one
// (__shfl_down_sync). The highest delta lanes have no such neighbour and get
// their own v back, which is what makes the halving reduction loop terminate
// without a special case.
func (c Ctx) ShuffleDownF32(v float32, delta int) float32 {
	return f32(c.shuffle("ShuffleDownF32", bits32(v), c.LaneID()+delta))
}

// ShuffleDownI32 is ShuffleDownF32 for an int32 (__shfl_down_sync).
func (c Ctx) ShuffleDownI32(v int32, delta int) int32 {
	return int32(c.shuffle("ShuffleDownI32", uint32(v), c.LaneID()+delta))
}

// Ballot returns one bit per lane of this warp, set where that lane passed a
// true predicate (__ballot_sync). Bit n is lane n; a lane that is not taking
// part contributes a zero.
func (c Ctx) Ballot(pred bool) uint32 {
	mask, _, _ := c.vote("Ballot", pred)
	return mask
}

// Any reports whether any lane of this warp passed a true predicate
// (__any_sync).
func (c Ctx) Any(pred bool) bool {
	mask, _, _ := c.vote("Any", pred)
	return mask != 0
}

// All reports whether every lane of this warp passed a true predicate
// (__all_sync).
//
// "Every lane" means every lane that took part, so in a short warp, or after
// some lanes have left the kernel, it is a vote among fewer threads than
// WarpSize -- exactly as the device's mask semantics would have it, and with
// the same caveat about which threads those are.
func (c Ctx) All(pred bool) bool {
	mask, here, ok := c.vote("All", pred)
	// A vote that never took place is false rather than vacuously true: the
	// launch is about to panic with the diagnosis, and the false is the
	// harmless answer of the two.
	return ok && mask == here
}

// ActiveMask returns one bit per lane of this warp that is taking part
// (__activemask). Under the emulator that is the set of threads that arrived
// at this rendezvous, which is not the same question the device answers; see
// the note at the top of this file.
func (c Ctx) ActiveMask() uint32 {
	w, lane := c.warp()
	if w == nil {
		return 1
	}
	mask, ok := w.meet(lane, "ActiveMask")
	if !ok {
		return 0
	}
	w.part(lane)
	return mask
}

// SyncWarp is __syncwarp(): a barrier across the threads of one warp.
//
// On the device it is also a memory fence within the warp, which the emulator
// does not model -- it has no store buffer to flush. It is here so that a
// kernel which reads shared memory written by its own warp can say so, and so
// that the same source runs on both backends.
func (c Ctx) SyncWarp() {
	w, lane := c.warp()
	if w == nil {
		return
	}
	if _, ok := w.meet(lane, "SyncWarp"); !ok {
		return
	}
	w.part(lane)
}

// warp resolves the caller's warp and lane, or (nil, 0) for a Ctx that belongs
// to no launch. A kernel body called as a plain Go function is one thread on
// its own, so every primitive degenerates to a warp of one rather than
// refusing to run at all -- the same choice SharedF32 makes in handing out a
// private buffer.
func (c Ctx) warp() (*warpState, int) {
	if c.thread == nil || c.thread.warp == nil {
		return nil, 0
	}
	return c.thread.warp, c.thread.lane
}

// shuffle is the rendezvous every Shuffle* method is: publish this lane's
// value, wait for the warp, read the lane asked for, wait again so that no
// thread can race ahead and overwrite a slot somebody is still reading.
func (c Ctx) shuffle(op string, v uint32, src int) uint32 {
	w, lane := c.warp()
	if w == nil {
		return v
	}
	w.vals[lane] = v
	mask, ok := w.meet(lane, op)
	if !ok {
		return v
	}
	got := v
	// An absent lane -- one that walked off the end of the warp, one that
	// never existed in a short warp, one that has left the kernel -- yields
	// the caller's own value.
	if src >= 0 && src < WarpSize && mask&(1<<uint(src)) != 0 {
		got = w.vals[src]
	}
	w.part(lane)
	return got
}

// vote is the rendezvous behind Ballot, Any and All. It returns the predicates
// of the warp as a mask and the set of lanes that took part, because All is a
// question about both.
func (c Ctx) vote(op string, pred bool) (mask, here uint32, ok bool) {
	w, lane := c.warp()
	if w == nil {
		if pred {
			return 1, 1, true
		}
		return 0, 1, true
	}
	var bit uint32
	if pred {
		bit = 1
	}
	w.vals[lane] = bit
	here, ok = w.meet(lane, op)
	if !ok {
		return 0, 0, false
	}
	for l := range WarpSize {
		if here&(1<<uint(l)) != 0 && w.vals[l] != 0 {
			mask |= 1 << uint(l)
		}
	}
	w.part(lane)
	return mask, here, true
}

// warpState is the state the lanes of one warp share. It mirrors blockState
// one level down: the slots a rendezvous exchanges, and somewhere to record a
// diagnosis rather than panic on a thread's own goroutine.
type warpState struct {
	bar   *warpBarrier
	block *blockState
	bid   int // which block, for the diagnosis
	id    int // which warp of that block
	size  int // lanes this warp started with: 32, or what is left of a short block

	// vals and ops are written by the owning lane before it arrives and read
	// by its siblings only between the two waits, so the barrier is what
	// orders them. Each lane touches its own element and no other.
	vals [WarpSize]uint32
	ops  [WarpSize]string
}

// meet is the first half of a rendezvous: publish which operation this lane is
// performing, wait for the warp, and check that everybody agreed about what
// they came for. It returns the mask of lanes that arrived, and false when the
// warp never assembled -- in which case a diagnosis has been recorded and the
// caller must return something harmless rather than wait again.
func (w *warpState) meet(lane int, op string) (uint32, bool) {
	w.ops[lane] = op
	mask, ok := w.bar.wait(lane)
	if !ok {
		w.block.report(fmt.Sprintf(
			"gpu: warp rendezvous timed out in block %d, warp %d (%d lanes), lane %d, waiting in %s. "+
				"Every thread of a warp must reach every warp-level call: a thread that took another branch and is "+
				"blocked elsewhere (at SyncThreads, say) can never arrive, and the two wait for each other. "+
				"Hoist the call out of the conditional the threads disagree about.",
			w.bid, w.id, w.size, lane, op))
		return 0, false
	}
	for l := range WarpSize {
		if mask&(1<<uint(l)) != 0 && w.ops[l] != op {
			w.block.report(fmt.Sprintf(
				"gpu: threads of block %d, warp %d met in different warp-level calls: lane %d is in %s, lane %d is in %s. "+
					"Lanes are matched by the order they arrive in, so every thread of a warp must reach the same calls in the same order.",
				w.bid, w.id, lane, op, l, w.ops[l]))
			break
		}
	}
	return mask, true
}

// part is the second half of a rendezvous: nobody leaves until everybody has
// read, so that the fastest thread cannot reach its next warp call and
// overwrite a slot a slower sibling is still reading.
func (w *warpState) part(lane int) { w.bar.wait(lane) }

// warpBarrier is a barrier for the lanes of one warp. It is a second
// implementation rather than a use of barrier, for two reasons that barrier
// cannot serve: a waiter has to learn *who* took part, which is what
// ActiveMask and Ballot are, and a waiter has to be able to give up, because a
// warp is the one place in the emulator where two threads can wait for each
// other for good. sync.Cond has no timed wait, so the generation is a channel.
type warpBarrier struct {
	mu    sync.Mutex
	n     int    // lanes still in the kernel
	count int    // lanes arrived in the generation being assembled
	mask  uint32 // which ones
	cur   *warpGen
}

// warpGen is one rendezvous. mask is written before done is closed and read
// only after, so the close is what publishes it -- and a waiter can therefore
// never be handed a later generation's mask, however long it takes to wake.
type warpGen struct {
	done chan struct{}
	mask uint32
}

func newWarpBarrier(n int) *warpBarrier {
	return &warpBarrier{n: n, cur: &warpGen{done: make(chan struct{})}}
}

// wait blocks until every lane still in the kernel has arrived, and returns
// the mask of those that did. It returns false if it gave up first, which is a
// deadlock by any other name: see warpStallTimeout.
func (b *warpBarrier) wait(lane int) (uint32, bool) {
	b.mu.Lock()
	b.mask |= 1 << uint(lane)
	b.count++
	gen := b.cur
	if b.count >= b.n {
		mask := b.releaseLocked()
		b.mu.Unlock()
		return mask, true
	}
	b.mu.Unlock()

	timer := time.NewTimer(warpStallTimeout)
	defer timer.Stop()
	select {
	case <-gen.done:
		return gen.mask, true
	case <-timer.C:
		b.mu.Lock()
		defer b.mu.Unlock()
		if b.cur != gen {
			// Released between the timer firing and the lock being taken. The
			// warp did assemble, so report nothing.
			return gen.mask, true
		}
		// Take the arrival back, so that a later generation is not released by
		// a thread that is no longer waiting in this one.
		b.mask &^= 1 << uint(lane)
		b.count--
		return 0, false
	}
}

// abandon removes a lane from the warp for good, and is called when a thread
// leaves the kernel -- by returning as much as by panicking.
//
// It is what keeps ordinary divergence from hanging. A thread that took the
// other branch and ran to the end of the kernel never arrives at the
// rendezvous its siblings are waiting in, and the wait is therefore over the
// threads still present rather than over 32: arrived + finished == warp size
// is the release condition, spelled as a count against a shrinking n. CUDA
// says the same thing in its own terms -- a thread that has exited need not
// take part in a _sync built-in.
//
// It needs no lane, because a lane's bit is in the mask only while it is
// waiting, and a thread that is waiting has not left.
func (b *warpBarrier) abandon() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.n--
	if b.n > 0 && b.count >= b.n {
		b.releaseLocked()
	}
}

// releaseLocked opens the current generation and starts the next.
func (b *warpBarrier) releaseLocked() uint32 {
	gen := b.cur
	gen.mask = b.mask
	b.cur = &warpGen{done: make(chan struct{})}
	b.mask, b.count = 0, 0
	close(gen.done)
	return gen.mask
}

// bits32 and f32 move a float through the exchange, which carries raw bits so
// that one set of slots serves both element types. The round trip is exact for
// every value including NaNs, which a conversion through another numeric type
// would not be.
func bits32(v float32) uint32 { return math.Float32bits(v) }
func f32(v uint32) float32    { return math.Float32frombits(v) }
