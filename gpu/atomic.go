package gpu

import "sync"

// Atomic read-modify-write on global or shared memory.
//
// Each function takes the buffer and an index into it rather than a pointer,
// because the kernel subset has no address-of: "&s[i]" is something the
// transpiler writes and a kernel can never say. Each returns the value the
// element held *before* the operation, which is what the CUDA built-in
// returns and what makes AtomicAddI32 usable as a work queue rather than only
// as a counter.
//
// The vocabulary is exactly the set of CUDA overloads that exist at
// compute_75 with no header included, which is all NVRTC has: it compiles a
// bare string. The gaps are the hardware's, not the transpiler's, and they are
// worth naming so their absence reads as a decision:
//
//   - atomicMin, atomicMax and atomicCAS have no float overload, so there is
//     no AtomicMinF32. Doing it on floats means __float_as_int bit tricks,
//     which is a different feature.
//   - atomicAdd has no "long long" overload, only "unsigned long long", so
//     64-bit variants would each need a different reinterpret cast and the
//     one-Go-name-to-one-built-in table would stop holding.
//   - atomicAdd(double*) does exist at compute_75, but float64 is opt-in
//     behind //gocuda:float64 and package gpu has no float64 vocabulary to
//     add it to.

// atomicMu serialises every read-modify-write the functions below perform.
//
// Global memory in the emulator is the caller's own Go slice, captured by the
// kernel closure, so there is nowhere per-buffer to hang a lock and nothing
// the emulator is handed that it could key one on. One lock for the package is
// therefore the mechanism, and the emulator can afford it: RunCPU already
// spends a goroutine per thread, so a mutex is not what makes it slow.
//
// What the lock deliberately does *not* do matters more than the contention.
// It creates a happens-before edge only between goroutines that take it, so a
// kernel that also writes the same element plainly has no edge with the atomic
// writers and "go test -race" reports it. That is correct: a plain write
// racing an atomic is a real race on the device too.
var atomicMu sync.Mutex

// The limits of this emulation, since they are invisible from the call site:
//
//   - It imposes a total order over every atomic in the process. CUDA
//     guarantees atomicity per address and nothing about order across
//     addresses. No kernel can observe the difference, but a benchmark can:
//     the emulator's atomic throughput says nothing about the device's.
//   - float32 addition is not associative, and the order in which goroutines
//     win the lock is not the order in which the device's threads arrive, so a
//     CPU and a GPU float sum of the same data differ in the last bits. Tests
//     that compare them must say so.
//   - It models atomicity only, never visibility: there is no __threadfence
//     here, and no memory scopes (atomicAdd_block, atomicAdd_system). A kernel
//     that needs a fence to be correct is not diagnosed by running it here.
//
// Every function below unlocks through defer. That is load-bearing rather than
// stylistic: an out-of-range index panics with the lock held, RunCPU's per
// thread recover turns that panic into a report instead of a crash, and the
// process would then carry a permanently locked mutex that deadlocks every
// later launch. One bad kernel would hang the whole test binary.

// AtomicAddF32 adds v to s[i] and returns the previous value (atomicAdd).
func AtomicAddF32(s []float32, i int, v float32) float32 {
	atomicMu.Lock()
	defer atomicMu.Unlock()
	old := s[i]
	s[i] = old + v
	return old
}

// AtomicAddI32 adds v to s[i] and returns the previous value (atomicAdd).
func AtomicAddI32(s []int32, i int, v int32) int32 {
	atomicMu.Lock()
	defer atomicMu.Unlock()
	old := s[i]
	s[i] = old + v
	return old
}

// AtomicMinI32 stores the smaller of s[i] and v into s[i] and returns the
// previous value (atomicMin).
func AtomicMinI32(s []int32, i int, v int32) int32 {
	atomicMu.Lock()
	defer atomicMu.Unlock()
	old := s[i]
	if v < old {
		s[i] = v
	}
	return old
}

// AtomicMaxI32 stores the larger of s[i] and v into s[i] and returns the
// previous value (atomicMax).
func AtomicMaxI32(s []int32, i int, v int32) int32 {
	atomicMu.Lock()
	defer atomicMu.Unlock()
	old := s[i]
	if v > old {
		s[i] = v
	}
	return old
}

// AtomicExchI32 stores v into s[i] and returns the previous value
// (atomicExch). The name is CUDA's rather than Go's "swap", because every
// other name here is the built-in's too.
func AtomicExchI32(s []int32, i int, v int32) int32 {
	atomicMu.Lock()
	defer atomicMu.Unlock()
	old := s[i]
	s[i] = v
	return old
}

// AtomicCASI32 stores val into s[i] if s[i] equals compare, and returns the
// previous value either way (atomicCAS).
//
// The caller learns whether it won by comparing the result against compare,
// which is why returning the old value unconditionally is not a detail: it is
// the whole interface.
func AtomicCASI32(s []int32, i int, compare, val int32) int32 {
	atomicMu.Lock()
	defer atomicMu.Unlock()
	old := s[i]
	if old == compare {
		s[i] = val
	}
	return old
}
