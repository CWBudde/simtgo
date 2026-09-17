//go:build !cuda

package cuda

// This file keeps the package's API available when the module is built
// without the "cuda" build tag, so the transpiler and its tests compile on
// machines with no CUDA toolchain. Every call fails with ErrNoCUDA.

// Init reports that CUDA support was not compiled in.
func Init() error { return ErrNoCUDA }

// Available reports whether a usable CUDA device is present.
func Available() bool { return false }

// Context is a placeholder for the real driver context.
type Context struct{}

// NewContext always fails in a build without the "cuda" tag.
func NewContext(device int) (*Context, error) { return nil, ErrNoCUDA }

// Arch reports the virtual architecture of the device.
func (c *Context) Arch() string { return "" }

// ComputeCapability reports the device's compute capability. With no device to
// ask this build answers (0, 0), which callers must read as "unknown" rather
// than as a device older than every CUDA GPU ever built.
func (c *Context) ComputeCapability() (major, minor int) { return 0, 0 }

// MaxSharedMemPerBlock reports how much shared memory a single block may use,
// in bytes. Without a device to ask there is no limit to report, so this build
// always answers zero, which callers must read as "unknown, skip the check"
// rather than as a device offering no shared memory at all.
func (c *Context) MaxSharedMemPerBlock() int { return 0 }

// Name reports the device name.
func (c *Context) Name() string { return "" }

// Close releases the context.
func (c *Context) Close() error { return ErrNoCUDA }

// Sync waits for outstanding device work.
func (c *Context) Sync() error { return ErrNoCUDA }

// Alloc reserves device memory.
func (c *Context) Alloc(n int) (DevPtr, error) { return 0, ErrNoCUDA }

// Free releases device memory.
func (c *Context) Free(p DevPtr) error { return ErrNoCUDA }

// LoadPTX loads a PTX module.
func (c *Context) LoadPTX(ptx []byte) (*Module, error) { return nil, ErrNoCUDA }

// LoadPTXCached loads a module into the context at most once per key. Since
// nothing can be loaded here, build is never called.
func (c *Context) LoadPTXCached(key string, build func() ([]byte, error)) (*Module, error) {
	return nil, ErrNoCUDA
}

// Module is a placeholder for a loaded PTX module.
type Module struct{}

// Function looks up a kernel by name.
func (m *Module) Function(name string) (*Function, error) { return nil, ErrNoCUDA }

// Unload releases the module ahead of its context's Close.
func (m *Module) Unload() error { return ErrNoCUDA }

// Function is a placeholder for a kernel entry point.
type Function struct{}

// Launch runs the kernel.
func (f *Function) LaunchSync(grid, block Dim3, sharedBytes int, args ...Arg) error {
	return ErrNoCUDA
}

// Compile JIT-compiles CUDA C to PTX.
func Compile(src, name, arch string) (*PTX, error) { return nil, ErrNoCUDA }

// NVRTCVersion reports the version of the NVRTC library.
func NVRTCVersion() (major, minor int, err error) { return 0, 0, ErrNoCUDA }

// Slice is a placeholder for a typed device buffer.
type Slice[T any] struct{ n int }

// NewSlice allocates a device buffer.
func NewSlice[T any](c *Context, n int) (*Slice[T], error) { return nil, ErrNoCUDA }

// Upload copies a host slice to the device.
func Upload[T any](c *Context, xs []T) (*Slice[T], error) { return nil, ErrNoCUDA }

// Download copies the buffer back to the host.
func (s *Slice[T]) Download() ([]T, error) { return nil, ErrNoCUDA }

// Len reports the number of elements.
func (s *Slice[T]) Len() int { return s.n }

// Arg passes the buffer's device pointer to a kernel.
func (s *Slice[T]) Arg() Arg { return Arg{} }

// KernelArgs expands the slice into a pointer and length pair.
func (s *Slice[T]) KernelArgs() []Arg { return nil }

// Free releases the buffer.
func (s *Slice[T]) Free() {}
