// Package gocuda is a proof of concept for writing CUDA kernels in Go.
//
// It follows the two tracks NVIDIA describes for CUDA Rust:
//
//   - package simt lowers kernels written as ordinary Go functions to CUDA C,
//     the way the SIMT track lowers annotated Rust functions to PTX;
//   - package tile records a graph of tile operations and fuses it into a
//     single generated kernel, the way the tile track builds a kernel from an
//     embedded AST.
//
// Both compile through NVRTC at run time and launch through the CUDA driver
// API; see package cuda.
package gocuda

import (
	"embed"
	"io/fs"
)

// Kernel sources are embedded rather than read from disk so that a binary
// carries everything the transpiler needs, in the same spirit as CUDA Rust
// embedding the kernel AST in the host binary.
//
//go:embed kernels/*.go
var kernelFS embed.FS

// Kernels returns the embedded kernel sources, ready to hand to simt.Build.
func Kernels() fs.FS {
	sub, err := fs.Sub(kernelFS, "kernels")
	if err != nil {
		panic(err) // the embed pattern is a compile-time constant
	}
	return sub
}
