//go:build !cuda

package main

import "github.com/CWBudde/gocuda/cuda"

// compile without the "cuda" build tag cannot reach NVRTC. This file exists so
// that the command still compiles everywhere -- a build failure here would be
// indistinguishable from a missing toolkit, which is exactly the confusion the
// separate process was introduced to avoid.
func compile(Request) (*Response, error) { return nil, cuda.ErrNoCUDA }
