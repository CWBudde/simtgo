// Command gocuda-nvrtc compiles generated CUDA C to PTX for "gocuda generate".
//
// It exists only because NVRTC needs cgo and gocuda does not. Building the
// generator itself with the "cuda" tag would make one //go:generate line wrong
// on every machine without a CUDA toolkit: the failure would happen inside
// "go build" of the tool, as `fatal error: nvrtc.h: No such file or directory`,
// before any of its code could explain itself. Keeping the compiler in a child
// process lets the parent catch that and say what to do about it.
//
// It reads a JSON request on stdin and writes a JSON response on stdout, one
// batch per invocation rather than one process per kernel.
//
// Phase 1.2 of PLAN.md loads libnvrtc at run time, at which point this command
// has no reason to exist and the generator can call cuda.Compile directly.
package main

import (
	"encoding/json"
	"fmt"
	"os"
)

// Request is a batch of kernels to compile, all for the same architecture.
type Request struct {
	Arch  string `json:"arch"`
	Units []Unit `json:"units"`
}

// Unit is one kernel's generated CUDA C.
type Unit struct {
	Name   string `json:"name"`
	Source string `json:"source"`
}

// Response mirrors Request.Units one for one.
type Response struct {
	NVRTCMajor int      `json:"nvrtcMajor"`
	NVRTCMinor int      `json:"nvrtcMinor"`
	Results    []Result `json:"results"`
}

// Result is one kernel's outcome. PTX is base64 in the JSON, since it travels
// as bytes; Error is empty when the kernel compiled.
type Result struct {
	Name  string `json:"name"`
	PTX   []byte `json:"ptx,omitempty"`
	Log   string `json:"log,omitempty"`
	Error string `json:"error,omitempty"`
}

func main() {
	var req Request
	if err := json.NewDecoder(os.Stdin).Decode(&req); err != nil {
		fail("reading the request: %v", err)
	}
	resp, err := compile(req)
	if err != nil {
		fail("%v", err)
	}
	if err := json.NewEncoder(os.Stdout).Encode(resp); err != nil {
		fail("writing the response: %v", err)
	}
}

func fail(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "gocuda-nvrtc: "+format+"\n", args...)
	os.Exit(2)
}
