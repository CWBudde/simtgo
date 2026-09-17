//go:build cuda

package main

import "github.com/CWBudde/gocuda/cuda"

// compile is deliberately indifferent to whether a device is present. NVRTC
// needs only its own library, while cuda.Available would call cuInit and so
// demand a driver and a GPU -- which a build machine holding the toolkit does
// not have to have, and which is why the cgo LDFLAGS already link -lcuda from
// lib64/stubs.
func compile(req Request) (*Response, error) {
	major, minor, err := cuda.NVRTCVersion()
	if err != nil {
		return nil, err
	}
	resp := &Response{NVRTCMajor: major, NVRTCMinor: minor}
	for _, u := range req.Units {
		ptx, err := cuda.Compile(u.Source, u.Name+".cu", req.Arch)
		if err != nil {
			// A kernel NVRTC refuses is reported per kernel rather than
			// aborting the batch: the others are still worth generating, and
			// the caller decides what a refusal means.
			resp.Results = append(resp.Results, Result{Name: u.Name, Error: err.Error()})
			continue
		}
		resp.Results = append(resp.Results, Result{Name: u.Name, PTX: ptx.Bytes, Log: ptx.Log})
	}
	return resp, nil
}
