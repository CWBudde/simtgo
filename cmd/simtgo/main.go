// Command simtgo is the build-time half of the toolchain: it reports kernels
// that cannot be lowered, before they are built at run time.
//
//	simtgo vet ./...                     # check the kernels in a package
//	go vet -vettool=$(which simtgo) ./...  # the same check, through go vet
//
// It is deliberately free of cgo. Package simt imports the CUDA driver
// bindings, so a tool that imported simt would need a CUDA toolchain to build
// -- an absurd requirement for a static analysis tool, and one that would stop
// it running in the CI job that has no GPU. Everything here works from
// internal/lower, which is pure Go.
package main

import (
	"fmt"
	"os"
	"strings"

	"golang.org/x/tools/go/analysis/singlechecker"

	"github.com/CWBudde/simtgo/analysis/simtcheck"
)

func main() {
	args := os.Args[1:]

	// go vet -vettool invokes the tool with no subcommand at all: first
	// "-V=full", then "-flags", and finally a single argument naming a .cfg
	// file. singlechecker.Main already speaks that whole protocol, as well as
	// the ordinary "run over these packages" one, so the only job here is to
	// recognise which invocations are ours.
	if len(args) > 0 && (strings.HasPrefix(args[0], "-") || strings.HasSuffix(args[0], ".cfg")) {
		singlechecker.Main(simtcheck.Analyzer)
		return
	}

	switch {
	case len(args) > 0 && args[0] == "vet":
		os.Args = append([]string{os.Args[0] + " vet"}, args[1:]...)
		singlechecker.Main(simtcheck.Analyzer)
	case len(args) > 0 && args[0] == "generate":
		os.Exit(generate(args[1:]))
	case len(args) == 0, args[0] == "help", args[0] == "-h", args[0] == "--help":
		usage(os.Stdout)
	default:
		fmt.Fprintf(os.Stderr, "simtgo: unknown subcommand %q\n\n", args[0])
		usage(os.Stderr)
		os.Exit(2)
	}
}

func usage(w *os.File) {
	fmt.Fprint(w, `simtgo is the build-time toolchain for Go CUDA kernels.

Usage:

	simtgo vet [packages]    report kernels that cannot be lowered to CUDA C
	simtgo generate [flags]  lower kernels ahead of time and embed their PTX

It also implements the go vet tool protocol:

	go vet -vettool=$(which simtgo) ./...

A bare package pattern is not accepted without a subcommand, so that it can
never be mistaken for the .cfg file go vet passes.
`)
}
