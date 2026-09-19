# gocuda justfile
#
# One local entry point for the checks that otherwise live in five remembered
# command lines. It *mirrors* .github/workflows/ci.yml rather than being called
# by it: the workflow keeps its explicit step list, so a CI runner needs no
# `just` and a red job names the step that failed in the GitHub UI.
#
# That makes the workflow authoritative. If the two ever disagree, the workflow
# is right and this file is stale -- `just check` and the workflow's `test`,
# `lint` and `format` jobs are meant to run the same commands in the same
# order, and `just check` is the cheaper way to find out before pushing.

default: check

#################################
# Format and lint
#################################

# Format everything treefmt knows about
fmt:
    treefmt

# Is everything formatted? (writes nothing)
fmt-check:
    treefmt --ci

# Lint the untagged tree -- the transpiler, the analyzer and the emulator
lint:
    golangci-lint run ./...

# Worth doing before a release: two findings have hidden there already.
# Lint the half behind //go:build cuda, which neither `lint` nor CI can see
lint-cuda:
    golangci-lint run --build-tags cuda ./...

# Lint with the fixes the linters can apply themselves
lint-fix:
    golangci-lint run --fix ./...

# Fix what can be fixed, then format
fix: lint-fix fmt

#################################
# Build
#################################

# Build every package
build:
    go build ./...

# The driver must build with no cgo and no toolkit installed
build-nocgo:
    CGO_ENABLED=0 go build -tags cuda ./...

# Vet both build configurations; the tagged half compiles nowhere else
vet:
    go vet ./...
    go vet -tags cuda ./...

#################################
# Tests
#################################

# Transpiler, golden files and the CPU emulator -- no GPU needed
test:
    go test ./...

# The emulator runs one kernel from many goroutines over shared slices
test-race:
    go test -race ./gpu/

# On a machine with a GPU this is the CPU/GPU parity suite instead.
# Compile every tagged test; run those needing neither device nor toolkit
test-cuda:
    go test -tags cuda ./...

# One golden file, or all of them
test-golden:
    go test -run TestGolden ./simt/

#################################
# Generated artifacts
#################################

# Is the committed CUDA C current?
check-generated:
    go run ./cmd/gocuda generate -check

# Regenerate kernels/prebuilt -- needs NVRTC
generate:
    go generate ./...

# Regenerate it without a toolkit: refreshes the gate, never calls NVRTC
generate-no-ptx:
    go run ./cmd/gocuda generate -pkg ./kernels -out ./kernels/prebuilt -no-ptx

# Refresh simt/testdata/*.cu after an emitter change
golden-update:
    GOCUDA_UPDATE=1 go test -run TestGolden ./simt/

# Refuse kernels that cannot be lowered
vet-kernels:
    go run ./cmd/gocuda vet ./kernels

#################################
# Checks
#################################

# Everything CI decides without a device, in CI's order
check: build vet test test-race build-nocgo check-generated test-cuda lint fmt-check

# GOCUDA_REQUIRE_DEVICE turns the "no CUDA device" skip into a failure, because
# a sweep that launched nothing is green and means nothing; --error-exitcode is
# what makes it a gate at all. A clean run prints nothing: go test discards a
# passing binary's stdout.
# compute-sanitizer over every kernel, one tool at a time. Needs a device.
#
# It is taken from PATH, and on a machine with both the toolkit's copy and the
# distribution's nvidia-cuda-toolkit package installed, /usr/bin wins and can
# be years older than the driver. That failure looks nothing like an install
# problem -- every package "FAIL"s in milliseconds with "Unable to find
# injection library libsanitizer-collection.so" -- so if that is what comes
# back, put the toolkit's own directory first:
#   PATH=/usr/local/cuda-12.8/compute-sanitizer:$PATH just sanitize-all
sanitize tool="memcheck":
    GOCUDA_REQUIRE_DEVICE=1 go test -tags cuda -count=1 -timeout 0 \
        -exec "compute-sanitizer --tool={{tool}} --error-exitcode 1 --report-api-errors no --target-processes application-only" \
        ./simt/ ./tile/ ./cuda/

# All four sanitizer tools
sanitize-all: (sanitize "memcheck") (sanitize "racecheck") (sanitize "initcheck") (sanitize "synccheck")

#################################
# Dev
#################################

# Install what `check` needs that Go does not ship
setup-deps:
    #!/usr/bin/env bash
    set -euo pipefail

    # golangci-lint has to be built with this module's own Go: the version
    # check compares the toolchain that built the linter against go.mod, and a
    # release binary built with an older Go refuses the module outright.
    command -v golangci-lint >/dev/null 2>&1 || {
        echo "Installing golangci-lint (from source -- see the comment above)..."
        go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@v2.13.2
    }

    # treefmt cannot be `go install`ed at any version: a path in its own test
    # data contains a character the module zip format forbids, so the proxy
    # fails to build the zip. Building from a pinned tag is the way in.
    command -v treefmt >/dev/null 2>&1 || {
        echo "Installing treefmt (from a pinned tag -- see the comment above)..."
        tmp=$(mktemp -d)
        git clone --depth 1 --branch v2.6.0 https://github.com/numtide/treefmt "$tmp"
        go build -C "$tmp" -o "$(go env GOPATH)/bin/treefmt" .
        rm -rf "$tmp"
    }

    # prettier formats the Markdown, YAML and JSON. Pinned for the same reason
    # CI pins it: a formatter's idea of correct output changes between releases.
    command -v prettier >/dev/null 2>&1 || {
        echo "Installing prettier..."
        npm install --global prettier@3.8.1 || echo "prettier needs npm; install Node.js first."
    }

    echo "Done. $(go env GOPATH)/bin must be on PATH."

# Run an example on a device
example name="fir":
    go run -tags cuda ./examples/{{name}}
