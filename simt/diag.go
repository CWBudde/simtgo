package simt

import (
	"go/token"
	"strings"

	"github.com/CWBudde/simtgo/internal/lower"
)

// A Diagnostic is one reason a kernel could not be lowered, with its position
// already resolved against the source it was found in.
type Diagnostic struct {
	Pos token.Position
	Msg string
}

func (d Diagnostic) Error() string {
	if !d.Pos.IsValid() {
		return "simt: " + d.Msg
	}
	return "simt: " + d.Pos.String() + ": " + d.Msg
}

// UnsupportedError reports every construct that kept a kernel from being
// lowered.
//
// A kernel is usually refused for one reason, and then this reads exactly as a
// single error always has. When there are several, each is on its own line and
// names its own position, so the whole list can be pasted into an issue.
type UnsupportedError struct {
	Kernel string
	Diags  []Diagnostic
}

func (e *UnsupportedError) Error() string {
	msgs := make([]string, len(e.Diags))
	for i, d := range e.Diags {
		msgs[i] = d.Error()
	}
	return strings.Join(msgs, "\n")
}

// Unwrap exposes the diagnostics individually, so that errors.As can pick one
// out of a set.
func (e *UnsupportedError) Unwrap() []error {
	errs := make([]error, len(e.Diags))
	for i, d := range e.Diags {
		errs[i] = d
	}
	return errs
}

// newUnsupportedError resolves positions against fset, which is nil for
// diagnostics raised before a file set existed -- those already carry whatever
// location they have in their message.
func newUnsupportedError(fset *token.FileSet, kernel string, diags []lower.Diagnostic) *UnsupportedError {
	out := make([]Diagnostic, len(diags))
	for i, d := range diags {
		out[i] = Diagnostic{Msg: d.Msg}
		if fset != nil && d.Pos.IsValid() {
			out[i].Pos = fset.Position(d.Pos)
		}
	}
	return &UnsupportedError{Kernel: kernel, Diags: out}
}
