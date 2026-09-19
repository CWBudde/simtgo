package tile

import (
	"fmt"

	"github.com/CWBudde/simtgo/cuda"
)

// AliasError is a materialisation whose output buffer overlapped one of the
// graph's inputs.
//
// Input names the generated kernel parameter, as Tensor.Source spells it, and
// not anything the caller wrote: a graph's inputs have no user-visible names,
// and the parameter is what the promise is made about.
type AliasError struct {
	Input string
}

func (e *AliasError) Error() string {
	return fmt.Sprintf("tile: the output buffer overlaps input %s, which the kernel reads while it writes out; "+
		"the generated kernel declares every pointer __restrict__, which promises that they do not overlap",
		e.Input)
}

// checkAliasing refuses a launch that breaks the __restrict__ promise the
// generated kernel makes.
//
// args is what Tensor.run is about to marshal: the output first, then one
// argument per input in the order generate named them, so index i+1 is
// parameter pi. Only an overlap with the output matters -- restrict forbids
// reaching a *modified* object through another pointer, and out is the only
// thing a fused kernel writes -- so two inputs sharing a buffer is accepted,
// which is the same position simt.Kernel.checkAliasing takes and the case
// MaterializeStepwise reaches whenever one node feeds both operands.
//
// Today nothing can fail this check: Tensor.run allocates out with
// cuda.NewSlice on every call, so it cannot be a buffer the graph already
// holds. It is here for the step that changes that -- device-resident values,
// where a caller supplies the buffers -- because by then the qualifier will
// have been in the generated source for a while and an unchecked promise is
// how wrong numbers on one architecture start.
func checkAliasing(args []any) error {
	if len(args) == 0 {
		return nil
	}
	outRange, ok := args[0].(cuda.Ranger)
	if !ok {
		return nil
	}
	base, bytes := outRange.DeviceRange()
	if bytes == 0 {
		return nil
	}
	for i, a := range args[1:] {
		r, ok := a.(cuda.Ranger)
		if !ok {
			continue
		}
		inBase, inBytes := r.DeviceRange()
		if inBytes == 0 {
			continue
		}
		// Half-open intervals, so a partial overlap is caught and not only an
		// identical base.
		if base >= inBase+cuda.DevPtr(inBytes) || inBase >= base+cuda.DevPtr(bytes) {
			continue
		}
		return &AliasError{Input: fmt.Sprintf("p%d", i)}
	}
	return nil
}
