package lower

import (
	"go/ast"
	"go/types"
)

// This file answers one question: which of a function's slice parameters does
// it never write through? A parameter that survives it is emitted as
// `const T*`, which is a promise to the C++ compiler -- and a wrong promise is
// either a compile error about generated code or, if the write hides behind a
// call, a miscompile. So the analysis is deliberately pessimistic: every shape
// it cannot read is treated as a write.
//
// It has to be interprocedural, because a slice parameter is forwarded as a
// bare pointer: `fill(y, y_len)` gives a device function the kernel's own
// memory to write, and nothing at the call site says so. The three ways a
// kernel can write are an assignment to an index expression, an increment of
// one, and the buffer argument of a gpu.Atomic* call; everything else is a
// call, and a call inherits whatever the callee does to what it was handed.

// writtenParams reports which of fd's parameters it may write through,
// following calls into the rest of the package.
//
// The result is memoised per declaration: a helper reached by two paths is
// analysed once, and the answer does not depend on the call site.
func (t *transpiler) writtenParams(fd *ast.FuncDecl) map[types.Object]bool {
	if w, ok := t.written[fd]; ok {
		return w
	}
	params, ok := t.paramObjects(fd)
	if !ok || fd.Body == nil || t.analysing[fd] {
		// Either the declaration is one the lowering refuses anyway, or this
		// is a cycle -- recursion is refused when the kernel is lowered, but
		// this walk runs over declarations and has to terminate on its own.
		// Assuming every parameter is written costs a const and is never wrong.
		return allWritten(params)
	}
	if t.analysing == nil {
		t.analysing = map[*ast.FuncDecl]bool{}
	}
	t.analysing[fd] = true

	mine := make(map[types.Object]bool, len(params))
	for _, obj := range params {
		mine[obj] = true
	}
	w := map[types.Object]bool{}
	ast.Inspect(fd.Body, func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.AssignStmt:
			for _, lhs := range n.Lhs {
				t.markTarget(w, mine, lhs)
			}
		case *ast.IncDecStmt:
			t.markTarget(w, mine, n.X)
		case *ast.CallExpr:
			t.markCallWrites(w, mine, n)
		}
		return true
	})

	delete(t.analysing, fd)
	if t.written == nil {
		t.written = map[*ast.FuncDecl]map[types.Object]bool{}
	}
	t.written[fd] = w
	return w
}

// paramObjects lists the objects fd's parameters were checked into, and
// reports whether all of them resolved. An unnamed or unresolved parameter is
// refused by the lowering itself; here it only means the analysis cannot see
// what it would need to.
func (t *transpiler) paramObjects(fd *ast.FuncDecl) ([]types.Object, bool) {
	var out []types.Object
	if fd.Type.Params == nil {
		return nil, true
	}
	for _, f := range fd.Type.Params.List {
		if len(f.Names) == 0 {
			return out, false
		}
		for _, n := range f.Names {
			obj := t.info.Defs[n]
			if obj == nil {
				return out, false
			}
			out = append(out, obj)
		}
	}
	return out, true
}

func allWritten(params []types.Object) map[types.Object]bool {
	w := make(map[types.Object]bool, len(params))
	for _, obj := range params {
		w[obj] = true
	}
	return w
}

// markTarget records the buffer an assignment target writes into.
//
// A plain identifier is not one: assigning to a parameter assigns the copy the
// function was given, which no caller can see. Only an index expression
// reaches through the pointer a slice lowered to, and a selector may sit on
// top of one -- `ps[0].X = 1` writes ps.
func (t *transpiler) markTarget(w, mine map[types.Object]bool, e ast.Expr) {
	switch e := unparen(e).(type) {
	case *ast.IndexExpr:
		t.markBase(w, mine, e.X)
	case *ast.SelectorExpr:
		t.markTarget(w, mine, e.X)
	}
}

// markBase records the object an expression is ultimately indexed out of.
func (t *transpiler) markBase(w, mine map[types.Object]bool, e ast.Expr) {
	switch e := unparen(e).(type) {
	case *ast.Ident:
		if obj := t.objectOf(e); obj != nil && mine[obj] {
			w[obj] = true
		}
	case *ast.IndexExpr:
		t.markBase(w, mine, e.X)
	case *ast.SelectorExpr:
		t.markBase(w, mine, e.X)
	}
}

// markCallWrites records what a call does to the parameters handed to it.
func (t *transpiler) markCallWrites(w, mine map[types.Object]bool, c *ast.CallExpr) {
	fun := unparen(c.Fun)
	if tv, ok := t.info.Types[fun]; ok && tv.IsType() {
		return // a conversion, which takes a value and not a buffer
	}
	switch f := fun.(type) {
	case *ast.Ident:
		if _, isBuiltin := t.info.Uses[f].(*types.Builtin); isBuiltin {
			// len, min and max: the only builtins the subset has, and none of
			// them writes.
			return
		}
		if obj, ok := t.info.Uses[f].(*types.Func); ok {
			if decl := t.declOf(obj); decl != nil {
				t.markForwarded(w, mine, c, decl)
				return
			}
		}
	case *ast.SelectorExpr:
		if sel := t.info.Selections[f]; sel != nil {
			// A method. Only gpu.Ctx has any device meaning and none of its
			// methods takes a buffer; anything else is refused when it is
			// lowered.
			return
		}
		if id, ok := f.X.(*ast.Ident); ok {
			if pkg, ok := t.info.Uses[id].(*types.PkgName); ok && pkg.Imported().Path() == GPUPkgPath {
				if _, isAtomic := gpuAtomics[f.Sel.Name]; isAtomic && len(c.Args) > 0 {
					// The one write that is not an assignment: an atomic reads
					// and writes the element it is given.
					t.markBase(w, mine, c.Args[0])
				}
				// Every other gpu function takes and returns scalars.
				return
			}
		}
	}
	// Something this walk cannot resolve. Treating every argument as written
	// is the answer that cannot be wrong, and it costs a const on a call the
	// lowering is about to refuse anyway.
	for _, a := range c.Args {
		t.markBase(w, mine, a)
	}
}

// markForwarded propagates a callee's writes back to the caller's parameters.
func (t *transpiler) markForwarded(w, mine map[types.Object]bool, c *ast.CallExpr, decl *ast.FuncDecl) {
	callee, ok := t.paramObjects(decl)
	args := c.Args
	if ok && len(callee) > 0 && IsCtx(callee[0].Type()) && len(args) > 0 {
		// The gpu.Ctx is dropped from the C signature and from the call, so
		// the arguments line up one short -- exactly as deviceArgs renders
		// them.
		callee, args = callee[1:], args[1:]
	}
	if !ok || len(callee) != len(args) {
		for _, a := range c.Args {
			t.markBase(w, mine, a)
		}
		return
	}
	written := t.writtenParams(decl)
	for i, p := range callee {
		if written[p] {
			t.markBase(w, mine, args[i])
		}
	}
}

// aliasedArgs reports two arguments of one call that are the same buffer where
// the callee writes through at least one of them.
//
// The generated C declares every pointer parameter __restrict__, which says
// the callee may assume they do not overlap. At a launch that promise is
// checked against the buffers actually bound; inside the translation unit it
// has to be checked here, because `blend(y, y)` is one Go call with nothing
// wrong with it and two aliased restrict pointers in C.
func (t *transpiler) aliasedArgs(c *ast.CallExpr, decl *ast.FuncDecl) (a, b string, ok bool) {
	params, resolved := t.paramObjects(decl)
	args := c.Args
	if resolved && len(params) > 0 && IsCtx(params[0].Type()) && len(args) > 0 {
		params, args = params[1:], args[1:]
	}
	if !resolved || len(params) != len(args) {
		return "", "", false
	}
	written := t.writtenParams(decl)
	for i := range args {
		obji := t.argBuffer(args[i])
		if obji == nil {
			continue
		}
		for j := i + 1; j < len(args); j++ {
			if t.argBuffer(args[j]) != obji {
				continue
			}
			if written[params[i]] || written[params[j]] {
				return params[i].Name(), params[j].Name(), true
			}
		}
	}
	return "", "", false
}

// argBuffer is the object a call argument passes as a buffer, or nil when the
// argument is not one. Only an identifier can be: the subset has no slice
// expressions, so a buffer argument is always a parameter or a shared tile.
func (t *transpiler) argBuffer(e ast.Expr) types.Object {
	id, ok := unparen(e).(*ast.Ident)
	if !ok {
		return nil
	}
	obj := t.objectOf(id)
	if obj == nil {
		return nil
	}
	if _, isSlice := obj.Type().(*types.Slice); !isSlice {
		return nil
	}
	return obj
}
