package fuzz

import "go/token"

// Go refuses a declared variable nothing reads, and the subset inherits that
// because the transpiler type-checks the source before lowering it.
//
// It is the single largest source of refusals a generator like this produces,
// and it cannot be avoided while a declaration is being written: whether
// anything will read the variable is not known until the rest of the block
// exists. So it is repaired afterwards, over the finished tree, where the
// question has an exact answer -- exact because the reads are collected by
// variable rather than by name, so a shadowed declaration is not mistaken for
// a use of the one it shadows.
//
// The repair is `v = v`, which reads the variable and stores what it already
// held. It is a no-op for every value including a NaN and a negative zero,
// which matters: the variable may well be read later by the comparison, and a
// repair that changed it would be changing the program's answer to make it
// compile.

// fixUnused inserts a use for every local the function declares and never
// reads.
func fixUnused(fn *Func) {
	reads := map[*Var]bool{}
	readsInStmts(fn.Body, reads)
	fn.Body = repair(fn.Body, reads)
}

// useStmt reads a variable and writes back what it read.
func useStmt(v *Var) Stmt {
	if v.buffer() {
		zero := &Lit{K: KInt}
		return &Assign{LHS: Lvalue{V: v, Idx: zero}, Op: token.ASSIGN, RHS: &Index{Base: v, Idx: zero}}
	}
	return &Assign{LHS: Lvalue{V: v}, Op: token.ASSIGN, RHS: &Ref{V: v}}
}

// repair rewrites one statement list, inserting a use after each declaration
// whose variable nothing reads and descending into every nested block.
func repair(list []Stmt, reads map[*Var]bool) []Stmt {
	out := make([]Stmt, 0, len(list))
	for _, s := range list {
		switch s := s.(type) {
		case *Decl:
			out = append(out, s)
			if !reads[s.V] {
				out = append(out, useStmt(s.V))
			}
			continue
		case *ArrayDecl:
			out = append(out, s)
			if !reads[s.V] {
				out = append(out, useStmt(s.V))
			}
			continue
		case *If:
			s.Then = repair(s.Then, reads)
			s.Else = repair(s.Else, reads)
		case *For:
			// The loop variable is read by the condition, so only the body can
			// hold a declaration in need of repair.
			s.Body = repair(s.Body, reads)
		case *Range:
			s.Body = repair(s.Body, reads)
			var lead []Stmt
			if !reads[s.Key] {
				lead = append(lead, useStmt(s.Key))
			}
			if s.Val != nil && !reads[s.Val] {
				lead = append(lead, useStmt(s.Val))
			}
			s.Body = append(lead, s.Body...)
		case *Switch:
			for i := range s.Cases {
				s.Cases[i].Body = repair(s.Cases[i].Body, reads)
			}
		}
		out = append(out, s)
	}
	return out
}

func readsInStmts(list []Stmt, reads map[*Var]bool) {
	for _, s := range list {
		readsInStmt(s, reads)
	}
}

func readsInStmt(s Stmt, reads map[*Var]bool) {
	switch s := s.(type) {
	case *Decl:
		readsIn(s.Init, reads)
	case *ArrayDecl:
	case *Assign:
		readsInLvalue(s.LHS, reads)
		readsIn(s.RHS, reads)
	case *ParAssign:
		for _, l := range s.LHS {
			readsInLvalue(l, reads)
		}
		for _, e := range s.RHS {
			readsIn(e, reads)
		}
	case *IncDec:
		readsInLvalue(s.LHS, reads)
	case *If:
		readsIn(s.Cond, reads)
		readsInStmts(s.Then, reads)
		readsInStmts(s.Else, reads)
	case *For:
		readsInStmt(s.Init, reads)
		readsIn(s.Cond, reads)
		readsInStmt(s.Post, reads)
		readsInStmts(s.Body, reads)
	case *Range:
		if s.Over.Buf != nil {
			reads[s.Over.Buf] = true
		}
		readsIn(s.Over.N, reads)
		readsInStmts(s.Body, reads)
	case *Switch:
		readsIn(s.Tag, reads)
		for _, cc := range s.Cases {
			for _, v := range cc.Vals {
				readsIn(v, reads)
			}
			readsInStmts(cc.Body, reads)
		}
	case *Return:
		readsIn(s.X, reads)
	case *SharedDecl:
	case *CallStmt:
		for _, a := range s.Args {
			readsIn(a, reads)
		}
	case *Discard:
		readsIn(s.X, reads)
	}
}

// readsInLvalue counts the buffer an indexed store names. Go treats `a[i] = x`
// as using a, and only the scalar form leaves the variable unread.
func readsInLvalue(l Lvalue, reads map[*Var]bool) {
	if l.Idx != nil {
		reads[l.V] = true
		readsIn(l.Idx, reads)
	}
}

func readsIn(e Expr, reads map[*Var]bool) {
	switch e := e.(type) {
	case nil:
		return
	case *Ref:
		reads[e.V] = true
	case *Index:
		reads[e.Base] = true
		readsIn(e.Idx, reads)
	case *Len:
		reads[e.Base] = true
	case *Binary:
		readsIn(e.X, reads)
		readsIn(e.Y, reads)
	case *Unary:
		readsIn(e.X, reads)
	case *Conv:
		readsIn(e.X, reads)
	case *MathCall:
		readsInAll(e.Args, reads)
	case *MinMax:
		readsInAll(e.Args, reads)
	case *WarpCall:
		readsInAll(e.Args, reads)
	case *AtomicCall:
		reads[e.Buf] = true
		readsIn(e.Idx, reads)
		readsInAll(e.Args, reads)
	case *CallExpr:
		readsInAll(e.Args, reads)
	}
}

func readsInAll(es []Expr, reads map[*Var]bool) {
	for _, e := range es {
		readsIn(e, reads)
	}
}
