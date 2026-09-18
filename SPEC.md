# The supported subset

This is the contract. `README.md` explains why the subset looks as it does and
is the place to start reading; this file says what is in it, and is the file to
consult when the two disagree.

Two things make it a contract rather than a description:

- **Refusals are checked against the tests.** Every diagnostic quoted below is
  pinned by a case in `simt/errors_test.go`, and `simt/spec_test.go` fails in
  both directions — a refusal the transpiler enforces and this document omits,
  or a rule this document claims and no test holds it to.
- **The lowering has one definition.** `internal/lower` is it, and both callers
  share it: `simt.Transpile`, which type-checks kernel sources at run time, and
  `analysis/simtcheck`, which is handed a real package by `go/analysis`. They
  cannot disagree, because there is nothing to disagree with.

Numerics are a separate contract, in `NUMERICS.md`: what the device does to a
`float32`, and what a test may therefore assert.

## 1. What a kernel is

A **kernel** is a function whose first parameter is a `gpu.Ctx`. Nothing else
marks one — a directive somebody forgets restores exactly the "compiles fine,
dies in `main()`" failure the whole design removes.

```
KernelDecl = "func" identifier "(" "ctx" "gpu.Ctx" { "," ParamGroup } ")" Block .
```

It must be a plain function: no receiver, no results, no type parameters, and
every parameter named. A kernel package may import **only**
`github.com/CWBudde/gocuda/gpu`.

Three directives change what a declaration is, and each is a whole comment line
in a doc comment — a mention inside a sentence is prose about the directive:

| Directive          | On a function | On a file's package comment | Means                                                    |
| ------------------ | ------------- | --------------------------- | -------------------------------------------------------- |
| `//gocuda:ignore`  | yes           | yes                         | not a kernel, and not lowered at all                     |
| `//gocuda:device`  | yes           | **no**                      | a helper, not a kernel; refused if it takes no `gpu.Ctx` |
| `//gocuda:float64` | yes           | yes                         | this kernel may use double precision                     |

`//gocuda:float64` is a promise about what a launch costs, so it belongs to the
kernel and covers the whole translation unit including every helper the kernel
reaches. A helper may not carry its own — otherwise a kernel without the
directive could acquire double precision through a call. The same helper may
therefore lower as `float` in one kernel and `double` in another.

### Device functions

A kernel may call another function in its package. Each one it reaches is
emitted as a `__device__` function in the same translation unit — prototypes
first, so the order of the Go declarations does not matter — and a function two
paths reach is emitted once. A function nothing reaches is never lowered, and
so is never checked.

Refused: recursion (direct and mutual), methods, generics, variadics, more than
one result, a named result, and an array result.

## 2. Grammar

The subset is a subset of Go's grammar, and everything not derivable below is
refused. Go's own productions are assumed for identifiers, literals and types.

```
Stmt        = Declaration | SimpleStmt | IfStmt | ForStmt | RangeStmt
            | SwitchStmt | LabeledStmt | BranchStmt | ReturnStmt | Block
            | ExprStmt | EmptyStmt .

Declaration = "const" ConstSpec | "var" VarSpec .          /* no type, no import */
VarSpec     = IdentList Type [ "=" ExprList ] .            /* len(ExprList) == 0 or == len(IdentList) */

SimpleStmt  = Assignment | IncDecStmt | ShortVarDecl .
Assignment  = ExprList assign_op ExprList .                /* len(lhs) == len(rhs) */
assign_op   = "=" | "+=" | "-=" | "*=" | "/=" | "%="
            | "&=" | "|=" | "^=" | "<<=" | ">>=" .         /* note: no "&^=" */

IfStmt      = "if" [ SimpleStmt ";" ] Expr Block [ "else" ( IfStmt | Block ) ] .
ForStmt     = "for" [ [ SimpleStmt ] ";" [ Expr ] ";" [ SimpleStmt ] ] Block .
RangeStmt   = "for" Ident [ "," Ident ] ":=" "range" Expr Block .
SwitchStmt  = "switch" [ SimpleStmt ";" ] [ Expr ] "{" { CaseClause } "}" .
LabeledStmt = Label ":" ( ForStmt | RangeStmt ) .          /* a label goes on a loop, nothing else */
BranchStmt  = ( "break" | "continue" ) [ Label ] | "fallthrough" .

Expr        = UnaryExpr | Expr binary_op Expr .
UnaryExpr   = PrimaryExpr | ( "-" | "+" | "!" | "^" ) UnaryExpr .
PrimaryExpr = Operand | PrimaryExpr Index | PrimaryExpr Selector
            | Conversion | BuiltinCall | GPUCall | DeviceCall .
binary_op   = "||" | "&&" | "==" | "!=" | "<" | "<=" | ">" | ">="
            | "+" | "-" | "|" | "^" | "*" | "/" | "%" | "<<" | ">>" | "&" | "&^" .
BuiltinCall = ( "len" | "min" | "max" ) "(" ExprList ")" .
```

Three shapes in that grammar need a word, because Go and C disagree about what
the spelling means.

**A `switch` has two lowerings.** One whose tag is integral and whose every
case is a foldable constant becomes a C `switch`, with an explicit `break`
closing each clause because Go does not fall through, and `fallthrough`
honoured by leaving that `break` out. Any other switch — tagless, or testing a
variable — becomes the `if`/`else` chain Go's semantics actually describe, its
tag evaluated once into a temporary. A bare `break` inside _that_ form is
refused, because in C it would leave the enclosing loop.

**A labelled `break` or `continue` becomes a `goto`** to a target after the
loop or at the end of its body. Go's own `goto` is refused, which is what makes
a structured analysis of the Go source sound.

**Parallel assignment evaluates everything first.** `a, b = b, a` is a swap, so
every right-hand side goes through a temporary, and so does any index the
statement itself writes: `i, y[i] = 2, 7` stores into the old `i`'s element. It
works as a statement and is refused in a `for` clause, where the temporaries it
needs would have to be declarations and C's comma operator carries only
expressions. Assigning from a call that returns two values is refused
altogether: a device function lowers to one C return value.

## 3. Types

Two mappings, and which one applies is decided by **position**. `ctype` is the
by-value map — a cast, a return type, a local, a scalar parameter. `ctypeElem`
is the layout map — a slice element, an array element, a struct field. They are
not the same, and the difference is the most consequential rule here.

| Go                          | By value             | In a layout position              |
| --------------------------- | -------------------- | --------------------------------- |
| `float32`                   | `float`              | `float`                           |
| `float64`                   | `double`, opt-in     | `double`, opt-in                  |
| `int32`                     | `int`                | `int`                             |
| `int64`                     | `long long`          | `long long`                       |
| `uint32`                    | `unsigned int`       | `unsigned int`                    |
| `uint64`                    | `unsigned long long` | `unsigned long long`              |
| `bool`                      | `bool`               | `bool`                            |
| `int`                       | `int`                | **refused**                       |
| `int8`, `int16`             | **refused**          | `signed char`, `short`            |
| `uint8`, `uint16`           | **refused**          | `unsigned char`, `unsigned short` |
| `uint`, `uintptr`           | **refused**          | **refused**                       |
| a named struct of the above | a `struct`           | a `struct`                        |
| `[N]T` of the above         | —                    | `T name[N]`                       |

The 64-bit types are `long long` and never `long`, which is 8 bytes on Linux
and 4 on Windows.

**Go's `int` is 64-bit and CUDA's is 32-bit.** That narrowing is the one
deliberate infidelity, and it holds by value, where an index is bounded by the
grid. Where a value can leave those bounds it is refused rather than excused: a
left shift by a computed amount is the shape that does, and the differential
fuzzer found it by writing `o << (o & 31)` and reading the two answers, which
differ by more than the high word — see "Shifts, where the two widths
disagree". As a slice element, an array element or a struct field it is
not a lost high word but a different _stride_, and the host copies Go's layout
regardless — so `[]int` is refused. It used to lower cleanly and return the
wrong numbers.

**The narrow integers are storage, not arithmetic.** They may be a slice
element, an array element or a struct field — `[]uint8` is what an image buffer
is — and no operator accepts one. Convert to `int32`, compute there, convert
back. Some of those operators would in fact agree, and they are refused anyway:
a rule that holds for every operator is one a reader can keep in their head,
and relaxing a refusal later costs a line where retracting an acceptance costs
somebody a kernel that worked.

**A struct is laid out by Go and checked by CUDA.** Every hole Go leaves is
declared as an `unsigned char gocuda_padN[k]` member, the trailing one
included, and then `sizeof` and `alignof` are asserted against Go's numbers.
With every hole spelled out the members account for exactly Go's size, and C++
lays each member at or after the end of the one before it — so the size can
only match if nothing further was inserted, and every field is therefore where
Go put it. The trailing member is what closes that argument: without it a byte
inserted earlier could hide in the end slack. Offsets cannot be asserted
directly because NVRTC compiles a bare string with no include path, so it has
no `offsetof`. Struct literals lower positionally and step over the padding,
because designated initialisers are C++20 and NVRTC defaults to C++17.

**An array is storage, not a value.** It can be declared, indexed, measured
with `len`, ranged over and held as a struct field. It cannot be a parameter
(Go copies, C decays to a pointer, so a write inside the function would reach
the caller's array), nor a result, nor assigned whole, nor compared, nor
switched on, nor have zero length.

## 4. The `gpu` vocabulary

Every symbol, what it becomes, and what it requires. Adding anything to `gpu`
means adding it to `internal/lower/gpupkg.go` too; a drift test enforces that.

### Position

`ThreadIdx`/`Y`/`Z`, `BlockIdx`/`Y`/`Z`, `BlockDim`/`Y`/`Z`, `GridDim`/`Y`/`Z`,
`GlobalID` (also `GlobalIDX`), `GlobalIDY`, `GlobalIDZ`. Each is one CUDA
built-in, cast to `int` — the built-ins are unsigned, and mixing them into Go's
`int` comparisons would change answers. `GlobalID()` is
`(int)(blockIdx.x * blockDim.x + threadIdx.x)`.

### Barriers

`SyncThreads()` → `__syncthreads()`, `SyncWarp()` → `__syncwarp(0xffffffff)`.
See §6 for what a barrier requires of the threads that reach it.

### Shared memory

One constructor per element type: `SharedF32`, `SharedF64`, `SharedI32`,
`SharedI64`, `SharedU32`, `SharedU64`, each becoming `__shared__ T name[N]`.
There is no `SharedBool`: nothing here reads or writes a bool atomically, so
such a tile could only be filled under a barrier, which the `int32` tile
already does.

The dynamic forms — `SharedDynF32` and one per type — take no argument and
become `extern __shared__ T name[]`, with the length arriving as a generated
kernel parameter the launch fills.

Preconditions: assigned to a variable, at the top level of its function, a
constant size (static) or no argument (dynamic), `//gocuda:float64` for the
`F64` forms, **at most one dynamic tile per kernel**, and no dynamic tile in a
device function. A static tile in a device function is fine — that is
block-scoped storage CUDA allocates once per function.

### Math

`Sqrt`, `Abs`, `Hypot`, `Sin`, `Cos`, `Exp`, `Log`, `Fmin`, `Fmax` become
`sqrtf`, `fabsf`, `hypotf`, `sinf`, `cosf`, `expf`, `logf`, `fminf`, `fmaxf`.
The double-precision half is `Sqrt64` and the rest, becoming CUDA's unsuffixed
`sqrt`, `hypot`, `fmin` — legal only under `//gocuda:float64`, and the check
sits on the _call_, because `float32(gpu.Sqrt64(2))` names no `float64`
anywhere.

### Atomics

`AtomicAddF32`, `AtomicAddI32`, `AtomicMinI32`, `AtomicMaxI32`,
`AtomicExchI32`, `AtomicCASI32`, each returning the value the element held
before. They take **a buffer and an index** rather than a pointer, because the
subset has no address-of: `&s[i]` is something the emitter writes and a kernel
can never say. That shape is also what makes the first argument checkable — it
must be a slice parameter or a shared tile, the only two things whose address
means anything on the device.

The vocabulary stops where CUDA's overloads stop at `compute_75` with no
header: no float `atomicMin`/`atomicMax`/`atomicCAS`, and no `long long`
`atomicAdd`.

### Warp

`LaneID`, `ShuffleF32`/`I32`, `ShuffleXorF32`/`I32`, `ShuffleUpF32`/`I32`,
`ShuffleDownF32`/`I32`, `Ballot`, `Any`, `All`, `ActiveMask`, `SyncWarp`, and
the constant `WarpSize` (32, a Go constant because CUDA's own `warpSize` is a
variable no constant expression can use).

**None takes a participation mask.** CUDA's `_sync` forms do; the emitter
writes `0xffffffff` and the Go-level contract is that every thread of the warp
reaches the call. In a block that is not a multiple of 32 that mask names lanes
which do not exist, which CUDA leaves undefined — launch whole warps, and say
so with `AssumeBlockDim`. A negative _constant_ lane offset is refused, because
CUDA reads those as unsigned.

## 5. Launch contracts

These are not rules about source text. They are discovered while lowering and
travel with the unit, so a launch can be held to them.

- **`AssumeBlockDim(n)`** emits no code; it records the block size the kernel
  was written for, and both `Kernel.Launch` and the emulator refuse any other.
  What is counted is threads per block across all three axes, so a 16×16 block
  satisfies `AssumeBlockDim(256)`. At most one value per kernel, a positive
  constant, at the top level of the kernel, and never in a device function.
- **Shared memory** is checked twice: the static total against the device limit
  at `Build`, and `static + n × width` at `LaunchShared`, which is the first
  moment the dynamic half exists.
- **`LaunchShared` and `Launch` are separate calls**, and using the wrong one is
  refused. They are separate because the mistake is silent either way: a kernel
  with a dynamic tile launched through `Launch` gets zero bytes of it, and one
  without launched through `LaunchShared` is handed a size for a tile it does
  not have. `n` is an element count; bytes are the emitter's business.
- **Pointers carry `const` and `__restrict__`.** Read-only is proved across the
  call graph, conservatively — anything unreadable counts as a write. Because
  `__restrict__` promises something Go cannot, it is checked rather than
  assumed: `Kernel.Launch` refuses overlapping device ranges where the kernel
  writes through one of them, and passing one buffer twice to a helper that
  writes it is refused at lowering. Two _read-only_ parameters may share a
  buffer deliberately — what `restrict` forbids is reaching a modified object
  through another pointer, so `dot(x, x)` is sound. The CPU emulator checks
  none of this: it never sees the caller's slices, and there a kernel is
  ordinary Go, where aliasing is defined.

## 6. Barriers and divergence

A `__syncthreads()` that only some threads of a block reach is undefined
behaviour, and so is a `_sync` warp built-in that only some lanes reach. Three
static rules refuse the shapes that cause it: a barrier under a thread-varying
condition, a barrier inside a loop whose trip count is thread-varying, and a
barrier preceded by a `return` under a thread-varying condition. The analysis
follows calls, so a barrier inside a helper counts as one at the call site.

What is **thread-varying**: `ThreadIdx*`, `GlobalID*`, `LaneID`, the shuffles,
`Ballot`/`Any`/`All`/`ActiveMask`, the prior value an atomic returns, a load at
a varying index, and anything derived from those.

What is **block-uniform**, and therefore fine to put a barrier under:
`BlockIdx*`, `BlockDim*`, `GridDim*`, a scalar parameter, `len()`, and every
constant. `if ctx.BlockIdx() == 0 { ctx.SyncThreads() }` is legal.

Two limits, stated rather than left to be found. The warp primitives are held
to the block-uniform lattice, which is stricter than they need — `Any` and
`All` are warp-uniform — and the rules work over statements, so they do not see
into a condition: `ctx.LaneID() == 0 && ctx.Any(p)` short-circuits in C as it
does in Go, calling `__any_sync` on one lane with a full mask, and no rule here
catches it.

## 7. Refusals

Every entry is pinned by a case in `simt/errors_test.go`, and the quoted phrase
is what the diagnostic contains. `simt/spec_test.go` checks both directions.

### Kernel and function shape

- no context parameter — diagnostic: `first parameter must be gpu.Ctx`
- returns a value — diagnostic: `must not return values`
- a method call — diagnostic: `methods are not supported in kernels`
- a recursive device function — diagnostic: `down calls itself`
- mutually recursive device functions — diagnostic: `even, which is already being lowered`
- calling another kernel — diagnostic: `Other is a kernel`
- a device function returning two values — diagnostic: `must return at most one value`
- a device function naming its result — diagnostic: `must not name its result`
- a variadic device function — diagnostic: `must not be variadic`
- a gpu.Ctx helper that did not say it was a device function — diagnostic: `where is a kernel`
- the refusal names the device marker — diagnostic: `Mark it //gocuda:device`
- //gocuda:device on a function that takes no gpu.Ctx — diagnostic: `//gocuda:device does nothing on half, which takes no gpu.Ctx`
- //gocuda:device on a function nothing calls — diagnostic: `//gocuda:device does nothing on unused`

### Directives and double precision

- float64 without the directive — diagnostic: `float64 needs //gocuda:float64 on kernel K`
- //gocuda:float64 on a device function — diagnostic: `belongs on the kernel, not on device function half`
- a float64 helper whose type never surfaces — diagnostic: `gpu.Sqrt64 is double precision and needs //gocuda:float64 on kernel K`
- a float64 helper on a float64 kernel without the directive — diagnostic: `needs //gocuda:float64 on kernel K`

### Narrow integers, which are storage only

- a narrow local — diagnostic: `not a variable, a parameter or a result`
- arithmetic on narrow elements — diagnostic: ``so `*` on one can give a different answer``
- unary minus on a narrow element — diagnostic: ``so `-` on one can give a different answer``
- comparing narrow elements — diagnostic: ``so `<` on one can give a different answer``
- compound assignment into a narrow slot — diagnostic: `` `+=` is refused anyway ``
- incrementing a narrow slot — diagnostic: `` `++` is refused anyway ``
- min on narrow elements — diagnostic: `min has no overload for it`
- a shift by a narrow count — diagnostic: ``so `<<` on one can give a different answer``

### min and max, where the two disagree about NaN

- the builtin `min`/`max` on `float32` — diagnostic: `write gpu.Fmin(a, b)`
- the builtin `min`/`max` on `float64` — diagnostic: `write gpu.Fmax64(a, b)`

Go's `min` and `max` propagate a NaN operand; CUDA's `fminf` and `fmaxf` follow
IEEE minNum and ignore one, returning the number. So `min(0.0/0.0, x)` is a NaN
in Go and `x` on the device. `gpu.Fmin` and `gpu.Fmax` are the spelling that
means the device's answer, and the emulator implements them to match. The
**integer** overloads are untouched: no integer is a NaN.

### Shifts, where the two widths disagree

- a Go `int` shifted left by a computed amount — diagnostic:
  `is 64 bits in Go and 32 on the device`
- a shift by a constant the C type cannot take — diagnostic:
  `undefined on the device`

A _constant_ left shift of an `int` is accepted: its result is as bounded as
the value is, which is the case the narrowing above was always about. What is
**not** checked, and is stated here rather than left to be found: `int32` or
`int64` shifted by a computed amount. Both are the same width in both
languages, so the only disagreement left is a count that reaches the width at
run time — Go defines that as zero and C leaves it undefined — and telling
`x << (k & 31)`, which is how one writes it safely, from `x << k` needs a
range analysis the lowering does not have.

## 3a. Signed overflow wraps, as Go says it does

**Go defines signed integer overflow as wrapping**; C leaves it _undefined_,
which is not a wrong number but a licence for the compiler to assume it cannot
happen. So `+`, `-`, `*`, unary `-` and `<<` on a signed integer are emitted
through the **unsigned type of the same width** and converted back, which is
the only spelling of modular arithmetic C defines.

The conversion happens once per _region_, not once per operator, or the source
would disappear under casts:

```c
int y = (int)((unsigned int)(a) + (unsigned int)(b) * (unsigned int)(c));
```

`/`, `%` and `>>` end a region rather than joining it, because they mean
something different on unsigned operands; `&`, `|` and `^` would be safe either
way and are left outside too, so that the rule has no exception to remember.

Two things this does **not** do:

- **It does not make Go's `int` equal to C's.** Go wraps at 64 bits and the
  device at 32, which is the narrowing above. Wrapping makes the C _defined_,
  not _equal_.
- **`MinInt / -1` is still undefined on the device.** It is `MinInt` in Go, and
  routing it through unsigned cannot help — unsigned division is a different
  operation. Nothing refuses it, because the values are not known until the
  kernel runs.

### Types that cannot cross

- uint — diagnostic: `use uint32 or uint64`
- []int — diagnostic: `cannot cross to the device`
- int(x) from int64 — diagnostic: `truncates on the device`
- map — diagnostic: `unsupported type`

### Arrays

- an array parameter — diagnostic: `Go passes an array by value and C would pass a pointer to it`
- an array result — diagnostic: `C cannot return an array`
- whole-array assignment — diagnostic: `cannot be assigned`
- a zero-length array — diagnostic: `has no elements`

### Structs

- struct equality — diagnostic: `field by field in Go, which C cannot do`
- a field named like the emitted padding — diagnostic: `spelled like the padding gocuda emits`
- an embedded field — diagnostic: `embedded field`
- a blank struct field — diagnostic: `has a blank field`
- an anonymous struct type — diagnostic: `unsupported type struct{X float32} on the device`
- a switch over a struct — diagnostic: `cannot switch on kernels.P`

### Shared memory

- non-constant shared memory — diagnostic: `constant size`
- a runtime size on a typed tile — diagnostic: `use SharedDynI32, whose length the launch gives`
- a shared tile inside a conditional — diagnostic: `top level of the function`
- two dynamically sized shared tiles — diagnostic: `at most one dynamically sized shared tile, and a is already one`
- a dynamically sized shared tile inside a device function — diagnostic: `may only be declared in a kernel: its length is a launch parameter`
- a dynamic shared tile inside a marked device function — diagnostic: `device function`
- a dynamic tile's length collides with a parameter — diagnostic: `collides with parameter s_len`
- a dynamic tile's length collides with a slice's generated length — diagnostic: `collides with the length generated for slice parameter float`

### Launch contracts

- AssumeBlockDim inside a device function — diagnostic: `AssumeBlockDim may only be called in a kernel`
- non-constant block size — diagnostic: `constant block size`
- two different block sizes — diagnostic: `already declared a block size of 128`
- conditional block size — diagnostic: `top level of the kernel body`

### Atomics

- an atomic on a slice expression rather than a buffer — diagnostic: `needs the buffer itself as its first argument`
- an atomic on a package-level buffer — diagnostic: `declared outside the kernel`

### Warp primitives

- a negative source lane — diagnostic: `takes a lane offset and -1 is negative`
- a negative shuffle delta — diagnostic: `takes a lane offset and -2 is negative`

### Barriers and divergence

- a barrier under a thread-varying if — diagnostic: `ctx.SyncThreads() is under a thread-varying if`
- a barrier under a thread-varying switch — diagnostic: `ctx.SyncThreads() is under a thread-varying switch`
- a barrier inside a loop with a thread-varying trip count — diagnostic: `ctx.SyncThreads() is inside a loop whose trip count differs between threads`
- a barrier after a thread-varying return — diagnostic: `ctx.SyncThreads() is preceded by a return at`
- a barrier reached through a call, after a thread-varying return — diagnostic: `stage, which reaches ctx.SyncThreads(), is preceded by a return at`
- a warp primitive under a thread-varying if — diagnostic: `ctx.ShuffleF32 is under a thread-varying if`

### Statements and expressions

- goroutine — diagnostic: `unsupported statement`
- type switch — diagnostic: `type switches are not supported`
- assigning from a two-valued call — diagnostic: `assigning 2 values from one expression is not supported`
- a parallel assignment in a for clause — diagnostic: `multiple assignment is not supported in a for clause`
- the blank identifier in a parallel assignment — diagnostic: `the blank identifier is not supported in kernels`
- break inside a switch that lowered to an if/else chain — diagnostic: `cannot break out of a switch`
- a label on something other than a loop — diagnostic: `a label may only be placed on a for loop`
- range with a value, assigned rather than declared — diagnostic: ``only `for i := range x` and `for i, v := range x` are supported``

### Names the emitter generates

- parameter collides with a generated length — diagnostic: `x_len is the length generated for slice parameter x`
- a variable spelled like an escaped keyword — diagnostic: `int is a C++ keyword and is emitted as int_`

### Aliasing

- one buffer passed as two parameters of a helper that writes — diagnostic: `blend is passed the same buffer as both out and a`
