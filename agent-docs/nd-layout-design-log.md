# nd-layout design log

Agent scratch memory, not finished documentation. Human-written documentation belongs elsewhere. Keep the
sections current; append significant session notes at the bottom.

## Current working decisions

- `nd-layout` owns the index-to-offset vocabulary. It does not name element types, storage types,
  pointers, or allocation-owning memory abstractions.
- Unsafe is allowed. Extension points are unsafe traits with documented contracts. There is no separate
  `ValidLayout` marker: users should write `L: Layout`, while implementors write `unsafe impl Layout`.
- No panics. Safe APIs must be total and use `Option` or `Result` for invalid input.
- Performance is a goal from the start, but optimization is incremental. Preferred mechanisms:
  construction-time validation, unsafe unchecked variants with explicit contracts, iterators that yield
  valid indices, static-axis operations, and later compile-time extents.
- `no_std` and `no_alloc` are not hard constraints driving the design. Do not contort representations
  around them. Correctness and simple ownership come before small-inline optimizations.
- Shapes are read-only in the narrow sense that there is no public API for mutating individual axis
  lengths directly. Controlled invariant-preserving mutation and whole-value replacement are allowed.
  By-value operations remain the baseline because they can change type.
- No implicit broadcasting. If broadcasting exists, it is explicit. This removes pressure to build
  rank-max machinery early.
- The end state is mostly lazy and iterator-first. `map` produces a `Map`; it does not allocate or apply
  eagerly. Materialization is explicit. Elementwise computation and fusion live above `nd-layout`.
- Structural transformations should be lazy metadata/layout transformations, not data operations.
- Axis operations are by-value and use per-operation associated output types. A single `WithRank` GAT was
  explored and rejected.
- Runtime-axis operations erase static knowledge because the output type cannot depend on the runtime
  axis value. Constant-sized axes need a separate static-axis method set later.
- Compile-time shapes and strides are important end-state representations, but they are not the first
  implementation slice.

## Core vocabulary

| Concept | Working meaning | Excluded |
|---|---|---|
| `Rank` | compile-time knowledge of axis count | shape, index, stride |
| `Ranked` | bridge from `Rank` to runtime `ndim()` | owner of runtime rank |
| `Shape` | read-only axis lengths | direct mutation, strides |
| `Index` | canonical coordinate type for a layout | user-facing indexing syntax |
| `Layout` | total map from valid logical indices to signed element offsets | memory, elements, storage |
| `AxisOps` | invariant-preserving structural transformations | pointer manipulation |
| `Strided` | affine layout with observable signed strides | all layouts |
| offset bounds | minimum and maximum offsets reachable through the layout | allocation ownership |
| uniqueness | whether distinct valid indices can alias, conservatively reported | storage ownership uniqueness |

Offsets are signed element units. Strides are `isize` directly, not `usize` reinterpreted as `isize`.

`DynRank` means rank unknown at compile time, not a second runtime source of truth. Runtime rank comes
from the shape.

## Initial trait direction

Unsafe traits whose correctness can affect downstream memory safety: `Shape`, `Layout`, `AxisOps`, and
`Strided`. `Rank` and `Ranked` can remain safe vocabulary traits.

`Layout` initial surface, subject to iteration:

- associated `Shape` and `Index`
- borrowed shape access
- `offset_checked`
- unsafe `offset_unchecked`
- offset bounds, probably `Option<(isize, isize)>` with inclusive bounds and `None` for empty layouts
- conservative uniqueness query
- logical `first_index` / `next_index`

Deferred from initial `Layout`:

- `index_of(offset)` / inversion
- stride queries
- construction and mutation
- broadcasting and rank arithmetic
- memory-order fast paths

## Lazy and iterator-first implications

- `Map` is not a layout concern. It belongs to the computation/producer layer above `nd-layout`.
- Elementwise lazy operations preserve the logical index space and normally reuse the input evaluation
  layout or schedule.
- Structural lazy operations transform index spaces and layouts without touching elements.
- Layout design must leave room for order-aware traversal. Logical index iteration alone is not enough
  for high-performance evaluation. A later capability may describe memory order, stride order, offset
  runs, or another evaluation schedule.
- Materialization/copying is a terminal operation and should not be implicit.
- Open tension: if every lazy array must implement `Layout` as index-to-offset, generated arrays and
  fused pipelines may need artificial virtual offsets. The alternative is a broader shaped/index-space
  abstraction above or beside `Layout`. Do not settle this until the lazy producer layer is designed.
- Current lean from follow-up discussion: random access to a `Map` is evaluation, not memory access.
  Evaluation recursively delegates through the lazy expression tree; only backed leaves need `Layout` plus
  storage/access to reach an element. Thus `Map` itself should probably implement a producer/evaluation
  trait, not `Layout`.
- `std::ops::Index` is memory-biased because it must return a reference. A value-producing `Map` cannot
  implement it honestly. Lazy producers need an `eval`/`at`-style method returning a value; backed arrays
  can additionally support reference-returning indexing.
- This suggests a possible end-state split: shape/index-space vocabulary shared by everything, `Layout`
  for offset-addressable leaves, and producer/evaluation traits above. It may also mean logical index
  iteration does not belong directly on `Layout`.

## First implementation slices

1. Rank vocabulary seed: `Rank`, `ConstRank<N>`, `DynRank`, `Ranked`. Mostly compile-time assertions.
2. `Shape` plus `[usize; N]` implementation. Hegel tests for iteration and checked size semantics.
3. Fixed-rank dynamic-extent `StridedLayout<N>` using `[usize; N]` and `[isize; N]`, with checked
   construction and affine offset computation. Hegel tests for bounds, offsets, iteration, and offset
   bounds. Kani proofs for arithmetic-overflow invariants.
4. Offset bounds and conservative uniqueness, with enumerated small-layout tests and Kani where feasible.
5. First axis operation, probably `reverse_axes`, then one operation at a time: collapse, slice, invert,
   insert/remove, permute, merge.
6. A non-strided layout such as a tiled/block layout to pressure-test generality before optimizing.
7. Dynamic-rank owned representation using allocation, kept simple initially.
8. `Strided` trait, contiguity queries, and explicit fast-path capabilities.
9. Const-extent/const-stride families and static-axis operation traits.
10. Iteration-order/evaluation-schedule abstractions needed by the lazy producer layer.

## Reference findings worth preserving

- The old `Dimension` trait is a negative example: it mixes rank, shape, index, strides, mutation,
  construction, iteration, and arithmetic.
- The refactor branch's useful contribution is rank vocabulary and the rule that runtime rank lives in
  shape, not in `DynRank`.
- The spike validates the split into `Layout`, axis operations, and `Strided`.
- `BlockLayout` proves a non-strided layout can fit the traits, and shows that origin deltas are not a
  universal representation; they are part of the operation algebra and may be zero for layouts that
  internalize transformations.
- `ExtStrided<Sh, St, O>` shows that rank, axis lengths, stride values, and axis-order guarantees are
  independent static/dynamic axes.
- Axis-order guarantees must be enforced by constructors before they can be trusted as type-level facts.
- Per-operation associated types compose better than a single rank-changing GAT.
- Sparse is storage, not layout.

## Rejected or deferred

- `Cow` in shape/stride traits
- one single strided `Layout` trait
- porting `Dimension` wholesale
- a single `WithRank` GAT for rank-changing outputs
- typenum as the core rank representation
- implicit broadcasting
- small-inline dynamic-shape optimization in the first implementation
- compile-time extents as the first implementation slice

## Chronological log

### 2026-11-26

Initial design discussion after reading `layout-spike`, `upstream/refactor`, `shape-trait`, and relevant
`ndarray-design` branches. The user relaxed the earlier no-std/no-alloc/no-unsafe constraints, confirmed
unsafe extension traits, rejected panics, rejected implicit broadcasting, confirmed the initial layout
surface, and added the lazy/iterator-first end-state requirement: `map` produces a `Map` rather than
allocating or eagerly applying.
