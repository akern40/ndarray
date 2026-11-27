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
- Use the three-way split: `Shape` is axis-length geometry; `IndexSpace` is the object-level canonical
  index/validity/logical-iteration interface; `Layout` adds index-to-offset mapping. Lazy producers use
  `IndexSpace` plus evaluation rather than pretending to be layouts.
- Integration with the wider Rust ecosystem is a design requirement, especially `Vec`, slices, and
  arrays. However, raw Rust arrays and slices are not inherently ranked. `[usize; N]` can represent an
  N-axis shape, but as element storage it is a one-dimensional container. Use semantic wrappers,
  role-specific traits, and inexpensive conversions at boundaries rather than trait implementations that
  give one raw container conflicting meanings.
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

- `IndexSpace` as a supertrait, providing shape access, canonical indices, validity, and logical iteration
- `offset` as the safe checked form, returning `Option<isize>`
- unsafe `offset_unchecked`
- offset bounds, probably `Option<(isize, isize)>` with inclusive bounds and `None` for empty layouts
- conservative uniqueness query

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
- Resolved direction: use a broader object-level `IndexSpace` abstraction rather than making lazy arrays
  implement `Layout`. `Shape` remains axis-length geometry; `IndexSpace` adds canonical index type,
  validity, and logical iteration; `Layout` adds offset mapping.
- Random access to a `Map` is evaluation, not memory access. Evaluation recursively delegates through the
  lazy expression tree; only backed leaves need `Layout` plus storage/access to reach an element. Thus
  `Map` itself should implement producer/evaluation traits, not `Layout`.
- `std::ops::Index` is memory-biased because it must return a reference. A value-producing `Map` cannot
  implement it honestly. Lazy producers need an `eval`/`at`-style method returning a value; backed arrays
  can additionally support reference-returning indexing.
- End-state split: shape/index-space vocabulary shared by everything, `Layout` for offset-addressable
  leaves, and producer/evaluation traits above. Logical index iteration belongs to `IndexSpace`, not
  directly to `Layout`.

## First implementation slices

1. Rank vocabulary seed: `Rank`, `ConstRank<N>`, `DynRank`, `Ranked`. Mostly compile-time assertions.
2. `Shape` plus a semantic fixed-rank shape representation backed by `[usize; N]`, with inexpensive
   array/slice conversions. Hegel tests for axis access and checked size semantics.
3. `IndexSpace` plus logical index iteration for the fixed-rank shape representation. Hegel tests for
   validity, enumeration count, and round-tripping through first/next.
4. Fixed-rank dynamic-extent `StridedLayout<N>` using semantic shape/stride representations backed by
   `[usize; N]` and `[isize; N]`, with checked construction and affine offset computation. Hegel tests
   for bounds, offsets, iteration, and offset bounds. Kani proofs for arithmetic-overflow invariants.
5. Offset bounds and conservative uniqueness, with enumerated small-layout tests and Kani where feasible.
6. First axis operation, probably `reverse_axes`, then one operation at a time: collapse, slice, invert,
   insert/remove, permute, merge.
7. A non-strided layout such as a tiled/block layout to pressure-test generality before optimizing.
8. Dynamic-rank owned representation using allocation, kept simple initially, with `Vec`/slice conversion
   boundaries considered explicitly.
9. `Strided` trait, contiguity queries, and explicit fast-path capabilities.
10. Const-extent/const-stride families and static-axis operation traits.
11. Iteration-order/evaluation-schedule abstractions needed by the lazy producer layer.

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

Follow-up: the user approved the `Shape`/`IndexSpace`/`Layout` split and emphasized integration with
`Vec`, slices, and arrays. Raw arrays and slices must not acquire conflicting semantic meanings through
blanket trait implementations; use semantic wrappers, role-specific traits, and cheap conversions instead.
