# nd-layout design log

Agent scratch memory, not finished documentation. Human-written documentation belongs elsewhere. Keep the
sections current; append significant session notes at the bottom.

## Current working decisions

- `nd-layout` owns the index-to-offset vocabulary. It does not name element types, storage types,
  pointers, or allocation-owning memory abstractions. `nd-core` is the first consumer layer and owns
  backed views, storage access, and eventually ownership/memory-relevant invariants.
- Unsafe extension points are allowed, but unsafe markers are demand-driven. Do not introduce `unsafe`
  merely because a concept is expected to support unsafe consumers eventually. Introduce it at the first
  concrete boundary where memory access or another unsafe operation must trust an invariant that cannot
  be checked.
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
- `IndexSpace` exposes logical iteration through Rust iterators rather than a manual first/next protocol.
  This fits the iterator-first end state and integrates with the wider Rust ecosystem. This is canonical
  logical row-major traversal only; memory-order, stride-order, fusion-compatible, and parallel evaluation
  schedules are deliberately deferred because they require layout/producer information unavailable to
  `IndexSpace`.
- Integration with the wider Rust ecosystem is a design requirement, especially `Vec`, slices, and
  arrays. However, raw Rust arrays and slices are not inherently ranked. `[usize; N]` can represent an
  N-axis shape, but as element storage it is a one-dimensional container. Use semantic wrappers,
  role-specific traits, and inexpensive conversions at boundaries rather than trait implementations that
  give one raw container conflicting meanings.
- Structural transformations should be lazy metadata/layout transformations, not data operations.
- Backed storage needs an explicit origin: the storage position corresponding to layout offset zero.
  Logical zero is not generally the first storage position, even for positive strides. For nonempty
  rectangular layouts the zero index is valid, so origin must denote a valid storage element position;
  for empty layouts it may denote a slice boundary.
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

Unsafe traits and functions should be scarce and demand-driven. Mark code unsafe only when a concrete
consumer must rely on an invariant to preserve memory safety and cannot realistically validate or enforce
that invariant. Ordinary correctness bugs are not enough. The existence of a likely future unsafe consumer
is also not enough on its own in this ground-up rewrite: the exact proof obligation will be clearer when
that consumer exists.

`Rank`, `Ranked`, `Shape`, `IndexSpace`, and the initial `Layout` are safe traits. `Layout` still has
memory-relevant correctness semantics, but until storage code exists there is no concrete unsafe proof
boundary. When that boundary appears, choose deliberately between making `Layout` unsafe, adding an unsafe
marker trait such as `TrustedLayout`, validating at the storage boundary, or a combination. Evaluate
`AxisOps` and `Strided` when their concrete uses are designed.

`Layout` initial surface, subject to iteration:

- `IndexSpace` as a supertrait, providing shape access, canonical indices, validity, and logical iteration
- `offset` as the safe checked form, returning `Option<isize>`
- `offset_unchecked` as a safe arithmetic fast path; unchecked memory access remains a downstream concern
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
   validity, uniqueness, enumeration count, and row-major logical order.
4. Fixed-rank dynamic-extent `StridedLayout<N>` using semantic shape/stride representations backed by
   `[usize; N]` and `[isize; N]`, with checked construction and affine offset computation. Hegel tests
   for bounds, offsets, iteration, and offset bounds. Kani proofs for arithmetic-overflow invariants.
5. Offset bounds and conservative uniqueness, with enumerated small-layout tests and Kani where feasible.
6. Minimal read-only backed consumer in `nd-core`: a `SliceView` over a slice, generic in `Layout`, with
   explicit origin and checked value access. Use it to pressure-test associated index types, offset
   bounds, negative strides, zero strides, empty layouts, non-strided layouts, and the origin model before
   designing axis operations.
7. First axis operation, probably `reverse_axes`, now tested through backed views as well as layout algebra;
   then one operation at a time: collapse, slice, invert, insert/remove, permute, merge.
8. A non-strided layout such as a tiled/block layout to pressure-test generality before optimizing.
9. Dynamic-rank owned representation using allocation, kept simple initially, with `Vec`/slice conversion
   boundaries considered explicitly.
10. `Strided` trait, contiguity queries, and explicit fast-path capabilities.
11. Const-extent/const-stride families and static-axis operation traits.
12. Iteration-order/evaluation-schedule abstractions needed by the lazy producer layer.

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

Follow-up: unsafe-trait marking was narrowed after discussing the difference between correctness bugs and
memory-safety authority. `Shape` is safe.

Follow-up: the initial `Layout` design briefly used an unsafe trait and unsafe `offset_unchecked`. The
user then clarified the demand-driven process rule for this fresh rewrite: because no memory-accessing
consumer exists yet, it is premature to guess which implementation invariants must become unsafe proof
obligations. `Layout` and all its methods are therefore safe for now. Correctness invariants are still
tested and proved; when concrete unsafe consumers arrive, the design can introduce exactly the needed
unsafe trait, marker trait, or validation boundary.

Follow-up: the first consumer slice was reprioritized ahead of axis operations. The user approved a minimal
`nd-core` spike: a read-only fixed-rank `StridedView` over a slice, with explicit storage origin and safe
checked value access. This is intended to ground layout semantics in real value extraction before further
layout algebra. The discussion also clarified that logical row-major iteration is only one traversal order;
memory/evaluation-order abstractions remain deferred.

Follow-up: the spike was generalized from `StridedView` to `SliceView<'a, T, L>` where `L: Layout`. The
layout's associated index type now determines the index accepted by `get`, and the view delegates `Ranked`
and `IndexSpace` to its layout. Generic `get` accepts `&L::Index`: passing the associated index by value is
not generally appropriate because the core operation only borrows it and future index types may not be cheap
to move or copy. Ergonomic fixed-rank value indexing can be layered on later.

Follow-up: immutable logical iteration was added to `SliceView`. It traverses canonical row-major logical
indices, recomputes checked offsets for each index, and uses checked slice access. The private `get_valid`
helper is now the clearest future unchecked-access boundary: it represents access by an index already known
to be valid. It is currently safe and checked; incorrect layout implementations can cause panics but not
memory unsafety. Element references use the backing slice lifetime and can outlive the view object. The
opaque `impl Iterator` return hides capabilities such as exact size or double-ended iteration, so named or
associated iterator types may become necessary when those capabilities matter.

Follow-up: `reverse_axes` remains the preferred first axis operation because it changes logical index order
without changing storage, rank, offset bounds, uniqueness, or logical-zero origin. It is the smallest useful
structural transformation of an array-like object and can now be validated through observed iteration order,
not just layout algebra. It should expose how transformations compose with backed views before ownership,
mutability, or more origin-sensitive operations are designed.

Follow-up: benchmarking should be established incrementally alongside functionality. A benchmark should be
small, compare equivalent logical semantics rather than implementation names, keep construction outside the
measured region, assert correctness outside the measured region, and use `black_box` on benchmark inputs or
views when the intended measurement is runtime-opaque access. Comparison against upstream `ndarray` is
useful even when the new implementation is expected to lose; the purpose is to measure the size and shape of
the gap, not to require parity at every step.

Initial local wall-clock baseline for summing a 128 by 128 `u64` array in logical row-major order:

| implementation | layout | median |
|---|---|---:|
| Rust slice | contiguous | 1.322 µs |
| `ndarray` 0.17.2 | contiguous | 1.291 µs |
| `ndarray` 0.17.2 | row-strided | 2.103 µs |
| `SliceView` | contiguous | 50.08 µs |
| `SliceView` | row-strided | 48.52 µs |

The large gap is expected: `SliceView` performs complete checked affine offset calculation and checked
slice access for every element, while `ndarray` uses specialized counted or slice-backed iterators. This is
evidence for designing cursor/incremental traversal next rather than optimizing isolated random-access
`get` calls.

Follow-up: `reverse_axes` was implemented as a metadata-only transformation for `StridedLayout` and
`SliceView`. Reversing axis order reverses shape and strides but preserves the offset set exactly, so the
cached offset bounds can be reused without recomputation. Because the layout's zero offset is unchanged,
the backed view's explicit origin is also unchanged. This confirms that axis permutation is independent
of backing storage and does not require revalidation when it starts from a valid view. Hegel coverage
checks the algebraic correspondence `reversed[index] == original[reverse(index)]`, involution, bounds,
and uniqueness for rank-two layouts; fixed tests cover empty and rank-zero layouts. A backed-view test
checks both observed transposed value order and equivalence of point access under reversed indices.

Reverse-axes benchmark evidence for summing the same 128 by 128 `u64` data in logical row-major order:

| implementation | layout | median |
|---|---|---:|
| Rust slice | contiguous | 1.322 µs |
| `ndarray` 0.17.2 | contiguous | 1.301 µs |
| `ndarray` 0.17.2 | reversed contiguous axes | 6.249 µs |
| `ndarray` 0.17.2 | row-strided | 2.104 µs |
| `ndarray` 0.17.2 | reversed row-strided axes | 8.624 µs |
| `SliceView` | contiguous | 48.49 µs |
| `SliceView` | reversed contiguous axes | 47.37 µs |
| `SliceView` | row-strided | 47.87 µs |
| `SliceView` | reversed row-strided axes | 47.37 µs |

`SliceView` timing is essentially flat across these layouts because full affine offset computation and
checked indexing dominate memory effects. Upstream `ndarray` slows substantially for transposed logical
traversal, but remains much faster than `SliceView`; the gap narrows from roughly 37x contiguous to 7.6x
for reversed contiguous axes and 5.5x for reversed row-strided axes. This is direct evidence that traversal
specialization and incremental cursor state are more important than further micro-optimizing the current
random-access path.

Follow-up: upstream ndarray iteration provided two important reference points. First, logically contiguous
arrays iterate through the backing slice directly. Second, otherwise iteration uses a counted cursor whose
`fold` special-cases the innermost axis. Since operations such as `sum` are fold-driven, an iterator with
only an efficient `next` implementation would miss much of the relevant performance.

The first new traversal design therefore made offset iteration a layout-level counterpart to logical index
iteration: `Layout` now has an associated `Offsets` iterator in logical order. `LogicalOffsets` is the dumb,
obviously correct implementation for arbitrary layouts, checking the offset for every enumerated logical
index. `StridedOffsets` is the fast strided implementation, advancing logical index and offset together.
Its `fold` implementation consumes the innermost axis in a tight loop and only performs carry/reset work at
row boundaries. `SliceView::iter` now consumes layout offsets rather than logical indices. Its internal
valid-position path can use wrapping origin-plus-offset arithmetic because constructor validation has
already established bounds for every valid offset; memory access itself remains checked. Incorrect layout
implementations can still cause wrong values or panics, but not memory unsafety.

Benchmarks after the strided cursor and innermost-axis fold, again for 128 by 128 `u64` logical summation:

| implementation | layout | median |
|---|---|---:|
| Rust slice | contiguous | 1.322 µs |
| `ndarray` 0.17.2 | contiguous | 1.343 µs |
| `ndarray` 0.17.2 | reversed contiguous axes | 6.499 µs |
| `ndarray` 0.17.2 | row-strided | 2.124 µs |
| `ndarray` 0.17.2 | reversed row-strided axes | 8.749 µs |
| `SliceView` | contiguous | 6.749 µs |
| `SliceView` | reversed contiguous axes | 6.999 µs |
| `SliceView` | row-strided | 6.958 µs |
| `SliceView` | reversed row-strided axes | 8.874 µs |

This reduced `SliceView` logical iteration from roughly 48 µs to 7-9 µs. The remaining contiguous gap is
about 5x because upstream can use a backing-slice iterator, while `SliceView` still performs per-element
offset-to-position conversion and checked access. In reversed layouts the implementations are nearly at
parity because strided memory traversal dominates. The next clear capability is a layout-level contiguous
forward-offset query so `nd-core` can safely select a backing-slice fast path without unsafe code.

Follow-up: `Layout::forward_contiguous_len` now provides that capability query. `Some(len)` is intended to
mean that logical iteration produces exactly the offsets `0..len` in order. The default is conservative
`None`. `StridedLayout` computes the query exactly for positive-forward affine contiguity. The important
edge case is unit-length axes: their strides are inert because coordinate zero is the only valid coordinate,
so they must not prevent a layout from being contiguous. Empty layouts report length zero and rank-zero
layouts report length one.

`SliceView::iter` uses the query to select a backing-slice iterator when the corresponding storage range is
available. This is safe: an incorrect layout implementation can produce wrong values or fall back to the
generic path, but slicing still cannot access memory outside the backing slice. The opaque iterator return
now hides a small enum with backing-slice and layout-offset variants. Its `fold` forwards to the backing
slice's own `fold` in the contiguous case, preserving vectorization-friendly behavior.

Benchmarks after the contiguous fast path:

| implementation | layout | median |
|---|---|---:|
| Rust slice | contiguous | 1.332 µs |
| `ndarray` 0.17.2 | contiguous | 1.197 µs |
| `ndarray` 0.17.2 | reversed contiguous axes | 6.041 µs |
| `ndarray` 0.17.2 | row-strided | 2.103 µs |
| `ndarray` 0.17.2 | reversed row-strided axes | 8.707 µs |
| `SliceView` | contiguous | 1.307 µs |
| `SliceView` | reversed contiguous axes | 6.958 µs |
| `SliceView` | row-strided | 6.999 µs |
| `SliceView` | reversed row-strided axes | 8.916 µs |

Contiguous logical iteration is now at parity with raw slice and upstream ndarray iteration. Reversed
contiguous and reversed row-strided layouts are also close to upstream because their memory access patterns
dominate. The largest remaining gap is ordinary row-strided logical iteration: upstream remains about 3.3x
faster because it can exploit contiguous innermost rows with a tight pointer loop, while `SliceView` still
converts and checks every individual offset. This suggests the next iteration-design question: whether
layouts should expose offset runs or another innermost-run traversal capability, especially for fold-driven
evaluation, rather than only whole-layout contiguity and per-element offsets.
