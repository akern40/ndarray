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
- Rank-preserving axis operations are by-value. The current probe groups a closed operation set in
  `AxisOps` with direct methods and per-operation associated output types. Rank-changing operations remain
  a separate algebra. A single `WithRank` GAT was explored and rejected.
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

Follow-up: the first storage extraction is intentionally small and concrete. `SliceStorage<'a, T>` inside
`nd-core` owns the borrowed slice abstraction: construction-time validation, checked element access, and
checked contiguous range access. It does not know about layouts except in a validation method, and it does
not know the logical origin. `SliceView` is now a composition of storage, layout, and explicit origin. This
is not yet a general storage crate or trait hierarchy; those should wait for a second storage
representation or mutable access to provide evidence.

Construction validation now includes the upstream-inspired size invariants that can be checked at this
boundary: the product of nonzero axis lengths must fit `isize`; the element-distance span of the layout
must fit `isize`; and that span multiplied by `size_of::<T>()` must fit `isize` bytes. Byte validation is
necessarily in `nd-core`, not `nd-layout`, because layouts do not know element types. ZSTs are supported
under the same element-count and element-span caps even though their byte span is zero. This deliberately
rejects some theoretically constructible huge ZST shapes, matching the chosen ndarray-like cap rather than
redesigning offsets around ZST extremes.

The extraction also clarifies empty-layout behavior: with no accessible offsets there is no layout span to
validate, but the explicit origin must still be a valid slice boundary, including one-past-the-end. This is
a deliberately generalized-layout interpretation rather than upstream ndarray's raw-pointer interpretation:
movement is measured over valid indices, not over hypothetical axis movement in an empty index space. If
future storage implementations perform pointer arithmetic even for empty arrays, they may need a stronger
storage-specific invariant. Regression tests cover ordinary ZST iteration and rejection of a logically huge
empty layout. Benchmarks show no meaningful iteration regression; validation is construction-time and the
small storage methods inline away.

### 2026-11-27

Design discussion before trait formulation: a structural operation is storage-independent when its result
can be represented entirely as a transformation of index-space, layout, and placement metadata over the
same virtual storage-offset space. Such an operation preserves backing-storage identity, inspects no
elements, allocates or copies no storage, requires no host element access, and requires no device
synchronization. Examples should include slicing, transposing, reversing axes, broadcasting, inserting or
removing length-one axes, splitting, and metadata-only reshapes when representable.

Operations outside that definition are not structural view operations. Allocation, copying, compaction,
concatenation, format conversion, or production of a new storage identity belongs to explicit
materialization or adaptation operations so costs remain visible. An operation may be storage-independent
in principle even when a concrete layout type cannot represent the result; composing layouts, reporting
that the operation is unsupported for that layout, and explicit materialization are distinguishable
outcomes rather than hidden copies behind a nominally zero-copy operation.

This is currently a guideline, not a formulated trait boundary. No storage, handle, or access traits have
been designed from it yet.

Follow-up: the first grouped axis-algebra probe is now implemented. `AxisOps` is one layout trait with
per-operation associated output types rather than one trait per operation. It currently contains only
`reverse_axes` and `invert_axis`. `invert_axis` returns both the transformed layout and a signed uniform
translation of the view's base position. For strided layouts, inversion negates the axis stride and shifts
the base by `(len - 1) * old_stride`; relative offset bounds are recomputed because the position set is
preserved only after applying the translation. Unit-length axes and globally empty layouts currently treat
inversion as the identity because reversal is observationally trivial over their valid index spaces. This
avoids giving meaning to arbitrary inert strides, but it is a semantic choice worth revisiting if empty
or unit-axis metadata should transform even when no observed positions change.

The probe also tests the fluent axis selector discussed with the user: `layout.axis(axis).invert()` and
`view.axis(axis).invert()`. The layout-level `Axis<L>` value owns the target layout plus the selected axis.
It is deliberately unchecked at selection time; invalid axes make the eventual operation return `None`.
This preserves the desired chaining shape, but it means `Axis` is really an operation builder/selection,
not a proof that the axis is valid, and invalid operations consume their target just as current fallible
constructors consume invalid inputs. Whether axis validity should instead be established earlier is still
open.

`SliceView::reverse_axes` is now generic over `L: AxisOps` and returns the associated reversed layout type.
`SliceView::axis(..).invert()` applies the layout translation to the explicit origin, leaves storage
untouched, and revalidates the resulting composition before returning it. The revalidation uses only
storage extent/type metadata, not element access. It is protective because `AxisOps` is still a safe trait;
if later trusted contracts make transformation preservation a proof obligation, this validation may become
redundant for trusted implementations. The duplication between layout-level `Axis` and view-level
`SliceViewAxis` is intentional probe scaffolding, not a final API shape. A unified selector may need a more
general target algebra, or array-level selectors may remain a thin adaptation of layout algebra.

Follow-up revision, superseding the unchecked-selector description above: the axis probe now uses validated
value-level transformations. `Axis` is finite-index
evidence containing both the rank for which it is valid and the selected coordinate; invalid coordinates
cannot be constructed through its checked constructor. Axis transformations are currently the closed enum
`AxisTransform::{ReverseAxes, InvertAxis}` and are restricted to rank-preserving operations. `AxisOps` has
one required method, `apply_axis_transform`; `reverse_axes` and `invert_axis` are convenience wrappers that
forward to it. Implementors can still override those wrappers, although the initial `StridedLayout`
implementation does all specialization inside the required method.

All convenience transformations now return `TransformedLayout`, the pair of transformed layout and signed
placement translation. Reversal no longer strips or assumes a zero translation. For strided layouts reversal
does produce zero translation because permutation fixes logical zero, but translation is representation
normalization rather than part of the mathematical reindexing map, so the generic algebra should not discard
it. The intended semantic equation remains `translation + output.offset(x') = input.offset(phi(x'))` for
the transformation's logical reindexing `phi`. When transformations compose, translations add. A rank-two
property test now checks the permutation/inversion commutation law by composing reverse and inversion in
both orders and comparing both layout and accumulated translation.

`TransformedLayout` is itself an axis-transformable target but is not itself a `Layout`. This is why
`AxisOps` now extends `Ranked` rather than `Layout`: layouts are the initial transformation targets, while
the transformation envelope carries pending placement metadata during a chain. Its associated output type
still must be a layout. Applying another transformation to the envelope transforms the inner layout and
checked-adds the translations, so chains such as invert-then-reverse do not accidentally drop placement
metadata. The inverse-involution property test now exercises this composition path directly and expects the
round trip to reproduce the original layout with zero accumulated translation.

The fluent selector validates before constructing a selection: `layout.axis(index)` and `view.axis(index)`
return `None` rather than an invalid `AxisSelection`. In ordinary fallible code the intended shape is
`layout.axis(1)?.invert()?`; tests often use `.and_then(AxisSelection::invert)` because they do not return
`Option`. Selection still consumes its target before validation, so failed runtime-axis selection drops the
target. That is acceptable for the current small copyable layouts and views but is not a settled ownership
design for future owned arrays.

`SliceView` now treats reversal and inversion uniformly: apply the layout transformation, add its
translation to the explicit origin, revalidate the composition, and preserve the storage unchanged. The
revalidation uses storage metadata rather than element access. `StridedLayout` absorbs both operations
directly into shape and stride metadata, so no transformation chain or indexing wrapper is constructed for
these cases.

This is still only a probe of the one-required-method shape. Its visible costs are that reversal is
nominally fallible even though it is total for strided layouts, transformation inputs can still be
rank-mismatched when constructed outside the selector, and transformation values represent individual atomic
operations rather than a normalized composite transformation. There is no composed-fallback layout yet, and
stable Rust cannot provide one through a blanket implementation that coexists with optimized implementors.
A useful next question is whether rank-preserving transformations can normalize into bounded per-axis data
(permutation plus per-axis affine slicing/inversion state), which would matter for representation closure
and `no_alloc` composition even if construction-time rewriting itself is cheap.

Follow-up reversal, superseding the one-required-method experiment above: the `AxisTransform` enum and
required `apply_axis_transform` dispatch were removed after review. The enum did not provide a primitive
transformation from which the other operations were derived; implementors still wrote each operation inside
a dispatch. It therefore bought neither expressivity nor speed, while forcing unrelated operations into a
uniform fallible output signature and making reversal artificially awkward. The composition implementation
for `TransformedLayout` was bookkeeping glue rather than a new layout representation, so it was removed as
well.

The retained parts are the validated runtime `Axis`, the grouped but closed `AxisOps` trait, per-operation
associated output types, direct operation methods, the semantic reindexing equation, and algebraic property
tests. `reverse_axes` is infallible again in the current probe: reversal is treated as a baseline capability
of implementing the closed axis-operation trait. `invert_axis` remains fallible because a concrete strided
representation can be unable to represent the inverted signed stride even though the logical transformation
exists. Whether that asymmetry is the right contract should be tested against a non-strided layout rather
than decided abstractly.

Sequencing direction from design review: prioritize recovering ordinary CPU-array capability before GPU
access traits, and use concrete counterexamples earlier rather than generalizing from `SliceView` alone.
The next layout evidence should include at least one non-strided layout that can implement `AxisOps` and
eventually one that cannot. A hierarchical blocked/tiled layout is a promising positive example because
axis reversal permutes its metadata and inversion can negate both inner and outer strides with a uniform
translation when shapes divide evenly into blocks. Morton order or gathered/permuted offsets are possible
negative examples. Sparse formats should not be treated as merely exotic storage: their index-to-position
map generally depends on sparsity-pattern data, so they entangle layout and storage rather than testing the
storage boundary alone.

Follow-up on axis values: the public rank-carrying `Axis` type was removed. It did not actually prove that
an axis was valid for a particular target; only the consuming selector established that, while direct
operations still needed runtime validation. Axis-local operations therefore use `usize` internally, and
`axis(index)` is the validating fluent boundary. A separate validated axis token can be reconsidered if a
future consumer needs to reuse validation evidence across operations. This does not change the intended
user-facing direction for axis-local operations: `axis(usize).slice(..)`, `axis(usize).collapse(..)`, and
similar selectors rather than free-standing axis newtypes.

Blocked-layout positive probe: `BlockedLayout<N>` is the first non-strided layout in the rewrite. It uses
the restricted hierarchical formula `offset(x) = sum((x_a / block_len_a) outer_stride_a + (x_a %
block_len_a) inner_stride_a)` and requires every nonzero axis length to divide evenly into complete blocks.
That divisibility is a representation invariant even when another axis makes the overall index space empty;
zero-length axes themselves are divisible by any nonzero block length. The representation is deliberately
much simpler than the spike's windowed/permuted block layout.

The probe supports the current direct `AxisOps` shape. Reversal remains total and merely reverses shape,
block shape, inner strides, and outer strides. Inversion also remains representation-closed: it negates both
stride levels and returns one uniform translation equal to the outer-block endpoint contribution plus the
within-block endpoint contribution. Its fallibility is still arithmetic representability rather than a
structural blocked-layout failure. Rank-two property tests reuse the same permutation/inversion algebra as
strided layouts, supporting the decision to keep semantic algebraic laws without reifying a transformation
enum.

Exact complete-block geometry makes offset bounds exact: quotient and remainder digits vary independently
within each axis, so inner and outer signed stride ranges can be summed per axis. The constructor rejects
metadata whose checked bounds cannot be represented. Offsets currently iterate through the generic logical
index fallback and therefore pay division/remainder work per element. That is adequate for semantics but
should not be treated as a performance design; a specialized blocked-offset iterator is an obvious later
optimization.

`nd-core` required no implementation change for the new layout. `SliceView` construction, indexed access,
iteration, reversal, and inversion all worked through the existing `Layout`/`AxisOps` bounds and storage
validation. This is positive evidence for the layout/storage split: the storage layer only consumes offset
bounds and per-index offsets, while layout-specific hierarchy remains invisible to it.

The probe also exposes unfinished capability questions. `BlockedLayout::is_unique` is currently extremely
conservative and `forward_contiguous_len` remains the default `None`, even though some blocked layouts are
unique or contain contiguous tiles. This is consistent with the existing conservative strided uniqueness
analysis, but mutable-access work will need a clearer contract for conservative capability queries. The
complete-block restriction is also crucial to the current closure result: slicing or windowing can create
partial blocks and is unlikely to remain closed without additional window metadata or a more general block
representation. Thus this is a positive probe for reversal/inversion only, not evidence that the final
blocked layout is closed under the broader structural algebra.

Mutable borrowed-slice probe: `SliceViewMut<'a, T, L>` is now a concrete sibling of `SliceView`, not a
generalization over access traits. This was deliberate. A slice supports every access capability, so it
cannot discriminate the eventual `RefAccess`/`SliceAccess` boundary; mutable slices instead provide evidence
about lifetimes, aliasing, ownership movement, and traversal. Public access traits should still wait for a
second storage family such as paged storage.

Mutable construction now rejects nonempty layouts unless `Layout::is_unique` reports uniqueness. Empty
layouts remain constructible because they access no positions. Since uniqueness is conservative, this can
reject some mathematically unique layouts; the constructor requirement is provable uniqueness, not a claim
that the query is exact. Zero strides remain acceptable on unit axes because they cannot create aliasing
between distinct valid indices.

To make ordinary mutable arrays constructible, strided uniqueness now uses the upstream-inspired
superincreasing-stride test: sort effective axes by unsigned stride magnitude and require each stride to be
strictly greater than the sum of maximum contributions from all smaller-stride axes. This accepts ordinary
contiguous and strictly separated strided layouts, rejects zero or colliding strides, uses unsigned magnitudes
so `isize::MIN` does not overflow an absolute-value calculation, and remains conservative rather than exact.
Blocked layouts use the same idea after decomposing each axis into quotient and remainder digit axes. That
`2N` pseudo-axis analysis accepts the canonical complete-block example and still requires no allocation.
This duplication suggests a possible future layout capability for exposing weighted digit decompositions,
but it is too early to design that trait.

`SliceViewMut` currently provides checked `get`, reborrowed `get_mut`, immutable logical iteration,
forward-contiguous shared/mutable slice access, forward contiguous mutable iteration, reverse/invert axis
operations, and consuming conversion into an immutable `SliceView`. Random mutable access is deliberately
reborrowed from `&mut self`; it cannot return `&'a mut T`, because repeated calls through the same index
could otherwise produce aliasing mutable references even when the layout is unique. Backing-lifetime mutable
access must come from consuming or uniquely decomposing the view, not merely borrowing it.

No unsafe code was needed for this slice of functionality. All random access is checked, and contiguous
mutable traversal uses safe slice iterators. General non-contiguous mutable iteration remains deferred: it
would either need unsafe reference construction that trusts layout bounds and uniqueness, or pay for runtime
checking/reordering. That is now the clearest future demand-driven unsafe-boundary decision. If unsafe code
trusts `is_unique`, correctness of that method becomes memory-safety evidence and probably requires an
explicit trusted-layout contract rather than merely a safe conservative query.

The implementation duplication between immutable and mutable views is useful evidence rather than an
accident to hide immediately. Immutable slice storage can return backing-lifetime shared references because
`&'a [T]` is copyable; mutable slice storage can only reborrow shared or mutable references through `&self`
or `&mut self` unless the view is consumed. A lowest-common-denominator access trait could erase the stronger
immutable lifetime, so access traits need evidence from ownership and consuming operations as well as storage
capabilities. Structural transformations did not need access traits: moving the non-copy mutable storage
through reverse/invert worked directly, and `nd-layout` remained unaware of mutability or storage.

Storage-layer extraction: `nd-storage` now exists as an independent sibling of `nd-layout`; `nd-core`
depends on both, and neither lower crate depends on the other. Borrowed shared and mutable slice storage
moved into `nd-storage` as concrete position-based resources. They expose lengths, position access, and
contiguous ranges, but no layout validation. Layout-aware validation remains in `nd-core` because it is a
composition invariant involving storage extent, layout offset bounds, and placement. This is intended to
lock the dependency direction before designing storage traits.

The design discussion rejected a dedicated layout-plus-placement struct for now. The scalar zero position
is expected to be storage-related somehow, but its exact representation remains open. The first trait
question is therefore not necessarily access: position/extent/offset mechanics may be more primitive than
element references or contiguous slices. Access traits should be designed only after that lower boundary
is clear.
