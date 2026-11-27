use std::collections::HashSet;

use hegel::{TestCase, generators as gs};
use nd_layout::{AxisOps, IndexSpace, Layout, Shape, ShapeN, StridedLayout};

fn small_shape(tc: &TestCase) -> ShapeN<2> {
    ShapeN::new(tc.draw(gs::arrays(gs::integers::<usize>().max_value(3))))
}

fn small_strides(tc: &TestCase) -> [isize; 2] {
    tc.draw(gs::arrays(
        gs::integers::<isize>().min_value(-3).max_value(3),
    ))
}

fn reversed<T, const N: usize>(mut values: [T; N]) -> [T; N] {
    values.reverse();
    values
}

#[hegel::test]
fn offset_matches_affine_formula(tc: TestCase) {
    let shape = small_shape(&tc);
    tc.assume(shape.iter().all(|len| len > 0));
    let strides = small_strides(&tc);
    let index: [usize; 2] = tc.draw(gs::arrays(gs::integers::<usize>().max_value(2)));
    let layout = StridedLayout::try_new(shape, strides);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        tc.assume(layout.contains(&index));
        let [x, y] = index;
        let [x_stride, y_stride] = strides;
        let expected = x.cast_signed() * x_stride + y.cast_signed() * y_stride;

        assert_eq!(layout.offset(&index), Some(expected));
        assert_eq!(layout.offset_unchecked(&index), expected);
    }
}

#[hegel::test]
fn invalid_indices_have_no_offset(tc: TestCase) {
    let shape = small_shape(&tc);
    let strides = small_strides(&tc);
    let index: [usize; 2] = tc.draw(gs::arrays(gs::integers::<usize>().max_value(5)));
    let layout = StridedLayout::try_new(shape, strides);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        tc.assume(!layout.contains(&index));
        assert_eq!(layout.offset(&index), None);
    }
}

#[hegel::test]
fn reverse_axes_preserves_layout_algebra(tc: TestCase) {
    let shape = small_shape(&tc);
    let strides = small_strides(&tc);
    let layout = StridedLayout::try_new(shape, strides);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        let reversed_layout = layout.reverse_axes();

        assert_eq!(
            reversed_layout.shape().as_array(),
            &reversed(shape.into_array())
        );
        assert_eq!(reversed_layout.strides(), &reversed(strides));
        assert_eq!(reversed_layout.offset_bounds(), layout.offset_bounds());
        assert_eq!(reversed_layout.is_unique(), layout.is_unique());
        assert_eq!(reversed_layout.reverse_axes(), layout);

        for index in reversed_layout.indices() {
            let original_index = reversed(index);
            assert_eq!(
                reversed_layout.offset(&index),
                layout.offset(&original_index)
            );
        }
    }
}

#[hegel::test]
fn offsets_match_checked_index_offsets(tc: TestCase) {
    let shape = small_shape(&tc);
    let strides = small_strides(&tc);
    let layout = StridedLayout::try_new(shape, strides);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        let mut expected = Vec::new();
        for index in layout.indices() {
            let Some(offset) = layout.offset(&index) else {
                unreachable!()
            };
            expected.push(offset);
        }

        let collected: Vec<isize> = layout.offsets().collect();
        assert_eq!(collected, expected);

        let folded = layout.offsets().fold(Vec::new(), |mut offsets, offset| {
            offsets.push(offset);
            offsets
        });
        assert_eq!(folded, expected);
    }
}

#[hegel::test]
fn offset_bounds_match_enumerated_offsets(tc: TestCase) {
    let shape = small_shape(&tc);
    tc.assume(shape.iter().all(|len| len > 0));
    let strides = small_strides(&tc);
    let layout = StridedLayout::try_new(shape, strides);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        let mut enumerated_bounds: Option<(isize, isize)> = None;
        for index in layout.indices() {
            let offset = layout.offset(&index);
            assert!(offset.is_some());
            if let Some(offset) = offset {
                enumerated_bounds = match enumerated_bounds {
                    Some((min, max)) => Some((min.min(offset), max.max(offset))),
                    None => Some((offset, offset)),
                };
            }
        }

        assert_eq!(layout.offset_bounds(), enumerated_bounds);
        if let Some((min, max)) = layout.offset_bounds() {
            for index in layout.indices() {
                let offset = layout.offset(&index);
                assert!(offset.is_some());
                if let Some(offset) = offset {
                    assert!(min <= offset && offset <= max);
                }
            }
        }
    }
}

#[hegel::test]
fn reported_uniqueness_implies_no_aliasing(tc: TestCase) {
    let shape = small_shape(&tc);
    let strides = small_strides(&tc);
    let layout = StridedLayout::try_new(shape, strides);

    assert!(layout.is_some());
    if let Some(layout) = layout
        && layout.is_unique()
    {
        let offsets: Vec<isize> = layout
            .indices()
            .filter_map(|index| layout.offset(&index))
            .collect();
        let unique: HashSet<isize> = offsets.iter().copied().collect();
        assert_eq!(unique.len(), offsets.len());
    }
}

#[test]
fn empty_layout_has_no_bounds_or_indices() {
    let layout = StridedLayout::try_new(ShapeN::new([0, 3]), [isize::MIN, isize::MAX]);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        assert_eq!(layout.offset_bounds(), None);
        assert_eq!(layout.indices().count(), 0);
        assert_eq!(layout.offset(&[0, 0]), None);
        assert!(layout.is_unique());
    }
}

#[test]
fn rank_zero_layout_has_zero_offset() {
    let layout = StridedLayout::try_new(ShapeN::<0>::new([]), []);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        assert_eq!(layout.offset_bounds(), Some((0, 0)));
        assert_eq!(layout.offset(&[]), Some(0));
        assert_eq!(layout.offset_unchecked(&[]), 0);
        assert!(layout.is_unique());
    }
}

#[test]
fn negative_strides_have_signed_bounds() {
    let layout = StridedLayout::try_new(ShapeN::new([3, 2]), [-2, 1]);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        assert_eq!(layout.offset_bounds(), Some((-4, 1)));
        assert_eq!(layout.offset(&[2, 1]), Some(-3));
    }
}

#[test]
fn construction_rejects_offset_overflow() {
    assert_eq!(
        StridedLayout::try_new(ShapeN::new([3]), [isize::MAX]).map(|_| ()),
        None
    );
    assert_eq!(
        StridedLayout::try_new(ShapeN::new([2, 2]), [isize::MAX, isize::MAX]).map(|_| ()),
        None
    );
    assert_eq!(
        StridedLayout::try_new(ShapeN::new([2, 2]), [isize::MIN, isize::MIN]).map(|_| ()),
        None
    );
}

#[test]
fn construction_accepts_boundary_strides() {
    let positive = StridedLayout::try_new(ShapeN::new([2]), [isize::MAX]);
    let negative = StridedLayout::try_new(ShapeN::new([2]), [isize::MIN]);

    assert!(positive.is_some());
    assert!(negative.is_some());
    if let Some(layout) = positive {
        assert_eq!(layout.offset_bounds(), Some((0, isize::MAX)));
    }
    if let Some(layout) = negative {
        assert_eq!(layout.offset_bounds(), Some((isize::MIN, 0)));
    }
}

#[hegel::test]
fn forward_contiguous_len_means_offsets_are_ordinals(tc: TestCase) {
    let shape = small_shape(&tc);
    let strides = small_strides(&tc);
    let layout = StridedLayout::try_new(shape, strides);

    assert!(layout.is_some());
    if let Some(layout) = layout
        && let Some(len) = layout.forward_contiguous_len()
    {
        assert_eq!(len, layout.indices().count());
        let expected: Vec<isize> = (0..len).map(usize::cast_signed).collect();
        assert_eq!(layout.offsets().collect::<Vec<_>>(), expected);
    }
}

#[test]
fn forward_contiguous_len_handles_inert_unit_axes() {
    let cases: [([usize; 2], [isize; 2], Option<usize>); 6] = [
        ([1, 3], [99, 1], Some(3)),
        ([2, 1], [1, 99], Some(2)),
        ([1, 1], [99, -99], Some(1)),
        ([2, 3], [3, 1], Some(6)),
        ([2, 3], [256, 1], None),
        ([2, 3], [-3, -1], None),
    ];

    for (shape, strides, expected) in cases {
        let Some(layout) = StridedLayout::try_new(ShapeN::new(shape), strides) else {
            unreachable!()
        };
        assert_eq!(layout.forward_contiguous_len(), expected);
    }
}

#[test]
fn forward_contiguous_len_handles_empty_and_rank_zero_layouts() {
    let empty = StridedLayout::try_new(ShapeN::new([0, 3]), [5, 7]);
    assert!(empty.is_some());
    if let Some(empty) = empty {
        assert_eq!(empty.forward_contiguous_len(), Some(0));
    }

    let rank_zero = StridedLayout::try_new(ShapeN::<0>::new([]), []);
    assert!(rank_zero.is_some());
    if let Some(rank_zero) = rank_zero {
        assert_eq!(rank_zero.forward_contiguous_len(), Some(1));
    }
}

#[test]
fn offsets_fold_after_partial_iteration() {
    let cases = [
        ([2, 3], [3, 1]),
        ([2, 3], [1, 3]),
        ([2, 3], [-3, -1]),
        ([2, 3], [0, 1]),
        ([2, 3], [5, 0]),
    ];

    for (shape, strides) in cases {
        let Some(layout) = StridedLayout::try_new(ShapeN::new(shape), strides) else {
            unreachable!()
        };
        let all: Vec<isize> = layout.offsets().collect();
        for skip in 0..=all.len() {
            let mut offsets = layout.offsets();
            for _ in 0..skip {
                offsets.next();
            }
            let folded = offsets.fold(Vec::new(), |mut remaining, offset| {
                remaining.push(offset);
                remaining
            });
            let expected: Vec<isize> = all.iter().skip(skip).copied().collect();
            assert_eq!(folded, expected);
        }
    }
}

#[test]
fn rank_zero_offsets_iterate_once() {
    let layout = StridedLayout::try_new(ShapeN::<0>::new([]), []);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        assert_eq!(layout.offsets().collect::<Vec<_>>(), [0]);
        assert_eq!(layout.offsets().fold(0, |count, _| count + 1), 1);
    }
}

#[test]
fn reverse_axes_reverses_empty_layout_geometry() {
    let layout = StridedLayout::try_new(ShapeN::new([0, 3]), [5, 7]);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        let reversed_layout = layout.reverse_axes();
        assert_eq!(reversed_layout.shape().as_array(), &[3, 0]);
        assert_eq!(reversed_layout.strides(), &[7, 5]);
        assert_eq!(reversed_layout.offset_bounds(), None);
        assert_eq!(reversed_layout.indices().count(), 0);
        assert_eq!(reversed_layout.reverse_axes(), layout);
    }
}

#[test]
fn reverse_axes_is_identity_for_rank_zero() {
    let layout = StridedLayout::try_new(ShapeN::<0>::new([]), []);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        assert_eq!(layout.reverse_axes(), layout);
    }
}

#[hegel::test]
fn invert_axis_preserves_absolute_positions(tc: TestCase) {
    let shape = small_shape(&tc);
    tc.assume(shape.iter().all(|len| len > 0));
    let strides = small_strides(&tc);
    let axis = tc.draw(gs::integers::<usize>().max_value(1));
    let layout = StridedLayout::try_new(shape, strides);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        let inverted = layout.axis(axis).invert();
        assert!(inverted.is_some());
        if let Some((inverted, translation)) = inverted {
            assert_eq!(inverted.shape().as_array(), shape.as_array());

            let axis_len = shape.as_array()[axis];
            for index in inverted.indices() {
                let mut original_index = index;
                original_index[axis] = axis_len - 1 - index[axis];

                let inverted_offset = inverted.offset(&index);
                let original_offset = layout.offset(&original_index);
                assert!(inverted_offset.is_some());
                assert!(original_offset.is_some());
                if let (Some(inverted_offset), Some(original_offset)) =
                    (inverted_offset, original_offset)
                {
                    assert_eq!(translation + inverted_offset, original_offset);
                }
            }

            let twice = inverted.axis(axis).invert();
            assert!(twice.is_some());
            if let Some((twice, reverse_translation)) = twice {
                assert_eq!(twice, layout);
                assert_eq!(translation + reverse_translation, 0);
            }
        }
    }
}

#[test]
fn invert_axis_uses_fluent_axis_selection() {
    let layout = StridedLayout::try_new(ShapeN::new([2, 3]), [3, 1]);
    assert!(layout.is_some());
    if let Some(layout) = layout {
        let inverted = layout.axis(1).invert();
        assert!(inverted.is_some());
        if let Some((inverted, translation)) = inverted {
            assert_eq!(translation, 2);
            assert_eq!(inverted.strides(), &[3, -1]);
            assert_eq!(inverted.offset_bounds(), Some((-2, 3)));
            assert_eq!(inverted.offset(&[0, 0]), Some(0));
            assert_eq!(inverted.offset(&[1, 2]), Some(1));
        }
    }
}

#[test]
fn invert_axis_rejects_invalid_axes() {
    let layout = StridedLayout::try_new(ShapeN::new([2, 3]), [3, 1]);
    assert!(layout.is_some());
    if let Some(layout) = layout {
        assert!(layout.axis(2).invert().is_none());
    }
}

#[test]
fn invert_axis_is_identity_for_unit_axes() {
    let layout = StridedLayout::try_new(ShapeN::new([1, 3]), [99, 1]);
    assert!(layout.is_some());
    if let Some(layout) = layout {
        assert_eq!(layout.axis(0).invert(), Some((layout, 0)));
    }
}

#[test]
fn invert_axis_is_identity_for_empty_layouts() {
    let layout = StridedLayout::try_new(ShapeN::new([0, 3]), [isize::MIN, isize::MAX]);
    assert!(layout.is_some());
    if let Some(layout) = layout {
        assert_eq!(layout.axis(1).invert(), Some((layout, 0)));
    }
}
