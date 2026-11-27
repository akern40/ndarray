use hegel::{TestCase, generators as gs};
use nd_layout::{AxisOps, AxisSelection, BlockedLayout, IndexSpace, Layout, ShapeN};

fn small_block_geometry(tc: &TestCase) -> (ShapeN<2>, ShapeN<2>) {
    let block_lengths: [usize; 2] = tc.draw(gs::arrays(
        gs::integers::<usize>().min_value(1).max_value(3),
    ));
    let block_counts: [usize; 2] = tc.draw(gs::arrays(
        gs::integers::<usize>().min_value(1).max_value(3),
    ));
    let shape = ShapeN::new([
        block_lengths[0] * block_counts[0],
        block_lengths[1] * block_counts[1],
    ]);
    (shape, ShapeN::new(block_lengths))
}

fn small_strides(tc: &TestCase) -> ([isize; 2], [isize; 2]) {
    let outer = tc.draw(gs::arrays(
        gs::integers::<isize>().min_value(-4).max_value(4),
    ));
    let inner = tc.draw(gs::arrays(
        gs::integers::<isize>().min_value(-4).max_value(4),
    ));
    (outer, inner)
}

fn reversed<T, const N: usize>(mut values: [T; N]) -> [T; N] {
    values.reverse();
    values
}

#[test]
fn construction_rejects_invalid_block_geometry() {
    let shape = ShapeN::new([4, 6]);

    assert!(BlockedLayout::try_new(shape, ShapeN::new([0, 3]), [12, 6], [3, 1]).is_none());
    assert!(BlockedLayout::try_new(shape, ShapeN::new([3, 3]), [12, 6], [3, 1]).is_none());
    assert!(
        BlockedLayout::try_new(ShapeN::new([0, 5]), ShapeN::new([4, 2]), [12, 6], [3, 1]).is_none()
    );
}

#[test]
fn construction_accepts_exact_empty_geometry() {
    let layout = BlockedLayout::try_new(ShapeN::new([0, 6]), ShapeN::new([4, 3]), [12, 6], [3, 1]);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        assert_eq!(layout.offset_bounds(), None);
        assert_eq!(layout.indices().count(), 0);
    }
}

#[test]
fn row_major_tiles_have_hierarchical_offsets() {
    let layout = BlockedLayout::try_new(ShapeN::new([4, 6]), ShapeN::new([2, 3]), [12, 6], [3, 1]);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        assert_eq!(layout.offset(&[0, 0]), Some(0));
        assert_eq!(layout.offset(&[1, 0]), Some(3));
        assert_eq!(layout.offset(&[0, 3]), Some(6));
        assert_eq!(layout.offset(&[2, 0]), Some(12));
        assert_eq!(layout.offset(&[3, 5]), Some(23));
        assert_eq!(layout.offset(&[4, 0]), None);
        assert_eq!(layout.offset_bounds(), Some((0, 23)));
    }
}

#[hegel::test]
fn offset_matches_hierarchical_formula(tc: TestCase) {
    let (shape, block_shape) = small_block_geometry(&tc);
    let (outer_strides, inner_strides) = small_strides(&tc);
    let index: [usize; 2] = tc.draw(gs::arrays(gs::integers::<usize>().max_value(8)));
    let layout = BlockedLayout::try_new(shape, block_shape, outer_strides, inner_strides);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        tc.assume(layout.contains(&index));
        let [x, y] = index;
        let [x_block, y_block] = block_shape.into_array();
        let [x_outer, y_outer] = outer_strides;
        let [x_inner, y_inner] = inner_strides;
        let expected = (x / x_block).cast_signed() * x_outer
            + (x % x_block).cast_signed() * x_inner
            + (y / y_block).cast_signed() * y_outer
            + (y % y_block).cast_signed() * y_inner;

        assert_eq!(layout.offset(&index), Some(expected));
        assert_eq!(layout.offset_unchecked(&index), expected);
    }
}

#[hegel::test]
fn invalid_indices_have_no_offset(tc: TestCase) {
    let (shape, block_shape) = small_block_geometry(&tc);
    let (outer_strides, inner_strides) = small_strides(&tc);
    let index: [usize; 2] = tc.draw(gs::arrays(gs::integers::<usize>().max_value(10)));
    let layout = BlockedLayout::try_new(shape, block_shape, outer_strides, inner_strides);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        tc.assume(!layout.contains(&index));
        assert_eq!(layout.offset(&index), None);
    }
}

#[hegel::test]
fn offset_bounds_match_enumerated_offsets(tc: TestCase) {
    let (shape, block_shape) = small_block_geometry(&tc);
    let (outer_strides, inner_strides) = small_strides(&tc);
    let layout = BlockedLayout::try_new(shape, block_shape, outer_strides, inner_strides);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        let offsets: Vec<isize> = layout
            .indices()
            .filter_map(|index| layout.offset(&index))
            .collect();
        assert_ne!(offsets.len(), 0);
        let bounds = offsets
            .iter()
            .min()
            .copied()
            .zip(offsets.iter().max().copied());
        assert_eq!(layout.offset_bounds(), bounds);
    }
}

#[hegel::test]
fn reverse_axes_preserves_hierarchical_positions(tc: TestCase) {
    let (shape, block_shape) = small_block_geometry(&tc);
    let (outer_strides, inner_strides) = small_strides(&tc);
    let layout = BlockedLayout::try_new(shape, block_shape, outer_strides, inner_strides);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        let reversed_layout = layout.reverse_axes();
        assert_eq!(
            reversed_layout.shape().as_array(),
            &reversed(shape.into_array())
        );
        assert_eq!(
            reversed_layout.block_shape().as_array(),
            &reversed(block_shape.into_array())
        );
        assert_eq!(reversed_layout.outer_strides(), &reversed(outer_strides));
        assert_eq!(reversed_layout.inner_strides(), &reversed(inner_strides));
        assert_eq!(reversed_layout.offset_bounds(), layout.offset_bounds());
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
    let (shape, block_shape) = small_block_geometry(&tc);
    let (outer_strides, inner_strides) = small_strides(&tc);
    let layout = BlockedLayout::try_new(shape, block_shape, outer_strides, inner_strides);

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
    }
}

#[hegel::test]
fn invert_axis_preserves_absolute_positions(tc: TestCase) {
    let (shape, block_shape) = small_block_geometry(&tc);
    let (outer_strides, inner_strides) = small_strides(&tc);
    let axis = tc.draw(gs::integers::<usize>().max_value(1));
    let layout = BlockedLayout::try_new(shape, block_shape, outer_strides, inner_strides);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        let inverted = layout.axis(axis).and_then(AxisSelection::invert);
        assert!(inverted.is_some());
        if let Some((inverted, translation)) = inverted {
            assert_eq!(inverted.shape().as_array(), shape.as_array());
            assert_eq!(inverted.block_shape().as_array(), block_shape.as_array());

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

            let twice = inverted.axis(axis).and_then(AxisSelection::invert);
            assert!(twice.is_some());
            if let Some((twice, reverse_translation)) = twice {
                assert_eq!(twice, layout);
                assert_eq!(translation + reverse_translation, 0);
            }
        }
    }
}

#[hegel::test]
fn reverse_and_invert_obey_permutation_algebra(tc: TestCase) {
    let (shape, block_shape) = small_block_geometry(&tc);
    let (outer_strides, inner_strides) = small_strides(&tc);
    let axis = tc.draw(gs::integers::<usize>().max_value(1));
    let layout = BlockedLayout::try_new(shape, block_shape, outer_strides, inner_strides);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        let invert_after_reverse = layout
            .reverse_axes()
            .axis(axis)
            .and_then(AxisSelection::invert);

        let invert_before_reverse = layout
            .axis(1 - axis)
            .and_then(AxisSelection::invert)
            .map(|(inverted, translation)| (inverted.reverse_axes(), translation));

        assert_eq!(invert_after_reverse, invert_before_reverse);
    }
}

#[test]
fn invert_axis_is_identity_for_unit_axes_and_empty_layouts() {
    let unit = BlockedLayout::try_new(ShapeN::new([1, 6]), ShapeN::new([1, 3]), [12, 6], [7, 1]);
    assert!(unit.is_some());
    if let Some(unit) = unit {
        let inverted = unit.axis(0).and_then(AxisSelection::invert);
        assert_eq!(inverted, Some((unit, 0)));
    }

    let empty = BlockedLayout::try_new(ShapeN::new([0, 6]), ShapeN::new([2, 3]), [12, 6], [3, 1]);
    assert!(empty.is_some());
    if let Some(empty) = empty {
        let inverted = empty.axis(1).and_then(AxisSelection::invert);
        assert_eq!(inverted, Some((empty, 0)));
    }
}

#[test]
fn reverse_axes_is_identity_for_rank_zero() {
    let layout = BlockedLayout::try_new(ShapeN::<0>::new([]), ShapeN::<0>::new([]), [], []);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        assert_eq!(layout.reverse_axes(), layout);
        assert_eq!(layout.offset(&[]), Some(0));
    }
}

#[test]
fn construction_rejects_offset_overflow() {
    assert_eq!(
        BlockedLayout::try_new(ShapeN::new([3]), ShapeN::new([1]), [isize::MAX], [0],).map(|_| ()),
        None
    );
    assert_eq!(
        BlockedLayout::try_new(
            ShapeN::new([2, 2]),
            ShapeN::new([1, 1]),
            [isize::MAX, isize::MAX],
            [0, 0],
        )
        .map(|_| ()),
        None
    );
    assert_eq!(
        BlockedLayout::try_new(
            ShapeN::new([2, 2]),
            ShapeN::new([1, 1]),
            [isize::MIN, isize::MIN],
            [0, 0],
        )
        .map(|_| ()),
        None
    );
}

#[test]
fn construction_accepts_boundary_strides() {
    let positive = BlockedLayout::try_new(
        ShapeN::new([2]),
        ShapeN::new([1]),
        [isize::MAX],
        [isize::MIN],
    );
    let negative = BlockedLayout::try_new(
        ShapeN::new([2]),
        ShapeN::new([2]),
        [isize::MAX],
        [isize::MIN],
    );

    assert!(positive.is_some());
    assert!(negative.is_some());
    if let Some(layout) = positive {
        assert_eq!(layout.offset_bounds(), Some((0, isize::MAX)));
        assert_eq!(layout.offset(&[1]), Some(isize::MAX));
    }
    if let Some(layout) = negative {
        assert_eq!(layout.offset_bounds(), Some((isize::MIN, 0)));
        assert_eq!(layout.offset(&[1]), Some(isize::MIN));
    }
}

#[test]
fn inversion_rejects_unrepresentable_stride_negation() {
    let layout = BlockedLayout::try_new(ShapeN::new([2]), ShapeN::new([1]), [isize::MIN], [0]);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        assert_eq!(layout.offset_bounds(), Some((isize::MIN, 0)));
        assert!(layout.axis(0).and_then(AxisSelection::invert).is_none());
    }
}
