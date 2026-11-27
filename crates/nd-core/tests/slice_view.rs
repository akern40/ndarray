use hegel::{TestCase, generators as gs};
use nd_core::SliceView;
use nd_layout::{
    IndexSpace, Layout, LogicalOffsets, Ranked, RowMajorIndices, Shape, ShapeN, StridedLayout,
};

fn small_shape(tc: &TestCase) -> ShapeN<2> {
    ShapeN::new(tc.draw(gs::arrays(gs::integers::<usize>().max_value(3))))
}

fn small_strides(tc: &TestCase) -> [isize; 2] {
    tc.draw(gs::arrays(
        gs::integers::<isize>().min_value(-3).max_value(3),
    ))
}

fn position(origin: usize, offset: isize) -> Option<usize> {
    if offset >= 0 {
        origin.checked_add(usize::try_from(offset).ok()?)
    } else {
        origin.checked_sub(offset.unsigned_abs())
    }
}

#[derive(Debug)]
struct SquaredLayout {
    shape: ShapeN<1>,
}

impl SquaredLayout {
    fn three() -> Self {
        Self {
            shape: ShapeN::new([3]),
        }
    }
}

impl IndexSpace for SquaredLayout {
    type Shape = ShapeN<1>;
    type Index = [usize; 1];
    type Indices<'a>
        = RowMajorIndices<'a, 1>
    where
        Self: 'a;

    fn shape(&self) -> &Self::Shape {
        &self.shape
    }

    fn contains(&self, index: &Self::Index) -> bool {
        self.shape.contains(index)
    }

    fn indices(&self) -> Self::Indices<'_> {
        self.shape.indices()
    }
}

impl Layout for SquaredLayout {
    type Offsets<'a>
        = LogicalOffsets<'a, SquaredLayout>
    where
        Self: 'a;

    fn offset(&self, index: &Self::Index) -> Option<isize> {
        if !self.contains(index) {
            return None;
        }
        let position = isize::try_from(index[0]).ok()?;
        position.checked_mul(position)
    }

    fn offset_unchecked(&self, index: &Self::Index) -> isize {
        index[0].cast_signed().wrapping_mul(index[0].cast_signed())
    }

    fn offset_bounds(&self) -> Option<(isize, isize)> {
        Some((0, 4))
    }

    fn is_unique(&self) -> bool {
        true
    }

    fn offsets(&self) -> Self::Offsets<'_> {
        LogicalOffsets::new(self)
    }
}

#[hegel::test]
fn valid_indices_read_expected_positions(tc: TestCase) {
    let shape = small_shape(&tc);
    tc.assume(shape.iter().all(|len| len > 0));
    let strides = small_strides(&tc);
    let len = tc.draw(gs::integers::<usize>().min_value(1).max_value(24));
    let origin = tc.draw(gs::integers::<usize>().max_value(len));
    let data: Vec<usize> = (0..len).collect();
    let layout = StridedLayout::try_new(shape, strides);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        let view = SliceView::try_new(&data, layout, origin);
        tc.assume(view.is_some());
        if let Some(view) = view {
            for index in layout.indices() {
                let offset = layout.offset(&index);
                assert!(offset.is_some());
                if let Some(offset) = offset {
                    let expected_position = position(origin, offset);
                    assert!(expected_position.is_some());
                    if let Some(expected_position) = expected_position {
                        assert_eq!(view.get(&index), data.get(expected_position));
                    }
                }
            }
        }
    }
}

#[hegel::test]
fn iter_matches_enumerated_positions(tc: TestCase) {
    let shape = small_shape(&tc);
    tc.assume(shape.iter().all(|len| len > 0));
    let strides = small_strides(&tc);
    let len = tc.draw(gs::integers::<usize>().min_value(1).max_value(24));
    let origin = tc.draw(gs::integers::<usize>().max_value(len));
    let data: Vec<usize> = (0..len).collect();
    let layout = StridedLayout::try_new(shape, strides);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        let view = SliceView::try_new(&data, layout, origin);
        tc.assume(view.is_some());
        if let Some(view) = view {
            let mut expected = Vec::new();
            for index in layout.indices() {
                let offset = layout.offset(&index);
                assert!(offset.is_some());
                if let Some(offset) = offset {
                    let expected_position = position(origin, offset);
                    assert!(expected_position.is_some());
                    if let Some(expected_position) = expected_position {
                        assert!(expected_position < data.len());
                        expected.push(data[expected_position]);
                    }
                }
            }
            let actual: Vec<usize> = view.iter().copied().collect();

            assert_eq!(actual, expected);
        }
    }
}

#[test]
fn iter_uses_row_major_logical_order() {
    let data = [10, 11, 12, 13];
    let layout = StridedLayout::try_new(ShapeN::new([2, 2]), [1, 2]);
    assert!(layout.is_some());
    if let Some(layout) = layout {
        let view = SliceView::try_new(&data, layout, 0);
        assert!(view.is_some());
        if let Some(view) = view {
            let values: Vec<usize> = view.iter().copied().collect();
            assert_eq!(values, [10, 12, 11, 13]);
        }
    }
}

#[test]
fn iter_items_can_outlive_the_view() {
    let data = [10, 20, 30];
    let layout = StridedLayout::try_new(ShapeN::new([3]), [-1]);
    assert!(layout.is_some());
    let first = if let Some(layout) = layout {
        let view = SliceView::try_new(&data, layout, 2);
        assert!(view.is_some());
        view.and_then(|view| view.iter().next())
    } else {
        None
    };

    assert_eq!(first, Some(&30));
}

#[hegel::test]
fn construction_matches_enumerated_positions(tc: TestCase) {
    let shape = small_shape(&tc);
    let strides = small_strides(&tc);
    let len = tc.draw(gs::integers::<usize>().max_value(12));
    let origin = tc.draw(gs::integers::<usize>().max_value(len + 1));
    let data: Vec<usize> = (0..len).collect();
    let layout = StridedLayout::try_new(shape, strides);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        let view = SliceView::try_new(&data, layout, origin);
        let mut indexed_positions_are_valid = true;
        for index in layout.indices() {
            let offset = layout.offset(&index);
            assert!(offset.is_some());
            if let Some(offset) = offset {
                let valid = position(origin, offset).is_some_and(|position| position < len);
                indexed_positions_are_valid &= valid;
            }
        }
        let expected = if layout.shape().size() == Some(0) {
            origin <= len
        } else {
            indexed_positions_are_valid
        };

        assert_eq!(view.is_some(), expected);
    }
}

#[test]
fn negative_stride_reads_reverse_order() {
    let data = [10, 20, 30];
    let layout = StridedLayout::try_new(ShapeN::new([3]), [-1]);
    assert!(layout.is_some());
    if let Some(layout) = layout {
        let view = SliceView::try_new(&data, layout, 2);
        assert!(view.is_some());
        if let Some(view) = view {
            assert_eq!(view.get(&[0]), Some(&30));
            assert_eq!(view.get(&[1]), Some(&20));
            assert_eq!(view.get(&[2]), Some(&10));
            assert_eq!(view.get(&[3]), None);
            let values: Vec<usize> = view.iter().copied().collect();
            assert_eq!(values, [30, 20, 10]);
        }
    }
}

#[test]
fn positive_stride_can_skip_backing_positions() {
    let data = [10, 11, 12, 13, 14];
    let layout = StridedLayout::try_new(ShapeN::new([3]), [2]);
    assert!(layout.is_some());
    if let Some(layout) = layout {
        let view = SliceView::try_new(&data, layout, 0);
        assert!(view.is_some());
        if let Some(view) = view {
            assert_eq!(view.get(&[0]), Some(&10));
            assert_eq!(view.get(&[1]), Some(&12));
            assert_eq!(view.get(&[2]), Some(&14));
        }
    }
}

#[test]
fn reversed_axes_view_iterates_transposed_values() {
    let data = [0, 1, 2, 3, 4, 5];
    let layout = StridedLayout::try_new(ShapeN::new([2, 3]), [3, 1]);
    assert!(layout.is_some());
    if let Some(layout) = layout {
        let view = SliceView::try_new(&data, layout, 0);
        assert!(view.is_some());
        assert!(SliceView::try_new(&data, layout.reverse_axes(), 0).is_some());
        if let Some(view) = view {
            let reversed_view = view.reverse_axes();
            assert_eq!(reversed_view.shape().as_array(), &[3, 2]);
            assert_eq!(reversed_view.layout().strides(), &[1, 3]);
            let values: Vec<usize> = reversed_view.iter().copied().collect();
            assert_eq!(values, [0, 3, 1, 4, 2, 5]);

            for index in reversed_view.indices() {
                let [first, second] = index;
                assert_eq!(reversed_view.get(&index), view.get(&[second, first]));
            }
        }
    }
}

#[test]
fn constructor_validates_explicit_origin() {
    let data = [10, 20, 30];
    let positive = StridedLayout::try_new(ShapeN::new([3]), [1]);
    let negative = StridedLayout::try_new(ShapeN::new([3]), [-1]);

    assert!(positive.is_some());
    assert!(negative.is_some());
    if let Some(layout) = positive {
        assert!(SliceView::try_new(&data, layout, 0).is_some());
        assert!(SliceView::try_new(&data, layout, 1).is_none());
    }
    if let Some(layout) = negative {
        assert!(SliceView::try_new(&data, layout, 2).is_some());
        assert!(SliceView::try_new(&data, layout, 1).is_none());
    }
}

#[test]
fn zero_stride_indices_may_alias_for_shared_access() {
    let data = [42];
    let layout = StridedLayout::try_new(ShapeN::new([3]), [0]);
    assert!(layout.is_some());
    if let Some(layout) = layout {
        let view = SliceView::try_new(&data, layout, 0);
        assert!(view.is_some());
        if let Some(view) = view {
            assert_eq!(view.get(&[0]), Some(&42));
            assert_eq!(view.get(&[1]), Some(&42));
            assert_eq!(view.get(&[2]), Some(&42));
            assert_eq!(view.iter().count(), 3);
            assert!(view.iter().all(|value| *value == 42));
        }
    }
}

#[test]
fn empty_layout_has_no_accessible_indices() {
    let data: [usize; 0] = [];
    let layout = StridedLayout::try_new(ShapeN::new([0, 3]), [1, 1]);
    assert!(layout.is_some());
    if let Some(layout) = layout {
        let view = SliceView::try_new(&data, layout, 0);
        assert!(view.is_some());
        if let Some(view) = view {
            assert_eq!(view.get(&[0, 0]), None);
            assert_eq!(view.iter().count(), 0);
        }
        assert!(SliceView::try_new(&data, layout, 1).is_none());
    }
}

#[test]
fn empty_layout_permits_one_past_end_origin() {
    let data = [0];
    let layout = StridedLayout::try_new(ShapeN::new([0, 3]), [1, 1]);
    assert!(layout.is_some());
    if let Some(layout) = layout {
        assert!(SliceView::try_new(&data, layout, 1).is_some());
        assert!(SliceView::try_new(&data, layout, 2).is_none());
    }
}

#[test]
fn rank_zero_view_reads_one_element() {
    let data = [7];
    let layout = StridedLayout::try_new(ShapeN::<0>::new([]), []);
    assert!(layout.is_some());
    if let Some(layout) = layout {
        let view = SliceView::try_new(&data, layout, 0);
        assert!(view.is_some());
        if let Some(view) = view {
            assert_eq!(view.get(&[]), Some(&7));
            let values: Vec<usize> = view.iter().copied().collect();
            assert_eq!(values, [7]);
        }
    }
}

#[test]
fn non_strided_layout_uses_associated_index_space() {
    let data = [10, 11, 12, 13, 14];
    let view = SliceView::try_new(&data, SquaredLayout::three(), 0);

    assert!(view.is_some());
    if let Some(view) = view {
        assert_eq!(view.ndim(), 1);
        assert_eq!(view.shape().as_array(), &[3]);
        assert!(view.contains(&[2]));
        assert!(!view.contains(&[3]));
        assert_eq!(view.indices().count(), 3);
        assert_eq!(view.get(&[0]), Some(&10));
        assert_eq!(view.get(&[1]), Some(&11));
        assert_eq!(view.get(&[2]), Some(&14));
        assert_eq!(view.get(&[3]), None);
        let values: Vec<usize> = view.iter().copied().collect();
        assert_eq!(values, [10, 11, 14]);
    }
}
