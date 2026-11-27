use hegel::{TestCase, generators as gs};
use nd_core::StridedView;
use nd_layout::{IndexSpace, Layout, Shape, ShapeN, StridedLayout};

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
        let view = StridedView::try_new(&data, layout, origin);
        tc.assume(view.is_some());
        if let Some(view) = view {
            for index in layout.indices() {
                let offset = layout.offset(&index);
                assert!(offset.is_some());
                if let Some(offset) = offset {
                    let expected_position = position(origin, offset);
                    assert!(expected_position.is_some());
                    if let Some(expected_position) = expected_position {
                        assert_eq!(view.get(index), data.get(expected_position));
                    }
                }
            }
        }
    }
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
        let view = StridedView::try_new(&data, layout, origin);
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
        let view = StridedView::try_new(&data, layout, 2);
        assert!(view.is_some());
        if let Some(view) = view {
            assert_eq!(view.get([0]), Some(&30));
            assert_eq!(view.get([1]), Some(&20));
            assert_eq!(view.get([2]), Some(&10));
            assert_eq!(view.get([3]), None);
        }
    }
}

#[test]
fn positive_stride_can_skip_backing_positions() {
    let data = [10, 11, 12, 13, 14];
    let layout = StridedLayout::try_new(ShapeN::new([3]), [2]);
    assert!(layout.is_some());
    if let Some(layout) = layout {
        let view = StridedView::try_new(&data, layout, 0);
        assert!(view.is_some());
        if let Some(view) = view {
            assert_eq!(view.get([0]), Some(&10));
            assert_eq!(view.get([1]), Some(&12));
            assert_eq!(view.get([2]), Some(&14));
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
        assert!(StridedView::try_new(&data, layout, 0).is_some());
        assert!(StridedView::try_new(&data, layout, 1).is_none());
    }
    if let Some(layout) = negative {
        assert!(StridedView::try_new(&data, layout, 2).is_some());
        assert!(StridedView::try_new(&data, layout, 1).is_none());
    }
}

#[test]
fn zero_stride_indices_may_alias_for_shared_access() {
    let data = [42];
    let layout = StridedLayout::try_new(ShapeN::new([3]), [0]);
    assert!(layout.is_some());
    if let Some(layout) = layout {
        let view = StridedView::try_new(&data, layout, 0);
        assert!(view.is_some());
        if let Some(view) = view {
            assert_eq!(view.get([0]), Some(&42));
            assert_eq!(view.get([1]), Some(&42));
            assert_eq!(view.get([2]), Some(&42));
        }
    }
}

#[test]
fn empty_layout_has_no_accessible_indices() {
    let data: [usize; 0] = [];
    let layout = StridedLayout::try_new(ShapeN::new([0, 3]), [1, 1]);
    assert!(layout.is_some());
    if let Some(layout) = layout {
        let view = StridedView::try_new(&data, layout, 0);
        assert!(view.is_some());
        if let Some(view) = view {
            assert_eq!(view.get([0, 0]), None);
        }
        assert!(StridedView::try_new(&data, layout, 1).is_none());
    }
}

#[test]
fn empty_layout_permits_one_past_end_origin() {
    let data = [0];
    let layout = StridedLayout::try_new(ShapeN::new([0, 3]), [1, 1]);
    assert!(layout.is_some());
    if let Some(layout) = layout {
        assert!(StridedView::try_new(&data, layout, 1).is_some());
        assert!(StridedView::try_new(&data, layout, 2).is_none());
    }
}

#[test]
fn rank_zero_view_reads_one_element() {
    let data = [7];
    let layout = StridedLayout::try_new(ShapeN::<0>::new([]), []);
    assert!(layout.is_some());
    if let Some(layout) = layout {
        let view = StridedView::try_new(&data, layout, 0);
        assert!(view.is_some());
        if let Some(view) = view {
            assert_eq!(view.get([]), Some(&7));
        }
    }
}
