use hegel::{TestCase, generators as gs};
use nd_core::{SliceView, SliceViewMut, SliceViewMutAxis};
use nd_layout::{BlockedLayout, IndexSpace, Layout, ShapeN, StridedLayout};

fn small_shape(tc: &TestCase) -> ShapeN<2> {
    ShapeN::new(tc.draw(gs::arrays(
        gs::integers::<usize>().min_value(1).max_value(3),
    )))
}

fn position(origin: usize, offset: isize) -> Option<usize> {
    if offset >= 0 {
        origin.checked_add(usize::try_from(offset).ok()?)
    } else {
        origin.checked_sub(offset.unsigned_abs())
    }
}

#[test]
fn mutable_view_reads_and_writes_unique_strided_positions() {
    let mut data = [0, 1, 2, 3, 4, 5];
    let layout = StridedLayout::try_new(ShapeN::new([2, 3]), [3, 1]);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        let view = SliceViewMut::try_new(&mut data, layout, 0);
        assert!(view.is_some());
        if let Some(mut view) = view {
            assert_eq!(view.get(&[0, 0]), Some(&0));
            assert_eq!(view.get(&[1, 2]), Some(&5));
            assert_eq!(view.get(&[2, 0]), None);

            assert!(view.get_mut(&[1, 2]).is_some());
            if let Some(value) = view.get_mut(&[1, 2]) {
                *value = 25;
            }
            assert_eq!(view.get(&[1, 2]), Some(&25));
        }
    }

    assert_eq!(data, [0, 1, 2, 3, 4, 25]);
}

#[test]
fn mutable_construction_rejects_overlapping_layouts() {
    let mut data = [0, 1, 2, 3];
    let layout = StridedLayout::try_new(ShapeN::new([2, 2]), [0, 1]);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        assert!(SliceView::try_new(&data, layout, 0).is_some());
        assert!(SliceViewMut::try_new(&mut data, layout, 0).is_none());
    }
}

#[test]
fn mutable_construction_accepts_empty_overlapping_layouts() {
    let mut data = [0, 1, 2, 3];
    let layout = StridedLayout::try_new(ShapeN::new([0, 3]), [0, 0]);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        let view = SliceViewMut::try_new(&mut data, layout, 0);
        assert!(view.is_some());
        if let Some(view) = view {
            assert_eq!(view.indices().count(), 0);
            assert_eq!(view.get(&[0, 0]), None);
        }
    }
}

#[test]
fn mutable_view_supports_noncontiguous_unique_strides() {
    let mut data = [0, 1, 2, 3, 4, 5];
    let layout = StridedLayout::try_new(ShapeN::new([2, 2]), [4, 1]);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        let view = SliceViewMut::try_new(&mut data, layout, 0);
        assert!(view.is_some());
        if let Some(mut view) = view {
            assert!(view.forward_slice_mut().is_none());
            assert!(view.forward_iter_mut().is_none());
            assert_eq!(view.get(&[1, 1]), Some(&5));

            assert!(view.get_mut(&[1, 0]).is_some());
            if let Some(value) = view.get_mut(&[1, 0]) {
                *value = 40;
            }

            let values: Vec<usize> = view.iter().copied().collect();
            assert_eq!(values, [0, 1, 40, 5]);
        }
    }

    assert_eq!(data, [0, 1, 2, 3, 40, 5]);
}

#[test]
fn forward_mutable_iteration_writes_contiguous_logical_order() {
    let mut data = [0, 1, 2, 3, 4, 5];
    let layout = StridedLayout::try_new(ShapeN::new([2, 3]), [3, 1]);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        let view = SliceViewMut::try_new(&mut data, layout, 0);
        assert!(view.is_some());
        if let Some(mut view) = view {
            let iter = view.forward_iter_mut();
            assert!(iter.is_some());
            if let Some(iter) = iter {
                for (position, value) in iter.enumerate() {
                    *value += position;
                }
            }

            let slice = view.forward_slice();
            assert!(slice.is_some());
            if let Some(slice) = slice {
                assert_eq!(slice, [0, 2, 4, 6, 8, 10]);
            }
        }
    }

    assert_eq!(data, [0, 2, 4, 6, 8, 10]);
}

#[test]
fn reversed_mutable_view_writes_same_storage_positions() {
    let mut data = [0, 1, 2, 3, 4, 5];
    let layout = StridedLayout::try_new(ShapeN::new([2, 3]), [3, 1]);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        let view = SliceViewMut::try_new(&mut data, layout, 0);
        assert!(view.is_some());
        if let Some(view) = view {
            let mut reversed = view.reverse_axes();
            assert_eq!(reversed.shape().as_array(), &[3, 2]);

            assert!(reversed.get_mut(&[2, 1]).is_some());
            if let Some(value) = reversed.get_mut(&[2, 1]) {
                *value = 25;
            }
            assert_eq!(reversed.get(&[2, 1]), Some(&25));
        }
    }

    assert_eq!(data, [0, 1, 2, 3, 4, 25]);
}

#[test]
fn inverted_mutable_view_writes_same_storage_positions() {
    let mut data = [0, 1, 2, 3, 4, 5];
    let layout = StridedLayout::try_new(ShapeN::new([2, 3]), [3, 1]);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        let view = SliceViewMut::try_new(&mut data, layout, 0);
        assert!(view.is_some());
        if let Some(view) = view {
            let inverted = view.axis(1).and_then(SliceViewMutAxis::invert);
            assert!(inverted.is_some());
            if let Some(mut inverted) = inverted {
                assert_eq!(inverted.origin(), 2);
                assert_eq!(inverted.get(&[0, 0]), Some(&2));

                assert!(inverted.get_mut(&[0, 0]).is_some());
                if let Some(value) = inverted.get_mut(&[0, 0]) {
                    *value = 22;
                }
                assert_eq!(inverted.get(&[0, 0]), Some(&22));
                assert_eq!(inverted.get(&[1, 2]), Some(&3));
            }
        }
    }

    assert_eq!(data, [0, 1, 22, 3, 4, 5]);
}

#[test]
fn mutable_view_converts_into_immutable_view() {
    let mut data = [0, 1, 2, 3];
    let layout = StridedLayout::try_new(ShapeN::new([2, 2]), [2, 1]);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        let view = SliceViewMut::try_new(&mut data, layout, 0);
        assert!(view.is_some());
        if let Some(view) = view {
            let view = view.into_view();
            assert_eq!(view.get(&[1, 1]), Some(&3));
            let values: Vec<usize> = view.iter().copied().collect();
            assert_eq!(values, [0, 1, 2, 3]);
        }
    }
}

#[test]
fn mutable_view_reads_and_writes_unique_blocked_positions() {
    let mut data: Vec<usize> = (0..24).collect();
    let layout = BlockedLayout::try_new(ShapeN::new([4, 6]), ShapeN::new([2, 3]), [12, 6], [3, 1]);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        let view = SliceViewMut::try_new(&mut data, layout, 0);
        assert!(view.is_some());
        if let Some(mut view) = view {
            assert_eq!(view.get(&[3, 5]), Some(&23));
            assert!(view.forward_slice_mut().is_none());

            assert!(view.get_mut(&[2, 0]).is_some());
            if let Some(value) = view.get_mut(&[2, 0]) {
                *value = 42;
            }

            let values: Vec<usize> = view.iter().take(9).copied().collect();
            assert_eq!(values, [0, 1, 2, 6, 7, 8, 3, 4, 5]);
            assert_eq!(view.get(&[2, 0]), Some(&42));
        }
    }

    assert_eq!(data[12], 42);
}

#[test]
fn mutable_view_accepts_unique_negative_strides() {
    let mut data = [0, 1, 2, 3, 4, 5];
    let layout = StridedLayout::try_new(ShapeN::new([2, 3]), [-3, 1]);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        let view = SliceViewMut::try_new(&mut data, layout, 3);
        assert!(view.is_some());
        if let Some(mut view) = view {
            assert_eq!(view.get(&[1, 0]), Some(&0));
            assert_eq!(view.get(&[0, 0]), Some(&3));

            assert!(view.get_mut(&[1, 2]).is_some());
            if let Some(value) = view.get_mut(&[1, 2]) {
                *value = 12;
            }
            assert_eq!(view.get(&[1, 2]), Some(&12));
        }
    }

    assert_eq!(data, [0, 1, 12, 3, 4, 5]);
}

#[test]
fn mutable_view_accepts_inert_zero_strides_on_unit_axes() {
    let mut data = [0, 1, 2];
    let layout = StridedLayout::try_new(ShapeN::new([1, 3]), [0, 1]);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        let view = SliceViewMut::try_new(&mut data, layout, 0);
        assert!(view.is_some());
        if let Some(mut view) = view {
            assert!(view.get_mut(&[0, 2]).is_some());
            if let Some(value) = view.get_mut(&[0, 2]) {
                *value = 12;
            }
        }
    }

    assert_eq!(data, [0, 1, 12]);
}

#[test]
fn rank_zero_mutable_view_reads_and_writes_one_element() {
    let mut data = [7];
    let layout = StridedLayout::try_new(ShapeN::<0>::new([]), []);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        let view = SliceViewMut::try_new(&mut data, layout, 0);
        assert!(view.is_some());
        if let Some(mut view) = view {
            assert_eq!(view.get(&[]), Some(&7));
            assert!(view.get_mut(&[]).is_some());
            if let Some(value) = view.get_mut(&[]) {
                *value = 9;
            }
        }
    }

    assert_eq!(data, [9]);
}

#[hegel::test]
fn mutable_writes_hit_layout_positions(tc: TestCase) {
    let shape = small_shape(&tc);
    let first_magnitude = tc.draw(gs::integers::<usize>().min_value(1).max_value(3));
    let second_gap = tc.draw(gs::integers::<usize>().min_value(1).max_value(3));
    let second_magnitude = 2 * first_magnitude + second_gap;
    let first_negative = tc.draw(gs::integers::<usize>().max_value(1)) == 0;
    let second_negative = tc.draw(gs::integers::<usize>().max_value(1)) == 0;
    let signed = |magnitude: usize, negative: bool| {
        let magnitude = magnitude.cast_signed();
        if negative { -magnitude } else { magnitude }
    };
    let strides = [
        signed(first_magnitude, first_negative),
        signed(second_magnitude, second_negative),
    ];
    let len = 64;
    let origin = 32;
    let mut data: Vec<usize> = (0..len).collect();
    let mut expected = data.clone();
    let layout = StridedLayout::try_new(shape, strides);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        let view = SliceViewMut::try_new(&mut data, layout, origin);
        assert!(view.is_some());
        if let Some(mut view) = view {
            let indices: Vec<[usize; 2]> = view.indices().collect();
            for (ordinal, index) in indices.into_iter().enumerate() {
                let value = 1000 + ordinal;
                assert!(view.get_mut(&index).is_some());
                if let Some(element) = view.get_mut(&index) {
                    *element = value;
                }

                let Some(offset) = layout.offset(&index) else {
                    unreachable!()
                };
                let Some(position) = position(origin, offset) else {
                    unreachable!()
                };
                assert!(position < expected.len());
                if position < expected.len() {
                    expected[position] = value;
                }
            }
            assert_eq!(data, expected);
        }
    }
}

#[test]
fn reversed_blocked_mutable_view_writes_same_storage_positions() {
    let mut data: Vec<usize> = (0..24).collect();
    let layout = BlockedLayout::try_new(ShapeN::new([4, 6]), ShapeN::new([2, 3]), [12, 6], [3, 1]);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        let view = SliceViewMut::try_new(&mut data, layout, 0);
        assert!(view.is_some());
        if let Some(view) = view {
            let mut reversed = view.reverse_axes();
            assert_eq!(reversed.shape().as_array(), &[6, 4]);

            assert!(reversed.get_mut(&[5, 3]).is_some());
            if let Some(value) = reversed.get_mut(&[5, 3]) {
                *value = 43;
            }
            assert_eq!(reversed.get(&[5, 3]), Some(&43));
        }
    }

    assert_eq!(data[23], 43);
}

#[test]
fn inverted_blocked_mutable_view_writes_same_storage_positions() {
    let mut data: Vec<usize> = (0..24).collect();
    let layout = BlockedLayout::try_new(ShapeN::new([4, 6]), ShapeN::new([2, 3]), [12, 6], [3, 1]);

    assert!(layout.is_some());
    if let Some(layout) = layout {
        let view = SliceViewMut::try_new(&mut data, layout, 0);
        assert!(view.is_some());
        if let Some(view) = view {
            let inverted = view.axis(0).and_then(SliceViewMutAxis::invert);
            assert!(inverted.is_some());
            if let Some(mut inverted) = inverted {
                assert_eq!(inverted.origin(), 15);
                assert_eq!(inverted.get(&[0, 0]), Some(&15));

                assert!(inverted.get_mut(&[0, 0]).is_some());
                if let Some(value) = inverted.get_mut(&[0, 0]) {
                    *value = 45;
                }
                assert_eq!(inverted.get(&[0, 0]), Some(&45));
            }
        }
    }

    assert_eq!(data[15], 45);
}
