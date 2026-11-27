use std::collections::HashSet;

use hegel::{TestCase, generators as gs};
use nd_layout::{IndexSpace, Layout, Shape, ShapeN, StridedLayout};

fn small_shape(tc: &TestCase) -> ShapeN<2> {
    ShapeN::new(tc.draw(gs::arrays(gs::integers::<usize>().max_value(3))))
}

fn small_strides(tc: &TestCase) -> [isize; 2] {
    tc.draw(gs::arrays(
        gs::integers::<isize>().min_value(-3).max_value(3),
    ))
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
