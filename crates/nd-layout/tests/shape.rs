use hegel::{TestCase, generators as gs};
use nd_layout::{Ranked, Shape, ShapeN};

#[hegel::test]
fn axis_len_matches_backing_array(tc: TestCase) {
    let lengths: [usize; 4] = tc.draw(gs::arrays(gs::integers::<usize>()));
    let shape = ShapeN::new(lengths);

    for (axis, expected) in lengths.iter().enumerate() {
        assert_eq!(shape.axis_len(axis), Some(*expected));
    }
    assert_eq!(shape.axis_len(4), None);
    assert_eq!(shape.axis_len(usize::MAX), None);
}

#[hegel::test]
fn iter_yields_axis_lengths_in_order(tc: TestCase) {
    let lengths: [usize; 4] = tc.draw(gs::arrays(gs::integers::<usize>()));
    let shape = ShapeN::new(lengths);
    let collected: Vec<usize> = shape.iter().collect();

    assert_eq!(collected.as_slice(), &lengths[..]);
}

#[hegel::test]
fn size_matches_checked_product(tc: TestCase) {
    let lengths: [usize; 4] = tc.draw(gs::arrays(gs::integers::<usize>()));
    let shape = ShapeN::new(lengths);
    let expected = if lengths.contains(&0) {
        Some(0)
    } else {
        lengths
            .iter()
            .try_fold(1_usize, |product, len| product.checked_mul(*len))
    };

    assert_eq!(shape.size(), expected);
}

#[hegel::test]
fn array_conversions_round_trip(tc: TestCase) {
    let lengths: [usize; 4] = tc.draw(gs::arrays(gs::integers::<usize>()));
    let shape = ShapeN::new(lengths);
    let from_array = ShapeN::<4>::from(lengths);
    let back_into_array: [usize; 4] = shape.into();

    assert_eq!(shape, from_array);
    assert_eq!(back_into_array, lengths);
    assert_eq!(shape.as_array(), &lengths);
    assert_eq!(shape.as_slice(), &lengths[..]);
}

#[test]
fn rank_zero_shape_has_one_element() {
    let shape = ShapeN::<0>::new([]);

    assert_eq!(shape.ndim(), 0);
    assert_eq!(shape.axis_len(0), None);
    assert_eq!(shape.size(), Some(1));
    assert_eq!(shape.iter().count(), 0);
}

#[test]
fn size_is_none_when_product_overflows() {
    let shape = ShapeN::new([2, usize::MAX]);

    assert_eq!(shape.size(), None);
}

#[test]
fn zero_length_axis_makes_size_zero_regardless_of_order() {
    let shape = ShapeN::new([2, usize::MAX, 0]);

    assert_eq!(shape.size(), Some(0));
}
