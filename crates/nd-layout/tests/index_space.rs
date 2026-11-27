use std::collections::HashSet;

use hegel::{TestCase, generators as gs};
use nd_layout::{IndexSpace, Shape, ShapeN};

fn small_lengths(tc: &TestCase) -> [usize; 3] {
    tc.draw(gs::arrays(gs::integers::<usize>().max_value(3)))
}

#[hegel::test]
fn contains_matches_pointwise_bounds(tc: TestCase) {
    let lengths = small_lengths(&tc);
    let index: [usize; 3] = tc.draw(gs::arrays(gs::integers::<usize>().max_value(5)));
    let shape = ShapeN::new(lengths);
    let expected = index
        .iter()
        .zip(lengths.iter())
        .all(|(position, len)| *position < *len);

    assert_eq!(shape.contains(&index), expected);
}

#[hegel::test]
fn indices_are_unique_valid_and_count_matches_size(tc: TestCase) {
    let shape = ShapeN::new(small_lengths(&tc));
    let indices: Vec<[usize; 3]> = shape.indices().collect();
    let unique: HashSet<[usize; 3]> = indices.iter().copied().collect();

    assert_eq!(unique.len(), indices.len());
    assert!(indices.iter().all(|index| shape.contains(index)));
    assert_eq!(shape.size(), Some(indices.len()));
}

#[hegel::test]
fn indices_use_row_major_logical_order(tc: TestCase) {
    let lengths = small_lengths(&tc);
    let shape = ShapeN::new(lengths);
    let [x_len, y_len, z_len] = lengths;
    let mut expected = Vec::new();

    for x in 0..x_len {
        for y in 0..y_len {
            for z in 0..z_len {
                expected.push([x, y, z]);
            }
        }
    }

    let actual: Vec<[usize; 3]> = shape.indices().collect();
    assert_eq!(actual, expected);
}

#[test]
fn zero_length_axis_produces_no_indices() {
    for shape in [ShapeN::new([0, 2]), ShapeN::new([2, 0])] {
        assert_eq!(shape.indices().count(), 0);
    }
}

#[test]
fn rank_zero_shape_produces_one_empty_index() {
    let shape = ShapeN::<0>::new([]);
    let indices: Vec<[usize; 0]> = shape.indices().collect();

    assert!(shape.contains(&[]));
    assert_eq!(indices.len(), 1);
    assert!(indices.contains(&[]));
}
