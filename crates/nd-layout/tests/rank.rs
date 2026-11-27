use hegel::{TestCase, generators as gs};
use nd_layout::{DynRank, Rank, Ranked};

fn assert_statically_ranked<T: Ranked>(value: &T, expected: usize) {
    assert_eq!(<T::NDim as Rank>::N, Some(expected));
    assert_eq!(value.ndim(), expected);
}

fn assert_dynamically_ranked<T: Ranked<NDim = DynRank> + ?Sized>(value: &T, expected: usize) {
    assert_eq!(<T::NDim as Rank>::N, None);
    assert_eq!(value.ndim(), expected);
}

#[test]
fn arrays_are_statically_ranked() {
    assert_statically_ranked(&[0_usize; 0], 0);
    assert_statically_ranked(&[0], 1);
    assert_statically_ranked(&[0, 0], 2);
    assert_statically_ranked(&[0, 0, 0, 0, 0, 0], 6);
}

#[hegel::test]
fn slices_are_dynamically_ranked(tc: TestCase) {
    let values = tc.draw(gs::vecs(gs::integers::<usize>()));
    assert_dynamically_ranked(values.as_slice(), values.len());
}
