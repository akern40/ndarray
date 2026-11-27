use hegel::{TestCase, generators as gs};
use nd_layout::{ConstRank, DynRank, Rank, Ranked};

struct FixedRankShape<const N: usize>;

impl<const N: usize> Ranked for FixedRankShape<N> {
    type NDim = ConstRank<N>;

    fn ndim(&self) -> usize {
        N
    }
}

struct DynamicRankShape(Vec<usize>);

impl Ranked for DynamicRankShape {
    type NDim = DynRank;

    fn ndim(&self) -> usize {
        self.0.len()
    }
}

fn assert_statically_ranked<T: Ranked>(value: &T, expected: usize) {
    assert_eq!(<T::NDim as Rank>::N, Some(expected));
    assert_eq!(value.ndim(), expected);
}

fn assert_dynamically_ranked<T: Ranked<NDim = DynRank> + ?Sized>(value: &T, expected: usize) {
    assert_eq!(<T::NDim as Rank>::N, None);
    assert_eq!(value.ndim(), expected);
}

#[test]
fn types_can_expose_static_rank() {
    assert_statically_ranked(&FixedRankShape::<0>, 0);
    assert_statically_ranked(&FixedRankShape::<1>, 1);
    assert_statically_ranked(&FixedRankShape::<2>, 2);
    assert_statically_ranked(&FixedRankShape::<6>, 6);
}

#[hegel::test]
fn types_can_expose_dynamic_rank(tc: TestCase) {
    let values = tc.draw(gs::vecs(gs::integers::<usize>()));
    let shape = DynamicRankShape(values);
    assert_dynamically_ranked(&shape, shape.0.len());
}
