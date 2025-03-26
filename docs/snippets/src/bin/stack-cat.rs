use ndarray::prelude::*;
use ndarray::{concatenate, stack};

fn main()
{
    let a = array![1, 2, 3];
    let b = array![4, 5, 6];

    assert_eq!(stack!(Axis(0), a, b), array![
        [1, 2, 3],
        [4, 5, 6],
    ]);
    assert_eq!(stack!(Axis(1), a, b), array![
        [1, 4],
        [2, 5],
        [3, 6],
    ]);

    assert_eq!(concatenate!(Axis(0), a, b), array![1, 2, 3, 4, 5, 6]);
}
