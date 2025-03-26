use ndarray::prelude::*;

fn main()
{
    let a = array![[1., 2., 3.], [4., 5., 6.],];
    let col_sums = a.sum_axis(Axis(0));
    let row_sums = a.sum_axis(Axis(1));
    assert_eq!(col_sums, array![5., 7., 9.]);
    assert_eq!(row_sums, array![6., 15.]);
}
