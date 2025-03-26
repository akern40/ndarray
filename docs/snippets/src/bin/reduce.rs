use ndarray::prelude::*;

fn main()
{
    let a = array![[1., 2., 3., 4.]];
    assert_eq!(a.sum(), 10.);
}
