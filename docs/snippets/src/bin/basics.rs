use ndarray::prelude::*;

fn main()
{
    let a = array![
        [1., 2., 3.],
        [5., 5., 6.],
    ];
    assert_eq!(a.ndim(), 2); // The number of dimensions (dimensionality) of `a`
    assert_eq!(a.len(), 6); // The number of elements in `a`
    assert_eq!(a.shape(), [2, 3]); // The shape of `a`
    assert_eq!(a.is_empty(), false); // Check if the array has any elements

    println!("{:?}", a); // Print a debug representation of `a`
}
