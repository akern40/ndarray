use ndarray::prelude::*;
use std::f64::INFINITY as inf;

fn main()
{
    let a = array![
        [10., 20., 30., 40.],
    ];
    let b = Array::range(0., 4., 1.); // [0., 1., 2., 3.]

    // Each of these allocates a new array
    assert_eq!(&a + &b, array![[10., 21., 32., 43.]]);
    assert_eq!(&a - &b, array![[10., 19., 28., 37.]]);
    assert_eq!(&a * &b, array![[0., 20., 60., 120.,]]);

    // But if we pass in `a` instead of `&a`,
    // it will reuse `a` to avoid allocation...
    let a_location = a.as_ptr();
    let c = a / &b;
    assert_eq!(c, array![[inf, 20., 15., 13.333333333333334,]]);
    // ... so that `c` is now equal to the quotient
    assert_eq!(c, array![[inf, 20., 15., 13.333333333333334,]]);
    // ... and `c` still points to the same allocation that `a` did
    assert_eq!(c.as_ptr(), a_location);
}
