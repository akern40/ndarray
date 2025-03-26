use ndarray::prelude::*;

fn main()
{
    let a = array![ // A 3D array with shape (2, 2, 3)
        [[ 0,  2,  4],
         [ 1,  3,  5]],
        [[10, 12, 14],
         [11, 13, 15]]
    ];

    let mut sum = 0;
    for elem in a.iter() {
        sum += elem;
    }
    assert_eq!(sum, a.sum());

    let mut max = Array::from_elem((2, 3), i32::MIN);
    max.zip_mut_with(&mat, |x, a| {
        *x = (*x).max(*a);
    });
    assert_eq!(max, array![
        [10, 12, 14],
        [11, 13, 15]
    ]);
}
