use ndarray::prelude::*;

fn main()
{
    let a = array![ // A 3D array with shape (2, 2, 3)
        [[ 0,  2,  4],
         [ 1,  3,  5]],
        [[10, 12, 14],
         [11, 13, 15]]
    ];

    let tens: ArrayView<_, _> = a.slice(s![1, .., ..]);
    assert_eq!(tens, array![
        [10, 12, 14],
        [11, 13, 15]
    ]);

    // Get every other element from each row
    let first_last = a.slice(s![.., .., ..;2]);
    assert_eq!(first_last, array![
        [[ 0,  4],
         [ 1,  5]],
        [[10, 14],
         [11, 15]]
    ]);

    // Negative indexing from the back
    let last = a.slice(s![.., .., -1]);
    assert_eq!(last, array![[4, 5], [14, 15]]);
}
