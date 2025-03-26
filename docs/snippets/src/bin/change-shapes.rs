use ndarray::prelude::*;

fn main()
{
    let a = array![
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ];

    let b = a.flatten();
    assert_eq!(b, Array1::<i32>::range(1, 9, 1));

    // Consume `b` and generate `c` with new shape
    let c = b.into_shape_with_order((4, 2)).unwrap();
    assert_eq!(c, array![
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8]
    ]);
}
