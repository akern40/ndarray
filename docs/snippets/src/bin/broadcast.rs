use ndarray::prelude::*;

fn main()
{
    let a = array![
        [1., 2.],
        [3., 4.],
    ];

    let b = a.broadcast((3, 2, 2)).unwrap();
    assert_eq!(b, array![
        [
            [1, 2],
            [3, 4],
        ],
        [
            [1, 2],
            [3, 4],
        ],
        [
            [1, 2],
            [3, 4],
        ]
    ])
}
