use ndarray::prelude::*;

fn main()
{
    let a = array![
        [1, 2, 3],
        [4, 5, 6],
    ];
    let (first_row, second_row) = a.view().split_at(Axis(0), 1);
    assert_eq!(first_row, array![[1, 2, 3]]);
    assert_eq!(second_row, array![[4, 5, 6]]);

    let (first_col, other_cols) = a.view().split_at(Axis(1), 1);
    assert_eq!(first_col, array![
        [1],
        [4],
    ]);
    assert_eq!(other_cols, array![
        [2, 3],
        [5, 6],
    ]);
}
