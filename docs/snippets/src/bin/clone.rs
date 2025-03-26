use ndarray::array;

fn main()
{
    let mut a = array![[0, 1], [2, 3]];
    let b = a.clone();

    assert_eq!(a, b);

    a[[1, 1]] = 5;

    assert_eq!(a, array![[0, 1], [2, 5]]);
    assert_eq!(b, array![[0, 1], [2, 3]]);
}
