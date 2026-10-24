use ndarray::prelude::*;
use ndarray::SparseArray;

#[test]
fn reads_fall_back_to_fill_and_writes_store()
{
    let mut a = SparseArray::from_elem((3, 4), 0.0f64);
    assert_eq!(a.nnz(), 0);
    assert_eq!(a[[1, 2]], 0.0);
    a[[1, 2]] = 5.0;
    a[[2, 0]] = 7.0;
    assert_eq!(a.nnz(), 2);
    assert_eq!(a[[1, 2]], 5.0);
    assert_eq!(a[[2, 0]], 7.0);
    assert_eq!(a[[0, 0]], 0.0);
    assert_eq!(a.get((5, 5)), None);
}

#[test]
fn structural_ops_are_layout_only()
{
    let mut a = SparseArray::from_elem((3, 4), 0i32);
    a[[1, 2]] = 5;
    let t = a.t();
    assert_eq!(t.shape(), &[4, 3]);
    assert_eq!(t[[2, 1]], 5);
    let s = a.slice(s![1.., 2..]);
    assert_eq!(s.shape(), &[2, 2]);
    assert_eq!(s[[0, 0]], 5);
    let v = a.view();
    assert_eq!(v[[1, 2]], 5);
    assert_eq!(a.nnz(), 1);
}

#[test]
fn into_owned_densifies()
{
    let mut a = SparseArray::from_elem((2, 3), 1u8);
    a[[0, 1]] = 9;
    let dense: Array2<u8> = a.view().slice(s![.., 1..]).into_owned();
    assert_eq!(dense, array![[9, 1], [1, 1]]);
    let dense: Array2<u8> = a.into_owned();
    assert_eq!(dense, array![[1, 9, 1], [1, 1, 1]]);
}

#[test]
fn clone_is_deep()
{
    let mut a = SparseArray::from_elem((2, 2), 0i32);
    a[[1, 1]] = 3;
    let mut b = a.clone();
    b[[0, 0]] = 8;
    assert_eq!(a[[0, 0]], 0);
    assert_eq!(b[[0, 0]], 8);
    assert_eq!(b[[1, 1]], 3);
}
