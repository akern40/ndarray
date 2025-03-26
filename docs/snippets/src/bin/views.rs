use ndarray::{array, Axis};

fn main()
{
    let mut a = array![[0, 1, 2, 3], [4, 5, 6, 7]];

    {
        // Get views of `a`
        let (s1, s2) = a.view().split_at(Axis(1), 2);

        // With s as a view sharing the ref of a, we cannot update a here
        //
        // a[[1, 1]] = 8;
        //
        // error[E0502]: cannot borrow `a` as mutable because it is also borrowed as immutable
        // --> src/main.rs:11:9
        //    |
        // 8  |         let (s1, s2) = a.view().split_at(Axis(1), 2);
        //    |                        - immutable borrow occurs here
        // ...
        // 12 |         a[[1, 1]] = 8;
        //    |         ^ mutable borrow occurs here
        // 25 |         assert_eq!(s1, array![[0, 1], [4, 5]]);
        //    |         -------------------------------------- immutable borrow later used here

        assert_eq!(s1, array![[0, 1], [4, 5]]);
        assert_eq!(s2, array![[2, 3], [6, 7]]);
    }

    // Now we can update a again here, as views of s1, s2 are dropped already
    a[[1, 1]] = 8;

    let (s1, s2) = a.view().split_at(Axis(1), 2);
    assert_eq!(s1, array![[0, 1], [4, 8]]);
    assert_eq!(s2, array![[2, 3], [6, 7]]);
}
