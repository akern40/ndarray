use ndarray::prelude::*;

fn main()
{
    let a = array![[10., 20., 30., 40.]];
    let b = Array::range(0., 4., 1.); // b = [0., 1., 2., 3, ]
    println!("a shape {:?}", &a.shape());
    println!("b shape {:?}", &b.shape());

    let b = b.into_shape_with_order((4, 1)).unwrap(); // reshape b to shape [4, 1]
    println!("b shape after reshape {:?}", &b.shape());

    println!("{}", a.dot(&b)); // [1, 4] x [4, 1] -> [1, 1]
    println!("{}", a.t().dot(&b.t())); // [4, 1] x [1, 4] -> [4, 4]
}
