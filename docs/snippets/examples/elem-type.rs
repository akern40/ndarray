use ndarray::prelude::*;

fn main()
{
    let a = Array::<f64, _>::zeros((3, 2, 4));
    println!("{:?}", a);
}
