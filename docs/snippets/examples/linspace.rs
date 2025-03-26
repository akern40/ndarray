use ndarray::prelude::*;

fn main()
{
    let a = Array::linspace(0., 5., 11);
    println!("{:?}", a);
}
