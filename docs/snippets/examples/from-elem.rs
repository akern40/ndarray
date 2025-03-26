use ndarray::Array;

fn main()
{
    let a = Array::from_elem((3, 2, 4), false);
    println!("{:?}", a);
}
