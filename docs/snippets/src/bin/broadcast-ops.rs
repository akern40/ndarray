
use ndarray::array;

fn main() {
    let a = array![
        [1., 1.], 
        [1., 2.], 
        [0., 3.], 
        [0., 4.]
    ];

    let b = array![[0., 1.]];

    let c = array![
        [1., 2.], 
        [1., 3.], 
        [0., 4.], 
        [0., 5.]
    ];
    
    // We can add because the shapes are compatible even if not equal.
    // The `b` array is shape 1 × 2 but acts like a 4 × 2 array.
    assert!(c == a + b);
}