# Quickstart
This guide covers the basics of `ndarray` and common operations that users are likely to immediately need.
If you are familiar with Python's NumPy, you can also check out [`ndarray` for NumPy Users](numpy.md) after this quickstart.

## The Basics
You can create your first 2x3 floating-point multidimensional as follows: 
```rust
use ndarray::prelude::*;

fn main() {
    let a = array![
        [1., 2., 3.],
        [5., 5., 6.],
    ];
    assert_eq!(a.ndim(), 2);         // The number of dimensions (dimensionality) of `a`
    assert_eq!(a.len(), 6);          // The number of elements in `a`
    assert_eq!(a.shape(), [2, 3]);   // The shape of `a`
    assert_eq!(a.is_empty(), false); // Check if the array has any elements

    println!("{:?}", a); // Print a debug representation of `a`
}
```
which outputs
```
[[1.0, 2.0, 3.0],
 [5.0, 5.0, 6.0]], shape=[2, 3], strides=[3, 1], layout=Cc (0x5), const ndim=2
```
to the console.
We can see the shape and strides of the array, as well as a hint about the array's "layout" (row-major, column-major, etc) and the number of dimensions.

## Array Creation
We have already seen one way to create arrays: the [`array!` macro](https://docs.rs/ndarray/latest/ndarray/macro.array.html), which allows us to create array literals of up to 6 dimensions.
There are several ways to create arrays other than this literal syntax.

### Element Type and Dimensionality
One common array creation task is making an array full of `0` of a given shape.
Let's try to use the [`Array::zeros` function](https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#method.zeros) to make an array with shape `(3, 2, 4)`:
```rust
use ndarray::prelude::*;

fn main() {
    let a = Array::zeros((3, 2, 4));
    println!("{:?}", a);
}
```
Unfortunately, we hit our first compliation error with `ndarray`:
```shell
error[E0283]: type annotations needed for `ArrayBase<OwnedRepr<_>, Dim<[usize; 3]>>`
   --> src/main.rs:4:9
    |
4   |     let a = Array::zeros((3, 2, 4).f());
    |         ^   --------------------------- type must be known at this point
    ... # Additional error messages
```
Notice that we don't indicate anywhere what _type_ of element we want the our array to contain, and there is no "default" element type.
To fix this, we can provide the element type via turbofish syntax:
```rust
use ndarray::prelude::*;

fn main() {
    let a = Array::<f64, _>::zeros((3, 2, 4));
    println!("{:?}", a);
}
```
which outputs what we wanted: a `(3, 2, 4)` array of 64-bit floating point zeros:
```
[[[0.0, 0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0, 0.0]],

 [[0.0, 0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0, 0.0]],

 [[0.0, 0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0, 0.0]]], shape=[3, 2, 4], strides=[8, 4, 1], layout=Cc (0x5), const ndim=3
```

Now is a good time to point out that the `Array` type is parameterized by two things: the kind of elements it carries (usually denoted `A`) and the number of dimensions it has (usually denoted `D`), in that order.
So above, we specified the element type `f64` explicitly, but let the compiler infer the 3D dimensionality (by using `_`) from the `(3, 2, 4)` shape.
Unlike NumPy and most Python-based libraries, `ndarray` tries its best to always track the precise dimensionality of its arrays.
You'll notice that if you try replacing `Array::<f64, _>::zeros(...)` with `Array::<f64, Ix2>::zeros((3, 2, 4))` up above, you'll get a compiler error.
That's Rust's type system hard at work, trying to catch errors before runtime.

### Other Initial Values and Types
Of course, we can also initialize arrays by filling them with other values, using the [`from_elem`](https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#method.from_elem) method:
```rust
use ndarray::Array;

fn main() {
    let a = Array::from_elem((3, 2, 4), false);
    println!("{:?}", a);
}
```

### Common Constructor Functions
There are also a number of common "constructor functions" which create more complex initializations than just a single element; for example, [`linspace`](https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#method.linspace) creates a 1D array with `N` elements in "linear space" from some starting value to some stopping value (exclusive):
```rust
use ndarray::prelude::*;

fn main() {
    let a = Array::linspace(0., 5., 11);
    println!("{:?}", a);
}
```
which outputs
```
[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], shape=[11], strides=[1], layout=CFcf (0xf), const ndim=1
```

Other methods include, for example

- [`ones`](https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#method.ones) for arrays full of ones
- [`range`](https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#method.range) for controlling the step size rather than the number of steps
- [`logspace`](https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#method.logspace) for logarithmically distributed ranges
- [`eye`](https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#method.eye) for the identity matrix

## Arithmetic Operations
All of `ndarray`'s binary operators are element-wise; matrix products (next section) and other kinds of operations require specific method calls.
Generally, if an array's element type supports an operation (like `+`), then the array itself does as well:
```rust
use ndarray::prelude::*;
use std::f64::INFINITY as inf;

fn main() {
    let a = array![
        [10., 20., 30., 40.],
    ];
    let b = Array::range(0., 4., 1.); // [0., 1., 2., 3.]

    // Each of these allocates a new array
    assert_eq!(&a + &b, array![[10., 21., 32., 43.]]);
    assert_eq!(&a - &b, array![[10., 19., 28., 37.]]);
    assert_eq!(&a * &b, array![[0., 20., 60., 120.,]]);

    // But if we pass in `a` instead of `&a`,
    // it will reuse `a` to avoid allocation...
    let a_location = a.as_ptr();
    let c = a / &b;
    assert_eq!(c, array![[inf, 20., 15., 13.333333333333334,]]);
    // ... so that `c` is now equal to the quotient
    assert_eq!(c, array![[inf, 20., 15., 13.333333333333334,]]);
    // ... and `c` still points to the same allocation that `a` did
    assert_eq!(c.as_ptr(), a_location);
}
```

`ndarray` tries to be as efficient as possible, including avoiding array allocations, and will consume an owned array (`Array`) type during an operation.
The rules are as follows, for any binary operator `@`:

- `&A @ &A` produces a new `Array`
- `B @ A` consumes `B`, updates it with the result, and returns it
- `B @ &A` consumes `B`, updates it with the result, and returns it
- `C @= &A` performs an arithmetic operation in place

If you try removing all of the `&` signs in front of `a` and `b` above, you'll notice that `rustc` will tell you that `a` and `b` have been moved and are no longer available.

For more information, check out the [guide on binary operations](#explain/binary-ops.md).

### Reducing Operations
`ndarray` also has reducing operations, such as `sum`, which can "reduce" the array by summing all of its elements:
```rust
use ndarray::prelude::*;

fn main() {
    let a = array![[1., 2., 3., 4.]];
    assert_eq!(a.sum(), 10.);
}
```
These methods also tend to have variants with an `_axis` suffix, such as `sum_axis`, which take in a type parameter of `Axis` and perform the reduction across that axis:
```rust
use ndarray::prelude::*;

fn main() {
    let a = array![[1., 2., 3.], [4., 5., 6.],];
    let col_sums = a.sum_axis(Axis(0));
    let row_sums = a.sum_axis(Axis(1));
    assert_eq!(col_sums, array![5., 7., 9.]);
    assert_eq!(row_sums, array![6., 15.]);
}
```

### Matrix Product
Rust doesn't have an operator for matrix products (like Python's `@`), so they must be performed by using a method call; in the case of `ndarray`, that method is called `dot`.
```rust
use ndarray::prelude::*;

fn main() {
    let a = array![[10., 20., 30., 40.]];
    let b = Array::range(0., 4., 1.);     // b = [0., 1., 2., 3, ]
    println!("a shape {:?}", &a.shape());
    println!("b shape {:?}", &b.shape());
    
    let b = b.into_shape_with_order((4,1)).unwrap(); // reshape b to shape [4, 1]
    println!("b shape after reshape {:?}", &b.shape());
    
    println!("{}", a.dot(&b));            // [1, 4] x [4, 1] -> [1, 1] 
    println!("{}", a.t().dot(&b.t()));    // [4, 1] x [1, 4] -> [4, 4]
}
```
which outputs
```
a shape [1, 4]
b shape [4]
b shape after reshape [4, 1]
[[200]]
[[0, 10, 20, 30],
 [0, 20, 40, 60],
 [0, 30, 60, 90],
 [0, 40, 80, 120]]
```

An important note about this code: unlike [`nalgebra`](nalgebra.org), `ndarray` does not currently track the lengths of array axes in the type system.
As a result, there is no way for `ndarray` to know at compile-time whether two arrays are shape-compatible for a `dot` operation.
Commenting out the `into_shape_with_order` call will still compile, but will lead to a runtime panic due to shape mismatches.

## Indexing, Slicing, and Iteration
Like `Vec`s in Rust or NumPy arrays in Python, `ndarray`s can be indexed, sliced, and iterated over.
Let's start with indexing and slicing:
```rust
use ndarray::prelude::*;

fn main() {
    let a = array![ // A 3D array with shape (2, 2, 3)
        [[ 0,  2,  4],
         [ 1,  3,  5]],
        [[10, 12, 14],
         [11, 13, 15]]
    ];

    let tens: ArrayView::<_, _> = a.slice(s![1, .., ..]);
    assert_eq!(tens, array![
        [10, 12, 14],
        [11, 13, 15]
    ]);

    // Get every other element from each row
    let first_last = a.slice(s![.., .., ..;2]);
    assert_eq!(first_last, array![
        [[ 0,  4],
         [ 1,  5]],
        [[10, 14],
         [11, 15]]
    ]);

    // Negative indexing from the back
    let last = a.slice(s![.., .., -1]);
    assert_eq!(last, array![[4, 5], [14, 15]]);
}
```
Here we've introduced two new `ndarray` concepts: the slice macro, `s!`, and the `ArrayView` type.
We'll cover the `ArrayView` type in more detail below, but right now its enough to point out that `tens` is not a different allocation than `a`, just a "view" into `a`'s data.

The slice macro is `ndarray`'s way of enabling all of the slicing ability that you might see in NumPy: ranges, step sizes, negative indexing, and inserting new axes.
It's covered in more detail in [its API documentation](https://docs.rs/ndarray/latest/ndarray/macro.s.html).

Iteration in `ndarray` can happen either flattened over the entire array, via `.iter()`, or can happen across dimensions, e.g., via `.outer_iter()`:
```rust
use ndarray::prelude::*;

fn main() {
    let a = array![ // A 3D array with shape (2, 2, 3)
        [[ 0,  2,  4],
         [ 1,  3,  5]],
        [[10, 12, 14],
         [11, 13, 15]]
    ];

    let mut sum = 0;
    for elem in a.iter() {
        sum += elem;
    }
    assert_eq!(sum, a.sum());

    let mut max = Array::from_elem((2, 3), i32::MIN);
    for mat in a.outer_iter() {
        *max = max.maximum(mat);
    }
    assert_eq!(max, array![
        [10, 12, 13],
        [11, 13, 15]
    ]);
}
```

## Shape Manipulation
Multidimensional arrays in `ndarray` can have their shapes changed as well as their elements.
This includes reshaping, stacking, and splitting.

### Changing Shapes
The easiest way to change shapes is via methods such as `into_shape_with_order` or `flatten`:
```rust
use ndarray::prelude::*;

fn main() {
    let a = array![
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ];
    
    let b = a.flatten();
    assert_eq!(b, Array1::<i32>::range(1, 9, 1));

    // Consume `b` and generate `c` with new shape
    let c = b.into_shape_with_order((4, 2)).unwrap();
    assert_eq!(c, array![
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8]
    ]);
}
```

### Stacking and Concatenating
The `stack!` and `concatenate!` macros are helpful for stacking/concatenating arrays.
The `stack!` macro stacks arrays along a new axis, while the `concatenate!` macro concatenates arrays along an existing axis:
```rust
use ndarray::prelude::*;

fn main() {
    let a = array![1, 2, 3];
    let b = array![4, 5, 6];
    
    assert_eq!(stack![Axis(0), a, b], array![])

    println!("stack, axis 1:\n{:?}\n", stack![Axis(1), a, b]);
    println!("stack, axis 2:\n{:?}\n", stack![Axis(2), a, b]);
    println!("concatenate, axis 0:\n{:?}\n", concatenate![Axis(0), a, b]);
    println!("concatenate, axis 1:\n{:?}\n", concatenate![Axis(1), a, b]);
}
```
