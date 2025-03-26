# Quickstart
This guide covers the basics of `ndarray` and common operations that users are likely to immediately need.
All of the examples used here can be found as binaries on the `ndarray` GitHub under `docs/snippets`, with a few simple exceptions.
If you are familiar with Python's NumPy, you can also check out [`ndarray` for NumPy Users](numpy.md) after this quickstart.

## The Basics
You can create your first 2x3 floating-point multidimensional as follows: 
```rust
--8<-- "docs/quickstart/src/bin/basics.rs"
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
--8<-- "docs/quickstart/src/bin/elem-type.rs
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
--8<-- "docs/quickstart/src/bin/elem-type.rs
```

### Common Constructor Functions
There are also a number of common "constructor functions" which create more complex initializations than just a single element; for example, [`linspace`](https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#method.linspace) creates a 1D array with `N` elements in "linear space" from some starting value to some stopping value (exclusive):
```rust
--8<-- "docs/quickstart/src/bin/linspace.rs
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
--8<-- "docs/quickstart/src/bin/arithmetic.rs
```

`ndarray` tries to be as efficient as possible, including avoiding array allocations, and will consume an owned array (`Array`) type during an operation.
The rules are as follows, for any binary operator `@`:

- `&A @ &A` produces a new `Array`
- `B @ A` consumes `B`, updates it with the result, and returns it
- `B @ &A` consumes `B`, updates it with the result, and returns it
- `C @= &A` performs an arithmetic operation in place

If you try removing all of the `&` signs in front of `a` and `b` above, you'll notice that `rustc` will tell you that `a` and `b` have been moved and are no longer available.

For more information, check out the [guide on binary operations](explain/binary-ops.md).

### Reducing Operations
`ndarray` also has reducing operations, such as `sum`, which can "reduce" the array by summing all of its elements:
```rust
--8<-- "docs/quickstart/src/bin/reduce.rs
```
These methods also tend to have variants with an `_axis` suffix, such as `sum_axis`, which take in a type parameter of `Axis` and perform the reduction across that axis:
```rust
--8<-- "docs/quickstart/src/bin/reduce-axis.rs
```

### Matrix Product
Rust doesn't have an operator for matrix products (like Python's `@`), so they must be performed by using a method call; in the case of `ndarray`, that method is called `dot`.
```rust
--8<-- "docs/quickstart/src/bin/matmul.rs
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

An important note about this code: unlike [`nalgebra`](https://nalgebra.org), `ndarray` does not currently track the lengths of array axes in the type system.
As a result, there is no way for `ndarray` to know at compile-time whether two arrays are shape-compatible for a `dot` operation.
Commenting out the `into_shape_with_order` call will still compile, but will lead to a runtime panic due to shape mismatches.

## Indexing, Slicing, and Iteration
Like `Vec`s in Rust or NumPy arrays in Python, `ndarray`s can be indexed, sliced, and iterated over.
Let's start with indexing and slicing:
```rust
--8<-- "docs/quickstart/src/bin/indexing.rs
```
Here we've introduced two new `ndarray` concepts: the slice macro, `s!`, and the `ArrayView` type.
We'll cover the `ArrayView` type in more detail below, but right now its enough to point out that `tens` is not a different allocation than `a`, just a "view" into `a`'s data.

The slice macro is `ndarray`'s way of enabling all of the slicing ability that you might see in NumPy: ranges, step sizes, negative indexing, and inserting new axes.
It's covered in more detail in [its API documentation](https://docs.rs/ndarray/latest/ndarray/macro.s.html).

Iteration in `ndarray` can happen either flattened over the entire array, via `.iter()`, or can happen across dimensions, e.g., via `.outer_iter()`:
```rust
--8<-- "docs/quickstart/src/bin/iteration.rs
```

## Shape Manipulation
Multidimensional arrays in `ndarray` can have their shapes changed as well as their elements.
This includes reshaping, stacking, and splitting.

### Changing Shapes
The easiest way to change shapes is via methods such as `into_shape_with_order` or `flatten`:
```rust
--8<-- "docs/quickstart/src/bin/change-shapes.rs
```

### Stacking and Concatenating
The `stack!` and `concatenate!` macros are helpful for joining arrays together.
The `stack!` macro stacks arrays along a new axis, while the `concatenate!` macro concatenates arrays along an existing axis:
```rust
--8<-- "docs/quickstart/src/bin/stack-cat.rs
```

### Splitting Arrays
Of course, arrays can also be split into smaller pieces via the `split_at` method:
```rust
--8<-- "docs/quickstart/src/bin/split.rs
```
Two interesting note: first, notice that splitting does not remove axes.
This is because we cannot know at runtime whether either or both parts of a split will be "collapsible", and we therefore just keep length-one axes.

Second, we again run across the `ArrayView`s, this time via the `.view()` method.
This is because we cannot just "split" a single allocation into two allocations; instead, we can split a view into a single allocation into two views into the same allocation.
This is an excellent time to dig more deeply into this topic.

## Views and Copies
A few times in this quickstart guide, we've run across the concept of a "view": a look into an an array, perhaps from a slice or a split.
Other array libraries in other languages, like Python's NumPy, also have this concept, although it's not so explicitly dealt with.
But the concepts of array views and references are critical to extending Rust's ownership models to multidimensional arrays.

### Array Views
Views are immutable borrows of a _portion_ of an array, so when we have a view of an array, we can't mutate that array until we've released the view:
```rust
--8<-- "docs/quickstart/src/bin/views.rs
```
Mutable views are also possible, and similarly "lock" the array until the mutable view is released.

Views that cover the entire array - available through the `::view()` and `::view_mut()` functions - are essentially "shallow" copies.
They do not copy the data, but they have their own shape and stride information that can be independently altered from the original array.

### Array References
The other major array type in `ndarray` is `ArrayRef`, the array reference type.
You can read more about its relationship to the other array types in the [Array Types](explain/types.md) documentation.
Its main use is for writing functions, which is covered in the [Functions How-To Guide](how-to/functions.md).

### Deep Copies
Deep copies - copies of the underlying array data - are available via the usual `::clone()` method:
```rust
--8<-- "docs/quickstart/src/bin/clone.rs
```
Cloning an `ArrayView`, on the other hand, will not copy the underlying elements; it will act more like cloning a pointer.
Getting a "cloned" array from a view can be done via `::to_owned`.

## Broadcasting
Arrays support limited broadcasting, where arithmetic operations with array operands of different sizes can be carried out by repeating the elements of the array with fewer dimensions. 

```rust
--8<-- "docs/quickstart/src/bin/broadcast-ops.rs
```

See [.broadcast()](https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#method.broadcast) for a more detailed description.

And here is a short example of it:
```rust
--8<-- "docs/quickstart/src/bin/broadcast.rs
```

## Want to learn more?
That concludes the quickstart guide!
To learn more, check out the How-To Guides and Explainers on this website, or the [API documentation at docs.rs](https://docs.rs/ndarray/latest/ndarray).
