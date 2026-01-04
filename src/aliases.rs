//! Type aliases for common array sizes
//!

use crate::dimension::Dim;
use crate::layout::{NLayout, NShape};
use crate::{ArcArray, Array, ArrayRef, ArrayView, ArrayViewMut, Ix, IxDynImpl, Layout, LayoutRef};

/// Create a zero-dimensional index
#[allow(non_snake_case)]
#[inline(always)]
pub const fn Ix0() -> Ix0
{
    Dim::new([])
}
/// Create a one-dimensional index
#[allow(non_snake_case)]
#[inline(always)]
pub const fn Ix1(i0: Ix) -> Ix1
{
    Dim::new([i0])
}
/// Create a two-dimensional index
#[allow(non_snake_case)]
#[inline(always)]
pub const fn Ix2(i0: Ix, i1: Ix) -> Ix2
{
    Dim::new([i0, i1])
}
/// Create a three-dimensional index
#[allow(non_snake_case)]
#[inline(always)]
pub const fn Ix3(i0: Ix, i1: Ix, i2: Ix) -> Ix3
{
    Dim::new([i0, i1, i2])
}
/// Create a four-dimensional index
#[allow(non_snake_case)]
#[inline(always)]
pub const fn Ix4(i0: Ix, i1: Ix, i2: Ix, i3: Ix) -> Ix4
{
    Dim::new([i0, i1, i2, i3])
}
/// Create a five-dimensional index
#[allow(non_snake_case)]
#[inline(always)]
pub const fn Ix5(i0: Ix, i1: Ix, i2: Ix, i3: Ix, i4: Ix) -> Ix5
{
    Dim::new([i0, i1, i2, i3, i4])
}
/// Create a six-dimensional index
#[allow(non_snake_case)]
#[inline(always)]
pub const fn Ix6(i0: Ix, i1: Ix, i2: Ix, i3: Ix, i4: Ix, i5: Ix) -> Ix6
{
    Dim::new([i0, i1, i2, i3, i4, i5])
}

/// Create a dynamic-dimensional index
#[allow(non_snake_case)]
#[inline(always)]
pub fn IxDyn(ix: &[Ix]) -> IxDyn
{
    Dim(ix)
}

/// zero-dimensionial
pub type Ix0 = Dim<[Ix; 0]>;
/// one-dimensional
pub type Ix1 = Dim<[Ix; 1]>;
/// two-dimensional
pub type Ix2 = Dim<[Ix; 2]>;
/// three-dimensional
pub type Ix3 = Dim<[Ix; 3]>;
/// four-dimensional
pub type Ix4 = Dim<[Ix; 4]>;
/// five-dimensional
pub type Ix5 = Dim<[Ix; 5]>;
/// six-dimensional
pub type Ix6 = Dim<[Ix; 6]>;
/// dynamic-dimensional
///
/// You can use the `IxDyn` function to create a dimension for an array with
/// dynamic number of dimensions.  (`Vec<usize>` and `&[usize]` also implement
/// `IntoDimension` to produce `IxDyn`).
///
/// ```
/// use ndarray::ArrayD;
/// use ndarray::IxDyn;
///
/// // Create a 5 × 6 × 3 × 4 array using the dynamic dimension type
/// let mut a = ArrayD::<f64>::zeros(IxDyn(&[5, 6, 3, 4]));
/// // Create a 1 × 3 × 4 array using the dynamic dimension type
/// let mut b = ArrayD::<f64>::zeros(IxDyn(&[1, 3, 4]));
///
/// // We can use broadcasting to add arrays of compatible shapes together:
/// a += &b;
///
/// // We can index into a, b using fixed size arrays:
/// a[[0, 0, 0, 0]] = 0.;
/// b[[0, 2, 3]] = a[[0, 0, 2, 3]];
/// // Note: indexing will panic at runtime if the number of indices given does
/// // not match the array.
///
/// // We can keep them in the same vector because both the arrays have
/// // the same type `Array<f64, IxDyn>` a.k.a `ArrayD<f64>`:
/// let arrays = vec![a, b];
/// ```
pub type IxDyn = Dim<IxDynImpl>;

// One-dimensional strided layout
pub type L1 = NLayout<1>;
// Two-dimensional strided layout
pub type L2 = NLayout<2>;
// Three-dimensional strided layout
pub type L3 = NLayout<3>;
// Four-dimensional strided layout
pub type L4 = NLayout<4>;
// Five-dimensional strided layout
pub type L5 = NLayout<5>;
// Six-dimensional strided layout
pub type L6 = NLayout<6>;
// Seven-dimensional strided layout
pub type L7 = NLayout<7>;
// Eight-dimensional strided layout
pub type L8 = NLayout<8>;
// Nine-dimensional strided layout
pub type L9 = NLayout<9>;
// Ten-dimensional strided layout
pub type L10 = NLayout<10>;
// Eleven-dimensional strided layout
pub type L11 = NLayout<11>;
// Twelve-dimensional strided layout
pub type L12 = NLayout<12>;

pub type Sh1 = <L1 as Layout>::Shape;
pub type Sh2 = <L2 as Layout>::Shape;
pub type Sh3 = <L3 as Layout>::Shape;
pub type Sh4 = <L4 as Layout>::Shape;
pub type Sh5 = <L5 as Layout>::Shape;
pub type Sh6 = <L6 as Layout>::Shape;
pub type Sh7 = <L7 as Layout>::Shape;
pub type Sh8 = <L8 as Layout>::Shape;
pub type Sh9 = <L9 as Layout>::Shape;
pub type Sh10 = <L10 as Layout>::Shape;
pub type Sh11 = <L11 as Layout>::Shape;
pub type Sh12 = <L12 as Layout>::Shape;

/// zero-dimensional array
#[deprecated]
pub type Array0<A> = Array<A, Ix0>;
/// one-dimensional array
pub type Array1<A> = Array<A, L1>;
/// two-dimensional array
pub type Array2<A> = Array<A, L2>;
/// three-dimensional array
pub type Array3<A> = Array<A, L3>;
/// four-dimensional array
pub type Array4<A> = Array<A, L4>;
/// five-dimensional array
pub type Array5<A> = Array<A, L5>;
/// six-dimensional array
pub type Array6<A> = Array<A, L6>;
/// dynamic-dimensional array
pub type ArrayD<A> = Array<A, IxDyn>;

/// zero-dimensional array reference
pub type ArrayRef0<A> = ArrayRef<A, Ix0>;
/// one-dimensional array reference
pub type ArrayRef1<A> = ArrayRef<A, L1>;
/// two-dimensional array reference
pub type ArrayRef2<A> = ArrayRef<A, L2>;
/// three-dimensional array reference
pub type ArrayRef3<A> = ArrayRef<A, L3>;
/// four-dimensional array reference
pub type ArrayRef4<A> = ArrayRef<A, L4>;
/// five-dimensional array reference
pub type ArrayRef5<A> = ArrayRef<A, L5>;
/// six-dimensional array reference
pub type ArrayRef6<A> = ArrayRef<A, L6>;
/// dynamic-dimensional array reference
pub type ArrayRefD<A> = ArrayRef<A, IxDyn>;

/// zero-dimensional layout reference
pub type LayoutRef0<A> = LayoutRef<A, Ix0>;
/// one-dimensional layout reference
pub type LayoutRef1<A> = LayoutRef<A, L1>;
/// two-dimensional layout reference
pub type LayoutRef2<A> = LayoutRef<A, L2>;
/// three-dimensional layout reference
pub type LayoutRef3<A> = LayoutRef<A, L3>;
/// four-dimensional layout reference
pub type LayoutRef4<A> = LayoutRef<A, L4>;
/// five-dimensional layout reference
pub type LayoutRef5<A> = LayoutRef<A, L5>;
/// six-dimensional layout reference
pub type LayoutRef6<A> = LayoutRef<A, L6>;
/// dynamic-dimensional layout reference
pub type LayoutRefD<A> = LayoutRef<A, IxDyn>;

/// zero-dimensional array view
pub type ArrayView0<'a, A> = ArrayView<'a, A, Ix0>;
/// one-dimensional array view
pub type ArrayView1<'a, A> = ArrayView<'a, A, L1>;
/// two-dimensional array view
pub type ArrayView2<'a, A> = ArrayView<'a, A, L2>;
/// three-dimensional array view
pub type ArrayView3<'a, A> = ArrayView<'a, A, L3>;
/// four-dimensional array view
pub type ArrayView4<'a, A> = ArrayView<'a, A, L4>;
/// five-dimensional array view
pub type ArrayView5<'a, A> = ArrayView<'a, A, L5>;
/// six-dimensional array view
pub type ArrayView6<'a, A> = ArrayView<'a, A, L6>;
/// dynamic-dimensional array view
pub type ArrayViewD<'a, A> = ArrayView<'a, A, IxDyn>;

/// zero-dimensional read-write array view
pub type ArrayViewMut0<'a, A> = ArrayViewMut<'a, A, Ix0>;
/// one-dimensional read-write array view
pub type ArrayViewMut1<'a, A> = ArrayViewMut<'a, A, L1>;
/// two-dimensional read-write array view
pub type ArrayViewMut2<'a, A> = ArrayViewMut<'a, A, L2>;
/// three-dimensional read-write array view
pub type ArrayViewMut3<'a, A> = ArrayViewMut<'a, A, L3>;
/// four-dimensional read-write array view
pub type ArrayViewMut4<'a, A> = ArrayViewMut<'a, A, L4>;
/// five-dimensional read-write array view
pub type ArrayViewMut5<'a, A> = ArrayViewMut<'a, A, L5>;
/// six-dimensional read-write array view
pub type ArrayViewMut6<'a, A> = ArrayViewMut<'a, A, L6>;
/// dynamic-dimensional read-write array view
pub type ArrayViewMutD<'a, A> = ArrayViewMut<'a, A, IxDyn>;

/// one-dimensional shared ownership array
pub type ArcArray1<A> = ArcArray<A, L1>;
/// two-dimensional shared ownership array
pub type ArcArray2<A> = ArcArray<A, L2>;
