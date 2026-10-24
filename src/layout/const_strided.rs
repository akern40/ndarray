#![allow(missing_docs)] // spike
//! Spike: a strided layout whose shape and strides may be compile-time
//! constants, mdspan-style (`extents<3, dynamic_extent>` + `layout_stride`).
//!
//! Shape and strides are tuples of per-axis *extents*: `Const<N>` / `Dyn`
//! for lengths, `CStride<N>` / `DynStride` for strides. `offset` folds over
//! the tuple, so a fully-const layout compiles to constant multiplies.
//!
//! Finding: `AxisOps` takes `&mut self`, so an operation that changes which
//! axes are constant (slicing with a step, collapsing) cannot be expressed on
//! a const layout; it needs by-value methods with per-op output types. The
//! by-value `transpose`/`slice_axis0` below show what those look like.

use crate::layout::ranked::Ranked;
use crate::layout::traits::{Layout, Strided};
use crate::{Axis, Dim, Ix1, Ix2};

/// A per-axis length, static or dynamic.
pub trait Extent: Copy
{
    /// `Some(n)` if known at compile time.
    const STATIC: Option<usize>;
    fn get(self) -> usize;
}
/// A per-axis stride, static or dynamic.
pub trait StrideExtent: Copy
{
    const STATIC: Option<isize>;
    fn get(self) -> isize;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Const<const N: usize>;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dyn(pub usize);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CStride<const N: isize>;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DynStride(pub isize);

impl<const N: usize> Extent for Const<N>
{
    const STATIC: Option<usize> = Some(N);
    #[inline(always)]
    fn get(self) -> usize
    {
        N
    }
}
impl Extent for Dyn
{
    const STATIC: Option<usize> = None;
    #[inline(always)]
    fn get(self) -> usize
    {
        self.0
    }
}
impl<const N: isize> StrideExtent for CStride<N>
{
    const STATIC: Option<isize> = Some(N);
    #[inline(always)]
    fn get(self) -> isize
    {
        N
    }
}
impl StrideExtent for DynStride
{
    const STATIC: Option<isize> = None;
    #[inline(always)]
    fn get(self) -> isize
    {
        self.0
    }
}

/// A strided layout over extent tuples `Sh` (lengths) and `St` (strides).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CStrided<Sh, St>
{
    pub shape: Sh,
    pub strides: St,
    /// Materialised copies so `shape()`/`strides()` can return slices today.
    dim: [usize; 2],
    str: [isize; 2],
}

// Rank 2 only for the spike; a macro would produce ranks 0..=6.
impl<E0: Extent, E1: Extent, S0: StrideExtent, S1: StrideExtent> CStrided<(E0, E1), (S0, S1)>
{
    pub fn new(shape: (E0, E1), strides: (S0, S1)) -> Self
    {
        CStrided {
            shape,
            strides,
            dim: [shape.0.get(), shape.1.get()],
            str: [strides.0.get(), strides.1.get()],
        }
    }

    /// Whether the layout is C-contiguous *at compile time* (all extents static
    /// and strides equal to the row-major products). `None` = not decidable statically.
    pub const STATIC_C_CONTIGUOUS: Option<bool> = match (E0::STATIC, E1::STATIC, S0::STATIC, S1::STATIC) {
        (Some(_), Some(n1), Some(s0), Some(s1)) => Some(s1 == 1 && s0 == n1 as isize),
        _ => None,
    };

    /// By-value transpose: the output type is the permuted tuple.
    pub fn transpose(self) -> CStrided<(E1, E0), (S1, S0)>
    {
        CStrided::new((self.shape.1, self.shape.0), (self.strides.1, self.strides.0))
    }

    /// By-value step-slice of axis 0: that axis becomes dynamic in both
    /// length and stride; axis 1 keeps its (possibly const) types.
    #[allow(clippy::type_complexity)]
    pub fn slice_axis0(self, start: usize, end: usize, step: usize) -> (CStrided<(Dyn, E1), (DynStride, S1)>, isize)
    {
        assert!(step > 0 && start <= end && end <= self.dim[0]);
        let len = (end - start + step - 1) / step;
        let origin = start as isize * self.str[0];
        (
            CStrided::new((Dyn(len), self.shape.1), (DynStride(self.str[0] * step as isize), self.strides.1)),
            origin,
        )
    }
}

impl<E0: Extent, E1: Extent, S0: StrideExtent, S1: StrideExtent> Ranked for CStrided<(E0, E1), (S0, S1)>
{
    type NDim = <Ix2 as Ranked>::NDim;
    fn ndim(&self) -> usize
    {
        2
    }
}

impl<E0: Extent, E1: Extent, S0: StrideExtent, S1: StrideExtent> Layout for CStrided<(E0, E1), (S0, S1)>
{
    type Index = Ix2;

    fn shape(&self) -> &[usize]
    {
        &self.dim
    }
    fn len(&self) -> usize
    {
        self.shape.0.get() * self.shape.1.get()
    }
    fn len_of(&self, axis: Axis) -> usize
    {
        self.dim[axis.index()]
    }
    #[inline(always)]
    fn offset(&self, index: &Ix2) -> isize
    {
        // Const extents fold to immediates here.
        index[0] as isize * self.strides.0.get() + index[1] as isize * self.strides.1.get()
    }
    fn offset_checked(&self, index: &Ix2) -> Option<isize>
    {
        if index[0] < self.shape.0.get() && index[1] < self.shape.1.get() {
            Some(self.offset(index))
        } else {
            None
        }
    }
    fn first_index(&self) -> Option<Ix2>
    {
        if self.len() == 0 {
            None
        } else {
            Some(Dim([0, 0]))
        }
    }
    fn next_index(&self, mut index: Ix2) -> Option<Ix2>
    {
        index[1] += 1;
        if index[1] == self.shape.1.get() {
            index[1] = 0;
            index[0] += 1;
        }
        if index[0] == self.shape.0.get() {
            None
        } else {
            Some(index)
        }
    }
}

impl<E0: Extent, E1: Extent, S0: StrideExtent, S1: StrideExtent> Strided for CStrided<(E0, E1), (S0, S1)>
{
    fn strides(&self) -> &[isize]
    {
        &self.str
    }
    fn stride_of(&self, axis: Axis) -> isize
    {
        self.str[axis.index()]
    }
    #[inline(always)]
    fn is_contiguous(&self) -> bool
    {
        match Self::STATIC_C_CONTIGUOUS {
            Some(b) => b,
            None => self.str[1] == 1 && self.str[0] == self.dim[1] as isize,
        }
    }
    fn is_standard_layout(&self) -> bool
    {
        self.is_contiguous()
    }
    fn is_fortran_layout(&self) -> bool
    {
        self.str[0] == 1 && self.str[1] == self.dim[0] as isize
    }
    fn low_addr_offset(&self) -> usize
    {
        let mut off = 0isize;
        for k in 0..2 {
            if self.str[k] < 0 && self.dim[k] > 0 {
                off += (self.dim[k] as isize - 1) * self.str[k];
            }
        }
        (-off) as usize
    }
    fn has_overlap(&self) -> bool
    {
        // conservative: only the contiguous cases are known overlap-free here
        !(self.is_contiguous() || self.is_fortran_layout())
    }
}

#[allow(dead_code)]
type Row = Ix1;

#[cfg(test)]
mod tests
{
    use super::*;

    #[test]
    fn const_layout_is_statically_contiguous()
    {
        type L = CStrided<(Const<3>, Const<4>), (CStride<4>, CStride<1>)>;
        assert_eq!(L::STATIC_C_CONTIGUOUS, Some(true));
        let l = L::new((Const, Const), (CStride, CStride));
        assert_eq!(l.offset(&Dim([2, 3])), 11);
        assert_eq!(l.shape(), &[3, 4]);
        assert!(l.is_contiguous());
    }

    #[test]
    fn ops_change_the_type()
    {
        let l = CStrided::new((Const::<3>, Dyn(5)), (DynStride(5), CStride::<1>));
        assert_eq!(<CStrided<(Const<3>, Dyn), (DynStride, CStride<1>)>>::STATIC_C_CONTIGUOUS, None);
        assert!(l.is_contiguous());
        let t = l.transpose();
        assert_eq!(t.offset(&Dim([4, 2])), l.offset(&Dim([2, 4])));
        let (s, origin) = l.slice_axis0(1, 3, 2);
        assert_eq!(origin, 5);
        assert_eq!(s.shape(), &[1, 5]);
        assert_eq!(s.offset(&Dim([0, 2])) + origin, l.offset(&Dim([1, 2])));
    }
}
