#![allow(missing_docs)] // spike
//! Spike: a strided layout whose shape and strides may be compile-time
//! constants, mdspan-style (`extents<3, dynamic_extent>` + `layout_stride`).
//!
//! Shape and strides are tuples of per-axis *extents*: `Const<N>` / `Dyn`
//! for lengths, `CStride<N>` / `DynStride` for strides. `offset` folds over
//! the tuple, so a fully-const layout compiles to constant multiplies.
//!
//! Finding: runtime-axis ops erase all axes; `reverse_axes`/`into_dyn` are the
//! only type-preserving-or-better ops; keeping per-axis const types under
//! slicing needs const-axis ops (`slice_axis0`).

use crate::layout::ranked::Ranked;
use crate::layout::strided::StridedLayout;
use crate::layout::traits::{AxisOps, Layout, Strided};
use crate::{Axis, Dim, Ix1, Ix2, Ix3, IxDyn, Slice};

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

    #[allow(clippy::wrong_self_convention)]
    fn to_strided(&self) -> StridedLayout<Ix2>
    {
        StridedLayout::new(Dim(self.dim), Dim([self.strides.0.get() as usize, self.strides.1.get() as usize]))
    }

    fn from_strided(layout: StridedLayout<Ix2>) -> CStrided<(Dyn, Dyn), (DynStride, DynStride)>
    {
        CStrided::new(
            (Dyn(layout.dim[0]), Dyn(layout.dim[1])),
            (DynStride(layout.strides[0] as isize), DynStride(layout.strides[1] as isize)),
        )
    }

    /// By-value step-slice of axis 0: because the axis is known statically,
    /// only that axis becomes dynamic in length and stride.
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

impl<E0: Extent, E1: Extent, S0: StrideExtent, S1: StrideExtent> AxisOps for CStrided<(E0, E1), (S0, S1)>
{
    type Collapsed = CStrided<(Dyn, Dyn), (S0, S1)>;
    type Sliced = CStrided<(Dyn, Dyn), (DynStride, DynStride)>;
    type Inverted = CStrided<(E0, E1), (DynStride, DynStride)>;
    type Permuted = CStrided<(Dyn, Dyn), (DynStride, DynStride)>;
    type Reversed = CStrided<(E1, E0), (S1, S0)>;
    type Merged = CStrided<(Dyn, Dyn), (DynStride, DynStride)>;
    type Smaller = StridedLayout<Ix1>;
    type Larger = StridedLayout<Ix3>;
    type Dyn = StridedLayout<IxDyn>;

    fn collapse_axis(self, axis: Axis, index: usize) -> (Self::Collapsed, isize)
    {
        let mut layout = self.to_strided();
        let offset = layout.collapse_axis_inplace(axis, index);
        (CStrided::new((Dyn(layout.dim[0]), Dyn(layout.dim[1])), self.strides), offset)
    }

    fn slice_axis(self, axis: Axis, slice: Slice) -> (Self::Sliced, isize)
    {
        let mut layout = self.to_strided();
        let offset = layout.slice_axis_inplace(axis, slice);
        (Self::from_strided(layout), offset)
    }

    fn invert_axis(self, axis: Axis) -> (Self::Inverted, isize)
    {
        let shape = self.shape;
        let mut layout = self.to_strided();
        let offset = layout.invert_axis_inplace(axis);
        (
            CStrided::new(shape, (DynStride(layout.strides[0] as isize), DynStride(layout.strides[1] as isize))),
            offset,
        )
    }

    fn swap_axes(self, a: usize, b: usize) -> Self::Permuted
    {
        Self::from_strided(self.to_strided().swap_axes(a, b))
    }

    fn permute_axes(self, axes: &Self::Index) -> Self::Permuted
    {
        Self::from_strided(self.to_strided().permute_axes(axes))
    }

    fn reverse_axes(self) -> Self::Reversed
    {
        CStrided::new((self.shape.1, self.shape.0), (self.strides.1, self.strides.0))
    }

    fn merge_axes(self, take: Axis, into: Axis) -> Result<Self::Merged, Self>
    {
        match self.to_strided().merge_axes(take, into) {
            Ok(layout) => Ok(Self::from_strided(layout)),
            Err(_) => Err(self),
        }
    }

    fn insert_axis(self, axis: Axis) -> Self::Larger
    {
        self.to_strided().insert_axis(axis)
    }

    fn remove_axis(self, axis: Axis) -> Self::Smaller
    {
        self.to_strided().remove_axis(axis)
    }

    fn into_dyn(self) -> Self::Dyn
    {
        self.to_strided().into_dyn()
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

#[cfg(test)]
mod tests
{
    use super::*;
    use crate::layout::traits::Layout;

    fn assert_same<L>(left: &L, right: &StridedLayout<L::Index>, origins: (isize, isize))
    where L: Layout
    {
        assert_eq!(origins.0, origins.1);
        assert_eq!(left.shape(), right.shape());
        let mut index = left.first_index();
        while let Some(index_value) = index {
            assert_eq!(left.offset(&index_value), right.offset(&index_value));
            index = left.next_index(index_value);
        }
    }

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
    fn ops_agree_with_runtime_layout()
    {
        let l = CStrided::new((Const::<3>, Const::<4>), (CStride::<4>, CStride::<1>));
        let r = l.to_strided();

        let (collapsed, collapsed_origin) = l.collapse_axis(Axis(1), 2);
        let (runtime_collapsed, runtime_collapsed_origin) = r.clone().collapse_axis(Axis(1), 2);
        assert_same(&collapsed, &runtime_collapsed, (collapsed_origin, runtime_collapsed_origin));

        let (sliced, sliced_origin) = l.slice_axis(Axis(0), Slice::new(0, None, 2));
        let (runtime_sliced, runtime_sliced_origin) = r.clone().slice_axis(Axis(0), Slice::new(0, None, 2));
        assert_same(&sliced, &runtime_sliced, (sliced_origin, runtime_sliced_origin));

        let (inverted, inverted_origin) = l.invert_axis(Axis(1));
        let (runtime_inverted, runtime_inverted_origin) = r.clone().invert_axis(Axis(1));
        assert_same(&inverted, &runtime_inverted, (inverted_origin, runtime_inverted_origin));

        let swapped = l.swap_axes(0, 1);
        let runtime_swapped = r.clone().swap_axes(0, 1);
        assert_same(&swapped, &runtime_swapped, (0, 0));

        let permuted = l.permute_axes(&Dim([1, 0]));
        let runtime_permuted = r.clone().permute_axes(&Dim([1, 0]));
        assert_same(&permuted, &runtime_permuted, (0, 0));

        let reversed = l.reverse_axes();
        let runtime_reversed = r.clone().reverse_axes();
        assert_same(&reversed, &runtime_reversed, (0, 0));

        let merged = l.clone().merge_axes(Axis(0), Axis(1)).unwrap();
        let runtime_merged = r.clone().merge_axes(Axis(0), Axis(1)).unwrap();
        assert_same(&merged, &runtime_merged, (0, 0));

        let (collapsed, collapsed_origin) = l.collapse_axis(Axis(1), 2);
        let (runtime_collapsed, runtime_collapsed_origin) = r.clone().collapse_axis(Axis(1), 2);
        let removed = collapsed.remove_axis(Axis(1));
        let runtime_removed = runtime_collapsed.remove_axis(Axis(1));
        assert_same(&removed, &runtime_removed, (collapsed_origin, runtime_collapsed_origin));

        let inserted = l.clone().insert_axis(Axis(1));
        let runtime_inserted = r.clone().insert_axis(Axis(1));
        assert_same(&inserted, &runtime_inserted, (0, 0));

        let dynamic = l.into_dyn();
        let runtime_dynamic = r.into_dyn();
        assert_same(&dynamic, &runtime_dynamic, (0, 0));
    }

    #[test]
    fn ops_have_the_documented_types()
    {
        type L = CStrided<(Const<3>, Const<4>), (CStride<4>, CStride<1>)>;
        let l = L::new((Const, Const), (CStride, CStride));

        let (_collapsed, _): (CStrided<(Dyn, Dyn), (CStride<4>, CStride<1>)>, isize) = l.collapse_axis(Axis(1), 2);
        let (_sliced, _): (CStrided<(Dyn, Dyn), (DynStride, DynStride)>, isize) =
            l.slice_axis(Axis(0), Slice::new(0, None, 2));
        let (_inverted, _): (CStrided<(Const<3>, Const<4>), (DynStride, DynStride)>, isize) = l.invert_axis(Axis(1));
        let _swapped: CStrided<(Dyn, Dyn), (DynStride, DynStride)> = l.swap_axes(0, 1);
        let _permuted: CStrided<(Dyn, Dyn), (DynStride, DynStride)> = l.permute_axes(&Dim([1, 0]));
        type Reversed = CStrided<(Const<4>, Const<3>), (CStride<1>, CStride<4>)>;
        let _reversed: Reversed = l.reverse_axes();
        assert_eq!(Reversed::STATIC_C_CONTIGUOUS, Some(false));
        let _merged: CStrided<(Dyn, Dyn), (DynStride, DynStride)> = l.merge_axes(Axis(0), Axis(1)).unwrap();
        let _smaller: StridedLayout<Ix1> = l.remove_axis(Axis(1));
        let _larger: StridedLayout<Ix3> = l.insert_axis(Axis(1));
        let _dynamic: StridedLayout<IxDyn> = l.into_dyn();
    }
}
