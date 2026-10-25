//! Spike: a tiled ("blocked") layout that is *not* strided, to test whether
//! [`Layout`] and [`AxisOps`] are satisfiable without strides.
//!
//! Elements live in row-major tiles of shape `tile`; the tiles themselves are
//! stored row-major. A logical index maps to (tile number, offset within tile).
//! Slicing, transposing and inverting are kept as a per-axis window
//! (`start`, `step`) and a permutation over the physical axes, because the
//! index → offset map is not affine and cannot absorb them into an origin
//! offset the way a strided layout does.

use crate::dimension::do_slice;
use crate::layout::ranked::Ranked;
use crate::layout::traits::{AxisOps, Layout};
use crate::{Axis, Dimension, IxDyn, Slice};

/// A logical axis that is not backed by a physical axis (inserted, length 1).
const UNIT: usize = usize::MAX;

/// A tiled layout: row-major tiles of shape `tile`, stored row-major.
#[derive(Clone, Debug)]
pub struct BlockLayout<D>
{
    /// Physical (whole-array) shape and tile shape, same rank.
    phys: IxDyn,
    tile: IxDyn,
    /// Coordinate used on a physical axis that no logical axis maps to any
    /// more (after `remove_axis`).
    base: IxDyn,
    /// Per logical axis: physical axis (or `UNIT`), length, window start
    /// coordinate on the physical axis, and signed step (stored as `usize`).
    perm: D,
    dim: D,
    start: D,
    step: D,
}

impl<D: Dimension> BlockLayout<D>
{
    /// A layout of `shape`, tiled by `tile` (same rank; sides nonzero).
    pub fn new(shape: D, tile: D) -> Self
    {
        assert_eq!(shape.ndim(), tile.ndim());
        assert!(tile.slice().iter().all(|&t| t > 0), "tile sides must be nonzero");
        let mut perm = D::zeros(shape.ndim());
        for (k, p) in perm.slice_mut().iter_mut().enumerate() {
            *p = k;
        }
        let mut step = D::zeros(shape.ndim());
        step.slice_mut().iter_mut().for_each(|s| *s = 1);
        BlockLayout {
            base: IxDyn::zeros(shape.ndim()),
            phys: shape.clone().into_dyn(),
            tile: tile.into_dyn(),
            perm,
            dim: shape.clone(),
            start: D::zeros(shape.ndim()),
            step,
        }
    }

    fn step_of(&self, k: usize) -> isize
    {
        self.step[k] as isize
    }

    /// Physical coordinate of logical index `i` on logical axis `k`.
    fn coord(&self, k: usize, i: usize) -> usize
    {
        (self.start[k] as isize + i as isize * self.step_of(k)) as usize
    }

    fn blocks_along(&self, p: usize) -> usize
    {
        (self.phys[p] + self.tile[p] - 1) / self.tile[p]
    }

    /// Offset of physical coordinates `c`: tile number × tile size + offset in tile.
    fn phys_offset(&self, c: &[usize]) -> isize
    {
        let mut block = 0usize;
        let mut within = 0usize;
        for (p, &cp) in c.iter().enumerate() {
            block = block * self.blocks_along(p) + cp / self.tile[p];
            within = within * self.tile[p] + cp % self.tile[p];
        }
        (block * self.tile.size() + within) as isize
    }
}

impl<D: Dimension> Ranked for BlockLayout<D>
{
    type NDim = D::NDim;

    fn ndim(&self) -> usize
    {
        self.dim.ndim()
    }
}

impl<D: Dimension> Layout for BlockLayout<D>
{
    type Index = D;

    fn shape(&self) -> &[usize]
    {
        self.dim.slice()
    }

    fn len(&self) -> usize
    {
        self.dim.size()
    }

    fn len_of(&self, axis: Axis) -> usize
    {
        self.dim[axis.index()]
    }

    fn offset(&self, index: &D) -> isize
    {
        let mut c = self.base.clone();
        for k in 0..self.dim.ndim() {
            let p = self.perm[k];
            if p != UNIT {
                c[p] = self.coord(k, index[k]);
            }
        }
        self.phys_offset(c.slice())
    }

    fn offset_checked(&self, index: &D) -> Option<isize>
    {
        if index
            .slice()
            .iter()
            .zip(self.dim.slice())
            .all(|(&i, &n)| i < n)
        {
            Some(self.offset(index))
        } else {
            None
        }
    }

    fn first_index(&self) -> Option<D>
    {
        self.dim.first_index()
    }

    fn next_index(&self, index: D) -> Option<D>
    {
        self.dim.next_for(index)
    }
}

impl<D: Dimension> AxisOps for BlockLayout<D>
{
    type Collapsed = Self;
    type Sliced = Self;
    type Inverted = Self;
    type Permuted = Self;
    type Reversed = Self;
    type Merged = Self;
    type Smaller = BlockLayout<D::Smaller>;
    type Larger = BlockLayout<D::Larger>;
    type Dyn = BlockLayout<IxDyn>;

    // The window absorbs the move; nothing for the handle to do.
    fn collapse_axis(mut self, axis: Axis, index: usize) -> (Self::Collapsed, isize)
    {
        let a = axis.index();
        assert!(index < self.dim[a]);
        self.start[a] = self.coord(a, index);
        self.dim[a] = 1;
        (self, 0)
    }

    fn slice_axis(mut self, axis: Axis, slice: Slice) -> (Self::Sliced, isize)
    {
        let a = axis.index();
        let mut len = self.dim[a];
        let mut stride = 1usize;
        let first = do_slice(&mut len, &mut stride, slice);
        if len > 0 {
            self.start[a] = self.coord(a, first as usize);
        }
        self.step[a] = (self.step_of(a) * stride as isize) as usize;
        self.dim[a] = len;
        (self, 0)
    }

    fn invert_axis(mut self, axis: Axis) -> (Self::Inverted, isize)
    {
        let a = axis.index();
        if self.dim[a] > 0 {
            self.start[a] = self.coord(a, self.dim[a] - 1);
        }
        self.step[a] = (-self.step_of(a)) as usize;
        (self, 0)
    }

    fn swap_axes(mut self, a: usize, b: usize) -> Self::Permuted
    {
        self.perm.slice_mut().swap(a, b);
        self.dim.slice_mut().swap(a, b);
        self.start.slice_mut().swap(a, b);
        self.step.slice_mut().swap(a, b);
        self
    }

    fn reverse_axes(mut self) -> Self::Reversed
    {
        self.perm.slice_mut().reverse();
        self.dim.slice_mut().reverse();
        self.start.slice_mut().reverse();
        self.step.slice_mut().reverse();
        self
    }

    // Two logical axes are never one physical run in a tiled layout.
    fn merge_axes(self, _take: Axis, _into: Axis) -> Result<Self::Merged, Self>
    {
        Err(self)
    }

    fn insert_axis(self, axis: Axis) -> Self::Larger
    {
        let a = axis.index();
        let mut perm = self.perm.insert_axis(axis);
        perm[a] = UNIT;
        let mut dim = self.dim.insert_axis(axis);
        dim[a] = 1;
        let mut start = self.start.insert_axis(axis);
        start[a] = 0;
        let mut step = self.step.insert_axis(axis);
        step[a] = 1;
        BlockLayout {
            phys: self.phys,
            tile: self.tile,
            base: self.base,
            perm,
            dim,
            start,
            step,
        }
    }

    // The physical axis stays; its coordinate is frozen at the window start
    // (which is what `index_axis` leaves after `collapse_axis`).
    fn remove_axis(self, axis: Axis) -> Self::Smaller
    {
        assert!(self.dim.ndim() > 0);
        let a = axis.index();
        let mut base = self.base;
        if self.perm[a] != UNIT {
            base[self.perm[a]] = self.start[a];
        }
        BlockLayout {
            phys: self.phys,
            tile: self.tile,
            base,
            perm: self.perm.try_remove_axis(axis),
            dim: self.dim.try_remove_axis(axis),
            start: self.start.try_remove_axis(axis),
            step: self.step.try_remove_axis(axis),
        }
    }

    fn permute_axes(mut self, axes: &D) -> Self::Permuted
    {
        let n = self.dim.ndim();
        assert_eq!(axes.ndim(), n, "the permutation must list every axis");
        let mut perm = D::zeros(n);
        let mut dim = D::zeros(n);
        let mut start = D::zeros(n);
        let mut step = D::zeros(n);
        for (k, &src) in axes.slice().iter().enumerate() {
            perm[k] = self.perm[src];
            dim[k] = self.dim[src];
            start[k] = self.start[src];
            step[k] = self.step[src];
        }
        self.perm = perm;
        self.dim = dim;
        self.start = start;
        self.step = step;
        self
    }

    fn into_dyn(self) -> Self::Dyn
    {
        BlockLayout {
            phys: self.phys,
            tile: self.tile,
            base: self.base,
            perm: self.perm.into_dyn(),
            dim: self.dim.into_dyn(),
            start: self.start.into_dyn(),
            step: self.step.into_dyn(),
        }
    }
}

#[cfg(test)]
mod tests
{
    use super::*;
    use crate::{Ix1, Ix2};

    fn a() -> BlockLayout<Ix2>
    {
        BlockLayout::new(Ix2(4, 6), Ix2(2, 3))
    }

    #[test]
    fn offsets_are_tile_then_within()
    {
        let l = a();
        assert_eq!(l.offset(&Ix2(0, 0)), 0);
        assert_eq!(l.offset(&Ix2(1, 0)), 3);
        assert_eq!(l.offset(&Ix2(0, 3)), 6);
        assert_eq!(l.offset(&Ix2(2, 0)), 12);
        assert_eq!(l.offset(&Ix2(3, 5)), 23);
        assert_eq!(l.offset_checked(&Ix2(4, 0)), None);
    }

    #[test]
    fn iteration_covers_every_slot_once()
    {
        let l = a();
        let mut seen = vec![false; 24];
        let mut i = l.first_index();
        while let Some(ix) = i {
            let o = l.offset(&ix) as usize;
            assert!(!seen[o]);
            seen[o] = true;
            i = l.next_index(ix);
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn transpose_slice_invert_are_windows()
    {
        let l = a();
        let t = l.clone().swap_axes(0, 1);
        assert_eq!(t.shape(), &[6, 4]);
        assert_eq!(t.offset(&Ix2(5, 3)), l.offset(&Ix2(3, 5)));

        let (s, origin) = l.clone().slice_axis(Axis(1), Slice::new(1, Some(6), 2));
        assert_eq!(origin, 0);
        assert_eq!(s.shape(), &[4, 3]);
        for i in 0..4 {
            for k in 0..3 {
                assert_eq!(s.offset(&Ix2(i, k)), l.offset(&Ix2(i, 1 + 2 * k)));
            }
        }

        let (inv, origin) = s.clone().invert_axis(Axis(1));
        assert_eq!(origin, 0);
        for i in 0..4 {
            for k in 0..3 {
                assert_eq!(inv.offset(&Ix2(i, k)), s.offset(&Ix2(i, 2 - k)));
            }
        }
    }

    #[test]
    fn rank_changing_ops()
    {
        let l = a();
        let (row, origin) = l.clone().collapse_axis(Axis(0), 3);
        assert_eq!(origin, 0);
        let row: BlockLayout<Ix1> = row.remove_axis(Axis(0));
        assert_eq!(row.shape(), &[6]);
        for j in 0..6 {
            assert_eq!(row.offset(&Ix1(j)), l.offset(&Ix2(3, j)));
        }
        let up = row.insert_axis(Axis(0));
        assert_eq!(up.shape(), &[1, 6]);
        assert_eq!(up.offset(&Ix2(0, 4)), l.offset(&Ix2(3, 4)));
        let d = up.into_dyn();
        assert_eq!(d.offset(&IxDyn(&[0, 4])), l.offset(&Ix2(3, 4)));
    }
}
