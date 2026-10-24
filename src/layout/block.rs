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
        for p in 0..c.len() {
            block = block * self.blocks_along(p) + c[p] / self.tile[p];
            within = within * self.tile[p] + c[p] % self.tile[p];
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
    type Smaller = BlockLayout<D::Smaller>;
    type Larger = BlockLayout<D::Larger>;
    type Dyn = BlockLayout<IxDyn>;

    // The window absorbs the move; nothing for the handle to do.
    fn collapse_axis(&mut self, axis: Axis, index: usize) -> isize
    {
        let a = axis.index();
        assert!(index < self.dim[a]);
        self.start[a] = self.coord(a, index);
        self.dim[a] = 1;
        0
    }

    fn slice_axis(&mut self, axis: Axis, slice: Slice) -> isize
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
        0
    }

    fn invert_axis(&mut self, axis: Axis) -> isize
    {
        let a = axis.index();
        if self.dim[a] > 0 {
            self.start[a] = self.coord(a, self.dim[a] - 1);
        }
        self.step[a] = (-self.step_of(a)) as usize;
        0
    }

    fn swap_axes(&mut self, a: usize, b: usize)
    {
        self.perm.slice_mut().swap(a, b);
        self.dim.slice_mut().swap(a, b);
        self.start.slice_mut().swap(a, b);
        self.step.slice_mut().swap(a, b);
    }

    fn reverse_axes(&mut self)
    {
        self.perm.slice_mut().reverse();
        self.dim.slice_mut().reverse();
        self.start.slice_mut().reverse();
        self.step.slice_mut().reverse();
    }

    // Two logical axes are never one physical run in a tiled layout.
    fn merge_axes(&mut self, _take: Axis, _into: Axis) -> bool
    {
        false
    }

    fn insert_axis(&self, axis: Axis) -> Self::Larger
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
            phys: self.phys.clone(),
            tile: self.tile.clone(),
            base: self.base.clone(),
            perm,
            dim,
            start,
            step,
        }
    }

    // The physical axis stays; its coordinate is frozen at the window start
    // (which is what `index_axis` leaves after `collapse_axis`).
    fn remove_axis(&self, axis: Axis) -> Self::Smaller
    {
        assert!(self.dim.ndim() > 0);
        let a = axis.index();
        let mut base = self.base.clone();
        if self.perm[a] != UNIT {
            base[self.perm[a]] = self.start[a];
        }
        BlockLayout {
            phys: self.phys.clone(),
            tile: self.tile.clone(),
            base,
            perm: self.perm.try_remove_axis(axis),
            dim: self.dim.try_remove_axis(axis),
            start: self.start.try_remove_axis(axis),
            step: self.step.try_remove_axis(axis),
        }
    }

    fn permute_axes(&self, axes: &D) -> Self
    {
        let mut out = self.clone();
        for (k, &src) in axes.slice().iter().enumerate() {
            out.perm[k] = self.perm[src];
            out.dim[k] = self.dim[src];
            out.start[k] = self.start[src];
            out.step[k] = self.step[src];
        }
        out
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
        let mut t = l.clone();
        t.swap_axes(0, 1);
        assert_eq!(t.shape(), &[6, 4]);
        assert_eq!(t.offset(&Ix2(5, 3)), l.offset(&Ix2(3, 5)));

        let mut s = l.clone();
        assert_eq!(s.slice_axis(Axis(1), Slice::new(1, Some(6), 2)), 0);
        assert_eq!(s.shape(), &[4, 3]);
        for i in 0..4 {
            for k in 0..3 {
                assert_eq!(s.offset(&Ix2(i, k)), l.offset(&Ix2(i, 1 + 2 * k)));
            }
        }

        let mut inv = s.clone();
        inv.invert_axis(Axis(1));
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
        let mut row = l.clone();
        row.collapse_axis(Axis(0), 3);
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
