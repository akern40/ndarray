use crate::{
    AxisOps, ConstRank, IndexSpace, Layout, LogicalOffsets, Ranked, RowMajorIndices, Shape, ShapeN,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockedLayout<const N: usize> {
    shape: ShapeN<N>,
    block_shape: ShapeN<N>,
    outer_strides: [isize; N],
    inner_strides: [isize; N],
    offset_bounds: Option<(isize, isize)>,
}

impl<const N: usize> BlockedLayout<N> {
    #[must_use]
    pub fn try_new(
        shape: ShapeN<N>,
        block_shape: ShapeN<N>,
        outer_strides: [isize; N],
        inner_strides: [isize; N],
    ) -> Option<Self> {
        for (len, block_len) in shape.iter().zip(block_shape.iter()) {
            if block_len == 0 || len % block_len != 0 {
                return None;
            }
        }

        let offset_bounds = if shape.as_array().contains(&0) {
            None
        } else {
            Some(compute_offset_bounds(
                &shape,
                &block_shape,
                &outer_strides,
                &inner_strides,
            )?)
        };

        Some(Self {
            shape,
            block_shape,
            outer_strides,
            inner_strides,
            offset_bounds,
        })
    }

    #[must_use]
    pub const fn block_shape(&self) -> &ShapeN<N> {
        &self.block_shape
    }

    #[must_use]
    pub const fn outer_strides(&self) -> &[isize; N] {
        &self.outer_strides
    }

    #[must_use]
    pub const fn inner_strides(&self) -> &[isize; N] {
        &self.inner_strides
    }
}

fn stride_bounds(steps: usize, stride: isize) -> Option<(isize, isize)> {
    if steps == 0 || stride == 0 {
        return Some((0, 0));
    }

    let steps = isize::try_from(steps).ok()?;
    let span = steps.checked_mul(stride)?;
    if stride > 0 {
        Some((0, span))
    } else {
        Some((span, 0))
    }
}

fn compute_offset_bounds<const N: usize>(
    shape: &ShapeN<N>,
    block_shape: &ShapeN<N>,
    outer_strides: &[isize; N],
    inner_strides: &[isize; N],
) -> Option<(isize, isize)> {
    let mut min = 0_isize;
    let mut max = 0_isize;

    for axis in 0..N {
        let len = shape.as_array()[axis];
        let block_len = block_shape.as_array()[axis];
        let blocks = len / block_len;
        let (outer_min, outer_max) = stride_bounds(blocks - 1, outer_strides[axis])?;
        let (inner_min, inner_max) = stride_bounds(block_len - 1, inner_strides[axis])?;
        min = min.checked_add(outer_min.checked_add(inner_min)?)?;
        max = max.checked_add(outer_max.checked_add(inner_max)?)?;
    }

    Some((min, max))
}

fn checked_stride_term(position: usize, stride: isize) -> Option<isize> {
    if stride == 0 {
        return Some(0);
    }

    let position = isize::try_from(position).ok()?;
    position.checked_mul(stride)
}

fn unchecked_stride_term(position: usize, stride: isize) -> isize {
    if stride == 0 {
        0
    } else {
        position.cast_signed().wrapping_mul(stride)
    }
}

fn checked_offset<const N: usize>(layout: &BlockedLayout<N>, index: &[usize; N]) -> Option<isize> {
    if !layout.contains(index) {
        return None;
    }

    let mut offset = 0_isize;
    for axis in 0..N {
        let position = index[axis];
        let block_len = layout.block_shape.as_array()[axis];
        let outer = checked_stride_term(position / block_len, layout.outer_strides[axis])?;
        let inner = checked_stride_term(position % block_len, layout.inner_strides[axis])?;
        offset = offset.checked_add(outer.checked_add(inner)?)?;
    }
    Some(offset)
}

fn unchecked_offset<const N: usize>(layout: &BlockedLayout<N>, index: &[usize; N]) -> isize {
    let mut offset = 0_isize;
    for axis in 0..N {
        let position = index[axis];
        let block_len = layout.block_shape.as_array()[axis];
        offset = offset.wrapping_add(unchecked_stride_term(
            position / block_len,
            layout.outer_strides[axis],
        ));
        offset = offset.wrapping_add(unchecked_stride_term(
            position % block_len,
            layout.inner_strides[axis],
        ));
    }
    offset
}

fn conservative_uniqueness<const N: usize>(layout: &BlockedLayout<N>) -> bool {
    if layout.shape.as_array().contains(&0) {
        return true;
    }

    let mut inner_used = [false; N];
    let mut outer_used = [false; N];
    let mut previous_offsets = 0_usize;

    for _ in 0..N {
        for _ in 0..2 {
            let mut next_axis = None;
            for axis in 0..N {
                if !inner_used[axis] {
                    let magnitude = layout.inner_strides[axis].unsigned_abs();
                    if next_axis.is_none_or(|(best, _, _)| magnitude < best) {
                        next_axis = Some((magnitude, false, axis));
                    }
                }
                if !outer_used[axis] {
                    let magnitude = layout.outer_strides[axis].unsigned_abs();
                    if next_axis.is_none_or(|(best, _, _)| magnitude < best) {
                        next_axis = Some((magnitude, true, axis));
                    }
                }
            }

            let Some((stride, outer, axis)) = next_axis else {
                return false;
            };
            if outer {
                outer_used[axis] = true;
            } else {
                inner_used[axis] = true;
            }

            let len = if outer {
                layout.shape.as_array()[axis] / layout.block_shape.as_array()[axis]
            } else {
                layout.block_shape.as_array()[axis]
            };
            if len <= 1 {
                continue;
            }
            if stride <= previous_offsets {
                return false;
            }
            let Some(axis_offsets) = len
                .checked_sub(1)
                .and_then(|steps| steps.checked_mul(stride))
            else {
                return false;
            };
            let Some(next_offsets) = previous_offsets.checked_add(axis_offsets) else {
                return false;
            };
            previous_offsets = next_offsets;
        }
    }

    true
}

impl<const N: usize> AxisOps for BlockedLayout<N> {
    type Reversed = Self;
    type Inverted = Self;

    fn reverse_axes(self) -> Self {
        let mut shape = self.shape.into_array();
        let mut block_shape = self.block_shape.into_array();
        let mut outer_strides = self.outer_strides;
        let mut inner_strides = self.inner_strides;
        shape.reverse();
        block_shape.reverse();
        outer_strides.reverse();
        inner_strides.reverse();

        Self {
            shape: ShapeN::new(shape),
            block_shape: ShapeN::new(block_shape),
            outer_strides,
            inner_strides,
            offset_bounds: self.offset_bounds,
        }
    }

    fn invert_axis(self, axis: usize) -> Option<(Self, isize)> {
        if axis >= N {
            return None;
        }

        let len = self.shape.as_array()[axis];
        if len <= 1 || self.shape.as_array().contains(&0) {
            return Some((self, 0));
        }

        let block_len = self.block_shape.as_array()[axis];
        let blocks = len / block_len;
        let outer_stride = self.outer_strides[axis];
        let inner_stride = self.inner_strides[axis];

        let outer_steps = isize::try_from(blocks - 1).ok()?;
        let inner_steps = isize::try_from(block_len - 1).ok()?;
        let translation = outer_steps
            .checked_mul(outer_stride)?
            .checked_add(inner_steps.checked_mul(inner_stride)?)?;

        let inverted_outer_stride = outer_stride.checked_neg()?;
        let inverted_inner_stride = inner_stride.checked_neg()?;

        let mut outer_strides = self.outer_strides;
        let mut inner_strides = self.inner_strides;
        outer_strides[axis] = inverted_outer_stride;
        inner_strides[axis] = inverted_inner_stride;
        let offset_bounds = Some(compute_offset_bounds(
            &self.shape,
            &self.block_shape,
            &outer_strides,
            &inner_strides,
        )?);

        Some((
            Self {
                shape: self.shape,
                block_shape: self.block_shape,
                outer_strides,
                inner_strides,
                offset_bounds,
            },
            translation,
        ))
    }
}

impl<const N: usize> Ranked for BlockedLayout<N> {
    type NDim = ConstRank<N>;

    fn ndim(&self) -> usize {
        N
    }
}

impl<const N: usize> IndexSpace for BlockedLayout<N> {
    type Shape = ShapeN<N>;
    type Index = [usize; N];
    type Indices<'a>
        = RowMajorIndices<'a, N>
    where
        Self: 'a;

    fn shape(&self) -> &Self::Shape {
        &self.shape
    }

    fn contains(&self, index: &Self::Index) -> bool {
        self.shape.contains(index)
    }

    fn indices(&self) -> Self::Indices<'_> {
        self.shape.indices()
    }
}

impl<const N: usize> Layout for BlockedLayout<N> {
    type Offsets<'a>
        = LogicalOffsets<'a, Self>
    where
        Self: 'a;

    fn offset(&self, index: &Self::Index) -> Option<isize> {
        checked_offset(self, index)
    }

    fn offset_unchecked(&self, index: &Self::Index) -> isize {
        unchecked_offset(self, index)
    }

    fn offset_bounds(&self) -> Option<(isize, isize)> {
        self.offset_bounds
    }

    fn is_unique(&self) -> bool {
        conservative_uniqueness(self)
    }

    fn offsets(&self) -> Self::Offsets<'_> {
        LogicalOffsets::new(self)
    }
}
