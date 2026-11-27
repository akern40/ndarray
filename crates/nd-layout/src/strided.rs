use crate::{AxisOps, ConstRank, IndexSpace, Layout, Ranked, RowMajorIndices, Shape, ShapeN};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StridedLayout<const N: usize> {
    shape: ShapeN<N>,
    strides: [isize; N],
    offset_bounds: Option<(isize, isize)>,
}

impl<const N: usize> StridedLayout<N> {
    #[must_use]
    pub fn try_new(shape: ShapeN<N>, strides: [isize; N]) -> Option<Self> {
        let offset_bounds = if shape.as_array().contains(&0) {
            None
        } else {
            Some(compute_offset_bounds(&shape, &strides)?)
        };
        Some(Self {
            shape,
            strides,
            offset_bounds,
        })
    }

    #[must_use]
    pub const fn strides(&self) -> &[isize; N] {
        &self.strides
    }
}

#[derive(Debug, Clone)]
pub struct StridedOffsets<'a, const N: usize> {
    shape: &'a ShapeN<N>,
    strides: &'a [isize; N],
    index: Option<[usize; N]>,
    offset: isize,
}

impl<'a, const N: usize> StridedOffsets<'a, N> {
    fn new(layout: &'a StridedLayout<N>) -> Self {
        let index = layout
            .shape
            .as_array()
            .iter()
            .all(|len| *len > 0)
            .then_some([0; N]);
        Self {
            shape: &layout.shape,
            strides: &layout.strides,
            index,
            offset: 0,
        }
    }
}

impl<const N: usize> Iterator for StridedOffsets<'_, N> {
    type Item = isize;

    fn next(&mut self) -> Option<Self::Item> {
        let mut index = self.index?;
        let current = self.offset;
        self.index = advance_offsets(self.shape, self.strides, &mut index, &mut self.offset)
            .then_some(index);
        Some(current)
    }

    fn fold<Acc, G>(mut self, init: Acc, mut f: G) -> Acc
    where
        G: FnMut(Acc, Self::Item) -> Acc,
    {
        let mut accum = init;
        if N == 0 {
            if self.index.is_some() {
                accum = f(accum, self.offset);
            }
            return accum;
        }

        while let Some(mut index) = self.index {
            let Some(len) = self.shape.as_array().last().copied() else {
                unreachable!()
            };
            let Some(stride) = self.strides.last().copied() else {
                unreachable!()
            };
            let Some(element_index) = index.last().copied() else {
                unreachable!()
            };

            let mut remaining = len - element_index;
            while remaining > 0 {
                accum = f(accum, self.offset);
                self.offset = self.offset.wrapping_add(stride);
                remaining -= 1;
            }

            self.offset = self.offset.wrapping_sub(stride);
            let Some(last) = index.last_mut() else {
                unreachable!()
            };
            *last = len - 1;
            self.index = advance_offsets(self.shape, self.strides, &mut index, &mut self.offset)
                .then_some(index);
        }
        accum
    }
}

fn advance_offsets<const N: usize>(
    shape: &ShapeN<N>,
    strides: &[isize; N],
    index: &mut [usize; N],
    offset: &mut isize,
) -> bool {
    for ((value, len), stride) in index
        .iter_mut()
        .rev()
        .zip(shape.as_array().iter().rev())
        .zip(strides.iter().rev())
    {
        if let Some(incremented) = value.checked_add(1)
            && incremented < *len
        {
            *value = incremented;
            *offset = offset.wrapping_add(*stride);
            return true;
        }
        let reset = (*value).cast_signed().wrapping_mul(*stride);
        *offset = offset.wrapping_sub(reset);
        *value = 0;
    }
    false
}

fn compute_offset_bounds<const N: usize>(
    shape: &ShapeN<N>,
    strides: &[isize; N],
) -> Option<(isize, isize)> {
    let mut min = 0_isize;
    let mut max = 0_isize;
    for (len, stride) in shape.as_array().iter().zip(strides.iter()) {
        let steps = len.checked_sub(1)?;
        if steps == 0 || *stride == 0 {
            continue;
        }
        let steps = isize::try_from(steps).ok()?;
        let span = steps.checked_mul(*stride)?;
        if *stride > 0 {
            max = max.checked_add(span)?;
        } else {
            min = min.checked_add(span)?;
        }
    }
    Some((min, max))
}

fn checked_offset<const N: usize>(layout: &StridedLayout<N>, index: &[usize; N]) -> Option<isize> {
    if !layout.contains(index) {
        return None;
    }

    let mut offset = 0_isize;
    for (position, stride) in index.iter().zip(layout.strides.iter()) {
        if *stride == 0 {
            continue;
        }
        let position = isize::try_from(*position).ok()?;
        let term = position.checked_mul(*stride)?;
        offset = offset.checked_add(term)?;
    }
    Some(offset)
}

fn unchecked_offset<const N: usize>(strides: &[isize; N], index: &[usize; N]) -> isize {
    let mut offset = 0_isize;
    for (position, stride) in index.iter().zip(strides.iter()) {
        if *stride != 0 {
            offset = offset.wrapping_add((*position).cast_signed().wrapping_mul(*stride));
        }
    }
    offset
}

fn forward_contiguous_len<const N: usize>(
    shape: &ShapeN<N>,
    strides: &[isize; N],
) -> Option<usize> {
    let len = shape.size()?;
    if len == 0 || N == 0 {
        return Some(len);
    }

    let mut inner_len = 1_usize;
    for axis in (0..N).rev() {
        let axis_len = shape.as_array()[axis];
        if axis_len > 1 {
            let expected = isize::try_from(inner_len).ok()?;
            if strides[axis] != expected {
                return None;
            }
        }
        inner_len = inner_len.checked_mul(axis_len)?;
    }
    Some(len)
}

fn conservative_uniqueness<const N: usize>(shape: &ShapeN<N>, strides: &[isize; N]) -> bool {
    if shape.as_array().contains(&0) {
        return true;
    }

    let mut order = [0_usize; N];
    for (axis, value) in order.iter_mut().enumerate() {
        *value = axis;
    }
    for sorted in 1..N {
        let mut axis = sorted;
        while axis > 0
            && strides[order[axis]].unsigned_abs() < strides[order[axis - 1]].unsigned_abs()
        {
            order.swap(axis, axis - 1);
            axis -= 1;
        }
    }

    let mut previous_offsets = 0_usize;
    for axis in order {
        let len = shape.as_array()[axis];
        if len <= 1 {
            continue;
        }

        let stride = strides[axis].unsigned_abs();
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
    true
}

impl<const N: usize> AxisOps for StridedLayout<N> {
    type Reversed = Self;
    type Inverted = Self;

    fn reverse_axes(self) -> Self {
        let mut shape = self.shape.into_array();
        let mut strides = self.strides;
        shape.reverse();
        strides.reverse();

        Self {
            shape: ShapeN::new(shape),
            strides,
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

        let stride = self.strides[axis];
        let steps = len.checked_sub(1)?;
        let steps = isize::try_from(steps).ok()?;
        let translation = steps.checked_mul(stride)?;
        let inverted_stride = stride.checked_neg()?;

        let mut strides = self.strides;
        strides[axis] = inverted_stride;
        let offset_bounds = Some(compute_offset_bounds(&self.shape, &strides)?);

        Some((
            Self {
                shape: self.shape,
                strides,
                offset_bounds,
            },
            translation,
        ))
    }
}

impl<const N: usize> Ranked for StridedLayout<N> {
    type NDim = ConstRank<N>;

    fn ndim(&self) -> usize {
        N
    }
}

impl<const N: usize> IndexSpace for StridedLayout<N> {
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

impl<const N: usize> Layout for StridedLayout<N> {
    type Offsets<'a>
        = StridedOffsets<'a, N>
    where
        Self: 'a;

    fn offset(&self, index: &Self::Index) -> Option<isize> {
        checked_offset(self, index)
    }

    fn offset_unchecked(&self, index: &Self::Index) -> isize {
        unchecked_offset(&self.strides, index)
    }

    fn offset_bounds(&self) -> Option<(isize, isize)> {
        self.offset_bounds
    }

    fn is_unique(&self) -> bool {
        conservative_uniqueness(&self.shape, &self.strides)
    }

    fn forward_contiguous_len(&self) -> Option<usize> {
        forward_contiguous_len(&self.shape, &self.strides)
    }

    fn offsets(&self) -> Self::Offsets<'_> {
        StridedOffsets::new(self)
    }
}

#[cfg(kani)]
mod proofs {
    use super::*;
    use crate::{IndexSpace, Layout};

    #[kani::proof]
    fn valid_indices_have_bounded_offsets_2d() {
        let lengths: [usize; 2] = kani::any();
        let strides: [isize; 2] = kani::any();
        let [x_len, y_len] = lengths;
        let [x_stride, y_stride] = strides;

        kani::assume(x_len > 0 && x_len <= 4);
        kani::assume(y_len > 0 && y_len <= 4);
        kani::assume((-4..=4).contains(&x_stride));
        kani::assume((-4..=4).contains(&y_stride));

        let shape = ShapeN::new([x_len, y_len]);
        if let Some(layout) = StridedLayout::try_new(shape, [x_stride, y_stride]) {
            let index: [usize; 2] = kani::any();
            kani::assume(layout.contains(&index));

            let checked = layout.offset(&index);
            assert!(checked.is_some());
            if let Some(checked) = checked {
                let unchecked = layout.offset_unchecked(&index);
                assert_eq!(unchecked, checked);
                if let Some((min, max)) = layout.offset_bounds() {
                    assert!(min <= checked && checked <= max);
                }
            }
        }
    }
}
