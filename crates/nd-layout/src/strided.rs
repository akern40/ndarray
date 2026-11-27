use crate::{ConstRank, IndexSpace, Layout, Ranked, RowMajorIndices, ShapeN};

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

    #[must_use]
    pub fn reverse_axes(self) -> Self {
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

fn conservative_uniqueness<const N: usize>(shape: &ShapeN<N>, strides: &[isize; N]) -> bool {
    if shape.as_array().contains(&0) {
        return true;
    }

    let mut effective_axes = 0_usize;
    for (len, stride) in shape.as_array().iter().zip(strides.iter()) {
        if *len > 1 {
            effective_axes += 1;
            if *stride == 0 {
                return false;
            }
        }
    }
    effective_axes <= 1
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
