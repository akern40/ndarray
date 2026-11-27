use crate::{ConstRank, Ranked};

pub trait Shape: Ranked {
    type Iter<'a>: Iterator<Item = usize>
    where
        Self: 'a;

    #[must_use]
    fn axis_len(&self, axis: usize) -> Option<usize>;

    #[must_use]
    fn size(&self) -> Option<usize> {
        let mut product = Some(1_usize);
        for len in self.iter() {
            if len == 0 {
                return Some(0);
            }
            if let Some(value) = product {
                product = value.checked_mul(len);
            }
        }
        product
    }

    fn iter(&self) -> Self::Iter<'_>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShapeN<const N: usize> {
    lengths: [usize; N],
}

impl<const N: usize> ShapeN<N> {
    #[must_use]
    pub const fn new(lengths: [usize; N]) -> Self {
        Self { lengths }
    }

    #[must_use]
    pub const fn as_array(&self) -> &[usize; N] {
        &self.lengths
    }

    #[must_use]
    pub fn as_slice(&self) -> &[usize] {
        &self.lengths
    }

    #[must_use]
    pub const fn into_array(self) -> [usize; N] {
        self.lengths
    }
}

impl<const N: usize> From<[usize; N]> for ShapeN<N> {
    fn from(lengths: [usize; N]) -> Self {
        Self::new(lengths)
    }
}

impl<const N: usize> From<ShapeN<N>> for [usize; N] {
    fn from(shape: ShapeN<N>) -> Self {
        shape.into_array()
    }
}

impl<const N: usize> AsRef<[usize]> for ShapeN<N> {
    fn as_ref(&self) -> &[usize] {
        self.as_slice()
    }
}

impl<const N: usize> Ranked for ShapeN<N> {
    type NDim = ConstRank<N>;

    fn ndim(&self) -> usize {
        N
    }
}

impl<const N: usize> Shape for ShapeN<N> {
    type Iter<'a>
        = core::iter::Copied<core::slice::Iter<'a, usize>>
    where
        Self: 'a;

    fn axis_len(&self, axis: usize) -> Option<usize> {
        self.lengths.get(axis).copied()
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.lengths.iter().copied()
    }
}
