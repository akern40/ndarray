use crate::{Layout, Ranked};

pub trait AxisOps: Layout {
    type Reversed: AxisOps;
    type Inverted: AxisOps;

    #[must_use]
    fn reverse_axes(self) -> Self::Reversed;

    fn invert_axis(self, axis: Axis) -> Option<(Self::Inverted, isize)>;

    fn axis(self, index: usize) -> Option<AxisSelection<Self>>
    where
        Self: Sized,
    {
        let axis = Axis::new(self.shape().ndim(), index)?;
        Some(AxisSelection::new(self, axis))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Axis {
    ndim: usize,
    index: usize,
}

impl Axis {
    #[must_use]
    pub const fn new(ndim: usize, index: usize) -> Option<Self> {
        if index < ndim {
            Some(Self { ndim, index })
        } else {
            None
        }
    }

    #[must_use]
    pub const fn ndim(self) -> usize {
        self.ndim
    }

    #[must_use]
    pub const fn index(self) -> usize {
        self.index
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AxisSelection<T> {
    target: T,
    axis: Axis,
}

impl<T> AxisSelection<T> {
    const fn new(target: T, axis: Axis) -> Self {
        Self { target, axis }
    }

    #[must_use]
    pub const fn axis(&self) -> Axis {
        self.axis
    }
}

impl<T: AxisOps> AxisSelection<T> {
    pub fn invert(self) -> Option<(T::Inverted, isize)> {
        self.target.invert_axis(self.axis)
    }
}
