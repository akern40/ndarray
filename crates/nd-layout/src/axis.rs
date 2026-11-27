use crate::{Layout, Ranked};

pub trait AxisOps: Layout {
    type Reversed: AxisOps;
    type Inverted: AxisOps;

    #[must_use]
    fn reverse_axes(self) -> Self::Reversed;

    fn invert_axis(self, axis: usize) -> Option<(Self::Inverted, isize)>;

    fn axis(self, index: usize) -> Option<AxisSelection<Self>>
    where
        Self: Sized,
    {
        (index < self.shape().ndim()).then(|| AxisSelection::new(self, index))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AxisSelection<T> {
    target: T,
    axis: usize,
}

impl<T> AxisSelection<T> {
    const fn new(target: T, axis: usize) -> Self {
        Self { target, axis }
    }

    #[must_use]
    pub const fn axis(&self) -> usize {
        self.axis
    }
}

impl<T: AxisOps> AxisSelection<T> {
    pub fn invert(self) -> Option<(T::Inverted, isize)> {
        self.target.invert_axis(self.axis)
    }
}
