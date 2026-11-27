use crate::Layout;

pub trait AxisOps: Layout {
    type Reversed: AxisOps;
    type Inverted: AxisOps;

    #[must_use]
    fn reverse_axes(self) -> Self::Reversed;

    fn invert_axis(self, axis: usize) -> Option<(Self::Inverted, isize)>;

    fn axis(self, axis: usize) -> Axis<Self>
    where
        Self: Sized,
    {
        Axis::new(self, axis)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Axis<L> {
    layout: L,
    axis: usize,
}

impl<L> Axis<L> {
    const fn new(layout: L, axis: usize) -> Self {
        Self { layout, axis }
    }
}

impl<L: AxisOps> Axis<L> {
    pub fn invert(self) -> Option<(L::Inverted, isize)> {
        self.layout.invert_axis(self.axis)
    }
}
