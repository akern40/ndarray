use crate::{Layout, Ranked};

pub trait AxisOps: Ranked {
    type Transformed: AxisOps + Layout;

    fn apply_axis_transform(
        self,
        transform: AxisTransform,
    ) -> Option<TransformedLayout<Self::Transformed>>;

    fn reverse_axes(self) -> Option<TransformedLayout<Self::Transformed>>
    where
        Self: Sized,
    {
        self.apply_axis_transform(AxisTransform::ReverseAxes)
    }

    fn invert_axis(self, axis: Axis) -> Option<TransformedLayout<Self::Transformed>>
    where
        Self: Sized,
    {
        self.apply_axis_transform(AxisTransform::InvertAxis(axis))
    }

    fn axis(self, index: usize) -> Option<AxisSelection<Self>>
    where
        Self: Sized,
    {
        let axis = Axis::new(self.ndim(), index)?;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AxisTransform {
    ReverseAxes,
    InvertAxis(Axis),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransformedLayout<L> {
    layout: L,
    translation: isize,
}

impl<L> TransformedLayout<L> {
    #[must_use]
    pub const fn new(layout: L, translation: isize) -> Self {
        Self {
            layout,
            translation,
        }
    }

    #[must_use]
    pub const fn layout(&self) -> &L {
        &self.layout
    }

    #[must_use]
    pub const fn translation(&self) -> isize {
        self.translation
    }

    #[must_use]
    pub fn into_parts(self) -> (L, isize) {
        (self.layout, self.translation)
    }
}

impl<L: Ranked> Ranked for TransformedLayout<L> {
    type NDim = L::NDim;

    fn ndim(&self) -> usize {
        self.layout.ndim()
    }
}

impl<L: AxisOps> AxisOps for TransformedLayout<L> {
    type Transformed = L::Transformed;

    fn apply_axis_transform(
        self,
        transform: AxisTransform,
    ) -> Option<TransformedLayout<Self::Transformed>> {
        let (layout, first_translation) = self.into_parts();
        let transformed = layout.apply_axis_transform(transform)?;
        let (layout, second_translation) = transformed.into_parts();
        let translation = first_translation.checked_add(second_translation)?;
        Some(TransformedLayout::new(layout, translation))
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
    pub fn invert(self) -> Option<TransformedLayout<T::Transformed>> {
        self.target.invert_axis(self.axis)
    }
}
