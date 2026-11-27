use crate::{Shape, ShapeN};

pub trait IndexSpace {
    type Shape: Shape;
    type Index;
    type Indices<'a>: Iterator<Item = Self::Index>
    where
        Self: 'a;

    #[must_use]
    fn shape(&self) -> &Self::Shape;

    #[must_use]
    fn contains(&self, index: &Self::Index) -> bool;

    fn indices(&self) -> Self::Indices<'_>;
}

#[derive(Debug, Clone)]
pub struct ShapeNIndices<'a, const N: usize> {
    shape: &'a ShapeN<N>,
    next: Option<[usize; N]>,
}

impl<'a, const N: usize> ShapeNIndices<'a, N> {
    fn new(shape: &'a ShapeN<N>) -> Self {
        let next = shape
            .as_array()
            .iter()
            .all(|len| *len > 0)
            .then_some([0; N]);
        Self { shape, next }
    }
}

impl<const N: usize> Iterator for ShapeNIndices<'_, N> {
    type Item = [usize; N];

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.next?;
        let mut following = current;
        self.next = advance(self.shape, &mut following).then_some(following);
        Some(current)
    }
}

fn advance<const N: usize>(shape: &ShapeN<N>, index: &mut [usize; N]) -> bool {
    for (value, len) in index.iter_mut().rev().zip(shape.as_array().iter().rev()) {
        if let Some(incremented) = value.checked_add(1)
            && incremented < *len
        {
            *value = incremented;
            return true;
        }
        *value = 0;
    }
    false
}

impl<const N: usize> IndexSpace for ShapeN<N> {
    type Shape = Self;
    type Index = [usize; N];
    type Indices<'a>
        = ShapeNIndices<'a, N>
    where
        Self: 'a;

    fn shape(&self) -> &Self::Shape {
        self
    }

    fn contains(&self, index: &Self::Index) -> bool {
        index
            .iter()
            .zip(self.as_array())
            .all(|(position, len)| *position < *len)
    }

    fn indices(&self) -> Self::Indices<'_> {
        ShapeNIndices::new(self)
    }
}
