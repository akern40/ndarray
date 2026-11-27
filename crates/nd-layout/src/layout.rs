use crate::IndexSpace;

pub trait Layout: IndexSpace {
    type Offsets<'a>: Iterator<Item = isize>
    where
        Self: 'a;

    #[must_use]
    fn offset(&self, index: &Self::Index) -> Option<isize>;

    #[must_use]
    fn offset_unchecked(&self, index: &Self::Index) -> isize;

    #[must_use]
    fn offset_bounds(&self) -> Option<(isize, isize)>;

    #[must_use]
    fn is_unique(&self) -> bool;

    fn offsets(&self) -> Self::Offsets<'_>;
}

#[derive(Debug)]
pub struct LogicalOffsets<'a, L: Layout> {
    layout: &'a L,
    indices: L::Indices<'a>,
}

impl<'a, L: Layout> LogicalOffsets<'a, L> {
    #[must_use]
    pub fn new(layout: &'a L) -> Self {
        Self {
            layout,
            indices: layout.indices(),
        }
    }
}

impl<L: Layout> Iterator for LogicalOffsets<'_, L> {
    type Item = isize;

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.indices.next()?;
        let Some(offset) = self.layout.offset(&index) else {
            unreachable!()
        };
        Some(offset)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.indices.size_hint()
    }
}
