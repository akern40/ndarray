use crate::IndexSpace;

pub trait Layout: IndexSpace {
    #[must_use]
    fn offset(&self, index: &Self::Index) -> Option<isize>;

    #[must_use]
    fn offset_unchecked(&self, index: &Self::Index) -> isize;

    #[must_use]
    fn offset_bounds(&self) -> Option<(isize, isize)>;

    #[must_use]
    fn is_unique(&self) -> bool;
}
