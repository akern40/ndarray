use nd_layout::Layout;

use crate::array::{ArrayAxis, ArrayBase};
use crate::storage::{PositionSpace, RefAccess, SliceStorage, SliceStorageMut};

pub type SliceView<'a, T, L> = ArrayBase<SliceStorage<'a, T>, L>;
pub type SliceViewAxis<'a, T, L> = ArrayAxis<SliceStorage<'a, T>, L>;
pub type SliceViewMut<'a, T, L> = ArrayBase<SliceStorageMut<'a, T>, L>;
pub type SliceViewMutAxis<'a, T, L> = ArrayAxis<SliceStorageMut<'a, T>, L>;

impl<'a, T, L: Layout> ArrayBase<SliceStorage<'a, T>, L> {
    #[must_use]
    pub fn try_new(data: &'a [T], layout: L, origin: usize) -> Option<Self> {
        let storage = SliceStorage::new(data);
        Self::try_compose(storage, layout, origin)
    }

    pub fn iter(&self) -> impl Iterator<Item = &'a T> + '_ {
        if let Some(len) = self.layout.forward_contiguous_len()
            && let Some(slice) = self.storage.slice(self.zero_position, len)
        {
            return SliceViewIter::Contiguous(slice.iter());
        }

        SliceViewIter::Offsets {
            view: self,
            offsets: self.layout.offsets(),
        }
    }

    fn get_valid_offset(&self, offset: isize) -> &'a T {
        let Some(position) = self.storage.displace(&self.zero_position, offset) else {
            unreachable!()
        };
        let Some(value) = self.storage.get(&position) else {
            unreachable!()
        };
        value
    }

    #[must_use]
    pub const fn origin(&self) -> usize {
        self.zero_position
    }
}

enum SliceViewIter<'a, 'view, T, L: Layout> {
    Contiguous(core::slice::Iter<'a, T>),
    Offsets {
        view: &'view SliceView<'a, T, L>,
        offsets: L::Offsets<'view>,
    },
}

impl<'a, T, L: Layout> Iterator for SliceViewIter<'a, '_, T, L> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Contiguous(iter) => iter.next(),
            Self::Offsets { view, offsets } => {
                offsets.next().map(|offset| view.get_valid_offset(offset))
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Contiguous(iter) => iter.size_hint(),
            Self::Offsets { offsets, .. } => offsets.size_hint(),
        }
    }

    fn fold<Acc, F>(self, init: Acc, mut f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        match self {
            Self::Contiguous(iter) => iter.fold(init, f),
            Self::Offsets { view, offsets } => offsets.fold(init, |accum, offset| {
                f(accum, view.get_valid_offset(offset))
            }),
        }
    }
}

impl<'a, T, L: Layout> ArrayBase<SliceStorageMut<'a, T>, L> {
    #[must_use]
    pub fn try_new(data: &'a mut [T], layout: L, origin: usize) -> Option<Self> {
        let storage = SliceStorageMut::new(data);
        if layout.offset_bounds().is_some() && !layout.is_unique() {
            return None;
        }
        Self::try_compose(storage, layout, origin)
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> + '_ {
        if let Some(len) = self.layout.forward_contiguous_len()
            && let Some(slice) = self.storage.slice(self.zero_position, len)
        {
            return SliceViewMutIter::Contiguous(slice.iter());
        }

        SliceViewMutIter::Offsets {
            view: self,
            offsets: self.layout.offsets(),
        }
    }

    #[must_use]
    pub fn forward_slice(&self) -> Option<&[T]> {
        let len = self.layout.forward_contiguous_len()?;
        self.storage.slice(self.zero_position, len)
    }

    #[must_use]
    pub fn forward_slice_mut(&mut self) -> Option<&mut [T]> {
        let len = self.layout.forward_contiguous_len()?;
        self.storage.slice_mut(self.zero_position, len)
    }

    #[must_use]
    pub fn forward_iter_mut(&mut self) -> Option<core::slice::IterMut<'_, T>> {
        self.forward_slice_mut().map(|slice| slice.iter_mut())
    }

    fn get_valid_offset(&self, offset: isize) -> &T {
        let Some(position) = self.storage.displace(&self.zero_position, offset) else {
            unreachable!()
        };
        let Some(value) = self.storage.get(&position) else {
            unreachable!()
        };
        value
    }

    #[must_use]
    pub const fn origin(&self) -> usize {
        self.zero_position
    }

    #[must_use]
    pub fn into_view(self) -> SliceView<'a, T, L> {
        let ArrayBase {
            storage,
            layout,
            zero_position,
        } = self;
        ArrayBase::new_unchecked(storage.into_shared(), layout, zero_position)
    }
}

enum SliceViewMutIter<'a, 'view, T, L: Layout> {
    Contiguous(core::slice::Iter<'view, T>),
    Offsets {
        view: &'view SliceViewMut<'a, T, L>,
        offsets: L::Offsets<'view>,
    },
}

impl<'view, T, L: Layout> Iterator for SliceViewMutIter<'_, 'view, T, L> {
    type Item = &'view T;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Contiguous(iter) => iter.next(),
            Self::Offsets { view, offsets } => {
                offsets.next().map(|offset| view.get_valid_offset(offset))
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Contiguous(iter) => iter.size_hint(),
            Self::Offsets { offsets, .. } => offsets.size_hint(),
        }
    }

    fn fold<Acc, F>(self, init: Acc, mut f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        match self {
            Self::Contiguous(iter) => iter.fold(init, f),
            Self::Offsets { view, offsets } => offsets.fold(init, |accum, offset| {
                f(accum, view.get_valid_offset(offset))
            }),
        }
    }
}
