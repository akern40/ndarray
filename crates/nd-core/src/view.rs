use nd_layout::{AxisOps, IndexSpace, Layout, Ranked};

use crate::storage::{SliceStorage, SliceStorageMut, Storage, validate_layout};

#[derive(Debug, Clone, Copy)]
pub struct SliceView<'a, T, L: Layout> {
    storage: SliceStorage<'a, T>,
    layout: L,
    origin: usize,
}

impl<'a, T, L: Layout> SliceView<'a, T, L> {
    #[must_use]
    pub fn try_new(data: &'a [T], layout: L, origin: usize) -> Option<Self> {
        let storage = SliceStorage::new(data);
        validate_layout(&storage, &layout, &origin)?;
        Some(Self {
            storage,
            layout,
            origin,
        })
    }

    #[must_use]
    pub fn get(&self, index: &L::Index) -> Option<&'a T> {
        let offset = self.layout.offset(index)?;
        let position = self.storage.displace(&self.origin, offset)?;
        self.storage.get(position)
    }

    pub fn iter(&self) -> impl Iterator<Item = &'a T> + '_ {
        if let Some(len) = self.layout.forward_contiguous_len()
            && let Some(slice) = self.storage.slice(self.origin, len)
        {
            return SliceViewIter::Contiguous(slice.iter());
        }

        SliceViewIter::Offsets {
            view: self,
            offsets: self.layout.offsets(),
        }
    }

    fn get_valid_offset(&self, offset: isize) -> &'a T {
        let Some(position) = self.storage.displace(&self.origin, offset) else {
            unreachable!()
        };
        let Some(value) = self.storage.get(position) else {
            unreachable!()
        };
        value
    }

    #[must_use]
    pub const fn origin(&self) -> usize {
        self.origin
    }

    #[must_use]
    pub const fn layout(&self) -> &L {
        &self.layout
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

impl<'a, T, L: AxisOps> SliceView<'a, T, L> {
    #[must_use]
    pub fn reverse_axes(self) -> SliceView<'a, T, L::Reversed> {
        SliceView {
            storage: self.storage,
            layout: self.layout.reverse_axes(),
            origin: self.origin,
        }
    }

    pub fn axis(self, index: usize) -> Option<SliceViewAxis<'a, T, L>> {
        (index < self.layout.shape().ndim()).then(|| SliceViewAxis {
            view: self,
            axis: index,
        })
    }
}

#[derive(Debug)]
pub struct SliceViewAxis<'a, T, L: AxisOps> {
    view: SliceView<'a, T, L>,
    axis: usize,
}

impl<'a, T, L: AxisOps> SliceViewAxis<'a, T, L> {
    pub fn invert(self) -> Option<SliceView<'a, T, L::Inverted>> {
        let (layout, translation) = self.view.layout.invert_axis(self.axis)?;
        let origin = self.view.storage.displace(&self.view.origin, translation)?;
        validate_layout(&self.view.storage, &layout, &origin)?;

        Some(SliceView {
            storage: self.view.storage,
            layout,
            origin,
        })
    }
}

impl<T, L: Layout> Ranked for SliceView<'_, T, L> {
    type NDim = <L::Shape as Ranked>::NDim;

    fn ndim(&self) -> usize {
        self.layout.shape().ndim()
    }
}

impl<T, L: Layout> IndexSpace for SliceView<'_, T, L> {
    type Shape = L::Shape;
    type Index = L::Index;
    type Indices<'a>
        = L::Indices<'a>
    where
        Self: 'a;

    fn shape(&self) -> &Self::Shape {
        self.layout.shape()
    }

    fn contains(&self, index: &Self::Index) -> bool {
        self.layout.contains(index)
    }

    fn indices(&self) -> Self::Indices<'_> {
        self.layout.indices()
    }
}

#[derive(Debug)]
pub struct SliceViewMut<'a, T, L: Layout> {
    storage: SliceStorageMut<'a, T>,
    layout: L,
    origin: usize,
}

impl<'a, T, L: Layout> SliceViewMut<'a, T, L> {
    #[must_use]
    pub fn try_new(data: &'a mut [T], layout: L, origin: usize) -> Option<Self> {
        let storage = SliceStorageMut::new(data);
        validate_layout(&storage, &layout, &origin)?;
        if layout.offset_bounds().is_some() && !layout.is_unique() {
            return None;
        }
        Some(Self {
            storage,
            layout,
            origin,
        })
    }

    #[must_use]
    pub fn get(&self, index: &L::Index) -> Option<&T> {
        let offset = self.layout.offset(index)?;
        let position = self.storage.displace(&self.origin, offset)?;
        self.storage.get(position)
    }

    #[must_use]
    pub fn get_mut(&mut self, index: &L::Index) -> Option<&mut T> {
        let offset = self.layout.offset(index)?;
        let position = self.storage.displace(&self.origin, offset)?;
        self.storage.get_mut(position)
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> + '_ {
        if let Some(len) = self.layout.forward_contiguous_len()
            && let Some(slice) = self.storage.slice(self.origin, len)
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
        self.storage.slice(self.origin, len)
    }

    #[must_use]
    pub fn forward_slice_mut(&mut self) -> Option<&mut [T]> {
        let len = self.layout.forward_contiguous_len()?;
        self.storage.slice_mut(self.origin, len)
    }

    #[must_use]
    pub fn forward_iter_mut(&mut self) -> Option<core::slice::IterMut<'_, T>> {
        self.forward_slice_mut().map(|slice| slice.iter_mut())
    }

    fn get_valid_offset(&self, offset: isize) -> &T {
        let Some(position) = self.storage.displace(&self.origin, offset) else {
            unreachable!()
        };
        let Some(value) = self.storage.get(position) else {
            unreachable!()
        };
        value
    }

    #[must_use]
    pub const fn origin(&self) -> usize {
        self.origin
    }

    #[must_use]
    pub const fn layout(&self) -> &L {
        &self.layout
    }

    #[must_use]
    pub fn into_view(self) -> SliceView<'a, T, L> {
        let Self {
            storage,
            layout,
            origin,
        } = self;
        SliceView {
            storage: storage.into_shared(),
            layout,
            origin,
        }
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

impl<'a, T, L: AxisOps> SliceViewMut<'a, T, L> {
    #[must_use]
    pub fn reverse_axes(self) -> SliceViewMut<'a, T, L::Reversed> {
        let Self {
            storage,
            layout,
            origin,
        } = self;
        SliceViewMut {
            storage,
            layout: layout.reverse_axes(),
            origin,
        }
    }

    pub fn axis(self, index: usize) -> Option<SliceViewMutAxis<'a, T, L>> {
        (index < self.layout.shape().ndim()).then(|| SliceViewMutAxis {
            view: self,
            axis: index,
        })
    }
}

#[derive(Debug)]
pub struct SliceViewMutAxis<'a, T, L: AxisOps> {
    view: SliceViewMut<'a, T, L>,
    axis: usize,
}

impl<'a, T, L: AxisOps> SliceViewMutAxis<'a, T, L> {
    pub fn invert(self) -> Option<SliceViewMut<'a, T, L::Inverted>> {
        let Self { view, axis } = self;
        let SliceViewMut {
            storage,
            layout: original_layout,
            origin: original_origin,
        } = view;
        let (layout, translation) = original_layout.invert_axis(axis)?;
        let origin = storage.displace(&original_origin, translation)?;
        validate_layout(&storage, &layout, &origin)?;

        Some(SliceViewMut {
            storage,
            layout,
            origin,
        })
    }
}

impl<T, L: Layout> Ranked for SliceViewMut<'_, T, L> {
    type NDim = <L::Shape as Ranked>::NDim;

    fn ndim(&self) -> usize {
        self.layout.shape().ndim()
    }
}

impl<T, L: Layout> IndexSpace for SliceViewMut<'_, T, L> {
    type Shape = L::Shape;
    type Index = L::Index;
    type Indices<'a>
        = L::Indices<'a>
    where
        Self: 'a;

    fn shape(&self) -> &Self::Shape {
        self.layout.shape()
    }

    fn contains(&self, index: &Self::Index) -> bool {
        self.layout.contains(index)
    }

    fn indices(&self) -> Self::Indices<'_> {
        self.layout.indices()
    }
}
