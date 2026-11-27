use nd_layout::{Axis, AxisOps, IndexSpace, Layout, Ranked};

use crate::storage::{SliceStorage, offset_position};

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
        storage.validate_layout(&layout, origin)?;
        Some(Self {
            storage,
            layout,
            origin,
        })
    }

    #[must_use]
    pub fn get(&self, index: &L::Index) -> Option<&'a T> {
        let offset = self.layout.offset(index)?;
        let position = offset_position(self.origin, offset)?;
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
        let position = self.origin.wrapping_add_signed(offset);
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
        let axis = Axis::new(self.layout.shape().ndim(), index)?;
        Some(SliceViewAxis { view: self, axis })
    }
}

#[derive(Debug)]
pub struct SliceViewAxis<'a, T, L: AxisOps> {
    view: SliceView<'a, T, L>,
    axis: Axis,
}

impl<'a, T, L: AxisOps> SliceViewAxis<'a, T, L> {
    pub fn invert(self) -> Option<SliceView<'a, T, L::Inverted>> {
        let (layout, translation) = self.view.layout.invert_axis(self.axis)?;
        let origin = offset_position(self.view.origin, translation)?;
        self.view.storage.validate_layout(&layout, origin)?;

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
