use nd_layout::{IndexSpace, Layout, Ranked, StridedLayout};

#[derive(Debug, Clone, Copy)]
pub struct SliceView<'a, T, L: Layout> {
    data: &'a [T],
    layout: L,
    origin: usize,
}

impl<'a, T, L: Layout> SliceView<'a, T, L> {
    #[must_use]
    pub fn try_new(data: &'a [T], layout: L, origin: usize) -> Option<Self> {
        match layout.offset_bounds() {
            Some((min, max)) => {
                offset_position(origin, min)?;
                let max_position = offset_position(origin, max)?;
                if max_position >= data.len() {
                    return None;
                }
            }
            None if origin > data.len() => return None,
            None => {}
        }

        Some(Self {
            data,
            layout,
            origin,
        })
    }

    #[must_use]
    pub fn get(&self, index: &L::Index) -> Option<&'a T> {
        let offset = self.layout.offset(index)?;
        let position = offset_position(self.origin, offset)?;
        self.data.get(position)
    }

    pub fn iter(&self) -> impl Iterator<Item = &'a T> + '_ {
        let view = self;
        self.layout
            .indices()
            .map(move |index| view.get_valid(&index))
    }

    fn get_valid(&self, index: &L::Index) -> &'a T {
        let Some(offset) = self.layout.offset(index) else {
            unreachable!()
        };
        let Some(position) = offset_position(self.origin, offset) else {
            unreachable!()
        };
        let Some(value) = self.data.get(position) else {
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

impl<T, const N: usize> SliceView<'_, T, StridedLayout<N>> {
    #[must_use]
    pub fn reverse_axes(self) -> Self {
        Self {
            data: self.data,
            layout: self.layout.reverse_axes(),
            origin: self.origin,
        }
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

fn offset_position(origin: usize, offset: isize) -> Option<usize> {
    if offset >= 0 {
        origin.checked_add(usize::try_from(offset).ok()?)
    } else {
        origin.checked_sub(offset.unsigned_abs())
    }
}
