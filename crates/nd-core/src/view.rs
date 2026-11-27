use nd_layout::{Layout, StridedLayout};

#[derive(Debug, Clone, Copy)]
pub struct StridedView<'a, T, const N: usize> {
    data: &'a [T],
    layout: StridedLayout<N>,
    origin: usize,
}

impl<'a, T, const N: usize> StridedView<'a, T, N> {
    #[must_use]
    pub fn try_new(data: &'a [T], layout: StridedLayout<N>, origin: usize) -> Option<Self> {
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
    pub fn get(&self, index: [usize; N]) -> Option<&'a T> {
        let offset = self.layout.offset(&index)?;
        let position = offset_position(self.origin, offset)?;
        self.data.get(position)
    }

    #[must_use]
    pub const fn origin(&self) -> usize {
        self.origin
    }

    #[must_use]
    pub const fn layout(&self) -> &StridedLayout<N> {
        &self.layout
    }
}

fn offset_position(origin: usize, offset: isize) -> Option<usize> {
    if offset >= 0 {
        origin.checked_add(usize::try_from(offset).ok()?)
    } else {
        origin.checked_sub(offset.unsigned_abs())
    }
}
