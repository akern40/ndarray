use crate::{PositionSpace, position_space::displace_position};

#[derive(Debug, Clone, Copy)]
pub struct SliceStorage<'a, T> {
    data: &'a [T],
}

impl<'a, T> SliceStorage<'a, T> {
    #[must_use]
    pub const fn new(data: &'a [T]) -> Self {
        Self { data }
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.data.len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[must_use]
    pub fn get(&self, position: usize) -> Option<&'a T> {
        let data = self.data;
        data.get(position)
    }

    #[must_use]
    pub fn slice(&self, start: usize, len: usize) -> Option<&'a [T]> {
        let end = start.checked_add(len)?;
        let data = self.data;
        data.get(start..end)
    }
}

impl<T> PositionSpace for SliceStorage<'_, T> {
    type Position = usize;

    fn displace(&self, position: &Self::Position, offset: isize) -> Option<Self::Position> {
        displace_position(*position, offset)
    }
}

#[derive(Debug)]
pub struct SliceStorageMut<'a, T> {
    data: &'a mut [T],
}

impl<'a, T> SliceStorageMut<'a, T> {
    #[must_use]
    pub const fn new(data: &'a mut [T]) -> Self {
        Self { data }
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.data.len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[must_use]
    pub fn get(&self, position: usize) -> Option<&T> {
        self.data.get(position)
    }

    #[must_use]
    pub fn get_mut(&mut self, position: usize) -> Option<&mut T> {
        self.data.get_mut(position)
    }

    #[must_use]
    pub fn slice(&self, start: usize, len: usize) -> Option<&[T]> {
        let end = start.checked_add(len)?;
        self.data.get(start..end)
    }

    #[must_use]
    pub fn slice_mut(&mut self, start: usize, len: usize) -> Option<&mut [T]> {
        let end = start.checked_add(len)?;
        self.data.get_mut(start..end)
    }

    #[must_use]
    pub fn into_shared(self) -> SliceStorage<'a, T> {
        SliceStorage::new(self.data)
    }
}

impl<T> PositionSpace for SliceStorageMut<'_, T> {
    type Position = usize;

    fn displace(&self, position: &Self::Position, offset: isize) -> Option<Self::Position> {
        displace_position(*position, offset)
    }
}

#[cfg(test)]
mod tests {
    use super::{SliceStorage, SliceStorageMut};
    use crate::{PositionSpace, position_space::displace_position};

    #[test]
    fn displacement_is_checked() {
        let data = [10, 20, 30];
        let storage = SliceStorage::new(&data);

        assert_eq!(storage.displace(&1, 1), Some(2));
        assert_eq!(storage.displace(&1, -1), Some(0));
        assert_eq!(storage.displace(&0, -1), None);
        assert_eq!(storage.displace(&usize::MAX, 1), None);
    }

    #[test]
    fn shared_slice_access_is_position_based() {
        let data = [10, 20, 30];
        let storage = SliceStorage::new(&data);

        assert_eq!(storage.len(), 3);
        assert_eq!(storage.get(1), Some(&20));
        assert_eq!(storage.get(3), None);
        assert_eq!(storage.slice(1, 2), Some(&data[1..3]));
        assert_eq!(storage.slice(2, 2), None);
    }

    #[test]
    fn mutable_slice_access_is_position_based() {
        let mut data = [10, 20, 30];
        let mut storage = SliceStorageMut::new(&mut data);

        assert_eq!(storage.len(), 3);
        assert_eq!(storage.get(1), Some(&20));
        if let Some(value) = storage.get_mut(1) {
            *value = 40;
        }
        assert_eq!(storage.get(1), Some(&40));
        assert!(storage.get_mut(3).is_none());
    }

    #[test]
    fn mutable_slice_ranges_are_checked() {
        let mut data = [10, 20, 30];
        let mut storage = SliceStorageMut::new(&mut data);

        assert_eq!(storage.slice(1, 2), Some(&[20, 30][..]));
        if let Some(slice) = storage.slice_mut(1, 2) {
            slice[0] = 40;
        }
        assert_eq!(storage.slice(1, 2), Some(&[40, 30][..]));
        assert!(storage.slice_mut(2, 2).is_none());
    }

    #[test]
    fn mutable_storage_converts_to_shared_storage() {
        let mut data = [10, 20, 30];
        let storage = SliceStorageMut::new(&mut data).into_shared();

        assert_eq!(storage.len(), 3);
        assert_eq!(storage.get(2), Some(&30));
    }

    #[derive(Debug, PartialEq, Eq)]
    struct PagePosition {
        page: usize,
        offset: usize,
    }

    struct Pages {
        page_len: usize,
    }

    impl PositionSpace for Pages {
        type Position = PagePosition;

        fn displace(&self, position: &Self::Position, offset: isize) -> Option<Self::Position> {
            let linear = position
                .page
                .checked_mul(self.page_len)?
                .checked_add(position.offset)?;
            let linear = super::displace_position(linear, offset)?;
            Some(PagePosition {
                page: linear.checked_div(self.page_len)?,
                offset: linear.checked_rem(self.page_len)?,
            })
        }
    }

    #[test]
    fn position_spaces_control_position_representation() {
        let pages = Pages { page_len: 4 };
        let position = PagePosition { page: 1, offset: 3 };

        assert_eq!(
            pages.displace(&position, 2),
            Some(PagePosition { page: 2, offset: 1 })
        );
        assert_eq!(
            pages.displace(&position, -7),
            Some(PagePosition { page: 0, offset: 0 })
        );
        assert_eq!(pages.displace(&position, -8), None);
    }
}
