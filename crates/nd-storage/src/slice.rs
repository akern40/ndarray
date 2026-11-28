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

#[cfg(test)]
mod tests {
    use super::{SliceStorage, SliceStorageMut};

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
}
