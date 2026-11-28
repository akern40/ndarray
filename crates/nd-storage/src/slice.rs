use core::mem::size_of;

use crate::Storage;

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

impl<T> Storage for SliceStorage<'_, T> {
    type Elem = T;
    type Position = usize;

    fn displace(&self, position: &Self::Position, offset: isize) -> Option<Self::Position> {
        displace_position(*position, offset)
    }

    fn validate_offset_bounds(
        &self,
        zero: &Self::Position,
        bounds: Option<(isize, isize)>,
    ) -> Option<()> {
        validate_slice_offset_bounds::<T>(self.len(), *zero, bounds)
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

impl<T> Storage for SliceStorageMut<'_, T> {
    type Elem = T;
    type Position = usize;

    fn displace(&self, position: &Self::Position, offset: isize) -> Option<Self::Position> {
        displace_position(*position, offset)
    }

    fn validate_offset_bounds(
        &self,
        zero: &Self::Position,
        bounds: Option<(isize, isize)>,
    ) -> Option<()> {
        validate_slice_offset_bounds::<T>(self.len(), *zero, bounds)
    }
}

fn displace_position(position: usize, offset: isize) -> Option<usize> {
    if offset >= 0 {
        position.checked_add(usize::try_from(offset).ok()?)
    } else {
        position.checked_sub(offset.unsigned_abs())
    }
}

fn validate_slice_offset_bounds<T>(
    len: usize,
    zero: usize,
    bounds: Option<(isize, isize)>,
) -> Option<()> {
    match bounds {
        Some((min, max)) => {
            let span = element_span(min, max)?;
            validate_byte_span::<T>(span)?;

            displace_position(zero, min)?;
            let max_position = displace_position(zero, max)?;
            (max_position < len).then_some(())
        }
        None => (zero <= len).then_some(()),
    }
}

fn element_span(min: isize, max: isize) -> Option<usize> {
    let span = max.checked_sub(min)?;
    usize::try_from(span).ok()
}

fn validate_byte_span<T>(element_span: usize) -> Option<()> {
    let byte_span = element_span.checked_mul(size_of::<T>())?;
    (byte_span <= isize::MAX.unsigned_abs()).then_some(())
}

#[cfg(test)]
mod tests {
    use super::{SliceStorage, SliceStorageMut, element_span, validate_byte_span};
    use crate::Storage;

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
    fn offset_bound_validation_uses_storage_extent() {
        let data = [10, 20, 30, 40];
        let storage = SliceStorage::new(&data);

        assert!(storage.validate_offset_bounds(&1, Some((-1, 2))).is_some());
        assert!(storage.validate_offset_bounds(&1, Some((-1, 3))).is_none());
        assert!(storage.validate_offset_bounds(&4, None).is_some());
        assert!(storage.validate_offset_bounds(&5, None).is_none());
    }

    #[test]
    fn element_span_must_fit_isize() {
        assert_eq!(
            element_span(isize::MIN + 1, 0),
            Some(isize::MAX.unsigned_abs())
        );
        assert_eq!(element_span(isize::MIN, 0), None);
        assert_eq!(element_span(isize::MIN, isize::MAX), None);
    }

    #[test]
    fn byte_span_must_fit_isize() {
        assert!(validate_byte_span::<u8>(isize::MAX.unsigned_abs()).is_some());
        assert!(validate_byte_span::<u16>(isize::MAX.unsigned_abs()).is_none());
        assert!(validate_byte_span::<()>(usize::MAX).is_some());
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
}
