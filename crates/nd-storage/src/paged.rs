use alloc::vec::Vec;

use crate::{PositionSpace, position_space::displace_position};

#[derive(Debug)]
pub struct PagedStorage<T> {
    pages: Vec<Vec<T>>,
    page_len: usize,
    len: usize,
}

impl<T> PagedStorage<T> {
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[must_use]
    pub const fn page_len(&self) -> usize {
        self.page_len
    }

    #[must_use]
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    #[must_use]
    pub fn get(&self, position: usize) -> Option<&T> {
        let (page, offset) = self.page_index(position)?;
        self.pages.get(page)?.get(offset)
    }

    #[must_use]
    pub fn get_mut(&mut self, position: usize) -> Option<&mut T> {
        let (page, offset) = self.page_index(position)?;
        self.pages.get_mut(page)?.get_mut(offset)
    }

    #[must_use]
    pub fn contiguous_len(&self, position: usize) -> Option<usize> {
        if position > self.len {
            return None;
        }
        if position == self.len {
            return Some(0);
        }

        let offset = position.checked_rem(self.page_len)?;
        let page_remaining = self.page_len.checked_sub(offset)?;
        let len_remaining = self.len.checked_sub(position)?;
        Some(page_remaining.min(len_remaining))
    }

    #[must_use]
    pub fn slice(&self, start: usize, len: usize) -> Option<&[T]> {
        if len > self.contiguous_len(start)? {
            return None;
        }
        if len == 0 {
            return Some(&[]);
        }

        let (page, offset) = self.page_index(start)?;
        let end = offset.checked_add(len)?;
        self.pages.get(page)?.get(offset..end)
    }

    #[must_use]
    pub fn slice_mut(&mut self, start: usize, len: usize) -> Option<&mut [T]> {
        if len > self.contiguous_len(start)? {
            return None;
        }
        if len == 0 {
            return Some(&mut []);
        }

        let (page, offset) = self.page_index(start)?;
        let end = offset.checked_add(len)?;
        self.pages.get_mut(page)?.get_mut(offset..end)
    }

    fn page_index(&self, position: usize) -> Option<(usize, usize)> {
        if position >= self.len {
            return None;
        }
        Some((
            position.checked_div(self.page_len)?,
            position.checked_rem(self.page_len)?,
        ))
    }
}

impl<T: Clone> PagedStorage<T> {
    #[must_use]
    pub fn new_filled(len: usize, page_len: usize, value: T) -> Option<Self> {
        if page_len == 0 {
            return None;
        }

        let quotient = len.checked_div(page_len)?;
        let remainder = len.checked_rem(page_len)?;
        let page_count = if remainder == 0 {
            quotient
        } else {
            quotient.checked_add(1)?
        };

        let mut pages = Vec::with_capacity(page_count);
        let mut remaining = len;
        for _ in 0..page_count {
            let current_len = remaining.min(page_len);
            let mut page = Vec::with_capacity(page_len);
            page.resize(current_len, value.clone());
            pages.push(page);
            remaining = remaining.checked_sub(current_len)?;
        }

        Some(Self {
            pages,
            page_len,
            len,
        })
    }
}

impl<T> PositionSpace for PagedStorage<T> {
    type Position = usize;

    fn displace(&self, position: &Self::Position, offset: isize) -> Option<Self::Position> {
        displace_position(*position, offset)
    }
}

#[cfg(test)]
mod tests {
    use crate::{PagedStorage, PositionSpace};

    #[test]
    fn construction_rejects_zero_page_length() {
        assert!(PagedStorage::new_filled(4, 0, 10).is_none());
    }

    #[test]
    fn positions_resolve_across_pages() {
        let storage = PagedStorage::new_filled(7, 3, 10).unwrap();

        assert_eq!(storage.len(), 7);
        assert_eq!(storage.page_count(), 3);
        assert_eq!(storage.get(0), Some(&10));
        assert_eq!(storage.get(2), Some(&10));
        assert_eq!(storage.get(3), Some(&10));
        assert_eq!(storage.get(6), Some(&10));
        assert_eq!(storage.get(7), None);
    }

    #[test]
    fn mutable_positions_resolve_across_pages() {
        let mut storage = PagedStorage::new_filled(7, 3, 10).unwrap();

        if let Some(value) = storage.get_mut(4) {
            *value = 40;
        }

        assert_eq!(storage.get(4), Some(&40));
        assert!(storage.get_mut(7).is_none());
    }

    #[test]
    fn displacement_is_position_space_arithmetic_not_access_validation() {
        let storage = PagedStorage::new_filled(7, 3, 10).unwrap();

        assert_eq!(storage.displace(&2, 2), Some(4));
        assert_eq!(storage.displace(&2, 10), Some(12));
        assert_eq!(storage.get(12), None);
    }

    #[test]
    fn contiguous_len_stops_at_page_boundaries() {
        let storage = PagedStorage::new_filled(7, 3, 10).unwrap();

        assert_eq!(storage.contiguous_len(0), Some(3));
        assert_eq!(storage.contiguous_len(1), Some(2));
        assert_eq!(storage.contiguous_len(3), Some(3));
        assert_eq!(storage.contiguous_len(5), Some(1));
        assert_eq!(storage.contiguous_len(6), Some(1));
        assert_eq!(storage.contiguous_len(7), Some(0));
        assert_eq!(storage.contiguous_len(8), None);
    }

    #[test]
    fn slices_do_not_cross_page_boundaries() {
        let storage = PagedStorage::new_filled(7, 3, 10).unwrap();

        assert_eq!(storage.slice(1, 2), Some(&[10, 10][..]));
        assert_eq!(storage.slice(1, 3), None);
        assert_eq!(storage.slice(3, 3), Some(&[10, 10, 10][..]));
        assert_eq!(storage.slice(5, 2), None);
        assert_eq!(storage.slice(6, 1), Some(&[10][..]));
    }

    #[test]
    fn empty_slices_are_available_at_boundaries() {
        let storage = PagedStorage::new_filled(7, 3, 10).unwrap();

        assert_eq!(storage.slice(0, 0), Some(&[][..]));
        assert_eq!(storage.slice(7, 0), Some(&[][..]));
        assert_eq!(storage.slice(8, 0), None);
    }

    #[test]
    fn mutable_slices_do_not_cross_page_boundaries() {
        let mut storage = PagedStorage::new_filled(7, 3, 10).unwrap();

        assert!(storage.slice_mut(1, 3).is_none());
        if let Some(slice) = storage.slice_mut(3, 3) {
            assert_eq!(slice.len(), 3);
            slice[0] = 30;
            slice[2] = 50;
        }

        assert_eq!(storage.get(3), Some(&30));
        assert_eq!(storage.get(5), Some(&50));
    }

    #[test]
    fn empty_storage_has_boundary_but_no_elements() {
        let storage = PagedStorage::new_filled(0, 3, 10).unwrap();

        assert_eq!(storage.len(), 0);
        assert_eq!(storage.page_count(), 0);
        assert_eq!(storage.get(0), None);
        assert_eq!(storage.contiguous_len(0), Some(0));
        assert_eq!(storage.slice(0, 0), Some(&[][..]));
        assert_eq!(storage.contiguous_len(1), None);
    }
}
