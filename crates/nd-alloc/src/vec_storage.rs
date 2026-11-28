use alloc::vec::Vec;

use nd_storage::{IntoElements, MutAccess, PositionSpace, RefAccess, Storage};

use crate::linear::displace_in_linear_space;

#[derive(Debug)]
pub struct VecStorage<T> {
    data: Vec<T>,
}

impl<T> VecStorage<T> {
    #[must_use]
    pub const fn new(data: Vec<T>) -> Self {
        Self { data }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[must_use]
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }
}

impl<T> From<Vec<T>> for VecStorage<T> {
    fn from(data: Vec<T>) -> Self {
        Self::new(data)
    }
}

impl<T> PositionSpace for VecStorage<T> {
    type Position = usize;

    fn displace(&self, position: &Self::Position, offset: isize) -> Option<Self::Position> {
        displace_in_linear_space(*position, offset, self.capacity())
    }
}

impl<T> Storage for VecStorage<T> {
    type Elem = T;
}

impl<T> RefAccess for VecStorage<T> {
    type Ref<'s>
        = &'s T
    where
        Self: 's;

    fn get(&self, position: &Self::Position) -> Option<Self::Ref<'_>> {
        self.data.get(*position)
    }
}

impl<T> MutAccess for VecStorage<T> {
    type Mut<'s>
        = &'s mut T
    where
        Self: 's;

    fn get_mut(&mut self, position: &Self::Position) -> Option<Self::Mut<'_>> {
        self.data.get_mut(*position)
    }
}

impl<T> IntoElements for VecStorage<T> {
    type Elements = alloc::vec::IntoIter<T>;

    fn into_elements(self) -> Self::Elements {
        self.data.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use nd_storage::{IntoElements, MutAccess, PositionSpace, RefAccess};

    use crate::VecStorage;

    #[test]
    fn positions_access_and_mutate_elements() {
        let mut storage = VecStorage::from(vec![10, 20, 30]);

        assert_eq!(storage.displace(&2, 1), Some(3));
        assert_eq!(storage.get(&1), Some(&20));
        if let Some(value) = storage.get_mut(&1) {
            *value = 40;
        }
        assert_eq!(storage.get(&1), Some(&40));
    }

    #[test]
    fn consumption_yields_each_element_in_storage_order() {
        let storage = VecStorage::from(vec![10, 20, 30]);

        assert!(storage.into_elements().eq([10, 20, 30]));
    }
    #[test]
    fn uninitialized_capacity_is_positional_but_not_dereferenceable() {
        let mut data = vec![10, 20, 30];
        data.reserve(2);
        let storage = VecStorage::from(data);
        let boundary = storage.capacity();

        assert!(boundary > storage.len());
        assert_eq!(storage.displace(&3, 0), Some(3));
        assert_eq!(storage.get(&3), None);
        assert_eq!(storage.displace(&boundary, 0), Some(boundary));
        assert_eq!(storage.displace(&boundary, 1), None);
    }
}
