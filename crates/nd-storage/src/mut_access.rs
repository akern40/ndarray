use core::ops::DerefMut;

use crate::RefAccess;

pub trait MutAccess: RefAccess {
    type Mut<'a>: DerefMut<Target = Self::Elem>
    where
        Self: 'a;

    fn get_mut(&mut self, position: &Self::Position) -> Option<Self::Mut<'_>>;
}

#[cfg(test)]
mod tests {
    use crate::{MutAccess, RefAccess, SliceStorageMut};

    fn replace_at<S>(storage: &mut S, position: usize, value: S::Elem) -> bool
    where
        S: MutAccess<Position = usize>,
    {
        if let Some(mut target) = storage.get_mut(&position) {
            *target = value;
            true
        } else {
            false
        }
    }

    #[test]
    fn generic_mutable_access_works_for_borrowed_slices() {
        let mut data = [10, 20, 30];
        let mut storage = SliceStorageMut::new(&mut data);

        assert!(replace_at(&mut storage, 1, 40));
        assert!(!replace_at(&mut storage, 3, 50));
        assert_eq!(storage.get(&1).copied(), Some(40));
    }
}
