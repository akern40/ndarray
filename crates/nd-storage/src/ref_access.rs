use core::ops::Deref;

use crate::Storage;

pub trait RefAccess: Storage {
    type Ref<'a>: Deref<Target = Self::Elem>
    where
        Self: 'a;

    fn get(&self, position: &Self::Position) -> Option<Self::Ref<'_>>;
}

#[cfg(test)]
mod tests {
    use crate::{RefAccess, SliceStorage, SliceStorageMut};

    fn cloned_at<S>(storage: &S, position: usize) -> Option<S::Elem>
    where
        S: RefAccess<Position = usize>,
        S::Elem: Clone,
    {
        storage.get(&position).map(|value| (*value).clone())
    }

    #[test]
    fn shared_slice_references_use_the_backing_lifetime() {
        let data = [10, 20, 30];
        let value = {
            let storage = SliceStorage::new(&data);
            RefAccess::get(&storage, &1).unwrap()
        };

        assert_eq!(*value, 20);
    }

    #[test]
    fn generic_access_works_for_borrowed_slice_storage() {
        let data = [10, 20, 30];
        let shared = SliceStorage::new(&data);
        assert_eq!(cloned_at(&shared, 1), Some(20));
        assert_eq!(cloned_at(&shared, 3), None);

        let mut mutable_data = [10, 20, 30];
        let mutable = SliceStorageMut::new(&mut mutable_data);
        assert_eq!(cloned_at(&mutable, 2), Some(30));
    }
}
