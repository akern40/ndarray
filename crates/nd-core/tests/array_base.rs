use nd_alloc::{PagedStorage, VecStorage};
use nd_core::ArrayBase;
use nd_layout::{ShapeN, StridedLayout};
use nd_storage::MutAccess;

#[test]
fn vec_storage_composes_with_layouts() {
    let layout = StridedLayout::try_new(ShapeN::new([2, 2]), [2, 1]).unwrap();
    let storage = VecStorage::from(vec![0, 1, 2, 3]);
    let array = ArrayBase::try_compose(storage, layout, 0).unwrap();

    assert_eq!(array.get(&[0, 0]), Some(&0));
    assert_eq!(array.get(&[1, 1]), Some(&3));
    assert_eq!(array.get(&[2, 0]), None);
}

#[test]
fn paged_storage_composes_with_layouts() {
    let layout = StridedLayout::try_new(ShapeN::new([2, 2]), [3, 1]).unwrap();
    let mut storage = PagedStorage::new_filled(6, 2, 0).unwrap();
    for position in 0..6 {
        if let Some(value) = storage.get_mut(&position) {
            *value = position;
        }
    }
    let array = ArrayBase::try_compose(storage, layout, 0).unwrap();

    assert_eq!(array.get(&[0, 0]), Some(&0));
    assert_eq!(array.get(&[0, 1]), Some(&1));
    assert_eq!(array.get(&[1, 0]), Some(&3));
    assert_eq!(array.get(&[1, 1]), Some(&4));
}

#[test]
fn mutable_owned_compositions_write_through_layouts() {
    let layout = StridedLayout::try_new(ShapeN::new([2]), [1]).unwrap();
    let storage = VecStorage::from(vec![10, 20]);
    let mut array = ArrayBase::try_compose(storage, layout, 0).unwrap();

    if let Some(value) = array.get_mut(&[1]) {
        *value = 40;
    }

    assert_eq!(array.get(&[1]), Some(&40));
}
