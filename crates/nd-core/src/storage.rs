use nd_layout::{Layout, Shape};

pub(crate) use nd_storage::{SliceStorage, SliceStorageMut, Storage};

pub(crate) fn validate_layout<S: Storage, L: Layout>(
    storage: &S,
    layout: &L,
    zero: &S::Position,
) -> Option<()> {
    validate_nonzero_axis_product(layout.shape())?;
    storage.validate_offset_bounds(zero, layout.offset_bounds())
}

fn validate_nonzero_axis_product<S: Shape + ?Sized>(shape: &S) -> Option<()> {
    let mut product = 1_usize;
    for len in shape.iter() {
        if len != 0 {
            product = product.checked_mul(len)?;
            if product > isize::MAX.unsigned_abs() {
                return None;
            }
        }
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use nd_layout::ShapeN;

    use super::validate_nonzero_axis_product;

    #[test]
    fn nonzero_axis_product_must_fit_isize() {
        assert!(validate_nonzero_axis_product(&ShapeN::new([4, 0, 3])).is_some());
        assert!(validate_nonzero_axis_product(&ShapeN::new([usize::MAX, 0])).is_none());
    }
}
