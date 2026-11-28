use core::mem::size_of;

use nd_layout::{Layout, Shape};

pub(crate) use nd_storage::{PositionSpace, RefAccess, SliceStorage, SliceStorageMut};

pub(crate) fn validate_storage_layout<S: RefAccess, L: Layout>(
    storage: &S,
    layout: &L,
    zero_position: &S::Position,
) -> Option<()> {
    validate_nonzero_axis_product(layout.shape())?;

    match layout.offset_bounds() {
        Some((min, max)) => {
            let element_span = element_span(min, max)?;
            validate_byte_span::<S::Elem>(element_span)?;

            let min_position = storage.displace(zero_position, min)?;
            let max_position = storage.displace(zero_position, max)?;
            storage.get(&min_position)?;
            storage.get(&max_position)?;
        }
        None => {
            storage.displace(zero_position, 0)?;
        }
    }

    Some(())
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
    use nd_layout::ShapeN;

    use super::{element_span, validate_byte_span, validate_nonzero_axis_product};

    #[test]
    fn nonzero_axis_product_must_fit_isize() {
        assert!(validate_nonzero_axis_product(&ShapeN::new([4, 0, 3])).is_some());
        assert!(validate_nonzero_axis_product(&ShapeN::new([usize::MAX, 0])).is_none());
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
}
