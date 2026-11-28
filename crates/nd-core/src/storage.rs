use core::mem::size_of;

use nd_layout::{Layout, Shape};

pub(crate) use nd_storage::{SliceStorage, SliceStorageMut};

pub(crate) fn validate_slice_layout<T, L: Layout>(
    len: usize,
    layout: &L,
    origin: usize,
) -> Option<()> {
    validate_nonzero_axis_product(layout.shape())?;

    match layout.offset_bounds() {
        Some((min, max)) => {
            let element_span = element_span(min, max)?;
            validate_byte_span::<T>(element_span)?;

            offset_position(origin, min)?;
            let max_position = offset_position(origin, max)?;
            if max_position >= len {
                return None;
            }
        }
        None if origin > len => return None,
        None => {}
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

pub(crate) fn offset_position(origin: usize, offset: isize) -> Option<usize> {
    if offset >= 0 {
        origin.checked_add(usize::try_from(offset).ok()?)
    } else {
        origin.checked_sub(offset.unsigned_abs())
    }
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
