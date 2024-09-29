/// A trait for types that hold references to allocated data.
pub trait Ref
{
    type Elem;
}

/// A trait for types that hold allocations of data for multidimensional arrays.
// TODO: Should `Ref` be an associated type, or a generic?
pub unsafe trait Owner
{
    type Ref: Ref;

    unsafe fn ref_from_offset_unchecked(&self, offset: isize) -> Self::Ref;

    fn ref_from_offset(&self, offset: isize) -> Self::Ref;
}
