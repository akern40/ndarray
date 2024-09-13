//! Array backend trait.
//!
//! The backend trait allows users to create their own data storage
//! while still taking advantage of code written generically for `ndarray`.

pub unsafe trait Backend
{
    /// The element type of this backend.
    type Elem;

    /// The type carried around by array references.
    type Ref;

    /// The type carried around by owning arrays.
    type Owned;

    fn ref_from_owner_offset(owner: &Self::Owned, offset: isize) -> Self::Ref;
}
