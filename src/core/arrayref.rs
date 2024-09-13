//! Array reference structures.

use core::{
    fmt::Debug,
    ops::{Deref, DerefMut},
};

/// A raw reference type for arrays.
///
/// Not necessarily aligned or uniquely held, but the offset rules should hold.
///
/// Instead of being parameterized by the full backend, this is just parameterized by
/// `R` for the backend's reference type. Many backends will share one reference
/// type (e.g., all dynamically-allocated CPU-based backends, and probably more).
/// It's much more ergonomic to be able to make blanket implementations based on the
/// reference type than the entire backend.
#[derive(Debug)]
pub struct RawRefBase<L, R>
{
    pub layout: L,
    pub storage: R,
}

/// A reference type for arrays.
///
/// Data is always aligned and, when received as a mutable reference (`&mut RefBase`),
/// uniquely held and therefore safe to mutate.
///
/// See [`RawRefBase`] for why this is parameterized by `R` and not by the backend.
#[derive(Debug)]
pub struct RefBase<L, R>(RawRefBase<L, R>);

impl<L, R> Deref for RefBase<L, R>
{
    type Target = RawRefBase<L, R>;

    fn deref(&self) -> &Self::Target
    {
        &self.0
    }
}

impl<L, R> DerefMut for RefBase<L, R>
{
    fn deref_mut(&mut self) -> &mut Self::Target
    {
        &mut self.0
    }
}

/// Blanket AsRef implementation, as instructed by the Rust documentation.
impl<L, R, T> AsRef<T> for RefBase<L, R>
where
    T: ?Sized,
    <RefBase<L, R> as Deref>::Target: AsRef<T>,
{
    fn as_ref(&self) -> &T
    {
        self.deref().as_ref()
    }
}
