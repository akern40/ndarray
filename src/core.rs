//! The module declaration for an eventual `ndarray-core` crate.

mod traits;
mod array;
mod arrayref;

pub use array::*;
pub use arrayref::*;
pub use traits::*;

/// The core types of `ndarray`, redefined in terms of the new infrastructure.
/// 
/// This is a separate module to make it clear that its components will not be
/// extracted into `ndarray-core`.
pub mod ndcore {
    use core::ptr::NonNull;

    use crate::{OwnedRepr, RawData};

    use super::{Owner, RawRefBase, Ref, RefBase};

    impl<T> Ref for NonNull<T> {
        type Elem = T;
    }

    pub struct Layout<D> {
        pub(crate) dim: D,
        pub(crate) strides: D,
    }

    // pub trait HeapOwner<T>: Owner<Ref = NonNull<T>> {}

    unsafe impl<T> Owner for OwnedRepr<T> {
        type Ref = NonNull<T>;

        unsafe fn ref_from_offset_unchecked(&self, offset: isize) -> Self::Ref {
            self.as_nonnull().offset(offset)
        }

        fn ref_from_offset(&self, offset: isize) -> Self::Ref {
            todo!("Check for in-bounds!");
            // unsafe { self.ref_from_offset_unchecked(offset) }
        }
    }

    pub type ArrayBase<S, D> = crate::core::array::ArrayBase<Layout<D>, NonNull<<S as RawData>::Elem>, S>;
    pub type RawViewBase<S, D> = crate::core::array::RawArrayViewBase<Layout<D>, NonNull<<S as RawData>::Elem>, S>;

    // impl<T> HeapOwner<T> for OwnedRepr<T> {}

    // /// This type should act almost entirely equivalently to the existing `Array`
    // pub type Array<A, D> = OwnedBase<Layout<D>, OwnedRepr<A>>;

    // /// A very simple Uniqueable, since this type is always unique.
    // unsafe impl<A, D> Uniqueable for Array<A, D> {
    //     fn try_ensure_unique(&mut self) {}

    //     fn try_is_unique(&self) -> Option<bool> {
    //         Some(true)
    //     }
    // }

    // /// A blanket NdArray implementation for any backend whose reference type is `NonNull<A>`.
    // ///
    // /// As described in the `arrayref` module of `core`, this kind of reference-focused `impl`
    // /// is very ergonomic and makes sense, since most behavior relies on the reference type,
    // /// not the owned type.
    // unsafe impl<L, O, A> NdArray<O::Ref> for OwnedBase<L, O>
    // where
    //     O: Owner<Ref = NonNull<A>>,
    // {
    //     fn as_ptr(&self) -> *mut <<O as Owner>::Ref as Ref>::Elem {
    //         self.storage().as_ptr()
    //     }
    // }

    // impl<T> Owner for Arc<OwnedRepr<T>> {
    //     type Ref = NonNull<T>;

    //     unsafe fn ref_from_offset_unchecked(&self, offset: isize) -> Self::Ref {
    //         self.as_nonnull().offset(offset)
    //     }

    //     fn ref_from_offset(&self, offset: isize) -> Self::Ref {
    //         todo!("Check for in-bounds!");
    //         // unsafe { self.ref_from_offset_unchecked(offset) }
    //     }
    // }

    // impl<T> HeapOwner<T> for Arc<OwnedRepr<T>> {}

    // /// Again, should be almost entirely equivalent to the existing `ArcArray`
    // pub type ArcArray<A, D> = OwnedBase<D, Arc<OwnedRepr<A>>>;

    // // Uniqueable implementation, with the uniqueness logic stolen from what already exists
    // unsafe impl<A, D: Dimension> Uniqueable for ArcArray<A, D> {
    //     fn try_ensure_unique(&mut self) {
    //         if Arc::get_mut(self.own_mut()).is_some() {
    //             return;
    //         }
    //         if self.layout().size() <= self.own().len() / 2 {
    //             todo!(".to_owned().to_shared()");
    //         }
    //         let ptr = self.as_ptr();
    //         let rcvec = &mut self.own();
    //         let a_size = mem::size_of::<A>() as isize;
    //         let our_off = if a_size != 0 {
    //             (ptr as isize - Arc::as_ptr(&rcvec) as isize) / a_size
    //         } else {
    //             0
    //         };
    //         self.with_reference(self.own().ref_from_offset(our_off));
    //     }

    //     // We're using strong count here so that we don't need &mut self.
    //     // This is not as strong as the original `try_is_unique`, but it doesn't matter.
    //     // The original does not hold on to the mutable reference provided by [`Arc::get_mut`],
    //     // so the moment the call is over the uniqueness guarantee does not hold.
    //     // This is a weaker guarantee of instantaneous uniqueness, but neither this nor the
    //     // original implementation are good enough to safely do anything with that uniqueness anyway.
    //     fn try_is_unique(&self) -> Option<bool> {
    //         Some(Arc::strong_count(&self.own()) == 1)
    //     }
    // }

    // impl<'a, T: Clone> Owner for Cow<'a, OwnedRepr<T>> {
    //     type Ref = NonNull<T>;
    
    //     unsafe fn ref_from_offset_unchecked(&self, offset: isize) -> Self::Ref {
    //         self.as_nonnull().offset(offset)
    //     }
    
    //     fn ref_from_offset(&self, offset: isize) -> Self::Ref {
    //         todo!("Check for in-bounds!");
    //         // unsafe { self.ref_from_offset_unchecked(offset) }
    //     }
    // }

    // /// These aliases work for a view into any array whose reference type is `NonNull`:

    // pub type ArrayView<'a, A, D> = ViewBase<'a, D, NonNull<A>>;
    // pub type ArrayViewMut<'a, A, D> = ViewBaseMut<'a, D, NonNull<A>>;
    // pub type RawArrayView<A, D> = RawViewBase<D, NonNull<A>>;
    // pub type RawArrayViewMut<A, D> = RawViewBaseMut<D, NonNull<A>>;
    pub type RawArrayRef<A, D> = RawRefBase<D, NonNull<A>>;
    pub type ArrayRef<A, D> = RefBase<D, NonNull<A>>;
}
