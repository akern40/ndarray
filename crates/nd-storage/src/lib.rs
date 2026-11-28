#![no_std]

extern crate alloc;

mod mut_access;
mod paged;
mod position_space;
mod ref_access;
mod slice;

pub use mut_access::MutAccess;
pub use paged::PagedStorage;
pub use position_space::PositionSpace;
pub use ref_access::RefAccess;
pub use slice::{SliceStorage, SliceStorageMut};
