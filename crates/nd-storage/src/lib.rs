#![no_std]

extern crate alloc;

mod paged;
mod position_space;
mod ref_access;
mod slice;

pub use paged::PagedStorage;
pub use position_space::PositionSpace;
pub use ref_access::RefAccess;
pub use slice::{SliceStorage, SliceStorageMut};
