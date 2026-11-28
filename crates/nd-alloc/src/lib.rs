#![no_std]

extern crate alloc;

mod linear;
mod paged;
mod vec_storage;

pub use paged::PagedStorage;
pub use vec_storage::VecStorage;
