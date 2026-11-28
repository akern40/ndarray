#![no_std]

mod slice;
mod storage;

pub use slice::{SliceStorage, SliceStorageMut};
pub use storage::Storage;
