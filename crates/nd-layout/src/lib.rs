#![no_std]
#![forbid(unsafe_code)]

mod rank;

pub use rank::{ConstRank, DynRank, Rank, Ranked};
