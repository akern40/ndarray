#![no_std]

mod rank;
mod shape;

pub use rank::{ConstRank, DynRank, Rank, Ranked};
pub use shape::{Shape, ShapeN};
