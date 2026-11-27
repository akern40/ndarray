#![no_std]

mod index_space;
mod rank;
mod shape;

pub use index_space::{IndexSpace, ShapeNIndices};
pub use rank::{ConstRank, DynRank, Rank, Ranked};
pub use shape::{Shape, ShapeN};
