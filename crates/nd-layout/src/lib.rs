#![no_std]

mod index_space;
mod layout;
mod rank;
mod shape;
mod strided;

pub use index_space::{IndexSpace, RowMajorIndices};
pub use layout::{Layout, LogicalOffsets};
pub use rank::{ConstRank, DynRank, Rank, Ranked};
pub use shape::{Shape, ShapeN};
pub use strided::{StridedLayout, StridedOffsets};
