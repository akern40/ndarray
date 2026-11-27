#![no_std]

mod axis;
mod index_space;
mod layout;
mod rank;
mod shape;
mod strided;

pub use axis::{Axis, AxisOps, AxisSelection};
pub use index_space::{IndexSpace, RowMajorIndices};
pub use layout::{Layout, LogicalOffsets};
pub use rank::{ConstRank, DynRank, Rank, Ranked};
pub use shape::{Shape, ShapeN};
pub use strided::{StridedLayout, StridedOffsets};
