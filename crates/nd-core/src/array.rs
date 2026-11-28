use core::fmt;

use nd_layout::{AxisOps, IndexSpace, Layout, Ranked};
use nd_storage::{MutAccess, RefAccess, Storage};

use crate::storage::validate_storage_layout;

#[derive(Debug, Clone, Copy)]
pub struct ArrayBase<S: Storage, L: Layout> {
    pub(crate) storage: S,
    pub(crate) layout: L,
    pub(crate) zero_position: S::Position,
}

impl<S: Storage, L: Layout> ArrayBase<S, L> {
    pub(crate) const fn new_unchecked(storage: S, layout: L, zero_position: S::Position) -> Self {
        Self {
            storage,
            layout,
            zero_position,
        }
    }

    #[must_use]
    pub const fn storage(&self) -> &S {
        &self.storage
    }

    #[must_use]
    pub const fn layout(&self) -> &L {
        &self.layout
    }

    #[must_use]
    pub const fn zero_position(&self) -> &S::Position {
        &self.zero_position
    }
}

impl<S: RefAccess, L: Layout> ArrayBase<S, L> {
    #[must_use]
    pub fn try_compose(storage: S, layout: L, zero_position: S::Position) -> Option<Self> {
        validate_storage_layout(&storage, &layout, &zero_position)?;
        Some(Self::new_unchecked(storage, layout, zero_position))
    }

    #[must_use]
    pub fn get(&self, index: &L::Index) -> Option<S::Ref<'_>> {
        let offset = self.layout.offset(index)?;
        let position = self.storage.displace(&self.zero_position, offset)?;
        self.storage.get(&position)
    }
}

impl<S: MutAccess, L: Layout> ArrayBase<S, L> {
    #[must_use]
    pub fn get_mut(&mut self, index: &L::Index) -> Option<S::Mut<'_>> {
        let offset = self.layout.offset(index)?;
        let position = self.storage.displace(&self.zero_position, offset)?;
        self.storage.get_mut(&position)
    }
}

impl<S: Storage, L: AxisOps> ArrayBase<S, L> {
    #[must_use]
    pub fn reverse_axes(self) -> ArrayBase<S, L::Reversed> {
        ArrayBase {
            storage: self.storage,
            layout: self.layout.reverse_axes(),
            zero_position: self.zero_position,
        }
    }

    pub fn axis(self, index: usize) -> Option<ArrayAxis<S, L>> {
        (index < self.layout.shape().ndim()).then(|| ArrayAxis {
            view: self,
            axis: index,
        })
    }
}

pub struct ArrayAxis<S: Storage, L: AxisOps> {
    view: ArrayBase<S, L>,
    axis: usize,
}

impl<S, L> fmt::Debug for ArrayAxis<S, L>
where
    S: Storage + fmt::Debug,
    S::Position: fmt::Debug,
    L: AxisOps + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ArrayAxis")
            .field("view", &self.view)
            .field("axis", &self.axis)
            .finish()
    }
}

impl<S: RefAccess, L: AxisOps> ArrayAxis<S, L> {
    pub fn invert(self) -> Option<ArrayBase<S, L::Inverted>> {
        let ArrayBase {
            storage,
            layout: original_layout,
            zero_position: original_zero,
        } = self.view;
        let (layout, translation) = original_layout.invert_axis(self.axis)?;
        let zero_position = storage.displace(&original_zero, translation)?;
        validate_storage_layout(&storage, &layout, &zero_position)?;

        Some(ArrayBase {
            storage,
            layout,
            zero_position,
        })
    }
}

impl<S: Storage, L: Layout> Ranked for ArrayBase<S, L> {
    type NDim = <L::Shape as Ranked>::NDim;

    fn ndim(&self) -> usize {
        self.layout.shape().ndim()
    }
}

impl<S: Storage, L: Layout> IndexSpace for ArrayBase<S, L> {
    type Shape = L::Shape;
    type Index = L::Index;
    type Indices<'a>
        = L::Indices<'a>
    where
        Self: 'a;

    fn shape(&self) -> &Self::Shape {
        self.layout.shape()
    }

    fn contains(&self, index: &Self::Index) -> bool {
        self.layout.contains(index)
    }

    fn indices(&self) -> Self::Indices<'_> {
        self.layout.indices()
    }
}
