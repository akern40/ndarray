use alloc::boxed::Box;
use alloc::vec::Vec;
use core::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};
use core::ops::{Deref, DerefMut, Index, IndexMut};
use core::slice::Iter;
use num_traits::Zero;

use crate::layout::rank::DynRank;
use crate::layout::ranked::Ranked;
use crate::layout::shape::{IntoShape, Shape};
use crate::Axis;

const CAP: usize = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DynAxesRepr<T>
{
    Inline(usize, [T; CAP]),
    Alloc(Box<[T]>),
}

pub type DShape = DynAxesRepr<usize>;

impl<T> DynAxesRepr<T>
{
    fn ndim(&self) -> usize
    {
        match self {
            DynAxesRepr::Inline(len, _) => *len,
            DynAxesRepr::Alloc(items) => items.len(),
        }
    }
}

impl<T> Deref for DynAxesRepr<T>
{
    type Target = [T];

    fn deref(&self) -> &Self::Target
    {
        match self {
            DynAxesRepr::Inline(len, arr) => {
                debug_assert!(*len <= arr.len());
                unsafe { arr.get_unchecked(..*len) }
            }
            DynAxesRepr::Alloc(items) => items,
        }
    }
}

impl<T> DerefMut for DynAxesRepr<T>
{
    fn deref_mut(&mut self) -> &mut Self::Target
    {
        todo!()
    }
}

impl<T> Index<usize> for DynAxesRepr<T>
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output
    {
        &(**self)[index]
    }
}

impl<T> IndexMut<usize> for DynAxesRepr<T>
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output
    {
        &mut (**self)[index]
    }
}

impl<T> Index<Axis> for DynAxesRepr<T>
{
    type Output = T;

    fn index(&self, index: Axis) -> &Self::Output
    {
        self.index(index.0)
    }
}

impl<T> IndexMut<Axis> for DynAxesRepr<T>
{
    fn index_mut(&mut self, index: Axis) -> &mut Self::Output
    {
        self.index_mut(index.0)
    }
}

impl<T> Default for DynAxesRepr<T>
where T: Zero + Copy
{
    fn default() -> Self
    {
        Self::Inline(1, [T::zero(); 4])
    }
}

impl Zero for DShape
{
    fn zero() -> Self
    {
        Self::Inline(1, [0usize; CAP])
    }

    fn is_zero(&self) -> bool
    {
        self.iter().all(|e| e.is_zero())
    }

    fn set_zero(&mut self)
    {
        self.iter_mut().for_each(|e| e.set_zero());
    }
}

impl<Rhs, T> From<Rhs> for DynAxesRepr<T>
where
    Rhs: AsRef<[T]>,
    T: Default + Copy,
{
    fn from(value: Rhs) -> Self
    {
        let value = value.as_ref();
        let n = value.len();
        if n <= CAP {
            let mut inline = [T::default(); CAP];
            inline.split_at_mut(n).0.copy_from_slice(value);
            Self::Inline(n, inline)
        } else {
            Self::Alloc(value.into())
        }
    }
}

impl<T> Ranked for DynAxesRepr<T>
{
    type NDim = DynRank;

    fn ndim(&self) -> usize
    {
        match self {
            DynAxesRepr::Inline(d, _) => d.clone(),
            DynAxesRepr::Alloc(items) => items.len(),
        }
    }
}

macro_rules! impl_op {
    ($op_trait:ty, $op_fn:ident, $op_assign_trait:ty, $op_assign_fn:ident) => {
        /// *Panics* if the two dimensionalities are different
        impl<Rhs> $op_trait for DShape
        where
            Rhs: IntoShape<NDim = DynRank>,
        {
            type Output = DShape;

            fn $op_fn(self, rhs: Rhs) -> <Self as $op_trait>::Output {
                let mut output = self.clone();
                output.$op_assign_fn(rhs);
                output
            }
        }

        /// *Panics* if the two dimensionalities are different
        impl<Rhs> $op_assign_trait for DShape
        where
            Rhs: IntoShape<NDim = DynRank>,
        {
            fn $op_assign_fn(&mut self, rhs: Rhs) {
                let other = rhs.into_shape();
                for i in 0..self.ndim().max(other.ndim()) {
                    self[i].$op_assign_fn(other[i]);
                }
            }
        }
    };
}

impl_op!(Add<Rhs>, add, AddAssign<Rhs>, add_assign);
impl_op!(Sub<Rhs>, sub, SubAssign<Rhs>, sub_assign);
impl_op!(Mul<Rhs>, mul, MulAssign<Rhs>, mul_assign);

impl<T> IntoIterator for DynAxesRepr<T>
where T: Clone
{
    type Item = T;
    type IntoIter = alloc::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter
    {
        match self {
            DynAxesRepr::Inline(len, arr) => Vec::from(arr[..len].to_vec()).into_iter(),
            DynAxesRepr::Alloc(b) => b.into_vec().into_iter(),
        }
    }
}

impl Shape for DynAxesRepr<usize>
{
    type Iter<'a>
        = Iter<'a, usize>
    where Self: 'a;

    fn iter(&self) -> Self::Iter<'_>
    {
        (**self).iter()
    }
}
