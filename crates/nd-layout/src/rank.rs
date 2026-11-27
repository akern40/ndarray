use core::fmt::Debug;

pub trait Rank: Copy + Eq + Debug + Send + Sync {
    const N: Option<usize>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstRank<const N: usize>;

impl<const N: usize> Rank for ConstRank<N> {
    const N: Option<usize> = Some(N);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DynRank;

impl Rank for DynRank {
    const N: Option<usize> = None;
}

pub trait Ranked {
    type NDim: Rank;

    #[must_use]
    fn ndim(&self) -> usize;
}

impl<const N: usize> Ranked for [usize; N] {
    type NDim = ConstRank<N>;

    fn ndim(&self) -> usize {
        N
    }
}

impl Ranked for [usize] {
    type NDim = DynRank;

    fn ndim(&self) -> usize {
        self.len()
    }
}
