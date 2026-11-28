pub trait PositionSpace {
    type Position;

    fn displace(&self, position: &Self::Position, offset: isize) -> Option<Self::Position>;
}

pub(crate) fn displace_position(position: usize, offset: isize) -> Option<usize> {
    if offset >= 0 {
        position.checked_add(usize::try_from(offset).ok()?)
    } else {
        position.checked_sub(offset.unsigned_abs())
    }
}
