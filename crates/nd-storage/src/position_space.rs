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

pub(crate) fn displace_in_linear_space(
    position: usize,
    offset: isize,
    len: usize,
) -> Option<usize> {
    if position > len {
        return None;
    }
    let position = displace_position(position, offset)?;
    (position <= len).then_some(position)
}
