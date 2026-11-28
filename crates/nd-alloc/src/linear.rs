pub(crate) fn displace_in_linear_space(
    position: usize,
    offset: isize,
    len: usize,
) -> Option<usize> {
    if position > len {
        return None;
    }
    let position = if offset >= 0 {
        position.checked_add(usize::try_from(offset).ok()?)?
    } else {
        position.checked_sub(offset.unsigned_abs())?
    };
    (position <= len).then_some(position)
}
