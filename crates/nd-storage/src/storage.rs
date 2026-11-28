pub trait Storage {
    type Elem;
    type Position;

    fn displace(&self, position: &Self::Position, offset: isize) -> Option<Self::Position>;

    fn validate_offset_bounds(
        &self,
        zero: &Self::Position,
        bounds: Option<(isize, isize)>,
    ) -> Option<()>;
}
