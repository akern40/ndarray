pub trait PositionSpace {
    type Position;

    fn displace(&self, position: &Self::Position, offset: isize) -> Option<Self::Position>;
}
