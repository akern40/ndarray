use crate::PositionSpace;

pub trait Storage: PositionSpace {
    type Elem;
}
