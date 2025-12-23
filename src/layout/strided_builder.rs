use crate::layout::dimensionality::Dimensionality;

pub trait StridedBuilder
{
    type Dimality: Dimensionality;
}
