use crate::Storage;

pub trait IntoElements: Storage + Sized {
    type Elements: Iterator<Item = Self::Elem>;

    fn into_elements(self) -> Self::Elements;
}
