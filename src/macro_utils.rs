/// This assertion is always enabled but only verbose (formatting when
/// debug assertions are enabled).
#[cfg(debug_assertions)]
macro_rules! ndassert {
    ($e:expr, $($t:tt)*) => { assert!($e, $($t)*) };
}

#[cfg(not(debug_assertions))]
macro_rules! ndassert {
    ($e:expr, $($_ignore:tt)*) => {
        assert!($e)
    };
}

macro_rules! expand_if {
    (@bool [true] $($body:tt)*) => { $($body)* };
    (@bool [false] $($body:tt)*) => { };
    (@nonempty [$($if_present:tt)+] $($body:tt)*) => {
        $($body)*
    };
    (@nonempty [] $($body:tt)*) => { };
}

// Macro to insert more informative out of bounds message in debug builds
#[cfg(debug_assertions)]
macro_rules! debug_bounds_check {
    ($self_:ident, $index:expr) => {
        if $index.index_checked(&$self_._dim(), &$self_._strides()).is_none() {
            panic!(
                "ndarray: index {:?} is out of bounds for array of shape {:?}",
                $index,
                $self_.shape()
            );
        }
    };
}

#[cfg(not(debug_assertions))]
macro_rules! debug_bounds_check {
    ($self_:ident, $index:expr) => {};
}

#[cfg(debug_assertions)]
macro_rules! debug_bounds_check_ref {
    ($self_:ident, $index:expr) => {
        if $index.index_checked(&$self_._dim(), &$self_._strides()).is_none() {
            panic!(
                "ndarray: index {:?} is out of bounds for array of shape {:?}",
                $index,
                $self_.shape()
            );
        }
    };
}

#[cfg(not(debug_assertions))]
macro_rules! debug_bounds_check_ref {
    ($self_:ident, $index:expr) => {};
}
