//! Implementations to mimic the [`num`](https://docs.rs/num/0.4.3/num/) crate's capabilities.

use core::num::FpCategory;

use num_integer::{ExtendedGcd, Integer};
use num_traits::{Euclid, Float, Inv, Pow, Signed};
use paste::paste;

use crate::{broadcast::Broadcastable, dimension::broadcast::co_broadcast, Array, ArrayRef, DimMax, Dimension, Zip};

/// Implement functions inside a generic scope that map another function to an array's elements.
///
/// This is useful for implementing numeric functions succinctly.
/// The macro takes three arguments:
///     1. A module identifier, indicating which module the mapping function is from
///     2. A literal, either "ref" to map by reference or "owned" to map by value
///     3. A parenthesized list of tuples, each one is (function_name, output_type)
///
/// # Example
/// ```no_run
/// use num::Float;
///
/// impl<A, D> ArrayRef<A, D>
/// where
///     A: Float + Clone,
///     D: Dimension,
/// {
///     impl_singles!(Float, "owned", ((is_infinite, bool), (is_finite, bool)));
/// }
/// ```
/// expands to the following block:
/// ```no_run
/// use num::Float;
///
/// impl<A, D> ArrayRef<A, D>
/// where
///     A: Float + Clone,
///     D: Dimension,
/// {
///     fn is_infinite(&self) -> Array<bool, D> {
///         self.mapv(Float::is_infinite)
///     }
///
///     fn is_finite(&self) -> Array<bool, D> {
///         self.mapv(Float::is_finite)
///     }
/// }
/// ```
macro_rules! impl_singles {
    ($mod:ident, "ref", ($(($fn:ident, $output:ty)),+)) => {
        $(
            #[must_use = "method returns a new array and does not mutate the original value"]
            fn $fn(&self) -> Array<$output, D> {
                self.map($mod::$fn)
            }
        )+
    };
    ($mod:ident, "owned", ($(($fn:ident, $output:ty)),+)) => {
        $(
            #[must_use = "method returns a new array and does not mutate the original value"]
            fn $fn(&self) -> Array<$output, D> {
                self.mapv($mod::$fn)
            }
        )+
    };
}

/// Implement pairs of immutable and mutable functions inside a generic scope
/// that map another function to an array's elements.
///
/// This is useful for implementing numeric functions succinctly.
/// The macro takes three arguments:
///     1. A module identifier, indicating which module the mapping function is from
///     2. A literal, either "ref" to map by reference or "owned" to map by value
///     3. A parenthesized list of function names from that module
///
/// # Example
/// ```no_run
/// use num::Float;
///
/// impl<A, D> ArrayRef<A, D>
/// where
///     A: Float + Clone,
///     D: Dimension,
/// {
///     impl_pairs!(Float, "owned", (ceil));
/// }
/// ```
/// expands to
/// ```no_run
/// use num::Float;
///
/// impl<A, D> ArrayRef<A, D>
/// where
///     A: Float + Clone,
///     D: Dimension,
/// {
///     fn ceil(&self) -> Array<A, D> {
///         self.mapv(Float::ceil)
///     }
///
///     fn ceil_assign(&mut self) {
///         self.mapv_inplace(Float::ceil)
///     }
/// }
/// ```
macro_rules! impl_pairs {
    ($mod:ident, "ref", ($($fn:ident),+)) => {
        impl_singles!($mod, "ref", ($(($fn, A)),+));
        $(
            paste! {
                fn [<$fn _assign>](&mut self) {
                    self.map_inplace(|v| *v = $mod::$fn(v))
                }
            }
        )+
    };
    ($mod:ident, "owned", ($($fn:ident),+)) => {
        impl_singles!($mod, "owned", ($(($fn, A)),+));
        $(
            paste! {
                fn [<$fn _assign>](&mut self) {
                    self.mapv_inplace($mod::$fn)
                }
            }
        )+
    };
}

/// Implement functions inside a generic scope that map another function to an array's elements,
/// with an additional argument that can be a scalar or an array.
///
/// This is useful for implementing numeric functions succinctly.
/// The macro takes three arguments:
///     1. A module identifier, indicating which module the mapping function is from
///     2. A literal, either "ref" to map by reference or "owned" to map by value
///     3. A parenthesized list of (function_name, argument_name, output_type)
///
/// This macro makes heavy use of [`Broadcastable`]; see its documentation for more details.
///
/// # Example
/// ```no_run
/// use num::Integer;
///
/// impl<A, D, T, const ARR: bool> ArrayRef<A, D>
/// where
///     A: Integer + Clone,
///     D: Dimension,
///     T: Broadcastable<A, ARR>,
/// {
///     impl_broadcast_singles!(Integer, "ref", (is_multiple_of, other, bool));
/// }
/// ```
/// expands to
/// ```no_run
/// use num::Integer;
///
/// impl<A, D, T, const ARR: bool> ArrayRef<A, D>
/// where
///     A: Integer + Clone,
///     D: Dimension,
///     T: Broadcastable<A, ARR>,
/// {
///     fn is_multiple_of(&self, other: &T) -> Array<bool, D> {
///         Zip::from(self)
///             .and(other.maybe_broadcast(self.raw_dim()).unwrap())
///             .map_collect(Integer::is_multiple_of)
///     }
/// }
/// ```
///
/// # Panics
/// Functions created by this macro will panic when the additional argument is an array
/// that is not broadcastable-compatible (i.e., has an incompatible shape) with the
/// `self` array.
/// ```
macro_rules! impl_broadcast_singles {
    ($mod:ident, "ref", ($(($fn:ident, $arg:ident, $output:ty)),+)) => {
        $(
            #[must_use = "method returns a new array and does not mutate the original value"]
            fn $fn<B, const ARR: bool>(&self, $arg: &B) -> Array<$output, <D as DimMax<B::PseudoDim>>::Output>
            where
                B: Broadcastable<A, ARR>,
                D: DimMax<B::PseudoDim>
            {
                let shape = co_broadcast::<_, _, <D as DimMax<B::PseudoDim>>::Output>(
                    &self.raw_dim(), &$arg.broadcast_dim()
                ).unwrap();
                Zip::from(self.broadcast(shape.clone()).unwrap())
                    .and($arg.maybe_broadcast(shape).unwrap())
                    .map_collect($mod::$fn)
            }
        )+
    };
    ($mod:ident, "owned", ($(($fn:ident, $arg:ident, $output:ty)),+)) => {
        $(
            #[must_use = "method returns a new array and does not mutate the original value"]
            fn $fn<B, const ARR: bool>(&self, $arg: &B) -> Array<$output, <D as DimMax<B::PseudoDim>>::Output>
            where
                B: Broadcastable<A, ARR>,
                D: DimMax<B::PseudoDim>
            {
                let shape = co_broadcast::<_, _, <D as DimMax<B::PseudoDim>>::Output>(
                    &self.raw_dim(), &$arg.broadcast_dim()
                ).unwrap();
                Zip::from(self.broadcast(shape.clone()).unwrap())
                    .and($arg.maybe_broadcast(shape).unwrap())
                    .map_collect(|s, a| s.$fn(*a))
            }
        )+
    };
}

/// Implement pairs of immutable and mutable functions inside a generic scope that map
/// another function to an array's elements, with an additional argument that can be a scalar or an array.
///
/// This is useful for implementing numeric functions succinctly.
/// The macro takes three arguments:
///     1. A module identifier, indicating which module the mapping function is from
///     2. A literal, either "ref" to map by reference or "owned" to map by value
///     3. A parenthesized list of (function_name, argument_name)
///
/// This macro makes heavy use of [`Broadcastable`]; see its documentation for more details.
///
/// # Example
/// ```no_run
/// use num::Integer;
///
/// impl<A, D, T, const ARR: bool> ArrayRef<A, D>
/// where
///     A: Integer + Clone,
///     D: Dimension,
///     T: Broadcastable<A, ARR>,
/// {
///     impl_broadcast_pairs!(Integer, "ref", (div_mod, other));
/// }
/// ```
/// expands to
/// ```no_run
/// use num::Integer;
///
/// impl<A, D, T, const ARR: bool> ArrayRef<A, D>
/// where
///     A: Integer + Clone,
///     D: Dimension,
///     T: Broadcastable<A, ARR>,
/// {
///     fn div_mod(&self, other: &T) -> Array<bool, D> {
///         Zip::from(self)
///             .and(other.maybe_broadcast(self.raw_dim()).unwrap())
///             .map_collect(Integer::is_multiple_of)
///     }
///
///     fn div_mod_assign(&mut self, other: &T) {
///         self.zip_mut_with(&other.maybe_broadcast(self.raw_dim()).unwrap(), |s, o| {
///             *s = s.div_mod(o)
///         });
///     }
/// }
/// ```
///
/// # Panics
/// Functions created by this macro will panic when the additional argument is an array
/// that is not broadcastable-compatible (i.e., has an incompatible shape) with the
/// `self` array.
/// ```
macro_rules! impl_broadcast_pairs {
    ($mod:ident, "ref", ($(($fn:ident, $arg:ident)),+)) => {
        impl_broadcast_singles!($mod, "ref", ($(($fn, $arg, A)),+));
        $(
            paste! {
                fn [<$fn _assign>]<B, const ARR: bool>(&mut self, $arg: &B)
                where B: Broadcastable<A, ARR>
                {
                    self.zip_mut_with(&$arg.maybe_broadcast(self.raw_dim()).unwrap(), |s, o| {
                        *s = s.$fn(o)
                    });
                }
            }
        )+
    };
    ($mod:ident, "owned", ($(($fn:ident, $arg:ident)),+)) => {
        impl_broadcast_singles!($mod, "owned", ($(($fn, $arg, A)),+));
        $(
            paste! {
                fn [<$fn _assign>]<B, const ARR: bool>(&mut self, $arg: &B)
                where B: Broadcastable<A, ARR>
                {
                    self.zip_mut_with(&$arg.maybe_broadcast(self.raw_dim()).unwrap(), |s, o| {
                        *s = s.$fn(*o)
                    });
                }
            }
        )+
    };
}

/// Functions that forward to [`num_traits::Signed`]
impl<A, D> ArrayRef<A, D>
where
    A: Signed + Clone,
    D: Dimension,
{
    impl_pairs!(Signed, "ref", (abs, signum));
    impl_singles!(Signed, "ref", ((is_positive, bool), (is_negative, bool)));
    impl_broadcast_pairs!(Signed, "ref", ((abs_sub, other)));
}

/// Functions that forward to [`num_traits::Pow`]
impl<A, D> ArrayRef<A, D>
where D: Dimension
{
    fn pow<B, C, const ARR: bool>(&self, rhs: &C) -> Array<A::Output, <D as DimMax<C::PseudoDim>>::Output>
    where
        A: Pow<B> + Clone,
        B: Clone,
        C: Broadcastable<B, ARR>,
        D: DimMax<C::PseudoDim>,
    {
        let shape =
            co_broadcast::<_, _, <D as DimMax<C::PseudoDim>>::Output>(&self.raw_dim(), &rhs.broadcast_dim()).unwrap();
        Zip::from(self.broadcast(shape.clone()).unwrap())
            .and(rhs.maybe_broadcast(shape).unwrap())
            .map_collect(|s, r| s.clone().pow(r.clone()))
    }

    fn pow_assign<B, C, const ARR: bool>(&mut self, rhs: &C)
    where
        A: Pow<B, Output = A> + Clone,
        B: Clone,
        C: Broadcastable<B, ARR>,
    {
        self.zip_mut_with(&rhs.maybe_broadcast(self.raw_dim()).unwrap(), |s, r| *s = s.clone().pow(r.clone()));
    }
}

/// Functions that forward to [`num_traits::Float`]
impl<A, D> ArrayRef<A, D>
where
    A: Float,
    D: Dimension,
{
    impl_pairs!(
        Float,
        "owned",
        (
            floor, ceil, round, trunc, fract, recip, sqrt, exp, exp2, ln, log2, log10, cbrt, sin,
            cos, tan, asin, acos, atan, exp_m1, ln_1p, sinh, cosh, tanh, asinh, acosh, atanh
        )
    );
    impl_singles!(
        Float,
        "owned",
        (
            (is_nan, bool),
            (is_infinite, bool),
            (is_finite, bool),
            (is_normal, bool),
            (classify, FpCategory),
            (integer_decode, (u64, i16, i8)),
            (sin_cos, (A, A))
        )
    );
    impl_broadcast_pairs!(
        Float,
        "owned",
        (
            (powf, n),
            (log, base),
            (max, other),
            (min, other),
            (hypot, other),
            (atan2, other)
        )
    );

    fn mul_add<B, T, const ARR: bool, const ARR2: bool>(
        &self, a: &B, b: &T,
    ) -> Array<A, <<D as DimMax<B::PseudoDim>>::Output as DimMax<T::PseudoDim>>::Output>
    where
        B: Broadcastable<A, ARR>,
        T: Broadcastable<A, ARR2>,
        D: DimMax<B::PseudoDim>,
        <D as DimMax<B::PseudoDim>>::Output: DimMax<T::PseudoDim>,
    {
        let shape =
            co_broadcast::<_, _, <D as DimMax<B::PseudoDim>>::Output>(&self.raw_dim(), &a.broadcast_dim()).unwrap();
        let shape: <<D as DimMax<B::PseudoDim>>::Output as DimMax<T::PseudoDim>>::Output =
            co_broadcast(&shape, &b.broadcast_dim()).unwrap();
        Zip::from(self.broadcast(shape.clone()).unwrap())
            .and(&a.maybe_broadcast(shape.clone()).unwrap())
            .and(&b.maybe_broadcast(shape).unwrap())
            .map_collect(|s, a, b| s.mul_add(*a, *b))
    }

    fn mul_add_assign<B, C, const ARR: bool, const ARR2: bool>(&mut self, a: &B, b: &C)
    where
        B: Broadcastable<A, ARR>,
        C: Broadcastable<A, ARR2>,
    {
        let shape = self.raw_dim();
        Zip::from(self)
            .and(&a.maybe_broadcast(shape.clone()).unwrap())
            .and(&b.maybe_broadcast(shape).unwrap())
            .map_collect(|s, a, b| s.mul_add(*a, *b));
    }

    fn powi<B, const ARR: bool>(&self, n: &B) -> Array<A, <D as DimMax<B::PseudoDim>>::Output>
    where
        B: Broadcastable<i32, ARR>,
        D: DimMax<B::PseudoDim>,
    {
        let shape: <D as DimMax<B::PseudoDim>>::Output = co_broadcast(&self.raw_dim(), &n.broadcast_dim()).unwrap();
        Zip::from(self.broadcast(shape.clone()).unwrap())
            .and(&n.maybe_broadcast(shape).unwrap())
            .map_collect(|s, n| s.powi(*n))
    }

    fn powi_assign<B, const ARR: bool>(&mut self, n: &B)
    where B: Broadcastable<i32, ARR>
    {
        self.zip_mut_with(&n.maybe_broadcast(self.raw_dim()).unwrap(), |s, n| *s = s.powi(*n));
    }
}

/// Functions that forward to [`num_integer::Integer`]
impl<A, D> ArrayRef<A, D>
where
    A: Integer + Clone,
    D: Dimension,
{
    impl_singles!(Integer, "ref", ((is_even, bool), (is_odd, bool)));

    fn dec(&mut self)
    where Self: Clone
    {
        self.map_inplace(A::dec);
    }

    fn inc(&mut self)
    where Self: Clone
    {
        self.map_inplace(A::inc);
    }

    impl_broadcast_pairs!(
        Integer,
        "ref",
        (
            (div_floor, other),
            (mod_floor, other),
            (gcd, other),
            (lcm, other),
            (div_ceil, other),
            (next_multiple_of, other),
            (prev_multiple_of, other)
        )
    );
    impl_broadcast_singles!(
        Integer,
        "ref",
        (
            (is_multiple_of, other, bool),
            (div_rem, other, (A, A)),
            (gcd_lcm, other, (A, A)),
            (div_mod_floor, other, (A, A)),
            (extended_gcd, other, ExtendedGcd<A>)
        )
    );

    fn extended_gcd_lcm<B, const ARR: bool>(
        &self, other: &B,
    ) -> Array<(ExtendedGcd<A>, A), <D as DimMax<B::PseudoDim>>::Output>
    where
        Self: Clone + Signed,
        B: Broadcastable<A, ARR>,
        D: DimMax<B::PseudoDim>,
    {
        let shape: <D as DimMax<B::PseudoDim>>::Output = co_broadcast(&self.raw_dim(), &other.broadcast_dim()).unwrap();
        Zip::from(
            self.broadcast(shape.clone())
                .expect("Shape derived from co_broadcast should be ok"),
        )
        .and(
            &other
                .maybe_broadcast(shape)
                .expect("Shape derived from co_broadcast should be ok"),
        )
        .map_collect(|s, o| (s.extended_gcd(o), s.lcm(o)))
    }
}

/// Functions that forward to [`num_traits::Euclid`]
impl<A, D> ArrayRef<A, D>
where
    A: Euclid + Clone,
    D: Dimension,
{
    impl_broadcast_pairs!(Euclid, "ref", ((div_euclid, v), (rem_euclid, v)));
    impl_broadcast_singles!(Euclid, "ref", ((div_rem_euclid, v, (A, A))));
}

/// Functions that forward to [`num_traits::Inv`]
impl<A, D> ArrayRef<A, D>
where
    A: Inv + Clone,
    D: Dimension,
{
    fn inv(&self) -> Array<A::Output, D>
    {
        self.mapv(Inv::inv)
    }
}
