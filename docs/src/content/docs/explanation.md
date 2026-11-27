---
title: Explanation
---

# Mistakes and Panics
The `ndarray` library is designed to minimize panics, instead reporting errors through return values or optional types.
There are three main categories of errors that ndarray handles:
1. **User input errors**: These are errors that occur when a user provides invalid data or inputs, such as a mismatched shape or out-of-bound indices. These kinds of errors should never panic, and instead return `Option` or `Result` types.
2. **Internal invariant violations**: These are bugs in the implementation of the library. These _may_ panic, but should be reported as bugs.
3. **Trait contract violations**: These are errors that occur when a trait's contract is violated, such as a trait that is not implemented correctly. This breaks down into two cases:
    - **Safe trait violations**: These are contract violations on safe traits. These may panic, but should be reported as bugs.
    - **Unsafe trait violations**: These are contract violations on unsafe traits. When detected, these violations _must_ panic.

So, in short: panics in the library should almost always be considered bugs and reported as such.

# Safety Invariants
The `ndarray` library relies on maintaining the following invariants.
Any trait that requires these invariants to be upheld must be marked as unsafe, and the implementor must ensure that they are upheld.
1. the product of nonzero axis lengths must not exceed `isize::MAX`
2. the maximum absolute movement across the layout, measured in elements, must not exceed `isize::MAX`
3. the same movement measured in bytes must not exceed `isize::MAX`
4. nonempty arrays must access elements strictly inside the backing storage
5. empty arrays may use positions up to one-past-the-end because such pointers can be formed but not dereferenced
6. owned/mutable access requires nonaliasing; shared read-only access may alias
7. raw pointer arithmetic must stay within one allocation or one byte past its end
