# Overview

The `ndarray` crate provides multidimensional containers in Rust.
`ndarray` supports [owned arrays](explain/types.md#owned-arrays), [views](explain/types.md#array-views), and [references](explain/types.md#array-references), [slicing](how-to/slice.md), [iteration](how-to/iterate.md), [broadcasting](how-to/broadcast.md) and more.

## Installing
`ndarray` is installed using [Cargo](https://doc.rust-lang.org/cargo/), Rust's package manager.
Add `ndarray` as a dependency by navigating to your project directory and running
``` shell
>>> cargo add ndarray
    Updating crates.io index
      Adding ndarray v0.17.0 to dependencies
             Features:
             + std
             - approx
             - blas
             - matrixmultiply-threading
             - portable-atomic-critical-section
             - rayon
             - serde
    ...
```
The `+` sign next to `std` indicates that the `std` feature is turned on; this is true by default.
The `-` signs are extra features that `ndarray` has; they're discussed more in the Feature Flags sections.

## Getting Started
Take a look at the [quickstart](quickstart.md) for an overview of common operations.
If you're coming from NumPy, take a look at [`ndarray` for NumPy Users](numpy.md).
