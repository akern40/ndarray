# Overview

The `ndarray` crate provides an *n*-dimensional container in Rust.

## Installing
`ndarray` is installed using [Cargo](https://doc.rust-lang.org/cargo/), Rust's package manager.
Add `ndarray` as a dependency by navigating to your project directory and running
``` shell
>>> cargo add ndarray
    Updating crates.io index
      Adding ndarray v0.16.1 to dependencies
             Features:
             + std
             - approx
             - blas
             - matrixmultiply-threading
             - portable-atomic-critical-section
             - rayon
             - serde
    Updating crates.io index
     Locking 10 packages to latest compatible versions
```
The `-` signs are extra features that `ndarray` has; they're discussed more in the Extra Features sections.

## Getting Started
