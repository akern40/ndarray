# Approximate Equality with `approx`

In Rust, the [`approx`](https://docs.rs/approx/latest/approx) crate provides traits for testing the approximate equality of floating-point types.
`ndarray` implements `approx`'s traits by doing element-by-element comparisons between two arrays.

## Enabling `approx`
This integration is only available after being enabled by running
```shell
cargo add --features approx ndarray
```
in the root of your Rust project's directory.
This will add `ndarray` as a dependency or, if it was already a dependency, just add the `approx` feature.

## Usage
`ndarray` implements the following traits from `approx`:
- `AbsDiffEq`
- `RelativeEq`
- `UlpsEq`

For convenience, `ndarray` provides two "forwarding" methods for `approx`: `::abs_diff_eq` and `::relative_eq`.
