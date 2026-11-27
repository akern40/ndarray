# Agent rules

This branch (`next`) is a ground-up rewrite of ndarray as layered crates. It shares no history with
`master`. Build one trait or struct at a time; never port spike code wholesale.

## Environment

This project uses `mise` for provisioning and managing development tools and commands.
Checks, docs, and pinned tools can be run via `mise exec --`.

## Reference Material

Since this branch is a ground-up rewrite, it shares no history with `master`. When designing its
new elements, the following reference sources are important:
- The original `master` branch, as a reference for what ndarray can do now and how it does it
- The `rust-ndarray/ndarray` `refactor` branch, as a reference for previous incremental design work
- The `akern40/ndarray` `layout-spike` branch for some agent-driven experimentation
- The `akern40/ndarray-design` repository, multiple branches, for design notes and exploration, both human-written and agent-generated

## Comments and docs are human-only

Agents never write comments of any kind: no `//`, `///`, `//!`, `/* */` in Rust, no `#` lines in
TOML or YAML, no doc comments, no `// SAFETY:` notes. Never edit, move or delete an existing
comment. Where a lint asks for documentation (`missing_docs`, `missing_errors_doc`,
`missing_panics_doc`, `missing_crate_level_docs`, `undocumented_unsafe_blocks`), leave the warning
in place and list it in your report so the human can write it.

Agents never write documentation prose either: nothing under `docs/src/content/`, no READMEs, no
design notes, no changelog text. Agents may draft comments and docs in the chat; the human writes
them into files.

## Documentation

- The site is Starlight, in `docs/`. Its content follows Diátaxis, one quadrant per page:
  - Tutorials (`tutorials/`): learning by doing, a guided path for a newcomer.
  - How-to guides (`how-to/`): the steps to accomplish one task, for a reader who knows the basics.
  - Reference (`reference/`): dry, complete description; the API reference itself is rustdoc.
  - Explanation (`explanation/`): background, design reasoning and trade-offs. Design notes live
    here.
- When drafting a page in chat, name its quadrant and stay inside it. Material that belongs to
  another quadrant becomes a separate page.
- Rust code shown in the docs lives in real `.rs` files that cargo compiles and runs in CI. Pages
  include whole files, never fragments marked out by comments.

## Commits

- Conventional Commits: `type(scope): subject`, e.g. `feat(nd-layout): add Rank`. Types: `feat`,
  `fix`, `perf`, `refactor`, `test`, `docs`, `build`, `ci`, `chore`. Scopes are crate names.
  Breaking changes use `!` and a `BREAKING CHANGE:` footer.
- `committed` checks every commit in CI, and release-plz builds the changelog from them. Never
  hand-edit a `CHANGELOG.md`.

## Lints are not negotiable

- Never edit `Cargo.toml` lint tables, `clippy.toml`, `rustfmt.toml` or `rust-toolchain.toml` to
  make code pass. Fix the code.
- No `#[allow(...)]`. A `#[expect(lint, reason = "...")]` needs the human's approval first.
- Deny-level lints fail every local build. The documentation lints above are warn-level locally and
  fail only in CI (`-D warnings`), so undocumented code can be iterated on but not merged.

## Verify before reporting

```
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features
cargo nextest run --workspace --all-features --no-tests=warn
cargo test --doc --workspace --all-features
cargo build -p nd-layout --target thumbv7em-none-eabihf
```

Tests run under nextest; doctests run separately because nextest does not run them.

When `docs/` changes, also run `npm --prefix docs ci` and `npm --prefix docs run build`.

Clippy must show no warnings other than the documentation lints listed above.

## Testing

- Property tests with hegel are the default: `#[hegel::test]`, in `tests/` or a `#[cfg(test)]` module.
- Specific behaviours get plain `#[test]`s.
- A shrunk failure becomes a permanent regression as a `#[hegel::explicit_test_case]` with the values written out. `#[hegel::reproduce_failure]` is for local debugging and is never committed.
- Proofs are Kani harnesses: `#[kani::proof]` in a `#[cfg(kani)]` module. Kani runs on Linux and macOS (`cargo kani --workspace`) and in CI, not on Windows.
- Benchmarks use divan (the `codspeed-divan-compat` package, imported as `divan`) in `crates/*/benches/`. `cargo bench` gives wall-clock timings locally; CI runs the same benchmarks under CodSpeed.

## Ground rules

- Stable Rust only, pinned in `rust-toolchain.toml`. Edition 2024.
- No sealed traits; extension points are `unsafe trait`s with documented contracts.
- No dependency without the human's approval.
- Push to `akern40/*` forks freely. Never open PRs, comment or push in the `rust-ndarray` org.
