# `ndarray` Documentation
`ndarray` maintains two kinds of documentation: API docs and a website.
While much of the information overlaps, each has its place.

## API Docs
The API docs are hosted on [crates.io](https://docs.rs/ndarray/latest/ndarray/) and are generated from doc comments embedded in the `ndarray` source code.
As a result, any documentation that is specific to a particular struct, trait, function, or method will appear there.
These docs are the "reference" material: they try to succinctly describe the library's various components.
However, this is not the right venue for all kinds of documentation, and that's where the website comes in.

## Website
The website is hosted using GitHub pages and is generated from markdown files that are separate from the source code itself.
Rather than documenting individual components, the website is concerned with all of the other information necessary for a strong user experience: tutorials, explainers, and "how-to"s.

### Building the Website
The `ndarray` website uses [MkDocs](https://www.mkdocs.org) with the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/#everything-you-would-expect) framework and theme.
This allows us to generate great documentation using just markdown files, meaning developers and users should be able to contribute without learning a whole new paradigm (smaller edits and additions won't even need an understanding of the MkDocs approach).

Material for MkDocs is distributed as a Python package; if you're already comfortable with Python workflows, the quickest way to get started is the follow the [installation instructions](https://squidfunk.github.io/mkdocs-material/getting-started/) on the Material for MkDocs website.

However, new users may find it more expedient to [install uv](https://docs.astral.sh/uv/getting-started/installation/), a new Rust-based, Cargo-like package manager (+more) for Python.
After installing, you can simply run
```shell
uvx --with mkdocs-material mkdocs build -d target/site
```
from the `ndarray` root directory.
This will download and isolate all of the necessary dependencies and build the docs into the `target/site` directory.

### Website Development
While writing documentation, its easiest to use the MkDocs server, which includes hot reloads.
If you're using `uvx`, just run
```shell
uvx --with mkdocs-material mkdocs serve
```
to start a developer server on port 8000.
Navigate to `http://127.0.0.1:8000/ndarray/` to start browsing the docs.
