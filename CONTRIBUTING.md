# Contributing to Quackformers

*Generated with Claude. Human-reviewed before merge — see the policy below.*

## On handcrafting, and on AI

This is a personal project and I want it handcrafted. I use AI a lot — but the
parts of this codebase that make it what it is, I write myself.

**Written by hand:**

- Model implementations and the embedding math (`src/embed_utils/`)
- The DuckDB binding layer (`src/lib.rs`)
- Anything that decides what a vector ends up being

**AI-generated contributions are welcome for:**

- Boilerplate and scaffolding
- Tests, fixtures, and the scripts that generate them
- CI configuration and build tooling
- Documentation and design notes

### Two rules for AI-generated files

**1. Say so, in the file.** Every AI-generated file carries a marker in its
header — the module docstring for Python, `//!` for Rust, an HTML comment or a
line under the title for Markdown, a leading `#` comment for YAML and SQL:

```
Generated with Claude. Human-reviewed before merge — see CONTRIBUTING.md.
```

Name the tool. Keep the phrase `Generated with` intact so the marker stays
greppable:

```sh
grep -rn "Generated with" --include="*.rs" --include="*.py" --include="*.md" .
```

**2. A human reads it before it merges.** Not skims — reads. The marker is a
statement that this happened, not a disclaimer that it didn't. A file carrying
the marker that nobody reviewed is worse than no marker at all.

Partially AI-assisted files don't need the marker. It's for files that were
generated substantially whole.

## Branching

`main` is always releasable.

| Branch | Purpose |
|---|---|
| `main` | Trunk. Short-lived branches off it, one PR per issue |
| `v2` | Long-lived integration branch for breaking (2.0.0) work only |

Breaking changes can't ship in halves, so 2.0.0 work accumulates on `v2`, and
`main` is merged **into** `v2` on every 1.x release so it absorbs DuckDB
version bumps continuously rather than facing one large conflict at the end.

Everything else — fixes, additive features, chores — branches off `main`.

## Versioning

Tags are **plain quackformers semver**. The supported DuckDB version lives in
the release notes and in `description.yml`, not in the tag.

Tags before this convention (`v1.5.5`, `v1.5.2`, `v1.4.3`) track the *DuckDB*
version instead. That discontinuity is deliberate and documented.

## The vector-change rule

Two kinds of compatibility matter here, and only one of them is loud:

| Kind | Failure mode |
|---|---|
| **API compatibility** | Queries error. You find out immediately. |
| **Vector compatibility** | **Silent.** Stored indexes quietly degrade. Nothing errors. |

If a change alters the floats `embed()` returns — a different model, a
different device, a different candle version, a fixed masking bug — it changes
vector compatibility. Someone's stored embeddings become subtly wrong with no
error to tell them.

So: **label any such issue or PR `vector-change`.** It cannot ship in a minor
release without a prominent release note. This does not line up with the
bug/feature distinction — plenty of bug fixes change vectors, and plenty of
features don't. Judge it on the floats, not on the category.

## Building and testing

```sh
make configure      # one-time: venv, platform detection, submodules
make debug          # build
make test_debug     # run test/sql/*.test
```

`extension-ci-tools` is a submodule and its pin must match the
`ci_tools_version` input in `.github/workflows/MainDistributionPipeline.yml`.
When they drift, local and CI silently run different test harnesses and local
results stop meaning anything.

Rust unit tests need `--lib`, because `cargo test` otherwise tries to build
every target:

```sh
cargo test --lib
```

Tests that call into DuckDB won't run under `cargo test` — its symbols resolve
from the host process at load time. Test the model internals in Rust and the
SQL surface in `test/sql/`.

## Opening an issue

Label it `type:bug` / `type:feat` / `type:chore`, add `breaking` and/or
`vector-change` if they apply, and put it on the milestone it belongs to.
