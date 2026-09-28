# Changelog

Notable changes to quackformers. Versions are **plain quackformers semver** —
the supported DuckDB version is stated per release, not encoded in the number.
See [CONTRIBUTING.md](CONTRIBUTING.md#versioning).

## [1.6.0-rc.1] — unreleased

Built against **DuckDB v1.5.5**.

> **This is a supported build, not a preview.** DuckDB community extensions
> have no separate pre-release channel — `INSTALL quackformers FROM community`
> serves this build to everyone. The `-rc.1` says only that the 1.6.0 milestone
> is not finished; it does not mean "not for production". The crash fixes below
> are shipped ahead of the rest of the milestone precisely because waiting
> would leave them unfixed in the wild.

> [!IMPORTANT]
> **`embed()` output changes in this release.** MiniLM inputs longer than 128
> tokens now embed differently. If you store vectors or have built an HNSW
> index over `embed()` output, re-embed after upgrading — nothing will error,
> the results just quietly get worse.
>
> **`embed_jina()` output is unchanged here, but will change in the next
> release** ([#33](https://github.com/martin-conur/quackformers/issues/33)).
> If you are about to build a large Jina index, it may be worth waiting.

### Fixed

- **`embed()` could crash the DuckDB process.** Casting output to text, or
  passing it to a list aggregate, read past the end of the vector's child
  buffer — `SELECT embed(t)::VARCHAR` segfaulted, and `list_sum(embed(t))`
  silently returned garbage such as `-2.9e+137`. The list's child size was set
  to the row count rather than the element count. Present since the first
  release that returned a list.
  ([#89](https://github.com/martin-conur/quackformers/issues/89))
- **`embed(NULL)` could crash the DuckDB process.** The input validity mask was
  never consulted, so a NULL slot yielded an arbitrary pointer and length: a
  garbage vector on macOS, an `ACCESS_VIOLATION` on Windows. NULL now
  propagates as NULL, with row alignment preserved when NULLs are skipped.
  ([#35](https://github.com/martin-conur/quackformers/issues/35))
- **MiniLM truncated at 128 tokens instead of 256.** The limit was inherited
  from `tokenizer.json` rather than taken from `sentence_bert_config.json`, so
  any input over 128 tokens disagreed with every other sentence-transformers
  consumer. Truncation is now set explicitly per model — 256 for MiniLM, 512
  for Jina. **This is the vector change noted above.**
  ([#34](https://github.com/martin-conur/quackformers/issues/34))

### Performance

- **`LOAD` is roughly 130× faster and uses ~2.2 GB less memory.** The Jina
  ALiBi bias was built once at the model's maximum context of 8192 tokens —
  a 3.2 GB tensor allocated at load time and then sliced down per query. It is
  now built at the sequence length actually being processed.
  ([#32](https://github.com/martin-conur/quackformers/issues/32))

  | | before | after |
  |---|---|---|
  | `LOAD` (debug build) | 167.56 s | 1.26 s |
  | `LOAD` (release build) | — | 0.15 s |
  | peak RSS | ~3.2 GB | ~1.0 GB |

  Embedding output is unchanged — verified against the reference
  implementation.

- **Embedding calls now run concurrently.** A process-wide mutex serialised
  every call to `embed()` and `embed_jina()`, so DuckDB's thread pool could
  only ever run one forward pass at a time. Measured on a 12-core machine over
  a 320-row multi-file scan:

  | `SET threads` | wall clock | speedup |
  |---|---|---|
  | 1 | 16.48 s | 1.0× |
  | 2 | 8.31 s | 2.0× |
  | 4 | 5.55 s | 3.0× |
  | 8 | 5.01 s | 3.3× |

  Concurrency is deliberately capped at `cores / 2`, maximum 4. Activation
  memory scales with the number of simultaneous forward passes — roughly
  400 MB per in-flight batch of 32 rows at 512 tokens — so an uncapped version
  could turn a query that used to be slow into one that exhausts memory. The
  plateau between 4 and 8 threads above is that cap working as intended.

  Output is unaffected by thread count: the checksum over all 320 rows is
  identical at every setting above.
  ([#37](https://github.com/martin-conur/quackformers/issues/37), thanks to
  [@mikemikimike](https://github.com/mikemikimike))

### Added

- A golden-vector regression suite that asserts embedding **values** against
  reference vectors generated from `sentence-transformers`, covering empty,
  accented, non-Latin, near-limit and over-limit inputs for both models.
  ([#29](https://github.com/martin-conur/quackformers/issues/29))
- Rust unit tests now run in CI.
  ([#80](https://github.com/martin-conur/quackformers/issues/80))

### Changed

- Intel MKL is enabled on Linux x86_64. It is deliberately **not** used on
  Windows, where it links dynamically against `libiomp5md.dll` and the
  extension builds but fails to load.
- Build artifacts are no longer committed to the repository. Releases are
  distributed through the DuckDB community extensions repository.
  ([#73](https://github.com/martin-conur/quackformers/issues/73))

---

## Earlier releases

Tags before this release (`v1.5.5`, `v1.5.2`, `v1.4.3`, …) tracked the *DuckDB*
version rather than quackformers' own. That discontinuity is deliberate; see
[CONTRIBUTING.md](CONTRIBUTING.md#versioning).
