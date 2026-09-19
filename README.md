# Quackformers: A DuckDB Extension for LLM-Related Functionality

**Quackformers**, a DuckDB extension for LLM-related tasks. For embedding and RAG-like features on DuckDB:

```sql
LOAD 'build/debug/quackformers.duckdb_extension'; -- IF BUILDING LOCALLY

-- IMPORTING FROM DUCKDB COMMUNITY
INSTALL quackformers FROM community;
LOAD quackformers;

CREATE TEMP TABLE QUESTIONS(random_questions) AS
VALUES
    ('What is the capital of France?'),
    ('How does a car engine work?'),
    ('What is the tallest mountain in the world?'),
    ('How do airplanes stay in the air?'),
    ('What is the speed of light?')
;

SELECT embed(RANDOM_QUESTIONS)::FLOAT[384] embedded_questions FROM QUESTIONS;
```

### Example: RAG with Just DUCKDB

```sql
INSTALL quackformers FROM community;
LOAD quackformers;

INSTALL vss;
LOAD vss;

CREATE TABLE vector_table AS
SELECT *, embed(text)::FLOAT[384] as embedded_text FROM read_csv_auto('some/path/to/file/filename.csv');
CREATE INDEX hnsw_index on vector_table USING HNSW (embedded_text);

-- GETTING MOST IMPORTANT CHUNKS BASED ON QUESTION
SELECT text FROM vector_table
ORDER BY array_distance(embedded_text, embed('Some question related to the file?')::FLOAT[384])
LIMIT 5;

SELECT text FROM vector_table
ORDER BY array_distance(embedded_text, embed('Another question related to the file?')::FLOAT[384])
LIMIT 5;
```

For more examples, check out the [examples folder](examples/).

If building locally or calling from repo, you should use the -unsigned tag.
```shell
duckdb -unsigned 
```

Features:
- No DuckDB build required
- No C++ or C code required
- CI/CD chain preconfigured
- (Coming soon) Works with community extensions

## Cloning

Clone the repo with submodules

```shell
git clone --recurse-submodules git@github.com:martin-conur/quackformers.git
```

## Dependencies
In principle, these extensions can be compiled with the Rust toolchain alone. However, this template relies on some additional
tooling to make life a little easier and to be able to share CI/CD infrastructure with extension templates for other languages:

- Python3
- Python3-venv
- [Make](https://www.gnu.org/software/make)
- Git

Installing these dependencies will vary per platform:
- For Linux, these come generally pre-installed or are available through the distro-specific package manager.
- For MacOS, [homebrew](https://formulae.brew.sh/).
- For Windows, [chocolatey](https://community.chocolatey.org/).

## Building
After installing the dependencies, building is a two-step process. Firstly run:
```shell
make configure
```
This will ensure a Python venv is set up with DuckDB and DuckDB's test runner installed. Additionally, depending on configuration,
DuckDB will be used to determine the correct platform for which you are compiling.

Then, to build the extension run:
```shell
make debug
```
This delegates the build process to cargo, which will produce a shared library in `target/debug/<shared_lib_name>`. After this step, 
a script is run to transform the shared library into a loadable extension by appending a binary footer. The resulting extension is written
to the `build/debug` directory.

To create optimized release binaries, simply run `make release` instead.

## CI/CD

GitHub Actions builds the extension for every supported platform on pushes to
`main` and `v2`, and on every pull request. Documentation-only changes are
skipped.

**Built:** `linux_amd64`, `linux_arm64`, `osx_amd64`, `osx_arm64`, `windows_amd64`.

**Not built:** `wasm_mvp`, `wasm_eh`, `wasm_threads`, `linux_amd64_musl`,
`windows_amd64_mingw`. WASM is excluded deliberately — see [#72](https://github.com/martin-conur/quackformers/issues/72)
for the blockers.

Per-run binaries can be downloaded from the workflow run's artifacts. Released
binaries are distributed through the DuckDB community extensions repository
(`INSTALL quackformers FROM community`). Local builds land in `build/debug` and
`build/release`.

## Testing
This extension uses the DuckDB Python client for testing. This should be automatically installed in the `make configure` step.
The tests themselves are written in the SQLLogicTest format, just like most of DuckDB's tests. A sample test can be found in
`test/sql/<extension_name>.test`. To run the tests using the *debug* build:

```shell
make test_debug
```

or for the *release* build:
```shell
make test_release
```

### Version switching
Switching to different DuckDB versions is now automated with a single command:

```shell
make set-version VERSION=v1.4.0
```

This command will automatically:
- Update the `Makefile` (TARGET_DUCKDB_VERSION, DUCKDB_TEST_VERSION)
- Update `Cargo.toml` (duckdb and libduckdb-sys dependencies)
- Update `.github/workflows/MainDistributionPipeline.yml` (CI workflow versions)
- Update `Cargo.lock` (via cargo update)

After changing the version, rebuild and test:
```shell
make clean_all
make configure
make debug
make test_debug
```

**Example**: To switch to DuckDB v1.3.2:
```shell
make set-version VERSION=v1.3.2
make clean_all && make configure && make debug && make test_debug
```

### Known issues
This is a bit of a footgun, but the extensions produced by this template may (or may not) be broken on windows on python3.11 
with the following error on extension load:
```shell
IO Error: Extension '<name>.duckdb_extension' could not be loaded: The specified module could not be found
```
This was resolved by using python 3.12

## Roadmap

Planned work is tracked on the [quackformers v2 board](https://github.com/users/martin-conur/projects/5):

| Milestone | Theme |
|---|---|
| [1.6.0](https://github.com/martin-conur/quackformers/milestone/1) | Correctness and load behaviour, plus `embed_bge()` and `embed_multilingual()` |
| [1.7.0](https://github.com/martin-conur/quackformers/milestone/2) | Migration to DuckDB 2.0's stable V2 C API |
| [1.8.0](https://github.com/martin-conur/quackformers/milestone/3) | `token_count()`, `split_text()`, `embed_models()`, settings, Metal opt-in |
| [2.0.0](https://github.com/martin-conur/quackformers/milestone/4) | Breaking: new default model, unified `embed(text, model := ...)` dispatch |

Longer term: a cross-encoder `rerank()`, late chunking, and quantised outputs.

**On document splitting.** An earlier plan called for a `read_split(path)` table
function. That is superseded by a scalar `split_text()` consumed with `UNNEST`:
DuckDB table functions cannot take a correlated column as an argument, and
`split_text()` already composes with DuckDB's own file reader, which handles
path resolution, globs and remote filesystems for free.

```sql
SELECT u.chunk_index, embed(u.chunk_text)
FROM docs, UNNEST(split_text(docs.body, chunk_size := 256)) AS u;

-- from a file, no dedicated reader needed
SELECT u.chunk_text FROM UNNEST(split_text(read_text('docs/readme.md'))) AS u;
```

## Open Discussion

If you have ideas for custom embedding models or additional features you'd like to see in **Quackformers**, feel free to open a discussion or create an issue in the repository. We welcome your feedback and contributions!