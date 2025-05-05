# Quackformers: A DuckDB Extension for LLM-Related Functionality

**Quackformers**, a DuckDB extension for LLM-related tasks. For embedding and RAG-like features on DuckDB:

```sql
LOAD 'build/debug/quackformers.duckdb_extension'; -- IF BUILDING LOCALLY

-- IMPORTING FROM DUCKDB COMMUNITY
INSTALL quackformers fROM  community;
LOAD quackformers;

-- IMPORTING FROM GITHUB REPO
LOAD quackformers FROM 'https://github.com/martin-conur/quackformers';

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
INSTALL quackformers fROM  community;
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
git clone git@github.com:martin-conur/quackformers.git
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
Testing with different DuckDB versions is really simple:

First, run 
```
make clean_all
```
to ensure the previous `make configure` step is deleted.

Then, run 
```
DUCKDB_TEST_VERSION=v1.1.2 make configure
```
to select a different duckdb version to test with

Finally, build and test with 
```
make debug
make test_debug
```

### Known issues
This is a bit of a footgun, but the extensions produced by this template may (or may not) be broken on windows on python3.11 
with the following error on extension load:
```shell
IO Error: Extension '<name>.duckdb_extension' could not be loaded: The specified module could not be found
```
This was resolved by using python 3.12

## Roadmap

Here are the planned features and improvements for **Quackformers**:

1. **Faster Embedding Implementation** ✅  
   - Currently, embeddings are generated row by row. The goal is to implement a more efficient method to embed entire columns or chunks at once.

2. **Jina Integration** ✅  
   - Add support for Jina-based embedding functionality. This feature is currently a work in progress (WIP) and will be available soon.

3. **Return Arrays Instead of Strings** ✅  
   - Modify the output format to return arrays directly instead of strings for better usability and performance.

All planned features have been implemented!

## Open Discussion

If you have ideas for custom embedding models or additional features you'd like to see in **Quackformers**, feel free to open a discussion or create an issue in the repository. We welcome your feedback and contributions!