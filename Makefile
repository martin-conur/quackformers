.PHONY: clean clean_all test_rust

PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

EXTENSION_NAME=quackformers

# Set to 1 to enable Unstable API (binaries will only work on TARGET_DUCKDB_VERSION, forwards compatibility will be broken)
# Note: currently extension-template-rs requires this, as duckdb-rs relies on unstable C API functionality
USE_UNSTABLE_C_API=1

# Target DuckDB version
TARGET_DUCKDB_VERSION=v1.5.5
DUCKDB_TEST_VERSION=1.5.5

all: configure debug

# Include makefiles from DuckDB
include extension-ci-tools/makefiles/c_api_extensions/base.Makefile
include extension-ci-tools/makefiles/c_api_extensions/rust.Makefile

configure: venv platform extension_version

debug: build_extension_library_debug build_extension_with_metadata_debug
release: build_extension_library_release build_extension_with_metadata_release

test: test_debug test_rust
test_debug: test_extension_debug
test_release: test_extension_release

# Rust unit tests. Separate from the sqllogictest suite: these cover model
# internals (pooling, ALiBi construction, tokenizer config) that never cross
# the DuckDB boundary, so they need no extension load and run in seconds
# rather than the ~3 minutes LOAD currently costs.
#
# --lib is deliberate. Tests that call into DuckDB cannot run here: with the
# loadable-extension feature its symbols resolve from the host process at load
# time, so there is nothing to call in a standalone test binary. Test model
# internals in Rust, the SQL surface in test/sql.
test_rust:
	cargo test --lib --locked

clean: clean_build clean_rust
clean_all: clean_configure clean

# Set DuckDB version across all configuration files
set-version:
	@if [ -z "$(VERSION)" ]; then \
		echo "Error: VERSION not specified"; \
		echo "Usage: make set-version VERSION=v1.4.0"; \
		exit 1; \
	fi
	@chmod +x set_duckdb_version.sh
	@./set_duckdb_version.sh $(VERSION)
