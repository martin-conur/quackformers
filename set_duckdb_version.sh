#!/usr/bin/env bash
#
# Script to update DuckDB version across all project files
#
# Usage: ./scripts/set_duckdb_version.sh v1.4.0
#

set -e

VERSION=$1

if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version>"
    echo ""
    echo "Examples:"
    echo "  $0 v1.4.0    # Latest stable"
    echo "  $0 v1.3.0    # Previous version"
    echo "  $0 v1.5.0    # Upcoming version"
    echo ""
    echo "This script updates:"
    echo "  - Makefile (TARGET_DUCKDB_VERSION, DUCKDB_TEST_VERSION)"
    echo "  - Cargo.toml (duckdb and libduckdb-sys versions)"
    echo "  - .github/workflows/MainDistributionPipeline.yml"
    echo "  - Cargo.lock (via cargo update)"
    exit 1
fi

# Validate version format
if [[ ! "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Version must be in format vX.Y.Z (e.g., v1.4.0)"
    exit 1
fi

# Remove 'v' prefix for versions that don't use it
VERSION_NO_V=${VERSION#v}

# Compute the duckdb-rs / libduckdb-sys encoded crate version.
# The crate does NOT use the plain DuckDB version. It encodes it as:
#   crate = 1.{major*10000 + minor*100 + patch}.0
# e.g. DuckDB 1.5.2 -> 1.10502.0, DuckDB 1.5.4 -> 1.10504.0
IFS='.' read -r DDB_MAJOR DDB_MINOR DDB_PATCH <<< "$VERSION_NO_V"
CRATE_MINOR=$((DDB_MAJOR * 10000 + DDB_MINOR * 100 + DDB_PATCH))
CRATE_VERSION="1.${CRATE_MINOR}.0"

echo "🔄 Updating DuckDB version to ${VERSION} (crate version ${CRATE_VERSION})..."
echo ""

# Check if we're in the project root
if [ ! -f "Makefile" ] || [ ! -f "Cargo.toml" ]; then
    echo "Error: Must be run from project root directory"
    exit 1
fi

# Function to update file with backup
update_file() {
    local file=$1
    local pattern=$2
    local desc=$3

    if [ -f "$file" ]; then
        # Use different sed syntax for macOS vs Linux
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "$pattern" "$file"
        else
            sed -i "$pattern" "$file"
        fi
        echo "✓ Updated $desc in $file"
    else
        echo "⚠ Warning: $file not found"
    fi
}

# 1. Update Makefile
echo "📝 Updating Makefile..."
update_file "Makefile" \
    "s/^TARGET_DUCKDB_VERSION=.*/TARGET_DUCKDB_VERSION=${VERSION}/" \
    "TARGET_DUCKDB_VERSION"

update_file "Makefile" \
    "s/^DUCKDB_TEST_VERSION=.*/DUCKDB_TEST_VERSION=${VERSION_NO_V}/" \
    "DUCKDB_TEST_VERSION"

# 2. Update Cargo.toml
# Pin exactly (=) so the caret range does not silently float to a newer
# DuckDB minor/patch. The patterns tolerate the column-aligned whitespace
# in Cargo.toml by matching any run of spaces before '= { version ='.
# '^duckdb[[:space:]]' avoids matching 'libduckdb-sys' and
# 'duckdb-loadable-macros' (neither is followed by whitespace after 'duckdb').
echo "📝 Updating Cargo.toml..."
update_file "Cargo.toml" \
    "s/^\(duckdb[[:space:]]*= { version = \)\"[^\"]*\"/\1\"=${CRATE_VERSION}\"/" \
    "duckdb dependency"

update_file "Cargo.toml" \
    "s/^\(libduckdb-sys[[:space:]]*= { version = \)\"[^\"]*\"/\1\"=${CRATE_VERSION}\"/" \
    "libduckdb-sys dependency"

# 3. Update GitHub Actions workflow
echo "📝 Updating GitHub Actions workflow..."
update_file ".github/workflows/MainDistributionPipeline.yml" \
    "s|uses: duckdb/extension-ci-tools/.github/workflows/_extension_distribution.yml@v[0-9.]*|uses: duckdb/extension-ci-tools/.github/workflows/_extension_distribution.yml@${VERSION}|" \
    "CI tools workflow version"

update_file ".github/workflows/MainDistributionPipeline.yml" \
    "s/duckdb_version: v[0-9.]*/duckdb_version: ${VERSION}/" \
    "duckdb_version parameter"

update_file ".github/workflows/MainDistributionPipeline.yml" \
    "s/ci_tools_version: v[0-9.]*/ci_tools_version: ${VERSION}/" \
    "ci_tools_version parameter"

update_file ".github/workflows/MainDistributionPipeline.yml" \
    "s/DUCKDB_VERSION=\"v[0-9.]*\"/DUCKDB_VERSION=\"${VERSION}\"/g" \
    "DUCKDB_VERSION in bash scripts"

# 4. Update Cargo.lock (requires cargo to be installed)
echo ""
echo "🔧 Updating Cargo.lock..."
if command -v cargo &> /dev/null; then
    # Update only duckdb-related dependencies
    cargo update -p duckdb -p libduckdb-sys 2>&1 | grep -v "Updating" || true
    echo "✓ Updated Cargo.lock"
else
    echo "⚠ Warning: cargo not found. You'll need to run 'cargo update' manually."
fi

echo ""
echo "✅ Successfully updated DuckDB version to ${VERSION}"
echo ""
echo "📋 Modified files:"
echo "  - Makefile"
echo "  - Cargo.toml"
echo "  - Cargo.lock"
echo "  - .github/workflows/MainDistributionPipeline.yml"
echo ""
echo "🔍 Next steps:"
echo "  1. Review changes:       git diff"
echo "  2. Clean build:          make clean_all"
echo "  3. Reconfigure:          make configure"
echo "  4. Build:                make debug"
echo "  5. Test:                 make test_debug"
echo "  6. Commit changes:       git add -A && git commit -m 'Bump DuckDB to ${VERSION}'"
echo "  7. Push to trigger CI:   git push origin master"
echo ""
