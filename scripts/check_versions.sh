#!/usr/bin/env bash
#
# Verify Cargo.toml, pyproject.toml, and (optionally) a git tag all encode
# the same version. Cargo uses SemVer (2.0.0-rc6), PEP 440 drops the hyphen
# (2.0.0rc6); we compare on the PEP 440 form.
#
# Usage:
#   scripts/check_versions.sh              # Cargo vs pyproject
#   scripts/check_versions.sh v2.0.0rc6    # + tag vs pyproject (strip leading v)
#
# Emits `::error::` annotations so GitHub Actions surfaces them nicely; also
# prints a plain success line for local runs.

set -euo pipefail

here=$(dirname "$(readlink -f "$0")")
root=$(dirname "$here")

cargo=$(awk -F'"' '/^version/ {print $2; exit}' "$root/Cargo.toml")
py=$(awk -F'"' '/^version/ {print $2; exit}' "$root/pyproject.toml")
cargo_pep440="${cargo//-/}"

fail=0
if [ "$cargo_pep440" != "$py" ]; then
    echo "::error file=pyproject.toml::Version mismatch: Cargo.toml='$cargo' (PEP 440: '$cargo_pep440') vs pyproject.toml='$py'"
    fail=1
fi

tag="${1:-}"
if [ -n "$tag" ]; then
    tag_stripped="${tag#v}"
    if [ "$tag_stripped" != "$py" ]; then
        echo "::error::Tag '$tag' (stripped: '$tag_stripped') does not match pyproject.toml='$py'"
        fail=1
    fi
    if [ "$tag_stripped" != "$cargo_pep440" ]; then
        echo "::error::Tag '$tag' (stripped: '$tag_stripped') does not match Cargo.toml PEP 440 form='$cargo_pep440'"
        fail=1
    fi
fi

if [ "$fail" = 1 ]; then exit 1; fi

if [ -n "$tag" ]; then
    echo "✓ versions consistent: Cargo='$cargo', pyproject='$py', tag='$tag'"
else
    echo "✓ versions consistent: Cargo='$cargo', pyproject='$py'"
fi
