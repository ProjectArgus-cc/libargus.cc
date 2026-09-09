#!/usr/bin/env bash
set -euo pipefail
# Compatibility entry point; the catalog is the only platform/resource mapping.
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [ "$#" -ne 1 ]; then
  echo 'Usage: scripts/package-artifacts.sh <directory containing native-<catalog-id> artifacts>' >&2
  exit 2
fi
exec python3 "$ROOT/scripts/release/candidate.py" stage "$1"
