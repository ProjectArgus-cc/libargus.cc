#!/usr/bin/env bash
set -euo pipefail
# Bootstrap depends only on the runner and a pinned first-party checkout.
DIR=$(mktemp -d)
trap 'rm -rf "$DIR"' EXIT
curl --fail --location --silent --show-error \
  https://github.com/rhysd/actionlint/releases/download/v1.7.12/actionlint_1.7.12_linux_amd64.tar.gz \
  -o "$DIR/actionlint.tar.gz"
echo '8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8  actionlint.tar.gz' > "$DIR/checksums"
(cd "$DIR" && sha256sum --check checksums && tar -xzf actionlint.tar.gz actionlint)
"$DIR/actionlint" -color
