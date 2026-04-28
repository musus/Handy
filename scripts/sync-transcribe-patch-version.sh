#!/usr/bin/env bash
set -euo pipefail

cargo_toml="src-tauri/Cargo.toml"
patch_toml="src-tauri/transcribe-rs-patch/Cargo.toml"

upstream_ver=$(grep -m1 -E '^transcribe-rs = \{ version = "' "$cargo_toml" \
  | sed -E 's/.*version = "([^"]+)".*/\1/')

if [[ -z "${upstream_ver}" ]]; then
  echo "error: could not extract transcribe-rs version from $cargo_toml" >&2
  exit 1
fi

current_ver=$(grep -m1 -E '^version = "' "$patch_toml" \
  | sed -E 's/.*version = "([^"]+)".*/\1/')

if [[ "${current_ver}" == "${upstream_ver}" ]]; then
  echo "transcribe-rs-patch already at ${upstream_ver}"
  exit 0
fi

sed -i.bak -E "0,/^version = \"[^\"]+\"$/{s/^version = \"[^\"]+\"$/version = \"${upstream_ver}\"/}" "$patch_toml"
rm "${patch_toml}.bak"
echo "synced transcribe-rs-patch ${current_ver} -> ${upstream_ver}"
