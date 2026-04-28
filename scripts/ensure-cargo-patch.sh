#!/usr/bin/env bash
set -euo pipefail

toml="src-tauri/Cargo.toml"

if grep -qF 'transcribe-rs = { path = "transcribe-rs-patch" }' "$toml"; then
  echo "[patch.crates-io] transcribe-rs already present"
  exit 0
fi

printf '\n[patch.crates-io]\ntranscribe-rs = { path = "transcribe-rs-patch" }\n' >> "$toml"
echo "appended [patch.crates-io] transcribe-rs entry"
