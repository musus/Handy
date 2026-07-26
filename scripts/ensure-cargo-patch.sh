#!/usr/bin/env bash
set -euo pipefail

# upstream sync のマージ時、src-tauri/Cargo.toml のコンフリクトは
# "theirs" (upstream版) を丸ごと採用して解決する。そのため、
# フォーク独自にこのファイルへ追加した依存はすべて消える。
# ここで復元する。新しいフォーク独自依存を Cargo.toml に足したら
# このスクリプトにも追記すること。

toml="src-tauri/Cargo.toml"

insert_after_marker() {
  local marker="$1" entry="$2"
  local line
  line=$(grep -nF "$marker" "$toml" | head -1 | cut -d: -f1)
  if [[ -z "$line" ]]; then
    echo "warning: marker not found, skip: ${marker}" >&2
    return 1
  fi
  awk -v n="$line" -v entry="$entry" 'NR==n{print; print entry; next} {print}' "$toml" > "${toml}.tmp"
  mv "${toml}.tmp" "$toml"
}

# [patch.crates-io] transcribe-rs -> ローカルパッチ (.ort フォールバック, BaseJa variant)
entry='transcribe-rs = { path = "transcribe-rs-patch" }'
if grep -qF "$entry" "$toml"; then
  echo "[patch.crates-io] transcribe-rs already present"
elif grep -qF '[patch.crates-io]' "$toml"; then
  # 既存の [patch.crates-io] テーブル(最初の出現)直下に追記する。
  # TOML は同名テーブルの重複定義を許さないため、新規セクションを
  # 追加すると "duplicate key" でパースエラーになる。
  insert_after_marker '[patch.crates-io]' "$entry"
  echo "inserted transcribe-rs into existing [patch.crates-io]"
else
  printf '\n[patch.crates-io]\n%s\n' "$entry" >> "$toml"
  echo "appended [patch.crates-io] transcribe-rs entry"
fi

# macos target deps -> coreaudio-sys (system audio ducking, フォーク独自機能)
entry='coreaudio-sys = { version = "0.2" }'
marker='[target.'"'"'cfg(target_os = "macos")'"'"'.dependencies]'
if grep -qF "$entry" "$toml"; then
  echo "macos coreaudio-sys dependency already present"
else
  insert_after_marker "$marker" "$entry"
  echo "inserted coreaudio-sys into macos target deps"
fi
