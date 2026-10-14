Read @AGENTS.md

## このフォークについて (musus/Handy)

本リポジトリは [cjpais/Handy](https://github.com/cjpais/Handy) のフォークで、**Moonshine Base Japanese** モデルサポートを追加している。

### リモート構成

- `origin` → `git@github.com:musus/Handy.git`（このフォーク）
- `upstream` → `https://github.com/cjpais/Handy.git`（本家）

### フォーク独自の変更点

1. **`moonshine-base-ja` モデルの追加**
   - `src-tauri/src/managers/model.rs`: モデルエントリ追加（134MB、日本語専用）
   - ダウンロードURL: `https://github.com/musus/Handy/releases/download/v0.8.0-moonshine-ja/moonshine-base-ja.tar.gz`
   - モデルファイルは `.ort` 形式（`encoder_model.ort`, `decoder_model_merged.ort`, `tokenizer.json`）
2. **`src-tauri/src/managers/transcription.rs`**
   - `moonshine-base-ja` → `MoonshineVariant::BaseJa` にマッピング
3. **`src-tauri/transcribe-rs-patch/`** — `[patch.crates-io]` 経由で `transcribe-rs` を置換するローカルパッチ
   - `src/onnx/session.rs::resolve_model_path` に `.ort` フォールバックを追加
   - `src/onnx/moonshine/mod.rs` に `MoonshineVariant::BaseJa` を追加（`token_rate=13` CJK対応、`num_layers=8`, `head_dim=52`）
   - `src/onnx/moonshine/model.rs` の `MoonshineTokenizer::decode` で CJK 間スペース除去と末尾反復除去（`strip_trailing_repeats`）
4. **moonshine-base-ja の rebatch ライブプレビュー** — `transcription.rs` の `run_rebatch_stream_loop` / `rebatch_transcribe`（`REBATCH_*` 定数）
5. **macOS システム音量 ducking** — `managers/audio.rs`（`macos_coreaudio`、`MuteState.ducked`、`restore_audio()`）、`actions.rs`（`duck_now`）、`settings.rs` の `duck_volume_percent` / `duck_fade_ms`、`shortcut/mod.rs` の `change_duck_*`、`DuckLevelSlider` / `DuckFadeSlider`、全ロケールの `settings.sound.duckLevel` / `duckFade`、`Cargo.toml` の `coreaudio-sys`
6. **CJK 文字間スペース除去** — `audio_toolkit/text.rs` の `remove_cjk_interchar_spaces`（`normalize_transcription_output` 内）
7. **fork 配布モデルを legacy 扱いしない** — `src/components/onboarding/ModelCard.tsx` の `FORK_URL_MODEL_IDS`
8. **updater / CI** — `tauri.conf.json` の updater endpoint・pubkey、`.github/workflows` は fork 版に固定（upstream-sync）

### upstream マージ手順

```bash
git fetch upstream
git merge upstream/main
# Cargo.lock でコンフリクトが出たら upstream 版で解決
git checkout upstream/main -- src-tauri/Cargo.lock
```

**重要な落とし穴**: upstream が `transcribe-rs` のバージョンを上げると、`transcribe-rs-patch` のバージョンが要件を満たさず「`patch was not used in the crate graph`」警告が出て、パッチが黙って無視される。この場合：

1. `transcribe-rs-patch/Cargo.toml` と `Cargo.toml.orig` の `version` を upstream `src-tauri/Cargo.toml` で要求されるバージョンに合わせる
2. 新しい upstream `transcribe-rs` の src を cargo キャッシュから取得：
   `~/.cargo/registry/src/index.crates.io-*/transcribe-rs-<VER>/src/` を `transcribe-rs-patch/src/` に上書きコピー
3. フォーク独自の変更（上記`.ort` フォールバックと `BaseJa` バリアント）を再適用
4. `transcribe-rs-patch/Cargo.toml` の `whisper-cpp` feature に `whisper-rs/raw-api` が含まれているか確認（upstream の `gpu.rs` が `whisper_rs::whisper_rs_sys` を使うため必須）
5. `cargo update transcribe-rs` してから再ビルド

**コンフリクトにならない取りこぼしにも注意**（v0.9.8 で実際に発生）：

- git が自動マージした fork コードが、upstream の関数分割・型変更に追従せずコンパイルエラーになる。マージ後は必ずビルドを通す。例: `filter_transcription_output` が `remove_filler_words` + `normalize_transcription_output` に分割、`StreamCmd::Finalize` の返信型が `Option<FinalizedStreamText>` に変更 → rebatch ループの追従が必要だった
- upstream 版の `Cargo.lock` をそのまま採用すると `coreaudio-sys` と transcribe-rs パッチが lock から落ちる。ビルド後に再生成された `src-tauri/Cargo.lock` をコミットする
- upstream が新しいロケールを追加したら、duck 系の翻訳キーも足す（`bun run check:translations`）

### ビルド環境 (macOS)

必要なツール（どれか欠けるとビルド失敗）：

- **Rust/cargo**: `~/.cargo/bin/cargo`（インストール: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`）
- **Bun**: `~/.bun/bin/bun`（インストール: `curl -fsSL https://bun.sh/install | bash`）
- **CMake**: `brew install cmake`（`whisper-rs-sys` のビルドに必要）
- **Xcode**: フル版が必要（Command Line Tools だけだと `TargetConditionals.h` が見つからず `ring` ビルドが失敗）

ビルド実行時は環境変数が必要：

```bash
export PATH="$HOME/.cargo/bin:$HOME/.bun/bin:/opt/homebrew/bin:$PATH"
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)
bun run tauri build
```

- `xcode-select` が CommandLineTools を指していてもビルドは通るが、Apple Intelligence（後処理プロバイダ）が stub になる。`DEVELOPER_DIR` でフル Xcode を指定し、ログに `Building with Apple Intelligence support.` が出ることを確認する
- Xcode 更新後はライセンス未同意で `xcrun` が止まる。`sudo /Applications/Xcode.app/Contents/Developer/usr/bin/xcodebuild -license accept`（`xcode-select` を変えずに同意できる）

アプリ本体の生成後に出る「`A public key has been found, but no private key. Make sure to set TAURI_SIGNING_PRIVATE_KEY`」エラーはアップデーター署名の警告で、`.app`/`.dmg` 自体は正常に生成されている。

### 動作確認

ビルド成果物：

- `src-tauri/target/release/bundle/macos/Handy.app`
- `src-tauri/target/release/bundle/dmg/Handy_<version>_aarch64.dmg`

モデル配置先（既にダウンロード済みの場合）：
`~/Library/Application Support/com.pais.handy/models/moonshine-base-ja/`
