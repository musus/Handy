Read @AGENTS.md

## このフォークについて (musus/Handy)

本リポジトリは [cjpais/Handy](https://github.com/cjpais/Handy) のフォークで、日本語での利用（Moonshine Base Japanese）向けの調整と macOS の音量 ducking を加えている。

v0.10.0 からは upstream がすべてのモデルを transcribe-cpp（GGUF）で動かすようになり、`moonshine-base-ja` も公式カタログ（`handy-computer/moonshine-base-ja-gguf`）に入った。そのため、以前フォークで持っていた `.ort` 版モデル・`transcribe-rs-patch`・rebatch ライブプレビューは廃止し、upstream の仕組みの上に必要な差分だけを載せている。

### リモート構成

- `origin` → `git@github.com:musus/Handy.git`（このフォーク）
- `upstream` → `https://github.com/cjpais/Handy.git`（本家）

### フォーク独自の変更点

1. **長尺分割の切り位置を 30 秒寄りにする** — `src-tauri/src/audio_toolkit/chunking.rs` の `EARLY_CUT_PENALTY`（upstream は 0.05、フォークは 0.15）。upstream の重みだと 22 秒付近の長いポーズで切られやすく、moonshine-base-ja は約 22 秒のチャンクで冒頭の文を落とす（30 秒なら落とさない）。127 秒の日本語録音で確認済み
2. **非ストリーミングモデルのライブ表示（末尾プレビュー）** — 録音中、まだチャンクになっていない末尾の音声を約 0.4 秒おきにデコードし直し、確定済みチャンクの後ろに仮テキストとして表示する。旧 rebatch ライブプレビューの置き換え
   - `managers/transcription/chunked.rs` の `TailPreview` / `ChunkedDecode::preview` / `with_preview`（`PREVIEW_*` 定数）
   - `managers/transcription/live.rs` の `LiveDecode::Chunked` 分岐（`chunk_preview` が立っていれば `with_preview()` と `stream_active`）
   - `managers/transcription.rs` の `chunk_preview` フィールド・`can_preview_chunks()`・`set_chunk_preview()`
   - `actions.rs` の録音開始処理（overlay が Live かつロード済みモデルが分割対象のとき、Live パネルを開いてプレビューを有効化）
   - `audio_toolkit/chunking.rs` の `Chunker::pending()`
3. **macOS システム音量 ducking** — `managers/audio.rs`（`macos_coreaudio`、`MuteState.ducked`、`saved_volume`、`restore_audio()`、`cancel_recording` での `remove_mute()`）、`actions.rs`（`duck_now`）、`settings.rs` の `duck_volume_percent` / `duck_fade_ms`、`shortcut/mod.rs` の `change_duck_*`、`lib.rs` のコマンド登録、`DuckLevelSlider` / `DuckFadeSlider`、`GeneralSettings.tsx`、`settingsStore.ts`、`bindings.ts`、全ロケールの `settings.sound.duckLevel` / `duckFade`、`Cargo.toml` の `coreaudio-sys`
4. **CJK 文字間スペース除去** — `audio_toolkit/text.rs` の `remove_cjk_interchar_spaces`（`normalize_transcription_output` 内）
5. **旧 `.ort` 版 moonshine-base-ja の移行** — `managers/model.rs` の `RETIRED_ONNX_MODELS` に `moonshine-base-ja` を追加。`models/moonshine-base-ja/` が残っていれば、upstream の移行画面が GGUF 版への置き換えと旧ファイル削除を案内する
6. **途中で止まったデコードの結果を捨てない** — `engine_supervisor/worker.rs` の `Request::Run` 分岐。transcribe-cpp が繰り返し検出（status 19）や生成上限（status 18）で止めたとき、エラーに付いている途中結果（繰り返しは除去済み）をそのまま返す。upstream はエラーにして全文を捨てるため、マイク音量が小さい環境（Studio Display のマイク、入力音量 29% など）では moonshine-base-ja の短い発話がほぼ毎回「文字起こしに失敗しました」になる。旧 0.9.8 フォークも途中結果を返していた
7. **updater / CI** — `tauri.conf.json` の updater endpoint・pubkey・`signingIdentity`（HandyDevCert の SHA-1 で固定。下の「署名証明書」を参照）・`createUpdaterArtifacts`、`.github/workflows` は fork 版に固定（`upstream-sync.yml`）、`.github/upstream-version`

### upstream マージ手順

```bash
git fetch upstream
git merge upstream/main
# Cargo.lock でコンフリクトが出たら upstream 版で解決し、ビルド後に再生成されたものをコミットする
git checkout upstream/main -- src-tauri/Cargo.lock
```

- マージ後は必ずビルドを通す。git が自動マージしたフォークのコードが、upstream の関数分割・型変更に追従せずコンパイルエラーになることがある（v0.9.8 で発生）
- `Cargo.toml` を upstream 版（theirs）で解決すると `coreaudio-sys` が落ちる。手で戻す
- upstream が新しいロケールを追加したら、duck 系の翻訳キーも足す（`bun run check:translations`）
- upstream の構造が大きく変わったリリース（v0.10.0 など）は、マージより「新しいタグからブランチを切ってフォークの変更を載せ直し、旧フォークのブランチを `git merge -s ours` で履歴だけ取り込む」方が早い

### ビルド環境 (macOS)

必要なツール（どれか欠けるとビルド失敗）：

- **Rust/cargo**: `~/.cargo/bin/cargo`（インストール: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`）
- **Bun**: `~/.bun/bin/bun`（インストール: `curl -fsSL https://bun.sh/install | bash`）
- **CMake**: `brew install cmake`（transcribe-cpp のビルドに必要）
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

### 署名証明書（HandyDevCert）

macOS は入力監視・アクセシビリティ・マイクの許可を「バンドル識別子 + 署名証明書」で覚えている。リリースごとに証明書が変わると、更新のたびに許可が外れる。特に入力監視が外れると、ショートカットが修飾キー（fn・command 単体）にしか反応しなくなる。v0.10.0-fork.1（MBA でビルド）→ fork.2（MacStudio でビルド）で、Mac ごとに別々に作った自己署名証明書が使われて実際に起きた。

- `tauri.conf.json` の `signingIdentity` は名前ではなく証明書の SHA-1 `B3088F191FAE021DA24088E2286BB6090C9A179E` で固定している。この証明書を持たない Mac では署名の段階でビルドが失敗する（別の証明書で黙って署名されるのを防ぐため）
- 証明書が入っているのは MacStudio（`MacStudio2023`）だけ。リリース用のビルドは MacStudio で行う。MBA にある HandyDevCert（`92D5A63D…`）は fork.1 を署名した古いもので、使わない
- 他の Mac でビルドしたい場合は、MacStudio のキーチェーンアクセスから HandyDevCert を .p12 で書き出し、`security import HandyDevCert.p12 -k ~/Library/Keychains/login.keychain-db -T /usr/bin/codesign` で入れる
- SSH 越しのビルドは署名で `errSecInternalComponent` になる（キーチェーンがロックされている）。MacStudio の画面上のターミナルでビルドする
- HandyDevCert を新しく作り直さない。作り直すと、全ての Mac で許可が外れる
- リリース前に `codesign -d -r- src-tauri/target/release/bundle/macos/Handy.app` の `certificate leaf` が上のハッシュと一致することを確かめる

### 動作確認

ビルド成果物：

- `src-tauri/target/release/bundle/macos/Handy.app`
- `src-tauri/target/release/bundle/dmg/Handy_<version>_aarch64.dmg`

録音ファイルをアプリを開かずに文字起こしして比べられる（`--transcribe-file`）。普段の設定や履歴を汚さないよう、ポータブルモードで動かす：

```bash
# .app と同じ構成にしないと VAD モデル（resources）のパスが解決できず、長尺分割が効かない
mkdir -p /tmp/h/H.app/Contents/{MacOS,Resources}
cp src-tauri/target/release/handy /tmp/h/H.app/Contents/MacOS/
cp -R src-tauri/resources /tmp/h/H.app/Contents/Resources/
printf "Handy Portable Mode" > /tmp/h/H.app/Contents/MacOS/portable
# HF_HOME を明示すると、ポータブルモードでも ~/.cache/huggingface のモデルを使う
cd /tmp/h/H.app/Contents/MacOS && HF_HOME=$HOME/.cache/huggingface ./handy \
  --transcribe-file <録音.wav> \
  --model handy-computer/moonshine-base-ja-gguf/moonshine-base-ja-Q8_0.gguf
```

録音ファイルの場所: `~/Library/Application Support/com.pais.handy/recordings/`
