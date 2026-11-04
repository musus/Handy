//! Long-form chunked decoding, shared by a live recording and a batch clip:
//! both cut the same audio at the same pauses and decode it the same way.

use std::time::{Duration, Instant};

use log::{debug, info};
use transcribe_cpp::Transcript;

use super::{emit_stream_text, PreviewScript, RunSetup, TranscriptionManager};
use crate::audio_toolkit::chunking::{join_transcripts, Chunker};
use crate::engine_supervisor::{CancelMark, EngineError, LoadedInfo};
use crate::language_detection::Plan;
use crate::managers::audio::create_detector;
use crate::settings::{get_settings, AppSettings, VadBackend};

/// Fork (tail preview): shortest pause between two preview decodes, counted
/// from the end of the last one.
const PREVIEW_INTERVAL: Duration = Duration::from_millis(400);
/// Fork (tail preview): audio not yet in a chunk shorter than this isn't
/// previewed (the model rejects < 0.1 s, and so little yields no useful text).
const PREVIEW_MIN_SAMPLES: usize = 16_000 / 5;
/// Fork (tail preview): nor is audio longer than this. Live cuts wait for up
/// to 45 s, and one run that long can overrun a model's output cap
/// (moonshine's is 194 tokens); the preview just holds until the next cut.
const PREVIEW_MAX_SAMPLES: usize = 40 * 16_000;

/// How a chunked decode's runs are set up: known up front (a whole clip), or
/// once the first chunk's audio can pick the language (a live recording).
pub(super) enum ChunkSetup {
    Ready(Box<RunSetup>),
    Pending(Box<PendingSetup>),
}

/// What a chunked decode's setup is resolved from.
pub(super) struct PendingSetup {
    pub plan: Plan,
    pub info: LoadedInfo,
    pub settings: AppSettings,
}

impl ChunkSetup {
    /// The setup, resolving a pending one with `audio`.
    fn resolve(&mut self, tm: &TranscriptionManager, audio: &[f32]) -> &RunSetup {
        if let Self::Pending(pending) = self {
            let setup = tm.setup_for(&pending.info, &pending.settings, &pending.plan, audio);
            *self = Self::Ready(Box::new(setup));
        }
        match self {
            Self::Ready(setup) => setup,
            Self::Pending(_) => unreachable!("resolved above"),
        }
    }
}

/// Cuts audio at pauses as it arrives and decodes each chunk once its cut is
/// certain, so a recording has only its last chunk left when it stops.
pub(super) struct ChunkedDecode<'a> {
    tm: &'a TranscriptionManager,
    /// Every chunk's run stops with one cancel.
    mark: CancelMark,
    setup: ChunkSetup,
    model_id: String,
    chunker: Chunker,
    transcripts: Vec<Transcript>,
    /// The first run that failed or was cancelled; nothing more is decoded
    /// after it.
    failure: Option<EngineError>,
    /// Live text for a model that can't stream (fork); `None` for a clip.
    preview: Option<TailPreview>,
}

/// Fork: live text for a model that can't stream. The audio not yet in a
/// chunk is decoded whole again each time new audio has arrived and the last
/// preview finished at least [`PREVIEW_INTERVAL`] ago, and shown as tentative
/// text after the chunks decoded so far. Audio fed while a preview decodes
/// queues on the channel and is all pushed before the next one.
struct TailPreview {
    /// Made once the run setup is known.
    script: Option<PreviewScript>,
    last_done: Option<Instant>,
    /// [`Chunker::samples`] at the last preview.
    previewed: usize,
}

impl<'a> ChunkedDecode<'a> {
    pub(super) fn new(
        tm: &'a TranscriptionManager,
        mark: CancelMark,
        setup: ChunkSetup,
        model_id: &str,
        backend: VadBackend,
    ) -> Self {
        // The VAD is only loaded if the audio turns out long enough to cut.
        let app_handle = tm.app_handle.clone();
        Self {
            tm,
            mark,
            setup,
            model_id: model_id.to_string(),
            chunker: Chunker::new(Box::new(move || create_detector(backend, &app_handle))),
            transcripts: Vec::new(),
            failure: None,
            preview: None,
        }
    }

    /// Shows live text while recording (fork): see [`TailPreview`].
    pub(super) fn with_preview(mut self) -> Self {
        self.preview = Some(TailPreview {
            script: None,
            last_done: None,
            previewed: 0,
        });
        self
    }

    /// A whole clip's chunks, decoded in order, and the setup they ran with.
    pub(super) fn decode_clip(
        tm: &'a TranscriptionManager,
        mark: CancelMark,
        setup: ChunkSetup,
        audio: Vec<f32>,
        model_id: &str,
        backend: VadBackend,
    ) -> Result<(Vec<Transcript>, RunSetup), EngineError> {
        let mut decode = Self::new(tm, mark, setup, model_id, backend);
        decode.push(audio);
        decode.finish()
    }

    /// Samples pushed so far.
    pub(super) fn samples(&self) -> usize {
        self.chunker.samples()
    }

    /// Chunks decoded so far.
    pub(super) fn decoded(&self) -> usize {
        self.transcripts.len()
    }

    /// Whether a failure already ended the decode; [`finish`](Self::finish)
    /// reports it.
    pub(super) fn failed(&self) -> bool {
        self.failure.is_some()
    }

    /// Adds audio, decoding each chunk it makes certain.
    pub(super) fn push(&mut self, pcm: Vec<f32>) {
        if self.failure.is_some() {
            return;
        }
        let chunks = self.chunker.push(pcm);
        let cut = !chunks.is_empty();
        self.decode(chunks);
        self.preview(cut);
    }

    /// Decodes the audio not yet in a chunk and shows it after the decoded
    /// chunks (fork). `cut` forces it: the last preview's text is now in a
    /// chunk.
    fn preview(&mut self, cut: bool) {
        let samples = self.chunker.samples();
        let Some(preview) = self.preview.as_mut() else {
            return;
        };
        if self.failure.is_some() || (!cut && samples == preview.previewed) {
            return;
        }
        if !cut
            && preview
                .last_done
                .is_some_and(|done| done.elapsed() < PREVIEW_INTERVAL)
        {
            return;
        }
        // The first chunk picks the language for every chunk. Until it is
        // decoded, a plan that detects the language from audio isn't fixed
        // yet, so there is nothing to preview with; any other plan is fixed
        // whatever the audio, and resolving it now is what the first chunk
        // would do.
        if let ChunkSetup::Pending(pending) = &self.setup {
            if pending.plan.detects() {
                return;
            }
        }
        let setup = self.setup.resolve(self.tm, &[]);
        let run_options = setup.run_options.clone();
        let languages = setup.languages.clone();
        let script = preview.script.get_or_insert_with(|| {
            let settings = get_settings(&self.tm.app_handle);
            PreviewScript::new(
                settings.chinese_script,
                &setup.output_language(&settings, None),
            )
        });

        let committed = join_transcripts(
            &self
                .transcripts
                .iter()
                .map(|t| t.text.as_str())
                .collect::<Vec<_>>(),
        );
        let tail = self.chunker.pending();
        let tentative = if (PREVIEW_MIN_SAMPLES..=PREVIEW_MAX_SAMPLES).contains(&tail.len()) {
            match self
                .tm
                .run_batch(self.mark, tail.to_vec(), &run_options, &self.model_id)
            {
                Ok(transcript) => Some(transcript.text),
                // A preview that fails shows nothing new; the chunk decode
                // reports a real failure.
                Err(e) => {
                    debug!("Live preview decode skipped: {}", e);
                    None
                }
            }
        } else {
            None
        };
        preview.previewed = samples;
        preview.last_done = Some(Instant::now());
        // After a cut the old tentative text is in `committed`; without a new
        // preview it is dropped rather than shown twice.
        if tentative.is_none() && !cut {
            return;
        }
        let (committed, tentative) =
            script.convert(&committed, tentative.as_deref().unwrap_or(""), &languages);
        emit_stream_text(&self.tm.app_handle, &committed, &tentative);
    }

    /// Decodes what is left. Every chunk's transcript in order and the setup
    /// they ran with, or the run that failed or was cancelled.
    pub(super) fn finish(mut self) -> Result<(Vec<Transcript>, RunSetup), EngineError> {
        if self.failure.is_none() {
            let chunks = self.chunker.finish();
            self.decode(chunks);
        }
        if let Some(failure) = self.failure {
            return Err(failure);
        }
        if self.transcripts.len() > 1 {
            info!(
                "Decoded {:.1}s of audio in {} chunks",
                self.samples() as f64 / 16_000.0,
                self.transcripts.len()
            );
        }
        // No chunk at all (no audio) still needs a setup to finish with.
        self.setup.resolve(self.tm, &[]);
        let ChunkSetup::Ready(setup) = self.setup else {
            unreachable!("resolved above")
        };
        Ok((self.transcripts, *setup))
    }

    fn decode(&mut self, chunks: Vec<Vec<f32>>) {
        for chunk in chunks {
            if self.failure.is_some() {
                return;
            }
            // The first chunk picks the language for every chunk.
            let run_options = self.setup.resolve(self.tm, &chunk).run_options.clone();
            let audio_secs = chunk.len() as f64 / 16_000.0;
            let started = Instant::now();
            match self
                .tm
                .run_batch(self.mark, chunk, &run_options, &self.model_id)
            {
                Ok(transcript) => {
                    self.transcripts.push(transcript);
                    // Real-time factor: audio seconds decoded per second of
                    // wall clock. Floored so a near-instant decode reports a
                    // large factor instead of infinity.
                    let elapsed_secs = started.elapsed().as_secs_f64().max(1e-6);
                    info!(
                        "Decoded long-form chunk {} ({:.1}s of audio) in {:.2}s ({:.0}x real time)",
                        self.transcripts.len(),
                        audio_secs,
                        elapsed_secs,
                        audio_secs / elapsed_secs
                    );
                }
                Err(e) => {
                    info!("Chunk decoding stopped: {}", e);
                    self.failure = Some(e);
                }
            }
        }
    }
}
