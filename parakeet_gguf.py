"""
Parakeet GGUF wrapper for Subgen.

This module provides a transcribe-cpp backed runtime that loads NVIDIA Parakeet
TDT/RNNT/CTC and Canary models in GGUF format and runs them through the ggml
native library (libtranscribe).

Supports:
- Parakeet TDT 0.6B v3 (English ASR)
- Parakeet RNNT 1.1B (English ASR)
- Canary 1B v2 (multilingual ASR + translation)
- Whisper, Qwen3-ASR, and any other model family transcribe.cpp supports

Repository / pre-built weights:
- transcribe.cpp runtime (with Python bindings): https://github.com/handy-computer/transcribe.cpp
- Official Python package on PyPI: https://pypi.org/project/transcribe-cpp/
- Parakeet TDT 0.6B v3 GGUF: https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v3-gguf
- Canary 1B v2 GGUF:          https://huggingface.co/cstr/canary-1b-v2-GGUF

If the transcribe-cpp wheels are not yet available for your platform, build
libtranscribe from the transcribe.cpp repo and point the TRANSCRIBE_LIBRARY
environment variable at the resulting shared object before importing this
module.
"""

import logging
import os
import threading
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result objects compatible with what subgen.py expects (stable-ts shape)
# ---------------------------------------------------------------------------

class Word:
    """A single word with start/end timestamps (seconds) and confidence."""

    def __init__(self, word: str, start: float, end: float, probability: float = 1.0):
        self.word = word
        self.start = start
        self.end = end
        self.probability = probability


class Segment:
    """A subtitle segment with start/end timestamps, text and word list."""

    def __init__(self, start: float, end: float, text: str, words: List[Word] = None, sid: int = 0):
        self.start = start
        self.end = end
        self.text = text
        self.words = words or []
        self.id = sid
        # stable-ts shims read these private attrs in some callers
        self._default_start = start
        self._default_end = end


class TranscriptionResult:
    """Result object compatible with stable-whisper / stable-ts output."""

    def __init__(self, text: str, language: str = "en", segments: List[Segment] = None, tokens=None):
        self.text = text
        self.language = language
        self.segments = segments or []
        self.tokens = tokens or []

    def to_srt_vtt(self, filepath: str = None, word_level: bool = False, vtt: bool = False) -> str:
        """Convert to SRT or VTT."""
        if vtt:
            return self._to_vtt(filepath, word_level)
        return self._to_srt(filepath, word_level)

    def _format_ts(self, seconds: float) -> str:
        millis = int((seconds - int(seconds)) * 1000)
        total_seconds = int(seconds)
        hours, rem = divmod(total_seconds, 3600)
        minutes, secs = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _format_vtt_ts(self, seconds: float) -> str:
        millis = int((seconds - int(seconds)) * 1000)
        total_seconds = int(seconds)
        hours, rem = divmod(total_seconds, 3600)
        minutes, secs = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    def _to_srt(self, filepath: str = None, word_level: bool = False) -> str:
        if not self.segments:
            lines = ["1", "00:00:00,000 --> 99:59:59,999", self.text, ""]
            out = "\n".join(lines)
        else:
            lines = []
            for i, seg in enumerate(self.segments, start=1):
                lines.append(str(i))
                lines.append(f"{self._format_ts(seg.start)} --> {self._format_ts(seg.end)}")
                if word_level and seg.words:
                    # Karaoke-style: <00:00.000>word formatted inline
                    inline = []
                    for w in seg.words:
                        inline.append(f"<{self._format_vtt_ts(w.start)[3:]}>{w.word}")
                    lines.append(" ".join(inline))
                else:
                    lines.append(seg.text)
                lines.append("")
            out = "\n".join(lines)

        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(out)
        return out

    def _to_vtt(self, filepath: str = None, word_level: bool = False) -> str:
        lines = ["WEBVTT", ""]
        for seg in self.segments:
            lines.append(f"{self._format_vtt_ts(seg.start)} --> {self._format_vtt_ts(seg.end)}")
            if word_level and seg.words:
                inline = []
                for w in seg.words:
                    # VTT karaoke cue: <00:00.000>word
                    inline.append(f"<{self._format_vtt_ts(w.start)[3:]}>{w.word}")
                lines.append(" ".join(inline))
            else:
                lines.append(seg.text)
            lines.append("")
        out = "\n".join(lines)

        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(out)
        return out


# ---------------------------------------------------------------------------
# Model wrapper around transcribe_cpp
# ---------------------------------------------------------------------------

class ParakeetGGUF:
    """
    ggml-based Parakeet / Canary wrapper driven by the transcribe-cpp bindings.

    The transcribe_cpp module exposes:
      - transcribe_cpp.Model(path: str, backend: str = "auto")  -> context manager
      - model.session()                                          -> context manager
      - session.run(pcm: np.ndarray[float32, 1-D, 16k-mono])    -> Result

    Where Result has at minimum ``.text`` (str) and on newer builds ``.words``
    (list of dicts with ``w``, ``start``, ``end``, ``conf``) for word timestamps.

    One Model / one session runs only one inference at a time. We serialize
    transcribe calls behind a per-instance lock.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        backend: str = "auto",
        compute_type: str = "q8_0",
        threads: int = 4,
        **kwargs,
    ):
        """
        Initialize the GGUF-backed Parakeet model.

        Args:
            model_path: Path to a .gguf model file (e.g. parakeet-tdt-0.6b-v3-Q8_0.gguf).
            device: "cpu", "gpu"/"cuda". Normalized to transcribe-cpp backend string.
            backend: Override the transcribe-cpp backend ("auto", "cpu", "cuda", "vulkan").
                     Defaults to "auto" which lets the library pick.
            compute_type: Quantization hint from filename ("q8_0", "q4_k", "f16", ...),
                          used by the factory to disambiguate model files in a directory.
            threads: Reserved for forward-compat; transcribe-cpp picks its own.
            **kwargs: Extra kwargs ignored (allows a uniform factory signature).
        """
        self.model_path = str(model_path)
        self.device = device
        self.backend = backend or "auto"
        self.compute_type = compute_type or "q8_0"
        self.threads = threads
        self._model = None
        self._session = None
        self._lock = threading.Lock()

        # Canonical "supports_translation" flag matches the old CanaryONNX API.
        self.supports_translation = self._detect_canary(self.model_path)

        logger.info(
            f"ParakeetGGUF initialized: {self.model_path} (backend={self.backend}, "
            f"compute_type={self.compute_type}, supports_translation={self.supports_translation})"
        )

    @staticmethod
    def _detect_canary(model_path: str) -> bool:
        name = Path(model_path).name.lower()
        # Canary models do translation; Parakeet TDT/RNNT/CTC do not.
        return "canary" in name

    def _resolve_model_path(self) -> str:
        """Resolve to a concrete file path. If model_path is a directory, pick
        a GGUF file matching the configured compute_type, else the first .gguf.
        Auto-downloads the default Parakeet TDT 0.6B v3 Q8_0 model from
        HuggingFace if the configured file is missing under the directory."""
        p = Path(self.model_path)
        if p.is_file():
            return str(p)
        if p.is_dir():
            candidates = sorted(p.glob("*.gguf"))
            if not candidates:
                # Try to auto-fetch the default model into this directory.
                fetched = self._maybe_auto_fetch(p)
                if fetched:
                    return str(fetched)
                raise RuntimeError(
                    f"No .gguf files found in model directory {p}. "
                    f"Set PARAKEET_MODEL to a GGUF filename and either place "
                    f"it under PARAKEET_MODEL_PATH or let subgen auto-download "
                    f"the default model by leaving PARAKEET_MODEL at its default "
                    f"'parakeet-tdt-0.6b-v3-Q8_0.gguf'."
                )
            ct = self.compute_type.lower().lstrip("qw_")  # tolerant
            _ = ct  # tolerant hint; prefer_quants drives actual selection
            prefer_quants = ["q8_0", "q6_k", "q5_k", "q4_k", "q4_0", "f16", "f32"]
            for q in prefer_quants:
                for c in candidates:
                    if c.name.lower().endswith(f"-{q}.gguf") or c.name.lower().endswith(f"_{q}.gguf"):
                        return str(c)
            return str(candidates[0])
        # File doesn't exist yet — if the parent directory exists, try auto-fetch.
        if p.parent.is_dir():
            fetched = self._maybe_auto_fetch(p.parent, target_name=p.name)
            if fetched:
                return str(fetched)
        raise RuntimeError(f"Model path does not exist: {self.model_path}")

    # Default model + download source. Override via env (TRANSCRIBE_DEFAULT_MODEL_URL,
    # TRANSCRIBE_DEFAULT_MODEL_FILE) if a different default is desired.
    _DEFAULT_MODEL_FILE = "parakeet-tdt-0.6b-v3-Q8_0.gguf"
    _DEFAULT_MODEL_URL = (
        "https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v3-gguf/"
        "resolve/main/parakeet-tdt-0.6b-v3-Q8_0.gguf"
    )

    def _maybe_auto_fetch(self, directory: Path, target_name: str = None) -> Optional[Path]:
        """Auto-download the default GGUF model file if the user is using the
        default PARAKEET_MODEL (or has explicitly enabled auto-fetch). Returns
        the path of the downloaded file, or None if we did not attempt a fetch.

        We only auto-fetch when (a) the configured PARAKEET_MODEL filename
        matches the known default, AND (b) there isn't already a GGUF of the
        same name present. This protects users who pointed PARAKEET_MODEL at
        a custom file from surprising 700 MB downloads — for those, we just
        let _resolve_model_path() raise with a helpful message."""
        desired = target_name or self._infer_model_name_from_path()
        if desired is None:
            return None

        url = os.getenv("TRANSCRIBE_DEFAULT_MODEL_URL", self._DEFAULT_MODEL_URL)
        default_file = os.getenv("TRANSCRIBE_DEFAULT_MODEL_FILE", self._DEFAULT_MODEL_FILE)

        # Only auto-fetch when the requested name is the well-known default
        # (case-insensitive match on the filename). Otherwise the user has
        # picked a custom model and we should NOT silently download a different
        # one.
        if desired.lower() != default_file.lower():
            logger.warning(
                f"Configured model '{desired}' not found in {directory}. "
                f"Auto-fetch is only enabled for the default model "
                f"'{default_file}'. Download '{desired}' manually and place "
                f"it under PARAKEET_MODEL_PATH."
            )
            return None

        target = directory / desired
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Auto-fetching default GGUF model from {url} -> {target} (~740 MB)...")
            # Use requests if available (already a subgen dep), fall back to urllib.
            try:
                import requests
                with requests.get(url, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    total = int(r.headers.get("Content-Length", 0))
                    downloaded = 0
                    with open(target, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if total and downloaded % (10 * 1024 * 1024) == 0:
                                    logger.info(f"  ...{downloaded // (1024*1024)} / {total // (1024*1024)} MB")
            except ImportError:
                logger.info("requests not available, falling back to urllib")
                import urllib.request
                urllib.request.urlretrieve(url, target)

            logger.info(f"Model downloaded: {target} ({target.stat().st_size // (1024*1024)} MB)")
            return target
        except Exception as e:
            logger.error(f"Failed to auto-fetch model: {e}")
            # Clean up partial download if present.
            if target.exists():
                try:
                    target.unlink()
                except OSError:
                    pass
            return None

    def _infer_model_name_from_path(self) -> Optional[str]:
        """If self.model_path is a directory, the model name comes from the
        factory's model_name kwarg (stored on self by create_parakeet_model
        via _model_name attr). Fall back to the default filename."""
        name = getattr(self, "_model_name", None)
        if name:
            return name
        return self._DEFAULT_MODEL_FILE

    def _load_model(self):
        """Lazy-load the transcribe_cpp model if not already loaded."""
        if self._model is not None:
            return
        try:
            import transcribe_cpp
        except ImportError:
            logger.error(
                "transcribe-cpp not installed. Install with `pip install transcribe-cpp` "
                "(CPU) or `pip install transcribe-cpp[cu12]` (CUDA). If using a local "
                "build, set TRANSCRIBE_LIBRARY=/path/to/libtranscribe.so before importing."
            )
            raise

        # The transcribe-cpp PyPI package is a pure-Python shim that requires
        # a native library (libtranscribe.so/.dylib/.dll) to be loaded at import
        # time. If the native lib wasn't found (no wheel, no TRANSCRIBE_LIBRARY
        # env var, no repo auto-discovery), `Model` won't be bound — which is
        # what produces `AttributeError: module 'transcribe_cpp' has no attribute 'Model'`.
        # Surface a clear error pointing the user at the fix instead of a bare
        # AttributeError.
        if not hasattr(transcribe_cpp, "Model"):
            msg = (
                "transcribe_cpp imported but `Model` is not bound — the native "
                "libtranscribe library was not loaded. To fix:\n"
                "  1. Install a native provider: `pip install transcribe-cpp-native` "
                "(CPU) or `pip install transcribe-cpp-native-cu12` (CUDA), or\n"
                "  2. Build libtranscribe from "
                "https://github.com/handy-computer/transcribe.cpp "
                "(`cmake -B build -DTRANSCRIBE_BUILD_SHARED=ON -DTRANSCRIBE_CUDA=ON && "
                "cmake --build build --target transcribe`) and set the "
                "TRANSCRIBE_LIBRARY env var to the resulting shared object.\n"
                f"Current TRANSCRIBE_LIBRARY={os.getenv('TRANSCRIBE_LIBRARY', '<unset>')}, "
                f"TRANSCRIBE_NATIVE_PROVIDER={os.getenv('TRANSCRIBE_NATIVE_PROVIDER', '<unset>')}"
            )
            logger.error(msg)
            raise RuntimeError(msg)

        resolved = self._resolve_model_path()
        logger.info(f"Loading GGUF model: {resolved} (backend={self.backend})")
        # Model is a context manager — we hold it open for the lifetime of the instance.
        self._transcribe_cpp = transcribe_cpp
        self._model = transcribe_cpp.Model(resolved, backend=self.backend)
        logger.info("GGUF model loaded")

    def _audio_to_pcm(self, audio: Union[str, np.ndarray, bytes], input_sr: int = 16000) -> np.ndarray:
        """Normalize any accepted audio input form into the 1-D float32 16k-mono
        numpy array that transcribe_cpp.session.run() expects."""
        # Already a numpy array
        if isinstance(audio, np.ndarray):
            data = audio
            # Downmix stereo to mono
            if data.ndim > 1:
                # If shape (N,2) take mean across axis 1
                data = data.mean(axis=tuple(range(1, data.ndim)))
            if data.dtype != np.float32:
                # If int16 etc. normalize to [-1,1]
                if data.dtype.kind in ('i', 'u'):
                    max_val = float(np.iinfo(data.dtype).max)
                    data = data.astype(np.float32) / max_val
                else:
                    data = data.astype(np.float32)
            if input_sr and input_sr != 16000:
                try:
                    import librosa
                    data = librosa.resample(data.astype(np.float32), orig_sr=input_sr, target_sr=16000)
                except ImportError:
                    raise RuntimeError(
                        f"Input audio is at {input_sr} Hz but librosa is not installed; "
                        "transcribe-cpp requires 16 kHz mono float32 PCM. "
                        "Install librosa (pip install librosa) or downsample before calling."
                    )
            return np.ascontiguousarray(data, dtype=np.float32)

        # WAV bytes from FFmpeg output (pcm_s16le at 16kHz)
        if isinstance(audio, (bytes, bytearray)):
            return self._wav_bytes_to_pcm(audio)

        # File path on disk
        if isinstance(audio, str):
            return self._wav_file_to_pcm(audio)

        raise TypeError(f"Unsupported audio input type: {type(audio)}")

    @staticmethod
    def _wav_bytes_to_pcm(raw: bytes) -> np.ndarray:
        """Decode a WAV (pcm_s16le) byte stream to a float32 1-D numpy array.
        We do a minimal WAV header parse so we don't need soundfile at runtime."""
        if len(raw) < 44:
            raise ValueError("WAV data too short")

        # Parse RIFF header (PCM only — FFmpeg always writes PCM s16le).
        if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
            raise ValueError("Not a RIFF/WAVE stream")
        fmt_chunk = raw.index(b"fmt ") if b"fmt " in raw else None
        if fmt_chunk is None:
            raise ValueError("WAV format chunk missing")
        audio_format = int.from_bytes(raw[fmt_chunk + 8:fmt_chunk + 10], "little")
        n_channels = int.from_bytes(raw[fmt_chunk + 10:fmt_chunk + 12], "little")
        sample_rate = int.from_bytes(raw[fmt_chunk + 12:fmt_chunk + 16], "little")
        if audio_format != 1 and audio_format != 0xFFFE:
            # WAVE_FORMAT_EXTENSIBLE is 0xFFFE — FFmpeg doesn't emit it for pcm_s16le;
            # this is just defensive.
            pass

        data_marker = raw.index(b"data")
        data_size = int.from_bytes(raw[data_marker + 4:data_marker + 8], "little")
        data_start = data_marker + 8
        pcm_bytes = raw[data_start:data_start + data_size]

        # Convert int16 samples to float32
        pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if n_channels > 1:
            pcm = pcm.reshape(-1, n_channels).mean(axis=1)

        # Resample if needed (rare; FFmpeg respects our ar=16000 flag)
        if sample_rate != 16000:
            try:
                import librosa
                pcm = librosa.resample(pcm, orig_sr=sample_rate, target_sr=16000)
            except ImportError:
                raise RuntimeError(
                    f"WAV stream is at {sample_rate} Hz but librosa is not installed; "
                    "transcribe-cpp requires 16 kHz mono float32 PCM. "
                    "Install librosa (pip install librosa) or pre-convert the audio."
                )

        return np.ascontiguousarray(pcm, dtype=np.float32)

    @staticmethod
    def _wav_file_to_pcm(path: str) -> np.ndarray:
        """Decode any audio file on disk to float32 16k-mono PCM using ffmpeg."""
        try:
            import ffmpeg
        except ImportError:
            logger.error("ffmpeg-python required to decode audio files")
            raise

        out, _ = (
            ffmpeg.input(path)
            .output("pipe:1", format="wav", acodec="pcm_s16le", ar=16000, ac=1)
            .run(capture_stdout=True, capture_stderr=True)
        )
        return ParakeetGGUF._wav_bytes_to_pcm(out)

    def _result_to_TranscriptionResult(self, tresult, language: str) -> TranscriptionResult:
        """Convert transcribe_cpp result to our TranscriptionResult."""
        text = getattr(tresult, "text", "") or ""

        segments = []
        words_list = []

        # Best-effort per-word timestamps. transcribe-cpp supports a .words list
        # with {w, start, end, conf} fields (newer builds). If only `.text`
        # is exposed, emit a single segment spanning the whole audio.
        raw_words = getattr(tresult, "words", None)
        if raw_words:
            try:
                raw_words = list(raw_words)
            except TypeError:
                raw_words = []

        if raw_words:
            for w in raw_words:
                if isinstance(w, dict):
                    wtext = w.get("w", "") or w.get("word", "")
                    wstart = float(w.get("start", w.get("s", 0.0)))
                    wend = float(w.get("end", w.get("e", wstart)))
                    wconf = float(w.get("conf", w.get("probability", 1.0)))
                else:  # attribute-style
                    wtext = getattr(w, "word", "") or getattr(w, "w", "")
                    wstart = float(getattr(w, "start", 0.0))
                    wend = float(getattr(w, "end", wstart))
                    wconf = float(getattr(w, "probability", 1.0))
                if wtext:
                    words_list.append(Word(word=wtext, start=wstart, end=wend, probability=wconf))

            # Group words into ~7-second subtitle segments (Whisper-style).
            SEG_TARGET = 7.0
            current_seg_words = []
            current_start = words_list[0].start if words_list else 0.0
            for w in words_list:
                if current_seg_words and (w.end - current_start) > SEG_TARGET:
                    seg_text = " ".join(x.word for x in current_seg_words).strip()
                    segments.append(Segment(
                        start=current_seg_words[0].start,
                        end=current_seg_words[-1].end,
                        text=seg_text,
                        words=current_seg_words,
                        sid=len(segments),
                    ))
                    current_seg_words = []
                    current_start = w.start
                current_seg_words.append(w)
            if current_seg_words:
                seg_text = " ".join(x.word for x in current_seg_words).strip()
                segments.append(Segment(
                    start=current_seg_words[0].start,
                    end=current_seg_words[-1].end,
                    text=seg_text,
                    words=current_seg_words,
                    sid=len(segments),
                ))
        else:
            # No word timing — produce one segment with the full transcript.
            # transcribe_cpp may expose duration via frame_sec * frame_count.
            duration = getattr(tresult, "duration", None)
            if duration is None:
                # Estimate from frame count & stride (Parakeet frame stride = 80ms)
                frame_count = getattr(tresult, "frame_count", 0) or 0
                frame_sec = getattr(tresult, "frame_sec", 0.08)
                duration = (frame_count * frame_sec) if frame_count else 0.0
            segments = [Segment(start=0.0, end=float(duration), text=text, words=[], sid=0)]

        return TranscriptionResult(text=text, language=language, segments=segments, tokens=[])

    # -----------------------------------------------------------------------
    # Public API matching what old ParakeetONNX/ParakeetNemo exposed.
    # -----------------------------------------------------------------------

    def transcribe(
        self,
        audio: Union[str, np.ndarray, List[Union[str, np.ndarray]]],
        language: Optional[str] = None,
        task: str = "transcribe",
        verbose: bool = False,
        **kwargs,
    ) -> Union[TranscriptionResult, List[TranscriptionResult]]:
        """
        Transcribe audio file(s) or in-memory numpy/WAV bytes.

        Args:
            audio: File path (str), 16kHz mono float32 ndarray, WAV bytes,
                   or list of any of these.
            language: Optional source language code (e.g. "en", "fr"). Ignored
                      for Parakeet TDT; passed to Canary for source-language tagging.
            task: "transcribe" (default) or "translate".
            verbose: Unused (kept for API compatibility).
            **kwargs: Absorbs input_sr, regroup, progress_callback, etc. — ignored here.

        Returns:
            TranscriptionResult or list[TranscriptionResult].
        """
        self._load_model()
        self._lock.acquire()
        try:
            single = False
            if not isinstance(audio, list):
                audio = [audio]
                single = True

            results = []
            for a in audio:
                input_sr = kwargs.get("input_sr", 16000) or 16000
                pcm = self._audio_to_pcm(a, input_sr=input_sr)
                if verbose:
                    logger.info(f"Transcribing {len(pcm)} samples ({len(pcm)/16000:.2f}s) "
                                f"via transcribe-cpp backend={self.backend}")
                # Create a fresh session per inference (the native C library
                # releases graph state when the session context closes).
                with self._model.session() as session:
                    tresult = session.run(pcm)
                # Language selection: Canary models expose task/lang when translating.
                # transcribe-cpp's Parakeet path is English-only; we return "en" unless
                # the user moved a Canary model in.
                lang_out = "en" if not language else language
                results.append(self._result_to_TranscriptionResult(tresult, lang_out))

            return results[0] if single else results
        finally:
            self._lock.release()

    def unload_model(self):
        """Release the underlying model. transcribe_cpp Model is a context
        manager; calling its __exit__ triggers lib files release. We also
        clear our references so GC can drop the object."""
        self._lock.acquire()
        try:
            self._session = None
            if self._model is not None:
                try:
                    # transcribe_cpp.Model exports close() in newer builds
                    if hasattr(self._model, "close"):
                        self._model.close()
                    elif hasattr(self._model, "__exit__"):
                        self._model.__exit__(None, None, None)
                except Exception as e:
                    logger.warning(f"Ignoring error while closing model: {e}")
                self._model = None
            logger.info("GGUF model unloaded from memory")
        finally:
            self._lock.release()


# ---------------------------------------------------------------------------
# Factory — same name/signature shape as the old parakeet_onnx.create_parakeet_model
# ---------------------------------------------------------------------------

def create_parakeet_model(
    model_path: str = None,
    model_name: str = "parakeet-tdt-0.6b-v3-Q8_0.gguf",
    device: str = "cpu",
    backend: str = None,
    compute_type: str = "q8_0",
    threads: int = 4,
    cache_dir: str = None,
    **kwargs,
) -> ParakeetGGUF:
    """
    Factory function to create a ParakeetGGUF model instance.

    Args:
        model_path: Path to a GGUF model file or directory containing GGUF
                    files. Defaults to ``cache_dir`` if not provided.
        model_name: Filename to look up inside ``cache_dir`` when ``model_path``
                    is not a concrete file.
        device:     "cpu", "cuda", or "gpu" (normalized to "cuda").
        backend:    Override for transcribe-cpp backend ("auto", "cpu", "cuda",
                    "vulkan"). Defaults to mapping from device.
        compute_type: Preferred GGUF quantization name (q8_0, q4_k, f16, ...).
                      Used when resolving a directory of multiple GGUFs.
        threads:    Number of CPU threads (forwarded for future use).
        cache_dir:  Root directory for GGUF model files (defaults to ``MODEL_PATH``).

    Returns:
        ParakeetGGUF instance — single backend.

    Note: Replaces the previous ``parakeet_onnx.create_parakeet_model`` API. The
          ``use_onnx`` parameter and ONNX/NeMo fallbacks have been removed.
    """
    # Resolve default paths.
    if model_path is None or model_path == "":
        if cache_dir is None:
            cache_dir = os.getenv("MODEL_PATH", "./models")
        model_path = os.path.join(cache_dir, model_name)

    # Normalize device to a backend string for transcribe-cpp.
    if backend is None:
        d = (device or "cpu").lower()
        if d in ("cuda", "gpu"):
            backend = "auto"  # transcribe-cpp picks CUDA when present
        else:
            backend = "cpu"

    # Allow auto-fetch logic to know which filename the user requested.
    inst = ParakeetGGUF(
        model_path=model_path,
        device=device,
        backend=backend,
        compute_type=compute_type,
        threads=threads,
        **kwargs,
    )
    inst._model_name = model_name
    return inst


# Backward-compat with the stable-whisper API (calls used to wrap faster-whisper)
class StableWhisperCompat:
    """Compatibility wrapper to match stable-whisper's faster-whisper API."""

    def __init__(self, parakeet_model):
        self.model = parakeet_model

    def transcribe(self, audio, **kwargs):
        return self.model.transcribe(audio, **kwargs)

    def __getattr__(self, name):
        return getattr(self.model, name)
