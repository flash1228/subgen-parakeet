"""
conftest.py — patches heavy ML/AV dependencies before any test module imports subgen.

The mock setup must happen here, at collection time, before subgen.py is imported.
subgen.py starts worker threads at import time; they're daemon threads and are harmless.

GGUF migration note: nemo / onnxruntime / sentencepiece / torchaudio / soundfile /
librosa are no longer used at runtime (subgen now ships transcribe-cpp / ggml).
They are still mocked here so legacy test imports don't blow up if you happen to
install them in dev environments.
"""
import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock heavy dependencies that are not installed in CI
# ---------------------------------------------------------------------------
_MOCKED_MODULES = [
    # Legacy Parakeet stack — no longer imported by subgen.py but kept mocked
    # so any leftover third-party imports inside tests still resolve.
    "nemo",
    "nemo.collections",
    "nemo.collections.asr",
    "nemo.collections.asr.models",
    "nemo.collections.asr.parts",
    "nemo.collections.asr.parts.mixins",
    "nemo.collections.asr.parts.submodules",
    "nemo.collections.asr.parts.utils",
    "nemo.utils",
    "nemo.utils.export_utils",
    "lightning",
    "lightning.pytorch",
    "omegaconf",
    "torch",
    "av",
    "ffmpeg",
    "watchdog",
    "watchdog.observers",
    "watchdog.observers.polling",
    "watchdog.events",
    "numpy",
    # Legacy audio stack — transcribe-cpp handles audio internally now, but
    # some helper paths still have optional librosa import guards.
    "torchaudio",
    "soundfile",
    "librosa",
    "sentencepiece",
    "onnxruntime",
    # New GGUF runtime
    "transcribe_cpp",
]

for _mod in _MOCKED_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Mock NeMo ASR model classes (kept for any test that still references them)
sys.modules["nemo.collections.asr.models"].ASRModel = MagicMock()
sys.modules["nemo.collections.asr.models"].EncDecRNNTBPEModel = MagicMock()
sys.modules["nemo.collections.asr.models"].EncDecCTCModelBPE = MagicMock()
sys.modules["nemo.collections.asr.models"].EncDecHybridRNNTCTCModel = MagicMock()

# Mock NeMo TranscribeConfig
sys.modules["nemo.collections.asr.parts.mixins"].TranscribeConfig = MagicMock()

# Mock torch CUDA shims used by the optional VRAM-clear path
sys.modules["torch"].cuda.is_available = MagicMock(return_value=False)
sys.modules["torch"].cuda.empty_cache = MagicMock()

# Ensure watchdog attribute imports work
# e.g. `from watchdog.observers.polling import PollingObserver as Observer`
sys.modules["watchdog.observers.polling"].PollingObserver = MagicMock()
# e.g. `from watchdog.events import FileSystemEventHandler`
sys.modules["watchdog.events"].FileSystemEventHandler = object

# Mock the legacy parakeet_onnx module in case any test still imports it
# (subgen.py itself no longer imports it after the GGUF migration)
sys.modules["parakeet_onnx"] = MagicMock()
sys.modules["parakeet_onnx"].create_parakeet_model = MagicMock()
sys.modules["parakeet_onnx"].TranscriptionResult = MagicMock()
sys.modules["parakeet_onnx"].ParakeetONNX = MagicMock()
sys.modules["parakeet_onnx"].CanaryONNX = MagicMock()
sys.modules["parakeet_onnx"].ParakeetNemo = MagicMock()

# Mock parakeet_gguf module (the new GGUF backend)
sys.modules["parakeet_gguf"] = MagicMock()
sys.modules["parakeet_gguf"].create_parakeet_model = MagicMock()
sys.modules["parakeet_gguf"].TranscriptionResult = MagicMock()
sys.modules["parakeet_gguf"].ParakeetGGUF = MagicMock()
sys.modules["parakeet_gguf"].Word = MagicMock()
sys.modules["parakeet_gguf"].Segment = MagicMock()
sys.modules["parakeet_gguf"].StableWhisperCompat = MagicMock()
