"""
conftest.py — patches heavy ML/AV dependencies before any test module imports subgen.

The mock setup must happen here, at collection time, before subgen.py is imported.
subgen.py starts worker threads at import time; they're daemon threads and are harmless.
"""
import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock heavy dependencies that are not installed in CI
# ---------------------------------------------------------------------------
_MOCKED_MODULES = [
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
    "torchaudio",
    "soundfile",
    "librosa",
    "sentencepiece",
    "onnxruntime",
]

for _mod in _MOCKED_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Mock NeMo ASR model classes
sys.modules["nemo.collections.asr.models"].ASRModel = MagicMock()
sys.modules["nemo.collections.asr.models"].EncDecRNNTBPEModel = MagicMock()
sys.modules["nemo.collections.asr.models"].EncDecCTCModelBPE = MagicMock()
sys.modules["nemo.collections.asr.models"].EncDecHybridRNNTCTCModel = MagicMock()

# Mock NeMo TranscribeConfig
sys.modules["nemo.collections.asr.parts.mixins"].TranscribeConfig = MagicMock()

# Mock torch
sys.modules["torch"].cuda.is_available = MagicMock(return_value=False)
sys.modules["torch"].cuda.empty_cache = MagicMock()

# Ensure watchdog attribute imports work
# e.g. `from watchdog.observers.polling import PollingObserver as Observer`
sys.modules["watchdog.observers.polling"].PollingObserver = MagicMock()
# e.g. `from watchdog.events import FileSystemEventHandler`
sys.modules["watchdog.events"].FileSystemEventHandler = object

# Mock parakeet_onnx module
sys.modules["parakeet_onnx"] = MagicMock()
sys.modules["parakeet_onnx"].create_parakeet_model = MagicMock()
sys.modules["parakeet_onnx"].TranscriptionResult = MagicMock()
sys.modules["parakeet_onnx"].ParakeetONNX = MagicMock()
sys.modules["parakeet_onnx"].CanaryONNX = MagicMock()
sys.modules["parakeet_onnx"].ParakeetNemo = MagicMock()