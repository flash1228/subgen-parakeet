# Subgen — Agent Instructions

## Project Overview
Subgen is a Python FastAPI service that generates subtitles (`.srt`, `.lrc`) from audio/video using `faster-whisper` + `stable-ts`. Integrates with **Bazarr** (Whisper provider), **Plex**, **Jellyfin**, **Emby**, **Tautulli** via webhooks. Runs on CPU or NVIDIA GPU (CUDA); experimental AMD/ROCm support.

## Key Entry Points
| File | Purpose |
|------|---------|
| `subgen.py` | Main FastAPI app — all webhooks, queue, transcription worker, skip logic |
| `language_code.py` | `LanguageCode` enum (ISO 639-1/2/T/B, names) |
| `launcher.py` | Standalone installer/updater with Bazarr wizard |
| `entrypoint.sh` | Docker entrypoint (PUID/PGID, permissions, rootless) |

## Developer Commands
```bash
# Install test deps (ML packages mocked in CI)
pip install -r requirements-test.txt

# Run tests (pytest, timeout 60s)
pytest tests/ -v --tb=short --timeout=60

# Lint (ruff 0.15.12)
ruff check --ignore E402,PLW1508 subgen.py language_code.py
ruff check --select I --ignore E402,PLW1508 subgen.py language_code.py

# Run standalone (Python 3.9–3.11 + ffmpeg)
python3 launcher.py -u -i -s   # update, install deps, setup bazarr wizard
```

## Architecture Notes
- **Single-file app**: `subgen.py` is ~2500 lines — all logic inline (webhooks, queue, worker, skip logic, Plex/Jellyfin API helpers).
- **Queue**: `DeduplicatedQueue` (priority: detect_language=0, asr=1, transcribe=2) with thread-safe deduplication.
- **Workers**: `CONCURRENT_TRANSCRIPTIONS` daemon threads run `transcription_worker()`.
- **Model lifecycle**: Lazy-loaded via `start_model()`; VRAM cleanup scheduled via `delete_model()` → `schedule_model_cleanup()` → `perform_model_cleanup()` (delay `MODEL_CLEANUP_DELAY`, default 30s).
- **Skip logic**: `should_skip_file()` — checks audio tracks, embedded/external subs, `.subgen_skip` marker files in directories.
- **Audio offset fix**: `get_audio_start_time()` uses `ffprobe` to detect container audio `start_time`; `apply_timestamp_offset()` shifts Whisper timestamps (fixes Amazon WEB-DL early subs).

## Testing Conventions
- **Heavy deps mocked** in `tests/conftest.py`: `stable_whisper`, `faster_whisper`, `torch`, `av`, `ffmpeg`, `watchdog`, `numpy`.
- Tests patch module globals (`monkeypatch.setattr(subgen, "var", value)`) and filesystem calls.
- Test files: `test_skip_logic.py`, `test_queue.py`, `test_language_code.py`, `test_integration.py`, `test_helpers.py`, `test_endpoints.py`, `test_bug_fixes.py`, `test_audio_tracks.py`.

## CI / Pre-commit
- **GitHub Actions**: `test.yml` (Python 3.11, 3.12), `lint.yml` (ruff), `build_CPU.yml`, `build_GPU.yml`, `build_amd.yml`, `bump_version.yml`.
- **Pre-commit hook** (`.githooks/pre-commit`): Auto-bumps `subgen_version` (YYYY.MM.patch) when `subgen.py` is staged.

## Environment Variables (Key Ones)
All read via `get_env_with_fallback(new_name, old_name, default, convert_func)` for backwards compat.

| Variable | Default | Notes |
|----------|---------|-------|
| `TRANSCRIBE_DEVICE` | `cpu` | `cpu`, `gpu`/`cuda` |
| `WHISPER_MODEL` | `medium` | `large-v3-turbo` = transcribe only (no translate) |
| `CONCURRENT_TRANSCRIPTIONS` | `2` | Worker threads |
| `WEBHOOK_PORT` | `9000` | |
| `MODEL_PATH` | `./models` | Model cache dir |
| `CLEAR_VRAM_ON_COMPLETE` | `True` | |
| `MODEL_CLEANUP_DELAY` | `30` | Seconds |
| `SKIP_STARTUP_SCAN` | `False` | Gates lifespan thread only; `/batch` unaffected |
| `IGNORE_FORCED_SUBTITLES` | `True` | Excludes forced tracks from skip coverage |

## Docker
- Images: `mccloud/subgen:latest` (CPU+GPU), `:cpu`, `:cuda`, `:amd` (ROCm).
- `docker-compose.yml` mounts `${TV}:/tv`, `${MOVIES}:/movies`, `${APPDATA}/subgen/models:/subgen/models`.
- GPU: uncomment `deploy.resources.reservations.devices` for NVIDIA; see compose for AMD/ROCm flags (`CT2_CUDA_ALLOCATOR`, `HSA_OVERRIDE_GFX_VERSION`).

## Common Gotchas
- **Bazarr**: Must set "Pass Video Name" + mount media paths identically for audio offset fix.
- **Skip logic**: `.subgen_skip` in a directory skips that entire subtree (startup scan + monitor).
- **Version**: `subgen_version = '2026.07.3'` at top of `subgen.py`; auto-bumped by pre-commit.
- **Legacy env names**: `PLEXTOKEN` → `PLEX_TOKEN`, `PLEXSERVER` → `PLEX_SERVER`, etc. (see docstring in `subgen.py`).
- **Tests**: Never install `stable-ts`/`faster-whisper`/`torch` in CI — they're mocked.