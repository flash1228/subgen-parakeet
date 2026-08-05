# Subgen

> **Original work by [McCloudS](https://github.com/McCloudS/subgen)**  
> This fork was created with AI-assisted coding to replace Whisper with **NVIDIA Parakeet** (FastConformer TDT/RNNT) for faster, more accurate transcription.

[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/donate/?hosted_button_id=SU4QQP6LH5PF6)
<img src="https://raw.githubusercontent.com/flash1228/subgen-parakeet/main/icon.png" width="200">

<details>
<summary><strong>Updates:</strong></summary>

7 Jun 2026: Fixed a bug where files containing only a **forced** embedded subtitle track were incorrectly treated as having full subtitle coverage and skipped. Forced tracks cover only a small fraction of dialogue (typically foreign-language inserts) and should not count as full coverage. Added `IGNORE_FORCED_SUBTITLES` (default `True`) to control this behaviour.

11 Apr 2026: Fixed subtitle timing on files with audio stream offsets (common in Amazon WEB-DL). Parakeet ignores silence padding, causing subtitles to be early by the offset amount. Subgen now detects this via ffprobe and compensates automatically when the source video file is accessible. See [Audio Start-Time Offset Fix](#-audio-start-time-offset-fix) for details.

27 Mar 2026: Potentially added ROCm support for AMD GPU/APUs. Image is: `flash1228/subgen-parakeet:amd`.

17 Mar 2026: Added `WEBHOOK_URL_COMPLETED`. When a task finishes, Subgen will send a POST request with a JSON structure.

4 Mar 2026: Reformatted the readme.

Feb 2026: Contributor helped cut the GPU container size in half. Added `ASR_TIMEOUT` as environment variable to timeout ASR endpoint transcriptions after X seconds.

31 Jan 2026: Added the ability to run the container 'rootless', accepts `PUID` and `PGID` as environment variables.

13 Jan 2026: Fixed runaway memory problems for CPU only. Added `MODEL_CLEANUP_DELAY` which will wait X seconds before purging the model to clear up (V)RAM.

26 Aug 2025: Renamed environment variables to make them slightly easier to understand. Currently maintains backwards compatibility.

12 Aug 2025: Added distil-large-v3.5

7 Feb 2025: Fixed (V)RAM clearing, added PLEX_QUEUE_SEASON, other fixes.

23 Dec 2025: Added PLEX_QUEUE_NEXT_EPISODE and PLEX_QUEUE_SERIES.

4 Dec 2025: Added more ENV settings: DETECT_LANGUAGE_OFFSET, PREFERRED_AUDIO_LANGUAGES, SKIP_IF_AUDIO_TRACK_IS, ONLY_SKIP_IF_SUBGEN_SUBTITLE, SKIP_UNKNOWN_LANGUAGE, SKIP_IF_LANGUAGE_IS_NOT_SET_BUT_SUBTITLES_EXIST, SHOULD_PARAKEET_DETECT_AUDIO_LANGUAGE

30 Nov 2024: Significant refactoring. Added language code class, audio track separation. New ENV Variables: SUBTITLE_LANGUAGE_NAMING_TYPE, SKIP_IF_AUDIO_TRACK_IS, PREFERRED_AUDIO_LANGUAGE, SKIP_IF_TO_TRANSCRIBE_SUB_ALREADY_EXIST

22 Nov 2024: Updated to support large-v3-turbo

30 Sept 2024: Removed webui

5 Sept 2024: Fixed Emby response to a test message/notification.

14 Aug 2024: Cleaned up usage of kwargs across the board. Added ability for /asr to encode or not.

3 Aug 2024: Added SUBGEN_KWARGS environment variable.

21 Apr 2024: Fixed queuing with thanks to xhzhu0628.

31 Mar 2024: Removed `/subsync` endpoint and general refactoring.

24 Mar 2024: Added a 'webui' to configure environment variables (removed later).

23 Mar 2024: Added `CUSTOM_REGROUP` to try to 'clean up' subtitles.

22 Mar 2024: Added LRC capability.

21 Mar 2024: Added a 'wizard' into the launcher. Removed 'Transformers' option. Added `USE_MODEL_PROMPT` and `CUSTOM_MODEL_PROMPT`.

19 Mar 2024: Added a `MONITOR` environment variable.

6 Mar 2024: Added a `/subsync` endpoint (removed later).

5 Mar 2024: Cleaned up logging. Added timestamps option.

4 Mar 2024: Updated Dockerfile CUDA to 12.2.2. Added endpoint `/status`. Can use distil models.

29 Feb 2024: Changed default port to align with whisper-asr.

11 Feb 2024: Added a 'launcher.py' file for Docker to prevent huge image downloads. Added APPEND watermark.

10 Feb 2024: Added features from JaiZed's branch. Added `/batch` endpoint. Added CLEAR_VRAM_ON_COMPLETE.

8 Feb 2024: Added FORCE_DETECTED_LANGUAGE_TO. Fixed asr to actually use the language passed to it.

5 Feb 2024: General housekeeping.

28 Jan 2024: Fixed issue with ffmpeg python module. Removed separate GPU/CPU containers.

19 Dec 2023: Added the ability for Plex and Jellyfin to automatically update metadata.

31 Oct 2023: Added Bazarr support via Whisper provider.

25 Oct 2023: Added Emby support and TRANSCRIBE_FOLDERS.

23 Oct 2023: There are now two docker images.

22 Oct 2023: The script should have backwards compatibility.

19 Oct 2023: And we're back! Uses faster-whisper and stable-ts.

2 Feb 2023: Added Tautulli webhooks back in.

31 Jan 2023 : Rewrote the script substantially to remove Tautulli.

</details>

---

## 🎬 What is this?

Subgen transcribes your personal media to create subtitles (`.srt` or `.lrc`) from audio/video files. It uses **NVIDIA Parakeet** (via NeMo toolkit or ONNX Runtime) for high-accuracy ASR — faster and more accurate than Whisper on supported hardware.

**Parakeet TDT 1.1B** (default): English-only transcription, state-of-the-art accuracy  
**Canary 1B** (optional): Multilingual (25 languages) + translation support

It integrates perfectly with **Bazarr** (Whisper Provider compatible), or runs via webhooks triggered directly by your **Plex, Emby, Jellyfin, or Tautulli** servers whenever media is added or played.

---

## 🤔 Why?

Some shows just won't have subtitles available, or embedded subtitles might be wildly out of sync. This gap-fills everything else by generating highly accurate subtitles locally on your own hardware.

---

## ⚡ Quick Start: Bazarr (The Bare Minimum)

If you just want to plug Subgen into Bazarr and get going, here is the absolute minimum you need to configure in your Subgen Docker container. **No path mapping or media mounts are needed!**

**1. Set your Environment Variables in Subgen:**

* `TRANSCRIBE_DEVICE`: Set to `cuda` if you have an Nvidia GPU (highly recommended for speed), otherwise leave as `cpu`.
* `PARAKEET_MODEL`: Default is `nvidia/parakeet-tdt-1.1b`. Use `nvidia/canary-1b` for multilingual + translation support.
* `PARAKEET_USE_ONNX`: Default `True` (faster inference via ONNX Runtime). Set `False` to use NeMo toolkit directly.
* `CONCURRENT_TRANSCRIPTIONS`: Default is `2`. Lower to `1` if you are running out of RAM/VRAM.

**2. Configure Bazarr:**

* In Bazarr, go to **Settings > Whisper Provider**.
* Select **Whisper** as the provider.
* Set the **Docker Endpoint** to your Subgen IP and port: `http://<your-ip>:9000` *(Note: Do not use `127.0.0.1` if Bazarr is also in a Docker container).*
* Save! Subgen will now act as an invisible, self-hosted API for Bazarr's transcription requests.

**3. Disable Auto-Sync for Subgen subtitles (important):**

Subgen already produces accurately timed subtitles. If you have Bazarr's **Automatic Subtitles Audio Synchronization** enabled, you must exclude `whisperai` from it — otherwise Bazarr will run ffsubsync on top of already-synced subtitles and degrade their quality.
* In Bazarr, go to **Settings > Subtitles > Audio Synchronization**.
* Under **"Do not sync subtitles downloaded from those providers"**, add **`whisperai`**.

---

## 🔧 Audio Start-Time Offset Fix

Some media containers — particularly Amazon WEB-DL files — have an audio stream that starts later than the video stream (e.g., audio `start_time` of ~4 seconds). When Bazarr extracts audio from these files, it compensates by prepending silence via ffmpeg's `adelay` filter. However, Parakeet's speech recognition completely ignores this digital silence, producing timestamps that are early by the offset amount (e.g., every subtitle appears ~4 seconds too early).

Subgen now automatically detects and compensates for this. When the source video file is accessible, it uses `ffprobe` to read the audio stream's `start_time` metadata, then shifts all Parakeet timestamps forward by that amount after transcription.

**This fix is fully backwards compatible.** If the video file is not accessible, or has no audio offset (i.e. `start_time` is 0), behaviour is completely unchanged.

### How to enable it (Bazarr)

1. **Mount your media into the Subgen container** with the same paths that Bazarr sees. For example, if Bazarr sees TV shows at `/tv`, add a volume mount so Subgen also sees `/tv`:
   ```yaml
   volumes:
     - /path/to/your/tv:/tv
     - /path/to/your/movies:/movies
   ```

2. **Enable "Pass Video Name" in Bazarr.** Go to **Settings > Whisper Provider** and check the **Pass Video Name** option. This tells Bazarr to send the video file path alongside the audio, allowing Subgen to look up the source file and detect any audio offset.

That's it. No new environment variables are required. Files without an audio offset are unaffected.

---

## 🛠 Installation & Setup

### 1. Docker (Recommended)

The easiest way to run Subgen is via Docker. We maintain images on Docker Hub (`flash1228/subgen-parakeet`):

* `flash1228/subgen-parakeet:latest` (Supports both CPU and GPU/CUDA)
* `flash1228/subgen-parakeet:cpu` (Smaller image, CPU only)
* `flash1228/subgen-parakeet:amd` (AMD/ROCm experimental)

**Crucial Note on Volume Mapping:** If you are using Plex/Emby/Jellyfin/Tautulli webhooks, **Subgen must see your media paths exactly identically to how your media server sees them.** For example, if Plex uses `/Share/media/TV:/tv`, Subgen needs that exact same volume mount. *(Note: This does not apply to Bazarr, which sends audio over HTTP).*

### 2. Standalone (Without Docker)

1. Install Python 3.9–3.11 and `ffmpeg`.
2. Ensure you have the proper NVIDIA drivers/CUDA toolkit installed (if using GPU).
3. Download `launcher.py` from this repository and run:
   > `python3 launcher.py -u -i -s`

*(Launcher includes a wizard to help standalone users easily configure common variables).*

**ONNX Export (Optional — for faster inference with `PARAKEET_USE_ONNX=True`):**

By default, Subgen uses **ONNX Runtime** (`PARAKEET_USE_ONNX=True`) for faster, lighter inference. This requires exported ONNX models. If you prefer to use NeMo toolkit directly (no export needed), set `PARAKEET_USE_ONNX=False`.

```bash
# Only needed if PARAKEET_USE_ONNX=True (default)
python export_parakeet_onnx.py --model parakeet-tdt-1.1b --output-dir ./models/parakeet_onnx
python export_parakeet_onnx.py --model canary-1b --output-dir ./models/canary_onnx
```

| Mode | Requires ONNX Export? | Pros |
|------|----------------------|------|
| `PARAKEET_USE_ONNX=True` (default) | **Yes** | Faster inference, lower memory, no PyTorch at runtime |
| `PARAKEET_USE_ONNX=False` | No | Uses NeMo directly, simpler setup, full NeMo features |

### 3. Unraid

While Unraid doesn't have an app or template for quick install, with minor manual work, you can easily install it. See [this discussion thread](https://github.com/flash1228/subgen-parakeet/discussions) for pictures and steps.

---

## 🔌 Integrations & Webhooks Setup

Choose your preferred integration below. **Do not enable multiple webhooks for the same media events** (e.g., don't use both Tautulli and Plex webhooks for "playback start"), or you will generate duplicate subtitles!

### 🟠 Plex
Requires Plex Pass. Plex and Subgen must have identical path configurations (or use Path Mapping).
1. In Plex, go to **Settings > Webhooks**.
2. Add a new webhook pointing to your Subgen instance: `http://<your-ip>:9000/plex`
3. You will also need to generate a [Plex Token](https://support.plex.tv/articles/204059436-finding-an-authentication-token-x-plex-token/).
4. **Relevant Variables:** `PLEX_SERVER`, `PLEX_TOKEN`.

### 🔵 Jellyfin
Jellyfin and Subgen must have identical path configurations (or use Path Mapping).
1. Install the **Webhooks** plugin in Jellyfin.
2. Click **Add Generic Destination**. 
3. Name it whatever you like, and set the Webhook URL to: `http://<your-ip>:9000/jellyfin`
4. Check **Item Added**, **Playback Start**, and **Send All Properties**.
5. Click **Add Request Header**. Set Key: `Content-Type` and Value: `application/json`.
6. **Relevant Variables:** `JELLYFIN_SERVER`, `JELLYFIN_TOKEN`.

### 🟢 Emby
Emby and Subgen must have identical path configurations (or use Path Mapping). Emby responses contain full info, so no API tokens are required!
1. In Emby, create a webhook pointing to: `http://<your-ip>:9000/emby`
2. Set **Request content type** to `multipart/form-data`.
3. Configure your desired events (Usually `New Media Added`, `Start`, and `Unpause`).

### 🟣 Tautulli
Tautulli and Subgen must have identical path configurations (or use Path Mapping).
Create two separate Webhooks in Tautulli pointing to `http://<your-ip>:9000/tautulli` using the **POST** method.

**Webhook 1: Playback Start**
*   **Trigger:** Playback Start
*   **JSON Header:** `{"source": "Tautulli"}`
*   **Data (JSON):** 
    > `{"event": "played", "file": "{file}", "filename": "{filename}", "mediatype": "{media_type}"}`

**Webhook 2: Recently Added**
*   **Trigger:** Recently Added
*   **JSON Header:** `{"source": "Tautulli"}`
*   **Data (JSON):** 
    > `{"event": "added", "file": "{file}", "filename": "{filename}", "mediatype": "{media_type}"}`

---

## ⚙️ Configuration (Environment Variables)

*Note: Subgen recently standardized environment variables (e.g., `PLEX_TOKEN`). Legacy names (e.g., `PLEXTOKEN`) are still fully supported for backwards compatibility!*

### 🧠 Core Parakeet & AI Settings

| Variable | Default | Description |
|---|---|---|
| `TRANSCRIBE_DEVICE` | `cpu` | Device to transcribe on: `cpu`, `gpu`, or `cuda`. |
| `PARAKEET_MODEL` | `nvidia/parakeet-tdt-1.1b` | Model to use: `nvidia/parakeet-tdt-1.1b` (English), `nvidia/parakeet-rnnt-1.1b` (English RNNT), `nvidia/canary-1b` (Multilingual 25 langs + translation). |
| `PARAKEET_MODEL_DIR` | `./models/parakeet_onnx` | Directory containing exported ONNX models. |
| `PARAKEET_USE_ONNX` | `True` | Use ONNX Runtime for inference (faster, lighter) vs NeMo toolkit. |
| `PARAKEET_COMPUTE_TYPE` | `float32` | Precision: `float32`, `float16`, `bfloat16`. |
| `PARAKEET_THREADS` | `4` | Number of CPU threads for inference. |
| `CONCURRENT_TRANSCRIPTIONS` | `2` | Number of files to process in parallel. |
| `CLEAR_VRAM_ON_COMPLETE` | `True` | Do garbage collection and clear the model from VRAM when the queue is empty. |
| `MODEL_CLEANUP_DELAY` | `30` | Seconds to wait before clearing the Parakeet model from memory. |
| `ASR_TIMEOUT` | `18000` | Seconds to wait before timing out a transcription request (default 5 hours). |
| `SUBGEN_KWARGS` | `{}` | JSON dict to pass pure kwargs to Parakeet (e.g. `{'vad': True}`). For advanced users. |

### ⚡ Processing Triggers & Queuing

*(Not relevant for Bazarr users)*

| Variable | Default | Description |
|---|---|---|
| `PROCESS_ADDED_MEDIA` | `True` | Generate subs for newly added media (when triggered by webhook). |
| `PROCESS_MEDIA_ON_PLAY` | `True` | Generate subs for media when it is played (when triggered by webhook). |
| `TRANSCRIBE_FOLDERS` | `''` | Pipe-separated list (e.g., `/tv\|/movies`) to recurse through and queue existing media. |
| `MONITOR` | `False` | Actively watches `TRANSCRIBE_FOLDERS` in real-time for newly pasted files. |
| `SKIP_STARTUP_SCAN` | `False` | Skips the startup scan of `TRANSCRIBE_FOLDERS` entirely. Subgen will still watch for new files if `MONITOR` is enabled, but won't iterate existing files on start. |
| `PLEX_QUEUE_NEXT_EPISODE` | `False` | Auto-queues the *next* Plex episode when Subgen is triggered. |
| `PLEX_QUEUE_SEASON` | `False` | Auto-queues the *entire remaining season* when Subgen is triggered. |
| `PLEX_QUEUE_SERIES` | `False` | Auto-queues the *entire remaining series* when Subgen is triggered. |
| `WEBHOOK_URL_COMPLETED` | `''` | Sends a POST to the `WEBHOOK_URL_COMPLETED` URL with a JSON containing the completed file info. |

### ⏭️ Skip Logic & Audio Targeting

*Prevent Subgen from wasting time on files that don't need subtitles.*

**Directory skip marker:** Place an empty `.subgen_skip` file in any directory to tell Subgen to skip that directory and all of its subdirectories — both during the startup scan and when watching for new files. Useful for completed shows or any section of your library that will never need new subtitles.
```bash
touch "/tv/The Simpsons/.subgen_skip"   # skips all seasons
touch "/tv/Some Show/Season 1/.subgen_skip"  # skips just that season
```

| Variable | Default | Description |
|---|---|---|
| `SKIP_IF_TARGET_SUBTITLES_EXIST` | `True` | Skips if an auto-generated subtitle in your desired language already exists. |
| `SKIP_IF_EXTERNAL_SUBTITLES_EXIST`| `False` | Skips if an external subtitle matching `SUBTITLE_LANGUAGE_NAME` is found. |
| `SKIP_IF_INTERNAL_SUBTITLES_LANGUAGE`| `eng` | Skips if the file contains an embedded sub with this 3-letter code. |
| `SKIP_SUBTITLE_LANGUAGES` | `''` | Pipe-separated list (e.g., `eng\|spa`). Skips if the file *has audio* in these languages. |
| `SKIP_IF_AUDIO_LANGUAGES` | `''` | Pipe-separated list (ISO 639-2). Skips generation if the file has audio tracks in these languages. |
| `PREFERRED_AUDIO_LANGUAGES` | `eng` | Pipe-separated list. If multiple audio tracks exist, prefer transcribing this one. |
| `LIMIT_TO_PREFERRED_AUDIO_LANGUAGE`| `False` | If True, skips files that do not have any audio tracks matching your preferred list. |
| `FORCE_DETECTED_LANGUAGE_TO` | `''` | Force model to this 2-letter language code if it keeps incorrectly detecting audio. |
| `DETECT_LANGUAGE_LENGTH` | `30` | Number of seconds to analyze audio to determine the language. |
| `DETECT_LANGUAGE_OFFSET` | `0` | Number of seconds to skip forward before detecting language (good for avoiding theme songs). |
| `SHOULD_PARAKEET_DETECT_AUDIO_LANGUAGE` | `False` | Should Parakeet detect language if there is no audio language tagged in the media file. |
| `SKIP_UNKNOWN_LANGUAGE` | `False` | Skip processing if Parakeet cannot detect the audio language. |
| `SKIP_ONLY_SUBGEN_SUBTITLES` | `False` | Skips generation only if the file has "subgen" somewhere in the existing subtitle filename. |
| `SKIP_IF_NO_LANGUAGE_BUT_SUBTITLES_EXIST`| `False` | Skips generation if file doesn't have an audio stream marked with a language, but subtitles exist. |
| `IGNORE_FORCED_SUBTITLES` | `True` | When `True`, forced embedded subtitle tracks are excluded from all skip-coverage checks. A file whose only matching subtitle tracks are forced will be treated as having no coverage and transcribed normally. Set to `False` to count forced tracks as full coverage (old behaviour). |

> **Not sure why a file is being skipped?** Use the [Skip Logic Calculator](https://htmlpreview.github.io/?https://github.com/flash1228/subgen-parakeet/blob/main/docs/skip-logic-calculator.html) — configure your settings, describe what subtitle files and streams exist, and it shows exactly what triggered the skip.

### 📝 Subtitle Formatting & Preferences

| Variable | Default | Description |
|---|---|---|
| `TRANSCRIBE_OR_TRANSLATE` | `transcribe` | `transcribe` (matches input language) or `translate` (outputs English). **Translation requires Canary model.** |
| `SUBTITLE_LANGUAGE_NAME` | `aa` | Subtitle file name language code (e.g. `en`). Defaults to `aa` so it floats to the top of Plex's list. |
| `SUBTITLE_LANGUAGE_NAMING_TYPE`| `ISO_639_2_B` | Format to name files (`ISO_639_1`, `ISO_639_2_T`, `NAME`, `NATIVE`). |
| `LRC_FOR_AUDIO_FILES` | `True` | Generates `.lrc` instead of `.srt` if processing pure audio files (e.g., mp3, flac). |
| `WORD_LEVEL_HIGHLIGHT` | `False` | Highlights words dynamically as they are spoken in the subtitle. |
| `APPEND` | `False` | Appends a "Transcribed by Parakeet..." watermark at the very end of the `.srt`. |
| `SHOW_IN_SUBNAME_SUBGEN` | `True` | Adds `.subgen` to the output file name. |
| `SHOW_IN_SUBNAME_MODEL` | `True` | Adds the model used (e.g., `.parakeet-tdt-1.1b`) to the output file name. |
| `CUSTOM_REGROUP` | `cm_sl=84_sl=42++++++1` | Stable-TS grouping. Try to 'clean up' subtitles a bit. Set to `default` to use base Stable-TS. |

### 📂 System, Paths & Network Settings

| Variable | Default | Description |
|---|---|---|
| `WEBHOOK_PORT` | `9000` | Port used to listen for webhooks and Bazarr requests. |
| `PUID` / `PGID` | `99` / `100` | Run container as a specific user/group (helps with file permissions). |
| `DEBUG` | `True` | Outputs extra logs, helpful for troubleshooting paths or webhook hits. |
| `RELOAD_SCRIPT_ON_CHANGE` | `False` | (Dev) Auto-reloads uvicorn if `subgen.py` is edited. |
| `UPDATE` | `False` | (Standalone) Will pull the latest `subgen.py` from repo via `launcher.py`. |
| `USE_PATH_MAPPING` | `False` | Set to True if your media server and Subgen map their volumes differently. |
| `PATH_MAPPING_FROM` | `/tv` | Example: The media path on Plex. |
| `PATH_MAPPING_TO` | `/Volumes/TV` | Example: What Subgen natively sees that same path as. |
| `MODEL_PATH` | `./models` | Path where AI models are downloaded and stored. |

### 🎬 Media Server Integration (Metadata Refreshing)

*Required if you want Subgen to automatically generate Subtitles off of Webhook Events from Plex or Jellyfin or to tell Plex or Jellyfin to refresh the show's metadata so the subtitle immediately appears after generation.*

| Variable | Default | Description |
|---|---|---|
| `PLEX_SERVER` | *(None)* | Local Plex address (e.g., `http://192.168.1.100:32400`). |
| `PLEX_TOKEN` | *(None)* | Your Plex Token for API access. |
| `JELLYFIN_SERVER` | *(None)* | Local Jellyfin address (e.g., `http://192.168.1.100:8096`). |
| `JELLYFIN_TOKEN` | *(None)* | Generated API token from Jellyfin UI. |

---

## 🌎 Supported Languages

**Parakeet TDT/RNNT 1.1B:** English only  
**Canary 1B:** 25 European languages (en, es, fr, de, it, pt, pl, ru, zh, ja, ko, ar, hi, tr, vi, th, nl, sv, da, no, fi, cs, hu, ro, uk) with automatic language detection + translation to/from English

---

## 🔗 OpenAI-Compatible API Endpoints

Subgen exposes two endpoints that match the [OpenAI Whisper API](https://platform.openai.com/docs/api-reference/audio), so it can be used as a drop-in backend for any client that targets that API (Open WebUI, Obsidian plugins, etc.).

| Endpoint | Description |
|---|---|
| `POST /v1/audio/transcriptions` | Transcribe audio to text in the source language |
| `POST /v1/audio/translations` | Transcribe and translate audio to English (requires Canary model) |

**Supported parameters:**

| Parameter | Description |
|---|---|
| `file` | Audio file (any format ffmpeg can decode) |
| `language` | Source language ISO-639-1 code (transcriptions only; auto-detected if omitted) |
| `prompt` | Optional context passed to Parakeet |
| `response_format` | `json` (default), `text`, `srt`, `vtt`, `verbose_json` |
| `model` | Accepted but ignored — subgen uses its configured model |
| `temperature` | Accepted but ignored |

**Example:**
```bash
curl -X POST http://<your-ip>:9000/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "response_format=json"
# {"text": "..."}
```

`verbose_json` returns segments with start/end timestamps and word-level timestamps.

---

## 🪲 Known Issues

* Parakeet TDT/RNNT is English-only — use Canary for other languages
* Translation requires Canary model (Parakeet TDT/RNNT cannot translate)
* It uses trained AI models; there *will* occasionally be mistranslations or hallucinations based on background noise

---

## ❤️ Credits

* [NVIDIA Parakeet](https://github.com/NVIDIA/NeMo) — FastConformer TDT/RNNT ASR models
* [NVIDIA Canary](https://github.com/NVIDIA/NeMo) — Multilingual ASR + Translation
* [NeMo Toolkit](https://github.com/NVIDIA/NeMo) — NVIDIA's ASR framework
* [ONNX Runtime](https://onnxruntime.ai/) — Fast inference engine
* [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) for original implementation
* Google & FFmpeg
* [stable-ts](https://github.com/jianfch/stable-ts) — Subtitle timestamp alignment
* [Whisper ASR Webservice](https://github.com/ahmetoner/whisper-asr-webservice) for Bazarr HTTP webhook logic
* Community Contributors