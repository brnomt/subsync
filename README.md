# subsync

Align an `.srt` file to a movie with a **single global offset** (not line by line).

It is meant for subtitles that are in a different language than the audio (for example, Spanish subs on an English musical): it never compares text with the speech, it compares **when there is voice**.

## What it does

1. Extracts audio windows (by default the first N minutes, plus a slice at 10-15 min and another at 60-65 min) to 16 kHz mono WAV, from the main audio track.
2. Transcribes them with **faster-whisper `large-v3-turbo`** (CPU, int8) with word timestamps. Transcription is forced to English (`language="en"`).
3. Estimates the offset two ways:
   - activity correlation: Whisper speech intervals vs. SRT cues, as 10 ms binary activity envelopes, searched over ±20 s;
   - word alignment: the first tokens of each SRT cue matched against Whisper word timestamps (up to 6 tokens, at most 1 mismatch). This one only works when the subtitle is in the same language as the audio, so cross-language runs fall back to activity correlation.
4. Checks **drift**: offsets from different windows (start vs. ~minute 60) must agree within 0.3 s, otherwise the subtitle is not shifted (exit code 2).
5. Optional: cross-checks the result with **ffsubsync** (VAD). Outlier windows more than 1 s from the median are discarded as VAD failures; if Whisper and ffsubsync disagree by more than 0.3 s, nothing is written (exit code 3).

## Usage

From this folder, with the venv already created:

```bat
.venv\Scripts\python.exe subsync.py "C:\path\movie.mkv" "C:\path\subs.srt" --minutes 5
```

That only computes and reports. To write the result:

```bat
.venv\Scripts\python.exe subsync.py "C:\path\movie.mkv" "C:\path\subs.srt" --apply --out "C:\path\subs.fixed.spa.srt"
```

`--apply` never overwrites the input file. Without `--out` it writes `<name>.synced.srt` next to the original.

Other options:

| Flag | Meaning |
| --- | --- |
| `--model large-v3-turbo` | faster-whisper model name |
| `--minutes 5` | length of each analysis window, in minutes |
| `--search 20` | offset search range, seconds |
| `--no-ffs` | skip the ffsubsync cross-check |
| `--cache-dir work` | where WAV / Whisper JSON are cached |
| `--force-offset -1.5` | skip detection and use this offset (needs `--apply` to write) |

Exit codes: `0` ok, `1` no estimate, `2` drift detected, `3` Whisper/ffsubsync disagreement or nothing written.

## Environment

Python 3.11 (the venv in this folder):

```bat
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

`ffmpeg` must be on `PATH`. The model is downloaded the first time (~1.5 GB) into `%USERPROFILE%\.cache\huggingface`.

If Hugging Face complains about an expired token, the script disables the implicit token and downloads anonymously.

The video and the original `.srt` are never deleted or modified. The `work\` folder is cache only (WAV, Whisper JSON, per-window SRTs) and can be removed at any time.
