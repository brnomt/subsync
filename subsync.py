#!/usr/bin/env python3
"""Align a subtitle file to a video with one global offset.

Usage:
  subsync.py <video> <srt> [--minutes 5] [--apply] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

# Avoid a stale cached Hugging Face token blocking public model downloads.
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

import numpy as np

STEP = 0.01
DEFAULT_SEARCH = 20.0
DRIFT_LIMIT = 0.3
AGREE_LIMIT = 0.3


def ms_of(t: str) -> int:
    h, m, rest = t.split(":")
    s, milli = rest.split(",")
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(milli)


def fmt_ms(ms: int) -> str:
    if ms < 0:
        ms = 0
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, milli = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{milli:03d}"


def fmt_off(seconds: float) -> str:
    sign = "+" if seconds >= 0 else "-"
    return f"{sign}{abs(seconds):.3f}s"


def read_srt(path: Path):
    raw = path.read_bytes()
    has_bom = raw.startswith(b"\xef\xbb\xbf")
    body = raw[3:] if has_bom else raw
    try:
        text = body.decode("utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        text = body.decode("cp1252")
        encoding = "cp1252"
    if "\r\n" in text:
        nl = "\r\n"
        norm = text.replace("\r\n", "\n")
    elif "\r" in text:
        nl = "\r"
        norm = text.replace("\r", "\n")
    else:
        nl = "\n"
        norm = text
    cues = []
    for block in re.split(r"\n\s*\n", norm.strip()):
        lines = [ln for ln in block.split("\n") if ln.strip() != ""]
        if len(lines) < 2:
            continue
        m = re.match(
            r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})(.*)$",
            lines[1],
        )
        if not m:
            continue
        cues.append(
            {
                "start_ms": ms_of(m.group(1)),
                "end_ms": ms_of(m.group(2)),
                "start": ms_of(m.group(1)) / 1000.0,
                "end": ms_of(m.group(2)) / 1000.0,
                "rest": m.group(3),
                "text": " ".join(lines[2:]),
                "text_lines": lines[2:],
            }
        )
    return {
        "cues": cues,
        "encoding": encoding,
        "has_bom": has_bom,
        "nl": nl,
        "path": path,
    }


def write_shifted(src: dict, dest: Path, offset_s: float) -> None:
    offset_ms = int(round(offset_s * 1000))
    out_lines = []
    n = 0
    dropped = 0
    for cue in src["cues"]:
        s = cue["start_ms"] + offset_ms
        e = cue["end_ms"] + offset_ms
        if e <= 0:
            dropped += 1
            continue
        if s < 0:
            s = 0
        n += 1
        out_lines.append(str(n))
        out_lines.append(f"{fmt_ms(s)} --> {fmt_ms(e)}{cue['rest']}")
        out_lines.extend(cue["text_lines"])
        out_lines.append("")
    payload = src["nl"].join(out_lines)
    if not payload.endswith(src["nl"]):
        payload += src["nl"]
    data = payload.encode(src["encoding"])
    if src["has_bom"]:
        data = b"\xef\xbb\xbf" + data
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    print(f"Wrote {dest}  cues={n} dropped={dropped} offset={fmt_off(offset_s)}")


def which_ffmpeg() -> str:
    exe = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if not exe:
        sys.exit("ffmpeg not found on PATH")
    return exe


def ffprobe_audio_map(ffmpeg: str, video: Path) -> str:
    """Return ffmpeg -map for the main/English audio track (default 0:a:0)."""
    ffprobe = shutil.which("ffprobe") or shutil.which("ffprobe.exe")
    if not ffprobe:
        return "0:a:0"
    r = subprocess.run(
        [
            ffprobe,
            "-hide_banner",
            "-loglevel",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index:stream_tags=language,title",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        return "0:a:0"
    try:
        streams = json.loads(r.stdout).get("streams") or []
    except json.JSONDecodeError:
        return "0:a:0"
    for s in streams:
        lang = str((s.get("tags") or {}).get("language") or "").lower()
        if lang in {"eng", "en", "english"}:
            return f"0:{s['index']}"
    if streams:
        return f"0:{streams[0]['index']}"
    return "0:a:0"


def extract_wav(ffmpeg: str, video: Path, wav: Path, start_s: float, dur_s: float, amap: str) -> None:
    if wav.exists() and wav.stat().st_size > 1000:
        return
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-y",
        "-ss",
        str(start_s),
        "-t",
        str(dur_s),
        "-i",
        str(video),
        "-map",
        amap,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(wav),
    ]
    subprocess.run(cmd, check=True)


def transcribe(wav: Path, json_path: Path, model_name: str) -> dict:
    if json_path.exists() and json_path.stat().st_size > 100:
        return json.loads(json_path.read_text(encoding="utf-8"))
    from faster_whisper import WhisperModel

    print(f"Loading {model_name} (CPU int8)...", flush=True)
    model = transcribe.model if getattr(transcribe, "model", None) and transcribe.model_name == model_name else None
    if model is None:
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        transcribe.model = model
        transcribe.model_name = model_name
    print(f"Transcribing {wav.name}...", flush=True)
    segments, info = model.transcribe(
        str(wav),
        language="en",
        word_timestamps=True,
        vad_filter=False,
        beam_size=5,
    )
    segs = []
    for s in segments:
        words = []
        if s.words:
            words = [
                {
                    "start": float(w.start),
                    "end": float(w.end),
                    "word": w.word,
                    "prob": float(w.probability) if w.probability is not None else None,
                }
                for w in s.words
            ]
        segs.append(
            {
                "start": float(s.start),
                "end": float(s.end),
                "text": s.text,
                "words": words,
            }
        )
    payload = {
        "file": wav.name,
        "language": info.language,
        "duration": float(info.duration) if info.duration is not None else None,
        "segments": segs,
        "n_segments": len(segs),
        "speech_s": sum(s["end"] - s["start"] for s in segs),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  {len(segs)} segments, {payload['speech_s']:.1f}s speech -> {json_path.name}", flush=True)
    return payload


transcribe.model = None
transcribe.model_name = None


def activity(intervals, length_s, step=STEP):
    n = int(round(length_s / step))
    a = np.zeros(n, dtype=np.float32)
    for it in intervals:
        i0 = max(0, int(it["start"] / step))
        i1 = min(n, int(np.ceil(it["end"] / step)))
        if i1 > i0:
            a[i0:i1] = 1.0
    return a


def xcorr_offset(cues, segments, window_start, window_dur, search=DEFAULT_SEARCH, step=STEP):
    pad = search
    t0 = window_start - pad
    length = window_dur + 2 * pad
    w_rel = [{"start": s["start"] + pad, "end": s["end"] + pad} for s in segments if (s["end"] - s["start"]) >= 0.12]
    w_act = activity(w_rel, length, step)
    s_rel = [
        {"start": c["start"] - t0, "end": c["end"] - t0}
        for c in cues
        if c["end"] >= t0 - 1 and c["start"] <= t0 + length + 1
    ]
    s_act = activity(s_rel, length, step)
    corr = np.correlate(w_act, s_act, mode="full")
    lags = np.arange(-(len(s_act) - 1), len(w_act))
    max_lag = int(round(search / step))
    mask = (lags >= -max_lag) & (lags <= max_lag)
    lags, corr = lags[mask], corr[mask]
    w_speech = float(w_act.sum())
    s_speech = float(s_act.sum())
    denom = min(w_speech, s_speech) if min(w_speech, s_speech) else 1.0
    scores = corr / denom
    best_i = int(np.argmax(scores))
    best = float(lags[best_i] * step)
    # runner-up outside +/- 0.3 s
    other = scores.copy()
    near = int(round(0.3 / step))
    other[max(0, best_i - near) : min(len(other), best_i + near + 1)] = -1
    ru_i = int(np.argmax(other))
    return {
        "offset_s": best,
        "score": float(scores[best_i]),
        "runner_up_offset_s": float(lags[ru_i] * step),
        "runner_up_score": float(other[ru_i]),
        "whisper_speech_s": w_speech * step,
    }


def tok(s: str):
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.findall(r"[a-z0-9']+", s)


def word_offsets(cues, payload, window_start, search=DEFAULT_SEARCH):
    words = []
    for s in payload["segments"]:
        for w in s.get("words") or []:
            ts = tok(w["word"])
            if not ts:
                continue
            words.append({"t": ts[0], "start": window_start + float(w["start"])})
    offs = []
    lo, hi = window_start - 5, window_start + 305
    for c in cues:
        if c["end"] < lo or c["start"] > hi:
            continue
        q = tok(c["text"])
        if len(q) < 3:
            continue
        n = min(6, len(q))
        qn = q[:n]
        best = None
        for i in range(len(words) - n + 1):
            got = [words[i + k]["t"] for k in range(n)]
            miss = sum(a != b for a, b in zip(got, qn))
            if miss <= 1:
                d = words[i]["start"] - c["start"]
                if abs(d) <= search and (best is None or abs(d) < abs(best)):
                    best = d
        if best is not None:
            offs.append(best)
    if not offs:
        return None
    arr = np.array(offs)
    med = float(np.median(arr))
    tight = arr[np.abs(arr - med) <= 1.0]
    if len(tight) == 0:
        tight = arr
    return {
        "n": int(len(tight)),
        "n_raw": int(len(arr)),
        "offset_s": float(np.median(tight)),
        "std": float(np.std(tight)),
    }


def window_srt(cues, dest: Path, start_s: float, dur_s: float, pad_s: float = 20.0) -> int:
    lo, hi = start_s - pad_s, start_s + dur_s + pad_s
    out = []
    n = 0
    for c in cues:
        if c["end"] < lo or c["start"] > hi:
            continue
        n += 1
        s = max(0, int(round((c["start"] - start_s) * 1000)))
        e = max(s, int(round((c["end"] - start_s) * 1000)))
        out.append(str(n))
        out.append(f"{fmt_ms(s)} --> {fmt_ms(e)}")
        out.extend(c["text_lines"])
        out.append("")
    dest.write_bytes(("\n".join(out) + "\n").encode("utf-8"))
    return n


def run_ffsubsync(wav: Path, srt_in: Path, srt_out: Path, log_dir: Path, search: float):
    ffs = shutil.which("ffsubsync") or shutil.which("ffsubsync.exe")
    if not ffs:
        # venv next to this script
        cand = Path(sys.executable).with_name("ffsubsync.exe")
        ffs = str(cand) if cand.exists() else None
    if not ffs:
        return None
    log_dir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [
            ffs,
            str(wav),
            "-i",
            str(srt_in),
            "-o",
            str(srt_out),
            "--max-offset-seconds",
            str(int(search)),
            "--no-fix-framerate",
            "--encoding",
            "utf-8",
            "--output-encoding",
            "utf-8",
            "--log-dir-path",
            str(log_dir),
        ],
        capture_output=True,
        text=True,
    )
    log = log_dir / "ffsubsync.log"
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else (r.stdout + r.stderr)
    off = score = None
    m = re.search(r"offset seconds:\s*([+-]?\d+(?:\.\d+)?)", text)
    if m:
        off = float(m.group(1))
    m = re.search(r"score:\s*([+-]?\d+(?:\.\d+)?)", text)
    if m:
        score = float(m.group(1))
    return {"offset_s": off, "score": score, "ok": r.returncode == 0}


def parse_args():
    p = argparse.ArgumentParser(description="Compute one global SRT offset from the soundtrack.")
    p.add_argument("video", type=Path, help="Movie / audio file")
    p.add_argument("srt", type=Path, help="Subtitle file to align (not modified unless you pass --out on top of it)")
    p.add_argument("--minutes", type=float, default=5.0, help="Length of each analysis window in minutes")
    p.add_argument("--model", default="large-v3-turbo", help="faster-whisper model name")
    p.add_argument("--search", type=float, default=DEFAULT_SEARCH, help="Offset search range, seconds")
    p.add_argument("--apply", action="store_true", help="Write a shifted SRT if checks pass")
    p.add_argument("--out", type=Path, help="Output SRT path (default: <srt>.synced.srt)")
    p.add_argument("--no-ffs", action="store_true", help="Skip the ffsubsync cross-check")
    p.add_argument("--cache-dir", type=Path, help="Where to store WAV/JSON (default: ./work next to this script)")
    p.add_argument("--force-offset", type=float, help="Skip detection and apply this offset (seconds) when used with --apply")
    return p.parse_args()


def main():
    args = parse_args()
    video = args.video.expanduser().resolve()
    srt_path = args.srt.expanduser().resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")
    if not srt_path.exists():
        sys.exit(f"srt not found: {srt_path}")
    cache = (args.cache_dir or (Path(__file__).resolve().parent / "work")).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    src = read_srt(srt_path)
    print(f"Loaded {len(src['cues'])} cues from {srt_path} ({src['encoding']}, nl={src['nl']!r})")

    if args.force_offset is not None:
        if not args.apply:
            print(f"Forced offset {fmt_off(args.force_offset)} (pass --apply to write)")
            return 0
        dest = args.out or srt_path.with_name(srt_path.stem + ".synced.srt")
        write_shifted(src, dest, args.force_offset)
        return 0

    ffmpeg = which_ffmpeg()
    amap = ffprobe_audio_map(ffmpeg, video)
    print(f"Audio map {amap}")
    dur = args.minutes * 60.0
    windows = [
        ("00-05", 0.0, dur),
        ("10-15", 10 * 60.0, dur),
        ("60-65", 60 * 60.0, dur),
    ]

    whisper_rows = []
    ffs_rows = []
    for name, start, length in windows:
        wav = cache / f"audio_{name}.wav"
        print(f"\n=== Window {name} ({start:.0f}s, {length:.0f}s) ===")
        try:
            extract_wav(ffmpeg, video, wav, start, length, amap)
        except subprocess.CalledProcessError as e:
            print(f"  skip: could not extract audio ({e})")
            continue
        payload = transcribe(wav, cache / f"audio_{name}.whisper.json", args.model)
        xc = xcorr_offset(src["cues"], payload["segments"], start, length, search=args.search)
        wo = word_offsets(src["cues"], payload, start, search=args.search)
        print(f"  whisper activity xcorr: {fmt_off(xc['offset_s'])}  score={xc['score']:.3f}  "
              f"runner {fmt_off(xc['runner_up_offset_s'])} score={xc['runner_up_score']:.3f}  "
              f"speech={xc['whisper_speech_s']:.1f}s")
        if wo:
            print(f"  whisper word-align:     {fmt_off(wo['offset_s'])}  n={wo['n']}/{wo['n_raw']} std={wo['std']:.3f}")
        whisper_rows.append({"name": name, "xcorr": xc, "words": wo, "speech": payload["speech_s"]})

        if not args.no_ffs:
            win_srt = cache / f"window_{name}.srt"
            n = window_srt(src["cues"], win_srt, start, length, pad_s=args.search)
            if n:
                ffs = run_ffsubsync(wav, win_srt, cache / f"window_{name}.ffs.srt", cache / f"ffs_{name}", args.search)
                if ffs and ffs["offset_s"] is not None:
                    print(f"  ffsubsync:              {fmt_off(ffs['offset_s'])}  score={ffs['score']}")
                    ffs_rows.append({"name": name, **ffs, "speech": payload["speech_s"]})

    word_vals = [(r["name"], r["words"]["offset_s"], r["words"]["n"]) for r in whisper_rows if r["words"] and r["words"]["n"] >= 5]
    x_vals = [(r["name"], r["xcorr"]["offset_s"], r["xcorr"]["score"], r["speech"]) for r in whisper_rows]

    print("\n=== Consensus ===")
    whisper_off = None
    if word_vals:
        # prefer the window with most word matches
        best = max(word_vals, key=lambda x: x[2])
        whisper_off = best[1]
        print(f"Whisper word-align (best window {best[0]}, n={best[2]}): {fmt_off(whisper_off)}")
        if len(word_vals) >= 2:
            early = [v[1] for v in word_vals if v[0] != "60-65"]
            late = [v[1] for v in word_vals if v[0] == "60-65"]
            if early and late:
                drift = late[0] - (early[-1] if early else early[0])
                print(f"Drift (late window minus earlier word-align): {fmt_off(drift)}")
                if abs(drift) > DRIFT_LIMIT:
                    print(f"DRIFT > {DRIFT_LIMIT}s — not a single global offset. Not writing.")
                    return 2
            span = max(v[1] for v in word_vals) - min(v[1] for v in word_vals)
            print(f"Word-align span across windows: {span:.3f}s")
            if span > DRIFT_LIMIT:
                print(f"DRIFT > {DRIFT_LIMIT}s — not a single global offset. Not writing.")
                return 2
    elif x_vals:
        # fall back to activity; ignore windows with almost continuous speech (flat peak)
        usable = [v for v in x_vals if v[3] < (args.minutes * 60.0) * 0.85]
        pick = usable or x_vals
        whisper_off = float(np.median([v[1] for v in pick]))
        print(f"Whisper activity median: {fmt_off(whisper_off)}")

    ffs_off = None
    if ffs_rows:
        # drop likely VAD failures: low speech-gap windows already flagged, or outliers > 1s from median
        offs = [r["offset_s"] for r in ffs_rows if r["offset_s"] is not None]
        if offs:
            med = float(np.median(offs))
            good = [r for r in ffs_rows if r["offset_s"] is not None and abs(r["offset_s"] - med) <= 1.0]
            bad = [r for r in ffs_rows if r not in good]
            if good:
                ffs_off = float(np.median([r["offset_s"] for r in good]))
                print(f"ffsubsync (reliable windows {[r['name'] for r in good]}): {fmt_off(ffs_off)}")
            if bad:
                print(f"ffsubsync discarded as VAD outliers: " +
                      ", ".join(f"{r['name']}={fmt_off(r['offset_s'])}" for r in bad))

    if whisper_off is None:
        print("Could not estimate an offset.")
        return 1

    agreed = True
    if ffs_off is not None:
        delta = abs(ffs_off - whisper_off)
        print(f"Whisper vs ffsubsync disagreement: {delta:.3f}s")
        if delta > AGREE_LIMIT:
            agreed = False
            print(f"DISAGREE > {AGREE_LIMIT}s — not writing. Use the numbers above.")
    else:
        print("No ffsubsync cross-check available.")

    # If there is a same-language embedded SRT, a round number near it is OK,
    # but we apply the Whisper estimate (source of truth for the audio).
    chosen = whisper_off
    print(f"Chosen offset: {fmt_off(chosen)}  (add this many seconds to every subtitle timestamp)")

    if not args.apply:
        print("Dry run. Pass --apply to write a shifted SRT.")
        return 0 if agreed else 3

    if not agreed:
        return 3

    dest = args.out.expanduser() if args.out else srt_path.with_name(srt_path.stem + ".synced.srt")
    if dest.resolve() == srt_path.resolve():
        sys.exit("Refusing to overwrite the input SRT; pass a different --out")
    write_shifted(src, dest, chosen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())