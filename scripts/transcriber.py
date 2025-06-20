# scripts/transcriber.py
# Simplified timestamped transcript for manual speaker labeling,
# with keyword highlighting, silence‐duration markers,
# sentiment per chunk, segmented fast transcription for long interviews,
# and suppressed Whisper/malloc warnings.

import warnings
import contextlib
import os
import subprocess
import json
import uuid
import glob
from typing import Dict, Any, List
from pathlib import Path
from multiprocessing import Pool, cpu_count
import re

# ─── SUPPRESS WHISPER FP16 WARNINGS ─────────────────────────────────────────────
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# ─── IMPORT WHISPER AND MONKEY‐PATCH transcribe TO SILENCE stderr ──────────────
import whisper
_original_transcribe = whisper.Whisper.transcribe

def _silent_transcribe(self, *args, **kwargs):
    # Redirect any C‐level stderr (e.g. malloc logging) to /dev/null
    with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
        return _original_transcribe(self, *args, **kwargs)

whisper.Whisper.transcribe = _silent_transcribe

# ─── OTHER IMPORTS ──────────────────────────────────────────────────────────────
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from scripts.analysis import (
    depression_keywords,
    hopelessness_keywords,
    anxiety_keywords,
    adhd_keywords,
    filler_keywords
)

# Ensure VADER lexicon is available
nltk.download('vader_lexicon', quiet=True)

# ─── CONFIG ────────────────────────────────────────────────────────────────────
CHUNK_LEN      = 30       # Length of each audio chunk for parallel route (sec)
SEGMENT_LEN    = 300      # Length of each segment for segmented fast route (sec)
MAX_WORKERS    = 4        # Parallel worker count
SILENCE_THRESH = 0.4      # Silence threshold (sec), only for debug info
FAST_THRESHOLD = 120.0    # Up to 2 min → fast; above → segmented fast

# compile a master set of all keywords
ALL_KEYWORDS = (
    depression_keywords
    | hopelessness_keywords
    | anxiety_keywords
    | adhd_keywords
    | filler_keywords
)

# regex to split words while keeping punctuation separate
WORD_RE = re.compile(r"\b[\w’']+\b")

# sentiment analyzer instance
sia = SentimentIntensityAnalyzer()

def _highlight_keywords(text: str) -> str:
    """
    Wrap any word in ALL_KEYWORDS with ** for visibility.
    Matching is case-insensitive and ignores punctuation.
    """
    def repl(match):
        w = match.group(0)
        return f"**{w}**" if w.lower() in ALL_KEYWORDS else w
    return WORD_RE.sub(repl, text)

# ─── AUDIO SEGMENTER ──────────────────────────────────────────────────────────

def _segment_audio_file(input_path: str, segment_len: int = SEGMENT_LEN) -> List[str]:
    """
    Splits input audio into fixed-length segments (WAV) for safe fast transcription.
    Returns list of file paths.
    """
    tmpdir = os.path.join("/tmp", f"split_{uuid.uuid4().hex}")
    os.makedirs(tmpdir, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-f", "segment",
        "-segment_time", str(segment_len),
        "-c", "copy",
        f"{tmpdir}/chunk_%03d.wav"
    ], check=True)
    return sorted(glob.glob(f"{tmpdir}/chunk_*.wav"))

# ─── TRANSCRIBE ROUTES ────────────────────────────────────────────────────────

def transcribe_fast(input_path: str) -> Dict[str, Any]:
    """
    Single-model fast transcription for short files.
    """
    model = whisper.load_model("small", device="cpu")
    result = model.transcribe(input_path, language="en")
    merged = {
        "text": result.get("text", "").strip(),
        "chunks": [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
            for seg in result.get("segments", [])
        ]
    }
    _write_transcript_files(input_path, merged)
    return merged

def transcribe_segmented_fast(input_path: str, segment_len: int = SEGMENT_LEN) -> Dict[str, Any]:
    """
    Splits a long file into segments and transcribes each with fast model,
    adjusting timestamps to original timeline, with progress logs.
    """
    model = whisper.load_model("small", device="cpu")
    segment_paths = _segment_audio_file(input_path, segment_len)
    total = len(segment_paths)
    merged = {"text": "", "chunks": []}

    print(f"🧠 Transcribing {total} segments ({segment_len}s each)...", flush=True)
    for idx, seg_path in enumerate(segment_paths, start=1):
        offset = (idx - 1) * segment_len
        print(f"  → Segment {idx}/{total}: {os.path.basename(seg_path)} (offset={offset}s)", flush=True)

        result = model.transcribe(seg_path, language="en")
        merged["text"] += result.get("text", "").strip() + " "
        for seg in result.get("segments", []):
            merged["chunks"].append({
                "start": seg["start"] + offset,
                "end":   seg["end"]   + offset,
                "text":  seg["text"].strip()
            })

    # clean up temp segments
    tmpdir = os.path.dirname(segment_paths[0]) if segment_paths else None
    if tmpdir:
        subprocess.run(["rm", "-rf", tmpdir], check=True)

    print("✅ Segmented transcription complete.\n", flush=True)
    _write_transcript_files(input_path, merged)
    return merged

# ─── PARALLEL TRANSCRIBE (fallback) ───────────────────────────────────────────

def _init_worker():
    global model
    model = whisper.load_model("small", device="cpu")

def _split_audio(input_path: str, workdir: str) -> List[str]:
    os.makedirs(workdir, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-ac", "1", "-ar", "16000",
        "-f", "segment", "-segment_time", str(CHUNK_LEN),
        "-c", "pcm_s16le", f"{workdir}/chunk_%03d.wav"
    ], check=True)
    return sorted(glob.glob(f"{workdir}/chunk_*.wav"))

def _transcribe_chunk(args):
    path, idx = args
    result = model.transcribe(path, language="en")
    offset = idx * CHUNK_LEN
    return {
        "chunks": [
            {"start": seg["start"] + offset,
             "end":   seg["end"]   + offset,
             "text":  seg["text"].strip()}
            for seg in result.get("segments", [])
        ]
    }

def transcribe_parallel(input_path: str) -> Dict[str, Any]:
    tmp   = os.path.join("/tmp", f"par_{uuid.uuid4().hex}")
    files = _split_audio(input_path, tmp)
    jobs  = [(p, i) for i, p in enumerate(files)]
    workers = min(MAX_WORKERS, cpu_count(), len(jobs))
    with Pool(workers, initializer=_init_worker) as pool:
        results = pool.map(_transcribe_chunk, jobs)

    merged = {"chunks": []}
    for r in results:
        merged["chunks"].extend(r["chunks"])
    subprocess.run(["rm", "-rf", tmp], check=True)
    _write_transcript_files(input_path, merged)
    return merged

# ─── TRANSCRIPT WRITER ─────────────────────────────────────────────────────────

def _write_transcript_files(input_path: str, merged: Dict[str, Any]) -> None:
    """
    Writes a timestamped transcript for manual speaker labeling.
    Highlights keywords, notes silence gaps, and sentiment per chunk.
    """
    base = Path(input_path).stem
    outdir = Path("transcripts")
    outdir.mkdir(exist_ok=True)

    chunks       = merged.get("chunks", [])
    debug_info: List[str] = []
    silence_gaps: List[float] = []
    prev_end     = 0.0

    # collect debug info and silence gaps
    for idx, seg in enumerate(chunks):
        start, end = seg["start"], seg["end"]
        gap = start - prev_end
        silence_gaps.append(gap)
        debug_info.append(f"Chunk {idx}: start={start:.2f}s, prev_end={prev_end:.2f}s, gap={gap:.2f}s")
        prev_end = end

    def _fmt_srt(t: float) -> str:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int((t - int(t)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    # TXT with debug header
    txt_file = outdir / f"{base}.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("## ---- DEBUG INFO (chunk gaps) ----\n")
        for info in debug_info:
            f.write(f"## {info}\n")
        f.write("\n## ---- TRANSCRIPT (fill in speakers) ----\n\n")

        for idx, seg in enumerate(chunks):
            start, end, raw_text = seg["start"], seg["end"], seg["text"].strip()
            silence = silence_gaps[idx]
            sent_score = sia.polarity_scores(raw_text)["compound"]
            text = _highlight_keywords(raw_text)
            m0, s0 = divmod(int(start), 60)
            m1, s1 = divmod(int(end), 60)
            f.write(
                f"[{m0:02d}:{s0:02d} - {m1:02d}:{s1:02d}] "
                f"[SPEAKER?] [SILENCE: {silence:.2f}s] [SENT: {sent_score:+.2f}] "
                f"{text}\n"
            )

    # SRT without highlights
    srt_file = outdir / f"{base}.srt"
    with open(srt_file, "w", encoding="utf-8") as f:
        for i, seg in enumerate(chunks, start=1):
            start, end, text = seg["start"], seg["end"], seg["text"].strip()
            f.write(f"{i}\n")
            f.write(f"{_fmt_srt(start)} --> {_fmt_srt(end)}\n")
            f.write(f"{text}\n\n")

# ─── DURATION & DISPATCH ───────────────────────────────────────────────────────

def get_duration(input_path: str) -> float:
    cmd = ["ffprobe", "-v", "error",
           "-show_entries", "format=duration",
           "-of", "json", input_path]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    data = json.loads(out)
    return float(data.get("format", {}).get("duration", 0.0))

def transcribe_auto(input_path: str, threshold: float = FAST_THRESHOLD) -> Dict[str, Any]:
    dur = get_duration(input_path)
    if dur <= threshold:
        return transcribe_fast(input_path)
    else:
        return transcribe_segmented_fast(input_path, segment_len=SEGMENT_LEN)
