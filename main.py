#!/usr/bin/env python
# scripts/main.py
# Medium Phase
# This tool is intended for educational or synthetic interview data only. 
# Do not use on real clinical data without IRB approval or informed consent.
# Usage:
#   source venv/bin/activate
#   python -m scripts.main audio_files/your_file.mp3

import os
import argparse
import time
from datetime import datetime

# transcription imports
from scripts.transcriber         import transcribe_auto as transcribe_audio
from scripts.analysis           import analyze_transcript_chunks, save_analysis
from scripts.visualization      import animate_analysis
from scripts.word_visualization import animate_word_frequency
from scripts.summary            import generate_summary  # newly added

# ─── CONSTANTS ──────────────────────────────────────────────────────────────
def _log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ─── MAIN ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Run NEURONǒok pipeline on an interview audio file"
    )
    parser.add_argument("audio_path", help="e.g. audio_files/anxiety_1min.mp3")
    args = parser.parse_args()

    if not os.path.isfile(args.audio_path):
        parser.error(f"File not found: {args.audio_path}")
    # Check if the file is an audio file
    # pulls transcriber.py's is_audio_file function
    _log(f"1/5 Transcription → {args.audio_path}")
    t0 = time.time()
    result = transcribe_audio(args.audio_path)
    _log(f"   ↳ done in {time.time()-t0:.1f}s")

    base = os.path.splitext(os.path.basename(args.audio_path))[0].replace(" ", "_")
    _log(f"   ↳ Saving to {base}…")

    # extracts analysis (NLTK) results from the transcription
    _log("2/5 Analysis")
    t1 = time.time()
    analysis = analyze_transcript_chunks(result)
    save_analysis(analysis, base)
    _log(f"   ↳ done in {time.time()-t1:.1f}s")


    # develops a summary of the analysis; saves it to a text file
    _log("3/5 Generating summary")
    t2 = time.time()
    # checks for analysis directory; backup: creates one if it doesn't exist
    summary_dir = os.path.join("analysis_results")
    os.makedirs(summary_dir, exist_ok=True)
    # develop & save summary
    summary_text = generate_summary(analysis)
    summary_path = os.path.join(summary_dir, f"{base}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)
    _log(f"   ↳ summary saved to {summary_path} in {time.time()-t2:.1f}s")

    # visualizes analysis results
    _log("4/5 Visualization (keywords & sentiment)")
    t3 = time.time()
    animate_analysis(base)
    _log(f"   ↳ done in {time.time()-t3:.1f}s")

    # visualizes word frequency
    _log("5/5 Visualization (word frequency)")
    t4 = time.time()
    animate_word_frequency(base)
    _log(f"   ↳ done in {time.time()-t4:.1f}s")


    # final message
    _log("All done!")

# ─── ENTRY POINT ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
    try:
        main()
    except Exception as e:
        _log(f"Error: {e}")
