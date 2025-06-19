# scripts/visualization.py
# # Lighter Phase
# # # # # Analyzes the transcript chunks and returns the analysis results.
# # # # # The analysis results include the sentiment scores and keyword counts.
# # # # # The analysis results are saved to a JSON file.

import os, json, time
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Use a built-in serif font (DejaVu Serif) instead of Garamond
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif']  = ['DejaVu Serif', 'Times New Roman']

def _log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def load_analysis(base_name):
    path = os.path.join("analysis_results", f"{base_name}_analysis.json")
    _log(f"Loading analysis from {path}")
    with open(path, 'r') as f:
        return json.load(f)

def animate_analysis(base_name):
    # Animate sentiment and keyword trends over time into visualizations/
    data  = load_analysis(base_name)
    times = [d["end"] for d in data]

    sentiment = {
        "neg":      [d["neg"]      for d in data],
        "neu":      [d["neu"]      for d in data],
        "pos":      [d["pos"]      for d in data],
        "compound": [d["compound"] for d in data],
    }
    keywords = {
        "Depression":    [d["depression_count"]    for d in data],
        "Hopelessness":  [d["hopelessness_count"]  for d in data],
        "Anxiety":       [d["anxiety_count"]       for d in data],
        "ADHD":          [d["adhd_count"]          for d in data],
        "Filler":        [d["filler_count"]        for d in data],
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
    lines_s = {k: ax1.plot([], [], label=k)[0] for k in sentiment}
    lines_k = {k: ax2.plot([], [], label=k)[0] for k in keywords}

    # Sentiment plot styling
    ax1.set_title("Sentiment Scores Over Time", fontsize=14, fontfamily="DejaVu Serif")
    ax1.set_ylabel("Sentiment",              fontsize=12, fontfamily="DejaVu Serif")
    ax1.set_ylim(-1.1, 1.1)
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax1.legend(loc="upper left")

    # Keyword plot styling
    ax2.set_title("Keyword Counts Over Time", fontsize=14, fontfamily="DejaVu Serif")
    ax2.set_ylabel("Count",                   fontsize=12, fontfamily="DejaVu Serif")
    ax2.set_xlabel("Time (s)",                fontsize=12, fontfamily="DejaVu Serif")
    max_kw = max(max(v) for v in keywords.values()) or 1
    ax2.set_ylim(0, max_kw * 1.1)
    ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Force legend ordering
    order   = ["Depression", "Hopelessness", "Anxiety", "ADHD", "Filler"]
    handles = [lines_k[label] for label in order]
    ax2.legend(handles, order, loc="upper left")

    ax1.set_xlim(min(times), max(times))

    def update(i):
        t = times[: i+1]
        for k, ln in lines_s.items():
            ln.set_data(t, sentiment[k][: i+1])
        for k, ln in lines_k.items():
            ln.set_data(t, keywords[k][: i+1])
        return list(lines_s.values()) + list(lines_k.values())

    _log(f"Building combined animation ({len(times)} frames)…") 

    anim = FuncAnimation(fig, update, frames=len(times), interval=500, blit=True) # animate the plot
#    Save the animation to a file
    out_dir = "visualizations" # Directory for combined animation
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{base_name}_sentiment_keyword_trends.mp4")
    _log(f"Saving to {out_path}")
    anim.save(out_path, dpi=150, fps=2)
    plt.close(fig)
    _log("Combined animation done.")
