# script /word_visualization.py
# lighter phase
## # # # # # Analyzes the transcript chunks and returns the analysis results.
# # # # # # The analysis results keyword counts.

import os, json, time
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Use the same built-in serif font; formatting purposes
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif']  = ['DejaVu Serif', 'Times New Roman']

def _log(msg: str): # Log messages with a timestamp
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def load_analysis(base_name): #
    path = os.path.join("analysis_results", f"{base_name}_analysis.json")
    _log(f"Loading analysis from {path}")
    with open(path, 'r') as f:
        return json.load(f)

def animate_word_frequency(base_name):
    # Animate word frequency over time into visualizations/
    # This function creates an animation of the total word count per segment over time.
    data        = load_analysis(base_name)
    times       = [d["end"]         for d in data]
    word_counts = [d["total_words"] for d in data]

    fig, ax = plt.subplots()
    line, = ax.plot([], [], label="Total Words", marker='o', linestyle='-')

    ax.set_title("Total Words per Segment", fontsize=14, fontfamily="DejaVu Serif")
    ax.set_xlabel("Time (s)",                 fontsize=12, fontfamily="DejaVu Serif")
    ax.set_ylabel("Word Count",               fontsize=12, fontfamily="DejaVu Serif")

    ax.set_xlim(min(times), max(times))
    ax.set_ylim(0, max(word_counts) * 1.1 if word_counts else 1)

    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(loc="upper left")

    def update(i):
        t = times[: i+1]
        line.set_data(t, word_counts[: i+1])
        return (line,)

    _log(f"Building word-frequency animation ({len(times)} frames)…")
    anim = FuncAnimation(fig, update, frames=len(times), interval=500, blit=True)

    out_dir = "word_visualizations"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{base_name}_word_frequency.mp4")
    _log(f"Saving to {out_path}")
    anim.save(out_path, dpi=150, fps=2)
    plt.close(fig)
    _log("Word-frequency animation done.")
