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

    # Split sentiment into (a) proportions and (b) compound score
    sentiment_props = {
        "neg": [d["neg"] for d in data],
        "neu": [d["neu"] for d in data],
        "pos": [d["pos"] for d in data],
    }
    compound = [d["compound"] for d in data]

    keywords = {
        "Depression":    [d["depression_count"]    for d in data],
        "Hopelessness":  [d["hopelessness_count"]  for d in data],
        "Anxiety":       [d["anxiety_count"]       for d in data],
        "ADHD":          [d["adhd_count"]          for d in data],
        "Filler":        [d["filler_count"]        for d in data],
    }

    # 3 panels now: sentiment proportions, compound, keywords
    fig, (ax_props, ax_comp, ax_kw) = plt.subplots(3, 1, figsize=(6, 10), sharex=True)
    lines_props = {k: ax_props.plot([], [], label=k)[0] for k in sentiment_props}
    line_comp   = ax_comp.plot([], [], label="compound")[0]
    lines_kw    = {k: ax_kw.plot([], [], label=k)[0] for k in keywords}
    ax_props.set_title("Sentiment Proportions Over Time", fontsize=14, fontfamily="DejaVu Serif")
    ax_props.set_ylabel("Proportion", fontsize=12, fontfamily="DejaVu Serif")
    ax_props.set_ylim(0.0, 1.05)
    ax_props.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax_props.legend(loc="upper left")
    ax_comp.set_title("Compound Sentiment Over Time", fontsize=14, fontfamily="DejaVu Serif")
    ax_comp.set_ylabel("Compound", fontsize=12, fontfamily="DejaVu Serif")
    ax_comp.set_ylim(-1.1, 1.1)
    ax_comp.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax_comp.legend(loc="upper left")
    ax_kw.set_title("Keyword Counts Over Time", fontsize=14, fontfamily="DejaVu Serif")
    ax_kw.set_ylabel("Count", fontsize=12, fontfamily="DejaVu Serif")
    ax_kw.set_xlabel("Time", fontsize=12, fontfamily="DejaVu Serif")
    max_kw = max(max(v) for v in keywords.values()) or 1
    ax_kw.set_ylim(0, max_kw * 1.1)
    ax_kw.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    order   = ["Depression", "Hopelessness", "Anxiety", "ADHD", "Filler"]
    handles = [lines_kw[label] for label in order]
    ax_kw.legend(handles, order, loc="upper left")

    ax_props.set_xlim(min(times), max(times))

    def update(i):
        t = times[: i + 1]

        for k, ln in lines_props.items():
            ln.set_data(t, sentiment_props[k][: i + 1])

        line_comp.set_data(t, compound[: i + 1])

        for k, ln in lines_kw.items():
            ln.set_data(t, keywords[k][: i + 1])

        return list(lines_props.values()) + [line_comp] + list(lines_kw.values())

    _log(f"Building combined animation ({len(times)} frames)…")
    anim = FuncAnimation(fig, update, frames=len(times), interval=500, blit=True)

    out_dir = "visualizations"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{base_name}_sentiment_keyword_trends.mp4")
    _log(f"Saving to {out_path}")
    anim.save(out_path, dpi=150, fps=2)

    plt.close(fig)
    _log("Combined animation done.")

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
