"""
Module: summary.py
Location: scripts/summary.py
Purpose: Generate a concise, well-formatted clinical-style summary from aggregated analysis results.
Includes: summary of symptom themes, topical themes, sentiment, reading metrics, silences, question counts, and raw totals.
"""
from scripts.analysis import TOPICAL_KEYWORDS
def aggregate_results(raw_results, silence_threshold=3.0):
    """
    Convert a list of per-chunk analysis dicts into aggregated metrics, including:
    - symptom theme_counts: total counts per symptom
    - topical theme_counts: total counts per topical category (from chunk keys)
    - sentiment: average compound score
    - silences, reading metrics, question counts, etc.
    """
    # Initialize counts
    symptom_counts = { 'Depression': 0, 'Hopelessness': 0, 'Anxiety': 0, 'ADHD': 0 }
    topical_counts = { topic: 0 for topic in TOPICAL_KEYWORDS }
    sentiment_list = []
    silences = {}
    total_words = 0
    question_count = 0
    prev_end = None
    first_start = None
    last_end = None

    for chunk in raw_results:
        start = chunk.get('start', 0.0)
        end = chunk.get('end', 0.0)
        comp = chunk.get('compound', 0.0) or 0.0
        text = chunk.get('text', '') or ''
        words = chunk.get('total_words', 0)

        # Track session span
        if first_start is None:
            first_start = start
        last_end = end

        # Count questions
        question_count += text.count('?')

        # Aggregate symptom counts from analysis keys
        for sym in list(symptom_counts.keys()):
            key = f"{sym.lower()}_count"
            symptom_counts[sym] += chunk.get(key, 0)

        # Aggregate topical counts (fallback to text scan)
        for topic in topical_counts:
            key = f"{topic.replace(' ', '_').lower()}_count"
            if key in chunk:
                topical_counts[topic] += chunk.get(key, 0)
            else:
                topical_counts[topic] += sum(tok.strip('.,?!') in TOPICAL_KEYWORDS[topic]
                                              for tok in text.lower().split())

        # Sentiment
        sentiment_list.append(comp)

        # Word count
        total_words += words

        # Silence detection
        if prev_end is not None:
            silence_dur = round(start - prev_end, 2)
            if silence_dur >= silence_threshold:
                label = f"{prev_end:.2f}-{start:.2f}s"
                silences[label] = silence_dur

        prev_end = end

    # Clean zero counts
    symptom_counts = {k:v for k,v in symptom_counts.items() if v>0}
    topical_counts = {k:v for k,v in topical_counts.items() if v>0}

    # Averages and metrics
    avg_compound = (sum(sentiment_list)/len(sentiment_list)) if sentiment_list else 0.0
    duration = (last_end - first_start) if (first_start is not None and last_end is not None) else 0.0
    avg_wpm = (total_words/duration*60) if duration>0 else 0.0
    longest = max(silences.values()) if silences else 0.0
    shortest = min(silences.values()) if silences else 0.0
    longest_seg = max(silences, key=silences.get) if silences else None
    shortest_seg = min(silences, key=silences.get) if silences else None

    return {
        'symptom_counts': symptom_counts,
        'topical_counts': topical_counts,
        'sentiment': {'compound': avg_compound},
        'silences': silences,
        'total_words': total_words,
        'avg_wpm': avg_wpm,
        'longest_silence': (longest_seg, longest),
        'shortest_silence': (shortest_seg, shortest),
        'questions': question_count
    }


def generate_summary(raw_results):
    """
    Build a clinical-style session summary with both formatted and raw data sections.
    """
    data = raw_results if isinstance(raw_results, dict) else aggregate_results(raw_results)

    # Top symptoms
    sym = data.get('symptom_counts', {})
    top_sym = ', '.join(sorted(sym, key=sym.get, reverse=True)[:2]) or 'None'
    # Top topics
    topi = data.get('topical_counts', {})
    top_topics = ', '.join(sorted(topi, key=topi.get, reverse=True)[:2]) or 'None'
    # Overall sentiment label
    comp = data['sentiment']['compound']
    overall = 'Positive' if comp>=0.05 else 'Negative' if comp<=-0.05 else 'Neutral'

    # Formatted summary
    lines = ["==== Session Summary ====", ""]
    lines.append(f"• Prominent Symptoms : {top_sym}")
    lines.append(f"• Main Topics         : {top_topics}")
    lines.append(f"• Overall Sentiment   : {overall} ({comp:.2f})")
    lines.append(f"• Word Count          : {data['total_words']}")
    lines.append(f"• Avg WPM             : {data['avg_wpm']:.1f}")
    ls_seg, ls_dur = data['longest_silence']
    ss_seg, ss_dur = data['shortest_silence']
    lines.append(f"• Longest Silence     : {ls_seg or 'None'} ({ls_dur:.2f}s)")
    lines.append(f"• Shortest Silence    : {ss_seg or 'None'} ({ss_dur:.2f}s)")
    lines.append(f"• Questions Asked     : {data['questions']}")
    if data['silences']:
        lines.append(f"• Silences Detected   :")
        for seg, dur in data['silences'].items():
            lines.append(f"     - {seg} ({dur:.2f}s)")

    # Raw totals section
    lines.append("")
    lines.append("-- Raw Totals --")
    lines.append(f"Theme Counts          : {data['symptom_counts']}")
    lines.append(f"Topic Counts          : {data['topical_counts']}")
    lines.append(f"Avg Sentiment Score   : {data['sentiment']['compound']:.3f}")
    if data['silences']:
        lines.append("Silence Segments      :")
        for seg, dur in data['silences'].items():
            lines.append(f"     - {seg}")
    else:
        lines.append("Silence Segments      : None")
    lines.append(f"Questions Asked       : {data['questions']}")

    return '\n'.join(lines)
