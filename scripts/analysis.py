# scripts/analysis.py
# Lighter Phase
# # # # Analyzes the transcript chunks and returns the analysis results.
# # # # The analysis results include the sentiment scores, keyword counts, and topical theme counts.

import os
import time
import json
from datetime import datetime

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import wordpunct_tokenize  # <-- faster, no extra models
from tqdm import tqdm

# download once
nltk.download('vader_lexicon', quiet=True)

# ─── CONFIG ─────────────────────────────────────────────────────────────────
# --- keyword sets 
depression_keywords = {
    # core emotional states; General
    "depressed", "depressing", "hopeless", "sad", "down", "unhappy",
    "low", "numb", "empty", "worthless",
    
    # physical/behavioral symptoms
    "tired", "fatigued", "exhausted", "drained", "sluggish",
    "sleepy", "heavy", "slow", "can’t move", "no energy",
    "can’t get out of bed", "can’t do anything", "no motivation", # ADHD Conflict likely
    
    # suicidal ideation (watch for context)
    "suicidal", "don’t want to be here", "want it to stop", "no way out",
    "it’d be better if I weren’t here", "done with everything", "kill myself",
    
    # social withdrawal
    "alone", "lonely", "no one cares", "pushed away", "left out",
    "ignored", "abandoned", "unloved", "invisible", "nobody gets it",
    
    # emotional behavior
    "cry", "crying", "tears", "tearful", "broke down", "shut down",
    "can’t stop crying", "couldn't stop crying", "keep crying", 
    
    # ambiguous/self-deprecating language
    "messed up", "not okay", "not fine", "just tired", "it’s whatever",
    "I’m fine", "overwhelmed", # <-- CONTEXUAL; might be to be considered for erro 
    "checked out", "don’t care", "can’t do this", "burned out"
}

hopelessness_keywords = {
    "hopeless", "helpless", "stuck", "trapped",
    "giving up", "powerless",
    # new additions:
    "despair", "desperate", "no way", "nothing left",
    # cognitive symptoms
    "guilty", "worthless", "useless", "what’s the point",
    "hate myself", "hate my life", "can’t focus", "can’t think", "foggy",
    "not enough", "broken", "failed", "failure", "nothing matters",
    # Desperation and giving up
    "giving up", "gave up", "desperate", "nowhere to go", "give up",
    "can’t take it anymore", "breaking point", "last straw", "rock bottom", "burned out",
    "burnt out", "at the end of my rope", "no way out", "no way forward",
    # Futility and pointlessness
    "pointless", "what’s the point", "why bother", "nothing helps",
    "it doesn’t matter", "nothing changes", "waste of time", 
    # Negative self-evaluations (cognitive hopelessness)
    "worthless", "useless", "not enough", "failure", "broken", "messed up",
    "can’t do anything right", "keep messing up", "ruin everything",
}

anxiety_keywords = {
    "anxious", "anxiety", "nervous", "worried", "panic",
    "afraid", "scared", "tense", "stressed", "overwhelmed",
    "worry", "concerned", "jitters",
    "heart racing", "sweaty", "butterflies", "nauseous", "jumpy", "tense",
    "restless", "on edge", "lightheaded", "dizzy", "freaking out", 
    "spiraling", "can’t stop thinking", "mind racing",
     # Avoidant and indirect expressions
    "something’s wrong", "waiting for something bad", 
    "dreading it", "worst-case scenario", "bad feeling",
    "it’s too much", "always on alert", "paranoid", "unsafe",
}

adhd_keywords = {
    "distracted", "focus", "focused", "impulsive", "fidget",
    "bored", "forgetful", "procrastinate", "restless", "attention", "zoned out",
    "daydream", "drifted", "wandering", "zoning", "scatter",
    # Executive dysfunction
    "forgetful", "forget", "procrastinate", "procrastinating", "avoidance",
    "can’t start", "can’t finish", "keep forgetting", "disorganized",
    "missed it", "left it", "ran out of time", "late again",
     # 'Slang'(context-based)
    "brain fog", "brain won’t work", "all over the place", "head’s a mess",
    "spaced out", "mental chaos", "scatterbrained", "no filter", "chaotic brain"
}

filler_keywords = { # might have the most errors; this one should be checked w/ physical transcription
    # Classic fillers
    "um", "uh", "umm", "er", "ah", "eh", "hmm",

    # Common discourse markers
    "like", "you know", "i mean", "so", "well", "right", 
    "actually", "basically", "literally", "anyway", "okay", "alright",

    # Floor-holding / self-monitoring
    "let me think", "how do I put this", "sort of", "kind of", 
    "maybe", "probably", "i guess", "i suppose", "i don’t know", "i dunno",

    # Rephrasers and hedging
    "just", "honestly", "to be honest", "truthfully", "to be fair",
    "technically", "whatever", "stuff", "things", "you see",

    # Interruptive placeholders / vague expressions
    "and stuff", "or something", "and things", "whatnot", "thing is",
    "the thing is", "something like that", "or whatever",

    # Sentence softeners or pivots
    "anyways", "moving on", "like i said", "as I was saying",
    "you know what I mean", "does that make sense"
}

# --- topical keywords for conversation themes
TOPICAL_KEYWORDS = {
    'Work & Career': [
        'job', 'career', 'boss', 'promotion', 'unemployed', 'work', 'office', 'colleague', 'coworker', 'resume', 'interview', 'quit', 'laid off', 'fired', 'employed', 'internship', 'remote work', 'commute', 'overtime', 'paycheck', 'tasks', 'freelance', 'contract', 'salary', 'deadline', 'team meeting', 'burnout', 'job hunt', 'annual review'
    ],
    'Relationships': [
        'boyfriend', 'girlfriend', 'marriage', 'breakup', 'cheating', 'dating', 'partner', 'relationship', 'divorce', 'crush', 'fling', 'infidelity', 'love life', 'toxic', 'arguing', 'fight', 'affection', 'romantic', 'ex', 'hookup', 'situationship', 'trust issues', 'jealousy'
    ],
    'Family': [
        'mom', 'dad', 'parents', 'siblings', 'childhood', 'home', 'family', 'stepmom', 'stepdad', 'stepbrother', 'stepsister', 'cousins', 'grandma', 'grandpa', 'uncle', 'aunt', 'household', 'upbringing', 'family dinner', 'family issues', 'birth order', 'parenting', 'family reunion'
    ],
    'Health / Illness': [
        'hospital', 'sick', 'pain', 'diagnosis', 'medication', 'illness', 'disease', 'injury', 'doctor', 'therapy', 'surgery', 'treatment', 'health', 'symptoms', 'prescription', 'infection', 'recovery', 'check-up', 'mental health', 'chronic pain', 'side effects', 'specialist', 'ER visit'
    ],
    'School / Education': [
        'school', 'college', 'grades', 'exam', 'professor', 'class', 'university', 'test', 'homework', 'assignment', 'major', 'semester', 'credits', 'study', 'GPA', 'presentation', 'campus', 'graduation', 'midterms', 'thesis', 'drop out', 'tuition', 'library'
    ],
    'Financial Stress': [
        'money', 'debt', 'bills', 'rent', 'broke', 'budget', 'savings', 'expenses', 'paycheck', 'afford', 'financial aid', 'loan', 'scholarship', 'overdraft', 'poor', 'bankrupt', 'wallet', 'income', 'credit card', 'interest', 'rent overdue', 'late fees', 'utilities'
    ],
    'Loneliness / Isolation': [
        'alone', 'isolated', 'lonely', 'left out', 'disconnected', 'ignored', 'invisible', 'abandoned', 'solitude', 'alienated', 'unseen', 'excluded', 'missed', 'neglected', 'friendless', 'silent', 'nobody there', 'withdrawn'
    ],
    'Trauma / Abuse': [
        'violence', 'abuse', 'trauma', 'scared', 'ptsd', 'flashback', 'assault', 'harassment', 'molested', 'violated', 'shaken', 'nightmare', 'survivor', 'triggered', 'panic attack', 'screaming', 'bruised', 'fearful', 'escape'
    ],
    'Self-Worth': [
        'worthless', 'failure', 'ashamed', 'not enough', 'insecure', 'unlovable', 'unworthy', 'embarrassed', 'disgrace', 'inadequate', 'inferior', 'regret', 'self-esteem', 'hate myself', 'nobody cares', 'useless', 'self-hate', 'broken', 'not good enough', 'burden'
    ],
    'Future Plans': [
        'future', 'plan', 'goals', 'dreams', 'change', 'next step', 'path', 'direction', 'hope', 'ambition', 'what’s next', 'vision', 'motivation', 'new chapter', 'aspiration', 'long term', 'planning ahead', 'go forward', 'roadmap', 'growth mindset'
    ]
}

def _log(msg: str):  # Log messages with a timestamp
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def analyze_transcript_chunks(transcript_result):
    """Analyze transcript chunks for sentiment, symptom & topical keyword counts, and log a summary."""
    chunks = transcript_result.get("chunks", [])
    total = len(chunks)

    _log(f"Starting analysis of {total} chunks...")
    t0 = time.time()
    sia = SentimentIntensityAnalyzer()
    analysis = []

    # summary accumulators
    sum_words = 0
    sum_compound = 0.0
    sum_symptoms = { "depression":0, "hopelessness":0, "anxiety":0, "adhd":0, "filler":0 }
    sum_topical = { topic: 0 for topic in TOPICAL_KEYWORDS }

    for idx, chunk in enumerate(tqdm(chunks, desc="Analyzing chunks"), start=1):
        start_ts = chunk.get("start", 0.0)
        end_ts   = chunk.get("end",   0.0)
        text     = chunk.get("text",   "") or ""
        tokens   = wordpunct_tokenize(text.lower())

        # sentiment
        sentiment = sia.polarity_scores(text)
        sum_compound += sentiment["compound"]

        # symptom keyword counts
        dep = sum(tok in depression_keywords       for tok in tokens)
        hop = sum(tok in hopelessness_keywords     for tok in tokens)
        anx = sum(tok in anxiety_keywords          for tok in tokens)
        adh = sum(tok in adhd_keywords             for tok in tokens)
        fil = sum(tok in filler_keywords           for tok in tokens)
        total_words = len(tokens)

        # topical keyword counts
        topical_counts = {}
        for topic, keywords in TOPICAL_KEYWORDS.items():
            cnt = sum(tok.strip('.,?!') in keywords for tok in tokens)
            topical_counts[topic] = cnt
            sum_topical[topic] += cnt

        # accumulate
        sum_symptoms["depression"]   += dep
        sum_symptoms["hopelessness"] += hop
        sum_symptoms["anxiety"]      += anx
        sum_symptoms["adhd"]         += adh
        sum_symptoms["filler"]       += fil
        sum_words += total_words

        # build record
        record = {
            "start": start_ts,
            "end":   end_ts,
            "text":  text,
            **sentiment,
            "depression_count":   dep,
            "hopelessness_count": hop,
            "anxiety_count":      anx,
            "adhd_count":         adh,
            "filler_count":       fil,
            "total_words":        total_words,
        }
        # flatten topical counts
        for topic, cnt in topical_counts.items():
            key = topic.lower().replace(' ', '_') + '_count'
            record[key] = cnt

        analysis.append(record)

    elapsed = time.time() - t0
    _log(f"Finished analysis in {elapsed:.1f}s")

    # summary logging
    avg_compound = sum_compound / total if total else 0.0
    _log("======== Summary ========")
    _log(f"Total words             : {sum_words:,}")
    for symptom, val in sum_symptoms.items():
        _log(f"Total {symptom:12} : {val}")
    for topic, val in sum_topical.items():
        _log(f"Total {topic:20} : {val}")
    _log(f"Avg sentiment (compound): {avg_compound:.3f}")
    _log("==========================")

    return analysis

def save_analysis(analysis_data, base_name):
    os.makedirs("analysis_results", exist_ok=True)
    out_path = os.path.join("analysis_results", f"{base_name}_analysis.json")
    with open(out_path, "w") as f:
        json.dump(analysis_data, f, indent=2)
    _log(f"Saved analysis to {out_path}")
