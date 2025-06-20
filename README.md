NEURONǒok – Qualitative Psychology Interview Analyzer

NEURONǒok is a Python-based system for analyzing qualitative psychology interviews. 
It transcribes interview audio files, identifies psychological symptoms 
and contextual themes, analyzes silence durations, and visualizes results using 
sentiment and keyword trends. Designed with research and clinical utility in mind, 
this tool streamlines the traditionally manual process of qualitative analysis.

Features
- 🎙️ Automatic Transcription w/ OpenAI Whisper via Hugginface
- 💬 Sentiment Analysis: via NLTK VADER
- 🧠 Symptom Keyword Detection: (Depression, Anxiety, ADHD, Hopelessness)
- 🏷️ Contextual Theme Analysis: (Family, School, Finances, Trauma, etc.)
- 🔇 Silence Gap Detection: (≥3s)
- 📈 Animated Visualizations: (Sentiment + Keyword Trends)
- 📝 Structured Summary Reports

1) ## 🚀 Quick Start

1. Clone the Repository
git clone https://github.com/yourusername/NEURONook.git
cd NEURONook

2) Create and Activate a Virtual Environment
python -m venv venv
source venv/bin/activate

3) Install Required Packages
pip install -r requirements.txt
+
# macOS
brew install ffmpeg

# Ubuntu
sudo apt install ffmpeg

4) 4. Run the Analyzer
python -m scripts.main audio_files/your_interview.mp3

Project Structure: 

NEURONook/
├── audio_files/            # Place your input MP3 files here
├── transcripts/            # Output transcripts with timestamps and sentiment
├── analysis_results/       # JSON analysis + summaries files
├── visualizations/         # Animated plots of sentiment/keyword trends
└── scripts/                # Where the real work happens
    ├── main.py             # Main pipeline runner
    ├── transcriber.py      # Handles Whisper transcription
    ├── analysis.py         # Sentiment, symptom, and theme analysis
    ├── summary.py          # Summary report generator
    ├── visualization.py    # Animated sentiment/keyword plots
    └── word_visualization.py  # Word frequency plotting

! Ethical Use Disclaimer

This tool is intended for educational or synthetic interview data only.
Do not use NEURONǒok on real clinical data without IRB approval or informed consent.
Always follow ethical research guidelines.

Requirements
- Python 3.8+
- transformers
- torch
- torchaudio
- nltk
- matplotlib
- tqdm

Future Plans
- Speaker diarization (e.g., WhisperX or PyAnnote)
- Emotion classification (beyond sentiment)
- Interactive dashboard (e.g., Streamlit)
- Co-occurrence heatmaps for symptom-theme overlap

Credits
Developed by Abraham Rahman 
Project submitted for CSCI 120: Computer Science  Clark University (Spring 2025) + improved on afterwards



