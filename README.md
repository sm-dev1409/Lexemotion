# Emotion Arc Analysis
 
Mapping the emotional arc of public reaction to a major event - hour by hour - using NLP.
 
Built in 24 hours as part of the Stanford Venture Studio Fellowship application.  
No prior NLP experience. This was the point.
 
---
 
## What it does
 
Takes a collection of posts or comments timestamped by hour and runs two HuggingFace models on them:
 
- **Emotion classification** - anger, disgust, fear, joy, neutral, sadness, surprise  
- **Sentiment scoring** - a continuous −1 to +1 score per post
 
Then aggregates by hour and produces three charts showing how public mood evolves over time.
 
## Key finding
 
*Surprise dominates early but fades fast - it accounts for ~70% of the emotional signal at the moment of announcement, then drops sharply to ~20% within an hour.
As Surprise declines, Fear and Sadness fill the void, suggesting audiences shift from an initial "wow" reaction to a more anxious, processing mindset remarkably quickly.*
 
---
 
## Charts
 
| Chart | What it shows |
|---|---|
| `emotion_arc.png` | Stacked area - the emotional blend per hour |
| `sentiment_arc.png` | Sentiment score over time with dominant emotion labels |
| `emotion_heatmap.png` | Heatmap of emotion intensity across all hours |
 
---
 
## Dataset
 
The bundled dataset (`data/sample_data.json`) is 30 synthetic Reddit-style posts spanning 14 hours, representing public reaction to a major AI announcement. Each post has a source subreddit and hour timestamp.
 
You can swap in your own data - see **Usage** below.
 
---
 
## Setup
 
```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/emotion-arc-analysis.git
cd emotion-arc-analysis
 
# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows
 
# 3. Install dependencies
pip install -r requirements.txt
```
 
Models download automatically (~67MB) on first run. No API keys needed for the default mode.
 
---
 
## Usage
 
```bash
# Run on the bundled sample data (default)
python main.py
 
# Quick 5-post demo - useful for testing your setup first
python main.py --demo
 
# Run on your own text file (one post per line)
python main.py --file path/to/your_data.txt
 
# Fetch live from Reddit (requires Reddit API keys - see below)
python main.py --reddit MachineLearning "GPT-4"
```
 
Charts are saved to `output/`.
 
### Reddit API setup (optional)
 
1. Go to https://www.reddit.com/prefs/apps  
2. Create a "script" app  
3. Set environment variables:
 
```bash
export REDDIT_CLIENT_ID=your_id
export REDDIT_CLIENT_SECRET=your_secret
```
 
---
 
## Project structure
 
```
emotion-arc-analysis/
├── main.py              # Entry point - run this
├── requirements.txt
├── model/
│   ├── fetch_data.py    # Load from sample data, text file, or Reddit
│   ├── analyze.py       # HuggingFace models + aggregation
│   └── visualize.py     # Chart generation (Matplotlib)
├── data/
│   └── sample_data.json # Bundled dataset - 30 posts, 14 hours
└── output/              # Charts saved here after running
```
 
---
 
## Models used
 
- [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) - emotion classification  
- [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) - sentiment scoring
 
Both run on CPU. No GPU required.
 
---
 
## License
 
GNU GENERAL PUBLIC LICENSE (Version 3)
