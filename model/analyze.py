"""
analyze.py
Runs two HuggingFace models on each post:
  - Emotion classifier: anger, disgust, fear, joy, neutral, sadness, surprise
  - Sentiment classifier: positive, neutral, negative

Both models are ~67MB and download automatically on first run.
No GPU required — runs fine on CPU for datasets under ~500 posts.
"""

import numpy as np
from transformers import pipeline

# ── Model IDs ──────────────────────────────────────────────────────────────────
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
SENTIMENT_LABELS = ["negative", "neutral", "positive"]


def load_models():
    """Download (first run) and load both models. Prints progress."""
    print("Loading emotion model... (downloads ~67MB on first run)")
    emotion_classifier = pipeline(
        "text-classification",
        model=EMOTION_MODEL,
        top_k=None,  # return scores for ALL emotion labels
        truncation=True,
        max_length=512,
    )

    print("Loading sentiment model...")
    sentiment_classifier = pipeline(
        "text-classification",
        model=SENTIMENT_MODEL,
        top_k=None,
        truncation=True,
        max_length=512,
    )

    print("✓ Models ready\n")
    return emotion_classifier, sentiment_classifier


def _scores_to_dict(results, expected_labels):
    """Convert a list of {label, score} dicts into a clean {label: score} dict."""
    d = {item["label"].lower(): item["score"] for item in results}
    # Fill any missing labels with 0 (shouldn't happen, but defensive)
    return {label: d.get(label, 0.0) for label in expected_labels}


def analyse_posts(posts, emotion_classifier, sentiment_classifier):
    """
    Run both models over all posts.
    Returns a list of dicts, each with the original post data plus:
      - emotions: {anger: float, joy: float, ...}
      - dominant_emotion: str
      - sentiment: {negative: float, neutral: float, positive: float}
      - sentiment_score: float  (−1 to +1, useful for plotting)
    """
    print(f"Analysing {len(posts)} posts...")
    results = []

    for i, post in enumerate(posts):
        text = post["text"]

        # Run models
        emotion_output = emotion_classifier(text)[0]
        sentiment_output = sentiment_classifier(text)[0]

        # Parse scores
        emotions = _scores_to_dict(emotion_output, EMOTION_LABELS)
        sentiment = _scores_to_dict(sentiment_output, SENTIMENT_LABELS)

        # Derived values
        dominant_emotion = max(emotions, key=lambda k: emotions[k])

        sentiment_score = sentiment["positive"] - sentiment["negative"]  # −1 to +1

        results.append(
            {
                **post,
                "emotions": emotions,
                "dominant_emotion": dominant_emotion,
                "sentiment": sentiment,
                "sentiment_score": round(sentiment_score, 4),
            }
        )

        # Progress indicator every 5 posts
        if (i + 1) % 5 == 0 or (i + 1) == len(posts):
            print(f"  {i + 1}/{len(posts)} posts analysed")

    print(f"\n✓ Analysis complete\n")
    return results


def aggregate_by_hour(results):
    """
    Group results by hour and compute mean emotion + sentiment scores per hour.
    Returns a dict: {hour: {emotion_label: mean_score, ..., sentiment_score: mean}}
    """
    from collections import defaultdict

    hourly = defaultdict(list)
    for r in results:
        hourly[r["hour"]].append(r)

    aggregated = {}
    for hour, posts in sorted(hourly.items()):
        agg = {"hour": hour, "post_count": len(posts)}
        # Mean score for each emotion
        for label in EMOTION_LABELS:
            agg[label] = np.mean([p["emotions"][label] for p in posts])
        # Mean sentiment score
        agg["sentiment_score"] = np.mean([p["sentiment_score"] for p in posts])
        # Most common dominant emotion
        dominant_counts = {}
        for p in posts:
            e = p["dominant_emotion"]
            dominant_counts[e] = dominant_counts.get(e, 0) + 1
            agg["dominant_emotion"] = max(
                dominant_counts, key=lambda k: dominant_counts[k]
            )

        aggregated[hour] = agg

    return aggregated


def print_summary(results):
    """Print a readable summary of key findings to the terminal."""
    from collections import Counter

    print("=" * 60)
    print("  FINDINGS SUMMARY")
    print("=" * 60)

    dominant_emotions = [r["dominant_emotion"] for r in results]
    emotion_counts = Counter(dominant_emotions)

    print("\nDominant emotions across all posts:")
    for emotion, count in emotion_counts.most_common():
        bar = "█" * int(count * 40 / len(results))
        print(
            f"  {emotion:<12} {bar} {count} posts ({100 * count / len(results):.0f}%)"
        )

    avg_sentiment = np.mean([r["sentiment_score"] for r in results])
    print(
        f"\nOverall sentiment score: {avg_sentiment:+.3f}  (−1 = negative, +1 = positive)"
    )

    # Most surprising finding: biggest sentiment swing between any two consecutive hours
    hourly = aggregate_by_hour(results)
    hours = sorted(hourly.keys())
    if len(hours) > 1:
        swings = []
        for i in range(len(hours) - 1):
            h1, h2 = hours[i], hours[i + 1]
            swing = hourly[h2]["sentiment_score"] - hourly[h1]["sentiment_score"]
            swings.append((abs(swing), h1, h2, swing))
        swings.sort(reverse=True)
        _, h1, h2, swing = swings[0]
        direction = "positive" if swing > 0 else "negative"
        print(
            f"\nBiggest sentiment shift: Hour {h1} → Hour {h2}  ({swing:+.3f}, turned more {direction})"
        )

    print("\n" + "=" * 60)
