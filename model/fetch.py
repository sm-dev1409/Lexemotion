"""
fetch_data.py
Handles loading data from three sources:
  1. The bundled sample dataset (default, no setup required)
  2. A plain text file you provide
  3. Reddit (requires a free Reddit API key)
"""

import json
import os


def load_sample_data():
    """Load the bundled sample dataset about a major AI announcement."""
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "sample_data.json"
    )
    with open(data_path, "r") as f:
        posts = json.load(f)
    print(
        f"✓ Loaded {len(posts)} sample posts spanning {max(p['hour'] for p in posts) + 1} hours"
    )
    return posts


def load_from_text_file(filepath):
    """
    Load from a plain .txt file — one piece of text per line.
    Timestamps are assigned sequentially (no hour metadata).
    """
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    posts = []
    for i, line in enumerate(lines):
        posts.append(
            {
                "id": i + 1,
                "hour": i,  # treat each line as a sequential time step
                "text": line,
                "source": "custom",
            }
        )

    print(f"✓ Loaded {len(posts)} lines from {filepath}")
    return posts


def load_from_reddit(subreddit_name, query, limit=50):
    """
    Fetch posts from Reddit using PRAW.
    Requires REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET env vars.

    Setup:
      1. Go to https://www.reddit.com/prefs/apps
      2. Create a 'script' app
      3. Set environment variables:
           export REDDIT_CLIENT_ID=your_id
           export REDDIT_CLIENT_SECRET=your_secret
    """
    try:
        import praw
    except ImportError:
        raise ImportError("Run: pip install praw")

    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise ValueError(
            "Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables.\n"
            "See: https://www.reddit.com/prefs/apps"
        )

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent="emotion-arc-analysis/1.0",
    )

    subreddit = reddit.subreddit(subreddit_name)
    results = subreddit.search(query, limit=limit, sort="new")

    posts = []
    for i, post in enumerate(results):
        posts.append(
            {
                "id": i + 1,
                "hour": i,
                "text": post.title + " " + (post.selftext or ""),
                "source": f"r/{subreddit_name}",
            }
        )

    print(f"✓ Fetched {len(posts)} posts from r/{subreddit_name}")
    return posts
