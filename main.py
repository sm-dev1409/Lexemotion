"""
main.py - Emotion Arc Analysis
===============================
Analyses the emotional arc of public reaction to an event over time.
Uses two HuggingFace models (no API key, no GPU required).

Usage:
  python main.py                          # Run on bundled sample data
  python main.py --file path/to/file.txt  # Run on your own text file
  python main.py --reddit SUBREDDIT QUERY # Fetch live from Reddit (needs API keys)
  python main.py --demo                   # Quick demo on 5 posts (fast, for testing)
"""

import argparse
import os
import sys

# Make sure src/ is importable
sys.path.insert(0, os.path.dirname(__file__))

from model.analyze import aggregate_by_hour, analyse_posts, load_models, print_summary
from model.fetch import load_from_reddit, load_from_text_file, load_sample_data
from model.visualize import generate_all


def parse_args():
    parser = argparse.ArgumentParser(
        description="Emotion Arc Analysis - see how public mood shifts over time"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--file",
        type=str,
        metavar="PATH",
        help="Path to a plain text file (one text per line)",
    )
    group.add_argument(
        "--reddit",
        nargs=2,
        metavar=("SUBREDDIT", "QUERY"),
        help="Fetch from Reddit: --reddit MachineLearning 'GPT-4'",
    )
    group.add_argument(
        "--demo",
        action="store_true",
        help="Run a quick 5-post demo (fast, good for testing setup)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  EMOTION ARC ANALYSIS")
    print("  Mapping public emotion over time using NLP")
    print("=" * 60 + "\n")

    # ── Load data ──────────────────────────────────────────────────────────────
    if args.file:
        posts = load_from_text_file(args.file)

    elif args.reddit:
        subreddit, query = args.reddit
        posts = load_from_reddit(subreddit, query)

    elif args.demo:
        # Quick 5-post demo - useful for checking setup before the full run
        all_posts = load_sample_data()
        posts = all_posts[:5]
        print("(Demo mode: using first 5 posts only)\n")

    else:
        # Default: bundled sample data
        posts = load_sample_data()

    if not posts:
        print("No posts loaded. Exiting.")
        sys.exit(1)

    # ── Analyse ────────────────────────────────────────────────────────────────
    emotion_model, sentiment_model = load_models()
    results = analyse_posts(posts, emotion_model, sentiment_model)

    # ── Aggregate by hour ──────────────────────────────────────────────────────
    hourly_data = aggregate_by_hour(results)

    # ── Print summary ──────────────────────────────────────────────────────────
    print_summary(results)

    # ── Generate charts ────────────────────────────────────────────────────────
    if len(hourly_data) >= 2:
        chart_paths = generate_all(hourly_data)
        print("Charts saved:")
        for p in chart_paths:
            print(f"  → {p}")
    else:
        print(
            "Not enough time points to generate arc charts (need at least 2 hours of data)."
        )
        print(
            "Try the full sample dataset or add more data with different hour values."
        )

    print("\nDone. Open the output/ folder to see your charts.\n")


if __name__ == "__main__":
    main()
