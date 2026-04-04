"""
visualize.py
Generates two charts saved to the output/ folder:
  1. emotion_arc.png   — stacked area chart showing emotion blend across hours
  2. sentiment_arc.png — line chart showing sentiment score over time
  3. emotion_heatmap.png — heatmap of emotion intensity per hour (great for screen recording)
"""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

# Colour palette — each emotion gets a distinct, meaningful colour
EMOTION_COLOURS = {
    "joy": "#F5C542",  # warm yellow
    "surprise": "#F0A500",  # amber
    "neutral": "#A8DADC",  # soft teal
    "fear": "#9B5DE5",  # purple
    "sadness": "#457B9D",  # steel blue
    "anger": "#E63946",  # red
    "disgust": "#6D6875",  # muted mauve
}

EMOTION_ORDER = ["joy", "surprise", "neutral", "fear", "sadness", "anger", "disgust"]


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_emotion_arc(hourly_data):
    """
    Stacked area chart: emotional blend per hour.
    This is your hero visual — shows how the mood of a conversation
    changes over time as a layered, colourful wave.
    """
    _ensure_output_dir()

    hours = sorted(hourly_data.keys())
    x = np.array(hours)

    # Stack emotions in a meaningful order (positive → neutral → negative)
    emotion_stacks = {
        e: np.array([hourly_data[h][e] for h in hours]) for e in EMOTION_ORDER
    }

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#0F0F1A")
    ax.set_facecolor("#0F0F1A")

    # Normalise so stacks sum to 1 per hour (proportional view)
    totals = sum(emotion_stacks[e] for e in EMOTION_ORDER)
    totals = np.where(totals == 0, 1, totals)  # avoid divide-by-zero
    normalised = {e: emotion_stacks[e] / totals for e in EMOTION_ORDER}

    # Plot stacked areas
    baseline = np.zeros(len(x))
    for emotion in EMOTION_ORDER:
        values = normalised[emotion]
        ax.fill_between(
            x,
            baseline,
            baseline + values,
            color=EMOTION_COLOURS[emotion],
            alpha=0.85,
            label=emotion.capitalize(),
        )
        baseline += values

    # Style
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Hour after announcement", fontsize=13, color="#CCCCCC", labelpad=10)
    ax.set_ylabel(
        "Proportion of emotional signal", fontsize=13, color="#CCCCCC", labelpad=10
    )
    ax.set_title(
        "Emotional Arc of Public Reaction\nHour-by-hour breakdown across all posts",
        fontsize=16,
        fontweight="bold",
        color="white",
        pad=20,
    )

    ax.tick_params(colors="#AAAAAA", labelsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    # Legend
    legend_patches = [
        mpatches.Patch(color=EMOTION_COLOURS[e], label=e.capitalize())
        for e in EMOTION_ORDER
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper right",
        framealpha=0.2,
        facecolor="#111122",
        edgecolor="#444466",
        labelcolor="white",
        fontsize=10,
    )

    # Vertical hour markers
    for h in x:
        ax.axvline(x=h, color="#FFFFFF", alpha=0.04, linewidth=0.5)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "emotion_arc.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓ Saved: {path}")
    return path


def plot_sentiment_arc(hourly_data):
    """
    Line chart with shaded fill showing sentiment score (−1 to +1) over time.
    The zero line is prominent — crossing it is your story.
    """
    _ensure_output_dir()

    hours = sorted(hourly_data.keys())
    x = np.array(hours)
    y = np.array([hourly_data[h]["sentiment"] for h in hours])
    counts = [hourly_data[h]["post_count"] for h in hours]

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#0F0F1A")
    ax.set_facecolor("#0F0F1A")

    # Fill above/below zero in different colours

    # Main line
    ax.plot(x, y, color="white", linewidth=2.5, zorder=5)
    ax.scatter(x, y, color="white", s=60, zorder=6)

    # Zero line
    ax.axhline(0, color="#888888", linewidth=1, linestyle="--", alpha=0.6)

    # Annotate dominant emotion per hour
    for i, h in enumerate(hours):
        dom = hourly_data[h]["dominant_emotion"]
        ypos = y[i] + (0.04 if y[i] >= 0 else -0.06)
        ax.text(
            h,
            ypos,
            dom[:3].upper(),
            fontsize=7,
            color="#CCCCCC",
            ha="center",
            va="bottom",
            alpha=0.75,
        )

    ax.set_xlim(x[0] - 0.3, x[-1] + 0.3)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Hour after announcement", fontsize=13, color="#CCCCCC", labelpad=10)
    ax.set_ylabel(
        "Sentiment score  (−1 = negative, +1 = positive)", fontsize=12, color="#CCCCCC"
    )
    ax.set_title(
        "Sentiment Arc Over Time\nHow the mood shifted hour by hour",
        fontsize=16,
        fontweight="bold",
        color="white",
        pad=20,
    )

    ax.tick_params(colors="#AAAAAA", labelsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    ax.legend(
        framealpha=0.2,
        facecolor="#111122",
        edgecolor="#444466",
        labelcolor="white",
        fontsize=10,
    )

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "sentiment_arc.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓ Saved: {path}")
    return path


def plot_emotion_heatmap(hourly_data):
    """
    Heatmap: hours (x) × emotions (y), coloured by intensity.
    Ideal for screen recording — lots of colour, easy to read at a glance.
    """
    _ensure_output_dir()

    hours = sorted(hourly_data.keys())
    emotions = EMOTION_ORDER

    # Build matrix: rows = emotions, columns = hours
    matrix = np.array([[hourly_data[h][e] for h in hours] for e in emotions])

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#0F0F1A")
    ax.set_facecolor("#0F0F1A")

    im = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=0, vmax=matrix.max())

    # Axes
    ax.set_xticks(range(len(hours)))
    ax.set_xticklabels(
        [f"Hr {h}" for h in hours], color="#CCCCCC", fontsize=9, rotation=45
    )
    ax.set_yticks(range(len(emotions)))
    ax.set_yticklabels([e.capitalize() for e in emotions], color="#CCCCCC", fontsize=11)

    ax.set_title(
        "Emotion Intensity Heatmap  (darker = weaker, brighter = stronger)",
        fontsize=14,
        fontweight="bold",
        color="white",
        pad=15,
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(colors="#AAAAAA")
    cbar.set_label("Intensity", color="#AAAAAA", fontsize=10)

    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "emotion_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓ Saved: {path}")
    return path


def generate_all(hourly_data):
    """Run all three charts."""
    print("Generating visualisations...")
    paths = [
        plot_emotion_arc(hourly_data),
        plot_sentiment_arc(hourly_data),
        plot_emotion_heatmap(hourly_data),
    ]
    print("\n✓ All charts saved to output/\n")
    return paths
