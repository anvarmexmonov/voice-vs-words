"""Regenerate the README figures from the numbers reported in docs/.

The numbers are copied from the result tables (selected layers only).
Full per-layer curves come from the experiment scripts' own CSV outputs.
Run:  python scripts/make_figures.py
"""
import os
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREY = "#2b6cb0", "#dd6b20", "#a0aec0"

# ---------- Figure 1: Stage 1, middle vs top encoder layers ----------
rows = [  # label, VOICE on conflict, PULL -> words
    ("eGeMAPS\n(no words)", 75.2, 11.8),
    ("WavLM\nL6", 95.6, 9.7), ("WavLM\nL19", 78.4, 49.3),
    ("Whisper\nL6", 95.4, 6.6), ("Whisper\nfinal", 83.3, 50.1),
    ("Qwen enc\nL9", 94.6, 7.1), ("Qwen enc\nto LLM", 76.2, 38.0),
]
x = np.arange(len(rows)); w = 0.38
fig, ax = plt.subplots(figsize=(10, 4.3))
ax.bar(x - w/2, [r[1] for r in rows], w, color=BLUE, label="Voice readable on conflict clips (VOICE)")
ax.bar(x + w/2, [r[2] for r in rows], w, color=ORANGE, label="Matched-speech reader follows the WORDS (PULL)")
for i, r in enumerate(rows):
    ax.text(i - w/2, r[1] + 1, f"{r[1]:.0f}", ha="center", fontsize=9)
    ax.text(i + w/2, r[2] + 1, f"{r[2]:.0f}", ha="center", fontsize=9)
ax.axhline(25, color=GREY, ls="--", lw=1); ax.text(-0.75, 26.5, "chance", color="grey", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows]); ax.set_ylim(0, 105); ax.set_ylabel("% of conflict clips")
ax.set_title("Stage 1: middle encoder layers keep the voice; top layers lean toward the words")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_encoder_probes.png", dpi=160); plt.close(fig)

# ---------- Figure 2: Stage 3.0, steering by LLM layer ----------
layers = [4, 8, 12, 16, 20, 24, 28]
probe = [8, 8, 8, 12, 32, 42, 49]
rand = [8, 8, 7, 8, 8, 7, 12]
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.axvspan(16, 20, color="#fed7d7", alpha=0.6, label="voice is overwritten here")
ax.plot(layers, probe, "o-", color=BLUE, lw=2, label="Hand-made note (probe label)")
ax.plot([20, 24, 28], [33, 43, 51], "s--", color="#2f855a", lw=1.5, label="Hand-made note (true label)")
ax.plot(layers, rand, "o-", color=GREY, lw=2, label="Random direction, same size")
ax.axhline(7.5, color="black", ls=":", lw=1); ax.text(4.2, 9.5, "no note: 7.5%", fontsize=9)
ax.set_xticks(layers); ax.set_xlabel("LLM layer where the note is added (last token)")
ax.set_ylabel("Conflict clips answered with the VOICE (%)"); ax.set_ylim(0, 60)
ax.set_title("Stage 3.0: a note only works after layer ~16-20")
ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(0, 0.92), frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_steering_by_layer.png", dpi=160); plt.close(fig)

# ---------- Figure 3: Stage 3.0, energy not emotion ----------
emos = ["angry", "happy", "neutral", "sad"]; recall = [100, 35, 56, 50]
mostly = ["", "-> angry", "-> sad", "-> neutral"]
fig, ax = plt.subplots(figsize=(6.5, 3.8))
bars = ax.bar(emos, recall, color=[ORANGE, ORANGE, BLUE, BLUE])
for b, r, m in zip(bars, recall, mostly):
    ax.text(b.get_x() + b.get_width()/2, r + 2, f"{r}%", ha="center")
    if m: ax.text(b.get_x() + b.get_width()/2, 5, m, ha="center", color="white", fontsize=8.5, rotation=90, va="bottom")
ax.set_ylim(0, 112); ax.set_ylabel("Recall of the true voice (%)")
ax.set_title("Hand-made note (true label, L28): carries energy, not the exact emotion", fontsize=10.5)
ax.text(0.5, 106, "high energy", ha="center", color=ORANGE, fontsize=9); ax.text(2.5, 106, "low energy", ha="center", color=BLUE, fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_energy_not_emotion.png", dpi=160); plt.close(fig)
print("figures written to", os.path.abspath(OUT))
