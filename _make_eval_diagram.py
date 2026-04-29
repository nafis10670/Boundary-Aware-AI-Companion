"""Generate a vector diagram (SVG + PNG) of the evaluation methodology."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pathlib

OUT_DIR = pathlib.Path("plots")
OUT_DIR.mkdir(exist_ok=True)

# ── Colours ────────────────────────────────────────────────────────────────────
C_HL       = "#0077AA"
C_HL_LIGHT = "#E0EAF8"
C_GREEN    = "#008855"
C_GREEN_LT = "#E5F8EE"
C_WARN     = "#CC6600"
C_WARN_LT  = "#FBF0E8"
C_GRAY     = "#666688"
C_GRAY_LT  = "#F0F4FA"
C_TEXT     = "#111122"
C_BODY     = "#333355"
C_BORDER   = "#CCCCDD"
C_WHITE    = "#FFFFFF"
C_FILE_BG  = "#F7F9FF"
C_ARROW    = "#888899"

# ── Canvas ─────────────────────────────────────────────────────────────────────
FIG_W, FIG_H = 10, 14
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
fig.patch.set_facecolor(C_WHITE)
ax.set_facecolor(C_WHITE)


# ── Helper functions ───────────────────────────────────────────────────────────

def box(x, y, w, h, label, sublabel=None,
        fc=C_GRAY_LT, ec=C_BORDER, lw=1.5,
        label_size=13, label_weight="bold", label_color=C_TEXT,
        sub_size=10.5, sub_color=C_BODY, radius=0.18):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2,
    )
    ax.add_patch(rect)
    cy = y + h / 2 + (0.18 if sublabel else 0)
    ax.text(x + w / 2, cy, label,
            ha="center", va="center", fontsize=label_size,
            fontweight=label_weight, color=label_color, zorder=3)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 - 0.22, sublabel,
                ha="center", va="center", fontsize=sub_size,
                color=sub_color, zorder=3, style="italic")


def file_box(x, y, w, h, name, desc=None,
             fc=C_FILE_BG, ec=C_BORDER):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.12",
        facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=2,
        linestyle="--",
    )
    ax.add_patch(rect)
    cy = y + h / 2 + (0.12 if desc else 0)
    ax.text(x + w / 2, cy, name,
            ha="center", va="center", fontsize=10,
            fontweight="bold", color=C_BODY, zorder=3,
            fontfamily="monospace")
    if desc:
        ax.text(x + w / 2, y + h / 2 - 0.18, desc,
                ha="center", va="center", fontsize=8.5,
                color=C_GRAY, zorder=3)


def arrow(x1, y1, x2, y2, color=C_ARROW, lw=2.0, label=None):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color, lw=lw,
            mutation_scale=16,
        ),
        zorder=1,
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.12, my, label, fontsize=9, color=C_GRAY,
                va="center", style="italic")


def step_badge(x, y, label, color):
    circle = plt.Circle((x, y), 0.28, color=color, zorder=4)
    ax.add_patch(circle)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=C_WHITE, zorder=5)


def divider(x, y, w):
    ax.plot([x, x + w], [y, y], color=C_BORDER, lw=1.0, zorder=3)


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT  (y increases upward)
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Dataset ────────────────────────────────────────────────────────────────
box(2.2, 12.1, 5.6, 0.75,
    "INTIMA-MT Dataset",
    sublabel="375 conversations  ·  31 behaviour codes  ·  4 INTIMA categories",
    fc=C_HL_LIGHT, ec=C_HL, lw=2,
    label_size=13, label_color=C_HL)

arrow(5.0, 12.1, 5.0, 11.55, color=C_HL)

# ── 2. Exclusion note ─────────────────────────────────────────────────────────
ax.text(5.15, 11.82, "−9 ICL examples  (data leakage prevention)",
        ha="left", va="center", fontsize=8.5,
        color=C_GRAY, style="italic")

# ── 3. STEP 1 box ─────────────────────────────────────────────────────────────
S1_Y = 10.2
box(0.6, S1_Y, 8.8, 1.2, "", fc=C_HL_LIGHT, ec=C_HL, lw=2)
step_badge(1.08, S1_Y + 0.95, "1", C_HL)
ax.text(1.5, S1_Y + 0.95, "Step 1  —  Backbone Inference",
        ha="left", va="center", fontsize=13,
        fontweight="bold", color=C_HL, zorder=3)
ax.text(5.0, S1_Y + 0.38,
        "Run the agentic system and safety-prompted baseline on 366 conversations.\n"
        "Backbone: LLaMA 3.1 8B  ·  Qwen 2.5 72B  ·  Gemma 3 27B  ·  Mistral NeMo 12B",
        ha="center", va="center", fontsize=9.5, color=C_BODY, zorder=3)

# Output files from Step 1
arrow(3.2, S1_Y, 2.6, 9.35, color=C_HL)
arrow(6.8, S1_Y, 7.4, 9.35, color=C_HL)

file_box(1.0, 8.6, 3.2, 0.72,
         "responses.jsonl",
         "system & baseline responses\n+ conversation context",
         fc=C_FILE_BG, ec=C_HL)

file_box(5.8, 8.6, 3.2, 0.72,
         "run_metadata.json",
         "backbone model · dataset path\n· timestamp · conv count",
         fc=C_FILE_BG, ec=C_HL)

# merge arrows back to single flow
arrow(2.6, 8.6, 4.2, 8.0, color=C_HL)
arrow(7.4, 8.6, 5.8, 8.0, color=C_HL)

# ── 4. STEP 2 box ─────────────────────────────────────────────────────────────
S2_Y = 6.15
box(0.6, S2_Y, 8.8, 1.72, "", fc=C_GREEN_LT, ec=C_GREEN, lw=2)
step_badge(1.08, S2_Y + 1.38, "2", C_GREEN)
ax.text(1.5, S2_Y + 1.38, "Step 2  —  Multi-Judge Ensemble",
        ha="left", va="center", fontsize=13,
        fontweight="bold", color=C_GREEN, zorder=3)

# Three judge sub-boxes — centred horizontally in the outer box
JUDGE_MODELS = [
    ("Qwen 2.5 72B", "qwen"),
    ("Gemma 3 27B",  "gemma"),
    ("Mistral NeMo 12B", "mistral"),
]
J_W, J_H = 2.3, 0.58
J_Y = S2_Y + 0.67
for i, (name, family) in enumerate(JUDGE_MODELS):
    jx = 1.3 + i * 2.55
    rect = FancyBboxPatch((jx, J_Y), J_W, J_H,
        boxstyle="round,pad=0,rounding_size=0.1",
        facecolor=C_WHITE, edgecolor=C_GREEN, linewidth=1.2, zorder=3)
    ax.add_patch(rect)
    ax.text(jx + J_W / 2, J_Y + J_H / 2 + 0.1,
            f"Judge — {name}",
            ha="center", va="center", fontsize=9,
            fontweight="bold", color=C_GREEN, zorder=4)
    ax.text(jx + J_W / 2, J_Y + J_H / 2 - 0.1,
            f"family: {family}",
            ha="center", va="center", fontsize=8,
            color=C_GRAY, zorder=4, style="italic")
    if i < 2:
        ax.annotate("", xy=(jx + J_W + 0.25, J_Y + J_H / 2),
                    xytext=(jx + J_W + 0.02, J_Y + J_H / 2),
                    arrowprops=dict(arrowstyle="-|>", color=C_GREEN,
                                   lw=1.2, mutation_scale=12), zorder=4)
        ax.text(jx + J_W + 0.135, J_Y + J_H / 2 + 0.12,
                "seq.", fontsize=7.5, color=C_GRAY,
                ha="center", style="italic")

# Majority vote label — bottom strip of Step 2 box
ax.text(5.0, S2_Y + 0.28,
        "Majority vote  (≥ 2/3) → label:  boundary_maintaining  ·  companionship_reinforcing  ·  neutral",
        ha="center", va="center", fontsize=9, color=C_BODY,
        bbox=dict(boxstyle="round,pad=0.25", fc=C_WHITE, ec=C_GREEN,
                  lw=0.8, alpha=0.85))

arrow(5.0, S2_Y, 5.0, 5.65, color=C_GREEN)

# ── 5. judged_responses.jsonl ─────────────────────────────────────────────────
file_box(2.6, 5.0, 4.8, 0.62,
         "judged_responses.jsonl",
         "all responses.jsonl fields  +  system_judge  +  baseline_judge  +  per-judge votes",
         fc=C_FILE_BG, ec=C_GREEN)

arrow(5.0, 5.0, 5.0, 4.45, color=C_GREEN)

# ── 6. STEP 3 box ─────────────────────────────────────────────────────────────
S3_Y = 3.45
box(0.6, S3_Y, 8.8, 0.88, "", fc=C_WARN_LT, ec=C_WARN, lw=2)
step_badge(1.08, S3_Y + 0.63, "3", C_WARN)
ax.text(1.5, S3_Y + 0.63, "Step 3  —  Metrics",
        ha="left", va="center", fontsize=13,
        fontweight="bold", color=C_WARN, zorder=3)
ax.text(5.0, S3_Y + 0.20,
        "Boundary-maintaining rate  ·  Companionship-reinforcing rate  ·  Per-INTIMA-category breakdown  ·  Routing distribution",
        ha="center", va="center", fontsize=9, color=C_BODY, zorder=3)

# Output files from Step 3
arrow(2.8, S3_Y, 2.0, 2.88, color=C_WARN)
arrow(5.0, S3_Y, 5.0, 2.88, color=C_WARN)
arrow(7.2, S3_Y, 8.0, 2.88, color=C_WARN)

file_box(0.55, 2.22, 2.9, 0.63, "report.md",
         "overall results + per-category table", fc=C_FILE_BG, ec=C_WARN)
file_box(3.55, 2.22, 2.9, 0.63, "summary.csv",
         "per-category BM / CR rates", fc=C_FILE_BG, ec=C_WARN)
file_box(6.55, 2.22, 2.9, 0.63, "routing.csv",
         "per-behaviour-code routing", fc=C_FILE_BG, ec=C_WARN)

# ── 7. Legend ─────────────────────────────────────────────────────────────────
LY = 1.35
ax.plot([0.6, 9.4], [LY + 0.55, LY + 0.55], color=C_BORDER, lw=0.8)
legend_items = [
    (C_HL,    "Step 1 / Inference"),
    (C_GREEN, "Step 2 / Judging"),
    (C_WARN,  "Step 3 / Metrics"),
    (C_GRAY,  "Output file  (dashed border)"),
]
for i, (color, label) in enumerate(legend_items):
    lx = 0.8 + i * 2.28
    patch = FancyBboxPatch((lx, LY + 0.08), 0.22, 0.22,
        boxstyle="round,pad=0,rounding_size=0.04",
        facecolor=color, edgecolor="none", zorder=3)
    ax.add_patch(patch)
    ax.text(lx + 0.3, LY + 0.19, label,
            va="center", fontsize=8.5, color=C_BODY)

# ── Save ──────────────────────────────────────────────────────────────────────
for fmt in ("svg", "png"):
    path = OUT_DIR / f"07_eval_methodology_diagram.{fmt}"
    fig.savefig(path, format=fmt, dpi=200, bbox_inches="tight",
                facecolor=C_WHITE)
    print(f"Saved: {path}")

plt.close(fig)
