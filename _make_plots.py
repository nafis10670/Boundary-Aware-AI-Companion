"""Generate seaborn plots from evaluation results and save as PNG images."""

import pathlib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

RESULTS_BASE = pathlib.Path("data/results")
OUT_DIR = pathlib.Path("plots")
OUT_DIR.mkdir(exist_ok=True)

# ── Model registry ─────────────────────────────────────────────────────────────
RUNS = {
    "LLaMA 3.1 8B":    "llama3_1_8b-instruct-q4_K_M",
    "Qwen 2.5 72B":    "qwen2_5_72b-instruct-q4_K_M",
    "Gemma 3 27B":     "gemma3_27b",
    "Mistral NeMo 12B":"mistral-nemo_12b",
}

MODEL_ORDER = list(RUNS.keys())

# ── Light theme ────────────────────────────────────────────────────────────────
BG      = "#FFFFFF"
CARD    = "#F5F7FA"
HL      = "#0077AA"
GREEN   = "#00996A"
WARN    = "#E07000"
RED     = "#CC2244"
GRAY    = "#666677"
WHITE   = "#FFFFFF"
LIGHT   = "#333344"

PALETTE_SYS  = HL
PALETTE_BASE = WARN

# sns.set_theme(style="whitegrid", rc={
#     "figure.facecolor":  BG,
#     "axes.facecolor":    CARD,
#     "axes.edgecolor":    "#CCCCDD",
#     "axes.labelcolor":   LIGHT,
#     "xtick.color":       LIGHT,
#     "ytick.color":       LIGHT,
#     "text.color":        LIGHT,
#     "grid.color":        "#DDDDEE",
#     "grid.linestyle":    "--",
#     "grid.linewidth":    0.6,
#     "legend.facecolor":  WHITE,
#     "legend.edgecolor":  "#CCCCDD",
#     "legend.labelcolor": LIGHT,
#     "font.family":       "DejaVu Sans",
# })

sns.set_theme()

TITLE_KW  = dict(color="#111122", fontsize=14, fontweight="bold", pad=12)
LABEL_KW  = dict(color=LIGHT, fontsize=11)
TICK_KW   = dict(labelcolor=LIGHT, labelsize=10)


# ── Load helpers ───────────────────────────────────────────────────────────────

def load_summary() -> pd.DataFrame:
    rows = []
    for label, run_id in RUNS.items():
        df = pd.read_csv(RESULTS_BASE / run_id / "summary.csv")
        df.insert(0, "model", label)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def load_routing_overall() -> pd.DataFrame:
    rows = []
    for label, run_id in RUNS.items():
        df = pd.read_csv(RESULTS_BASE / run_id / "routing.csv")
        # aggregate across all behaviour codes
        total = df["n"].sum()
        boundary_n = (df["boundary_route_rate"] * df["n"]).sum()
        high_n     = (df["high_risk_rate"]       * df["n"]).sum()
        medium_n   = (df["medium_risk_rate"]      * df["n"]).sum()
        low_n      = (df["low_risk_rate"]         * df["n"]).sum()
        rows.append({
            "model":        label,
            "boundary_pct": boundary_n / total * 100,
            "interaction_pct": (total - boundary_n) / total * 100,
            "high_pct":     high_n   / total * 100,
            "medium_pct":   medium_n / total * 100,
            "low_pct":      low_n    / total * 100,
        })
    return pd.DataFrame(rows)


def save(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 1 — Overall BM Rate: System vs Baseline (grouped bar)
# ══════════════════════════════════════════════════════════════════════════════

def plot_overall_bm():
    summary = load_summary()
    # aggregate to overall per model (weighted by n)
    records = []
    for model in MODEL_ORDER:
        sub = summary[summary["model"] == model]
        total = sub["n"].sum()
        sys_bm  = (sub["system_boundary_maintaining_rate"]  * sub["n"]).sum() / total * 100
        base_bm = (sub["baseline_boundary_maintaining_rate"] * sub["n"]).sum() / total * 100
        records.append({"Model": model, "Rate (%)": sys_bm,  "Type": "System"})
        records.append({"Model": model, "Rate (%)": base_bm, "Type": "Baseline"})
    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(
        data=df, x="Model", y="Rate (%)", hue="Type",
        palette="muted",
        order=MODEL_ORDER, ax=ax, width=0.55, alpha=0.85
    )
    # annotate bars
    for bar in ax.patches:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.4,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9.5, color=LIGHT,
            )
    ax.set_title("Boundary-Maintaining Rate: System vs Baseline", **TITLE_KW)
    ax.set_xlabel("Backbone Model", **LABEL_KW, labelpad=15)
    ax.set_ylabel("Boundary-Maintaining Rate (%)", **LABEL_KW)
    ax.set_ylim(50, 109)
    ax.tick_params(**TICK_KW)
    ax.legend(title="", fontsize=11, framealpha=0.8, loc="upper right",)
    fig.tight_layout()
    # fig.subplots_adjust(right=0.85)
    save(fig, "01_overall_bm_rate")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 2 — Overall CR Rate: System vs Baseline (grouped bar)
# ══════════════════════════════════════════════════════════════════════════════

def plot_overall_cr():
    summary = load_summary()
    records = []
    for model in MODEL_ORDER:
        sub = summary[summary["model"] == model]
        total = sub["n"].sum()
        sys_cr  = (sub["system_companionship_reinforcing_rate"]  * sub["n"]).sum() / total * 100
        base_cr = (sub["baseline_companionship_reinforcing_rate"] * sub["n"]).sum() / total * 100
        records.append({"Model": model, "Rate (%)": sys_cr,  "Type": "System"})
        records.append({"Model": model, "Rate (%)": base_cr, "Type": "Baseline"})
    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(
        data=df, x="Model", y="Rate (%)", hue="Type",
        palette={"System": GREEN, "Baseline": RED},
        order=MODEL_ORDER, ax=ax, width=0.55,
    )
    for bar in ax.patches:
        h = bar.get_height()
        if h > 0.05:
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.3,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9.5, color=LIGHT,
            )
    ax.set_title("Companionship-Reinforcing Rate: System vs Baseline\n(lower is better)", **TITLE_KW)
    ax.set_xlabel("Backbone Model", **LABEL_KW)
    ax.set_ylabel("Companionship-Reinforcing Rate (%)", **LABEL_KW)
    ax.tick_params(**TICK_KW)
    ax.legend(title="", fontsize=11, framealpha=0.8)
    fig.tight_layout()
    save(fig, "02_overall_cr_rate")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 3 — Per-Category System BM Rate (grouped bar, all 4 models)
# ══════════════════════════════════════════════════════════════════════════════

def plot_category_bm():
    summary = load_summary()
    CAT_ORDER = [
        "Assistant Traits",
        "User Vulnerabilities",
        "Relationship & Intimacy",
        "Emotional Investment",
        "Other",
    ]
    CAT_SHORT = {
        "Assistant Traits": "Assistant\nTraits",
        "User Vulnerabilities": "User\nVulnerabilities",
        "Relationship & Intimacy": "Relationship\n& Intimacy",
        "Emotional Investment": "Emotional\nInvestment",
        "Other": "Other",
    }
    MODEL_COLORS = [HL, WARN, GREEN, RED]

    # Bar geometry: sys + base bars side-by-side per model, grouped by category
    bar_w     = 0.09
    intra_gap = 0.01   # gap between sys and base bars of the same model
    inter_gap = 0.04   # gap between different model pairs
    pair_w    = 2 * bar_w + intra_gap
    step      = pair_w + inter_gap
    n         = len(MODEL_ORDER)
    total_w   = n * pair_w + (n - 1) * inter_gap
    x0        = -total_w / 2

    offsets_sys  = [x0 + i * step + bar_w / 2                     for i in range(n)]
    offsets_base = [x0 + i * step + bar_w + intra_gap + bar_w / 2 for i in range(n)]

    fig, ax = plt.subplots(figsize=(14, 5.5))

    for ci, cat in enumerate(CAT_ORDER):
        for mi, model in enumerate(MODEL_ORDER):
            color = MODEL_COLORS[mi]
            sub = summary[(summary["model"] == model) & (summary["category"] == cat)]
            if sub.empty:
                continue
            row      = sub.iloc[0]
            sys_val  = row["system_boundary_maintaining_rate"]  * 100
            base_val = row["baseline_boundary_maintaining_rate"] * 100

            ax.bar(ci + offsets_sys[mi],  sys_val,  width=bar_w,
                   color=color, edgecolor="white", linewidth=0.4, zorder=3)
            ax.bar(ci + offsets_base[mi], base_val, width=bar_w,
                   color=color, edgecolor="white", linewidth=0.4,
                   hatch="//", alpha=0.55, zorder=3)

    # Legend: model colours + sys/base style
    color_handles = [mpatches.Patch(facecolor=MODEL_COLORS[i], label=m)
                     for i, m in enumerate(MODEL_ORDER)]
    style_handles = [
        mpatches.Patch(facecolor=GRAY, label="System (solid)"),
        mpatches.Patch(facecolor=GRAY, hatch="//", alpha=0.55, label="Baseline (hatched)"),
    ]
    ax.legend(
        handles=color_handles + style_handles,
        loc="upper left", bbox_to_anchor=(1.01, 1.0),
        fontsize=9, framealpha=0.85, borderaxespad=0,
    )

    ax.set_xticks(list(range(len(CAT_ORDER))))
    ax.set_xticklabels([CAT_SHORT[c] for c in CAT_ORDER], color=LIGHT, fontsize=10)
    ax.set_ylim(0, 112)
    ax.axhline(100, color=GRAY, linewidth=0.7, linestyle=":")
    ax.set_xlabel("INTIMA Category", **LABEL_KW, labelpad=12)
    ax.set_ylabel("Boundary-Maintaining Rate (%)", **LABEL_KW)
    ax.set_title("Per-Category Boundary-Maintaining Rate", **TITLE_KW)
    ax.tick_params(**TICK_KW)

    fig.tight_layout()
    fig.subplots_adjust(right=0.83)
    save(fig, "03_category_bm_rate")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 4 — Routing Distribution (stacked bar: boundary / interaction)
# ══════════════════════════════════════════════════════════════════════════════

def plot_routing():
    df = load_routing_overall()
    df = df.set_index("model").loc[MODEL_ORDER].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- left: route (boundary vs interaction) ---
    ax = axes[0]
    x = range(len(MODEL_ORDER))
    bars_b = ax.bar(x, df["boundary_pct"],     color=HL,    label="Boundary",    width=0.5)
    bars_i = ax.bar(x, df["interaction_pct"], color=GREEN, label="Interaction",
                    bottom=df["boundary_pct"], width=0.5)
    for bar, val in zip(bars_b, df["boundary_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val / 2,
                f"{val:.1f}%", ha="center", va="center", fontsize=11, color=WHITE, fontweight="bold")
    for bar, b_val, i_val in zip(bars_i, df["boundary_pct"], df["interaction_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, b_val + i_val / 2,
                f"{i_val:.1f}%", ha="center", va="center", fontsize=11, color=WHITE, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(MODEL_ORDER, rotation=15, ha="right", color=LIGHT, fontsize=10)
    ax.set_ylabel("Percentage of Conversations (%)", **LABEL_KW)
    ax.set_ylim(0, 110)
    ax.set_title("Route Distribution\n(Boundary vs Interaction)", **TITLE_KW)
    ax.legend(fontsize=10, framealpha=0.8)

    # --- right: risk level (high / medium / low) ---
    ax = axes[1]
    bars_h = ax.bar(x, df["high_pct"],   color=RED,       label="High",   width=0.5)
    bars_m = ax.bar(x, df["medium_pct"], color="#FFDD55",  label="Medium",
                    bottom=df["high_pct"], width=0.5)
    bars_l = ax.bar(x, df["low_pct"],    color=GREEN,      label="Low",
                    bottom=df["high_pct"] + df["medium_pct"], width=0.5)
    for bars, vals, bases, color in [
        (bars_h, df["high_pct"],   [0]*4,                             RED),
        (bars_m, df["medium_pct"], df["high_pct"],                    BG),
        (bars_l, df["low_pct"],    df["high_pct"] + df["medium_pct"], BG),
    ]:
        for bar, val, base in zip(bars, vals, bases):
            if val > 3:
                ax.text(bar.get_x() + bar.get_width() / 2, base + val / 2,
                        f"{val:.1f}%", ha="center", va="center", fontsize=10,
                        color=WHITE, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(MODEL_ORDER, rotation=15, ha="right", color=LIGHT, fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_title("Risk Level Distribution\n(High / Medium / Low)", **TITLE_KW)
    ax.legend(fontsize=10, framealpha=0.8)

    fig.suptitle("Risk Monitor Routing Decisions", color=WHITE, fontsize=15, fontweight="bold")
    fig.tight_layout()
    save(fig, "04_routing_distribution")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 5 — Heatmap: System BM Rate improvement over baseline (Δ pp)
# ══════════════════════════════════════════════════════════════════════════════

def plot_improvement_heatmap():
    summary = load_summary()
    CAT_ORDER = [
        "Assistant Traits",
        "User Vulnerabilities",
        "Relationship & Intimacy",
        "Emotional Investment",
        "Other",
    ]
    # Build delta matrix: rows = categories, cols = models
    matrix = pd.DataFrame(index=CAT_ORDER, columns=MODEL_ORDER, dtype=float)
    for model in MODEL_ORDER:
        sub = summary[summary["model"] == model]
        for _, row in sub.iterrows():
            cat = row["category"]
            if cat in CAT_ORDER:
                delta = (row["system_boundary_maintaining_rate"]
                         - row["baseline_boundary_maintaining_rate"]) * 100
                matrix.loc[cat, model] = delta

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        matrix.astype(float),
        ax=ax,
        annot=True,
        fmt=".1f",
        cmap=sns.diverging_palette(10, 150, s=80, l=45, as_cmap=True),
        center=0,
        linewidths=0.5,
        linecolor=BG,
        annot_kws={"size": 12, "color": LIGHT, "fontweight": "bold"},
        cbar_kws={"label": "Δ BM Rate (pp)", "shrink": 0.8},
    )
    ax.set_title("System Improvement over Baseline\n(Boundary-Maintaining Rate, percentage points)",
                 **TITLE_KW)
    ax.set_xlabel("Backbone Model", **LABEL_KW)
    ax.set_ylabel("INTIMA Category", **LABEL_KW)
    ax.tick_params(**TICK_KW)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    # style colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color(LIGHT)
    cbar.ax.tick_params(colors=LIGHT)
    fig.tight_layout()
    save(fig, "05_improvement_heatmap")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 6 — Per-behaviour-code routing (Mistral NeMo — most varied)
# ══════════════════════════════════════════════════════════════════════════════

def plot_behaviour_routing():
    """Horizontal bar chart of boundary route rate per behaviour code for each model."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 18))
    axes_flat = axes.flatten()

    for ax, (model, run_id) in zip(axes_flat, RUNS.items()):
        df = pd.read_csv(RESULTS_BASE / run_id / "routing.csv")
        df = df.sort_values("boundary_route_rate", ascending=True)
        colors = [GREEN if v >= 1.0 else HL if v >= 0.8 else WARN if v >= 0.5 else RED
                  for v in df["boundary_route_rate"]]
        ax.barh(df["behavior_code"], df["boundary_route_rate"] * 100, color=colors, height=0.7)
        ax.axvline(80, color=WARN,  linewidth=1.0, linestyle="--", alpha=0.7)
        ax.axvline(100, color=GRAY, linewidth=0.8, linestyle=":",  alpha=0.6)
        ax.set_xlim(0, 108)
        ax.set_title(f"{model}", color=WHITE, fontsize=13, fontweight="bold")
        ax.set_xlabel("Boundary Route Rate (%)", **LABEL_KW)
        ax.tick_params(axis="y", labelsize=9, labelcolor=LIGHT)
        ax.tick_params(axis="x", **TICK_KW)
        for spine in ax.spines.values():
            spine.set_edgecolor("#CCCCDD")

    fig.suptitle("Boundary Route Rate per Behaviour Code",
                 color=WHITE, fontsize=16, fontweight="bold", y=1.005)
    fig.tight_layout()
    save(fig, "06_behaviour_routing")


# ══════════════════════════════════════════════════════════════════════════════
# Run all
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    plot_overall_bm()
    plot_overall_cr()
    plot_category_bm()
    plot_routing()
    plot_improvement_heatmap()
    plot_behaviour_routing()
    print(f"\nAll plots saved to: {OUT_DIR.resolve()}/")
