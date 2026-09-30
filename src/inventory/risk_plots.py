"""
PROJECT FORESIGHT — Decision Grid Visualization
Generates the 2D Stockout vs. Overstock risk decisioning grid plot.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CHARTS_DIR
from src.utils.logger import get_logger

logger = get_logger("risk_plots")

COLOR_MAP = {
    "Reorder Now": "#E74C3C",        # Coral / Crimson
    "Markdown / Clear": "#8E44AD",   # Violet / Purple
    "Watch / Volatile": "#F39C12",   # Gold / Amber
    "Healthy": "#2ECC71",            # Green / Emerald
}


def plot_risk_decision_grid(
    risk_df: pd.DataFrame,
    save_path: Path = None,
) -> Path:
    """Plots every SKU on a 2D stockout vs overstock risk grid with bubble sizes."""
    if save_path is None:
        CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = CHARTS_DIR / "risk_decision_grid.png"

    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=150)

    # Reference threshold lines
    ax.axvline(0.50, color="#BDC3C7", ls="--", lw=1.2)
    ax.axhline(0.50, color="#BDC3C7", ls="--", lw=1.2)

    # Plot SKUs by quadrant
    for q_name, col in COLOR_MAP.items():
        sub = risk_df[risk_df["risk_quadrant"] == q_name]
        if len(sub) == 0:
            continue

        # Scale bubble size by total financial exposure (clamped for visual aesthetics)
        exposure = sub["lost_margin_risk"] + sub["excess_capital_locked"]
        sizes = np.clip(np.sqrt(exposure) * 0.8, 35, 450)

        ax.scatter(
            sub["overstock_risk_score"],
            sub["stockout_risk_score"],
            s=sizes,
            color=col,
            alpha=0.68,
            edgecolors="white",
            linewidth=1.2,
            label=f"{q_name} ({len(sub)} SKUs)",
        )

    # Quadrant background labels
    ax.text(
        0.05, 0.95, "REORDER NOW\n(High Stockout, Low Overstock)",
        fontsize=10, fontweight="bold", color="#C0392B", va="top"
    )
    ax.text(
        0.95, 0.95, "WATCH / VOLATILE\n(High on Both)",
        fontsize=10, fontweight="bold", color="#D35400", va="top", ha="right"
    )
    ax.text(
        0.05, 0.05, "HEALTHY\n(Low on Both)",
        fontsize=10, fontweight="bold", color="#27AE60", va="bottom"
    )
    ax.text(
        0.95, 0.05, "MARKDOWN / CLEAR\n(High Overstock, Low Stockout)",
        fontsize=10, fontweight="bold", color="#7D3C98", va="bottom", ha="right"
    )

    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Overstock Risk Score (0.0 to 1.0)", labelpad=8)
    ax.set_ylabel("Stockout Risk Score (0.0 to 1.0)", labelpad=8)
    ax.set_title("FORESIGHT 2D Inventory Decision Grid (Bubble Size = ₹ Exposure)", pad=14)

    ax.legend(frameon=True, facecolor="#F8F9FA", edgecolor="#E2E8F0", loc="center", markerscale=0.7)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

    logger.info("Saved 2D Risk Decision Grid to %s", save_path)
    return save_path
