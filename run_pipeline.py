import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Ensure project root is in sys.path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run_command(cmd_args, desc):
    """Executes a pipeline step and checks the exit code."""
    print(f"\n{'='*70}")
    print(f"[PIPELINE STEP] {desc}")
    print(f"{'='*70}")
    res = subprocess.run(cmd_args, cwd=str(PROJECT_ROOT))
    if res.returncode != 0:
        print(f"\n[ERROR] Step '{desc}' failed with exit code {res.returncode}")
        sys.exit(res.returncode)
    print(f"[COMPLETED] {desc} succeeded.\n")


def main():
    parser = argparse.ArgumentParser(
        description="FORESIGHT Master Pipeline Runner: Runs full end-to-end data, ML, and risk engine workflows."
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        help="Specify which phase to run: '2', '3', '4', '5', or 'all' (default: all)",
    )
    parser.add_argument(
        "--skip-data-gen",
        action="store_true",
        help="Skip synthetic data generation (Phase 2) and reuse existing datasets.",
    )
    args = parser.parse_args()

    python_exe = sys.executable

    print("\n" + "#" * 70)
    print("  PROJECT FORESIGHT — MASTER PIPELINE EXECUTION")
    print("  NorthBay Living Demand & Inventory Intelligence Platform")
    print("#" * 70)

    # Phase 2
    if args.phase in ["2", "all"] and not args.skip_data_gen:
        run_command(
            [python_exe, "scripts/run_phase2.py"],
            "Phase 2: Synthetic Data Generation & Relational Integrity Validation",
        )
    elif args.skip_data_gen:
        print("\n[INFO] Skipping Phase 2 data generation as requested (--skip-data-gen).")

    # Phase 3
    if args.phase in ["3", "all"]:
        run_command(
            [python_exe, "scripts/run_phase3.py"],
            "Phase 3: Data Quality Profiling, Weekly Aggregation & EDA Analysis",
        )

    # Phase 4
    if args.phase in ["4", "all"]:
        run_command(
            [python_exe, "scripts/run_phase4.py"],
            "Phase 4: Time-Series Feature Engineering, Model Training & Backtesting",
        )

    # Phase 5
    if args.phase in ["5", "all"]:
        run_command(
            [python_exe, "scripts/run_phase5.py"],
            "Phase 5: 2D Inventory Risk Engine, Decision Grid & Recommendations",
        )
        run_command(
            [python_exe, "scripts/make_priority_graphic.py"],
            "Phase 5 Add-on: Priority Replenishment & Liquidation Graphics",
        )

    print("\n" + "#" * 70)
    print("  FORESIGHT PIPELINE SUCCESSFULLY COMPLETED")
    print("  All data, models, forecasts, and risk scores are synchronized.")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
