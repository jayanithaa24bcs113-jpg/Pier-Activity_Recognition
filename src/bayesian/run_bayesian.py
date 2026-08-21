"""
Run Bayesian

Purpose: Run Bayesian smoothing over rule engine outputs.
Author: Student & Antigravity
Date: 2026-07-02
Version: 1.0.0

Run Bayesian smoothing over rule engine outputs.

This script reads `outputs/rules/rule_results.csv`, applies the
`BayesianFilter`, and writes CSV/JSON outputs to
`outputs/bayesian/`. The top-level script is intentionally simple and
meant for command-line runs. Add CLI argument parsing if needed.
"""

import os
import csv
import json

from src.bayesian.bayesian_filter import BayesianFilter

INPUT_FILE = "outputs/rules/rule_results.csv"

OUTPUT_FOLDER = "outputs/bayesian"

OUTPUT_CSV = os.path.join(OUTPUT_FOLDER, "bayesian_results.csv")

OUTPUT_JSON = os.path.join(OUTPUT_FOLDER, "bayesian_results.json")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

filter = BayesianFilter(alpha=0.4)

results_csv = []
results_json = []

with open(INPUT_FILE, "r") as f:

    reader = csv.DictReader(f)

    rows = list(reader)

print(f"\nFound {len(rows)} images\n")

for i, row in enumerate(rows, start=1):

    image = row["image"]

    score = float(row["score"])

    smoothed = filter.update({

        "Pier Stem Reinforcement": score

    })

    final_score = round(

        smoothed["Pier Stem Reinforcement"],

        3

    )

    print(

        f"[{i}/{len(rows)}] "

        f"{image} "

        f"Raw={score:.3f} "

        f"Smoothed={final_score:.3f}"

    )

    results_csv.append({

        "image": image,

        "raw_score": score,

        "smoothed_score": final_score

    })

    results_json.append({

        "image": image,

        "raw_score": score,

        "smoothed_score": final_score

    })

with open(OUTPUT_CSV, "w", newline="") as f:

    writer = csv.DictWriter(

        f,

        fieldnames=[

            "image",

            "raw_score",

            "smoothed_score"

        ]

    )

    writer.writeheader()

    writer.writerows(results_csv)

with open(OUTPUT_JSON, "w") as f:

    json.dump(results_json, f, indent=4)

print("\n==============================")

print("Bayesian Filter Completed")

print("==============================")

print(f"CSV  : {OUTPUT_CSV}")

print(f"JSON : {OUTPUT_JSON}")