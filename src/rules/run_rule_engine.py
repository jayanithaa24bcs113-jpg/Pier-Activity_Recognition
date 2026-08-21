"""
Run Rule Engine

Purpose: Run the rule engine against extracted attributes CSV.
Author: Student & Antigravity
Date: 2026-07-02
Version: 1.0.0

Run the rule engine against extracted attributes CSV.

Reads `outputs/attributes/attributes.csv`, evaluates configured rules,
and writes CSV/JSON summaries to `outputs/rules/`.
"""

import os
import csv
import json

from src.rules.rule_engine import RuleEngine

# -----------------------------------------
# Configuration
# -----------------------------------------

INPUT_CSV = "outputs/attributes/attributes.csv"

OUTPUT_FOLDER = "outputs/rules"

OUTPUT_CSV = os.path.join(OUTPUT_FOLDER, "rule_results.csv")

OUTPUT_JSON = os.path.join(OUTPUT_FOLDER, "rule_results.json")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------------------------
# Load Rule Engine
# -----------------------------------------

print("\nLoading Rule Engine...")

rule_engine = RuleEngine()

results_csv = []
results_json = []

# -----------------------------------------
# Read Attributes CSV
# -----------------------------------------

with open(INPUT_CSV, "r") as f:

    reader = csv.DictReader(f)

    rows = list(reader)

print(f"Found {len(rows)} images.\n")

# -----------------------------------------
# Evaluate Rules
# -----------------------------------------

for i, row in enumerate(rows, start=1):

    image_name = row["image"]

    # convert csv strings to numbers
    attributes = {}

    for key, value in row.items():

        if key == "image":
            continue

        try:

            if "." in value:
                attributes[key] = float(value)

            else:
                attributes[key] = int(value)

        except:

            attributes[key] = value

    scores = rule_engine.apply_rules(attributes)

    total_score = round(sum(scores.values()), 3)

    fired_rules = ", ".join(scores.keys())

    print(f"[{i}/{len(rows)}] {image_name}")

    print(f"Rules Fired : {fired_rules}")

    print(f"Total Score : {total_score}")

    print("---------------------------------------")

    results_csv.append({

        "image": image_name,

        "score": total_score,

        "rules": fired_rules

    })

    results_json.append({

        "image": image_name,

        "attributes": attributes,

        "rule_scores": scores,

        "total_score": total_score

    })

# -----------------------------------------
# Save CSV
# -----------------------------------------

with open(OUTPUT_CSV, "w", newline="") as f:

    writer = csv.DictWriter(

        f,

        fieldnames=["image", "score", "rules"]

    )

    writer.writeheader()

    writer.writerows(results_csv)

# -----------------------------------------
# Save JSON
# -----------------------------------------

with open(OUTPUT_JSON, "w") as f:

    json.dump(results_json, f, indent=4)

print("\n===================================")

print("Rule Engine Completed")

print("===================================")

print(f"Images : {len(rows)}")

print(f"CSV    : {OUTPUT_CSV}")

print(f"JSON   : {OUTPUT_JSON}")