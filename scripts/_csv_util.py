"""Tiny shared helper for the diagnostic scripts: merge rows into an
existing CSV instead of truncating it, so incremental re-runs (different
--configs/--folds subsets) accumulate results rather than destroying them."""

import csv
from pathlib import Path


def merge_csv_rows(csv_path, new_rows, key):
    """Merge `new_rows` into the CSV at `csv_path`, keyed on `key` columns.

    Existing rows with the same key are replaced by the new ones; all other
    existing rows are preserved. Returns the merged row list (existing-first,
    then genuinely new rows, in stable order).
    """
    csv_path = Path(csv_path)
    merged = {}
    if csv_path.exists():
        with open(csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                merged[tuple(row.get(k, "") for k in key)] = row
    for row in new_rows:
        merged[tuple(str(row.get(k, "")) for k in key)] = row
    return list(merged.values())


def write_csv_rows(csv_path, rows):
    """Write rows with the union of all fieldnames (order-preserving)."""
    fieldnames = list(dict.fromkeys(k for r in rows for k in r.keys()))
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        w.writeheader()
        w.writerows(rows)
