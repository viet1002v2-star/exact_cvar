#!/usr/bin/env python3
"""Summarize results_affine_certified_alns.csv."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_rows(path):
    with Path(path).open(
        newline="",
        encoding="utf-8",
    ) as handle:
        return list(csv.DictReader(handle))


def number(row, key):
    return float(row[key])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="results_affine_certified_alns.csv",
    )
    args = parser.parse_args()

    rows = read_rows(args.csv)
    if not rows:
        print("No rows found.")
        return

    gate = np.asarray(
        [number(row, "gate_speedup") for row in rows]
    )
    solver = np.asarray(
        [number(row, "solver_speedup") for row in rows]
    )
    prep = np.asarray(
        [
            number(
                row,
                "solver_speedup_with_prep",
            )
            for row in rows
        ]
    )
    coverage = np.asarray(
        [number(row, "coverage") for row in rows]
    )

    print("CERTIFIED CVaR INSIDE DETHLOFF ALNS")
    print("=" * 96)
    print(
        f"cells={len(rows)} "
        f"instances={len(set(row['instance'] for row in rows))} "
        f"median gate={np.median(gate):.2f}x "
        f"min gate={gate.min():.2f}x "
        f"median solver={np.median(solver):.2f}x "
        f"min solver={solver.min():.2f}x "
        f"median with-prep={np.median(prep):.2f}x "
        f"median coverage={np.median(coverage):.3f}"
    )

    if np.std(coverage) > 1e-12:
        print(
            "corr(coverage,solver-speed)="
            f"{np.corrcoef(coverage, solver)[0, 1]:.3f}"
        )

    groups = defaultdict(list)
    for row in rows:
        groups[
            (
                row["profile"],
                row["policy"],
                row["direction"],
            )
        ].append(row)

    print()
    print(
        f"{'profile/policy/direction':<36} "
        f"{'n':>4} {'gate':>8} {'solver':>8} "
        f"{'min':>8} {'prep':>8} {'cover':>8}"
    )

    for key, bucket in sorted(groups.items()):
        g = np.asarray(
            [number(row, "gate_speedup") for row in bucket]
        )
        s = np.asarray(
            [number(row, "solver_speedup") for row in bucket]
        )
        p = np.asarray(
            [
                number(
                    row,
                    "solver_speedup_with_prep",
                )
                for row in bucket
            ]
        )
        c = np.asarray(
            [number(row, "coverage") for row in bucket]
        )

        print(
            f"{'/'.join(key):<36} "
            f"{len(bucket):>4d} "
            f"{np.median(g):>8.2f} "
            f"{np.median(s):>8.2f} "
            f"{s.min():>8.2f} "
            f"{np.median(p):>8.2f} "
            f"{np.median(c):>8.3f}"
        )


if __name__ == "__main__":
    main()
