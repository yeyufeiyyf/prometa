#!/usr/bin/env python3
"""Verify the ProMeta-only release against the numerical response-letter tables.

The response letter reports values rounded to three decimals.  This audit reads
only the public ``results/released`` bundle, normalizes its table-specific
schemas, and compares the rounded mean, sample SD, and query-row counts with
``response_letter_expected.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "released"
EXPECTED = ROOT / "experiments" / "response_letter_expected.csv"
METRICS = ("AUROC", "AUPRC", "BALACC")


def setting_from_kq(row: pd.Series) -> str:
    q = int(row["q_query"])
    return f"K{int(row['k_shot'])}Q{'all' if q == 0 else q}"


def standard_actual(path: Path, table: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["response_table"] = table
    frame["setting"] = frame.apply(setting_from_kq, axis=1)
    return frame


def s7_actual() -> pd.DataFrame:
    frame = pd.read_csv(RESULTS / "s7_compact_prometa_variants.csv")
    out = pd.DataFrame(
        {
            "response_table": "S7",
            "direction": "average",
            "setting": "four fixed-query settings",
            "method": frame["method"],
            "AUROC_mean": frame["Mean AUROC"],
            "AUPRC_mean": frame["Mean AUPRC"],
            "BALACC_mean": frame["Mean BALACC"],
        }
    )
    return out


def s2_actual() -> pd.DataFrame:
    frame = pd.read_csv(RESULTS / "s2_intersection_summary.csv")
    frame["response_table"] = "S2"
    frame["n_seeds"] = frame["n_splits"]
    return frame


def rare_actual() -> pd.DataFrame:
    frame = pd.read_csv(RESULTS / "s4_s5_rare_e3_metrics.csv")
    frame = frame[frame["scope"].eq("common_query_label_agree")].copy()
    return pd.DataFrame(
        {
            "response_table": "S4-S5",
            "direction": "rare-E3",
            "setting": frame["ligase"],
            "method": frame["method"],
            "mean_n_query_eval": frame["n"],
            "AUROC_mean": frame["AUROC"],
            "AUPRC_mean": frame["AUPRC"],
            "BALACC_mean": frame["BALACC"],
        }
    )


def metric_mean(value: object) -> float:
    return float(str(value).split("+/-", 1)[0].strip())


def s8_actual() -> pd.DataFrame:
    frame = pd.read_csv(RESULTS / "s8_rdkit_3d_summary.csv")
    out = pd.DataFrame(
        {
            "response_table": "S8",
            "direction": frame["direction"],
            "setting": frame["setting"].str.replace("=", "", regex=False).str.replace(",", "", regex=False),
            "method": frame["method"],
            "mean_n_query_eval": frame["mean_n_query_eval"],
        }
    )
    for metric in METRICS:
        out[f"{metric}_mean"] = frame[metric].map(metric_mean)
    return out


def collect_actual() -> pd.DataFrame:
    frames = [
        standard_actual(RESULTS / "s3_bidirectional_summary.csv", "S3"),
        standard_actual(RESULTS / "s9_kq_sensitivity_summary.csv", "S9"),
        standard_actual(RESULTS / "s6_s7_prometa_variants_summary.csv", "S6"),
        s7_actual(),
        s2_actual(),
        rare_actual(),
        s8_actual(),
    ]
    return pd.concat(frames, ignore_index=True, sort=False)


def compare(expected: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    keys = ["response_table", "direction", "setting", "method"]
    merged = expected.merge(actual, on=keys, how="left", suffixes=("_expected", "_actual"), indicator=True)
    rows: list[dict[str, object]] = []
    fields = ["mean_n_query_eval"] + [f"{metric}_{suffix}" for metric in METRICS for suffix in ("mean", "std")]
    for _, item in merged.iterrows():
        for field in fields:
            expected_value = item.get(f"{field}_expected", np.nan)
            if pd.isna(expected_value):
                continue
            actual_value = item.get(f"{field}_actual", np.nan)
            if field == "mean_n_query_eval":
                passed = pd.notna(actual_value) and abs(float(actual_value) - float(expected_value)) < 1e-8
            else:
                passed = pd.notna(actual_value) and round(float(actual_value), 3) == round(float(expected_value), 3)
            rows.append(
                {
                    **{key: item[key] for key in keys},
                    "field": field,
                    "expected": expected_value,
                    "actual": actual_value,
                    "status": "PASS" if passed else "FAIL",
                }
            )
    return pd.DataFrame(rows)


def method_scope_audit() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(RESULTS.glob("*.csv")):
        if path.name in {
            "release_checksums.csv",
            "response_letter_consistency.csv",
            "prometa_only_scope_audit.csv",
        }:
            continue
        frame = pd.read_csv(path, low_memory=False)
        if "method" not in frame.columns:
            continue
        methods = sorted(frame["method"].dropna().astype(str).unique())
        invalid = [method for method in methods if not method.lower().startswith("prometa")]
        rows.append(
            {
                "file": str(path.relative_to(ROOT)),
                "methods": " | ".join(methods),
                "status": "PASS" if not invalid else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def vz185_audit() -> dict[str, object]:
    frame = pd.read_csv(RESULTS / "vz185_prometa_summary.csv")
    row = frame.loc[frame["cohort"].eq("legacy_excel_first12")].iloc[0]
    passed = int(row["n"]) == 12 and round(float(row["accuracy"]), 6) == round(11 / 12, 6)
    return {
        "response_table": "VZ185",
        "direction": "VHL case study",
        "setting": "legacy_excel_first12",
        "method": str(row["method"]),
        "field": "accuracy",
        "expected": 11 / 12,
        "actual": float(row["accuracy"]),
        "status": "PASS" if passed else "FAIL",
    }


def four_family_audit() -> pd.DataFrame:
    frame = pd.read_csv(RESULTS / "four_family_main_checkpoint_summary.csv")
    expected = {
        "CDK6 (n=14)": (14, 11),
        "NAMPT (n=12)": (12, 6),
        "BTK (n=10)": (10, 6),
        "BCL-xL (n=6)": (6, 6),
    }
    rows: list[dict[str, object]] = []
    for family, (expected_n, expected_correct) in expected.items():
        match = frame.loc[frame["family_short"].eq(family)]
        passed = (
            len(match) == 1
            and int(match.iloc[0]["n_compounds"]) == expected_n
            and int(match.iloc[0]["n_correct"]) == expected_correct
            and str(match.iloc[0]["checkpoint"]) == "main full-data checkpoint used in S3"
        )
        actual = int(match.iloc[0]["n_correct"]) if len(match) == 1 else float("nan")
        direction = str(match.iloc[0]["family_direction"]) if len(match) == 1 else ""
        rows.append(
            {
                "response_table": "R9",
                "direction": direction,
                "setting": family,
                "method": "ProMeta",
                "field": "n_correct",
                "expected": expected_correct,
                "actual": actual,
                "status": "PASS" if passed else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RESULTS / "response_letter_consistency.csv")
    parser.add_argument("--method-output", type=Path, default=RESULTS / "prometa_only_scope_audit.csv")
    args = parser.parse_args()

    expected = pd.read_csv(EXPECTED)
    actual = collect_actual()
    audit = compare(expected, actual)
    audit = pd.concat(
        [audit, pd.DataFrame([vz185_audit()]), four_family_audit()],
        ignore_index=True,
        sort=False,
    )
    scope = method_scope_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output, index=False)
    scope.to_csv(args.method_output, index=False)
    failures = int(audit["status"].ne("PASS").sum() + scope["status"].ne("PASS").sum())
    print(f"table checks: {len(audit)}; scope checks: {len(scope)}; failures: {failures}")
    if failures:
        print(audit[audit["status"].ne("PASS")].to_string(index=False))
        print(scope[scope["status"].ne("PASS")].to_string(index=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
