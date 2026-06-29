#!/usr/bin/env python3
"""Great Expectations Data Quality Validation Dashboard.

Runs a Great Expectations v1 validation suite on real EEG feature datasets
(epilepsy, depression, etc.) and returns structured pass/fail results
for every expectation, per-dataset.

Real library: great_expectations >=1.0 (pip install great_expectations).
Real data: data/epilepsy/sample/*.csv, data/depression/*.csv, etc.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _find_datasets():
    """Discover CSV feature datasets under data/."""
    datasets = []
    search_dirs = [
        ROOT / "data" / "epilepsy" / "sample",
        ROOT / "data" / "depression",
        ROOT / "data" / "alzheimer",
        ROOT / "data" / "parkinson",
        ROOT / "data" / "autism",
    ]
    for d in search_dirs:
        if d.is_dir():
            for csv in sorted(d.glob("*.csv"))[:3]:
                try:
                    df = pd.read_csv(csv, nrows=5)
                    if len(df.columns) >= 5:
                        datasets.append(csv)
                except Exception:
                    continue
    return datasets


def _disease_from_path(csv_path):
    """Extract disease name from path."""
    parts = csv_path.parts
    for known in ("epilepsy", "depression", "alzheimer", "parkinson", "autism"):
        if known in parts:
            return known
    return csv_path.parent.name


def _build_expectations(columns):
    """Build a list of GE v1 expectation objects for EEG feature data."""
    import great_expectations as gx

    expectations = []

    # Table-level
    expectations.append(
        gx.expectations.ExpectTableRowCountToBeBetween(min_value=10, max_value=500000)
    )
    expectations.append(
        gx.expectations.ExpectTableColumnCountToBeBetween(min_value=5, max_value=200)
    )

    numeric_cols = [c for c in columns
                    if c not in ("label", "subject_id", "class", "disease",
                                 "patient_id", "diagnosis", "group", "file")]

    # Non-null checks
    for col in numeric_cols[:25]:
        expectations.append(
            gx.expectations.ExpectColumnValuesToNotBeNull(column=col, mostly=0.5)
        )

    # Range checks for known EEG features
    range_checks = {
        "mean": (-100, 100),
        "std": (0, 1000),
        "spectral_entropy": (0, 20),
        "sample_entropy": (-1, 10),
        "hurst_exponent": (-0.5, 2.0),
        "dfa_alpha": (-5, 10),
        "hjorth_mobility": (0, 1000),
        "kurtosis": (-50, 500),
        "skewness": (-50, 50),
        "delta_power": (0, None),
        "theta_power": (0, None),
        "alpha_power": (0, None),
        "beta_power": (0, None),
        "gamma_power": (0, None),
        "total_power": (0, None),
    }
    for col, (lo, hi) in range_checks.items():
        if col in columns:
            kwargs = {"column": col, "mostly": 0.9}
            if lo is not None:
                kwargs["min_value"] = lo
            if hi is not None:
                kwargs["max_value"] = hi
            expectations.append(
                gx.expectations.ExpectColumnValuesToBeBetween(**kwargs)
            )

    # Label value set check
    if "label" in columns:
        expectations.append(
            gx.expectations.ExpectColumnValuesToBeInSet(
                column="label", value_set=[0, 1, 2, 3, 4, 5], mostly=0.95
            )
        )

    # Subject ID non-null
    if "subject_id" in columns:
        expectations.append(
            gx.expectations.ExpectColumnValuesToNotBeNull(column="subject_id", mostly=0.95)
        )

    return expectations


def _fmt_result(r):
    """Format a single expectation validation result."""
    result_dict = r.result or {}
    observed = result_dict.get("observed_value")
    if observed is None:
        observed = result_dict.get("unexpected_count")

    desc = r.expectation_config.type
    col = r.expectation_config.kwargs.get("column", "")
    if col:
        desc = f"{desc} [{col}]"

    return {
        "expectation_type": r.expectation_config.type,
        "column": col,
        "description": desc,
        "success": bool(r.success),
        "observed_value": observed,
        "element_count": result_dict.get("element_count"),
        "unexpected_count": result_dict.get("unexpected_count"),
        "unexpected_percent": result_dict.get("unexpected_percent"),
        "missing_count": result_dict.get("missing_count"),
        "missing_percent": result_dict.get("missing_percent"),
    }


def great_expectations_report():
    """Run the full GE validation and return structured JSON."""
    import great_expectations as gx

    datasets = _find_datasets()
    if not datasets:
        return {"available": False, "error": "No CSV datasets found under data/"}

    per_dataset = []
    total_pass = 0
    total_fail = 0
    total_expectations = 0

    for idx, csv_path in enumerate(datasets):
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            per_dataset.append({
                "file": csv_path.name,
                "disease": _disease_from_path(csv_path),
                "error": str(e),
                "results": [],
            })
            continue

        disease = _disease_from_path(csv_path)

        # Fresh context per dataset to avoid name collisions
        ctx = gx.get_context()

        # Build suite
        suite_name = f"eeg_suite_{idx}"
        suite = ctx.suites.add(gx.ExpectationSuite(name=suite_name))
        for exp in _build_expectations(df.columns.tolist()):
            suite.add_expectation(exp)

        # Create datasource + batch
        ds = ctx.data_sources.add_pandas(f"ds_{idx}")
        asset = ds.add_dataframe_asset(f"asset_{idx}")
        batch_def = asset.add_batch_definition_whole_dataframe(f"batch_{idx}")

        # Validation definition + run
        vd = ctx.validation_definitions.add(gx.ValidationDefinition(
            name=f"vd_{idx}", data=batch_def, suite=suite
        ))
        vr = vd.run(batch_parameters={"dataframe": df})

        results = [_fmt_result(r) for r in vr.results]
        passed = sum(1 for r in results if r["success"])
        failed = len(results) - passed
        total_pass += passed
        total_fail += failed
        total_expectations += len(results)

        # Column stats summary
        numeric_df = df.select_dtypes(include="number")
        col_stats = {}
        for col in numeric_df.columns[:15]:
            s = numeric_df[col]
            col_stats[col] = {
                "mean": round(float(s.mean()), 4) if not s.isna().all() else None,
                "std": round(float(s.std()), 4) if not s.isna().all() else None,
                "null_pct": round(float(s.isna().mean() * 100), 2),
                "min": round(float(s.min()), 4) if not s.isna().all() else None,
                "max": round(float(s.max()), 4) if not s.isna().all() else None,
            }

        per_dataset.append({
            "file": csv_path.name,
            "disease": disease,
            "rows": len(df),
            "columns": len(df.columns),
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / max(len(results), 1) * 100, 1),
            "column_stats": col_stats,
            "results": results,
        })

    # Aggregate by expectation type
    type_summary = {}
    for ds in per_dataset:
        for r in ds.get("results", []):
            etype = r["expectation_type"]
            if etype not in type_summary:
                type_summary[etype] = {"passed": 0, "failed": 0}
            if r["success"]:
                type_summary[etype]["passed"] += 1
            else:
                type_summary[etype]["failed"] += 1

    return {
        "available": True,
        "library": "great_expectations",
        "version": gx.__version__,
        "datasets_validated": len(per_dataset),
        "total_expectations": total_expectations,
        "total_passed": total_pass,
        "total_failed": total_fail,
        "overall_pass_rate": round(total_pass / max(total_expectations, 1) * 100, 1),
        "expectation_type_summary": type_summary,
        "per_dataset": per_dataset,
    }


if __name__ == "__main__":
    report = great_expectations_report()
    out = ROOT / "jobs" / "reports" / "great_expectations_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
