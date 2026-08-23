"""Narrow, deterministic verbalized-confidence repair (login-safe, CPU-only).

Scope (LOCKED with Jerry): repair ONLY the known EOS/control-token decoding
mismatch that caused otherwise-valid integer confidence fields to be rejected by
the fixed parser. This is an *implementation/parser repair* of a signal that was
part of the intended experimental design (verbalized confidence), NOT a new
scientific analysis and NOT a broadening of the parser.

Repair rule, applied only to the already-extracted ``raw_confidence_text``
(the content the fixed parser located after a terminal ``Answer:`` / adjacent
``Confidence:``):

1. strip a single trailing decoded control token of the form ``<|...|>``
   (e.g. ``<|im_end|>``);
2. strip a single wrapping angle-bracket pair around the number (``<95>``);
3. accept the result ONLY if it is a bare integer in ``[0, 100]``.

Anything not unambiguously recoverable under this rule (percent signs, literal
``<integer 0-100>`` template echoes, empty/junk) stays MISSING. Previously
``parsed`` values are never altered.

This module is pure string logic and reads only stored decoded text + parquet
columns; it never loads a model or dataset. Importable ``repair_confidence`` is
reused by the figure code so plotting and repair stay separate.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

# One trailing decoded EOS sentinel. Restricted to the ONLY control token that
# is actually observed contaminating recoverable integer confidence fields in
# this study's data: "<|im_end|>". We deliberately do NOT strip arbitrary
# "<|...|>" tokens, nor the "</think>" reasoning-close tag (the single
# "<100></think>" case stays missing on purpose; a broader audit rule counted it
# as the 47,228th checkpoint recovery, but broadening is out of scope).
_TRAILING_CONTROL_TOKEN = re.compile(r"<\|im_end\|>\s*$")
# A number wrapped in a single angle-bracket pair, e.g. "<100>".
_ANGLE_WRAPPED_INT = re.compile(r"<\s*(\d+)\s*>")
_BARE_INT = re.compile(r"\d+")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MERGED = (
    REPO_ROOT
    / "results/part1"
    / "6b29188239ffa494ddb5f409f6dba2db510bcb9c3b6f8f7ada81c524af4d1e7c"
    / "merged"
)
DEFAULT_OUT = REPO_ROOT / "analysis/final-r5000/confidence"


def repair_confidence(raw_text) -> tuple[int | None, bool]:
    """Apply the narrow salvage rule to one ``raw_confidence_text`` value.

    Returns ``(value, recovered)`` where ``value`` is an int in [0, 100] and
    ``recovered`` is True only when the value came from cleaning a malformed
    field. Null/absent text returns ``(None, False)``.
    """
    if raw_text is None or not isinstance(raw_text, str):
        return None, False
    s = _TRAILING_CONTROL_TOKEN.sub("", raw_text.strip()).strip()
    m = _ANGLE_WRAPPED_INT.fullmatch(s)
    if m:
        s = m.group(1)
    if _BARE_INT.fullmatch(s):
        value = int(s)
        if 0 <= value <= 100:
            return value, True
    return None, False


def apply_repair(df: pd.DataFrame) -> pd.DataFrame:
    """Add final confidence columns without mutating the fixed contract fields.

    Adds:
      - ``confidence_value_final``  : int in [0,100] or <NA> (parsed or recovered)
      - ``confidence_status_final`` : "parsed" | "recovered" | "missing"
      - ``confidence_recovered``    : bool (True only for newly salvaged values)
      - ``normalized_confidence_final`` : value_final / 100.0
    """
    out = df.copy()
    status = out["confidence_parse_status"].to_numpy()
    raw = out["raw_confidence_text"]

    value_final = []
    status_final = []
    recovered = []
    for st, rc, parsed in zip(status, raw, out["raw_parsed_confidence"]):
        if st == "parsed":
            value_final.append(int(parsed))
            status_final.append("parsed")
            recovered.append(False)
            continue
        val, rec = repair_confidence(rc)
        if rec:
            value_final.append(val)
            status_final.append("recovered")
            recovered.append(True)
        else:
            value_final.append(pd.NA)
            status_final.append("missing")
            recovered.append(False)

    out["confidence_value_final"] = pd.array(value_final, dtype="Int64")
    out["confidence_status_final"] = status_final
    out["confidence_recovered"] = recovered
    out["normalized_confidence_final"] = out["confidence_value_final"].astype("Float64") / 100.0
    return out


def _load(merged: Path):
    nat = pd.read_parquet(
        merged / "natural_results.parquet",
        columns=[
            "raw_record_id", "subject", "run_id", "question_id",
            "confidence_parse_status", "raw_confidence_text", "raw_parsed_confidence",
            "normalized_confidence", "answer_parse_status", "natural_answer",
            "natural_correct",
        ],
    )
    cp = pd.read_parquet(
        merged / "checkpoint_results.parquet",
        columns=[
            "checkpoint_record_id", "parent_raw_record_id", "subject", "run_id",
            "requested_fraction", "confidence_parse_status", "raw_confidence_text",
            "raw_parsed_confidence", "normalized_confidence", "checkpoint_local_correct",
        ],
    )
    return nat, cp


def build_validation(nat_r: pd.DataFrame, cp_r: pd.DataFrame, nat_raw: pd.DataFrame) -> dict:
    """Assemble the validation payload requested before Figure 1."""
    def status_counts(df):
        before = df["confidence_parse_status"].value_counts().to_dict()
        after = df["confidence_status_final"].value_counts().to_dict()
        return {"before": {k: int(v) for k, v in before.items()},
                "after": {k: int(v) for k, v in after.items()}}

    # Invariant: previously-parsed values are unchanged.
    parsed = nat_r[nat_r["confidence_parse_status"] == "parsed"]
    parsed_unchanged = bool(
        (parsed["confidence_value_final"].astype("int64") ==
         parsed["raw_parsed_confidence"].astype("int64")).all()
    )
    cp_parsed = cp_r[cp_r["confidence_parse_status"] == "parsed"]
    cp_parsed_unchanged = bool(
        (cp_parsed["confidence_value_final"].astype("int64") ==
         cp_parsed["raw_parsed_confidence"].astype("int64")).all()
    )

    # Invariant: all final (parsed+recovered) values are integers in [0,100].
    def in_range(df):
        v = df.loc[df["confidence_status_final"] != "missing", "confidence_value_final"]
        return bool(((v >= 0) & (v <= 100)).all())

    # Per-fraction checkpoint coverage after repair.
    cov = (cp_r.assign(ok=cp_r["confidence_status_final"] != "missing")
           .groupby("requested_fraction")
           .agg(n=("ok", "size"),
                parsed=("confidence_parse_status", lambda s: int((s == "parsed").sum())),
                recovered=("confidence_status_final", lambda s: int((s == "recovered").sum())),
                usable=("ok", "sum"))
           )
    cov["coverage_pct"] = (100 * cov["usable"] / cov["n"]).round(1)
    per_fraction = {str(f): {k: int(r[k]) if k != "coverage_pct" else float(r[k])
                             for k in ["n", "parsed", "recovered", "usable", "coverage_pct"]}
                    for f, r in cov.iterrows()}

    # Confidence usability for the RQ3 discrimination cohort: evaluable natural
    # answer (natural_correct not null) AND a final natural confidence value.
    evaluable = nat_r["natural_correct"].notna()
    usable_nat = evaluable & (nat_r["confidence_status_final"] != "missing")

    # Non-recovered malformed forms (why they stayed missing).
    def nonrecovered_forms(df, k=8):
        m = df[(df["confidence_parse_status"] == "malformed") &
               (df["confidence_status_final"] == "missing")]
        return {str(k_): int(v_) for k_, v_ in
                m["raw_confidence_text"].value_counts().head(k).items()}

    # Representative examples.
    rec_ex = (nat_r[nat_r["confidence_status_final"] == "recovered"]
              [["raw_confidence_text", "confidence_value_final"]].head(6))
    unrec_ex = (nat_r[(nat_r["confidence_parse_status"] == "malformed") &
                      (nat_r["confidence_status_final"] == "missing")]
                [["raw_confidence_text"]].head(6))

    # Recovered-confidence distribution (natural + checkpoint pooled and natural).
    nat_final = nat_r.loc[nat_r["confidence_status_final"] != "missing", "confidence_value_final"].astype("int64")
    cp_final = cp_r.loc[cp_r["confidence_status_final"] != "missing", "confidence_value_final"].astype("int64")

    def dist(v):
        return {"n": int(v.size), "mean": round(float(v.mean()), 2),
                "median": float(v.median()), "min": int(v.min()), "max": int(v.max()),
                "pct_ge_90": round(float((v >= 90).mean()) * 100, 1),
                "pct_eq_100": round(float((v == 100).mean()) * 100, 1)}

    return {
        "rule": "strip one trailing <|im_end|> sentinel + optional <..> wrap; accept int in [0,100]",
        "natural": {
            "total_rows": int(len(nat_r)),
            "status_counts": status_counts(nat_r),
            "malformed_total": int((nat_r["confidence_parse_status"] == "malformed").sum()),
            "malformed_recovered": int(nat_r["confidence_recovered"].sum()),
            "malformed_not_recovered": int(((nat_r["confidence_parse_status"] == "malformed") &
                                            (nat_r["confidence_status_final"] == "missing")).sum()),
            "parsed_values_unchanged": parsed_unchanged,
            "all_final_in_range_0_100": in_range(nat_r),
            "nonrecovered_forms": nonrecovered_forms(nat_r),
            "evaluable_natural_answers": int(evaluable.sum()),
            "usable_confidence_among_evaluable": int(usable_nat.sum()),
            "usable_confidence_among_evaluable_correct": int((usable_nat & (nat_r["natural_correct"] == True)).sum()),
            "usable_confidence_among_evaluable_incorrect": int((usable_nat & (nat_r["natural_correct"] == False)).sum()),
            "distribution_final": dist(nat_final),
            "examples_recovered": rec_ex.astype(str).values.tolist(),
            "examples_not_recovered": unrec_ex.astype(str).values.tolist(),
        },
        "checkpoint": {
            "total_rows": int(len(cp_r)),
            "status_counts": status_counts(cp_r),
            "malformed_total": int((cp_r["confidence_parse_status"] == "malformed").sum()),
            "malformed_recovered": int(cp_r["confidence_recovered"].sum()),
            "malformed_not_recovered": int(((cp_r["confidence_parse_status"] == "malformed") &
                                            (cp_r["confidence_status_final"] == "missing")).sum()),
            "parsed_values_unchanged": cp_parsed_unchanged,
            "all_final_in_range_0_100": in_range(cp_r),
            "nonrecovered_forms": nonrecovered_forms(cp_r),
            "per_fraction_coverage": per_fraction,
            "distribution_final": dist(cp_final),
        },
    }


def _md(v: dict) -> str:
    n, c = v["natural"], v["checkpoint"]
    lines = []
    lines.append("# Verbalized-confidence repair — validation\n")
    lines.append(f"**Rule:** {v['rule']}\n")
    lines.append("## Natural terminal confidence\n")
    lines.append(f"- rows: {n['total_rows']:,}\n")
    lines.append(f"- status before: {n['status_counts']['before']}\n")
    lines.append(f"- status after:  {n['status_counts']['after']}\n")
    lines.append(f"- malformed recovered: {n['malformed_recovered']:,} / {n['malformed_total']:,} "
                 f"(not recovered: {n['malformed_not_recovered']})\n")
    lines.append(f"- previously-parsed values unchanged: {n['parsed_values_unchanged']}\n")
    lines.append(f"- all final values integers in [0,100]: {n['all_final_in_range_0_100']}\n")
    lines.append(f"- non-recovered forms: {n['nonrecovered_forms']}\n")
    lines.append(f"- **evaluable natural answers:** {n['evaluable_natural_answers']:,}; "
                 f"**usable confidence among them:** {n['usable_confidence_among_evaluable']:,} "
                 f"({n['usable_confidence_among_evaluable_correct']:,} correct / "
                 f"{n['usable_confidence_among_evaluable_incorrect']:,} incorrect)\n")
    lines.append(f"- recovered/final distribution: {n['distribution_final']}\n")
    lines.append(f"- examples recovered: {n['examples_recovered']}\n")
    lines.append(f"- examples NOT recovered: {n['examples_not_recovered']}\n")
    lines.append("\n## Checkpoint confidence\n")
    lines.append(f"- rows: {c['total_rows']:,}\n")
    lines.append(f"- status before: {c['status_counts']['before']}\n")
    lines.append(f"- status after:  {c['status_counts']['after']}\n")
    lines.append(f"- malformed recovered: {c['malformed_recovered']:,} / {c['malformed_total']:,} "
                 f"(not recovered: {c['malformed_not_recovered']:,})\n")
    lines.append(f"- previously-parsed values unchanged: {c['parsed_values_unchanged']}\n")
    lines.append(f"- all final values integers in [0,100]: {c['all_final_in_range_0_100']}\n")
    lines.append(f"- non-recovered forms (top): {c['nonrecovered_forms']}\n")
    lines.append(f"- final distribution: {c['distribution_final']}\n")
    lines.append("\n### Per-fraction coverage after repair\n")
    lines.append("| fraction | n | parsed | recovered | usable | coverage % |\n")
    lines.append("|---:|---:|---:|---:|---:|---:|\n")
    for f, r in c["per_fraction_coverage"].items():
        lines.append(f"| {f} | {r['n']:,} | {r['parsed']:,} | {r['recovered']:,} | "
                     f"{r['usable']:,} | {r['coverage_pct']} |\n")
    return "".join(lines)


def main(merged: Path = DEFAULT_MERGED, out: Path = DEFAULT_OUT) -> None:
    out.mkdir(parents=True, exist_ok=True)
    nat, cp = _load(merged)
    nat_r = apply_repair(nat)
    cp_r = apply_repair(cp)

    keep_nat = ["raw_record_id", "subject", "run_id", "question_id", "natural_correct",
                "confidence_parse_status", "confidence_status_final",
                "confidence_value_final", "confidence_recovered", "normalized_confidence_final"]
    keep_cp = ["checkpoint_record_id", "parent_raw_record_id", "subject", "run_id",
               "requested_fraction", "checkpoint_local_correct",
               "confidence_parse_status", "confidence_status_final",
               "confidence_value_final", "confidence_recovered", "normalized_confidence_final"]
    nat_r[keep_nat].to_parquet(out / "recovered_confidence_natural.parquet", index=False)
    cp_r[keep_cp].to_parquet(out / "recovered_confidence_checkpoint.parquet", index=False)

    validation = build_validation(nat_r, cp_r, nat)
    (out / "confidence_repair_validation.json").write_text(json.dumps(validation, indent=2))
    (out / "confidence_repair_validation.md").write_text(_md(validation))
    print(_md(validation))
    print(f"\nWrote artifacts to {out}")


if __name__ == "__main__":
    main()
