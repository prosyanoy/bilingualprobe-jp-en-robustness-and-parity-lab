"""Turn result dicts into ranked failure categories and a Markdown report."""
from __future__ import annotations


def worst_categories(model_result: dict, top: int = 5, min_clean_correct: int = 5) -> list[dict]:
    """Highest failure rates over (perturbation, language) cells."""
    cells = []
    for pname, entry in model_result["perturbations"].items():
        for lang in ("en", "ja"):
            e = entry.get(lang)
            if e and e["failure_rate"] is not None and e["n_clean_correct"] >= min_clean_correct:
                cells.append({"perturbation": pname, "lang": lang, "failure_rate": e["failure_rate"],
                              "ci": e["failure_ci"], "n": e["n_clean_correct"]})
    cells.sort(key=lambda c: (-c["failure_rate"], c["perturbation"], c["lang"]))
    return cells[:top]


def summary_rows(results: dict) -> list[dict]:
    """One row per model: accuracies, parity gap + CI, mean failure rates."""
    rows = []
    for name, r in results["models"].items():
        par = r["parity"] or {}
        fr = {lang: [e[lang]["failure_rate"] for e in r["perturbations"].values()
                     if e.get(lang) and e[lang]["failure_rate"] is not None] for lang in ("en", "ja")}
        mean = lambda xs: sum(xs) / len(xs) if xs else None
        rows.append({
            "model": name,
            "backend": r["backend"],
            "acc_en": r["clean"]["en"]["value"] if r["clean"]["en"] else None,
            "acc_ja": r["clean"]["ja"]["value"] if r["clean"]["ja"] else None,
            "gap_en_minus_ja": par.get("gap"),
            "gap_ci_lo": par["ci"][0] if par else None,
            "gap_ci_hi": par["ci"][1] if par else None,
            "parity_ratio": par.get("ratio"),
            "mean_fail_en": mean(fr["en"]),
            "mean_fail_ja": mean(fr["ja"]),
        })
    return rows


def failure_matrix(model_result: dict) -> list[dict]:
    """Rows of perturbation x language failure rates (+ ja-en gap) for tables/heatmaps."""
    rows = []
    for pname, entry in model_result["perturbations"].items():
        g = entry.get("failure_gap_ja_minus_en")
        rows.append({
            "perturbation": pname,
            "fail_en": (entry.get("en") or {}).get("failure_rate"),
            "fail_ja": (entry.get("ja") or {}).get("failure_rate"),
            "gap_ja_minus_en": g["gap"] if g else None,
            "gap_significant": g["significant"] if g else None,
        })
    return rows


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def to_markdown(results: dict) -> str:
    meta = results["meta"]
    out = ["# BilingualProbe report", "",
           f"Data: **{meta['data_source']}** - train rows {meta['n_train_rows']}, test rows {meta['n_test_rows']}, "
           f"seed {meta['seed']}, bootstrap resamples {meta['n_boot']}.", "",
           "## Parity summary", "",
           "Gap = accuracy(EN) - accuracy(JA) with 95% bootstrap CI; ratio = min/max accuracy (1.0 = perfect parity).", "",
           "| model | backend | acc EN | acc JA | gap [95% CI] | ratio | mean fail EN | mean fail JA |",
           "|---|---|---|---|---|---|---|---|"]
    for r in summary_rows(results):
        gap = "n/a" if r["gap_en_minus_ja"] is None else \
            f"{100 * r['gap_en_minus_ja']:+.1f} pp [{100 * r['gap_ci_lo']:+.1f}, {100 * r['gap_ci_hi']:+.1f}]"
        ratio = "n/a" if r["parity_ratio"] is None else f"{r['parity_ratio']:.3f}"
        out.append(f"| {r['model']} | {r['backend']} | {_pct(r['acc_en'])} | {_pct(r['acc_ja'])} | {gap} | {ratio} | "
                   f"{_pct(r['mean_fail_en'])} | {_pct(r['mean_fail_ja'])} |")
    for name, r in results["models"].items():
        out += ["", f"## {name}", "", "Failure rate = P(wrong on perturbed | right on clean). "
                "Gap = JA - EN (positive: Japanese more fragile).", "",
                "| perturbation | fail EN | fail JA | gap JA-EN | significant |", "|---|---|---|---|---|"]
        for row in failure_matrix(r):
            g = "n/a" if row["gap_ja_minus_en"] is None else f"{100 * row['gap_ja_minus_en']:+.1f} pp"
            sig = "" if row["gap_significant"] is None else ("yes" if row["gap_significant"] else "no")
            out.append(f"| {row['perturbation']} | {_pct(row['fail_en'])} | {_pct(row['fail_ja'])} | {g} | {sig} |")
        out += ["", "Worst failure categories:", ""]
        for c in worst_categories(r):
            out.append(f"- `{c['perturbation']}` / {c['lang']}: {_pct(c['failure_rate'])} "
                       f"(CI {_pct(c['ci'][0])}-{_pct(c['ci'][1])}, n={c['n']})")
    return "\n".join(out) + "\n"
