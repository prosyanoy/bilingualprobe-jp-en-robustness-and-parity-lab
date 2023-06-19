"""Clean + perturbed evaluation with per-language scores and parity statistics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Example
from .perturb import Perturbation
from .stats import bootstrap_ci, bootstrap_gap, parity_ratio

LANGS = ("en", "ja")


@dataclass(frozen=True)
class PerturbedItem:
    index: int  # index into the clean example list
    perturbation: str
    lang: str
    pair_id: int
    text: str
    expected: str


def build_perturbed(examples: list[Example], perturbations: list[Perturbation], seed: int = 0) -> list[PerturbedItem]:
    """Apply every applicable perturbation once to every example (model-independent)."""
    items = []
    for p in perturbations:
        for i, ex in enumerate(examples):
            new = p.apply(ex.text, ex.lang, seed)
            if new is not None:
                items.append(PerturbedItem(i, p.name, ex.lang, ex.pair_id, new, p.expected(ex.label)))
    return items


def _rate(values, n_boot, seed):
    ci = bootstrap_ci(values, n_boot, seed)
    if ci is None:
        return None
    return {"value": ci[0], "ci": [ci[1], ci[2]], "n": len(values)}


def _gap(a, ids_a, b, ids_b, n_boot, seed):
    r = bootstrap_gap(a, ids_a, b, ids_b, n_boot, seed)
    if r is None:
        return None
    gap, lo, hi, paired = r
    return {"gap": gap, "ci": [lo, hi], "paired": paired, "significant": not (lo <= 0 <= hi)}


def evaluate_model(model, test: list[Example], perturbed: list[PerturbedItem], n_boot: int = 1000, seed: int = 0) -> dict:
    """Return a JSON-serialisable result dict for one fitted model.

    * ``clean``: accuracy per language and the parity block (gap = en - ja,
      min/max ratio, bootstrap CI of the gap).
    * ``perturbations``: per perturbation and language, ``failure_rate`` =
      P(wrong on perturbed | right on clean), plus the en/ja failure-rate gap
      (ja - en: positive means Japanese is more fragile).
    """
    preds = model.predict([e.text for e in test])
    correct = np.array([p == e.label for p, e in zip(preds, test)], dtype=float)
    lang_arr = np.array([e.lang for e in test])
    pair_ids = np.array([e.pair_id for e in test])

    clean: dict = {}
    sel = {}
    for lang in LANGS:
        m = lang_arr == lang
        sel[lang] = m
        clean[lang] = _rate(correct[m], n_boot, seed)
    parity = None
    if clean["en"] and clean["ja"]:
        g = _gap(correct[sel["en"]], pair_ids[sel["en"]], correct[sel["ja"]], pair_ids[sel["ja"]], n_boot, seed)
        parity = {**g, "ratio": parity_ratio(clean["en"]["value"], clean["ja"]["value"])}

    # perturbed predictions in one batch
    pred_p = model.predict([it.text for it in perturbed]) if perturbed else []
    by_key: dict[tuple[str, str], list[tuple[int, float, float]]] = {}
    for it, pp in zip(perturbed, pred_p):
        was_right = correct[it.index]
        now_right = float(pp == it.expected)
        by_key.setdefault((it.perturbation, it.lang), []).append((it.pair_id, was_right, now_right))

    results: dict = {}
    names = sorted({k[0] for k in by_key})
    for name in names:
        entry: dict = {}
        fail_vals: dict = {}
        for lang in LANGS:
            rows = by_key.get((name, lang), [])
            if not rows:
                entry[lang] = None
                continue
            right = [(pid, 1.0 - now) for pid, was, now in rows if was == 1.0]
            fail_vals[lang] = ([r[1] for r in right], [r[0] for r in right])
            fr = _rate(fail_vals[lang][0], n_boot, seed)
            entry[lang] = {
                "n": len(rows),
                "n_clean_correct": len(right),
                "accuracy": float(np.mean([now for _, _, now in rows])),
                "failure_rate": fr["value"] if fr else None,
                "failure_ci": fr["ci"] if fr else None,
            }
        gap = None
        if len(fail_vals) == 2 and fail_vals["en"][0] and fail_vals["ja"][0]:
            gap = _gap(fail_vals["ja"][0], fail_vals["ja"][1], fail_vals["en"][0], fail_vals["en"][1], n_boot, seed)
        entry["failure_gap_ja_minus_en"] = gap
        results[name] = entry

    return {
        "backend": getattr(model, "backend", ""),
        "features": getattr(model, "features", ""),
        "clean": clean,
        "parity": parity,
        "perturbations": results,
    }
