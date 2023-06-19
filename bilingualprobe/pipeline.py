"""End-to-end experiment: data -> train -> perturb -> evaluate -> report."""
from __future__ import annotations

import json
from pathlib import Path

from .config import Config
from .data import Example, load_public_dataset, make_splits, save_jsonl, split_examples
from .evaluate import build_perturbed, evaluate_model
from .models import MODEL_NAMES, build_model
from .perturb import get_perturbations
from .report import to_markdown


def load_data(cfg: Config) -> tuple[list[Example], list[Example], str]:
    if cfg.public_data:
        train, test = split_examples(load_public_dataset(cfg.public_data), seed=cfg.seed)
        return train, test, f"public:{Path(cfg.public_data).name}"
    train, test = make_splits(cfg.n_train, cfg.n_test, cfg.seed)
    return train, test, "synthetic-templates"


def train_models(train: list[Example], cfg: Config, names=MODEL_NAMES) -> dict:
    texts, labels = [e.text for e in train], [e.label for e in train]
    models = {}
    for n in names:
        models[n] = build_model(n, cfg.seed, cfg.use_embeddings, cfg.embed_model).fit(texts, labels)
    return models


def run_experiment(cfg: Config, model_names=MODEL_NAMES, perturbation_names=None, write: bool = True) -> dict:
    train, test, source = load_data(cfg)
    models = train_models(train, cfg, model_names)
    perturbed = build_perturbed(test, get_perturbations(perturbation_names), cfg.seed)
    results = {
        "meta": {"data_source": source, "seed": cfg.seed, "n_boot": cfg.n_boot,
                 "n_train_rows": len(train), "n_test_rows": len(test),
                 "perturbations": sorted({p.perturbation for p in perturbed})},
        "models": {n: evaluate_model(m, test, perturbed, cfg.n_boot, cfg.seed) for n, m in models.items()},
    }
    if write:
        out = Path(cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "report.md").write_text(to_markdown(results), encoding="utf-8")
        save_jsonl(test, out / "test_set.jsonl")
    return results
