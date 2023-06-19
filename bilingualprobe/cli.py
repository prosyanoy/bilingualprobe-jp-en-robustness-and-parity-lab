from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from .config import Config
from .data import make_splits, save_jsonl
from .models import MODEL_NAMES
from .perturb import PERTURBATIONS
from .pipeline import run_experiment
from .report import to_markdown


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bilingualprobe", description="JP/EN robustness and parity lab")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="write the synthetic parallel dataset as JSONL")
    g.add_argument("--out", default="artifacts/data")
    g.add_argument("--n-train", type=int)
    g.add_argument("--n-test", type=int)
    g.add_argument("--seed", type=int)

    r = sub.add_parser("run", help="train models, run perturbation suite, write results.json + report.md")
    r.add_argument("--out", dest="out_dir")
    r.add_argument("--seed", type=int)
    r.add_argument("--n-train", type=int)
    r.add_argument("--n-test", type=int)
    r.add_argument("--n-boot", type=int)
    r.add_argument("--public-data", help="CSV/TSV/JSONL with text,label,lang columns")
    r.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    r.add_argument("--perturbations", nargs="+", choices=[p.name for p in PERTURBATIONS])
    r.add_argument("--embeddings", action="store_true", help="try multilingual HF embeddings for the neural model")

    args = ap.parse_args(argv)
    cfg = Config.from_env()
    overrides = {k: v for k, v in vars(args).items() if k in {f.name for f in dataclasses.fields(Config)} and v is not None}
    if getattr(args, "embeddings", False):
        overrides["use_embeddings"] = True
    cfg = dataclasses.replace(cfg, **overrides)

    if args.cmd == "generate":
        train, test = make_splits(cfg.n_train, cfg.n_test, cfg.seed)
        save_jsonl(train, Path(args.out) / "train.jsonl")
        save_jsonl(test, Path(args.out) / "test.jsonl")
        print(f"wrote {len(train)} train and {len(test)} test rows to {args.out}")
        return 0

    results = run_experiment(cfg, args.models, args.perturbations)
    print(to_markdown(results))
    print(f"artifacts written to {cfg.out_dir}/")
    return 0
