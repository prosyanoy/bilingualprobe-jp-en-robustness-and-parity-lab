"""Configuration from environment variables (optionally a local ``.env`` file)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env reader; real environment variables win."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


@dataclass
class Config:
    seed: int = 0
    n_train: int = 600  # parallel pairs -> 2x rows
    n_test: int = 300
    n_boot: int = 1000
    out_dir: str = "artifacts"
    use_embeddings: bool = False
    embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    public_data: str = ""

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv()
        d = cls()
        e = os.environ.get
        return cls(
            seed=int(e("BP_SEED", d.seed)),
            n_train=int(e("BP_N_TRAIN", d.n_train)),
            n_test=int(e("BP_N_TEST", d.n_test)),
            n_boot=int(e("BP_N_BOOT", d.n_boot)),
            out_dir=e("BP_OUT_DIR", d.out_dir),
            use_embeddings=e("BP_USE_EMBEDDINGS", "0").lower() in {"1", "true", "yes"},
            embed_model=e("BP_EMBED_MODEL", d.embed_model),
            public_data=e("BP_PUBLIC_DATA", d.public_data),
        )
