"""Template-generated parallel JP/EN sentiment data, plus a public-data loader."""
from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .lexicon import ADJECTIVES, NOUNS, Adjective


@dataclass(frozen=True)
class Example:
    pair_id: int
    lang: str  # "en" | "ja"
    text: str
    label: str  # "positive" | "negative"
    style: str = ""  # Japanese register of the template: "polite" | "plain"


# Frames: (frame_id, supports_plain_japanese)
_FRAMES = ("copula", "think", "today", "honest", "contrast")


def _ja(frame: str, n1: str, a1: Adjective, n2: str, a2: Adjective, polite: bool) -> str:
    if frame == "copula":
        return f"{n1}は{a1.ja}{'です' if polite else ''}。"
    if frame == "think":
        return f"{n1}はとても{a1.ja}と{'思います' if polite else '思う'}。"
    if frame == "today":
        return f"今日の{n1}は{a1.ja}でした。"
    if frame == "honest":
        return f"正直、{n1}は{a1.ja}{'です' if polite else ''}。"
    # contrast: the *second* clause carries the overall sentiment
    return f"{n1}は{a1.ja}{'ですが' if polite else 'が'}、{n2}は{a2.ja}{'です' if polite else ''}。"


def _en(frame: str, n1: str, a1: Adjective, n2: str, a2: Adjective) -> str:
    if frame == "copula":
        return f"The {n1} is {a1.en}."
    if frame == "think":
        return f"I think the {n1} is very {a1.en}."
    if frame == "today":
        return f"Today the {n1} was {a1.en}."
    if frame == "honest":
        return f"Honestly, the {n1} is {a1.en}."
    return f"The {n1} is {a1.en}, but the {n2} is {a2.en}."


def render_pair(frame: str, noun1: int, adj1: int, noun2: int, adj2: int, polite: bool) -> tuple[str, str, str]:
    """Render (english, japanese, label) for one template instantiation."""
    a1, a2 = ADJECTIVES[adj1], ADJECTIVES[adj2]
    (j1, e1), (j2, e2) = NOUNS[noun1], NOUNS[noun2]
    if frame == "today":
        polite = True  # plain past of i-adjectives would need a different negation rule
    label = (a2 if frame == "contrast" else a1).polarity
    return _en(frame, e1, a1, e2, a2), _ja(frame, j1, a1, j2, a2, polite), label


def generate_pairs(
    n_pairs: int,
    seed: int = 0,
    exclude: set[str] | None = None,
    polite_ratio: float = 0.7,
) -> list[Example]:
    """Generate ``n_pairs`` parallel EN/JA examples (``2 * n_pairs`` rows).

    Both rows of a pair share ``pair_id`` and ``label``. Labels are balanced by
    construction. ``exclude`` lets the caller keep a test set free of strings
    seen in training (checked on both languages).
    """
    rng = random.Random(seed)
    exclude = set(exclude or ())
    seen: set[tuple] = set()
    out: list[Example] = []
    attempts = 0
    pair_id = 0
    while pair_id < n_pairs:
        attempts += 1
        if attempts > min(n_pairs * 200, 300_000):
            raise ValueError(f"template space exhausted: could only build {pair_id} of {n_pairs} pairs")
        frame = rng.choice(_FRAMES)
        want = "positive" if pair_id % 2 == 0 else "negative"
        pool = [i for i, a in enumerate(ADJECTIVES) if a.polarity == want]
        any_adj = list(range(len(ADJECTIVES)))
        a2 = rng.choice(pool)
        # for the contrast frame the first clause gets an arbitrary polarity
        a1 = rng.choice(any_adj) if frame == "contrast" else a2
        n1, n2 = rng.randrange(len(NOUNS)), rng.randrange(len(NOUNS))
        if frame == "contrast" and n1 == n2:
            continue
        polite = rng.random() < polite_ratio
        if frame == "today":
            polite = True
        key = (frame, n1, a1, n2, a2, polite) if frame == "contrast" else (frame, n1, a1, polite)
        if key in seen:
            continue
        en, ja, label = render_pair(frame, n1, a1, n2, a2, polite)
        if en in exclude or ja in exclude:
            continue
        seen.add(key)
        style = "polite" if polite else "plain"
        out.append(Example(pair_id, "en", en, label, style))
        out.append(Example(pair_id, "ja", ja, label, style))
        pair_id += 1
    return out


def make_splits(n_train: int, n_test: int, seed: int = 0) -> tuple[list[Example], list[Example]]:
    train = generate_pairs(n_train, seed=seed)
    seen = {e.text for e in train}
    test = generate_pairs(n_test, seed=seed + 10_007, exclude=seen)
    return train, test


# --------------------------------------------------------------------------
# Optional public dataset
# --------------------------------------------------------------------------
_DEFAULT_LABEL_MAP = {
    "0": "negative", "1": "positive", "neg": "negative", "pos": "positive",
    "negative": "negative", "positive": "positive",
}


def load_public_dataset(
    path: str | Path,
    text_col: str = "text",
    label_col: str = "label",
    lang_col: str = "lang",
    label_map: dict[str, str] | None = None,
    max_per_lang: int | None = None,
) -> list[Example]:
    """Load a binary-sentiment CSV/TSV/JSONL file with a language column.

    Suitable for exports of e.g. multilingual Amazon reviews or JGLUE-MARC.
    Rows are unpaired, so each gets its own ``pair_id`` and parity statistics
    fall back to independent bootstrap resampling.
    """
    path = Path(path)
    lmap = {str(k).lower(): v for k, v in (label_map or _DEFAULT_LABEL_MAP).items()}
    if path.suffix.lower() in {".jsonl", ".json"}:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        delim = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f, delimiter=delim))
    counts: dict[str, int] = {}
    out: list[Example] = []
    for i, row in enumerate(rows):
        for col in (text_col, label_col, lang_col):
            if col not in row:
                raise ValueError(f"column {col!r} missing in row {i} of {path}")
        lang = str(row[lang_col]).strip().lower()[:2]
        if lang not in {"en", "ja"}:
            continue
        raw = str(row[label_col]).strip().lower()
        if raw not in lmap:
            raise ValueError(f"unmapped label {raw!r}; pass label_map")
        if max_per_lang and counts.get(lang, 0) >= max_per_lang:
            continue
        counts[lang] = counts.get(lang, 0) + 1
        out.append(Example(len(out), lang, str(row[text_col]), lmap[raw]))
    if not out:
        raise ValueError(f"no usable en/ja rows in {path}")
    return out


def split_examples(examples: list[Example], test_frac: float = 0.3, seed: int = 0) -> tuple[list[Example], list[Example]]:
    rng = random.Random(seed)
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    cut = int(len(idx) * (1 - test_frac))
    return [examples[i] for i in idx[:cut]], [examples[i] for i in idx[cut:]]


def save_jsonl(examples: list[Example], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")
