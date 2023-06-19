"""Perturbation suite.

Each perturbation maps ``(text, lang, rng)`` to a new string, or ``None`` when
it does not apply (wrong language, no polite form to rewrite, no adjective to
negate, ...). Non-applicable items are excluded from failure-rate denominators.
Negation is the only label-flipping perturbation (positive <-> negative).
"""
from __future__ import annotations

import random
import re
import unicodedata
import zlib
from dataclasses import dataclass
from typing import Callable

from .lexicon import EN_ALL_ADJ, EN_TRAIN_ADJ, FLIP, JA_ALL_ADJ, JA_TRAIN_ADJ

TYPO_RATE = 0.08  # fraction of characters edited (at least one edit)

_EN_KEYS = "qwertyuiop asdfghjkl zxcvbnm".split()
_HIRAGANA = [chr(c) for c in range(0x3041, 0x3094)]


def _neighbors(ch: str) -> str:
    for row in _EN_KEYS:
        i = row.find(ch.lower())
        if i >= 0:
            return row[max(0, i - 1):i] + row[i + 1:i + 2]
    return ch


def typo(text: str, lang: str, rng: random.Random) -> str | None:
    """Random character-level noise: swap, delete, duplicate or substitute."""
    letters = (lambda c: c.isalpha()) if lang == "en" else (lambda c: not c.isspace() and c not in "。、，．,.")
    positions = [i for i, c in enumerate(text) if letters(c)]
    if len(positions) < 3:
        return None
    n_edits = max(1, round(TYPO_RATE * len(positions)))
    chars = list(text)
    for pos in sorted(rng.sample(positions, min(n_edits, len(positions))), reverse=True):
        op = rng.choice(("swap", "delete", "dup", "sub"))
        if op == "swap" and pos + 1 < len(chars):
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        elif op == "delete":
            del chars[pos]
        elif op == "dup":
            chars.insert(pos, chars[pos])
        else:
            if lang == "en":
                chars[pos] = rng.choice(_neighbors(chars[pos]) or chars[pos])
            else:
                chars[pos] = rng.choice(_HIRAGANA)
    out = "".join(chars)
    return out if out != text else None


# ---- width ---------------------------------------------------------------
def _build_halfwidth_katakana() -> dict[str, str]:
    table: dict[str, str] = {}
    for cp in range(0xFF61, 0xFFA0):  # halfwidth CJK punctuation + katakana
        half = chr(cp)
        full = unicodedata.normalize("NFKC", half)
        if full != half:
            table.setdefault(full, half)
    # voiced / semi-voiced katakana decompose to base + combining mark
    for cp in range(0x30A1, 0x30FB):
        ch = chr(cp)
        if ch in table:
            continue
        d = unicodedata.normalize("NFD", ch)
        if len(d) == 2 and d[0] in table:
            mark = {"゙": "ﾞ", "゚": "ﾟ"}.get(d[1])
            if mark:
                table[ch] = table[d[0]] + mark
    return table


_TO_HALF_KANA = _build_halfwidth_katakana()


def width(text: str, lang: str, rng: random.Random) -> str | None:
    """Half/full-width mismatch: halfwidth katakana for JA, fullwidth Latin for EN."""
    if lang == "ja":
        out = "".join(_TO_HALF_KANA.get(c, c) for c in text)
    else:
        out = "".join(chr(ord(c) + 0xFEE0) if "!" <= c <= "~" else c for c in text)
    return out if out != text else None


# ---- politeness (Japanese only) -------------------------------------------
_ADJ_PAST = re.compile("(" + "|".join(map(re.escape, JA_ALL_ADJ)) + ")でした")


def casualize(text: str, lang: str, rng: random.Random) -> str | None:
    """Polite -> casual spoken register (drop です/ます, add sentence-final particle)."""
    if lang != "ja" or not re.search("です|でした|思います", text):
        return None
    out = _ADJ_PAST.sub(lambda m: m.group(1)[:-1] + "かった", text)
    out = out.replace("ですが", "けど").replace("と思います", "と思う").replace("です", "")
    particle = rng.choice(("よ", "ね", "なあ"))
    out = out.rstrip("。") + particle + "。"
    return out


def keigo(text: str, lang: str, rng: random.Random) -> str | None:
    """Polite -> formal/honorific register (でございます, と存じます)."""
    if lang != "ja" or not re.search("です|でした|思います", text):
        return None
    return (text.replace("でした", "でございました").replace("です", "でございます")
            .replace("と思います", "と存じます"))


# ---- negation -------------------------------------------------------------
_JA_ADJ_RE = re.compile("|".join(map(re.escape, JA_ALL_ADJ)))
_EN_ADJ_RE = re.compile(r"\b(" + "|".join(map(re.escape, EN_ALL_ADJ)) + r")\b")
_EN_COPULA_RE = re.compile(r"\b(is|was)\b")


def negate(text: str, lang: str, rng: random.Random) -> str | None:
    """Negate the final predicate adjective (the clause that carries sentiment)."""
    if lang == "ja":
        matches = list(_JA_ADJ_RE.finditer(text))
        if not matches:
            return None
        m = matches[-1]
        rest = text[m.end():]
        stem = m.group()[:-1]
        if rest.startswith("でした"):
            return text[:m.start()] + stem + "くなかったです" + rest[3:]
        return text[:m.start()] + stem + "くない" + rest
    matches = list(_EN_ADJ_RE.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    cop = list(_EN_COPULA_RE.finditer(text[:m.start()]))
    if not cop:
        return None
    c = cop[-1]
    return text[:c.end()] + " not" + text[c.end():]


# ---- synonym swap ---------------------------------------------------------
_JA_SYN = {a.ja: a.ja_syn for a in JA_TRAIN_ADJ.values()}
_EN_SYN = {a.en: a.en_syn for a in EN_TRAIN_ADJ.values()}
_JA_TRAIN_RE = re.compile("|".join(map(re.escape, sorted(_JA_SYN, key=len, reverse=True))))
_EN_TRAIN_RE = re.compile(r"\b(" + "|".join(map(re.escape, _EN_SYN)) + r")\b")


def synonym(text: str, lang: str, rng: random.Random) -> str | None:
    """Replace training-set adjectives with held-out synonyms."""
    if lang == "ja":
        out = _JA_TRAIN_RE.sub(lambda m: _JA_SYN[m.group()], text)
    else:
        out = _EN_TRAIN_RE.sub(lambda m: _EN_SYN[m.group()], text)
    return out if out != text else None


@dataclass(frozen=True)
class Perturbation:
    name: str
    description: str
    fn: Callable[[str, str, random.Random], str | None]
    flips_label: bool = False
    languages: tuple[str, ...] = ("en", "ja")

    def apply(self, text: str, lang: str, seed: int = 0) -> str | None:
        if lang not in self.languages:
            return None
        rng = random.Random(seed * 1_000_003 + zlib.crc32(f"{self.name}|{text}".encode()))
        return self.fn(text, lang, rng)

    def expected(self, label: str) -> str:
        return FLIP.get(label, label) if self.flips_label else label


PERTURBATIONS: tuple[Perturbation, ...] = (
    Perturbation("typo", "random character swaps/deletions/duplications/substitutions", typo),
    Perturbation("width", "halfwidth katakana (JA) / fullwidth Latin (EN)", width),
    Perturbation("politeness_casual", "Japanese polite -> casual register", casualize, languages=("ja",)),
    Perturbation("politeness_keigo", "Japanese polite -> honorific register", keigo, languages=("ja",)),
    Perturbation("negation", "negate the sentiment predicate (label flips)", negate, flips_label=True),
    Perturbation("synonym", "swap in held-out synonyms", synonym),
)


def get_perturbations(names: list[str] | None = None) -> list[Perturbation]:
    if not names:
        return list(PERTURBATIONS)
    by_name = {p.name: p for p in PERTURBATIONS}
    unknown = [n for n in names if n not in by_name]
    if unknown:
        raise ValueError(f"unknown perturbations {unknown}; choose from {sorted(by_name)}")
    return [by_name[n] for n in names]
