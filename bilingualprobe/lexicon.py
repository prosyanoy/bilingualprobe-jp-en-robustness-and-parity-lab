"""Vocabulary shared by the dataset generator and the perturbation suite.

Adjectives are all Japanese i-adjectives so that polarity-flipping negation
(``ひどい`` -> ``ひどくない``) and politeness rewrites stay regular. Every
adjective has a *held-out synonym* that is never used to build training data;
the synonym-swap perturbation substitutes it to probe lexical generalisation.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Adjective:
    ja: str
    en: str
    polarity: str  # "positive" | "negative"
    ja_syn: str
    en_syn: str


ADJECTIVES: tuple[Adjective, ...] = (
    Adjective("素晴らしい", "wonderful", "positive", "すごい", "amazing"),
    Adjective("面白い", "interesting", "positive", "興味深い", "fascinating"),
    Adjective("楽しい", "enjoyable", "positive", "嬉しい", "delightful"),
    Adjective("心地よい", "comfortable", "positive", "気持ちいい", "pleasant"),
    Adjective("頼もしい", "reliable", "positive", "心強い", "dependable"),
    Adjective("ひどい", "terrible", "negative", "酷い", "awful"),
    Adjective("悪い", "bad", "negative", "まずい", "poor"),
    Adjective("汚い", "dirty", "negative", "汚らしい", "filthy"),
    Adjective("遅い", "slow", "negative", "のろい", "sluggish"),
    Adjective("騒がしい", "noisy", "negative", "うるさい", "loud"),
)

# (japanese, english)
NOUNS: tuple[tuple[str, str], ...] = (
    ("映画", "movie"),
    ("料理", "food"),
    ("サービス", "service"),
    ("ホテル", "hotel"),
    ("アプリ", "app"),
    ("スタッフ", "staff"),
    ("部屋", "room"),
    ("お店", "shop"),
    ("説明", "explanation"),
    ("ソフトウェア", "software"),
)

LABELS = ("negative", "positive")
FLIP = {"positive": "negative", "negative": "positive"}

JA_TRAIN_ADJ = {a.ja: a for a in ADJECTIVES}
EN_TRAIN_ADJ = {a.en: a for a in ADJECTIVES}
JA_ALL_ADJ = sorted({a.ja for a in ADJECTIVES} | {a.ja_syn for a in ADJECTIVES}, key=len, reverse=True)
EN_ALL_ADJ = sorted({a.en for a in ADJECTIVES} | {a.en_syn for a in ADJECTIVES}, key=len, reverse=True)
