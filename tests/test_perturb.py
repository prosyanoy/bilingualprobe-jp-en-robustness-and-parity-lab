import random
import unicodedata

from bilingualprobe.data import generate_pairs
from bilingualprobe.perturb import PERTURBATIONS, get_perturbations, negate, width

P = {p.name: p for p in PERTURBATIONS}


def test_negation_ja_and_en():
    assert P["negation"].apply("ホテルは素晴らしいです。", "ja") == "ホテルは素晴らしくないです。"
    assert P["negation"].apply("今日のホテルはひどいでした。", "ja") == "今日のホテルはひどくなかったです。"
    assert P["negation"].apply("The hotel is wonderful.", "en") == "The hotel is not wonderful."
    assert P["negation"].apply("The movie is bad, but the app was slow.", "en") == "The movie is bad, but the app was not slow."
    assert P["negation"].apply("映画は面白いですが、部屋は汚いです。", "ja") == "映画は面白いですが、部屋は汚くないです。"
    assert P["negation"].apply("hello", "en") is None
    assert P["negation"].expected("positive") == "negative"
    assert P["typo"].expected("positive") == "positive"


def test_width_roundtrips_under_nfkc():
    ja = "サービスはひどいです。"
    out = width(ja, "ja", random.Random(0))
    assert out != ja and "ｻ" in out and "ﾋﾞ" in out
    assert unicodedata.normalize("NFKC", out) == ja
    en = width("The app is bad.", "en", random.Random(0))
    assert en.startswith("Ｔｈｅ")
    assert unicodedata.normalize("NFKC", en) == "The app is bad."


def test_politeness_ja_only_and_rewrites():
    assert P["politeness_casual"].apply("The hotel is bad.", "en") is None
    kei = P["politeness_keigo"].apply("ホテルは素晴らしいと思います。", "ja")
    assert kei == "ホテルは素晴らしいと存じます。"
    assert P["politeness_keigo"].apply("今日のホテルはひどいでした。", "ja") == "今日のホテルはひどいでございました。"
    cas = P["politeness_casual"].apply("今日のホテルはひどいでした。", "ja")
    assert cas.startswith("今日のホテルはひどかった") and "です" not in cas
    assert P["politeness_casual"].apply("ホテルはひどい。", "ja") is None  # already plain


def test_synonym_uses_held_out_words():
    assert P["synonym"].apply("The hotel is wonderful.", "en") == "The hotel is amazing."
    assert P["synonym"].apply("ホテルはひどいです。", "ja") == "ホテルは酷いです。"
    assert P["synonym"].apply("The hotel is amazing.", "en") is None


def test_typo_is_seeded_and_changes_text():
    t = "The hotel is wonderful."
    a, b = P["typo"].apply(t, "en", 1), P["typo"].apply(t, "en", 1)
    assert a == b and a != t
    assert P["typo"].apply("ホテルは素晴らしいです。", "ja", 2) != "ホテルは素晴らしいです。"
    assert P["typo"].apply("ab", "en") is None


def test_every_generated_sentence_is_covered_by_negation():
    for ex in generate_pairs(60, seed=5):
        assert negate(ex.text, ex.lang, random.Random(0)) not in (None, ex.text)


def test_registry():
    assert [p.name for p in get_perturbations(["typo"])] == ["typo"]
    try:
        get_perturbations(["nope"])
    except ValueError:
        pass
    else:
        raise AssertionError
