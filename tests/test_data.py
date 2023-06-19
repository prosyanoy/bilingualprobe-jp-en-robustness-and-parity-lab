import json

import pytest

from bilingualprobe.data import generate_pairs, load_public_dataset, make_splits


def test_pairs_are_parallel_balanced_and_deterministic():
    a = generate_pairs(40, seed=3)
    assert a == generate_pairs(40, seed=3)
    assert len(a) == 80
    en, ja = a[0::2], a[1::2]
    for e, j in zip(en, ja):
        assert (e.lang, j.lang) == ("en", "ja")
        assert e.pair_id == j.pair_id and e.label == j.label
    assert sum(e.label == "positive" for e in en) == 20


def test_test_split_has_no_train_strings():
    train, test = make_splits(100, 50, seed=1)
    assert not {e.text for e in train} & {e.text for e in test}


def test_template_space_exhaustion_raises():
    with pytest.raises(ValueError):
        generate_pairs(10**6, seed=0)


def test_public_loader_csv_and_jsonl(tmp_path):
    csv_p = tmp_path / "d.csv"
    csv_p.write_text("text,label,lang\nいい映画,1,ja\nbad movie,0,en\nbonjour,1,fr\n", encoding="utf-8")
    rows = load_public_dataset(csv_p)
    assert [(r.lang, r.label) for r in rows] == [("ja", "positive"), ("en", "negative")]

    js = tmp_path / "d.jsonl"
    js.write_text("\n".join(json.dumps({"t": f"x{i}", "y": "pos", "l": "en"}) for i in range(5)), encoding="utf-8")
    rows = load_public_dataset(js, "t", "y", "l", max_per_lang=3)
    assert len(rows) == 3


def test_public_loader_rejects_unknown_label(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("text,label,lang\nx,maybe,en\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_public_dataset(p)
