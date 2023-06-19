import numpy as np

from bilingualprobe.data import make_splits
from bilingualprobe.evaluate import build_perturbed, evaluate_model
from bilingualprobe.models import MODEL_NAMES, NeuralClassifier, build_model
from bilingualprobe.perturb import get_perturbations
from bilingualprobe.report import summary_rows, to_markdown, worst_categories
from bilingualprobe.stats import bootstrap_ci, bootstrap_gap, parity_ratio


def test_bootstrap_ci_brackets_mean_and_handles_empty():
    m, lo, hi = bootstrap_ci([1, 0, 1, 1, 0, 1, 1, 1], n_boot=500)
    assert lo <= m <= hi
    assert bootstrap_ci([]) is None
    assert bootstrap_ci([1, 1, 1]) == (1.0, 1.0, 1.0)


def test_gap_paired_vs_independent_and_significance():
    a = np.ones(100)
    b = np.r_[np.ones(60), np.zeros(40)]
    gap, lo, hi, paired = bootstrap_gap(a, range(100), b, range(100), n_boot=500)
    assert paired and abs(gap - 0.4) < 1e-9 and lo > 0
    # shuffled id order must still pair by id
    order = list(reversed(range(100)))
    g2 = bootstrap_gap(a, range(100), b[order], order, n_boot=500)
    assert abs(g2[0] - 0.4) < 1e-9
    assert bootstrap_gap(a, range(100), b, range(1000, 1100), n_boot=200)[3] is False
    assert bootstrap_gap([], [], b, range(100)) is None


def test_parity_ratio():
    assert parity_ratio(0.9, 0.9) == 1.0
    assert parity_ratio(0.5, 1.0) == 0.5
    assert parity_ratio(0, 0) == 1.0


def _run(model_name, force_sklearn=False):
    train, test = make_splits(150, 80, seed=0)
    model = build_model(model_name, seed=0)
    if force_sklearn:
        model = NeuralClassifier(seed=0, force_sklearn=True)
    model.fit([e.text for e in train], [e.label for e in train])
    perturbed = build_perturbed(test, get_perturbations(), seed=0)
    return evaluate_model(model, test, perturbed, n_boot=200, seed=0), test, perturbed


def test_evaluation_structure_and_expected_behaviour():
    res, test, perturbed = _run("tfidf_lr")
    assert res["clean"]["en"]["value"] > 0.8
    lo, hi = res["parity"]["ci"]
    assert lo <= res["parity"]["gap"] <= hi and 0 <= res["parity"]["ratio"] <= 1
    p = res["perturbations"]
    assert p["politeness_keigo"]["en"] is None and p["politeness_keigo"]["ja"]["n"] > 0
    # models trained without negation cannot flip: negation must be the worst category
    assert p["negation"]["en"]["failure_rate"] > 0.8
    # without NFKC, width perturbation breaks English; with it, nothing breaks
    assert p["width"]["en"]["failure_rate"] > 0.2
    res_n, *_ = _run("tfidf_lr_nfkc")
    assert res_n["perturbations"]["width"]["en"]["failure_rate"] == 0.0
    assert res_n["perturbations"]["width"]["ja"]["failure_rate"] == 0.0


def test_neural_sklearn_backend_and_torch_if_available():
    res, *_ = _run("neural", force_sklearn=True)
    assert res["backend"] == "sklearn-mlp" and res["clean"]["en"]["value"] > 0.8
    try:
        import torch  # noqa: F401
    except ImportError:
        return
    res_t, *_ = _run("neural")
    assert res_t["backend"] == "torch" and res_t["clean"]["en"]["value"] > 0.8


def test_report_outputs():
    res, test, perturbed = _run("tfidf_lr")
    results = {"meta": {"data_source": "t", "seed": 0, "n_boot": 200, "n_train_rows": 300, "n_test_rows": len(test)},
               "models": {"tfidf_lr": res}}
    worst = worst_categories(res, top=3)
    assert len(worst) == 3 and worst[0]["failure_rate"] >= worst[1]["failure_rate"] >= worst[2]["failure_rate"]
    assert summary_rows(results)[0]["model"] == "tfidf_lr"
    md = to_markdown(results)
    assert "Parity summary" in md and "negation" in md


def test_all_model_names_build():
    for n in MODEL_NAMES:
        assert build_model(n).name == n
