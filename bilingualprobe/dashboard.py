"""Streamlit dashboard: ``streamlit run bilingualprobe/dashboard.py``."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from bilingualprobe.config import Config
from bilingualprobe.perturb import PERTURBATIONS
from bilingualprobe.pipeline import load_data, run_experiment, train_models
from bilingualprobe.report import failure_matrix, summary_rows, worst_categories

st.set_page_config(page_title="BilingualProbe", layout="wide")
cfg = Config.from_env()


@st.cache_data(show_spinner="Running experiment (a few seconds, fully offline)...")
def get_results() -> dict:
    path = Path(cfg.out_dir) / "results.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return run_experiment(cfg)


@st.cache_resource(show_spinner="Training models...")
def get_models():
    train, _, _ = load_data(cfg)
    return train_models(train, cfg)


results = get_results()
meta = results["meta"]
st.title("BilingualProbe - JP/EN robustness and parity lab")
st.caption(f"data: {meta['data_source']} | train rows {meta['n_train_rows']} | test rows {meta['n_test_rows']} "
           f"| seed {meta['seed']} | {meta['n_boot']} bootstrap resamples")

tab_parity, tab_fail, tab_probe = st.tabs(["Parity", "Failure categories", "Probe a sentence"])

with tab_parity:
    st.subheader("Cross-language parity (clean test set)")
    df = pd.DataFrame(summary_rows(results)).set_index("model")
    st.dataframe(df.style.format({c: "{:.3f}" for c in df.select_dtypes("number").columns}), use_container_width=True)
    st.caption("gap = acc(EN) - acc(JA); the CI is a paired bootstrap over parallel sentence pairs. "
               "If the CI excludes 0 the gap is statistically meaningful.")
    st.bar_chart(df[["acc_en", "acc_ja"]])

with tab_fail:
    model = st.selectbox("Model", list(results["models"]))
    r = results["models"][model]
    st.subheader("Worst failure categories")
    st.dataframe(pd.DataFrame(worst_categories(r, top=8)), use_container_width=True)
    st.subheader("Failure rate by perturbation and language")
    fm = pd.DataFrame(failure_matrix(r)).set_index("perturbation")
    st.bar_chart(fm[["fail_en", "fail_ja"]])
    st.dataframe(fm, use_container_width=True)
    st.caption("failure rate = P(wrong after perturbation | right before). "
               "gap_ja_minus_en > 0 means Japanese is more fragile.")

with tab_probe:
    text = st.text_input("Sentence", "ホテルは素晴らしいです。")
    lang = st.radio("Language", ["ja", "en"], horizontal=True)
    if text:
        models = get_models()
        rows = [{"perturbation": "(original)", "text": text,
                 **{n: m.predict([text])[0] for n, m in models.items()}}]
        for p in PERTURBATIONS:
            new = p.apply(text, lang)
            if new is not None:
                rows.append({"perturbation": p.name, "text": new,
                             **{n: m.predict([new])[0] for n, m in models.items()}})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.caption("negation flips the expected label; every other perturbation should keep the prediction.")
