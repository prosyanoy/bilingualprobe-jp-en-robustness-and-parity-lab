"""Local classifiers trained on mixed-language data.

* ``tfidf_lr``       TF-IDF char 1-3-grams + logistic regression
* ``tfidf_lr_nfkc``  same, with Unicode NFKC normalisation before vectorising
* ``neural``         small MLP on dense features; PyTorch when installed,
                     otherwise scikit-learn's MLPClassifier (same architecture)

Dense features for ``neural`` come from an optional multilingual sentence
encoder (Hugging Face) when ``BP_USE_EMBEDDINGS=1`` and it can be loaded;
otherwise from TF-IDF char n-grams, so everything works offline.
"""
from __future__ import annotations

import logging
import unicodedata
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline

log = logging.getLogger(__name__)

try:  # optional heavy dependency
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only without torch installed
    torch = None
    nn = None


def nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _tfidf(normalize: bool = False, max_features: int | None = None) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char", ngram_range=(1, 3), sublinear_tf=True, max_features=max_features,
        preprocessor=(lambda s: nfkc(s).lower()) if normalize else None,
    )


class SklearnTextClassifier:
    def __init__(self, name: str, normalize: bool, seed: int = 0):
        self.name = name
        self.backend = "sklearn"
        self.pipe = make_pipeline(_tfidf(normalize), LogisticRegression(C=10.0, max_iter=1000, random_state=seed))

    def fit(self, texts: Sequence[str], labels: Sequence[str]):
        self.pipe.fit(list(texts), list(labels))
        return self

    def predict(self, texts: Sequence[str]) -> list[str]:
        return [str(p) for p in self.pipe.predict(list(texts))]


class DenseEncoder:
    """Texts -> dense float32 matrix. Embeddings if requested+available, else TF-IDF."""

    def __init__(self, use_embeddings: bool = False, model_name: str = "", max_features: int = 3000):
        self.kind = "tfidf"
        self._st = None
        if use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer

                self._st = SentenceTransformer(model_name)
                self.kind = f"embeddings:{model_name}"
            except Exception as exc:  # offline, not installed, bad name ...
                log.warning("embeddings unavailable (%s); falling back to TF-IDF", exc)
        self._tfidf = _tfidf(False, max_features)

    def fit(self, texts: Sequence[str]):
        if self._st is None:
            self._tfidf.fit(list(texts))
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if self._st is not None:
            return np.asarray(self._st.encode(list(texts), show_progress_bar=False), dtype=np.float32)
        return self._tfidf.transform(list(texts)).toarray().astype(np.float32)


class NeuralClassifier:
    def __init__(self, seed: int = 0, use_embeddings: bool = False, embed_model: str = "",
                 hidden: int = 128, epochs: int = 40, force_sklearn: bool = False):
        self.name = "neural"
        self.seed, self.hidden, self.epochs = seed, hidden, epochs
        self.encoder = DenseEncoder(use_embeddings, embed_model)
        self.backend = "sklearn-mlp" if (torch is None or force_sklearn) else "torch"
        self.classes_: list[str] = []

    @property
    def features(self) -> str:
        return self.encoder.kind

    def fit(self, texts: Sequence[str], labels: Sequence[str]):
        X = self.encoder.fit(texts).transform(texts)
        self.classes_ = sorted(set(labels))
        y = np.array([self.classes_.index(l) for l in labels])
        if self.backend == "torch":
            torch.manual_seed(self.seed)
            self.net = nn.Sequential(
                nn.Linear(X.shape[1], self.hidden), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(self.hidden, len(self.classes_)),
            )
            opt = torch.optim.Adam(self.net.parameters(), lr=2e-3, weight_decay=1e-4)
            loss_fn = nn.CrossEntropyLoss()
            Xt, yt = torch.from_numpy(X), torch.from_numpy(y).long()
            gen = torch.Generator().manual_seed(self.seed)
            self.net.train()
            for _ in range(self.epochs):
                perm = torch.randperm(len(Xt), generator=gen)
                for i in range(0, len(perm), 64):
                    b = perm[i:i + 64]
                    opt.zero_grad()
                    loss_fn(self.net(Xt[b]), yt[b]).backward()
                    opt.step()
            self.net.eval()
        else:
            self.net = MLPClassifier((self.hidden,), alpha=1e-4, max_iter=300, random_state=self.seed)
            self.net.fit(X, y)
        return self

    def predict(self, texts: Sequence[str]) -> list[str]:
        X = self.encoder.transform(texts)
        if self.backend == "torch":
            with torch.no_grad():
                idx = self.net(torch.from_numpy(X)).argmax(dim=1).numpy()
        else:
            idx = self.net.predict(X)
        return [self.classes_[int(i)] for i in idx]


MODEL_NAMES = ("tfidf_lr", "tfidf_lr_nfkc", "neural")


def build_model(name: str, seed: int = 0, use_embeddings: bool = False, embed_model: str = ""):
    if name == "tfidf_lr":
        return SklearnTextClassifier(name, normalize=False, seed=seed)
    if name == "tfidf_lr_nfkc":
        return SklearnTextClassifier(name, normalize=True, seed=seed)
    if name == "neural":
        return NeuralClassifier(seed, use_embeddings, embed_model)
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
