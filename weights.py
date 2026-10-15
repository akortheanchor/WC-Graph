"""Density-ratio (likelihood-ratio) weights for covariate shift.

w(x) = dQ_X/dP_X(x) is estimated with a probabilistic domain classifier
(Bickel et al., 2009; Tibshirani et al., 2019):  w(x) = c * h(x)/(1-h(x)),
where h(x) = P(target | x) and c = n_source / n_target corrects class priors.
The classifier is fit on *source inputs that are not used for calibration*
(the agent's training inputs) versus unlabeled target inputs, so the estimated
weights are independent of the calibration sample.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class DomainClassifierRatio:
    """Stabilised density-ratio estimator used by *every* weighted method.

    * L2-regularised logistic regression, C chosen by 5-fold CV log-loss from
      {0.01, 0.1, 1, 10};
    * weights clipped at the ``clip_q`` quantile of the estimated weights on the
      classifier's own source sample (independent of the calibration data).
    Without these two safeguards the effective sample size of the calibration set
    collapsed to ~1 in 6/100 MNIST repetitions (README, 'Weight stabilisation').
    The bias they introduce is part of the Delta term of Theorem 1.
    """

    def __init__(self, Cs=(0.01, 0.1, 1.0, 10.0), clip_q: float | None = 0.99, seed: int = 0):
        self.Cs, self.clip_q, self.seed = list(Cs), clip_q, seed

    def fit(self, X_src: np.ndarray, X_tgt: np.ndarray) -> "DomainClassifierRatio":
        X = np.vstack([X_src, X_tgt])
        d = np.r_[np.zeros(len(X_src)), np.ones(len(X_tgt))]
        if len(self.Cs) == 1:
            clf = LogisticRegression(C=self.Cs[0], max_iter=2000)
        else:
            clf = LogisticRegressionCV(Cs=self.Cs, cv=5, scoring="neg_log_loss", max_iter=2000)
        self.model = make_pipeline(StandardScaler(), clf)
        with warnings.catch_warnings():            # sklearn>=1.8 API-transition notices
            warnings.simplefilter("ignore", FutureWarning)
            self.model.fit(X, d)
        self.prior = len(X_src) / len(X_tgt)
        self.clip = None
        if self.clip_q is not None:
            self.clip = float(np.quantile(self._raw(X_src), self.clip_q))
        return self

    def _raw(self, X: np.ndarray) -> np.ndarray:
        h = np.clip(self.model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        return self.prior * h / (1 - h)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        w = self._raw(X)
        if self.clip is not None:
            w = np.minimum(w, self.clip)
        return w


def effective_sample_size(w: np.ndarray) -> float:
    """Kish effective sample size of a weighted calibration set."""
    return float(w.sum() ** 2 / np.square(w).sum())


def weight_error_bound(w_hat_src: np.ndarray, w_true_src: np.ndarray) -> float:
    """Coverage-gap term Delta = 0.5 * E_P | w_hat/E_P[w_hat] - w | (Lei & Candes, 2021).

    Both arrays are evaluated on a large i.i.d. *source* sample, so the
    expectation is a Monte-Carlo average. Only computable when w is known.
    """
    wh = w_hat_src / w_hat_src.mean()
    wt = w_true_src / w_true_src.mean()
    return 0.5 * float(np.mean(np.abs(wh - wt)))
