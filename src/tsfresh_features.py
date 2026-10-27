"""
Approche de comparaison : extraction automatique de variables avec tsfresh.

tsfresh calcule des centaines de descripteurs par série (autocorrélations,
coefficients de Fourier, entropies, quantiles...). On l'applique sur les MÊMES
fenêtres (cœur de 1 s) que l'approche 1 et on évalue avec le MÊME protocole
(mêmes chiens, même CV groupée, même sélection de modèle), pour que la
comparaison ne porte que sur la manière de construire les variables.

Coût : tsfresh est lent. On l'applique
(1) aux 4 normes seulement et
(2) à un sous-échantillon du train (nombre maximal de fenêtres par classe). Pour une
comparaison équitable, l'approche 1 est réentraînée sur ce même sous-échantillon.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config as cfg

NORMS = [f"{s}_norm" for s in cfg.SENSORS]


def subsample_per_class(
    df: pd.DataFrame,
    max_per_class: int,
    seed: int = cfg.RANDOM_STATE,
    target: str = "target",
) -> pd.DataFrame:
    parts = [
        g.sample(n=min(len(g), max_per_class), random_state=seed)
        for _, g in df.groupby(target)
    ]
    return pd.concat(parts).sort_index()


def build_long_format(
    moves: pd.DataFrame,
    windows: pd.DataFrame,
    label_window_s: float = cfg.LABEL_WINDOW_S,
    fs: int = cfg.SAMPLING_RATE,
) -> pd.DataFrame:
    """
    Format « long » attendu par tsfresh : une ligne par (fenêtre, instant).
    `moves` doit être le tableau issu de data.load_moves() (même ordre de lignes
    que lors du fenêtrage : row_start y fait référence).
    """
    n_core = int(round(label_window_s * fs))
    pos = (
        windows["row_start"].to_numpy()[:, None] + np.arange(n_core)[None, :]
    ).ravel()
    axes = moves[cfg.SENSOR_COLUMNS].to_numpy(np.float32)[pos]
    norms = np.sqrt((axes.reshape(len(axes), len(cfg.SENSORS), 3) ** 2).sum(axis=2))
    long = pd.DataFrame(norms, columns=NORMS)
    long.insert(0, "time", np.tile(np.arange(n_core), len(windows)))
    long.insert(0, "id", np.repeat(windows["window_id"].to_numpy(), n_core))
    return long


def _clean_name(name: str) -> str:
    # XGBoost refuse les noms de variables contenant [, ] ou <
    return "tsf_" + re.sub(r"[\[\]<>,\s]", "_", name)


def extract_tsfresh(
    long: pd.DataFrame, settings: str = "efficient", n_jobs: int = 4
) -> pd.DataFrame:
    """Renvoie un DataFrame indexé par window_id (NON imputé : voir impute_from_train)."""
    from tsfresh import extract_features
    from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters

    params = {"efficient": EfficientFCParameters(), "minimal": MinimalFCParameters()}[
        settings
    ]
    X = extract_features(
        long,
        column_id="id",
        column_sort="time",
        default_fc_parameters=params,
        n_jobs=n_jobs,
        disable_progressbar=False,
    )
    X.columns = [_clean_name(c) for c in X.columns]
    X.index.name = "window_id"
    return X


def impute_from_train(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Remplace NaN / ±inf avec des statistiques calculées sur le train seulement
    (médiane pour NaN, max/min pour ±inf), puis retire les variables constantes
    dans le train. Imputer sur train+test serait une fuite.
    """
    X_test = X_test.reindex(columns=X_train.columns)
    tr = X_train.replace([np.inf, -np.inf], np.nan)
    med, vmax, vmin = tr.median(), tr.max(), tr.min()
    med, vmax, vmin = med.fillna(0), vmax.fillna(0), vmin.fillna(0)

    def _fix(X):
        X = X.mask(X == np.inf, vmax, axis=1).mask(X == -np.inf, vmin, axis=1)
        return X.fillna(med).astype(np.float32)

    X_train, X_test = _fix(X_train), _fix(X_test)
    keep = X_train.columns[X_train.nunique() > 1]
    return X_train[keep], X_test[keep]
