"""
Bonus : lissage temporel des prédictions.

Idée : un comportement dure en général plusieurs secondes ; une prédiction
isolée différente de ses voisines est souvent une erreur. On remplace la
probabilité de chaque fenêtre par la moyenne des probabilités des k fenêtres
voisines (fenêtre glissante centrée), puis on reprend l'argmax.

Pas de fuite de données : le lissage n'utilise que les prédictions (jamais les
vrais labels). La taille k est choisie sur des prédictions hors-pli du train,
jamais sur le test.
Limite : la fenêtre centrée utilise des prédictions « futures ». En temps réel,
il faudrait une fenêtre causale.
Le lissage ne traverse jamais une rupture temporelle (changement de segment ou
fenêtre manquante parce qu'exclue).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import f1_score

from . import config as cfg


def contiguous_runs(df: pd.DataFrame, step_s: float = cfg.STEP_S) -> np.ndarray:
    """Identifiant de « suite continue de fenêtres » (dans l'ordre de df)."""
    order = df.sort_values(["DogID", "TestNum", "segment", "t_center"]).index
    d = df.loc[order]
    new_run = (
        (d["DogID"].diff() != 0)
        | (d["TestNum"].diff() != 0)
        | (d["segment"].diff() != 0)
        | (d["t_center"].diff() > 1.5 * step_s)
    )
    runs = pd.Series(new_run.cumsum().to_numpy(), index=order)
    return runs.loc[df.index].to_numpy()


def smooth_proba(
    proba: np.ndarray, df: pd.DataFrame, k: int, step_s: float = cfg.STEP_S
) -> np.ndarray:
    """Moyenne glissante centrée de taille k (impair) des probabilités, par suite continue."""
    if k <= 1:
        return proba.copy()
    runs = contiguous_runs(df, step_s)
    order = np.lexsort((df["t_center"].to_numpy(), runs))
    out = np.empty_like(proba)
    sorted_runs = runs[order]
    bounds = np.flatnonzero(np.diff(sorted_runs)) + 1
    for idx in np.split(order, bounds):
        out[idx] = uniform_filter1d(proba[idx], size=k, axis=0, mode="nearest")
    return out


def choose_k(
    oof_proba: np.ndarray,
    y_true_encoded,
    df_train: pd.DataFrame,
    ks=(1, 3, 5, 7, 9, 11, 15),
) -> pd.DataFrame:
    """F1 macro hors-pli sur le TRAIN pour chaque taille de lissage k."""
    rows = []
    for k in ks:
        pred = smooth_proba(oof_proba, df_train, k).argmax(axis=1)
        rows.append(
            {
                "k": k,
                "duree_lissage_s": round(cfg.LABEL_WINDOW_S + (k - 1) * cfg.STEP_S, 1),
                "f1_macro_oof_train": f1_score(y_true_encoded, pred, average="macro"),
            }
        )
    return pd.DataFrame(rows)
