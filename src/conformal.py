"""
Bonus 3 : prédiction conforme (ensembles de prédiction à couverture garantie).

Au lieu d'un seul comportement, le modèle renvoie un ENSEMBLE de comportements
plausibles, construit pour contenir le vrai comportement avec une probabilité
d'au moins 1 - alpha (ex. 90 %). La taille de l'ensemble mesure l'incertitude.

Protocole (split conformal) :
  1. on sépare les chiens du TRAIN en « apprentissage » et « calibration » ;
  2. on réentraîne le modèle retenu (mêmes hyperparamètres) sur l'apprentissage ;
  3. on calibre le seuil sur les chiens de calibration ;
  4. on mesure couverture et taille des ensembles sur les chiens de TEST.

Hypothèse à discuter : la garantie suppose des données échangeables. Ici les
fenêtres d'un même chien sont corrélées ; en calibrant sur des chiens
différents, la garantie vaut « en moyenne sur des chiens » et la couverture
peut varier fortement d'un chien à l'autre (on la mesure par chien).

Deux implémentations : une version « maison » de la méthode LAC (quelques
lignes, pour montrer qu'on comprend la méthode) et la bibliothèque MAPIE (LAC
et APS), qui doivent donner des résultats cohérents.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from . import config as cfg


def calibration_split(train_df: pd.DataFrame, cal_size: float = 0.25, seed: int = cfg.RANDOM_STATE):
    dogs = np.sort(train_df["DogID"].unique())
    gss = GroupShuffleSplit(n_splits=1, test_size=cal_size, random_state=seed)
    tr, cal = next(gss.split(dogs, groups=dogs))
    return (train_df[train_df["DogID"].isin(dogs[tr])].copy(),
            train_df[train_df["DogID"].isin(dogs[cal])].copy())


# ---------------------------------------------------------------------------
# Implémentation maison (LAC : Least Ambiguous set-valued Classifier)
# ---------------------------------------------------------------------------
def lac_threshold(proba_cal: np.ndarray, y_cal: np.ndarray, alpha: float) -> float:
    """Score de non-conformité = 1 - probabilité de la vraie classe ; quantile corrigé."""
    scores = 1.0 - proba_cal[np.arange(len(y_cal)), y_cal]
    n = len(scores)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def lac_sets(proba: np.ndarray, qhat: float) -> np.ndarray:
    """Matrice booléenne (n_fenêtres, n_classes) : classe dans l'ensemble ?"""
    return proba >= 1.0 - qhat


# ---------------------------------------------------------------------------
# MAPIE (compatible API v1 et API 0.x)
# ---------------------------------------------------------------------------
def mapie_sets(fitted_model, X_cal, y_cal, X_test, alpha: float, method: str = "lac") -> np.ndarray:
    try:  # MAPIE >= 1.0
        from mapie.classification import SplitConformalClassifier
        m = SplitConformalClassifier(estimator=fitted_model, confidence_level=1 - alpha,
                                     conformity_score=method, prefit=True)
        m.conformalize(X_cal, y_cal)
        _, sets = m.predict_set(X_test)
    except ImportError:  # MAPIE 0.x
        from mapie.classification import MapieClassifier
        m = MapieClassifier(estimator=fitted_model, cv="prefit", method=method)
        m.fit(X_cal, y_cal)
        _, sets = m.predict(X_test, alpha=alpha)
    return np.asarray(sets)[:, :, 0].astype(bool)


# ---------------------------------------------------------------------------
# Mesures
# ---------------------------------------------------------------------------
def set_metrics(sets: np.ndarray, y_true: np.ndarray) -> dict:
    covered = sets[np.arange(len(y_true)), y_true]
    sizes = sets.sum(axis=1)
    return {"couverture": float(covered.mean()), "taille_moyenne": float(sizes.mean()),
            "part_singletons": float((sizes == 1).mean()), "part_vides": float((sizes == 0).mean())}


def coverage_by(sets: np.ndarray, y_true: np.ndarray, groups, name: str) -> pd.DataFrame:
    """Couverture et taille moyenne par classe ou par chien."""
    df = pd.DataFrame({name: np.asarray(groups),
                       "couvert": sets[np.arange(len(y_true)), y_true],
                       "taille": sets.sum(axis=1)})
    return df.groupby(name).agg(n_fenetres=("couvert", "size"), couverture=("couvert", "mean"),
                                taille_moyenne=("taille", "mean")).round(3).reset_index()
