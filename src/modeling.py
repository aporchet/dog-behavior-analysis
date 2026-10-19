"""
Modèles, validation croisée groupée, sélection et sauvegarde.

Règles de méthode (corrigent l'ancienne version)
------------------------------------------------
1. Les hyperparamètres ET le choix du meilleur modèle se font sur le score de
   validation croisée (F1 macro), JAMAIS sur le test.
2. Le test n'est utilisé qu'une fois, pour évaluer le modèle retenu.
3. La validation croisée est groupée par chien (StratifiedGroupKFold : chaque
   chien n'apparaît que dans un pli, et les classes sont réparties le plus
   équitablement possible entre les plis, ce qui évite qu'un pli d'entraînement
   ne contienne pas une classe rare).
4. Toute étape apprise (standardisation, sélection de variables) est dans le
   Pipeline : elle est réapprise dans chaque pli, sans fuite.
"""
from __future__ import annotations

import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from . import config as cfg

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None
    warnings.warn("xgboost n'est pas installé : le modèle XGBoost sera ignoré.")


# ---------------------------------------------------------------------------
# Définition des modèles
# ---------------------------------------------------------------------------
def make_cv(n_splits: int = cfg.N_CV_SPLITS, seed: int = cfg.RANDOM_STATE):
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def make_selector():
    """
    Sélection de variables par importance d'une forêt aléatoire rapide :
    on garde les variables d'importance supérieure à la moyenne. Justification :
    réduire le bruit et le temps de calcul, et produire une liste interprétable
    des variables utiles (à commenter en soutenance).
    """
    return SelectFromModel(
        RandomForestClassifier(n_estimators=50, class_weight="balanced", n_jobs=1,
                               random_state=cfg.RANDOM_STATE),
        threshold="mean")


def get_model_configs(quick: bool = cfg.QUICK) -> dict:
    """
    Dictionnaire nom -> {model, params, sample_weight}.
    `sample_weight=True` : on passe des poids équilibrés au classifieur
    (équivalent de class_weight='balanced' pour XGBoost, qui ne l'a pas).
    Les pipelines utilisent un cache disque : la sélection de variables, qui ne
    dépend pas des hyperparamètres du classifieur, n'est calculée qu'une fois par
    pli et non une fois par point de la grille.
    """
    memory = str(cfg.CACHE_DIR)
    configs = {
        "Dummy_majoritaire": {"model": DummyClassifier(strategy="most_frequent"),
                              "params": {}, "sample_weight": False},
        "Dummy_stratifie": {"model": DummyClassifier(strategy="stratified", random_state=cfg.RANDOM_STATE),
                            "params": {}, "sample_weight": False},
        "LogisticRegression": {
            "model": Pipeline([("scaler", StandardScaler()), ("selecteur", make_selector()),
                               ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))],
                              memory=memory),
            "params": {"clf__C": [1] if quick else [0.1, 1, 10]},
            "sample_weight": False},
        "RandomForest": {
            "model": Pipeline([("selecteur", make_selector()),
                               ("clf", RandomForestClassifier(class_weight="balanced", n_jobs=1,
                                                              random_state=cfg.RANDOM_STATE))],
                              memory=memory),
            "params": ({"clf__n_estimators": [50], "clf__max_depth": [10]} if quick else
                       {"clf__n_estimators": [100, 300], "clf__max_depth": [None, 15, 25],
                        "clf__min_samples_leaf": [1, 5]}),
            "sample_weight": False},
    }
    if XGBClassifier is not None:
        configs["XGBoost"] = {
            "model": Pipeline([("selecteur", make_selector()),
                               ("clf", XGBClassifier(eval_metric="mlogloss", tree_method="hist", n_jobs=1,
                                                     random_state=cfg.RANDOM_STATE))],
                              memory=memory),
            "params": ({"clf__n_estimators": [50], "clf__max_depth": [3], "clf__learning_rate": [0.1]} if quick else
                       {"clf__n_estimators": [200, 400], "clf__max_depth": [4, 6, 8],
                        "clf__learning_rate": [0.05, 0.1]}),
            "sample_weight": True}
    return configs


DUMMY_MODELS = ("Dummy_majoritaire", "Dummy_stratifie")


# ---------------------------------------------------------------------------
# Recherche d'hyperparamètres
# ---------------------------------------------------------------------------
def _fit_params(conf: dict, y) -> dict:
    return {"clf__sample_weight": compute_sample_weight("balanced", y)} if conf["sample_weight"] else {}


def run_grid_search(name: str, conf: dict, X, y, groups, cv=None, scoring: str = "f1_macro",
                    n_jobs: int = cfg.N_JOBS) -> dict:
    cv = cv or make_cv()
    gs = GridSearchCV(conf["model"], conf["params"], cv=cv, scoring=scoring, n_jobs=n_jobs,
                      refit=True, return_train_score=True, error_score="raise")
    t0 = time.time()
    gs.fit(X, y, groups=groups, **_fit_params(conf, y))
    bi = gs.best_index_
    folds = [gs.cv_results_[f"split{i}_test_score"][bi] for i in range(cv.get_n_splits())]
    return {"model_name": name, "estimator": gs.best_estimator_, "best_params": gs.best_params_,
            "cv_mean": float(np.mean(folds)), "cv_std": float(np.std(folds)), "cv_folds": folds,
            "train_mean": float(gs.cv_results_["mean_train_score"][bi]),
            "fit_time_s": round(time.time() - t0, 1),
            "cv_results": pd.DataFrame(gs.cv_results_)}


def selected_features(estimator, feature_cols) -> list:
    if isinstance(estimator, Pipeline) and "selecteur" in estimator.named_steps:
        return list(np.asarray(feature_cols)[estimator.named_steps["selecteur"].get_support()])
    return list(feature_cols)


def run_experiment(train_df: pd.DataFrame, feature_cols: list, experiment: str, task: str = "general",
                   models=None, target: str = "target", save: bool = True, verbose: bool = True):
    """
    Lance la recherche d'hyperparamètres pour chaque modèle sur le TRAIN.
    Renvoie (tableau récapitulatif trié par score CV, dict des bundles).
    Aucun score de test n'est calculé ici.
    """
    configs = get_model_configs()
    if models is not None:
        configs = {k: v for k, v in configs.items() if k in models}
    le = LabelEncoder().fit(train_df[target])
    X = train_df[feature_cols]
    y = le.transform(train_df[target])
    groups = train_df["DogID"].to_numpy()

    bundles, rows = {}, []
    for name, conf in configs.items():
        if verbose:
            print(f"[{experiment}] {name} ...", end=" ", flush=True)
        res = run_grid_search(name, conf, X, y, groups)
        bundle = {"model": res["estimator"], "label_encoder": le, "features": list(feature_cols),
                  "task": task, "experiment": experiment, "model_name": name,
                  "best_params": res["best_params"], "cv_mean": res["cv_mean"], "cv_std": res["cv_std"],
                  "cv_folds": res["cv_folds"], "sample_weight": conf["sample_weight"]}
        bundles[name] = bundle
        n_sel = len(selected_features(res["estimator"], feature_cols))
        rows.append({"experience": experiment, "tache": task, "modele": name,
                     "f1_macro_cv_moy": res["cv_mean"], "f1_macro_cv_std": res["cv_std"],
                     "f1_macro_train_moy": res["train_mean"],
                     "n_variables": len(feature_cols), "n_variables_retenues": n_sel,
                     "meilleurs_params": res["best_params"], "duree_s": res["fit_time_s"]})
        if verbose:
            print(f"F1 macro CV = {res['cv_mean']:.3f} ± {res['cv_std']:.3f} ({res['fit_time_s']} s)")
        if save:
            save_bundle(bundle)
    summary = pd.DataFrame(rows).sort_values("f1_macro_cv_moy", ascending=False).reset_index(drop=True)
    return summary, bundles


def best_bundle(bundles: dict, exclude=DUMMY_MODELS) -> dict:
    """Meilleur modèle selon le score de VALIDATION CROISÉE (jamais le test)."""
    candidates = {k: b for k, b in bundles.items() if k not in exclude}
    return max(candidates.values(), key=lambda b: b["cv_mean"])


# ---------------------------------------------------------------------------
# Prédiction, sauvegarde
# ---------------------------------------------------------------------------
def predict(bundle: dict, df: pd.DataFrame):
    """Renvoie (labels prédits en texte, matrice de probabilités)."""
    X = df[bundle["features"]]
    model = bundle["model"]
    proba = model.predict_proba(X)
    pred = bundle["label_encoder"].inverse_transform(proba.argmax(axis=1))
    return pred, proba


def bundle_path(bundle: dict):
    folder = cfg.MODELS_DIR / bundle["task"] / bundle["experiment"]
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{bundle['model_name']}.joblib"


def save_bundle(bundle: dict) -> None:
    model = bundle["model"]
    if isinstance(model, Pipeline):
        model.set_params(memory=None)  # le cache disque n'a pas de sens hors de cette machine
    joblib.dump(bundle, bundle_path(bundle))


def load_bundle(task: str, experiment: str, model_name: str) -> dict:
    return joblib.load(cfg.MODELS_DIR / task / experiment / f"{model_name}.joblib")


def fit_clone(bundle: dict, df: pd.DataFrame, target: str = "target"):
    """Réentraîne une copie du modèle (mêmes hyperparamètres) sur un autre jeu."""
    model = clone(bundle["model"])
    y = bundle["label_encoder"].transform(df[target])
    params = {"clf__sample_weight": compute_sample_weight("balanced", y)} if bundle["sample_weight"] else {}
    model.fit(df[bundle["features"]], y, **params)
    return model


def oof_predict_proba(bundle: dict, train_df: pd.DataFrame, target: str = "target", n_jobs: int = cfg.N_JOBS):
    """
    Probabilités « hors pli » sur le TRAIN (chaque fenêtre est prédite par un
    modèle qui n'a pas vu son chien). Sert à régler des post-traitements
    (lissage, etc.) sans jamais toucher au test.
    """
    model = clone(bundle["model"])
    if isinstance(model, Pipeline):
        model.set_params(memory=str(cfg.CACHE_DIR))
    y = bundle["label_encoder"].transform(train_df[target])
    params = {"clf__sample_weight": compute_sample_weight("balanced", y)} if bundle["sample_weight"] else {}
    return cross_val_predict(model, train_df[bundle["features"]], y, groups=train_df["DogID"].to_numpy(),
                             cv=make_cv(), method="predict_proba", params=params, n_jobs=n_jobs)


def split_sensitivity(bundle: dict, task_df: pd.DataFrame, seeds=(0, 1, 2, 3, 4), target: str = "target"):
    """
    Analyse de sensibilité au split : on refait le split train/test par chien
    avec plusieurs graines et on réentraîne le modèle retenu (mêmes
    hyperparamètres). N'intervient dans AUCUN choix : sert à montrer la
    variabilité du score dû au faible nombre de chiens de test.
    """
    from sklearn.metrics import accuracy_score, f1_score
    from .data import split_dogs

    classes = set(bundle["label_encoder"].classes_)
    task_df = task_df[task_df[target].isin(classes)]
    rows = []
    for seed in seeds:
        tr_ids, te_ids = split_dogs(task_df["DogID"].unique(), seed=seed)
        tr, te = task_df[task_df["DogID"].isin(tr_ids)], task_df[task_df["DogID"].isin(te_ids)]
        if set(tr[target]) != classes:  # une classe absente du train : split inutilisable
            continue
        model = fit_clone(bundle, tr, target)
        pred = bundle["label_encoder"].inverse_transform(model.predict(te[bundle["features"]]))
        rows.append({"graine_split": seed, "chiens_test": sorted(int(d) for d in te_ids),
                     "f1_macro_test": f1_score(te[target], pred, average="macro"),
                     "exactitude_test": accuracy_score(te[target], pred)})
    return pd.DataFrame(rows)
