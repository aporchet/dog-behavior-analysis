"""
Chargement, nettoyage et mise en cache des données ; split train/test par chien.

Le CSV brut (~10,6 M de lignes) est lu une seule fois puis enregistré au format
parquet (types compacts : float32, catégories) : les notebooks suivants le
rechargent en quelques secondes.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from . import config as cfg


# ---------------------------------------------------------------------------
# Entrées / sorties tabulaires
# ---------------------------------------------------------------------------
def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


def save_table(df: pd.DataFrame, path) -> None:
    """Sauvegarde en parquet (repli en pickle si pyarrow est absent)."""
    if _parquet_available():
        df.to_parquet(path, index=False)
    else:
        warnings.warn("pyarrow absent : sauvegarde en pickle à la place du parquet.")
        df.to_pickle(path)


def load_table(path, columns=None) -> pd.DataFrame:
    if _parquet_available():
        return pd.read_parquet(path, columns=columns)
    df = pd.read_pickle(path)
    return df[columns] if columns is not None else df


# ---------------------------------------------------------------------------
# Construction du cache
# ---------------------------------------------------------------------------
def _read_raw_moves() -> pd.DataFrame:
    dtypes = {c: "float32" for c in cfg.SENSOR_COLUMNS}
    dtypes.update(
        {c: "category" for c in cfg.BEHAVIOR_COLUMNS + ["Task", "PointEvent"]}
    )
    dtypes.update({"DogID": "int32", "TestNum": "int8", "t_sec": "float64"})
    return pd.read_csv(
        cfg.RAW_MOVE_FILE, na_values=cfg.NA_VALUES, dtype=dtypes, low_memory=False
    )


def clean_moves(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage minimal et justifiable :
    1. tri chronologique par séance (DogID, TestNum, t_sec) ;
    2. les étiquettes techniques de synchronisation deviennent des valeurs
       manquantes. On conserve les lignes car les supprimer créerait des « trous »
       dans le signal que le fenêtrage enjamberait sans le savoir.
       Les fenêtres non annotées sont filtrées plus tard.
    """
    df = df.sort_values(cfg.SESSION_KEYS + ["t_sec"], kind="stable").reset_index(
        drop=True
    )
    for col in cfg.BEHAVIOR_COLUMNS:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            to_remove = [
                lab for lab in cfg.TECHNICAL_LABELS if lab in df[col].cat.categories
            ]
            if to_remove:
                df[col] = df[col].cat.remove_categories(to_remove)
        else:
            df[col] = df[col].where(~df[col].isin(cfg.TECHNICAL_LABELS))
    return df


def build_cache(force: bool = False) -> None:
    """Lit les fichiers bruts, nettoie et écrit le cache dans data/interim/."""
    if cfg.MOVES_CACHE.exists() and cfg.DOGS_CACHE.exists() and not force:
        print("Cache déjà présent (force=True pour le reconstruire).")
        return
    print("Lecture du CSV brut (peut prendre une minute)...")
    moves = clean_moves(_read_raw_moves())
    save_table(moves, cfg.MOVES_CACHE)
    dogs = pd.read_excel(cfg.RAW_INFO_FILE)
    save_table(dogs, cfg.DOGS_CACHE)
    print(f"Cache écrit : {len(moves):,} lignes de mouvement, {len(dogs)} chiens.")


def load_moves(columns=None) -> pd.DataFrame:
    """Charge les séries temporelles nettoyées (depuis le cache)."""
    if not cfg.MOVES_CACHE.exists():
        build_cache()
    return load_table(cfg.MOVES_CACHE, columns=columns)


def load_dogs() -> pd.DataFrame:
    """
    Charge les informations fixes des chiens et ajoute deux variables
    catégorielles dérivées, utilisées uniquement pour l'analyse des
    performances par profil (pas comme variables d'entrée : elles sont des
    fonctions déterministes de Weight et Age months, donc redondantes).
    """
    if not cfg.DOGS_CACHE.exists():
        build_cache()
    dogs = load_table(cfg.DOGS_CACHE)
    dogs["Classe_Poids"] = pd.cut(
        dogs["Weight"], bins=cfg.WEIGHT_BINS, labels=cfg.WEIGHT_LABELS, right=False
    )
    dogs["Classe_Age"] = pd.cut(
        dogs["Age months"], bins=cfg.AGE_BINS, labels=cfg.AGE_LABELS, right=False
    )
    return dogs


# ---------------------------------------------------------------------------
# Contrôles qualité
# ---------------------------------------------------------------------------
def continuity_report(moves: pd.DataFrame) -> pd.DataFrame:
    """Durée, nombre de ruptures temporelles et plus grand trou par séance d'enregistrement."""
    expected_dt = 1.0 / cfg.SAMPLING_RATE
    rows = []
    for (dog, test), t in moves.groupby(cfg.SESSION_KEYS, sort=True)["t_sec"]:
        dt = np.diff(t.to_numpy())
        rows.append(
            {
                "DogID": dog,
                "TestNum": test,
                "n_lignes": len(t),
                "duree_min": round((t.iloc[-1] - t.iloc[0]) / 60, 2),
                "n_ruptures": int(np.sum((dt > 1.5 * expected_dt) | (dt <= 0))),
                "plus_grand_trou_s": float(dt.max()) if len(dt) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def label_coverage(moves: pd.DataFrame) -> pd.DataFrame:
    """Durée et pourcentage du temps annoté (Behavior_1/2/3) par séance."""
    g = moves.groupby(cfg.SESSION_KEYS, sort=True, observed=True)
    out = g.size().rename("n_lignes").to_frame()
    out["duree_min"] = (out["n_lignes"] / cfg.SAMPLING_RATE / 60).round(2)
    for i, col in enumerate(cfg.BEHAVIOR_COLUMNS, start=1):
        out[f"couverture_B{i}_%"] = (
            g[col].apply(lambda s: s.notna().mean()) * 100
        ).round(2)
    return out.reset_index()


# ---------------------------------------------------------------------------
# Split train / test par chien
# ---------------------------------------------------------------------------
def split_dogs(dog_ids, test_size: float = cfg.TEST_SIZE, seed: int = cfg.SPLIT_SEED):
    """
    Split par CHIEN : toutes les fenêtres d'un chien sont soit en train soit en
    test. Sans cela, des fenêtres voisines (donc corrélées) du même chien se
    retrouveraient de part et d'autre et gonfleraient les scores.
    """
    dog_ids = np.sort(np.unique(np.asarray(dog_ids)))
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(dog_ids, groups=dog_ids))
    return dog_ids[tr].tolist(), dog_ids[te].tolist()


def save_split(train_ids, test_ids, seed: int = cfg.SPLIT_SEED) -> None:
    payload = {
        "seed": seed,
        "test_size": cfg.TEST_SIZE,
        "train_dogs": [int(x) for x in train_ids],
        "test_dogs": [int(x) for x in test_ids],
    }
    cfg.SPLIT_FILE.write_text(json.dumps(payload, indent=2))


def load_split():
    """Renvoie (train_ids, test_ids) tels que figés par le notebook 02."""
    if not cfg.SPLIT_FILE.exists():
        raise FileNotFoundError(
            "Split introuvable : exécuter d'abord le notebook 02_split_train_test."
        )
    payload = json.loads(cfg.SPLIT_FILE.read_text())
    return payload["train_dogs"], payload["test_dogs"]
