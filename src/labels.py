"""
Définition des tâches de classification et gestion des classes.

Tâches
------
* 'general'    : prédire Behavior_1 (n'importe quel comportement) ;
* 'posture', 'locomotion', 'activite' : prédire le comportement de la famille,
  qu'il soit annoté en Behavior_1, 2 ou 3 (classes mutuellement exclusives).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg

TASKS = ["general"] + list(cfg.FAMILIES)


def task_columns(task: str):
    """(colonne du label, colonne de pureté) associées à une tâche."""
    if task == "general":
        return "behavior_1", "purity_1"
    if task in cfg.FAMILIES:
        return f"{task}_label", f"{task}_purity"
    raise ValueError(f"Tâche inconnue : {task}")


def select_task_windows(windows: pd.DataFrame, task: str, min_purity: float = cfg.MIN_PURITY) -> pd.DataFrame:
    """
    Garde les fenêtres dont le label de la tâche couvre au moins `min_purity`
    du cœur de la fenêtre, et ajoute une colonne 'target'.
    Les fenêtres à cheval sur deux comportements (ou peu annotées) sont exclues :
    leur label serait ambigu.
    """
    lab, pur = task_columns(task)
    keep = windows[lab].notna() & (windows[pur] >= min_purity)
    out = windows.loc[keep].copy()
    out["target"] = out[lab].astype(str)
    return out


def filter_rare_classes(train: pd.DataFrame, test: pd.DataFrame,
                        min_count: int = cfg.MIN_WINDOWS_PER_CLASS, target: str = "target"):
    """
    Exclut les classes ayant moins de `min_count` fenêtres dans le TRAIN (le
    seuil est décidé sur le train uniquement), dans le train ET le test.

    Conséquence à annoncer explicitement : les scores ne portent que sur les
    classes conservées. Le tableau renvoyé liste chaque classe avec ses
    effectifs et son statut, pour le rapport.
    """
    n_tr = train[target].value_counts()
    n_te = test[target].value_counts()
    table = pd.DataFrame({"n_fenetres_train": n_tr, "n_fenetres_test": n_te}).fillna(0).astype(int)
    table["conservee"] = table["n_fenetres_train"] >= min_count
    table = table.sort_values("n_fenetres_train", ascending=False)
    kept = table.index[table["conservee"]].tolist()
    part_exclue = 1 - table.loc[kept, "n_fenetres_test"].sum() / max(table["n_fenetres_test"].sum(), 1)
    table.attrs["part_test_exclue"] = float(part_exclue)
    return train[train[target].isin(kept)].copy(), test[test[target].isin(kept)].copy(), kept, table


def cooccurrence(moves: pd.DataFrame, col_a: str = "Behavior_1", col_b: str = "Behavior_2") -> pd.DataFrame:
    """Nombre de lignes (centièmes de seconde) où deux labels sont simultanés."""
    sub = moves[[col_a, col_b]].dropna()
    return pd.crosstab(sub[col_a].astype(str), sub[col_b].astype(str))


def family_of(label: str):
    for fam, labs in cfg.FAMILIES.items():
        if label in labs:
            return fam
    return None
