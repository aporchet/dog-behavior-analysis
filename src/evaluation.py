"""
Métriques, rapports et figures d'évaluation.

Vocabulaire : en français, « précision » désigne la
precision (VP / (VP + FP)). Le taux de bonnes réponses s'appelle l'EXACTITUDE
(accuracy).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ---------------------------------------------------------------------------
# Métriques globales
# ---------------------------------------------------------------------------
def _effective_labels(y_true, y_pred, labels):
    if labels is None:
        return None
    present = set(np.asarray(y_true)) | set(np.asarray(y_pred))
    return [lab for lab in labels if lab in present]


def compute_metrics(y_true, y_pred, labels=None) -> dict:
    """
    - exactitude : sensible au déséquilibre des classes ;
    - exactitude équilibrée = rappel macro : moyenne des rappels par classe ;
    - F1 macro : métrique principale du projet (chaque classe compte autant).

    `labels` restreint les classes prises en compte (ex. une famille). Les
    classes absentes à la fois des vraies valeurs et des prédictions sont
    ignorées : une classe sans aucune fenêtre de test (fréquent avec 9 chiens
    de test) ne doit pas compter pour un F1 nul.
    """
    labels = _effective_labels(y_true, y_pred, labels)
    kw = {"labels": labels, "zero_division": 0}
    return {
        "exactitude": accuracy_score(y_true, y_pred),
        "exactitude_equilibree": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", **kw),
        "f1_ponderee": f1_score(y_true, y_pred, average="weighted", **kw),
        "precision_macro": precision_score(y_true, y_pred, average="macro", **kw),
        "rappel_macro": recall_score(y_true, y_pred, average="macro", **kw),
        "n_fenetres": len(y_true),
    }


def named_report(y_true, y_pred, labels) -> pd.DataFrame:
    """Rapport de classification avec les VRAIS noms de comportements."""
    labels = _effective_labels(y_true, y_pred, labels)
    rep = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    df = pd.DataFrame(rep).T
    df["support"] = df["support"].astype(int)
    return df.round(3)


def plot_confusion(
    y_true, y_pred, labels, ax=None, title: str = "", normalize: str = "true"
):
    """Matrice de confusion normalisée par ligne : la diagonale = rappel par classe."""
    labels = _effective_labels(y_true, y_pred, labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    if ax is None:
        _, ax = plt.subplots(figsize=(1 + 0.7 * len(labels), 0.6 * len(labels) + 1))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_xlabel("Comportement prédit")
    ax.set_ylabel("Comportement réel")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    return ax


# ---------------------------------------------------------------------------
# Scores par chien et par profil
# ---------------------------------------------------------------------------
def per_dog_scores(
    df: pd.DataFrame, pred_col: str = "pred", target: str = "target"
) -> pd.DataFrame:
    """
    Un score par chien de test. C'est l'unité statistique pertinente : les
    fenêtres d'un même chien sont fortement corrélées. La dispersion entre
    chiens montre la fiabilité réelle du score global.
    """
    rows = []
    for dog, g in df.groupby("DogID"):
        rows.append(
            {
                "DogID": dog,
                "n_fenetres": len(g),
                "exactitude": accuracy_score(g[target], g[pred_col]),
                "f1_macro": f1_score(
                    g[target], g[pred_col], average="macro", zero_division=0
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_per_dog(
    per_dog: pd.DataFrame, metric: str = "f1_macro", ax=None, title: str = ""
):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    sns.boxplot(y=per_dog[metric], ax=ax, color="#9ecae1", width=0.4)
    sns.stripplot(y=per_dog[metric], ax=ax, color="black", size=6)
    for _, r in per_dog.iterrows():
        ax.annotate(str(int(r["DogID"])), (0.05, r[metric]), fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} par chien de test (n = {len(per_dog)})")
    return ax


def profile_table(
    per_dog: pd.DataFrame, dogs: pd.DataFrame, by: str, metric: str = "f1_macro"
) -> pd.DataFrame:
    """Score moyen par catégorie de profil, avec le nombre de chiens."""
    m = per_dog.merge(dogs[["DogID", by]], on="DogID", how="left")
    return (
        m.groupby(by, observed=True)[metric]
        .agg(n_chiens="count", moyenne="mean", ecart_type="std")
        .round(3)
        .reset_index()
    )


def plot_profiles(
    per_dog: pd.DataFrame,
    dogs: pd.DataFrame,
    by_cols=("Classe_Poids", "Classe_Age", "Gender", "NeuteringStatus"),
    metric: str = "f1_macro",
):
    """Un point = un chien. Lecture exploratoire uniquement (car effectifs très faibles)."""
    m = per_dog.merge(dogs, on="DogID", how="left")
    fig, axes = plt.subplots(
        1, len(by_cols), figsize=(4.5 * len(by_cols), 4), sharey=True
    )
    for ax, col in zip(np.atleast_1d(axes), by_cols):
        # Seules les catégories présentes parmi les chiens de test sont affichées, dans leur ordre naturel
        present = m[col].dropna().unique()
        if isinstance(m[col].dtype, pd.CategoricalDtype):
            order = [str(c) for c in m[col].cat.categories if c in present]
        else:
            order = [str(c) for c in sorted(present)]
        d = m.assign(**{col: m[col].astype(str)})
        sns.stripplot(
            data=d, x=col, y=metric, order=order, ax=ax, size=8, color="#3182bd"
        )
        sns.pointplot(
            data=d,
            x=col,
            y=metric,
            order=order,
            ax=ax,
            color="black",
            errorbar=None,
            markers="_",
            linestyles="none",
        )
        counts = d[col].value_counts()
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"{k}\n(n={counts[k]})" for k in order], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_xlabel("")
        ax.set_title(col)
    fig.suptitle(f"{metric} par profil de chien (1 point = 1 chien de test)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Évaluation « flexible »
# ---------------------------------------------------------------------------
def flexible_correct(df: pd.DataFrame, pred_col: str = "pred") -> pd.Series:
    """
    Vrai si la prédiction figure dans Behavior_1, 2 OU 3 de la fenêtre.
    Métrique complémentaire, plus indulgente que la métrique officielle :
    elle ne doit pas être comparée au score d'entraînement (qui vise Behavior_1).
    Les Behavior_2/3 sont ceux calculés lors du fenêtrage : ils sont alignés.
    """
    cols = ["behavior_1", "behavior_2", "behavior_3"]
    ok = np.zeros(len(df), dtype=bool)
    for c in cols:
        ok |= (df[c].astype(object) == df[pred_col].astype(object)).to_numpy()
    return pd.Series(ok, index=df.index)


# ---------------------------------------------------------------------------
# Comparaison modèle général / modèle spécifique
# ---------------------------------------------------------------------------
def compare_general_specific(
    general_pred: pd.DataFrame, specific_pred: pd.DataFrame, family_classes
) -> dict:
    """
    Comparaison à données identiques : fenêtres de test où le comportement de la
    famille est aussi le comportement principal (Behavior_1), et où les deux
    modèles ont produit une prédiction. Le F1 macro est calculé sur les classes
    de la famille : une prédiction du modèle général hors famille (ex. « Panting »
    au lieu de « Standing ») compte comme une erreur.

    general_pred / specific_pred : colonnes window_id, target, pred.
    """
    g = general_pred[["window_id", "pred"]].rename(columns={"pred": "pred_general"})
    s = specific_pred[["window_id", "target", "pred", "behavior_1"]].rename(
        columns={"pred": "pred_specifique"}
    )
    common = s.merge(g, on="window_id", how="inner")
    common = common[common["behavior_1"] == common["target"]]
    # Classes évaluées = classes de la famille présentes dans la vérité des fenêtres communes.
    # Sans cette restriction, une classe connue du seul modèle spécifique (ex. Galloping, exclue
    # du modèle général comme classe rare) mais absente des fenêtres communes compte pour un F1
    # nul dès qu'elle est prédite une fois : le F1 macro du spécifique est alors injustement
    # écrasé. Une prédiction erronée de cette classe reste comptée comme une erreur (rappel de
    # la vraie classe).
    present = set(common["target"])
    labels = [c for c in family_classes if c in present]
    out = {"n_fenetres_communes": len(common), "classes_evaluees": ", ".join(labels)}
    for who in ("general", "specifique"):
        m = compute_metrics(common["target"], common[f"pred_{who}"], labels=labels)
        out[f"f1_macro_{who}"] = m["f1_macro"]
        out[f"exactitude_{who}"] = m["exactitude"]
    out["n_fenetres_specifique_total"] = len(specific_pred)
    out["part_couverte_seulement_par_specifique"] = 1 - len(common) / max(
        len(specific_pred), 1
    )
    return out
