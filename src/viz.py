"""Visualisations : signaux bruts, frises chronologiques, UMAP des erreurs."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config as cfg

sns.set_theme(style="whitegrid")


def savefig(fig, name: str) -> None:
    fig.savefig(cfg.FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Exploration
# ---------------------------------------------------------------------------
def plot_session_signals(moves: pd.DataFrame, dogs: pd.DataFrame, dog_id: int, test_num: int,
                         sensor: str = "ABack", behavior_col: str = "Behavior_1"):
    """
    Signaux d'un capteur + comportement annoté, pour une séance.
    On filtre AVANT toute jointure (l'ancienne version fusionnait les 10 M de
    lignes avec df_dogs à chaque changement du menu déroulant).
    """
    d = moves[(moves["DogID"] == dog_id) & (moves["TestNum"] == test_num)]
    if d.empty:
        print(f"Aucune donnée pour le chien {dog_id}, test {test_num}.")
        return None
    breed = dogs.loc[dogs["DogID"] == dog_id, "Breed"]
    fig, ax1 = plt.subplots(figsize=(15, 6))
    for a in cfg.AXES:
        ax1.plot(d["t_sec"], d[f"{sensor}_{a}"], label=f"{sensor}_{a}", alpha=0.7, lw=0.6)
    ax1.set_xlabel("Temps (s)")
    ax1.set_ylabel("Accélération (g)" if sensor.startswith("A") else "Vitesse angulaire (°/s)")
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    lab = d[["t_sec", behavior_col]].dropna()
    ax2.scatter(lab["t_sec"], lab[behavior_col].astype(str), color="red", s=4, label=behavior_col)
    ax2.grid(False)
    ax1.set_title(f"Chien {dog_id} ({breed.iloc[0] if len(breed) else '?'}) - test {test_num} - {sensor}")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Frise chronologique interactive
# ---------------------------------------------------------------------------
def plot_timeline(session_df: pd.DataFrame, pred_col: str = "pred", show=("behavior_1", "behavior_2", "behavior_3"),
                  title: str = ""):
    """
    Frise d'une séance : vrais comportements (Behavior_1/2/3, alignés car issus
    du fenêtrage) et prédictions colorées (vert = dans Behavior_1/2/3, rouge
    sinon). L'axe horizontal est le VRAI temps (t_center) : les fenêtres
    exclues apparaissent comme des trous, au lieu d'être recollées.
    """
    import plotly.graph_objects as go
    from .evaluation import flexible_correct

    d = session_df.sort_values("t_center").copy()
    d["correct"] = flexible_correct(d, pred_col)
    styles = {"behavior_1": ("#3498db", "circle", 11), "behavior_2": ("#f39c12", "square", 9),
              "behavior_3": ("#9b59b6", "diamond", 9)}
    fig = go.Figure()
    for col in show:
        if d[col].notna().any():
            color, symbol, size = styles[col]
            fig.add_trace(go.Scatter(x=d["t_center"], y=d[col], mode="markers", name=f"Réel ({col})",
                                     marker=dict(color=color, symbol=symbol, size=size, opacity=0.5)))
    hover = [f"<b>t = {t:.1f} s</b><br>Prédit : {p} ({'OK' if c else 'erreur'})<br>B1 : {b1}<br>B2 : {b2}<br>B3 : {b3}"
             for t, p, c, b1, b2, b3 in zip(d["t_center"], d[pred_col], d["correct"],
                                            d["behavior_1"], d["behavior_2"], d["behavior_3"])]
    fig.add_trace(go.Scatter(
        x=d["t_center"], y=d[pred_col], mode="markers", name="Prédit",
        marker=dict(color=np.where(d["correct"], "green", "red"),
                    symbol=np.where(d["correct"], "circle", "x"), size=7),
        text=hover, hoverinfo="text"))
    fig.update_layout(title=title, xaxis_title="Temps (s)", yaxis_title="Comportement",
                      height=550, template="plotly_white", hovermode="closest")
    return fig


# ---------------------------------------------------------------------------
# UMAP des erreurs (bonus 2)
# ---------------------------------------------------------------------------
def umap_embedding(X: np.ndarray, n_neighbors: int = 30, min_dist: float = 0.1, seed: int = cfg.RANDOM_STATE):
    """
    Projection 2D des fenêtres (variables standardisées). Sert à l'ANALYSE des
    erreurs : les erreurs sont-elles à la frontière entre deux nuages de classes
    (confusion intrinsèque) ou au cœur d'un nuage (label douteux, chien atypique) ?
    """
    import umap
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(X)
    return umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed).fit_transform(Xs)


def plot_umap_errors(emb: np.ndarray, y_true, y_pred, dog_ids=None):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    err = y_true != y_pred
    n_cols = 3 if dog_ids is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6))
    sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=y_true, s=4, linewidth=0, ax=axes[0], palette="tab20")
    axes[0].set_title("Fenêtres de test colorées par comportement réel")
    axes[0].legend(fontsize=7, markerscale=3, loc="best")
    axes[1].scatter(emb[~err, 0], emb[~err, 1], s=3, c="lightgrey", label="correct")
    axes[1].scatter(emb[err, 0], emb[err, 1], s=3, c="red", label="erreur")
    axes[1].set_title(f"Erreurs ({err.mean():.1%} des fenêtres)")
    axes[1].legend(markerscale=4)
    if dog_ids is not None:
        sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=np.asarray(dog_ids).astype(str), s=4, linewidth=0,
                        ax=axes[2], palette="tab10")
        axes[2].set_title("Fenêtres colorées par chien (effet individu ?)")
        axes[2].legend(fontsize=7, markerscale=3, title="DogID")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    return fig
