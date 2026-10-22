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
def plot_session_signals(
    moves: pd.DataFrame,
    dogs: pd.DataFrame,
    dog_id: int,
    test_num: int,
    sensor: str = "ABack",
    behavior_col: str = "Behavior_1",
):
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
        ax1.plot(
            d["t_sec"], d[f"{sensor}_{a}"], label=f"{sensor}_{a}", alpha=0.7, lw=0.6
        )
    ax1.set_xlabel("Temps (s)")
    ax1.set_ylabel(
        "Accélération (g)" if sensor.startswith("A") else "Vitesse angulaire (°/s)"
    )
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    lab = d[["t_sec", behavior_col]].dropna()
    ax2.scatter(
        lab["t_sec"],
        lab[behavior_col].astype(str),
        color="red",
        s=4,
        label=behavior_col,
    )
    ax2.grid(False)
    ax1.set_title(
        f"Chien {dog_id} ({breed.iloc[0] if len(breed) else '?'}) - test {test_num} - {sensor}"
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Frise chronologique interactive
# ---------------------------------------------------------------------------
def plot_timeline(
    session_df: pd.DataFrame,
    pred_col: str = "pred",
    show=("behavior_1", "behavior_2", "behavior_3"),
    title: str = "",
):
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
    styles = {
        "behavior_1": ("#3498db", "circle", 11),
        "behavior_2": ("#f39c12", "square", 9),
        "behavior_3": ("#9b59b6", "diamond", 9),
    }
    fig = go.Figure()
    for col in show:
        if d[col].notna().any():
            color, symbol, size = styles[col]
            fig.add_trace(
                go.Scatter(
                    x=d["t_center"],
                    y=d[col],
                    mode="markers",
                    name=f"Réel ({col})",
                    marker=dict(color=color, symbol=symbol, size=size, opacity=0.5),
                )
            )
    hover = [
        f"<b>t = {t:.1f} s</b><br>Prédit : {p} ({'OK' if c else 'erreur'})<br>B1 : {b1}<br>B2 : {b2}<br>B3 : {b3}"
        for t, p, c, b1, b2, b3 in zip(
            d["t_center"],
            d[pred_col],
            d["correct"],
            d["behavior_1"],
            d["behavior_2"],
            d["behavior_3"],
        )
    ]
    fig.add_trace(
        go.Scatter(
            x=d["t_center"],
            y=d[pred_col],
            mode="markers",
            name="Prédit",
            marker=dict(
                color=np.where(d["correct"], "green", "red"),
                symbol=np.where(d["correct"], "circle", "x"),
                size=7,
            ),
            text=hover,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Temps (s)",
        yaxis_title="Comportement",
        height=550,
        template="plotly_white",
        hovermode="closest",
    )
    return fig


# ---------------------------------------------------------------------------
# UMAP des erreurs (bonus 2)
# ---------------------------------------------------------------------------
def umap_embedding(
    X, n_neighbors: int = 30, min_dist: float = 0.1, seed: int = cfg.RANDOM_STATE
):
    """
    Projection 2D robuste des fenêtres pour l'analyse des erreurs :
    1. valeurs non finies remplacées par la médiane de la colonne ;
    2. standardisation puis écrêtage à ±10 écarts-types (valeurs extrêmes) ;
    3. ACP à 50 composantes (pratique standard avant UMAP : moins de bruit, plus rapide) ;
    4. UMAP avec initialisation aléatoire ; si le résultat n'est pas fini, repli sur t-SNE.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=np.float64)
    n_bad = int((~np.isfinite(X)).sum())
    if n_bad:
        print(
            f"{n_bad} valeurs non finies dans les variables : remplacées par la médiane de la colonne"
        )
        X = np.where(np.isfinite(X), X, np.nan)
        med = np.nanmedian(X, axis=0)
        X = np.where(np.isnan(X), np.nan_to_num(med)[None, :], X)
    Xs = np.clip(StandardScaler().fit_transform(X), -10, 10)
    Xp = PCA(n_components=min(50, Xs.shape[1]), random_state=seed).fit_transform(Xs)

    emb = None
    try:
        import umap

        emb = umap.UMAP(
            n_neighbors=n_neighbors, min_dist=min_dist, init="random", random_state=seed
        ).fit_transform(Xp)
    except Exception as e:
        print("UMAP a échoué :", repr(e))
    if emb is None or not np.isfinite(emb).all():
        print("Projection UMAP non exploitable : repli sur t-SNE (quelques minutes)")
        from sklearn.manifold import TSNE

        emb = TSNE(
            n_components=2, perplexity=30, init="pca", random_state=seed
        ).fit_transform(Xp)
    return emb


def plot_umap_errors(emb, y_true, y_pred, dog_ids=None):
    """Projection UMAP : comportement réel, erreurs, chien (matplotlib pur, légendes fiables)."""
    emb = np.asarray(emb, dtype=float)
    y_true, y_pred = np.asarray(y_true).astype(str), np.asarray(y_pred).astype(str)
    ok = np.isfinite(emb).all(axis=1)
    if not ok.all():
        print(f"Attention : {(~ok).sum()} points non finis retirés de la projection")
    emb, y_true, y_pred = emb[ok], y_true[ok], y_pred[ok]
    dogs = None if dog_ids is None else np.asarray(dog_ids).astype(str)[ok]
    if not ok.any():
        raise ValueError(
            "Projection entièrement non finie : recalculer emb avec viz.umap_embedding."
        )
    err = y_true != y_pred

    # cadrage robuste : quelques points isolés ne doivent pas écraser le nuage principal
    lo, hi = np.percentile(emb, [0.5, 99.5], axis=0)
    marge = 0.05 * (hi - lo)

    def _par_groupe(ax, groupes, titre, cmap, titre_legende):
        base = plt.get_cmap(cmap)
        for i, v in enumerate(sorted(np.unique(groupes))):
            m = groupes == v
            ax.scatter(
                emb[m, 0],
                emb[m, 1],
                s=3,
                color=base(i % base.N),
                label=v,
                rasterized=True,
            )
        ax.set_title(titre)
        ax.legend(
            fontsize=7,
            markerscale=4,
            title=titre_legende,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )

    n_cols = 3 if dogs is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(7.5 * n_cols, 6))
    _par_groupe(
        axes[0],
        y_true,
        "Fenêtres de test par comportement réel",
        "tab20",
        "Comportement",
    )
    axes[1].scatter(
        emb[~err, 0], emb[~err, 1], s=3, c="lightgrey", label="correct", rasterized=True
    )
    axes[1].scatter(
        emb[err, 0], emb[err, 1], s=3, c="red", label="erreur", rasterized=True
    )
    axes[1].set_title(f"Erreurs ({err.mean():.1%} des fenêtres)")
    axes[1].legend(markerscale=4)
    if dogs is not None:
        _par_groupe(
            axes[2], dogs, "Fenêtres par chien (effet individu ?)", "tab10", "DogID"
        )
    for ax in axes:
        ax.set_xlim(lo[0] - marge[0], hi[0] + marge[0])
        ax.set_ylim(lo[1] - marge[1], hi[1] + marge[1])
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return fig
