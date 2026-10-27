"""
Fonctions d'affichage du notebook de présentation (08).

Aucun entraînement ici. On relit ce que les notebooks 00 à 07 ont
enregistré (tableaux CSV, figures PNG, prédictions, modèles) et on ne fait que
des calculs légers (prédictions de modèles déjà entraînés, métriques, lissage).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as cfg

# Palette catégorielle (ordre fixe) + neutres
BLEU, ORANGE, VERT, JAUNE = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRIS, GRIS_CLAIR = "#8a8984", "#d4d3ce"
COULEURS_MODELES = {"LogisticRegression": BLEU, "RandomForest": VERT, "XGBoost": ORANGE}

# Noms objectifs des expériences (les noms techniques restent ceux des fichiers)
NOMS_EXPERIENCES = {
    "ancienne_baseline_1s": "E1 · moy. + écart-type, 1 s",
    "complet_1s": "E2 · variables complètes, 1 s",
    "multi_echelle": "E3 · multi-échelle 1-2-4 s",
    "multi_echelle+profil": "E4 · multi-échelle + profil",
}
NOMS_TSFRESH = {
    "approche1_sous_echantillon": "Fenêtres E3 (sous-échantillon)",
    "tsfresh_1s": "tsfresh seul (4 normes, 1 s)",
    "approche1+tsfresh": "Fenêtres E3 + tsfresh",
}


# ---------------------------------------------------------------------------
# Relecture des résultats enregistrés
# ---------------------------------------------------------------------------
def figure(nom: str, width: int = 1000):
    """Affiche une figure enregistrée par un notebook précédent (reports/figures)."""
    from IPython.display import Image, display

    path = cfg.FIGURES_DIR / f"{nom}.png"
    if path.exists():
        display(Image(filename=str(path), width=width))
    else:
        print(f"Figure absente : {path.name} (exécuter le notebook correspondant)")


def table(nom: str) -> pd.DataFrame:
    return pd.read_csv(cfg.TABLES_DIR / f"{nom}.csv")


def load_windows_for_dogs(dog_ids) -> pd.DataFrame:
    """Fenêtres de quelques chiens seulement (lecture filtrée du parquet si possible)."""
    from . import data

    dog_ids = [int(d) for d in dog_ids]
    try:
        return pd.read_parquet(cfg.WINDOWS_FILE, filters=[("DogID", "in", dog_ids)])
    except Exception:
        w = data.load_table(cfg.WINDOWS_FILE)
        return w[w["DogID"].isin(dog_ids)].copy()


def durees(df: pd.DataFrame) -> str:
    s = float(np.nansum(df["duree_s"]))
    return (
        f"{int(s // 3600)} h {int(s % 3600 // 60):02d} min"
        if s >= 3600
        else f"{int(s // 60)} min {int(s % 60):02d} s"
    )


# ---------------------------------------------------------------------------
# 1. Exploration
# ---------------------------------------------------------------------------
def plot_profils(dogs: pd.DataFrame):
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.6))
    axes[0].hist(dogs["Weight"], bins=10, color=BLEU, edgecolor="white")
    axes[0].set_title("Poids (kg)")
    axes[1].hist(dogs["Age months"], bins=10, color=BLEU, edgecolor="white")
    axes[1].set_title("Âge (mois)")
    for ax, col, titre in [
        (axes[2], "Gender", "Genre (codé 1 / 2)"),
        (axes[3], "NeuteringStatus", "Stérilisé (0 / 1)"),
    ]:
        v = dogs[col].value_counts().sort_index()
        ax.bar(v.index.astype(str), v.values, color=BLEU, width=0.5)
        ax.set_title(titre)
    for ax in axes:
        ax.set_ylabel("nb de chiens")
    fig.suptitle(
        f"{len(dogs)} chiens, {dogs['Breed'].nunique()} races (race inexploitable : presque une par chien)",
        y=1.03,
    )
    fig.tight_layout()
    return fig


def plot_frequences(moves: pd.DataFrame, table_classes: pd.DataFrame):
    """Durée annotée par comportement, en distinguant classes conservées / exclues (< 500 fenêtres train)."""
    minutes = (
        moves["Behavior_1"].value_counts() / cfg.SAMPLING_RATE / 60
    ).sort_values()
    chiens = (
        moves.dropna(subset=["Behavior_1"])
        .groupby("Behavior_1", observed=True)["DogID"]
        .nunique()
    )
    gardees = set(table_classes.index[table_classes["conservee"]])
    fig, ax = plt.subplots(figsize=(10, 6))
    couleurs = [BLEU if c in gardees else GRIS_CLAIR for c in minutes.index]
    ax.barh(minutes.index.astype(str), minutes.values, color=couleurs)
    for i, (c, v) in enumerate(minutes.items()):
        ax.text(
            v,
            i,
            f"  {v:.0f} min · {chiens.get(c, 0)} chiens",
            va="center",
            fontsize=8,
            color="#52514e",
        )
    ax.set_xlabel("Durée annotée en Behavior_1 (minutes, tous chiens)")
    ax.set_xlim(0, minutes.max() * 1.35)
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color=BLEU, label="classe conservée"),
            Patch(color=GRIS_CLAIR, label="classe exclue (< 500 fenêtres de train)"),
        ],
        loc="lower right",
    )
    ax.set_title(
        "Classes très déséquilibrées : de quelques secondes à plusieurs heures"
    )
    fig.tight_layout()
    return fig


def plot_couverture(coverage: pd.DataFrame):
    c = coverage.sort_values("couverture_B1_%")
    lab = c["DogID"].astype(str) + "-" + c["TestNum"].astype(str)
    fig, ax = plt.subplots(figsize=(16, 3.8))
    ax.bar(lab, c["couverture_B1_%"], color=BLEU, label="Behavior_1")
    ax.bar(lab, c["couverture_B2_%"], color=ORANGE, width=0.45, label="Behavior_2")
    ax.axhline(c["couverture_B1_%"].mean(), color=GRIS, ls="--", lw=1)
    ax.text(
        len(c) - 0.5,
        c["couverture_B1_%"].mean() + 1.5,
        f"moyenne B1 : {c['couverture_B1_%'].mean():.0f} %",
        fontsize=8,
        ha="right",
    )
    ax.set_ylabel("% du temps annoté")
    ax.set_xlabel("séance (chien - test)")
    ax.set_title(
        "Couverture des annotations par séance (le reste = non annoté, exclu de la tâche)"
    )
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Split
# ---------------------------------------------------------------------------
def plot_split(moves: pd.DataFrame, dogs: pd.DataFrame, train_ids, test_ids):
    m = moves.dropna(subset=["Behavior_1"])
    jeu = np.where(m["DogID"].isin(test_ids), "Test", "Train")
    prop = (
        pd.crosstab(m["Behavior_1"].astype(str), jeu, normalize="columns") * 100
    ).sort_values("Train", ascending=False)
    d = dogs[dogs["DogID"].isin(list(train_ids) + list(test_ids))].copy()
    d["Set"] = np.where(d["DogID"].isin(test_ids), "Test", "Train")

    fig, axes = plt.subplots(
        1, 3, figsize=(18, 4.5), gridspec_kw={"width_ratios": [3, 1, 1]}
    )
    x = np.arange(len(prop))
    axes[0].bar(
        x - 0.2,
        prop["Train"],
        0.4,
        color=BLEU,
        label=f"Train ({len(train_ids)} chiens)",
    )
    axes[0].bar(
        x + 0.2, prop["Test"], 0.4, color=ORANGE, label=f"Test ({len(test_ids)} chiens)"
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(prop.index, rotation=45, ha="right")
    axes[0].set_ylabel("% du temps annoté")
    axes[0].set_title("Comportements : train vs test")
    axes[0].legend()
    for ax, col in zip(axes[1:], ["Weight", "Age months"]):
        for i, s in enumerate(["Train", "Test"]):
            v = d.loc[d["Set"] == s, col]
            ax.boxplot(v, positions=[i], widths=0.45, showfliers=False)
            ax.scatter(
                np.full(len(v), i)
                + np.random.default_rng(0).uniform(-0.1, 0.1, len(v)),
                v,
                s=14,
                color=[BLEU, ORANGE][i],
                zorder=3,
            )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Train", "Test"])
        ax.set_title({"Weight": "Poids (kg)", "Age months": "Âge (mois)"}[col])
    fig.tight_layout()
    return fig, prop


# ---------------------------------------------------------------------------
# 4. Approche par fenêtres
# ---------------------------------------------------------------------------
def plot_cv_experiences(cv: pd.DataFrame):
    cv = cv.copy()
    cv["exp"] = cv["experience"].map(NOMS_EXPERIENCES)
    exps = list(NOMS_EXPERIENCES.values())
    fig, ax = plt.subplots(figsize=(13, 4.8))
    x = np.arange(len(exps))
    for i, (mod, col) in enumerate(COULEURS_MODELES.items()):
        s = cv[cv["modele"] == mod].set_index("exp").loc[exps]
        ax.bar(
            x + (i - 1) * 0.26,
            s["f1_macro_cv_moy"],
            0.24,
            yerr=s["f1_macro_cv_std"],
            capsize=3,
            color=col,
            label=mod,
        )
    dummy = cv[cv["modele"] == "Dummy_stratifie"]["f1_macro_cv_moy"].max()
    ax.axhline(dummy, color=GRIS, ls="--", lw=1)
    ax.text(
        len(exps) - 0.5,
        dummy + 0.015,
        f"hasard stratifié ({dummy:.2f})",
        ha="right",
        fontsize=8,
        color="#52514e",
    )
    best = cv.loc[cv[~cv["modele"].str.startswith("Dummy")]["f1_macro_cv_moy"].idxmax()]
    ax.annotate(
        f"retenu : {best['f1_macro_cv_moy']:.3f}",
        (exps.index(best["exp"]) + 0.26, best["f1_macro_cv_moy"]),
        xytext=(0, 18),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        arrowprops=dict(arrowstyle="-", color="#52514e"),
    )
    ax.set_xticks(x)
    ax.set_xticklabels(exps)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1 macro (CV 5 plis par chien, ± écart-type)")
    ax.set_title("Validation croisée sur le train : seul critère de choix du modèle")
    ax.legend(loc="upper left", ncol=3)
    fig.tight_layout()
    return fig


def plot_surapprentissage(
    cv: pd.DataFrame, f1_test: dict | None = None, experience: str = "multi_echelle"
):
    """Train vs CV (vs test pour le modèle retenu) : l'écart train - CV mesure le surapprentissage."""
    s = cv[
        (cv["experience"] == experience) & ~cv["modele"].str.startswith("Dummy")
    ].set_index("modele")
    mods = [m for m in COULEURS_MODELES if m in s.index]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, m in enumerate(mods):
        tr, va = s.loc[m, "f1_macro_train_moy"], s.loc[m, "f1_macro_cv_moy"]
        ax.plot([va, tr], [i, i], color=GRIS_CLAIR, lw=3, zorder=1)
        ax.scatter(
            [tr],
            [i],
            color=GRIS,
            s=70,
            zorder=2,
            label="train (plis d'apprentissage)" if i == 0 else None,
        )
        ax.scatter([va], [i], color=COULEURS_MODELES[m], s=70, zorder=3)
        ax.text(va - 0.01, i + 0.22, f"CV {va:.3f}", ha="right", fontsize=8)
        ax.text(tr, i + 0.22, f"train {tr:.3f}", ha="center", fontsize=8)
        if f1_test and m in f1_test:
            ax.scatter(
                [f1_test[m]],
                [i],
                marker="D",
                color="black",
                s=40,
                zorder=4,
                label="test (chiens jamais vus)" if m == list(f1_test)[0] else None,
            )
    ax.set_yticks(range(len(mods)))
    ax.set_yticklabels(mods)
    ax.set_xlim(0.6, 1.03)
    ax.set_ylim(-0.6, len(mods) - 0.3)
    ax.set_xlabel("F1 macro")
    ax.set_title(
        f"Écart train / validation ({NOMS_EXPERIENCES.get(experience, experience)})"
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def selection_par_bloc(all_features, retained) -> pd.DataFrame:
    """Variables retenues par la sélection, par échelle et par capteur."""

    def echelle(c):
        return c.rsplit("_", 1)[-1] if c.rsplit("_", 1)[-1].startswith("w") else "-"

    def capteur(c):
        return c.split("_")[0] if c.split("_")[0] in cfg.SENSORS else c.split("_")[0]

    a = pd.DataFrame({"var": list(all_features)})
    a["retenue"] = a["var"].isin(set(retained))
    a["échelle"] = a["var"].map(echelle)
    a["capteur"] = a["var"].map(capteur)
    out = {}
    for k in ("échelle", "capteur"):
        t = a.groupby(k)["retenue"].agg(retenues="sum", total="size")
        t["part"] = (t["retenues"] / t["total"]).round(2)
        out[k] = t
    return out


# ---------------------------------------------------------------------------
# 5. Modèles spécifiques
# ---------------------------------------------------------------------------
def plot_general_vs_specifique(comp: pd.DataFrame):
    fams = comp["famille"].tolist()
    x = np.arange(len(fams))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    for ax, metr, titre in [
        (axes[0], "f1_macro", "F1 macro"),
        (axes[1], "exactitude", "Exactitude"),
    ]:
        g, s = comp[f"{metr}_general"], comp[f"{metr}_specifique"]
        ax.bar(x - 0.2, g, 0.38, color=BLEU, label="modèle général (11 classes)")
        ax.bar(x + 0.2, s, 0.38, color=ORANGE, label="modèle spécifique (famille)")
        for i in range(len(fams)):
            ax.text(
                x[i] - 0.2,
                g.iloc[i] + 0.01,
                f"{g.iloc[i]:.2f}",
                ha="center",
                fontsize=8,
            )
            ax.text(
                x[i] + 0.2,
                s.iloc[i] + 0.01,
                f"{s.iloc[i]:.2f}",
                ha="center",
                fontsize=8,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(fams)
        ax.set_ylim(0, 1.08)
        ax.set_title(
            f"{titre} - mêmes fenêtres de test (comportement de la famille = Behavior_1)"
        )
    axes[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. tsfresh
# ---------------------------------------------------------------------------
def plot_par_chien_apparie(
    par_chien: pd.DataFrame, col_a: str, col_b: str, titre: str, lab_a: str, lab_b: str
):
    """Une ligne par chien : comparaison appariée de deux approches."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for dog, r in par_chien.iterrows():
        up = r[col_b] > r[col_a]
        ax.plot(
            [0, 1],
            [r[col_a], r[col_b]],
            color=VERT if up else ORANGE,
            lw=1.5,
            marker="o",
        )
        ax.text(1.04, r[col_b], str(dog), va="center", fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([lab_a, lab_b])
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylabel("F1 macro du chien de test")
    ax.set_title(titre)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Prédiction conforme (courbe sans double axe)
# ---------------------------------------------------------------------------
def plot_courbe_conforme(courbe: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(
        courbe["couverture_visee"],
        courbe["couverture"],
        marker="o",
        color=BLEU,
        lw=2,
        label="obtenue (test)",
    )
    axes[0].plot([0.7, 1], [0.7, 1], ls="--", color=GRIS, lw=1, label="visée")
    axes[0].set_xlabel("couverture visée (1 - α)")
    axes[0].set_ylabel("couverture obtenue sur le test")
    axes[0].legend()
    axes[0].set_title("La garantie est tenue en moyenne")
    axes[1].plot(
        courbe["couverture_visee"],
        courbe["taille_moyenne"],
        marker="o",
        color=ORANGE,
        lw=2,
    )
    for _, r in courbe.iterrows():
        axes[1].text(
            r["couverture_visee"],
            r["taille_moyenne"] + 0.07,
            f"{r['taille_moyenne']:.2f}",
            ha="center",
            fontsize=8,
        )
    axes[1].set_xlabel("couverture visée (1 - α)")
    axes[1].set_ylabel("nb moyen de comportements proposés")
    axes[1].set_title("Prix de la certitude : taille des ensembles")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Synthèse
# ---------------------------------------------------------------------------
def plot_synthese(rows: pd.DataFrame):
    """rows : colonnes 'approche', 'f1_macro', 'groupe' (couleur), triées dans l'ordre d'affichage."""
    couleurs = {
        "référence": GRIS_CLAIR,
        "fenêtres": BLEU,
        "tsfresh (sous-échantillon)": VERT,
        "post-traitement": ORANGE,
    }
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(rows) + 1.2))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, rows["f1_macro"], color=[couleurs[g] for g in rows["groupe"]])
    for yi, v in zip(y, rows["f1_macro"]):
        ax.text(v + 0.005, yi, f"{v:.3f}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(rows["approche"])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("F1 macro sur les 9 chiens de test (11 classes, Behavior_1)")
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color=c, label=g)
            for g, c in couleurs.items()
            if g in set(rows["groupe"])
        ],
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        fontsize=8,
    )
    ax.set_title("Synthèse des résultats de test")
    fig.tight_layout()
    return fig
