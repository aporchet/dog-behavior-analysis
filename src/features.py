"""
Fenêtrage multi-échelle et extraction de variables (approche 1).

Principe
--------
* Les séries sont découpées en SEGMENTS continus : une fenêtre ne peut jamais
  enjamber une rupture temporelle.
* Chaque fenêtre est définie par un CŒUR de LABEL_WINDOW_S secondes (1 s), qui
  porte le label, et par plusieurs ÉCHELLES DE CONTEXTE centrées sur ce même
  cœur (1 s, 2 s, 4 s). Les comportements lents (renifler, manger) ont besoin
  d'un contexte plus long que la marche ou le galop.
* Toutes les échelles sont calculées sur EXACTEMENT les mêmes fenêtres : on peut
  comparer « 1 s seulement » et « multi-échelle » à données identiques.
* Chaque fenêtre conserve ses métadonnées (chien, séance, segment, temps de
  début/fin, position dans le CSV) ainsi que Behavior_1/2/3 calculés AU MOMENT
  de l'extraction. Cela corrige le désalignement de l'ancienne version, qui
  reconstruisait Behavior_2/3 a posteriori à partir d'un index de ligne faux
  dès qu'une fenêtre avait été sautée.

Variables par échelle et par signal
-----------------------------------
Signaux : 12 axes bruts, 4 normes (intensité globale du mouvement ou de la
rotation) et 4 « taux de variation de la norme » (dérivée de la norme,
anciennement appelée « jerk » ; ce n'est pas le jerk physique, qui serait la
dérivée du vecteur accélération).
Statistiques : moyenne, écart-type, min, max (la moyenne des axes
d'accélération encode l'orientation / la posture via la gravité).
Domaine fréquentiel (sur les 4 normes) : fréquence dominante, énergie,
entropie spectrale (séparent marche / trot / galop).
Ratios cou/dos : indépendance des mouvements de tête par rapport au corps.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg

STATS = ("mean", "std", "min", "max")
SPECTRAL = ("fdom", "energie", "entropie")
NORM_SIGNALS = [f"{s}_norm" for s in cfg.SENSORS]
DNORM_SIGNALS = [f"{s}_dnorm" for s in cfg.SENSORS]
SIGNALS = cfg.SENSOR_COLUMNS + NORM_SIGNALS + DNORM_SIGNALS
META_COLUMNS = ["window_id", "DogID", "TestNum", "segment", "t_start", "t_center", "t_end", "row_start"]


def scale_tag(scale_s: float) -> str:
    return f"w{scale_s:g}s"


# ---------------------------------------------------------------------------
# Signaux dérivés
# ---------------------------------------------------------------------------
def compute_signals(axes: np.ndarray, fs: int = cfg.SAMPLING_RATE) -> np.ndarray:
    """axes : (n, 12) dans l'ordre de cfg.SENSOR_COLUMNS -> (n, 20) signaux."""
    axes = axes.astype(np.float32, copy=False)
    norms = np.sqrt((axes.reshape(len(axes), len(cfg.SENSORS), 3) ** 2).sum(axis=2))
    dnorms = np.diff(norms, axis=0, prepend=norms[:1]) * fs  # unité : (g ou °/s) par seconde
    return np.concatenate([axes, norms, dnorms], axis=1).astype(np.float32)


def find_segments(t: np.ndarray, fs: int = cfg.SAMPLING_RATE):
    """Indices [début, fin) des portions sans rupture temporelle."""
    dt = np.diff(t)
    breaks = np.where((dt > 1.5 / fs) | (dt <= 0))[0] + 1
    bounds = np.concatenate([[0], breaks, [len(t)]])
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


# ---------------------------------------------------------------------------
# Calcul des variables sur un lot de fenêtres
# ---------------------------------------------------------------------------
def _stats_features(W: np.ndarray, tag: str) -> dict:
    """W : (n_fenêtres, L, n_signaux)."""
    out = {}
    agg = {"mean": W.mean(axis=1), "std": W.std(axis=1),
           "min": W.min(axis=1), "max": W.max(axis=1)}
    for stat, values in agg.items():
        for j, sig in enumerate(SIGNALS):
            out[f"{sig}_{stat}_{tag}"] = values[:, j]
    return out


def _spectral_features(Wn: np.ndarray, tag: str, fs: int) -> dict:
    """Wn : (n_fenêtres, L, 4) normes des capteurs."""
    L = Wn.shape[1]
    x = Wn - Wn.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(x, axis=1)) ** 2
    freqs = np.fft.rfftfreq(L, d=1.0 / fs)
    power, freqs = power[:, 1:, :], freqs[1:]          # on retire la composante continue
    total = power.sum(axis=1)
    p = power / (total[:, None, :] + 1e-12)
    entropy = -(p * np.log(p + 1e-12)).sum(axis=1) / np.log(power.shape[1])
    out = {}
    for j, sig in enumerate(NORM_SIGNALS):
        out[f"{sig}_fdom_{tag}"] = freqs[power[:, :, j].argmax(axis=1)]
        out[f"{sig}_energie_{tag}"] = total[:, j] / L
        out[f"{sig}_entropie_{tag}"] = entropy[:, j]
    return out


def _ratio_features(W: np.ndarray, tag: str) -> dict:
    idx = {s: SIGNALS.index(s) for s in NORM_SIGNALS}
    m = W.mean(axis=1)
    return {
        f"ratio_A_cou_dos_{tag}": m[:, idx["ANeck_norm"]] / (m[:, idx["ABack_norm"]] + 1e-6),
        f"ratio_G_cou_dos_{tag}": m[:, idx["GNeck_norm"]] / (m[:, idx["GBack_norm"]] + 1e-6),
    }


def _count_labels(codes: np.ndarray, n_labels: int) -> np.ndarray:
    """codes : (n_fenêtres, L) entiers (-1 = manquant) -> comptes (n_fenêtres, n_labels)."""
    return (codes[..., None] == np.arange(n_labels)).sum(axis=1)


def _mode(counts: np.ndarray, labels: np.ndarray):
    best = counts.argmax(axis=1)
    n_best = counts[np.arange(len(counts)), best]
    lab = labels[best].astype(object)
    lab[n_best == 0] = None
    return lab, n_best


# ---------------------------------------------------------------------------
# Construction du jeu de fenêtres
# ---------------------------------------------------------------------------
def build_windows(moves: pd.DataFrame,
                  scales_s=cfg.SCALES_S,
                  label_window_s: float = cfg.LABEL_WINDOW_S,
                  step_s: float = cfg.STEP_S,
                  families: dict = cfg.FAMILIES,
                  fs: int = cfg.SAMPLING_RATE,
                  batch_size: int = 2048,
                  verbose: bool = True) -> pd.DataFrame:
    """
    Construit une ligne par fenêtre (annotée ou non ; le filtrage selon la
    pureté du label se fait ensuite, tâche par tâche).

    Colonnes produites :
      - métadonnées : META_COLUMNS
      - behavior_1 / purity_1 / labelled_1 : label majoritaire de Behavior_1 sur
        le cœur, fraction du cœur portant ce label, fraction annotée ;
      - behavior_2 / behavior_3 : label majoritaire (s'il existe) sur le cœur ;
      - <famille>_label / <famille>_purity : label de la famille présent dans
        Behavior_1, 2 OU 3 (ex. la posture d'un chien qui halète est souvent
        dans Behavior_2) ;
      - variables : voir le docstring du module.
    """
    moves = moves.sort_values(cfg.SESSION_KEYS + ["t_sec"], kind="stable").reset_index(drop=True)
    n_core = int(round(label_window_s * fs))
    step = int(round(step_s * fs))
    scale_lengths = {scale_tag(s): int(round(s * fs)) for s in scales_s}
    half_max = max(max(scale_lengths.values()), n_core) // 2

    # Encodage commun des labels des trois colonnes
    all_labels = sorted(set().union(*[set(moves[c].dropna().unique()) for c in cfg.BEHAVIOR_COLUMNS]))
    label_array = np.array(all_labels, dtype=object)
    codes = np.stack([pd.Categorical(moves[c], categories=all_labels).codes
                      for c in cfg.BEHAVIOR_COLUMNS], axis=1).astype(np.int16)
    fam_idx = {f: np.array([all_labels.index(l) for l in labs if l in all_labels], dtype=int)
               for f, labs in families.items()}

    axes_all = moves[cfg.SENSOR_COLUMNS].to_numpy(np.float32)
    t_all = moves["t_sec"].to_numpy()
    blocks = []
    window_counter = 0
    sessions = moves.groupby(cfg.SESSION_KEYS, sort=True).indices

    for (dog, test), rows in sessions.items():
        s0, s1 = int(rows[0]), int(rows[-1]) + 1  # lignes contiguës après le tri
        for seg_id, (a, b) in enumerate(find_segments(t_all[s0:s1], fs)):
            a, b = a + s0, b + s0
            n = b - a
            centers = np.arange(half_max, n - half_max + 1, step)
            if len(centers) == 0:
                continue
            signals = compute_signals(axes_all[a:b], fs)
            seg_codes = codes[a:b]
            for k in range(0, len(centers), batch_size):
                c = centers[k:k + batch_size]
                feats = {}
                for tag, L in scale_lengths.items():
                    idx = c[:, None] + np.arange(-(L // 2), L - L // 2)[None, :]
                    W = signals[idx]
                    feats.update(_stats_features(W, tag))
                    feats.update(_spectral_features(W[:, :, [SIGNALS.index(s) for s in NORM_SIGNALS]], tag, fs))
                    feats.update(_ratio_features(W, tag))

                core = c[:, None] + np.arange(-(n_core // 2), n_core - n_core // 2)[None, :]
                core_codes = seg_codes[core]  # (nw, n_core, 3)
                counts = [_count_labels(core_codes[:, :, j], len(all_labels)) for j in range(3)]
                meta = {
                    "window_id": np.arange(window_counter, window_counter + len(c)),
                    "DogID": np.full(len(c), dog), "TestNum": np.full(len(c), test),
                    "segment": np.full(len(c), seg_id),
                    "t_start": t_all[a + core[:, 0]], "t_center": t_all[a + c],
                    "t_end": t_all[a + core[:, -1]], "row_start": a + core[:, 0],
                }
                window_counter += len(c)
                lab1, n1 = _mode(counts[0], label_array)
                meta["behavior_1"] = lab1
                meta["purity_1"] = n1 / n_core
                meta["labelled_1"] = (core_codes[:, :, 0] >= 0).mean(axis=1)
                meta["behavior_2"] = _mode(counts[1], label_array)[0]
                meta["behavior_3"] = _mode(counts[2], label_array)[0]
                total = counts[0] + counts[1] + counts[2]
                for fam, fidx in fam_idx.items():
                    fl, fn = _mode(total[:, fidx], label_array[fidx]) if len(fidx) else (np.full(len(c), None), np.zeros(len(c)))
                    meta[f"{fam}_label"] = fl
                    meta[f"{fam}_purity"] = np.minimum(fn / n_core, 1.0)
                block = pd.DataFrame(meta)
                feat_df = pd.DataFrame({k2: v.astype(np.float32) for k2, v in feats.items()})
                blocks.append(pd.concat([block, feat_df], axis=1))
        if verbose:
            print(f"  chien {dog} / test {test} : {window_counter:,} fenêtres cumulées")

    windows = pd.concat(blocks, ignore_index=True)
    for col in ["behavior_1", "behavior_2", "behavior_3"] + [f"{f}_label" for f in families]:
        windows[col] = windows[col].astype("object").where(windows[col].notna(), None)
    return windows


# ---------------------------------------------------------------------------
# Jeux de variables
# ---------------------------------------------------------------------------
def all_feature_columns(windows: pd.DataFrame) -> list:
    meta = set(META_COLUMNS) | {"behavior_1", "purity_1", "labelled_1", "behavior_2", "behavior_3"}
    meta |= {c for c in windows.columns if c.endswith("_label") or c.endswith("_purity")}
    return [c for c in windows.columns if c not in meta]


def feature_set(windows: pd.DataFrame, name: str) -> list:
    """
    Jeux de variables comparés dans le projet :
      - 'ancienne_baseline_1s' : moyenne et écart-type sur 1 s + ratios
        (réplique de la première version, 42 variables) ;
      - 'complet_1s' : toutes les variables à 1 s (+ min/max et fréquentiel) ;
      - 'multi_echelle' : toutes les variables à toutes les échelles.
    """
    cols = all_feature_columns(windows)
    tag1 = scale_tag(1.0)
    if name == "ancienne_baseline_1s":
        return [c for c in cols if c.endswith(tag1)
                and (c.startswith("ratio_") or any(f"_{s}_" in c for s in ("mean", "std")))]
    if name == "complet_1s":
        return [c for c in cols if c.endswith(tag1)]
    if name == "multi_echelle":
        return cols
    raise ValueError(f"Jeu de variables inconnu : {name}")


FEATURE_SET_NAMES = ["ancienne_baseline_1s", "complet_1s", "multi_echelle"]
