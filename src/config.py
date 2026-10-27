"""
Configuration centrale du projet.

Tous les chemins sont RELATIFS à la racine du projet : le projet fonctionne quelle que soit la machine.
Tous les partis pris (taille des fenêtres, seuils, graines aléatoires...) sont regroupés ici pour être faciles à modifier.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
# La variable d'environnement DOG_PROJECT_ROOT permet de pointer vers une autre
# racine si besoin
ROOT = Path(os.environ.get("DOG_PROJECT_ROOT", Path(__file__).resolve().parents[1]))

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # fichiers d'origine, jamais modifiés
INTERIM_DIR = (
    DATA_DIR / "interim"
)  # cache rapide (parquet) des données brutes nettoyées
PROCESSED_DIR = DATA_DIR / "processed"  # jeux de fenêtres, split, prédictions
MODELS_DIR = ROOT / "models"  # modèles entraînés (.joblib)
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
CACHE_DIR = ROOT / ".cache"  # cache joblib des pipelines (sélection de variables)

RAW_MOVE_FILE = RAW_DIR / "DogMoveData.csv"
RAW_INFO_FILE = RAW_DIR / "DogInfo.xlsx"

MOVES_CACHE = INTERIM_DIR / "dog_moves.parquet"
DOGS_CACHE = INTERIM_DIR / "dog_info.parquet"
SPLIT_FILE = PROCESSED_DIR / "split.json"
WINDOWS_FILE = PROCESSED_DIR / "windows.parquet"

for _d in (
    RAW_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    CACHE_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Description des données
# ---------------------------------------------------------------------------
SAMPLING_RATE = 100  # Hz (t_sec avance par pas de 0.01 s, fixé par le jeu de données)
SENSORS = ["ABack", "ANeck", "GBack", "GNeck"]  # A = accéléromètre, G = gyroscope
AXES = ["x", "y", "z"]
SENSOR_COLUMNS = [f"{s}_{a}" for s in SENSORS for a in AXES]
BEHAVIOR_COLUMNS = ["Behavior_1", "Behavior_2", "Behavior_3"]
SESSION_KEYS = ["DogID", "TestNum"]

NA_VALUES = ["<undefined>"]
# Étiquettes techniques (calage des capteurs avec la vidéo), pas des comportements.
TECHNICAL_LABELS = ["Synchronization", "Extra_Synchronization"]

# ---------------------------------------------------------------------------
# Reproductibilité
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
SPLIT_SEED = 2  # graine du split train/test par chien
TEST_SIZE = 0.20  # 20 % des chiens en test
N_CV_SPLITS = 5  # validation croisée groupée par chien

# ---------------------------------------------------------------------------
# Fenêtrage (partis pris de l'approche 1)
# ---------------------------------------------------------------------------
LABEL_WINDOW_S = 1.0  # fenêtre « cœur » sur laquelle est défini le label
STEP_S = 0.5  # pas entre deux fenêtres (=> 50 % de recouvrement des cœurs)
SCALES_S = (1.0, 2.0, 4.0)  # échelles de contexte centrées sur le même cœur
MIN_PURITY = 0.8  # fraction minimale du cœur portant le label majoritaire
# Classes plus rares (dans le TRAIN) exclues de la tâche.
MIN_WINDOWS_PER_CLASS = int(os.environ.get("DOG_MIN_WINDOWS_PER_CLASS", "500"))

# ---------------------------------------------------------------------------
# Familles de comportements (modèles spécifiques)
# ---------------------------------------------------------------------------
# Les labels de Behavior_1 ne sont pas mutuellement exclusifs : un chien peut
# haleter tout en étant debout. On sépare donc trois familles dont les classes,
# elles, s'excluent mutuellement.
FAMILIES = {
    "posture": ["Lying chest", "Sitting", "Standing"],
    "locomotion": ["Walking", "Trotting", "Pacing", "Galloping", "Jumping"],
    "activite": [
        "Sniffing",
        "Eating",
        "Drinking",
        "Playing",
        "Panting",
        "Shaking",
        "Carrying object",
        "Tugging",
        "Bowing",
    ],
}

# ---------------------------------------------------------------------------
# Profils des chiens (surtout pour l'analyse des performances)
# ---------------------------------------------------------------------------
WEIGHT_BINS = [0, 15, 30, 100]
WEIGHT_LABELS = ["Petit (<15 kg)", "Moyen (15-30 kg)", "Grand (>=30 kg)"]
AGE_BINS = [0, 24, 84, 400]
AGE_LABELS = ["Jeune (<2 ans)", "Adulte (2-7 ans)", "Senior (>=7 ans)"]
STATIC_FEATURES = ["Weight", "Age months", "Gender", "NeuteringStatus"]

# ---------------------------------------------------------------------------
# Mode rapide : grilles d'hyperparamètres réduites (tests, itérations rapides)
# Activer avec la variable d'environnement DOG_QUICK=1.
# ---------------------------------------------------------------------------
QUICK = os.environ.get("DOG_QUICK", "0") == "1"
N_JOBS = int(os.environ.get("DOG_N_JOBS", "-1"))
