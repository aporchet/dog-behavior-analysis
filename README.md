# Classification du comportement des chiens à partir de capteurs de mouvement

Projet de machine learning : prédire le comportement d'un chien (posture, locomotion, activités) à partir de séries temporelles d'accéléromètres et de gyroscopes placés sur le dos et sur le cou (100 Hz, 45 chiens).

Les notebooks 00 à 07 retraçent l'ensemble des expérimentations menées, de la préparation des données, leur exploration, jusqu'aux étapes de features engineering et d'entraînement de modèles en vue de leur comparaison. Le notebook 08 constitue une synthèse du travail mené dans les notebooks précédent et est le support de présentation pour la soutenance orale.

## Arborescence

```
dog-behavior-analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/          DogMoveData.csv, DogInfo.xlsx (à déposer ici, jamais modifiés)
│   ├── interim/      cache parquet des données nettoyées (notebook 00)
│   └── processed/    split.json, windows.parquet, prédictions, cache tsfresh
├── models/           modèles entraînés : models/<tâche>/<expérience>/<modèle>.joblib
├── reports/
│   ├── figures/      toutes les figures (PNG) produites par les notebooks
│   └── tables/       résultats (CSV) + meilleur_modele_general.json
├── src/              toute la logique réutilisable
│   ├── config.py           chemins relatifs et partis pris (fenêtres, seuils, graines)
│   ├── data.py             chargement, cache, contrôles qualité, split par chien
│   ├── features.py         fenêtrage multi-échelle et extraction des variables
│   ├── labels.py           tâches (générale / familles), classes rares
│   ├── modeling.py         modèles, CV groupée, choix par CV, sauvegarde
│   ├── evaluation.py       métriques, rapports, scores par chien et par profil
│   ├── presentation.py     fonctions d'affichage pour le notebook de présentation 08
│   ├── smoothing.py        bonus : lissage temporel des prédictions
│   ├── conformal.py        bonus : prédiction conforme (maison + MAPIE)
│   ├── tsfresh_features.py comparaison avec tsfresh
│   └── viz.py              signaux, frises interactives, UMAP
└── notebooks/        à exécuter dans l'ordre
    ├── 00_preparation_donnees.ipynb
    ├── 01_exploration.ipynb
    ├── 02_split_train_test.ipynb
    ├── 03_fenetres_et_variables.ipynb
    ├── 04_modeles_generaux.ipynb
    ├── 05_modeles_specifiques.ipynb
    ├── 06_comparaison_tsfresh.ipynb
    ├── 07_analyses_bonus.ipynb
    └── 08_presentation_soutenance.ipynb
```

## Installation et exécution

```bash
pip install -r requirements.txt
# déposer DogMoveData.csv et DogInfo.xlsx dans data/raw/
jupyter lab   # lancé depuis la racine du projet ou depuis notebooks/
```

Chaque notebook relit ce que les précédents ont écrit sur disque (cache, split, fenêtres, modèles) : on peut
reprendre à n'importe quelle étape sans tout relancer.

**Itérer vite** : `DOG_QUICK=1 jupyter lab` utilise des grilles d'hyperparamètres réduites. Les grilles
complètes (défaut) sont longues sur ~100 000 fenêtres ; on peut aussi restreindre la variable `MODELES`
en tête des notebooks 04 et 05.

## Partis pris principaux (détaillés dans les notebooks)

| Sujet | Choix |
|---|---|
| Split | par chien (80 / 20), toutes les séances d'un même chien dans le même jeu, figé dans `data/processed/split.json` |
| Validation | `StratifiedGroupKFold` à 5 plis, groupé par chien |
| Choix du modèle | sur le F1 macro de validation croisée, jamais sur le test |
| Métrique principale | F1 macro (car les classes sont déséquilibrées) ; exactitude et exactitude équilibrée (Behavior_2/3) en complément |
| Baselines | `DummyClassifier` (classe majoritaire et aléatoire stratifié) |
| Synchronisation | lignes conservées, label mis à manquant (pas de trou artificiel dans le signal) |
| Fenêtres | cœur de 1 s (label), pas de 0,5 s, contexte 1 / 2 / 4 s |
| Pureté | label majoritaire sur au moins 80 % du cœur |
| Classes rares | < 500 fenêtres dans le train : exclues du train et du test (tableau et part exclue affichés) |
| Familles | posture / locomotion / activité, label cherché dans Behavior_1, 2 ou 3 |
| Variables de profil | testées en expérience séparée ; poids et âge en classes utilisés seulement pour l'analyse |
| `Task` | jamais utilisée comme variable (information de protocole) |

## Données

Jeu de données publié avec Kumpulainen et al. (2021), *Dog behaviour classification with movement sensors
placed on the harness and the collar*, Applied Animal Behaviour Science. Téléchargé depuis Kaggle au lien suivant: https://www.kaggle.com/datasets/arashnic/animal-behavior-analysis
