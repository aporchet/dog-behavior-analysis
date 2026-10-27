"""Code source du projet de classification du comportement des chiens.

Modules :
    config            chemins et partis pris (fenêtres, seuils, graines)
    data              chargement, cache parquet, contrôles qualité, split par chien
    features          fenêtrage multi-échelle et extraction de variables
    labels            définition des tâches (général / familles), classes rares
    modeling          modèles, CV groupée, sélection par CV, sauvegarde
    evaluation        métriques, rapports, scores par chien et par profil
    smoothing         bonus : lissage temporel des prédictions
    conformal         bonus : prédiction conforme (maison + MAPIE)
    tsfresh_features  approche de comparaison tsfresh
    presentation      fonctions d'affiche du notebook de présentation
    viz               figures (signaux, frises, UMAP)
"""
