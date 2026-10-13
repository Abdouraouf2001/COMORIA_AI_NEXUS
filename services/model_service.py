"""Services de modélisation : pipelines scikit-learn, entraînement, comparaison."""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from config.settings import CLASSIFICATION, GRAINE, REGRESSION
from services.data_service import est_numerique

MODELES = {
    CLASSIFICATION: {
        "Forêt aléatoire": lambda: RandomForestClassifier(n_estimators=200, random_state=GRAINE),
        "Régression logistique": lambda: LogisticRegression(max_iter=1000),
        "Gradient boosting": lambda: GradientBoostingClassifier(random_state=GRAINE),
        "k plus proches voisins": lambda: KNeighborsClassifier(),
        "Arbre de décision": lambda: DecisionTreeClassifier(random_state=GRAINE),
    },
    REGRESSION: {
        "Forêt aléatoire": lambda: RandomForestRegressor(n_estimators=200, random_state=GRAINE),
        "Régression linéaire": lambda: LinearRegression(),
        "Gradient boosting": lambda: GradientBoostingRegressor(random_state=GRAINE),
        "k plus proches voisins": lambda: KNeighborsRegressor(),
        "Arbre de décision": lambda: DecisionTreeRegressor(random_state=GRAINE),
    },
}


def detecter_tache(serie: pd.Series) -> str:
    if not est_numerique(serie):
        return CLASSIFICATION
    valeurs = serie.dropna()
    if valeurs.nunique() <= 10 and bool((valeurs % 1 == 0).all()):
        return CLASSIFICATION
    return REGRESSION


def variables_par_defaut(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    """Écarte les colonnes de texte très variées (identifiants, noms…)."""
    return [c for c in candidates if est_numerique(df[c]) or df[c].nunique() <= 50]


def construire_pipeline(df: pd.DataFrame, colonnes: list[str], tache: str, nom_modele: str) -> Pipeline:
    numeriques = [c for c in colonnes if est_numerique(df[c])]
    categorielles = [c for c in colonnes if c not in numeriques]
    blocs = []
    if numeriques:
        blocs.append(
            (
                "num",
                Pipeline([("imputation", SimpleImputer(strategy="median")), ("echelle", StandardScaler())]),
                numeriques,
            )
        )
    if categorielles:
        blocs.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputation", SimpleImputer(strategy="most_frequent")),
                        ("encodage", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorielles,
            )
        )
    return Pipeline(
        [("preparation", ColumnTransformer(blocs)), ("modele", MODELES[tache][nom_modele]())]
    )


def profil_variables(X: pd.DataFrame) -> dict:
    profil = {}
    for colonne in X.columns:
        if est_numerique(X[colonne]):
            profil[colonne] = {"type": "num", "mediane": float(X[colonne].median())}
        else:
            valeurs = sorted(X[colonne].dropna().unique().tolist(), key=str)
            profil[colonne] = {"type": "cat", "valeurs": valeurs}
    return profil


def entrainer(df, cible, colonnes, nom_modele, tache, part_test, graine=GRAINE) -> dict:
    donnees = df.dropna(subset=[cible])
    X, y = donnees[colonnes], donnees[cible]
    stratification = None
    if tache == CLASSIFICATION and y.value_counts().min() >= 2:
        stratification = y
    X_ent, X_test, y_ent, y_test = train_test_split(
        X, y, test_size=part_test, random_state=graine, stratify=stratification
    )
    pipeline = construire_pipeline(donnees, colonnes, tache, nom_modele)
    debut = time.perf_counter()
    pipeline.fit(X_ent, y_ent)
    duree = time.perf_counter() - debut
    predictions = pipeline.predict(X_test)

    confusion = None
    if tache == CLASSIFICATION:
        metriques = {
            "Exactitude": accuracy_score(y_test, predictions),
            "Précision": precision_score(y_test, predictions, average="weighted", zero_division=0),
            "Rappel": recall_score(y_test, predictions, average="weighted", zero_division=0),
            "F1": f1_score(y_test, predictions, average="weighted", zero_division=0),
        }
        classes = list(pipeline.classes_)
        confusion = pd.DataFrame(
            confusion_matrix(y_test, predictions, labels=classes),
            index=[f"Réel : {c}" for c in classes],
            columns=[f"Prédit : {c}" for c in classes],
        )
    else:
        metriques = {
            "R²": r2_score(y_test, predictions),
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        }

    X_imp, y_imp = X_test, y_test
    if len(X_test) > 500:
        echantillon = X_test.sample(500, random_state=graine).index
        X_imp, y_imp = X_test.loc[echantillon], y_test.loc[echantillon]
    importance = permutation_importance(pipeline, X_imp, y_imp, n_repeats=5, random_state=graine)
    importances = (
        pd.DataFrame({"Variable": colonnes, "Importance": importance.importances_mean})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    maximum = importances["Importance"].max()
    importances["Importance relative"] = (
        (importances["Importance"].clip(lower=0) / maximum) if maximum > 0 else 0.0
    )

    return {
        "id": datetime.now().strftime("%H%M%S%f"),
        "heure": datetime.now().strftime("%H:%M:%S"),
        "pipeline": pipeline,
        "tache": tache,
        "nom_modele": nom_modele,
        "cible": cible,
        "colonnes": colonnes,
        "metriques": metriques,
        "confusion": confusion,
        "importances": importances,
        "reel": y_test.to_numpy(),
        "predit": predictions,
        "profil": profil_variables(X),
        "duree": duree,
        "n_train": len(X_ent),
        "n_test": len(X_test),
    }


def resume_modele(resultat: dict) -> dict:
    metrique, score = next(iter(resultat["metriques"].items()))
    return {
        "Heure": resultat["heure"],
        "Modèle": resultat["nom_modele"],
        "Cible": resultat["cible"],
        "Tâche": resultat["tache"],
        "Métrique": metrique,
        "Score": round(float(score), 4),
        "Durée (s)": round(resultat["duree"], 2),
    }


def comparer_modeles(df, cible, colonnes, noms, tache, plis, progression=None) -> pd.DataFrame:
    donnees = df.dropna(subset=[cible])
    X, y = donnees[colonnes], donnees[cible]
    mesure = "accuracy" if tache == CLASSIFICATION else "r2"
    lignes = []
    for i, nom in enumerate(noms):
        pipeline = construire_pipeline(donnees, colonnes, tache, nom)
        debut = time.perf_counter()
        scores = cross_val_score(pipeline, X, y, cv=plis, scoring=mesure)
        lignes.append(
            {
                "Modèle": nom,
                "Score moyen": float(scores.mean()),
                "Écart-type": float(scores.std()),
                "Durée (s)": round(time.perf_counter() - debut, 2),
            }
        )
        if progression:
            progression((i + 1) / len(noms))
    return (
        pd.DataFrame(lignes).sort_values("Score moyen", ascending=False).reset_index(drop=True)
    )
