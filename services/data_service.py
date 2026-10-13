"""Services de données : exemples, lecture CSV, nettoyage et préparation numérique."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from config.settings import STRATEGIE_REMPLACER, STRATEGIE_SUPPRIMER


# ---------- Types de colonnes ----------

def est_numerique(serie: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(serie) and not pd.api.types.is_bool_dtype(serie)


def colonnes_numeriques(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if est_numerique(df[c]) and df[c].notna().any()]


# ---------- Jeux d'exemple ----------

def _depuis_sklearn(chargeur) -> pd.DataFrame:
    donnees = chargeur(as_frame=True)
    df = donnees.frame.copy()
    df["classe"] = donnees.target.map(dict(enumerate(donnees.target_names)))
    return df.drop(columns="target")


def exemple_iris() -> pd.DataFrame:
    return _depuis_sklearn(load_iris)


def exemple_vin() -> pd.DataFrame:
    return _depuis_sklearn(load_wine)


def exemple_cancer() -> pd.DataFrame:
    return _depuis_sklearn(load_breast_cancer)


def exemple_boutiques(n: int = 400, graine: int = 42) -> pd.DataFrame:
    """Données synthétiques : ventes journalières de petites boutiques (en KMF)."""
    rng = np.random.default_rng(graine)
    ile = rng.choice(["Ngazidja", "Ndzouani", "Mwali"], n, p=[0.5, 0.35, 0.15])
    surface = rng.normal(35, 12, n).clip(10, 90).round(1)
    employes = rng.integers(1, 7, n)
    clients = (rng.normal(60, 20, n) + surface * 0.8).clip(10, None).round()
    prix = rng.normal(2500, 500, n).clip(800, None).round(-1)
    promotion = rng.choice(["oui", "non"], n, p=[0.3, 0.7])
    coefficient_ile = pd.Series(ile).map(
        {"Ngazidja": 1.15, "Ndzouani": 1.0, "Mwali": 0.85}
    ).to_numpy()
    ventes = (
        clients * prix * 0.35 * coefficient_ile * np.where(promotion == "oui", 1.12, 1.0)
        + rng.normal(0, 15000, n)
    ).clip(5000, None).round(-2)
    df = pd.DataFrame(
        {
            "ile": ile,
            "surface_m2": surface,
            "employes": employes,
            "clients_par_jour": clients,
            "prix_moyen_kmf": prix,
            "promotion": promotion,
            "ventes_par_jour_kmf": ventes,
        }
    )
    lignes = rng.choice(n, 20, replace=False)
    df.loc[lignes, "surface_m2"] = np.nan
    return df


EXEMPLES = {
    "Boutiques comoriennes (synthétique, régression)": exemple_boutiques,
    "Iris (classification)": exemple_iris,
    "Vin (classification)": exemple_vin,
    "Cancer du sein (classification binaire)": exemple_cancer,
}


# ---------- Lecture et nettoyage ----------

@st.cache_data(show_spinner=False)
def lire_csv(contenu: bytes) -> pd.DataFrame:
    for encodage in ("utf-8", "latin-1"):
        for separateur in (",", ";", "\t"):
            try:
                df = pd.read_csv(io.BytesIO(contenu), sep=separateur, encoding=encodage)
            except Exception:
                continue
            if df.shape[1] > 1:
                return df
    raise ValueError(
        "Impossible de lire ce fichier. Vérifie que c'est un CSV avec une ligne d'en-têtes."
    )


def nettoyer(df: pd.DataFrame, supprimer_doublons: bool, strategie: str) -> pd.DataFrame:
    resultat = df.copy()
    if supprimer_doublons:
        resultat = resultat.drop_duplicates()
    if strategie == STRATEGIE_REMPLACER:
        for colonne in resultat.columns:
            if not resultat[colonne].isna().any():
                continue
            if est_numerique(resultat[colonne]):
                resultat[colonne] = resultat[colonne].fillna(resultat[colonne].median())
            else:
                mode = resultat[colonne].mode(dropna=True)
                if not mode.empty:
                    resultat[colonne] = resultat[colonne].fillna(mode.iloc[0])
    elif strategie == STRATEGIE_SUPPRIMER:
        resultat = resultat.dropna()
    return resultat.reset_index(drop=True)


# ---------- Préparation numérique ----------

def matrice_numerique(df: pd.DataFrame, colonnes: list[str], standardiser: bool = True) -> np.ndarray:
    X = SimpleImputer(strategy="median").fit_transform(df[colonnes])
    return StandardScaler().fit_transform(X) if standardiser else X


def projection_2d(X: np.ndarray) -> pd.DataFrame:
    coordonnees = PCA(n_components=2, random_state=0).fit_transform(X)
    return pd.DataFrame(coordonnees, columns=["Composante 1", "Composante 2"])
