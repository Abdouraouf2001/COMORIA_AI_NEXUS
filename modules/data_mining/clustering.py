"""Clustering K-Means pour le module Data Mining."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from config.settings import GRAINE


def regrouper(X: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    """K-Means : renvoie les étiquettes de cluster et le score de silhouette."""
    etiquettes = KMeans(n_clusters=k, n_init=10, random_state=GRAINE).fit_predict(X)
    silhouette = silhouette_score(
        X, etiquettes, sample_size=min(len(X), 2000), random_state=GRAINE
    )
    return etiquettes, float(silhouette)


@st.cache_data(show_spinner=False)
def evaluer_k(X: np.ndarray, k_max: int = 8) -> pd.DataFrame:
    """Inertie et silhouette pour k = 2…k_max (aide au choix du nombre de clusters)."""
    lignes = []
    for k in range(2, min(k_max, len(X) - 1) + 1):
        modele = KMeans(n_clusters=k, n_init=10, random_state=GRAINE).fit(X)
        silhouette = silhouette_score(
            X, modele.labels_, sample_size=min(len(X), 2000), random_state=GRAINE
        )
        lignes.append({"k": k, "Inertie": modele.inertia_, "Silhouette": silhouette})
    return pd.DataFrame(lignes).set_index("k")
