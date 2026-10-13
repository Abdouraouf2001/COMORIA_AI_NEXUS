"""Détection d'anomalies (Isolation Forest) pour le module Cybersécurité IA."""

from __future__ import annotations

import numpy as np
import streamlit as st
from sklearn.ensemble import IsolationForest

from config.settings import GRAINE


@st.cache_data(show_spinner=False)
def detecter_anomalies(X: np.ndarray, contamination: float) -> tuple[np.ndarray, np.ndarray]:
    """Renvoie (étiquettes, scores) : étiquette -1 = anomalie, score élevé = plus inhabituel."""
    modele = IsolationForest(n_estimators=200, contamination=contamination, random_state=GRAINE)
    etiquettes = modele.fit_predict(X)
    scores = -modele.score_samples(X)
    return etiquettes, scores
