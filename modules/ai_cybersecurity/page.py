"""Module Cybersécurité IA : détection d'anomalies."""

from __future__ import annotations

import numpy as np
import streamlit as st

from config.settings import MODULE_PAR_DOSSIER
from core.components import entete_page, fmt_entier, panneau_donnees
from modules.ai_cybersecurity.detection import detecter_anomalies
from services.data_service import colonnes_numeriques, matrice_numerique, projection_2d

MODULE = MODULE_PAR_DOSSIER["ai_cybersecurity"]


def afficher() -> None:
    entete_page(MODULE)
    df = panneau_donnees()
    if df is None:
        st.info("Charge un jeu de données (journaux, transactions, mesures…) pour y chercher des anomalies.")
        return
    numeriques = colonnes_numeriques(df)
    if len(numeriques) < 2:
        st.warning("La détection d'anomalies demande au moins deux colonnes numériques.")
        return

    variables = st.multiselect(
        "Variables à surveiller", numeriques, default=numeriques[:8], key="cy_variables"
    )
    taux = st.slider("Part d'anomalies attendue (%)", 1, 20, 5, key="cy_taux") / 100
    if len(variables) < 2:
        st.info("Sélectionne au moins deux variables.")
        return

    X = matrice_numerique(df, variables, True)
    etiquettes, scores = detecter_anomalies(X, taux)
    est_anomalie = etiquettes == -1

    c = st.columns(3)
    c[0].metric("Observations analysées", fmt_entier(len(df)))
    c[1].metric("Anomalies signalées", fmt_entier(est_anomalie.sum()))
    c[2].metric("Part signalée", f"{est_anomalie.mean() * 100:.1f} %")

    projection = projection_2d(X)
    projection["Statut"] = np.where(est_anomalie, "Anomalie", "Normal")
    st.scatter_chart(
        projection.sample(min(len(projection), 3000), random_state=0),
        x="Composante 1",
        y="Composante 2",
        color="Statut",
    )

    resultat = df.copy()
    resultat["score_anomalie"] = scores.round(4)
    resultat["anomalie"] = est_anomalie
    st.markdown("#### Observations les plus inhabituelles")
    st.dataframe(resultat.sort_values("score_anomalie", ascending=False).head(20))
    st.caption(
        "Cette méthode n'utilise aucune étiquette : elle signale ce qui s'écarte du reste des données. "
        "Chaque alerte doit être vérifiée par une personne."
    )
    st.download_button(
        "Télécharger les anomalies (CSV)",
        data=resultat[est_anomalie].to_csv(index=False).encode("utf-8"),
        file_name="anomalies.csv",
        mime="text/csv",
        key="cy_telecharger",
    )
