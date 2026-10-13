"""Module Recherche scientifique : expériences reproductibles et comparaison de modèles."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from config.settings import CLASSIFICATION, MODULE_PAR_DOSSIER, REGRESSION
from core.components import entete_page, panneau_donnees
from services.data_service import est_numerique
from services.experiment_service import enregistrer_experience, lire_historique
from services.model_service import MODELES, comparer_modeles, detecter_tache, variables_par_defaut

MODULE = MODULE_PAR_DOSSIER["scientific_research"]


def afficher() -> None:
    entete_page(MODULE)
    df = panneau_donnees()
    if df is None:
        st.info("Charge un jeu de données pour définir une expérience.")
        return

    st.markdown("#### Définir l'expérience")
    colonnes_df = list(df.columns)
    c1, c2 = st.columns(2)
    cible = c1.selectbox("Variable à prédire", colonnes_df, index=len(colonnes_df) - 1, key="rech_cible")
    auto = detecter_tache(df[cible])
    tache = c2.radio(
        "Type de tâche",
        [CLASSIFICATION, REGRESSION],
        index=0 if auto == CLASSIFICATION else 1,
        horizontal=True,
        key=f"rech_tache_{cible}",
    )
    candidates = [c for c in colonnes_df if c != cible]
    variables = st.multiselect(
        "Variables explicatives",
        candidates,
        default=variables_par_defaut(df, candidates),
        key=f"rech_variables_{cible}",
    )
    noms = st.multiselect(
        "Modèles à comparer",
        list(MODELES[tache]),
        default=list(MODELES[tache])[:3],
        key=f"rech_modeles_{tache}",
    )
    plis = st.slider("Nombre de plis de validation croisée", 3, 10, 5, key="rech_plis")

    if st.button("Lancer l'expérience", key="rech_lancer"):
        if not variables or not noms:
            st.warning("Choisis au moins une variable et un modèle.")
        elif tache == REGRESSION and not est_numerique(df[cible]):
            st.error("Une régression demande une variable à prédire numérique.")
        else:
            barre = st.progress(0.0)
            try:
                resultats = comparer_modeles(
                    df, cible, variables, noms, tache, plis, progression=barre.progress
                )
            except Exception as erreur:
                st.error(f"L'expérience a échoué : {erreur}")
            else:
                st.session_state["derniere_experience"] = {
                    "resultats": resultats,
                    "meta": {
                        "date": datetime.now().isoformat(timespec="seconds"),
                        "jeu_de_donnees": st.session_state["df_nom"],
                        "cible": cible,
                        "tache": tache,
                        "variables": variables,
                        "plis": plis,
                        "metrique": "exactitude" if tache == CLASSIFICATION else "R²",
                    },
                }
            finally:
                barre.empty()

    experience = st.session_state["derniere_experience"]
    if experience is not None:
        resultats = experience["resultats"]
        meilleur = resultats.iloc[0]
        st.markdown("#### Résultats")
        st.success(
            f"Meilleur modèle : **{meilleur['Modèle']}** "
            f"({experience['meta']['metrique']} moyenne : {meilleur['Score moyen']:.3f})"
        )
        st.dataframe(resultats.round(3), hide_index=True)
        st.bar_chart(resultats.set_index("Modèle")["Score moyen"])
        note = st.text_area("Hypothèse ou remarques", key="rech_note")
        if st.button("Enregistrer l'expérience dans experiments/", key="rech_enregistrer"):
            chemin = enregistrer_experience(experience, note)
            st.success(f"Expérience enregistrée : experiments/{chemin.name}")

    historique = lire_historique()
    st.markdown("#### Expériences enregistrées")
    if historique.empty:
        st.caption("Aucune expérience enregistrée pour le moment.")
    else:
        st.dataframe(historique, hide_index=True)
