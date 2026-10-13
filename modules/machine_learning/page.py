"""Module Machine Learning : entraîner, évaluer et utiliser des modèles prédictifs."""

from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from config.settings import CLASSIFICATION, MODULE_PAR_DOSSIER, REGRESSION
from core.components import afficher_html, entete_page, fmt_entier, fmt_metrique, panneau_donnees
from services.data_service import est_numerique
from services.model_service import (
    MODELES,
    detecter_tache,
    entrainer,
    resume_modele,
    variables_par_defaut,
)
from services.storage_service import sauvegarder_modele

MODULE = MODULE_PAR_DOSSIER["machine_learning"]


def afficher_resultats(resultat: dict) -> None:
    st.markdown("#### Résultats")
    colonnes = st.columns(len(resultat["metriques"]))
    for colonne, (nom, valeur) in zip(colonnes, resultat["metriques"].items()):
        colonne.metric(nom, fmt_metrique(valeur))
    st.caption(
        f"{resultat['nom_modele']} · {fmt_entier(resultat['n_train'])} lignes d'entraînement, "
        f"{fmt_entier(resultat['n_test'])} de test · {resultat['duree']:.2f} s"
    )

    if resultat["tache"] == CLASSIFICATION:
        st.markdown("**Matrice de confusion** (lignes : vraie classe, colonnes : classe prédite)")
        st.dataframe(resultat["confusion"])
    else:
        st.markdown("**Valeurs réelles et valeurs prédites** (les points proches de la diagonale sont bien prédits)")
        st.scatter_chart(
            pd.DataFrame(
                {"Valeur réelle": resultat["reel"], "Valeur prédite": resultat["predit"]}
            ),
            x="Valeur réelle",
            y="Valeur prédite",
        )

    st.markdown("**Variables les plus influentes**")
    st.dataframe(
        resultat["importances"][["Variable", "Importance relative"]].head(15),
        hide_index=True,
        column_config={
            "Importance relative": st.column_config.ProgressColumn(
                "Importance relative", min_value=0.0, max_value=1.0, format="%.2f"
            )
        },
    )


def onglet_entrainement(df: pd.DataFrame) -> None:
    colonnes_df = list(df.columns)
    c1, c2 = st.columns(2)
    cible = c1.selectbox(
        "Variable à prédire", colonnes_df, index=len(colonnes_df) - 1, key="ml_cible"
    )
    auto = detecter_tache(df[cible])
    tache = c2.radio(
        "Type de tâche",
        [CLASSIFICATION, REGRESSION],
        index=0 if auto == CLASSIFICATION else 1,
        horizontal=True,
        key=f"ml_tache_{cible}",
    )
    candidates = [c for c in colonnes_df if c != cible]
    variables = st.multiselect(
        "Variables explicatives",
        candidates,
        default=variables_par_defaut(df, candidates),
        key=f"ml_variables_{cible}",
    )
    c3, c4 = st.columns(2)
    nom_modele = c3.selectbox("Algorithme", list(MODELES[tache]), key=f"ml_algo_{tache}")
    part_test = c4.slider("Part de test (%)", 10, 50, 20, 5, key="ml_test") / 100

    if tache == REGRESSION and not est_numerique(df[cible]):
        st.error("Une régression demande une variable à prédire numérique.")
    elif tache == CLASSIFICATION and df[cible].nunique() > 20:
        st.warning("Cette variable a plus de 20 valeurs distinctes : une régression est peut-être plus adaptée.")

    if st.button("Entraîner le modèle", key="ml_entrainer"):
        if not variables:
            st.warning("Choisis au moins une variable explicative.")
        elif tache == REGRESSION and not est_numerique(df[cible]):
            st.error("Entraînement impossible : la variable à prédire n'est pas numérique.")
        else:
            try:
                with st.spinner("Entraînement en cours…"):
                    resultat = entrainer(df, cible, variables, nom_modele, tache, part_test)
            except Exception as erreur:
                st.error(f"L'entraînement a échoué : {erreur}")
            else:
                st.session_state["modele_courant"] = resultat
                st.session_state["modeles"].append(resume_modele(resultat))

    resultat = st.session_state["modele_courant"]
    if resultat is not None:
        afficher_resultats(resultat)


def onglet_prediction() -> None:
    resultat = st.session_state["modele_courant"]
    if resultat is None:
        st.info("Entraîne d'abord un modèle dans l'onglet « Entraîner et évaluer ».")
        return
    st.markdown(
        f"Modèle : **{resultat['nom_modele']}** · cible : **{resultat['cible']}**. "
        "Renseigne les valeurs, puis lance la prédiction."
    )
    with st.form("formulaire_prediction"):
        colonnes = st.columns(3)
        saisies = {}
        for i, (nom, profil) in enumerate(resultat["profil"].items()):
            with colonnes[i % 3]:
                cle = f"pred_{resultat['id']}_{nom}"
                if profil["type"] == "num":
                    saisies[nom] = st.number_input(nom, value=profil["mediane"], key=cle)
                else:
                    saisies[nom] = st.selectbox(nom, profil["valeurs"], key=cle)
        envoye = st.form_submit_button("Prédire")

    if envoye:
        ligne = pd.DataFrame([saisies])
        pipeline = resultat["pipeline"]
        prediction = pipeline.predict(ligne)[0]
        if resultat["tache"] == REGRESSION:
            texte = f"{float(prediction):,.2f}".replace(",", " ")
        else:
            texte = str(prediction)
        afficher_html(
            f"""
            <div class="resultat">
            <div class="resultat-label">Prédiction pour « {escape(resultat['cible'])} »</div>
            <div class="resultat-valeur">{escape(texte)}</div>
            </div>
            """
        )
        if resultat["tache"] == CLASSIFICATION and hasattr(pipeline, "predict_proba"):
            probabilites = pd.DataFrame(
                {
                    "Classe": [str(c) for c in pipeline.classes_],
                    "Probabilité": pipeline.predict_proba(ligne)[0],
                }
            ).sort_values("Probabilité", ascending=False)
            st.dataframe(
                probabilites,
                hide_index=True,
                column_config={
                    "Probabilité": st.column_config.ProgressColumn(
                        "Probabilité", min_value=0.0, max_value=1.0, format="%.2f"
                    )
                },
            )


def onglet_historique() -> None:
    if not st.session_state["modeles"]:
        st.info("Les modèles entraînés pendant cette session apparaîtront ici.")
    else:
        st.dataframe(pd.DataFrame(st.session_state["modeles"]), hide_index=True)

    resultat = st.session_state["modele_courant"]
    if resultat is not None:
        st.markdown("#### Sauvegarder le modèle courant")
        st.caption(f"{resultat['nom_modele']} · cible : {resultat['cible']} · dossier models/")
        if st.button("Enregistrer dans models/", key="ml_sauvegarder"):
            chemin = sauvegarder_modele(resultat["pipeline"], resultat["nom_modele"])
            st.success(f"Modèle enregistré : models/{chemin.name}")


def afficher() -> None:
    entete_page(MODULE)
    df = panneau_donnees()
    if df is None:
        st.info("Charge un jeu de données pour entraîner un modèle.")
        return
    t1, t2, t3 = st.tabs(["Entraîner et évaluer", "Prédire", "Historique"])
    with t1:
        onglet_entrainement(df)
    with t2:
        onglet_prediction()
    with t3:
        onglet_historique()
