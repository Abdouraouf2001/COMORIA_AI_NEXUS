"""Module Data Mining : explorer, nettoyer et regrouper des données."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from config.settings import (
    MODULE_PAR_DOSSIER,
    STRATEGIE_AUCUNE,
    STRATEGIE_REMPLACER,
    STRATEGIE_SUPPRIMER,
)
from core.components import entete_page, fmt_entier, panneau_donnees
from core.state import definir_donnees, poster_message
from modules.data_mining.clustering import evaluer_k, regrouper
from services.data_service import (
    colonnes_numeriques,
    est_numerique,
    matrice_numerique,
    nettoyer,
    projection_2d,
)

MODULE = MODULE_PAR_DOSSIER["data_mining"]


def onglet_apercu(df: pd.DataFrame) -> None:
    numeriques = colonnes_numeriques(df)
    c = st.columns(4)
    c[0].metric("Lignes", fmt_entier(len(df)))
    c[1].metric("Colonnes", df.shape[1])
    c[2].metric("Numériques", len(numeriques))
    c[3].metric("Catégorielles", df.shape[1] - len(numeriques))
    st.markdown("#### Premières lignes")
    st.dataframe(df.head(100))
    st.markdown("#### Colonnes")
    types = pd.DataFrame(
        {
            "Colonne": df.columns,
            "Type": df.dtypes.astype(str).values,
            "Valeurs distinctes": df.nunique().values,
        }
    )
    st.dataframe(types, hide_index=True)


def onglet_qualite(df: pd.DataFrame) -> None:
    manquantes = df.isna().sum()
    total_manquant = int(manquantes.sum())
    doublons = int(df.duplicated().sum())
    completude = 100 - (total_manquant / max(df.size, 1) * 100)

    c = st.columns(3)
    c[0].metric("Valeurs manquantes", fmt_entier(total_manquant))
    c[1].metric("Lignes en double", fmt_entier(doublons))
    c[2].metric("Complétude", f"{completude:.1f} %")

    if total_manquant:
        tableau = pd.DataFrame(
            {
                "Colonne": df.columns,
                "Manquantes": manquantes.values,
                "% manquant": (manquantes.values / max(len(df), 1) * 100).round(1),
            }
        )
        st.dataframe(tableau[tableau["Manquantes"] > 0], hide_index=True)
    else:
        st.success("Aucune valeur manquante dans ce jeu de données.")

    st.markdown("#### Nettoyer")
    c1, c2 = st.columns(2)
    supprimer = c1.checkbox("Supprimer les lignes en double", value=doublons > 0, key="dm_doublons")
    strategie = c2.selectbox(
        "Valeurs manquantes",
        [STRATEGIE_AUCUNE, STRATEGIE_REMPLACER, STRATEGIE_SUPPRIMER],
        key="dm_strategie",
    )
    if st.button("Appliquer le nettoyage", key="dm_nettoyer"):
        propre = nettoyer(df, supprimer, strategie)
        nom = st.session_state["df_nom"]
        if not nom.endswith("(nettoyé)"):
            nom += " (nettoyé)"
        definir_donnees(propre, nom)
        poster_message(
            f"Nettoyage appliqué : {len(df) - len(propre)} ligne(s) supprimée(s), "
            f"{int(propre.isna().sum().sum())} valeur(s) manquante(s) restante(s)."
        )
        st.rerun()


def onglet_statistiques(df: pd.DataFrame) -> None:
    numeriques = colonnes_numeriques(df)
    if numeriques:
        st.markdown("#### Variables numériques")
        st.dataframe(df[numeriques].describe().T.round(3))
    categorielles = [c for c in df.columns if c not in numeriques]
    if categorielles:
        st.markdown("#### Variables catégorielles")
        st.dataframe(df[categorielles].describe().T.astype(str))

    st.markdown("#### Distribution")
    colonne = st.selectbox("Variable à examiner", list(df.columns), key="dm_variable")
    if est_numerique(df[colonne]):
        valeurs = df[colonne].dropna()
        if valeurs.empty:
            st.info("Cette colonne ne contient aucune valeur.")
        else:
            effectifs, bornes = np.histogram(valeurs, bins=20)
            centres = ((bornes[:-1] + bornes[1:]) / 2).round(3)
            st.bar_chart(pd.DataFrame({"Effectif": effectifs}, index=centres))
    else:
        st.bar_chart(df[colonne].value_counts().head(20).rename("Effectif"))

    if len(numeriques) >= 2:
        st.markdown("#### Variables les plus corrélées")
        corr = df[numeriques].corr()
        masque = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        paires = corr.where(masque).stack().dropna().reset_index()
        paires.columns = ["Variable A", "Variable B", "Corrélation"]
        ordre = paires["Corrélation"].abs().sort_values(ascending=False).index
        st.dataframe(paires.loc[ordre].head(10).round(3), hide_index=True)


def onglet_clustering(df: pd.DataFrame) -> None:
    numeriques = colonnes_numeriques(df)
    if len(numeriques) < 2:
        st.warning("Le clustering demande au moins deux colonnes numériques.")
        return
    variables = st.multiselect(
        "Variables utilisées", numeriques, default=numeriques[:6], key="dm_cluster_vars"
    )
    c1, c2 = st.columns(2)
    k = c1.slider("Nombre de clusters", 2, 10, 3, key="dm_k")
    standardiser = c2.checkbox("Standardiser les variables", value=True, key="dm_standard")
    if len(variables) < 2:
        st.info("Sélectionne au moins deux variables.")
        return
    if len(df) <= k:
        st.warning("Il y a trop peu de lignes pour ce nombre de clusters.")
        return

    X = matrice_numerique(df, variables, standardiser)
    etiquettes, silhouette = regrouper(X, k)

    c = st.columns(3)
    c[0].metric("Clusters", k)
    c[1].metric("Score de silhouette", f"{silhouette:.3f}")
    c[2].metric("Plus grand cluster", fmt_entier(np.bincount(etiquettes).max()))
    st.caption("Plus le score de silhouette est proche de 1, plus les groupes sont bien séparés.")

    projection = projection_2d(X)
    projection["Cluster"] = [f"Cluster {e}" for e in etiquettes]
    st.scatter_chart(
        projection.sample(min(len(projection), 3000), random_state=0),
        x="Composante 1",
        y="Composante 2",
        color="Cluster",
    )

    profil = df[variables].copy()
    profil["Cluster"] = etiquettes
    resume = profil.groupby("Cluster").mean(numeric_only=True).round(2)
    resume.insert(0, "Effectif", np.bincount(etiquettes))
    st.markdown("#### Profil moyen de chaque cluster")
    st.dataframe(resume)

    if st.checkbox("Aider à choisir le nombre de clusters (coude et silhouette)", key="dm_coude"):
        courbes = evaluer_k(X)
        g1, g2 = st.columns(2)
        g1.markdown("**Inertie** : cherche le « coude »")
        g1.line_chart(courbes["Inertie"])
        g2.markdown("**Silhouette** : plus haut, mieux c'est")
        g2.line_chart(courbes["Silhouette"])

    sortie = df.copy()
    sortie["cluster"] = etiquettes
    st.download_button(
        "Télécharger les données avec les clusters (CSV)",
        data=sortie.to_csv(index=False).encode("utf-8"),
        file_name="donnees_clusters.csv",
        mime="text/csv",
        key="dm_telecharger",
    )


def afficher() -> None:
    entete_page(MODULE)
    df = panneau_donnees()
    if df is None:
        st.info("Charge un jeu de données pour commencer l'exploration.")
        return
    t1, t2, t3, t4 = st.tabs(["Aperçu", "Qualité", "Statistiques", "Clustering"])
    with t1:
        onglet_apercu(df)
    with t2:
        onglet_qualite(df)
    with t3:
        onglet_statistiques(df)
    with t4:
        onglet_clustering(df)
