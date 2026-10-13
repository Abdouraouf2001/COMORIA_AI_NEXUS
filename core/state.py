"""État de session (st.session_state) partagé entre les modules."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config.settings import LABEL_ACCUEIL


def init_etat() -> None:
    st.session_state.setdefault("nav", LABEL_ACCUEIL)
    st.session_state.setdefault("df", None)
    st.session_state.setdefault("df_nom", None)
    st.session_state.setdefault("modeles", [])
    st.session_state.setdefault("modele_courant", None)
    st.session_state.setdefault("derniere_experience", None)
    st.session_state.setdefault("rl_resultat", None)


def aller_a(label: str) -> None:
    """Change de page (utilisable comme on_click d'un bouton)."""
    st.session_state["nav"] = label


def definir_donnees(df: pd.DataFrame, nom: str) -> None:
    """Définit le jeu de données actif et réinitialise les résultats qui en dépendent."""
    df = df.copy()
    for colonne in df.columns:
        if pd.api.types.is_bool_dtype(df[colonne]):
            df[colonne] = df[colonne].map({True: "oui", False: "non"})
    st.session_state["df"] = df
    st.session_state["df_nom"] = nom
    st.session_state["modele_courant"] = None
    st.session_state["derniere_experience"] = None


def poster_message(texte: str) -> None:
    """Message affiché une seule fois après le prochain rechargement."""
    st.session_state["message"] = texte


def retirer_message() -> str | None:
    return st.session_state.pop("message", None)
