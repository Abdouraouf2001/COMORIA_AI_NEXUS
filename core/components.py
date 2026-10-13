"""Composants d'interface réutilisables par tous les modules."""

from __future__ import annotations

from datetime import datetime
from html import escape

import pandas as pd
import streamlit as st

from config.settings import FICHIER_CSS, Module
from core.state import definir_donnees, retirer_message
from services.data_service import EXEMPLES, lire_csv


# ---------- Affichage ----------

def afficher_html(code: str) -> None:
    """Affiche du HTML sans que Markdown le prenne pour un bloc de code.

    Markdown traite toute ligne indentée de 4 espaces comme du code et les lignes
    vides cassent les blocs HTML : on retire donc l'indentation et les lignes vides.
    """
    lignes = [ligne.strip() for ligne in code.splitlines() if ligne.strip()]
    st.markdown("\n".join(lignes), unsafe_allow_html=True)


def charger_design() -> None:
    """Charge le design CSS de COMORIA AI NEXUS depuis assets/styles.css."""
    if FICHIER_CSS.exists():
        css = FICHIER_CSS.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        st.warning("Fichier assets/styles.css introuvable : le design n'est pas chargé.")


def fmt_entier(valeur: int) -> str:
    return f"{int(valeur):,}".replace(",", " ")


def fmt_metrique(valeur: float) -> str:
    return fmt_entier(round(valeur)) if abs(valeur) >= 100 else f"{valeur:.3f}"


def entete_page(module: Module) -> None:
    afficher_html(
        f"""
        <div class="page-titre">{module.icone} {escape(module.nom)}</div>
        <div class="page-sous-titre">{escape(module.resume)}</div>
        """
    )


def carte(icone: str, titre: str, description: str, actif: bool | None = None) -> None:
    pastille = ""
    if actif is True:
        pastille = '<span class="pill pill-on">Disponible</span>'
    elif actif is False:
        pastille = '<span class="pill pill-soon">En préparation</span>'
    afficher_html(
        f"""
        <div class="card">
        <div class="card-top"><span class="card-icon">{icone}</span>{pastille}</div>
        <div class="card-title">{escape(titre)}</div>
        <div class="card-text">{escape(description)}</div>
        </div>
        """
    )


def indicateur(titre: str, valeur: str, note: str) -> None:
    afficher_html(
        f"""
        <div class="metric-card">
        <div class="metric-label">{escape(titre)}</div>
        <div class="metric-value">{escape(valeur)}</div>
        <div class="metric-note">{escape(note)}</div>
        </div>
        """
    )


def pied_de_page() -> None:
    afficher_html(
        f"""
        <div class="pied">
        <strong>COMORIA AI NEXUS</strong><br>
        Une initiative de COMORIA Technologies<br>
        L'intelligence artificielle au service de l'innovation.<br>
        © {datetime.now().year} · Laboratoire universitaire d'intelligence artificielle
        </div>
        """
    )


# ---------- Jeu de données partagé ----------

def panneau_donnees() -> pd.DataFrame | None:
    """Sélecteur de jeu de données partagé par tous les modules."""
    message = retirer_message()
    if message:
        st.success(message)

    df = st.session_state["df"]
    titre = "📁 Jeu de données"
    if df is not None:
        titre += (
            f" : {st.session_state['df_nom']} "
            f"({fmt_entier(df.shape[0])} lignes × {df.shape[1]} colonnes)"
        )
    with st.expander(titre, expanded=df is None):
        source = st.radio(
            "Source", ["Exemple intégré", "Importer un CSV"], horizontal=True, key="source_donnees"
        )
        if source == "Exemple intégré":
            choix = st.selectbox("Exemple", list(EXEMPLES), key="choix_exemple")
            if st.button("Charger cet exemple", key="btn_exemple"):
                definir_donnees(EXEMPLES[choix](), choix)
                st.rerun()
        else:
            fichier = st.file_uploader("Fichier CSV", type=["csv"], key="upload_csv")
            if fichier is not None and st.session_state["df_nom"] != fichier.name:
                try:
                    definir_donnees(lire_csv(fichier.getvalue()), fichier.name)
                except ValueError as erreur:
                    st.error(str(erreur))
                else:
                    st.rerun()
    return st.session_state["df"]


# ---------- Modules en préparation ----------

def page_bientot(module: Module) -> None:
    entete_page(module)
    st.info("Ce module n'est pas encore développé. Voici ce qui est prévu.")
    st.markdown("#### Fonctionnalités prévues")
    for element in module.prevu:
        st.markdown(f"- {element}")
    if module.outils:
        st.caption(f"Outils envisagés : {module.outils}")
