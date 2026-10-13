"""Navigation : barre latérale, module courant et page d'accueil."""

from __future__ import annotations

from html import escape

import streamlit as st

from config.settings import (
    DOSSIER_EXPERIENCES,
    LABELS,
    MODULE_PAR_LABEL,
    MODULES,
    VERSION,
    Module,
)
from core.components import afficher_html, carte, fmt_entier, indicateur
from core.state import aller_a
from services.experiment_service import compter_experiences


def module_courant() -> Module | None:
    """Module sélectionné dans la barre latérale (None = page d'accueil)."""
    return MODULE_PAR_LABEL.get(st.session_state["nav"])


def barre_laterale() -> None:
    with st.sidebar:
        afficher_html(
            """
            <div class="marque-nom">🧠 COMORIA</div>
            <div class="marque-sous">AI NEXUS</div>
            """
        )
        st.caption("Laboratoire intégré d'intelligence artificielle")
        st.radio("Navigation principale", LABELS, key="nav", label_visibility="collapsed")
        st.markdown("---")
        df = st.session_state["df"]
        if df is None:
            afficher_html(
                """
                <div class="statut">
                <div class="statut-titre vide">● Aucun jeu de données</div>
                <div class="statut-texte">Charge un exemple ou un CSV depuis un module.</div>
                </div>
                """
            )
        else:
            afficher_html(
                f"""
                <div class="statut">
                <div class="statut-titre">● Jeu de données actif</div>
                <div class="statut-texte">{escape(str(st.session_state['df_nom']))}<br>
                {fmt_entier(df.shape[0])} lignes × {df.shape[1]} colonnes</div>
                </div>
                """
            )
        st.caption(f"Version {VERSION}")


def page_accueil() -> None:
    afficher_html(
        """
        <div class="hero">
        <div class="hero-label">ComorIA Technologies · Laboratoire d'IA</div>
        <div class="hero-titre">Explore. Build. <span>Intelligence.</span></div>
        <p class="hero-texte">Bienvenue dans COMORIA AI NEXUS, une plateforme intégrée dédiée à
        l'apprentissage, à l'expérimentation et au développement de solutions d'intelligence
        artificielle. Charge des données, entraîne des modèles et compare tes résultats au même endroit.</p>
        </div>
        """
    )

    actifs = sum(1 for m in MODULES if m.actif)
    nom_donnees = st.session_state["df_nom"] or "Aucun"
    colonnes = st.columns(4)
    valeurs = [
        ("Modules disponibles", f"{actifs} / {len(MODULES)}", "Les autres arrivent bientôt"),
        (
            "Jeu de données",
            nom_donnees if len(nom_donnees) <= 18 else nom_donnees[:16] + "…",
            "Partagé entre les modules",
        ),
        ("Modèles entraînés", str(len(st.session_state["modeles"])), "Dans cette session"),
        ("Expériences enregistrées", str(compter_experiences()), f"Dossier {DOSSIER_EXPERIENCES.name}/"),
    ]
    for colonne, (titre, valeur, note) in zip(colonnes, valeurs):
        with colonne:
            indicateur(titre, valeur, note)

    st.markdown("### Démarrage rapide")
    st.markdown(
        "1. **Data Mining** : charge un jeu de données et nettoie-le.\n"
        "2. **Machine Learning** : entraîne un modèle et teste une prédiction.\n"
        "3. **Recherche scientifique** : compare plusieurs modèles et enregistre l'expérience."
    )

    st.markdown("### Laboratoires")
    elements = list(MODULES) + [None]
    for debut in range(0, len(elements), 3):
        colonnes = st.columns(3)
        for colonne, element in zip(colonnes, elements[debut : debut + 3]):
            with colonne:
                if element is None:
                    carte(
                        "🇰🇲",
                        "ComorIA Technologies",
                        "Développer des solutions technologiques adaptées aux besoins locaux.",
                    )
                    continue
                carte(element.icone, element.nom, element.resume, element.actif)
                st.button(
                    "Ouvrir le module" if element.actif else "Voir le plan",
                    key=f"ouvrir_{element.nom}",
                    on_click=aller_a,
                    args=(element.label,),
                )
