"""Orchestrateur : relie la page choisie au bon module.

Chaque dossier de modules/ expose un fichier page.py avec une fonction afficher().
Pour ajouter un domaine, il suffit de le déclarer dans config/settings.py (MODULES)
et de créer modules/<dossier>/page.py : aucune autre modification n'est nécessaire.
"""

from __future__ import annotations

import importlib

import streamlit as st

from config.settings import Module
from core.components import charger_design, pied_de_page
from core.navigation import barre_laterale, module_courant, page_accueil
from core.state import init_etat


def charger_page(module: Module):
    """Importe modules/<dossier>/page.py ; affiche une erreur claire si l'import échoue."""
    try:
        return importlib.import_module(f"modules.{module.dossier}.page")
    except Exception as erreur:  # dépendance manquante, faute de frappe…
        st.error(f"Le module « {module.nom} » n'a pas pu être chargé.")
        st.exception(erreur)
        return None


def main() -> None:
    init_etat()
    charger_design()
    barre_laterale()

    module = module_courant()
    if module is None:
        page_accueil()
    else:
        page = charger_page(module)
        if page is not None:
            page.afficher()
    pied_de_page()
