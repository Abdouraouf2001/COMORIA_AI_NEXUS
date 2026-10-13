"""
COMORIA AI NEXUS — point d'entrée.

Lancer :  streamlit run app.py

Ce fichier ne contient presque rien : la configuration est dans config/,
l'interface commune dans core/, la logique dans services/ et chaque domaine
de l'IA dans son propre dossier de modules/.
"""

import streamlit as st

from config.settings import CONFIG_PAGE

# set_page_config doit être la toute première commande Streamlit.
st.set_page_config(**CONFIG_PAGE)

from core.orchestrateur import main  # noqa: E402

main()
