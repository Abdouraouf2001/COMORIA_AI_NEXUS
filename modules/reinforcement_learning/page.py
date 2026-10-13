"""Module Reinforcement Learning : un agent apprend à traverser une grille (Q-learning)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from config.settings import MODULE_PAR_DOSSIER
from core.components import afficher_html, entete_page
from modules.reinforcement_learning.agent import (
    FLECHES,
    chemin_glouton,
    entrainer_q,
    generer_trous,
)

MODULE = MODULE_PAR_DOSSIER["reinforcement_learning"]


def html_grille(n: int, trous: set, Q: np.ndarray, chemin: list) -> str:
    sur_le_chemin = set(chemin)
    cases = []
    for i in range(n):
        for j in range(n):
            if (i, j) == (0, 0):
                contenu, classe = "🚩", "g-depart"
            elif (i, j) == (n - 1, n - 1):
                contenu, classe = "🏁", "g-but"
            elif (i, j) in trous:
                contenu, classe = "🕳️", "g-trou"
            else:
                contenu = FLECHES[int(np.argmax(Q[i, j]))] if Q[i, j].any() else "·"
                classe = "g-chemin" if (i, j) in sur_le_chemin else ""
            cases.append(f'<div class="g-case {classe}">{contenu}</div>')
    return (
        f'<div class="grille" style="grid-template-columns:repeat({n},1fr)">'
        + "".join(cases)
        + "</div>"
    )


def afficher() -> None:
    entete_page(MODULE)
    st.markdown(
        "Un agent part du drapeau 🚩 et doit atteindre l'arrivée 🏁 sans tomber dans un trou 🕳️. "
        "Il apprend par essais et erreurs avec l'algorithme **Q-learning** : chaque pas coûte une petite "
        "pénalité, l'arrivée rapporte +1 et un trou fait perdre 1."
    )
    c1, c2, c3 = st.columns(3)
    n = c1.slider("Taille de la grille", 4, 8, 5, key="rl_taille")
    densite = c2.slider("Densité de trous (%)", 0, 30, 15, key="rl_densite") / 100
    graine = c3.number_input("Graine aléatoire", min_value=0, value=7, step=1, key="rl_graine")
    c4, c5, c6, c7 = st.columns(4)
    episodes = c4.slider("Épisodes", 100, 3000, 1000, 100, key="rl_episodes")
    alpha = c5.slider("Taux d'apprentissage", 0.05, 1.0, 0.3, 0.05, key="rl_alpha")
    gamma = c6.slider("Facteur d'actualisation", 0.5, 0.99, 0.95, 0.01, key="rl_gamma")
    epsilon = c7.slider("Exploration (ε)", 0.0, 0.5, 0.1, 0.05, key="rl_epsilon")

    if st.button("Entraîner l'agent", key="rl_lancer"):
        trous = generer_trous(n, densite, int(graine))
        with st.spinner("L'agent s'entraîne…"):
            Q, recompenses = entrainer_q(n, trous, episodes, alpha, gamma, epsilon, int(graine))
        chemin, succes = chemin_glouton(Q, n, trous)
        st.session_state["rl_resultat"] = {
            "n": n, "trous": trous, "Q": Q, "recompenses": recompenses,
            "chemin": chemin, "succes": succes, "episodes": episodes,
        }

    res = st.session_state.get("rl_resultat")
    if res is None:
        st.info("Règle les paramètres, puis lance l'entraînement.")
        return

    c = st.columns(3)
    c[0].metric("Arrivée atteinte", "Oui" if res["succes"] else "Non")
    c[1].metric("Pas du trajet appris", len(res["chemin"]) - 1)
    c[2].metric("Pas minimum possible", 2 * (res["n"] - 1))
    st.markdown("**Politique apprise** (flèche = meilleure action, cases bleues = trajet suivi)")
    afficher_html(html_grille(res["n"], res["trous"], res["Q"], res["chemin"]))
    st.markdown("**Récompense moyenne pendant l'apprentissage**")
    fenetre = max(1, res["episodes"] // 50)
    courbe = pd.Series(res["recompenses"]).rolling(fenetre, min_periods=1).mean()
    st.line_chart(courbe.rename("Récompense moyenne"))
    if not res["succes"]:
        st.caption("L'agent n'a pas encore trouvé le chemin : augmente le nombre d'épisodes ou l'exploration.")
