"""Q-learning sur une grille : environnement, entraînement et trajet appris."""

from __future__ import annotations

import numpy as np

from config.settings import GRAINE

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
FLECHES = ["↑", "↓", "←", "→"]


def _suivant(etat: tuple, action: int, n: int) -> tuple:
    di, dj = ACTIONS[action]
    return (min(max(etat[0] + di, 0), n - 1), min(max(etat[1] + dj, 0), n - 1))


def est_atteignable(n: int, trous: set, depart: tuple, but: tuple) -> bool:
    vus, pile = {depart}, [depart]
    while pile:
        i, j = pile.pop()
        if (i, j) == but:
            return True
        for action in range(4):
            voisin = _suivant((i, j), action, n)
            if voisin not in trous and voisin not in vus:
                vus.add(voisin)
                pile.append(voisin)
    return False


def generer_trous(n: int, densite: float, graine: int) -> set:
    """Place des trous au hasard en garantissant qu'un chemin existe jusqu'à l'arrivée."""
    rng = np.random.default_rng(graine)
    depart, but = (0, 0), (n - 1, n - 1)
    for _ in range(500):
        trous = {
            (i, j)
            for i in range(n)
            for j in range(n)
            if (i, j) not in (depart, but) and rng.random() < densite
        }
        if est_atteignable(n, trous, depart, but):
            return trous
    return set()


def entrainer_q(n, trous, episodes, alpha, gamma, epsilon, graine=GRAINE):
    rng = np.random.default_rng(graine)
    Q = np.zeros((n, n, 4))
    but = (n - 1, n - 1)
    recompenses = []
    for _ in range(episodes):
        etat, total = (0, 0), 0.0
        for _ in range(4 * n * n):
            if rng.random() < epsilon:
                action = int(rng.integers(4))
            else:
                valeurs = Q[etat]
                action = int(rng.choice(np.flatnonzero(valeurs == valeurs.max())))
            suivant = _suivant(etat, action, n)
            if suivant in trous:
                recompense, fini = -1.0, True
            elif suivant == but:
                recompense, fini = 1.0, True
            else:
                recompense, fini = -0.01, False
            objectif = recompense if fini else recompense + gamma * Q[suivant].max()
            Q[etat][action] += alpha * (objectif - Q[etat][action])
            total += recompense
            etat = suivant
            if fini:
                break
        recompenses.append(total)
    return Q, recompenses


def chemin_glouton(Q: np.ndarray, n: int, trous: set) -> tuple[list, bool]:
    """Suit la meilleure action à chaque case : renvoie le trajet et s'il atteint l'arrivée."""
    etat, but = (0, 0), (n - 1, n - 1)
    chemin = [etat]
    for _ in range(4 * n * n):
        etat = _suivant(etat, int(np.argmax(Q[etat])), n)
        chemin.append(etat)
        if etat in trous:
            return chemin, False
        if etat == but:
            return chemin, True
    return chemin, False
