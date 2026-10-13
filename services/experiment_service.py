"""Enregistrement et lecture des expériences scientifiques (dossier experiments/)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config.settings import DOSSIER_EXPERIENCES
from services.storage_service import chemin_libre, ecrire_json, lire_json, lister_fichiers

MOTIF = "exp_*.json"


def enregistrer_experience(experience: dict, note: str) -> Path:
    chemin = chemin_libre(DOSSIER_EXPERIENCES, "exp", ".json")
    contenu = {
        **experience["meta"],
        "note": note,
        "resultats": json.loads(experience["resultats"].to_json(orient="records")),
    }
    ecrire_json(chemin, contenu)
    return chemin


def compter_experiences() -> int:
    return len(lister_fichiers(DOSSIER_EXPERIENCES, MOTIF))


def lire_historique() -> pd.DataFrame:
    lignes = []
    for fichier in lister_fichiers(DOSSIER_EXPERIENCES, MOTIF):
        try:
            contenu = lire_json(fichier)
        except (OSError, ValueError):
            continue
        resultats = contenu.get("resultats") or [{}]
        meilleur = resultats[0]
        score = meilleur.get("Score moyen")
        lignes.append(
            {
                "Date": str(contenu.get("date", ""))[:16].replace("T", " "),
                "Jeu de données": contenu.get("jeu_de_donnees", ""),
                "Cible": contenu.get("cible", ""),
                "Tâche": contenu.get("tache", ""),
                "Meilleur modèle": meilleur.get("Modèle", ""),
                "Score": round(score, 3) if isinstance(score, (int, float)) else None,
                "Note": contenu.get("note", ""),
            }
        )
    return pd.DataFrame(lignes)
