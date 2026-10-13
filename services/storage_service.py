"""Lecture et écriture de fichiers : modèles (.joblib) et expériences (.json)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib

from config.settings import DOSSIER_MODELES


def horodatage() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sauvegarder_modele(pipeline, nom: str, dossier: Path | None = None) -> Path:
    dossier = dossier or DOSSIER_MODELES
    dossier.mkdir(parents=True, exist_ok=True)
    chemin = dossier / f"{nom}_{horodatage()}.joblib".replace(" ", "_")
    joblib.dump(pipeline, chemin)
    return chemin


def charger_modele(chemin: Path):
    return joblib.load(chemin)


def chemin_libre(dossier: Path, prefixe: str, extension: str) -> Path:
    """Renvoie un chemin qui n'existe pas encore (ajoute un numéro si besoin)."""
    dossier.mkdir(parents=True, exist_ok=True)
    base = f"{prefixe}_{horodatage()}"
    chemin = dossier / f"{base}{extension}"
    numero = 2
    while chemin.exists():
        chemin = dossier / f"{base}_{numero}{extension}"
        numero += 1
    return chemin


def ecrire_json(chemin: Path, contenu: dict) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(json.dumps(contenu, ensure_ascii=False, indent=2), encoding="utf-8")


def lire_json(chemin: Path) -> dict:
    return json.loads(chemin.read_text(encoding="utf-8"))


def lister_fichiers(dossier: Path, motif: str) -> list[Path]:
    """Fichiers du dossier correspondant au motif, du plus récent au plus ancien."""
    if not dossier.exists():
        return []
    return sorted(dossier.glob(motif), reverse=True)
