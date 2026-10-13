"""Page du module « nlp_assistant » (en préparation)."""

from __future__ import annotations

from config.settings import MODULE_PAR_DOSSIER
from core.components import page_bientot

MODULE = MODULE_PAR_DOSSIER["nlp_assistant"]


def afficher() -> None:
    page_bientot(MODULE)
