"""Page du module « computer_vision » (en préparation)."""

from __future__ import annotations

from config.settings import MODULE_PAR_DOSSIER
from core.components import page_bientot

MODULE = MODULE_PAR_DOSSIER["computer_vision"]


def afficher() -> None:
    page_bientot(MODULE)
