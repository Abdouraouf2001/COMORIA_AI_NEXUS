"""Page du module « robotics_hri » (en préparation)."""

from __future__ import annotations

from config.settings import MODULE_PAR_DOSSIER
from core.components import page_bientot

MODULE = MODULE_PAR_DOSSIER["robotics_hri"]


def afficher() -> None:
    page_bientot(MODULE)
