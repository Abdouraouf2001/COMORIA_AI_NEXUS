"""Page du module « deep_learning » (en préparation)."""

from __future__ import annotations

from config.settings import MODULE_PAR_DOSSIER
from core.components import page_bientot

MODULE = MODULE_PAR_DOSSIER["deep_learning"]


def afficher() -> None:
    page_bientot(MODULE)
