"""Page du module « generative_models » (en préparation)."""

from __future__ import annotations

from config.settings import MODULE_PAR_DOSSIER
from core.components import page_bientot

MODULE = MODULE_PAR_DOSSIER["generative_models"]


def afficher() -> None:
    page_bientot(MODULE)
