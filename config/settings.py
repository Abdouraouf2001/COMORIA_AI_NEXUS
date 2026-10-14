"""Configuration centrale de COMORIA AI NEXUS (aucune dépendance à Streamlit)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

VERSION = "0.3.0"

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
RACINE = Path(__file__).resolve().parent.parent
DOSSIER_ASSETS = RACINE / "assets"
DOSSIER_MODELES = RACINE / "models"
DOSSIER_EXPERIENCES = RACINE / "experiments"
FICHIER_CSS = DOSSIER_ASSETS / "styles.css"

# ---------------------------------------------------------------------------
# Page Streamlit
# ---------------------------------------------------------------------------
CONFIG_PAGE = {
    "page_title": "COMORIA AI NEXUS",
    "initial_sidebar_state": "expanded",
}

# ---------------------------------------------------------------------------
# Constantes métier
# ---------------------------------------------------------------------------
CLASSIFICATION = "classification"
REGRESSION = "régression"

STRATEGIE_AUCUNE = "Ne rien changer"
STRATEGIE_REMPLACER = "Remplacer (médiane ou valeur la plus fréquente)"
STRATEGIE_SUPPRIMER = "Supprimer les lignes concernées"

GRAINE = 42


# ---------------------------------------------------------------------------
# Modules de la plateforme
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Module:
    nom: str
    icone: str
    dossier: str  # nom du paquet Python dans modules/
    resume: str
    prevu: tuple = ()
    outils: str = ""
    actif: bool = False

    @property
    def label(self) -> str:
        return f"{self.icone} {self.nom}"


MODULES = [
    Module(
        nom="Data Mining", dossier="data_mining",
        resume="Explorer, nettoyer et regrouper des données.",
        actif=True,
    ),
    Module(
        nom="Machine Learning", dossier="machine_learning",
        resume="Entraîner, évaluer et utiliser des modèles prédictifs.",
        actif=True,
    ),
    Module(
        nom="Deep Learning", dossier="deep_learning",
        resume="Réseaux de neurones et apprentissage de représentations.",
        prevu=(
            "Perceptron multicouche sur données tabulaires",
            "Courbes d'apprentissage en direct",
            "Réseaux convolutifs et récurrents",
        ),
        outils="PyTorch ou Keras",
    ),
    Module(
        nom="Modèles génératifs", dossier="generative_models",
        resume="VAE, GAN et modèles de diffusion.",
        prevu=(
            "Auto-encodeur variationnel",
            "Génération de données synthétiques (dossier data/synthetic)",
            "Comparaison données réelles et générées",
        ),
        outils="PyTorch",
    ),
    Module(
        nom="NLP & Assistant IA", dossier="nlp_assistant",
        resume="Langage naturel, assistants conversationnels et RAG.",
        prevu=(
            "Analyse de textes : mots-clés et TF-IDF",
            "Assistant de questions-réponses",
            "RAG sur des documents locaux",
        ),
        outils="scikit-learn, Transformers, base vectorielle",
    ),
    Module(
        nom="Computer Vision", dossier="computer_vision",
        resume="Analyse d'images et reconnaissance visuelle.",
        prevu=(
            "Importer et prétraiter des images",
            "Classifier des images avec un réseau convolutif",
            "Détecter des objets",
        ),
        outils="OpenCV, PyTorch",
    ),
    Module(
        nom="Reinforcement Learning", dossier="reinforcement_learning",
        resume="Agents intelligents, récompenses et prise de décision.",
        actif=True,
    ),
    Module(
        nom="Robotique & HRI", dossier="robotics_hri",
        resume="Simulation robotique, contrôle et interaction humain-robot.",
        prevu=(
            "Simulation d'un bras ou d'un robot mobile",
            "Planification de trajectoire",
            "Commande vocale ou textuelle du robot",
        ),
        outils="PyBullet, Gymnasium",
    ),
    Module(
        nom="Cybersécurité IA", dossier="ai_cybersecurity",
        resume="Détection d'anomalies et analyse intelligente des menaces.",
        actif=True,
    ),
    Module(
        nom="Recherche scientifique", dossier="scientific_research",
        resume="Expériences, comparaison de modèles et résultats reproductibles.",
        actif=True,
    ),
]

MODULE_PAR_NOM = {m.nom: m for m in MODULES}
MODULE_PAR_LABEL = {m.label: m for m in MODULES}
MODULE_PAR_DOSSIER = {m.dossier: m for m in MODULES}

LABEL_ACCUEIL = " Accueil"
LABELS = [LABEL_ACCUEIL] + [m.label for m in MODULES]
