"""CortexAI: an experimental brain-inspired cognitive architecture."""
from .config import Settings, load_settings, offline_settings
from .organism import Organism

__all__ = ["Organism", "Settings", "load_settings", "offline_settings"]
__version__ = "0.1.0"
