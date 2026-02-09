"""
Core Package - Motore AI di Omni Eye
"""

from .ai_engine import AIEngine
from .memory import ConversationMemory
from .document_processor import DocumentProcessor

__all__ = ['AIEngine', 'ConversationMemory', 'DocumentProcessor']

# Lazy import del Pilot (non fallisce se la config manca)
def get_pilot(*args, **kwargs):
    """Factory per creare un'istanza Pilot"""
    from .ai_pilot import Pilot
    return Pilot(*args, **kwargs)
