"""
Core Package - Motore AI di Omni Eye
"""

# P3-18: Lazy imports to avoid loading heavy modules at package import
from .memory import ConversationMemory
from .document_processor import DocumentProcessor


def __getattr__(name):
    """Lazy import for AIEngine and AdvancedMemory — cached into globals()"""
    if name == 'AIEngine':
        from .ai_engine import AIEngine
        globals()['AIEngine'] = AIEngine
        return AIEngine
    if name == 'AdvancedMemory':
        from .advanced_memory import AdvancedMemory
        globals()['AdvancedMemory'] = AdvancedMemory
        return AdvancedMemory
    raise AttributeError(f"module 'core' has no attribute {name!r}")


# P3-19: Include AdvancedMemory in __all__
__all__ = ['AIEngine', 'ConversationMemory', 'DocumentProcessor', 'AdvancedMemory']

# Lazy import del Pilot (non fallisce se la config manca)
def get_pilot(*args, **kwargs):
    """Factory per creare un'istanza Pilot"""
    from .ai_pilot import Pilot
    return Pilot(*args, **kwargs)
