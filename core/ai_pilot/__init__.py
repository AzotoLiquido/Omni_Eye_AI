"""
AI Pilot - Sistema avanzato di orchestrazione AI

Package che trasforma il semplice chatbot Ollama in un agente
con memoria strutturata, tool execution e pianificazione ReAct.

Uso:
    from core.ai_pilot import Pilot

    pilot = Pilot()
    system_prompt = pilot.build_system_prompt("ciao")
    response, meta = pilot.process("ciao", ai_engine=engine)
"""

from .pilot import Pilot
from .config_loader import PilotConfig
from .memory_store import MemoryStore
from .tool_executor import ToolExecutor, ToolResult
from .prompt_builder import PromptBuilder
from .planner import ReActPlanner, SimplePlanner, PlanStep, create_planner
from .audit_logger import AuditLogger

__all__ = [
    "Pilot",
    "PilotConfig",
    "PromptBuilder",
    "MemoryStore",
    "ToolExecutor",
    "ToolResult",
    "ReActPlanner",
    "SimplePlanner",
    "PlanStep",
    "AuditLogger",
    "create_planner",
]
