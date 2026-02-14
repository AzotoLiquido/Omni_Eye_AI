"""
Model Router — Instradamento intelligente delle richieste al modello più adatto.

Classifica l'intento dell'utente (general, code, vision) e seleziona
automaticamente il modello Ollama migliore, evitando swap inutili.

Design:
  - Classificatore a regole (zero ML, <1ms) — niente overhead
  - Tracking del modello attualmente caricato in VRAM ("warm model")
  - Fallback al modello di default se quello specializzato non è installato
"""

import logging
import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# =========================================================================
# Intent classification
# =========================================================================

class Intent(Enum):
    """Categoria di intento rilevata dal messaggio utente."""
    GENERAL = "general"
    CODE = "code"
    VISION = "vision"


# --- Pattern per rilevamento codice ---
# Keyword esplicite di programmazione / richiesta codice
_CODE_KEYWORDS = re.compile(
    r'\b(?:'
    # Richieste esplicite
    r'scrivi\s+(?:un\s+)?(?:codice|script|programma|funzione|classe|metodo)'
    r'|genera\s+(?:un\s+)?(?:codice|script|programma|funzione|classe)'
    r'|implementa|refactor|debug(?:ga)?|compila|deploya'
    # Linguaggi di programmazione
    r'|python|javascript|typescript|java\b|c\+\+|c#|rust|golang|go\b|ruby'
    r'|html|css|sql|bash|shell|powershell|kotlin|swift|dart|php|perl'
    # Concetti tecnici di programmazione
    r'|funzione|classe|metodo|variabile|array|lista|dizionario|loop|ciclo'
    r'|api\b|endpoint|database|query|regex|regexp'
    r'|bug|errore\s+(?:nel\s+)?codice|stack\s?trace|exception|traceback'
    r'|import|require|from\s+\w+\s+import|def\s+\w+|class\s+\w+'
    r'|git\b|commit|branch|merge|docker|container|pip|npm|yarn'
    r'|algoritmo|struttura\s+dati|ricorsione|sorting|binary\s+search'
    r')\b',
    re.IGNORECASE,
)

# Code fence (```python, ```js, ecc.) o blocchi di codice inline
_CODE_FENCE = re.compile(r'```\w*\s*\n|`[^`]+`')

# Indentazione tipica di codice (def, for, if, class con indentazione)
_CODE_INDENT = re.compile(r'^\s{2,}(?:def |for |if |class |return |import )', re.MULTILINE)


def classify_intent(message: str, *, has_images: bool = False) -> Intent:
    """
    Classifica l'intento dell'utente in modo deterministico.

    Priority: VISION > CODE > GENERAL

    Args:
        message: Testo del messaggio utente
        has_images: True se il messaggio contiene immagini allegate

    Returns:
        Intent enum (GENERAL, CODE, VISION)
    """
    if has_images:
        return Intent.VISION

    if not message or not message.strip():
        return Intent.GENERAL

    text = message.strip()

    # Code fence esplicito → sicuramente codice
    if _CODE_FENCE.search(text):
        return Intent.CODE

    # Indentazione tipica di codice → sicuramente codice
    if _CODE_INDENT.search(text):
        return Intent.CODE

    # Keyword di programmazione
    if _CODE_KEYWORDS.search(text):
        return Intent.CODE

    return Intent.GENERAL


# =========================================================================
# Router configuration
# =========================================================================

@dataclass
class ModelMapping:
    """Mappa intento → modello Ollama preferito."""
    general: str = "gemma2:9b"
    code: str = "qwen2.5-coder:7b"
    vision: str = "minicpm-v"

    def get(self, intent: Intent) -> str:
        """Restituisce il modello per l'intento specificato."""
        return {
            Intent.GENERAL: self.general,
            Intent.CODE: self.code,
            Intent.VISION: self.vision,
        }[intent]

    def all_models(self) -> List[str]:
        """Lista di tutti i modelli configurati (unici)."""
        return list(dict.fromkeys([self.general, self.code, self.vision]))


# =========================================================================
# Model Router
# =========================================================================

class ModelRouter:
    """
    Seleziona il modello Ollama ottimale per ogni richiesta.

    Tiene traccia del modello "warm" (caricato in VRAM) per minimizzare
    gli swap costosi (~5-15 secondi).
    """

    def __init__(
        self,
        mapping: ModelMapping = None,
        fallback_model: str = None,
    ):
        """
        Args:
            mapping: Mappa intento→modello. Default usa gemma2:9b / qwen2.5-coder / minicpm-v.
            fallback_model: Modello di fallback se quello preferito non è installato.
        """
        self.mapping = mapping or ModelMapping()
        self.fallback = fallback_model or "llama3.2"
        self._warm_model: Optional[str] = None
        self._lock = threading.Lock()
        self._installed_cache: Optional[List[str]] = None
        self._installed_cache_lock = threading.Lock()
        logger.info(
            "ModelRouter inizializzato: general=%s, code=%s, vision=%s, fallback=%s",
            self.mapping.general, self.mapping.code, self.mapping.vision, self.fallback,
        )

    def refresh_installed(self, available_models: List[str]) -> None:
        """Aggiorna la cache dei modelli installati (chiamata a inizio sessione)."""
        with self._installed_cache_lock:
            self._installed_cache = [m.lower() for m in available_models]
        logger.debug("Cache modelli aggiornata: %d modelli", len(available_models))

    def _is_installed(self, model: str) -> bool:
        """Controlla se un modello è disponibile localmente."""
        with self._installed_cache_lock:
            if self._installed_cache is None:
                return True  # Se non abbiamo la cache, assumiamo sia installato
            model_lower = model.lower().split(":")[0]
            for m in self._installed_cache:
                m_base = m.split(":")[0]
                if model_lower == m_base or model.lower() == m:
                    return True
            return False

    def route(
        self,
        message: str,
        *,
        has_images: bool = False,
        force_intent: Intent = None,
    ) -> "RouteResult":
        """
        Determina il modello da usare per questa richiesta.

        Args:
            message: Messaggio utente
            has_images: True se ci sono immagini allegate
            force_intent: Forza un intento specifico (per testing/override)

        Returns:
            RouteResult con modello selezionato e metadata
        """
        intent = force_intent or classify_intent(message, has_images=has_images)
        preferred = self.mapping.get(intent)

        # Controlla se il modello preferito è installato
        if self._is_installed(preferred):
            model = preferred
            fallback_used = False
        else:
            logger.warning(
                "Modello %s non installato per intent=%s, fallback a %s",
                preferred, intent.value, self.fallback,
            )
            model = self.fallback
            fallback_used = True

        # Calcola se ci sarà uno swap di modello
        with self._lock:
            is_swap = self._warm_model is not None and self._warm_model != model
            self._warm_model = model

        if is_swap:
            logger.info(
                "Model swap: %s → %s (intent=%s)",
                self._warm_model, model, intent.value,
            )

        return RouteResult(
            model=model,
            intent=intent,
            is_swap=is_swap,
            fallback_used=fallback_used,
        )

    @property
    def warm_model(self) -> Optional[str]:
        """Modello attualmente caricato in VRAM."""
        with self._lock:
            return self._warm_model

    @warm_model.setter
    def warm_model(self, model: str) -> None:
        """Aggiorna manualmente il modello warm (es. dopo change_model da UI)."""
        with self._lock:
            self._warm_model = model


@dataclass
class RouteResult:
    """Risultato del routing — modello selezionato + metadata."""
    model: str
    intent: Intent
    is_swap: bool = False
    fallback_used: bool = False

    def to_dict(self) -> Dict:
        return {
            "model": self.model,
            "intent": self.intent.value,
            "is_swap": self.is_swap,
            "fallback_used": self.fallback_used,
        }
