"""
Planner ReAct - Orchestrazione ragionamento/azione per il Pilot

Implementa il ciclo:
  1. Pensiero  → ragionamento del modello
  2. Azione    → invocazione di un tool
  3. Osservazione → risultato del tool
  4. Ripeti o → Risposta Finale

Ottimizzato per modelli locali (7B-13B) con parsing robusto.
"""

import re
import json
import threading
from typing import Dict, List, Optional, Tuple

from .config_loader import PilotConfig
from .tool_executor import ToolExecutor, ToolResult


# ------------------------------------------------------------------
# Regex per parsing dell'output del modello
# ------------------------------------------------------------------

# Cattura: Pensiero: ... o Thought: ...
RE_THOUGHT = re.compile(
    r"(?:Pensiero|Thought)\s*:\s*(.+?)(?=(?:Azione|Action|Risposta Finale|Final Answer)|$)",
    re.DOTALL | re.IGNORECASE,
)

# P1-7: Use greedy match .*  to capture to LAST ')' on line, handling nested parens in JSON
RE_ACTION = re.compile(
    r"(?:Azione|Action)\s*:\s*(\w+)\s*\((.*)\)\s*$",
    re.MULTILINE | re.IGNORECASE,
)

# Cattura: Risposta Finale: ... o Final Answer: ...
RE_FINAL = re.compile(
    r"(?:Risposta Finale|Final Answer)\s*:\s*(.+)",
    re.DOTALL | re.IGNORECASE,
)


class PlanStep:
    """Un singolo passo del piano ReAct"""

    def __init__(self, step_num: int):
        self.step_num = step_num
        self.thought: str = ""
        self.action: Optional[str] = None       # tool_id
        self.action_params: Dict = {}
        self.observation: str = ""
        self.is_final: bool = False
        self.final_answer: str = ""
        self.raw_output: str = ""

    def to_dict(self) -> Dict:
        return {
            "step": self.step_num,
            "thought": self.thought,
            "action": self.action,
            "action_params": self.action_params,
            "observation": self.observation,
            "is_final": self.is_final,
            "final_answer": self.final_answer,
        }


class ReActPlanner:
    """Planner con ciclo ReAct multi-step"""

    def __init__(self, cfg: PilotConfig, tool_executor: ToolExecutor):
        self.cfg = cfg
        self.tool_executor = tool_executor
        self.max_steps = cfg.planner_max_steps
        self.steps: List[PlanStep] = []
        self._lock = threading.Lock()  # P1-7: thread safety per richieste concorrenti

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def needs_planning(self, user_message: str, available_tools: List[Dict]) -> bool:
        """
        Euristica a punteggio: determina se il messaggio necessita
        del ciclo ReAct (tool) o può essere risposto direttamente.
        Richiede almeno 2 keyword match per attivare (riduce falsi positivi).
        """
        if not available_tools:
            return False

        # Keyword bilingue che suggeriscono necessità di tool
        tool_keywords = [
            # Italiano
            "leggi il file", "apri il file", "esegui codice", "esegui script",
            "cerca in memoria", "cerca nel", "lista directory", "lista cartella",
            "ricorda che", "salva come", "analizza documento", "crea task",
            "scrivi file", "calcola",
            # Inglese
            "read file", "open file", "execute code", "run script",
            "search memory", "list directory", "remember that", "save as",
            "analyze document", "create task", "write file", "calculate",
        ]
        # Keyword singole (peso minore)
        weak_keywords = [
            "file", "memoria", "task", "documento", "codice", "script",
            "memory", "document", "code",
        ]

        msg_lower = user_message.lower()
        score = 0
        # Keyword forti valgono 2
        matched_strong = set()
        for kw in tool_keywords:
            if kw in msg_lower:
                score += 2
                matched_strong.update(kw.split())
        # Keyword deboli valgono 1 (skip se già coperte da una forte)
        score += sum(1 for kw in weak_keywords
                     if kw in msg_lower and kw not in matched_strong)

        return score >= 3

    def parse_model_output(self, output: str) -> PlanStep:
        """
        Parsa l'output del modello ed estrae pensiero, azione o risposta finale.
        Robusto per modelli locali che non sempre seguono il formato esatto.
        """
        step = PlanStep(len(self.steps) + 1)
        step.raw_output = output

        # 1. Cerca Risposta Finale
        match_final = RE_FINAL.search(output)
        if match_final:
            step.is_final = True
            step.final_answer = match_final.group(1).strip()
            return step

        # 2. Cerca Pensiero
        match_thought = RE_THOUGHT.search(output)
        if match_thought:
            step.thought = match_thought.group(1).strip()

        # 3. Cerca Azione
        match_action = RE_ACTION.search(output)
        if match_action:
            step.action = match_action.group(1).strip()
            raw_params = match_action.group(2).strip()
            step.action_params = self._parse_params(raw_params)
        else:
            # Fallback: cerca pattern semplificato  tool_name: param
            step.action, step.action_params = self._fallback_action_parse(output)

        # Se nessuna azione e nessun pensiero specifico, tratta come risposta diretta
        if not step.action and not step.thought:
            step.is_final = True
            step.final_answer = output.strip()

        return step

    def execute_step(self, step: PlanStep) -> Tuple[str, bool]:
        """Esegue l'azione di un PlanStep e restituisce (osservazione, success)"""
        if not step.action:
            return "", True

        result = self.tool_executor.execute(step.action, step.action_params)
        # P0-3 fix: ritorna il booleano success dal ToolResult
        if result.success:
            step.observation = result.output
        else:
            step.observation = f"ERRORE [{result.tool_id}]: {result.error}"
        self.steps.append(step)

        return step.observation, result.success

    def build_continuation_prompt(self, step: PlanStep) -> str:
        """
        Costruisce il prompt per chiedere al modello il passo successivo,
        includendo il risultato dell'osservazione.
        """
        lines = []

        if step.thought:
            lines.append(f"Pensiero: {step.thought}")
        if step.action:
            lines.append(f"Azione: {step.action}({json.dumps(step.action_params, ensure_ascii=False)})")
        if step.observation:
            lines.append(f"Osservazione: {step.observation}")

        lines.append("")
        lines.append(
            "Basandoti sull'osservazione, prosegui con un altro Pensiero/Azione "
            "oppure fornisci la Risposta Finale."
        )

        return "\n".join(lines)

    def get_history(self) -> List[Dict]:
        """Restituisce lo storico dei passi eseguiti"""
        with self._lock:
            return [s.to_dict() for s in self.steps]

    def reset(self) -> None:
        """Resetta lo stato del planner per una nuova richiesta"""
        with self._lock:
            self.steps = []

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_params(self, raw: str) -> Dict:
        """Tenta di parsare i parametri come JSON, con fallback"""
        raw = raw.strip()

        # Caso 1: JSON valido
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Caso 2: JSON con apici singoli → sostituisci solo quelli che delimitano
        # chiavi/valori (non apostrofi dentro le parole)
        # P2: Smarter quote replacement to avoid corrupting Italian apostrophes
        try:
            # Replace only single quotes adjacent to JSON structural chars
            fixed = re.sub(r"(?<=[:,\[{])\s*'|'\s*(?=[}\]:,])", '"', raw)
            # Also handle opening quote at start
            if fixed.startswith("{") or fixed.startswith("["):
                return json.loads(fixed)
        except (json.JSONDecodeError, ValueError):
            pass

        # Caso 3: Singolo valore stringa → parametro "query" o "path"
        if not raw.startswith("{"):
            # Se sembra un percorso file
            if "/" in raw or "\\" in raw or "." in raw:
                return {"path": raw.strip('"').strip("'")}
            else:
                return {"query": raw.strip('"').strip("'")}

        return {"raw": raw}

    def _fallback_action_parse(self, output: str) -> Tuple[Optional[str], Dict]:
        """
        Parsing fallback per modelli che non seguono il formato esatto.
        Cerca pattern come: "uso lo strumento fs per leggere..."
        """
        output_lower = output.lower()

        # Pattern: "uso/utilizzo <tool>" o "cerco in memoria"
        tool_patterns = {
            "fs":   [
                r"legg[io].*file", r"apri.*file", r"list[ao].*director", r"list[ao].*cartell",
                r"read.*file", r"open.*file", r"list.*director",
            ],
            "py":   [
                r"esegu[io].*codice", r"esegu[io].*python", r"calcol[ao]",
                r"run.*code", r"execute.*python", r"calculat",
            ],
            "db":   [
                r"cerc[ao].*memori", r"ricord[ao]", r"fatt[io].*not[io]",
                r"search.*memor", r"remember", r"find.*fact",
            ],
        }

        for tool_id, patterns in tool_patterns.items():
            for pat in patterns:
                match = re.search(pat, output_lower)
                if match:
                    # Prova a estrarre un parametro dal contesto
                    # Cerca testo tra virgolette
                    quoted = re.findall(r'["\']([^"\']+)["\']', output)
                    if quoted:
                        if tool_id == "fs":
                            return tool_id, {"action": "read", "path": quoted[0]}
                        elif tool_id == "db":
                            return tool_id, {"action": "search", "query": quoted[0]}
                        elif tool_id == "py":
                            return tool_id, {"code": quoted[0]}
                    # Nessun parametro trovato - provide sensible defaults
                    if tool_id == "fs":
                        return tool_id, {"action": "list", "path": "."}
                    elif tool_id == "db":
                        # Extract a search hint from the output itself
                        words = [w for w in output.split() if len(w) > 3]
                        hint = " ".join(words[:5]) if words else "ricerca"
                        return tool_id, {"action": "search", "query": hint}
                    return tool_id, {}

        return None, {}


class SimplePlanner:
    """
    Planner semplificato per quando la strategia è 'simple'.
    Risponde direttamente senza ciclo ReAct.
    """

    def needs_planning(self, user_message: str, available_tools: List[Dict]) -> bool:
        return False

    def parse_model_output(self, output: str) -> PlanStep:
        step = PlanStep(1)
        step.is_final = True
        step.final_answer = output
        return step

    def reset(self) -> None:
        pass

    def get_history(self) -> List[Dict]:
        """P2-4: interfaccia coerente con ReActPlanner"""
        return []


def create_planner(cfg: PilotConfig, tool_executor: ToolExecutor):
    """Factory: crea il planner giusto in base alla strategia configurata"""
    strategy = cfg.planner_strategy
    if strategy == "simple":
        return SimplePlanner()
    elif strategy in ("react", "tree_of_thought"):
        # tree_of_thought currently uses ReActPlanner (same implementation)
        return ReActPlanner(cfg, tool_executor)
    else:
        return SimplePlanner()
