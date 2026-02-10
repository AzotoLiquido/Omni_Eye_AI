"""
Pilot - Orchestratore principale del sistema AI-Pilot

Collega tutti i componenti:
  - Config     → caricamento e accesso tipizzato
  - Prompt     → generazione system prompt dinamici
  - Memory     → retrieval fatti/documenti SQLite + FTS5
  - Tools      → esecuzione sandboxed (filesystem, python, db)
  - Planner    → ciclo ReAct multi-step
  - Logger     → audit JSONL strutturato

Offre due modalità:
  1. process()        → risposta completa (blocca fino al risultato)
  2. process_stream() → genera chunk in streaming (per SSE)
"""

import json
import logging
import re
import threading
from typing import Callable, Dict, Generator, List, Optional, Tuple

from .config_loader import PilotConfig
from .prompt_builder import PromptBuilder
from .memory_store import MemoryStore
from .tool_executor import ToolExecutor
from .planner import ReActPlanner, SimplePlanner, PlanStep, create_planner
from .audit_logger import AuditLogger

_log = logging.getLogger(__name__)

# P1-4 fix: limite fatti estratti automaticamente per turno
_MAX_AUTO_FACTS_PER_TURN = 3


class Pilot:
    """Orchestratore principale del Pilot AI"""

    def __init__(self, config_path: str = None):
        """
        Inizializza tutti i sotto-sistemi.
        P1-6 fix: degrada con grazia se un sottosistema fallisce.

        Args:
            config_path: Percorso al file assistant.config.json (opzionale)
        """
        # 1. Carica configurazione (critica — senza config non possiamo procedere)
        self.cfg = PilotConfig(config_path=config_path)

        # P2-8 fix: lock per proteggere reload_config da richieste concorrenti
        self._reload_lock = threading.Lock()

        # P1-4 fix: contatore fatti estratti per turno
        self._auto_fact_count = 0

        # 2. Inizializza sotto-sistemi con degradazione graceful
        self.prompt_builder = PromptBuilder(self.cfg)

        try:
            self.memory = MemoryStore(self.cfg)
        except Exception as e:
            _log.error("MemoryStore init fallita, uso fallback in-memory: %s", e)
            self.memory = _NullMemoryStore()

        self.tools = ToolExecutor(self.cfg)
        self.tools.set_memory_store(self.memory)
        self.planner = create_planner(self.cfg, self.tools)

        try:
            self.logger = AuditLogger(self.cfg)
        except Exception as e:
            _log.error("AuditLogger init fallita, uso fallback no-op: %s", e)
            self.logger = _NullAuditLogger()

        # Log startup
        self.logger.log_startup({
            "name": self.cfg.name,
            "version": self.cfg.version,
            "engine": self.cfg.engine,
            "model": self.cfg.model_id,
            "planner": self.cfg.planner_strategy,
            "memory": self.memory.get_stats(),
        })

    # ==================================================================
    # API PRINCIPALE
    # ==================================================================

    def _prepare_turn(
        self,
        user_message: str,
        conv_id: str,
        extra_instructions: str = "",
    ) -> Tuple[str, List, bool]:
        """
        Logica condivisa per process() e process_stream():
        log turno utente, retrieval, build system prompt, planning decision.

        Returns:
            (system_prompt, available_tools, use_planning)
        """
        self.logger.log_conversation_turn(conv_id, "user", user_message)

        # Resetta limiti per-turno dei tool
        self.tools.reset_turn_limits()
        # P1-4 fix: resetta contatore fatti auto-estratti
        self._auto_fact_count = 0

        memory_context = self.memory.retrieve(user_message) if user_message else ""
        available_tools = self.tools.get_available_tools()

        system_prompt = self.prompt_builder.build_system_prompt(
            memory_context=memory_context,
            available_tools=available_tools or None,
            extra_instructions=extra_instructions,
        )

        use_planning = (
            hasattr(self.planner, 'needs_planning') and
            self.planner.needs_planning(user_message, available_tools)
        )

        return system_prompt, available_tools, use_planning

    def build_system_prompt(
        self,
        user_message: str = "",
        extra_instructions: str = "",
    ) -> str:
        """
        Costruisce il system prompt completo con contesto dalla memoria.

        Args:
            user_message:       Messaggio utente (per retrieval contestuale)
            extra_instructions: Istruzioni aggiuntive ad-hoc

        Returns:
            System prompt assemblato pronto per il modello
        """
        # Retrieval dalla memoria basato sul messaggio
        memory_context = ""
        if user_message:
            memory_context = self.memory.retrieve(user_message)

        # Tool disponibili
        available_tools = self.tools.get_available_tools()

        return self.prompt_builder.build_system_prompt(
            memory_context=memory_context,
            available_tools=available_tools if available_tools else None,
            extra_instructions=extra_instructions,
        )

    def process(
        self,
        user_message: str,
        conversation_history: List[Dict] = None,
        ai_engine=None,
        conv_id: str = "",
        extra_instructions: str = "",
    ) -> Tuple[str, Dict]:
        """
        Processa un messaggio con il ciclo completo del Pilot.

        Args:
            user_message:        Messaggio dell'utente
            conversation_history: Storico conversazione (formato Ollama)
            ai_engine:           Istanza di AIEngine per generare risposte
            conv_id:             ID conversazione per logging
            extra_instructions:  Istruzioni extra da aggiungere al system prompt

        Returns:
            (risposta_finale, metadata)
        """
        if not ai_engine:
            raise ValueError("ai_engine è richiesto per process()")

        system_prompt, available_tools, use_planning = self._prepare_turn(
            user_message, conv_id, extra_instructions
        )

        metadata = {
            "used_planning": use_planning,
            "steps": [],
            "tools_called": [],
            "memory_retrieved": bool(system_prompt and system_prompt.strip()),
        }

        if use_planning and isinstance(self.planner, ReActPlanner):
            response, plan_meta = self._run_react_loop(
                user_message, conversation_history, system_prompt, ai_engine
            )
            metadata.update(plan_meta)
        else:
            # Risposta diretta senza tool
            response = ai_engine.generate_response(
                user_message,
                conversation_history=conversation_history,
                system_prompt=system_prompt,
            )

        # Post-processing
        response = self._post_process(response)

        # Log turno assistente (sincrono — deve avvenire prima del return)
        self.logger.log_conversation_turn(conv_id, "assistant", response, metadata)

        # P0-1 fix: estrai fatti in thread separato (era sincrono, bloccava 2-5s)
        t = threading.Thread(
            target=self._extract_and_store_facts,
            args=(user_message, ai_engine),
            daemon=True,
        )
        t.start()

        return response, metadata

    def process_stream(
        self,
        user_message: str,
        conversation_history: List[Dict] = None,
        ai_engine=None,
        conv_id: str = "",
        images: List[str] = None,
        extra_instructions: str = "",
    ) -> Generator[str, None, None]:
        """
        Processa un messaggio con streaming.
        Per richieste ReAct, streamma aggiornamenti di stato intermedi
        prima della risposta finale (P1-4).

        Yields:
            Chunk della risposta
        """
        if not ai_engine:
            raise ValueError("ai_engine è richiesto per process_stream()")

        system_prompt, available_tools, use_planning = self._prepare_turn(
            user_message, conv_id, extra_instructions
        )

        # P0-2 fix: usa _stream_inner + post-processing in finally/GeneratorExit
        # per garantire logging e fact extraction anche se il client disconnette.
        response = ""
        try:
            if use_planning and isinstance(self.planner, ReActPlanner):
                # P1-4: streamma aggiornamenti intermedi del ciclo ReAct
                for msg_type, content in self._run_react_loop_streaming(
                    user_message, conversation_history, system_prompt, ai_engine
                ):
                    if msg_type == "status":
                        yield content
                    elif msg_type == "answer":
                        response = self._post_process(content)
                        for chunk in self._simulate_stream(response):
                            yield chunk
            else:
                # Streaming diretto dal modello
                full_response = ""
                for chunk in ai_engine.generate_response_stream(
                    user_message,
                    conversation_history=conversation_history,
                    system_prompt=system_prompt,
                    images=images,
                ):
                    full_response += chunk
                    yield chunk

                # Post-processing per il log (artefatti ReAct già inviati)
                response = self._post_process(full_response)
        except GeneratorExit:
            # Client disconnesso — esegui comunque cleanup
            pass
        finally:
            # Garantito: logging + fact extraction sempre eseguiti
            try:
                if response:
                    self.logger.log_conversation_turn(conv_id, "assistant", response)
                t = threading.Thread(
                    target=self._extract_and_store_facts,
                    args=(user_message, ai_engine),
                    daemon=True,
                )
                t.start()
            except Exception:
                pass  # Non bloccare per errori di logging

    # ==================================================================
    # CICLO REACT
    # ==================================================================

    def _run_react_loop(
        self,
        user_message: str,
        conversation_history: List[Dict],
        system_prompt: str,
        ai_engine,
    ) -> Tuple[str, Dict]:
        """
        Esegue il ciclo ReAct: Pensiero → Azione → Osservazione → ...

        Returns:
            (risposta_finale, metadata_piano)
        """
        # P1-1 fix: delega al core condiviso con on_step=None (non-streaming)
        final_answer = ""
        metadata: Dict = {}
        for msg_type, content in self._react_loop_core(
            user_message, conversation_history, system_prompt, ai_engine
        ):
            if msg_type == "answer":
                final_answer = content
            elif msg_type == "meta":
                metadata = content
        return final_answer, metadata

    def _run_react_loop_streaming(
        self,
        user_message: str,
        conversation_history: List[Dict],
        system_prompt: str,
        ai_engine,
    ) -> Generator[Tuple[str, str], None, None]:
        """
        P1-4: versione generator del ciclo ReAct che yield aggiornamenti
        intermedi per lo streaming, così l'utente non resta in attesa.

        Yields:
            ("status", testo_stato) — aggiornamento intermedio
            ("answer", risposta_finale) — risposta da post-processare e streammare
        """
        for msg_type, content in self._react_loop_core(
            user_message, conversation_history, system_prompt, ai_engine,
            emit_status=True,
        ):
            if msg_type != "meta":
                yield (msg_type, content)

    # ------------------------------------------------------------------
    # Cuore condiviso del ciclo ReAct (P1-1 DRY)
    # ------------------------------------------------------------------

    _MAX_CONTEXT_CHARS = 8000

    def _react_loop_core(
        self,
        user_message: str,
        conversation_history: List[Dict],
        system_prompt: str,
        ai_engine,
        emit_status: bool = False,
    ) -> Generator[Tuple[str, any], None, None]:
        """
        Implementazione unica del ciclo ReAct.
        Yield: ("status", text), ("answer", text), ("meta", dict).
        emit_status=True per streaming (emette "status"), False per sync.
        """
        self.planner.reset()
        metadata: Dict = {"steps": [], "tools_called": []}

        accumulated_context = ""
        output = ""
        max_tool_calls = self.cfg.max_tool_calls
        tools_called_count = 0

        for i in range(self.cfg.planner_max_steps):
            if emit_status:
                yield ("status", f"\n\n> 🔄 *Passo {i + 1}: ragionamento...*\n")

            full_prompt = user_message
            if accumulated_context:
                full_prompt = f"{accumulated_context}\n\nOra rispondi alla richiesta originale."

            output = ai_engine.generate_response(
                full_prompt,
                conversation_history=conversation_history,
                system_prompt=system_prompt,
            )

            step = self.planner.parse_model_output(output)
            self.logger.log_plan_step(step.to_dict())
            metadata["steps"].append(step.to_dict())

            if step.is_final:
                yield ("meta", metadata)
                yield ("answer", step.final_answer)
                return

            if step.action:
                tools_called_count += 1
                if tools_called_count > max_tool_calls:
                    self.logger.log_event("max_tool_calls_exceeded", {
                        "limit": max_tool_calls,
                    }, level="warn")
                    yield ("meta", metadata)
                    yield ("answer", output.strip())
                    return

                if emit_status:
                    yield ("status", f"> ⚙️ *Strumento: {step.action}*\n")

                observation, tool_success = self.planner.execute_step(step)
                metadata["tools_called"].append(step.action)

                if emit_status:
                    icon = "✅" if tool_success else "❌"
                    yield ("status", f"> {icon} *Risultato ottenuto*\n")

                self.logger.log_tool_call(
                    step.action, step.action_params, tool_success, observation,
                )

                new_context = self.planner.build_continuation_prompt(step)
                accumulated_context += "\n" + new_context
                accumulated_context = self._trim_context(accumulated_context)
            else:
                yield ("meta", metadata)
                yield ("answer", output.strip())
                return

        self.logger.log_event("react_max_steps", {
            "steps": len(metadata["steps"]),
            "max": self.cfg.planner_max_steps,
        }, level="warn")
        yield ("meta", metadata)
        yield ("answer", output.strip())

    @staticmethod
    def _trim_context(ctx: str, max_chars: int = 8000) -> str:
        """Tronca il contesto accumulato mantenendo primo + ultimo blocco."""
        if len(ctx) <= max_chars:
            return ctx
        first_break = ctx.find("\n\n", 200)
        if first_break == -1 or first_break > max_chars // 3:
            first_break = max_chars // 4
        first_part = ctx[:first_break]
        tail_budget = max_chars - len(first_part) - 30
        last_part = ctx[-tail_budget:]
        return first_part + "\n\n[...passi intermedi omessi...]\n\n" + last_part

    # ==================================================================
    # POST-PROCESSING
    # ==================================================================

    def _post_process(self, response: str) -> str:
        """Applica post-processing alla risposta"""
        if not response:
            return response

        # Rimuovi artefatti del formato ReAct se presenti
        response = re.sub(r"^(Pensiero|Thought)\s*:.*$", "", response, flags=re.MULTILINE)
        response = re.sub(r"^(Azione|Action)\s*:.*$", "", response, flags=re.MULTILINE)
        response = re.sub(r"^(Osservazione|Observation)\s*:.*$", "", response, flags=re.MULTILINE)
        response = re.sub(r"^(Risposta Finale|Final Answer)\s*:\s*", "", response, flags=re.MULTILINE)

        # Pulisci righe vuote multiple
        response = re.sub(r"\n{3,}", "\n\n", response)

        # Redazione segreti se configurato
        if self.cfg.redact_secrets:
            response = self._redact_secrets(response)

        return response.strip()

    def _redact_secrets(self, text: str) -> str:
        """Oscura pattern che sembrano segreti/credenziali"""
        patterns = [
            # API keys generiche
            (r'(?i)(api[_-]?key|token|secret|password|passwd|pwd)\s*[=:]\s*["\']?([a-zA-Z0-9_\-]{16,})["\']?',
             r'\1=***OSCURATO***'),
            # Bearer tokens
            (r'Bearer\s+[a-zA-Z0-9_\-\.]{20,}', 'Bearer ***OSCURATO***'),
            # JWT tokens (header.payload.signature)
            (r'eyJ[a-zA-Z0-9_-]{10,}\.eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}',
             '***JWT_OSCURATO***'),
            # AWS access keys
            (r'(?:AKIA|ASIA)[A-Z0-9]{16}', '***AWS_KEY_OSCURATO***'),
            # Connection strings
            (r'(?i)(?:mongodb|postgres|mysql|redis|amqp)://[^\s"\'>]+',
             '***CONN_STRING_OSCURATA***'),
            # SSH private keys
            (r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
             '***CHIAVE_PRIVATA_OSCURATA***'),
        ]
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text)
        return text

    # ==================================================================
    # APPRENDIMENTO (estrazione fatti)
    # ==================================================================

    def _extract_and_store_facts(self, user_message: str, ai_engine) -> None:
        """
        Usa il modello per estrarre fatti memorizzabili dal messaggio utente.
        Eseguita in un thread separato per non bloccare la risposta.
        P1-4 fix: applica rate-limit anche ai fatti estratti automaticamente.
        """
        # Messaggi troppo corti non contengono fatti
        if len(user_message) < 20:
            return

        try:
            prompt = self.prompt_builder.build_entity_extraction_prompt(user_message)
            extraction = ai_engine.generate_response(
                prompt,
                system_prompt="Sei un estrattore di informazioni. Rispondi SOLO in JSON valido.",
            )

            # Parsa il JSON
            extraction = extraction.strip()
            # Rimuovi eventuale code fence
            if extraction.startswith("```"):
                extraction = re.sub(r"```\w*\n?", "", extraction).strip()

            data = json.loads(extraction)
            facts = data.get("facts", [])

            for fact in facts:
                # P1-4 fix: rate-limit fatti auto-estratti
                if self._auto_fact_count >= _MAX_AUTO_FACTS_PER_TURN:
                    self.logger.log_event("auto_fact_limit_reached", {
                        "limit": _MAX_AUTO_FACTS_PER_TURN,
                    }, level="debug")
                    break

                key = fact.get("key", "").strip()
                value = fact.get("value", "").strip()
                if key and value:
                    self.memory.add_fact(key, value, source="auto_extraction")
                    self._auto_fact_count += 1
                    self.logger.log_memory_op("add_fact", {"key": key, "value": value[:100]})

        except (json.JSONDecodeError, KeyError, TypeError):
            pass  # Estrazione fallita, non critico
        except Exception as e:
            self.logger.log_error("Errore estrazione fatti", e)

    # ==================================================================
    # API MEMORIA (esposta per l'app)
    # ==================================================================

    def add_fact(self, key: str, value: str, source: str = "manual") -> int:
        """Aggiunge un fatto alla memoria"""
        fid = self.memory.add_fact(key, value, source)
        self.logger.log_memory_op("add_fact", {"key": key, "id": fid})
        return fid

    def search_memory(self, query: str) -> str:
        """Cerca nella memoria (fatti + documenti)"""
        return self.memory.retrieve(query)

    def add_document(self, path: str, content: str, tags: list = None) -> List[int]:
        """Indicizza un documento nella memoria"""
        ids = self.memory.add_document(path, content, tags)
        self.logger.log_memory_op("add_document", {"path": path, "chunks": len(ids)})
        return ids

    def add_task(self, title: str, due_at: str = None) -> int:
        """Crea un task"""
        tid = self.memory.add_task(title, due_at)
        self.logger.log_memory_op("add_task", {"title": title, "id": tid})
        return tid

    def get_open_tasks(self) -> List[Dict]:
        return self.memory.get_open_tasks()

    def get_memory_stats(self) -> Dict:
        return self.memory.get_stats()

    def get_all_facts(self) -> List[Dict]:
        return self.memory.get_all_facts()

    # ==================================================================
    # UTILITY
    # ==================================================================

    def _simulate_stream(self, text: str, chunk_size: int = 4) -> Generator[str, None, None]:
        """Simula streaming spezzando il testo in chunk che rispettano
        i confini delle parole (P2-5 fix: non taglia più a metà parola).
        """
        target = chunk_size * 5  # ~20 chars per chunk
        i = 0
        while i < len(text):
            end = min(i + target, len(text))
            # Se non siamo alla fine, evita di tagliare a metà parola
            if end < len(text) and text[end] not in (" ", "\n", "\t", "\r"):
                # Cerca il prossimo spazio/newline entro un margine ragionevole
                next_space = text.find(" ", end)
                next_nl = text.find("\n", end)
                candidates = [c for c in (next_space, next_nl) if c != -1 and c <= end + target]
                if candidates:
                    end = min(candidates) + 1
                else:
                    end = len(text)
            chunk = text[i:end]
            if chunk:
                yield chunk
            i = end

    def get_status(self) -> Dict:
        """Stato completo del Pilot"""
        return {
            "name": self.cfg.name,
            "version": self.cfg.version,
            "engine": self.cfg.engine,
            "model": self.cfg.model_id,
            "planner": self.cfg.planner_strategy,
            "tools": [t["id"] for t in self.tools.get_available_tools()],
            "memory": self.memory.get_stats(),
            "logs": self.logger.get_stats(),
        }

    def reload_config(self) -> None:
        """Ricarica la configurazione da disco e propaga ai subsystem.
        P2-8 fix: protetto da lock per evitare race condition con richieste in-flight.
        """
        with self._reload_lock:
            self.cfg.reload()
            self.tools = ToolExecutor(self.cfg)
            self.tools.set_memory_store(self.memory)
            self.planner = create_planner(self.cfg, self.tools)
            self.prompt_builder = PromptBuilder(self.cfg)
            self.logger.log_event("config_reload", {"version": self.cfg.version})

    def shutdown(self) -> None:
        """Chiusura pulita"""
        self.logger.log_event("pilot_shutdown", {})
        self.logger.flush()  # Svuota il buffer JSONL su disco
        self.memory.close()


# ======================================================================
# Fallback no-op per degradazione graceful (P1-6)
# ======================================================================

class _NullMemoryStore:
    """MemoryStore di fallback quando il DB non è disponibile."""
    def retrieve(self, *a, **kw) -> str: return ""
    def add_fact(self, *a, **kw) -> int: return -1
    def search_facts(self, *a, **kw) -> list: return []
    def add_document(self, *a, **kw) -> list: return []
    def add_task(self, *a, **kw) -> int: return -1
    def get_open_tasks(self, *a, **kw) -> list: return []
    def get_all_facts(self, *a, **kw) -> list: return []
    def get_stats(self) -> dict: return {"facts": 0, "tasks": 0, "document_chunks": 0, "db_path": "N/A (fallback)"}
    def close(self) -> None: pass


class _NullAuditLogger:
    """AuditLogger di fallback quando i log non possono essere scritti."""
    def log_event(self, *a, **kw) -> None: pass
    def log_conversation_turn(self, *a, **kw) -> None: pass
    def log_tool_call(self, *a, **kw) -> None: pass
    def log_plan_step(self, *a, **kw) -> None: pass
    def log_memory_op(self, *a, **kw) -> None: pass
    def log_error(self, *a, **kw) -> None: pass
    def log_startup(self, *a, **kw) -> None: pass
    def flush(self) -> None: pass
    def get_stats(self) -> dict: return {"events_count": 0, "conversations_count": 0}
