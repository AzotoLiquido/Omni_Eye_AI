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
import re
import threading
from typing import Dict, Generator, List, Tuple

from .config_loader import PilotConfig
from .prompt_builder import PromptBuilder
from .memory_store import MemoryStore
from .tool_executor import ToolExecutor
from .planner import ReActPlanner, SimplePlanner, PlanStep, create_planner
from .audit_logger import AuditLogger


class Pilot:
    """Orchestratore principale del Pilot AI"""

    def __init__(self, config_path: str = None):
        """
        Inizializza tutti i sotto-sistemi.

        Args:
            config_path: Percorso al file assistant.config.json (opzionale)
        """
        # 1. Carica configurazione
        self.cfg = PilotConfig(config_path=config_path)

        # 2. Inizializza sotto-sistemi
        self.prompt_builder = PromptBuilder(self.cfg)
        self.memory = MemoryStore(self.cfg)
        self.tools = ToolExecutor(self.cfg)
        self.tools.set_memory_store(self.memory)  # Inietta memoria nel tool db
        self.planner = create_planner(self.cfg, self.tools)
        self.logger = AuditLogger(self.cfg)

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
    ) -> Tuple[str, Dict]:
        """
        Processa un messaggio con il ciclo completo del Pilot.

        Args:
            user_message:        Messaggio dell'utente
            conversation_history: Storico conversazione (formato Ollama)
            ai_engine:           Istanza di AIEngine per generare risposte
            conv_id:             ID conversazione per logging

        Returns:
            (risposta_finale, metadata)
        """
        if not ai_engine:
            raise ValueError("ai_engine è richiesto per process()")

        # Log turno utente
        self.logger.log_conversation_turn(conv_id, "user", user_message)

        # Retrieval dalla memoria (una sola volta)
        memory_context = self.memory.retrieve(user_message) if user_message else ""
        available_tools = self.tools.get_available_tools()

        # Costruisci system prompt con il contesto già recuperato
        system_prompt = self.prompt_builder.build_system_prompt(
            memory_context=memory_context,
            available_tools=available_tools or None,
        )

        # Decidi se serve il planner
        use_planning = (
            hasattr(self.planner, 'needs_planning') and
            self.planner.needs_planning(user_message, available_tools)
        )

        metadata = {
            "used_planning": use_planning,
            "steps": [],
            "tools_called": [],
            "memory_retrieved": bool(memory_context),
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

        # Estrai fatti dal messaggio utente (apprendimento)
        self._extract_and_store_facts(user_message, ai_engine)

        # Log turno assistente
        self.logger.log_conversation_turn(conv_id, "assistant", response, metadata)

        return response, metadata

    def process_stream(
        self,
        user_message: str,
        conversation_history: List[Dict] = None,
        ai_engine=None,
        conv_id: str = "",
        images: List[str] = None,
    ) -> Generator[str, None, None]:
        """
        Processa un messaggio con streaming.
        Per richieste che richiedono tool, esegue il piano prima
        e poi genera lo streaming della risposta finale.

        Yields:
            Chunk della risposta
        """
        if not ai_engine:
            raise ValueError("ai_engine è richiesto per process_stream()")

        # Log turno utente
        self.logger.log_conversation_turn(conv_id, "user", user_message)

        # Retrieval dalla memoria (una sola volta)
        memory_context = self.memory.retrieve(user_message) if user_message else ""
        available_tools = self.tools.get_available_tools()

        # Costruisci system prompt con il contesto già recuperato
        system_prompt = self.prompt_builder.build_system_prompt(
            memory_context=memory_context,
            available_tools=available_tools or None,
        )

        # Decidi se serve il planner
        use_planning = (
            hasattr(self.planner, 'needs_planning') and
            self.planner.needs_planning(user_message, available_tools)
        )

        if use_planning and isinstance(self.planner, ReActPlanner):
            # Esegui il piano in modo sincrono, poi streamma il risultato finale
            response, _ = self._run_react_loop(
                user_message, conversation_history, system_prompt, ai_engine
            )
            response = self._post_process(response)

            # Streamma la risposta finale chunk per chunk
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

            # Applica post-processing (artefatti ReAct già inviati
            # in streaming non si possono ritirare, ma puliamo per il log)
            response = self._post_process(full_response)

        # Post-processing asincrono (estrazione fatti in thread separato)
        t = threading.Thread(
            target=self._extract_and_store_facts,
            args=(user_message, ai_engine),
            daemon=True,
        )
        t.start()

        # Log turno assistente
        self.logger.log_conversation_turn(conv_id, "assistant", response)

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
        self.planner.reset()
        metadata = {"steps": [], "tools_called": []}

        # Primo turno: chiedi al modello cosa fare
        current_prompt = user_message
        accumulated_context = ""
        output = ""

        for i in range(self.cfg.planner_max_steps):
            # Genera ragionamento dal modello
            full_prompt = current_prompt
            if accumulated_context:
                full_prompt = f"{accumulated_context}\n\nOra rispondi alla richiesta originale."

            output = ai_engine.generate_response(
                full_prompt,
                conversation_history=conversation_history,
                system_prompt=system_prompt,
            )

            # Parsa l'output
            step = self.planner.parse_model_output(output)

            # Log del passo
            self.logger.log_plan_step(step.to_dict())
            metadata["steps"].append(step.to_dict())

            # Se è la risposta finale, restituiscila
            if step.is_final:
                return step.final_answer, metadata

            # Se c'è un'azione, eseguila
            if step.action:
                observation = self.planner.execute_step(step)
                metadata["tools_called"].append(step.action)

                self.logger.log_tool_call(
                    step.action,
                    step.action_params,
                    "ERRORE" not in observation,
                    observation,
                )

                # Costruisci il contesto per il prossimo turno (accumula)
                accumulated_context += "\n" + self.planner.build_continuation_prompt(step)
            else:
                # Nessuna azione ma nemmeno risposta finale → forza uscita
                return output.strip(), metadata

        # Max step raggiunto
        self.logger.log_event("react_max_steps", {
            "steps": len(metadata["steps"]),
            "max": self.cfg.planner_max_steps,
        }, level="warn")

        # Usa l'ultimo output come risposta
        return output.strip(), metadata

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
            (r'(?i)(api[_-]?key|token|secret|password)\s*[=:]\s*["\']?([a-zA-Z0-9_\-]{16,})["\']?',
             r'\1=***OSCURATO***'),
            # Bearer tokens
            (r'Bearer\s+[a-zA-Z0-9_\-\.]{20,}', 'Bearer ***OSCURATO***'),
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
        """
        # Messaggi troppo corti non contengono fatti
        if len(user_message) < 20:
            return

        def _do_extract():
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
                    key = fact.get("key", "").strip()
                    value = fact.get("value", "").strip()
                    if key and value:
                        self.memory.add_fact(key, value, source="auto_extraction")
                        self.logger.log_memory_op("add_fact", {"key": key, "value": value[:100]})

            except (json.JSONDecodeError, KeyError, TypeError):
                pass  # Estrazione fallita, non critico
            except Exception as e:
                self.logger.log_error("Errore estrazione fatti", e)

        try:
            _do_extract()
        except Exception:
            pass  # Estrazione non critica, non blocca la risposta

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
        """Simula streaming spezzando il testo in piccoli chunk"""
        words = text.split(" ")
        buffer = []
        for word in words:
            buffer.append(word)
            if len(buffer) >= chunk_size:
                yield " ".join(buffer) + " "
                buffer = []
        if buffer:
            yield " ".join(buffer)

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
        """Ricarica la configurazione da disco"""
        self.cfg.reload()
        self.logger.log_event("config_reload", {"version": self.cfg.version})

    def shutdown(self) -> None:
        """Chiusura pulita"""
        self.logger.log_event("pilot_shutdown", {})
        self.logger.flush()  # Svuota il buffer JSONL su disco
        self.memory.close()
