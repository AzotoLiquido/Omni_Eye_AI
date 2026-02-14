"""
Config Loader - Carica e valida assistant.config.json contro lo schema
"""

import json
import os
from typing import Any, Dict, Optional
from pathlib import Path

# Directory contenente i JSON di configurazione
# _CODE_DIR = cartella del codice sorgente (core/ai_pilot/)
# _CFG_DIR  = cartella di configurazione (core/pilot_config/)
_CODE_DIR = Path(__file__).resolve().parent
_CFG_DIR = _CODE_DIR.parent / "pilot_config"

# P1-11: Verify file actually exists in fallback directory
if (_CFG_DIR / "assistant.config.json").exists():
    CONFIG_DIR = _CFG_DIR
elif (_CODE_DIR / "assistant.config.json").exists():
    CONFIG_DIR = _CODE_DIR
else:
    CONFIG_DIR = _CFG_DIR  # Will raise FileNotFoundError later

SCHEMA_PATH = CONFIG_DIR / "assistant.schema.json"
CONFIG_PATH = CONFIG_DIR / "assistant.config.json"


class ConfigValidationError(Exception):
    """Errore di validazione della configurazione"""
    pass


class PilotConfig:
    """Carica, valida e fornisce accesso tipizzato alla configurazione del Pilot"""

    def __init__(self, config_path: str = None, schema_path: str = None):
        self._config_path = Path(config_path) if config_path else CONFIG_PATH
        self._schema_path = Path(schema_path) if schema_path else SCHEMA_PATH
        self._raw: Dict[str, Any] = {}
        self._schema: Dict[str, Any] = {}
        self._load()

    # ------------------------------------------------------------------
    # Caricamento e validazione
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Carica schema e config da disco"""
        if not self._config_path.exists():
            raise FileNotFoundError(f"Config non trovata: {self._config_path}")

        with open(self._config_path, "r", encoding="utf-8-sig") as f:
            self._raw = json.load(f)

        if self._schema_path.exists():
            if self._schema_path.stat().st_size > 0:
                with open(self._schema_path, "r", encoding="utf-8-sig") as f:
                    self._schema = json.load(f)
            else:
                import logging as _log
                _log.getLogger(__name__).warning(
                    "Schema vuoto (%s), validazione JSON Schema saltata",
                    self._schema_path,
                )

        self._validate()

    def _validate(self) -> None:
        """Validazione strutturale (senza dipendenza jsonschema)"""
        required_top = ["meta", "runtime", "persona", "policies",
                        "memory", "tools", "orchestration", "logging"]
        missing = [k for k in required_top if k not in self._raw]
        if missing:
            raise ConfigValidationError(
                f"Sezioni mancanti nella config: {', '.join(missing)}"
            )

        # Validazione campi critici — supporta sia flat che nested
        rt = self._raw.get("runtime", {})
        has_model = (
            "model_id" in rt
            or ("model" in rt and "id" in rt.get("model", {}))
        )
        if not has_model:
            raise ConfigValidationError(
                "runtime.model_id o runtime.model.id è obbligatorio"
            )

        # Validazione jsonschema — solo se la config è in formato nested
        # (il formato flat non è conforme allo schema JSON, ma è supportato
        #  dal config_loader tramite le property di accesso ai dati)
        _is_nested = isinstance(rt.get("model"), dict)
        if _is_nested and self._schema:
            try:
                import jsonschema
                # Rimuovi $schema dal documento prima della validazione
                instance = {k: v for k, v in self._raw.items() if k != "$schema"}
                jsonschema.validate(instance=instance, schema=self._schema)
            except ImportError:
                pass  # Validazione base è sufficiente
            except jsonschema.ValidationError as e:
                raise ConfigValidationError(f"Validazione schema fallita: {e.message}")

    # ------------------------------------------------------------------
    # Accesso ai dati (proprietà)
    # ------------------------------------------------------------------

    @property
    def raw(self) -> Dict[str, Any]:
        import copy
        return copy.deepcopy(self._raw)

    # --- Meta ---
    @property
    def name(self) -> str:
        return self._raw["meta"]["name"]

    @property
    def version(self) -> str:
        return self._raw["meta"]["version"]

    @property
    def locale(self) -> str:
        m = self._raw.get("meta", {})
        return m.get("locale", self._raw.get("persona", {}).get("language", "it-IT"))

    # --- Runtime ---
    @property
    def engine(self) -> str:
        return self._raw["runtime"]["engine"]

    @property
    def model_id(self) -> str:
        rt = self._raw.get("runtime", {})
        # Flat: runtime.model_id  — Nested: runtime.model.id
        if "model_id" in rt:
            return rt["model_id"]
        model = rt.get("model", {})
        if isinstance(model, dict) and "id" in model:
            return model["id"]
        return "unknown"  # P3: safe fallback instead of KeyError

    @property
    def temperature(self) -> float:
        rt = self._raw["runtime"]
        if "temperature" in rt:
            return rt["temperature"]
        return rt.get("model", {}).get("temperature", 0.4)

    @property
    def top_p(self) -> float:
        rt = self._raw["runtime"]
        if "top_p" in rt:
            return rt["top_p"]
        return rt.get("model", {}).get("top_p", 1.0)

    @property
    def seed(self) -> Optional[int]:
        rt = self._raw["runtime"]
        if "seed" in rt:
            return rt["seed"]
        return rt.get("model", {}).get("seed")

    @property
    def max_tokens_out(self) -> int:
        rt = self._raw["runtime"]
        if "max_tokens" in rt:
            return rt["max_tokens"]
        return rt.get("limits", {}).get("max_tokens_out", 2048)

    @property
    def context_tokens(self) -> int:
        rt = self._raw["runtime"]
        if "context_window" in rt:
            return rt["context_window"]
        return rt.get("limits", {}).get("context_tokens", 4096)

    @property
    def tool_timeout_ms(self) -> int:
        return self._raw["runtime"].get("limits", {}).get("tool_timeout_ms", 45000)

    @property
    def max_tool_calls(self) -> int:
        rt = self._raw["runtime"]
        pol = self._raw.get("policies", {}).get("safety", {})
        if "max_tool_calls_per_turn" in pol:
            return pol["max_tool_calls_per_turn"]
        return rt.get("limits", {}).get("max_tool_calls", 12)

    @property
    def streaming(self) -> bool:
        return self._raw["runtime"].get("streaming", True)

    # --- Persona ---
    @property
    def tone(self) -> str:
        p = self._raw["persona"]
        if "tone" in p:
            return p["tone"]
        return p.get("style", {}).get("tone", "friendly")

    # P3: Class-level constant, not reallocated on every access
    _STR_MAP = {"minimal": 0, "low": 1, "brief": 2, "normal": 3,
                "balanced": 5, "detailed": 7, "verbose": 10}

    @property
    def verbosity(self):
        p = self._raw["persona"]
        val = p.get("verbosity", p.get("style", {}).get("verbosity", 2))
        if isinstance(val, int):
            return val
        return self._STR_MAP.get(str(val).lower(), 3)

    @property
    def formatting(self) -> Dict:
        fmt = self._raw["persona"].get("style", {}).get("formatting", {})
        # Defaults: code fences, liste e tabelle attivi se non specificato
        fmt.setdefault("code_fences", True)
        fmt.setdefault("use_lists", True)
        fmt.setdefault("use_tables", True)
        return fmt

    @property
    def primary_language(self) -> str:
        p = self._raw["persona"]
        lang = p.get("language", "it-IT")
        if isinstance(lang, dict):
            return lang.get("primary", "it-IT")
        return lang

    @property
    def avoid_english(self) -> bool:
        lang = self._raw["persona"].get("language", {})
        if isinstance(lang, dict):
            return lang.get("avoid_english_terms", True)
        return False

    @property
    def glossary(self) -> Dict[str, str]:
        lang = self._raw["persona"].get("language", {})
        if isinstance(lang, dict):
            return lang.get("glossary", {})
        return {}

    @property
    def output_format(self) -> str:
        p = self._raw.get("policies", {}).get("output", {})
        if "format" in p:
            return p["format"]
        return self._raw["persona"].get("output_format", {}).get("default", "markdown")

    @property
    def terminal_prefix(self) -> str:
        return self._raw["persona"].get("output_format", {}).get("terminal_prefix", "> ")

    @property
    def custom_instructions(self) -> str:
        return self._raw["persona"].get("custom_instructions", "")

    # --- Policies ---
    @property
    def refuse_categories(self) -> list:
        return self._raw["policies"]["safety"].get("refuse_categories", [])

    @property
    def redact_secrets(self) -> bool:
        return self._raw["policies"]["safety"].get("redact_secrets", True)

    @property
    def allow_shell_write(self) -> bool:
        return self._raw["policies"]["safety"].get("allow_shell_write", False)

    @property
    def store_conversations(self) -> bool:
        return self._raw["policies"].get("privacy", {}).get("store_conversations", True)

    @property
    def pii_handling(self) -> str:
        return self._raw["policies"].get("privacy", {}).get("pii_handling", "minimize")

    @property
    def data_paths_allowlist(self) -> list:
        return self._raw["policies"].get("privacy", {}).get("data_paths_allowlist", [])

    @property
    def web_access_enabled(self) -> bool:
        return self._raw["policies"].get("web_access", {}).get("enabled", False)

    # --- Memory ---
    @property
    def memory_enabled(self) -> bool:
        return self._raw["memory"].get("enabled", True)

    @property
    def memory_provider(self) -> str:
        mem = self._raw["memory"]
        return mem.get("backend", mem.get("provider", "sqlite"))

    @property
    def memory_storage_path(self) -> str:
        mem = self._raw["memory"]
        if "db_path" in mem:
            return mem["db_path"]
        return mem.get("storage", {}).get("path", "./data/memory.sqlite")

    @property
    def memory_encryption(self) -> bool:
        return self._raw["memory"].get("storage", {}).get("encryption", {}).get("enabled", False)

    @property
    def memory_schemas(self) -> Dict:
        mem = self._raw["memory"]
        return mem.get("collections", mem.get("schemas", {}))

    @property
    def retrieval_mode(self) -> str:
        return self._raw["memory"].get("retrieval", {}).get("mode", "fts5")

    @property
    def retrieval_top_k(self) -> int:
        return self._raw["memory"].get("retrieval", {}).get("top_k", 5)

    @property
    def retrieval_min_score(self) -> float:
        return self._raw["memory"].get("retrieval", {}).get("min_score", 0.2)

    @property
    def chunking_max_chars(self) -> int:
        return self._raw["memory"].get("retrieval", {}).get("chunking", {}).get("max_chars", 2000)

    @property
    def chunking_overlap(self) -> int:
        return self._raw["memory"].get("retrieval", {}).get("chunking", {}).get("overlap_chars", 200)

    # --- Tools ---
    @property
    def tool_registry(self) -> list:
        tools = self._raw["tools"]
        if isinstance(tools, list):
            return tools
        return tools.get("registry", [])

    @property
    def tool_routing_default(self) -> str:
        tools = self._raw["tools"]
        if isinstance(tools, list):
            return "auto"
        return tools.get("routing", {}).get("default_policy", "auto")

    @property
    def tool_routing_per_tool(self) -> Dict[str, str]:
        tools = self._raw["tools"]
        if isinstance(tools, list):
            return {t["id"]: t.get("policy", "auto") for t in tools if "id" in t}
        return tools.get("routing", {}).get("per_tool_policy", {})

    # --- Orchestration ---
    @property
    def planner_strategy(self) -> str:
        return self._raw["orchestration"]["planner"].get("strategy", "react")

    @property
    def planner_max_steps(self) -> int:
        return self._raw["orchestration"]["planner"].get("max_steps", 12)

    @property
    def stop_on_refusal(self) -> bool:
        return self._raw["orchestration"]["planner"].get("stop_on_refusal", True)

    @property
    def sandbox_enabled(self) -> bool:
        return self._raw["orchestration"].get("execution", {}).get("sandbox", {}).get("enabled", True)

    @property
    def sandbox_fs_root(self) -> str:
        return self._raw["orchestration"].get("execution", {}).get("sandbox", {}).get("fs_root", "./workspace")

    @property
    def sandbox_network(self) -> str:
        return self._raw["orchestration"].get("execution", {}).get("sandbox", {}).get("network", "off")

    @property
    def confirmations_required_for(self) -> list:
        return self._raw["orchestration"].get("execution", {}).get("confirmations", {}).get(
            "required_for", ["file_delete", "shell_exec", "network_access"]
        )

    @property
    def fallback_on_tool_error(self) -> str:
        return self._raw["orchestration"].get("fallback", {}).get("on_tool_error", "report_and_continue")

    @property
    def fallback_on_planner_error(self) -> str:
        return self._raw["orchestration"].get("fallback", {}).get("on_planner_error", "direct_response")

    # --- Logging ---
    @property
    def log_level(self) -> str:
        return self._raw["logging"]["level"]

    @property
    def log_events_path(self) -> str:
        log = self._raw["logging"]
        audit = log.get("audit", {})
        if "events_path" in audit:
            return audit["events_path"]
        return log.get("paths", {}).get("events", "./data/logs/events.jsonl")

    @property
    def log_conversations_path(self) -> str:
        log = self._raw["logging"]
        audit = log.get("audit", {})
        if "conversations_path" in audit:
            return audit["conversations_path"]
        return log.get("paths", {}).get("conversations", "./data/logs/conversations.jsonl")

    @property
    def audit_enabled(self) -> bool:
        return self._raw["logging"].get("audit", {}).get("enabled", True)

    @property
    def audit_log_prompts(self) -> bool:
        return self._raw["logging"].get("audit", {}).get("log_prompts", False)

    @property
    def audit_log_tool_io(self) -> bool:
        return self._raw["logging"].get("audit", {}).get("log_tool_io", True)

    # ------------------------------------------------------------------
    # Utilità
    # ------------------------------------------------------------------

    def get_enabled_tools(self) -> list:
        """Restituisce solo i tool abilitati"""
        return [t for t in self.tool_registry if t.get("enabled", True)]

    def get_tool_config(self, tool_id: str) -> Optional[Dict]:
        """Restituisce la config di un tool specifico"""
        for t in self.tool_registry:
            if t["id"] == tool_id:
                return t
        return None

    def get_tool_policy(self, tool_id: str) -> str:
        """Restituisce la policy di routing per un tool"""
        per_tool = self.tool_routing_per_tool
        if tool_id in per_tool:
            return per_tool[tool_id]
        return self.tool_routing_default

    def reload(self) -> None:
        """Ricarica la config da disco.
        
        P2: Load into temp variables first, then swap atomically
        so a parse error doesn't leave config in partial state.
        """
        old_raw = self._raw
        old_schema = self._schema
        try:
            self._load()
        except Exception:
            self._raw = old_raw
            self._schema = old_schema
            raise

    def __repr__(self) -> str:
        return f"<PilotConfig name={self.name!r} v{self.version} engine={self.engine}>"
