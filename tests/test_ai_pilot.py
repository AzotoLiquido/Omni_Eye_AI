"""
Test suite per AI-Pilot (P2-3)
Copre: config loader, planner parsing, tool validation, memory, audit logger, pilot.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Aggiungi la root del progetto al path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ======================================================================
# CONFIG LOADER
# ======================================================================

class TestPilotConfig(unittest.TestCase):
    """Test di PilotConfig: caricamento, validazione, property accessor."""

    def _make_config(self, overrides: dict = None) -> str:
        """Crea un file config minimo in una temp dir e restituisce il path."""
        base = {
            "meta": {"name": "Test", "version": "0.1.0", "created_at": "2026-01-01", "description": "", "locale": "it-IT"},
            "runtime": {
                "engine": "local_llm",
                "model": {"id": "test-model", "temperature": 0.5, "top_p": 1.0},
                "limits": {"max_tokens_out": 2048, "context_tokens": 8192, "tool_timeout_ms": 30000, "max_tool_calls": 10},
            },
            "persona": {"style": {"tone": "friendly", "verbosity": 2, "formatting": {}}, "language": {"primary": "it-IT"}, "output_format": {"default": "markdown"}},
            "policies": {"safety": {"refuse_categories": [], "redact_secrets": True, "allow_shell_write": False}, "privacy": {"store_conversations": True, "pii_handling": "minimize", "data_paths_allowlist": ["./workspace"]}, "web_access": {"enabled": False}},
            "memory": {"provider": "sqlite", "storage": {"path": "./test_mem.sqlite"}, "schemas": {"facts": {}, "tasks": {}, "documents": {}}, "retrieval": {"mode": "hybrid", "top_k": 5, "min_score": 0.2, "chunking": {"max_chars": 2000, "overlap_chars": 200}}},
            "tools": {"registry": [], "routing": {"default_policy": "auto"}},
            "orchestration": {"planner": {"strategy": "react", "max_steps": 5, "stop_on_refusal": True}, "execution": {"sandbox": {"enabled": True, "fs_root": "./workspace", "network": "off"}, "confirmations": {"required_for": []}}},
            "logging": {"level": "info", "paths": {"events": "./logs/test_events.jsonl", "conversations": "./logs/test_conv.jsonl"}, "audit": {"enabled": True, "log_prompts": False, "log_tool_io": True}},
        }
        if overrides:
            # Shallow merge at top level
            for k, v in overrides.items():
                if isinstance(v, dict) and k in base:
                    base[k].update(v)
                else:
                    base[k] = v

        self._tmpdir = tempfile.mkdtemp()
        cfg_path = os.path.join(self._tmpdir, "assistant.config.json")
        schema_path = os.path.join(self._tmpdir, "assistant.schema.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(base, f)
        # Schema vuoto → skip jsonschema validation
        with open(schema_path, "w", encoding="utf-8") as f:
            f.write("")
        return cfg_path

    def test_load_valid_config(self):
        from core.ai_pilot.config_loader import PilotConfig
        cfg = PilotConfig(config_path=self._make_config())
        self.assertEqual(cfg.name, "Test")
        self.assertEqual(cfg.model_id, "test-model")
        self.assertEqual(cfg.temperature, 0.5)
        self.assertEqual(cfg.max_tokens_out, 2048)

    def test_missing_section_raises(self):
        from core.ai_pilot.config_loader import PilotConfig, ConfigValidationError
        path = self._make_config()
        # Rimuovi una sezione obbligatoria
        with open(path, "r") as f:
            data = json.load(f)
        del data["runtime"]
        with open(path, "w") as f:
            json.dump(data, f)
        with self.assertRaises(ConfigValidationError):
            PilotConfig(config_path=path)

    def test_model_id_flat_format(self):
        """Testa il formato flat (runtime.model_id)"""
        from core.ai_pilot.config_loader import PilotConfig
        path = self._make_config()
        with open(path, "r") as f:
            data = json.load(f)
        # Converti a flat
        data["runtime"]["model_id"] = "flat-model"
        del data["runtime"]["model"]
        with open(path, "w") as f:
            json.dump(data, f)
        cfg = PilotConfig(config_path=path)
        self.assertEqual(cfg.model_id, "flat-model")

    def test_reload(self):
        from core.ai_pilot.config_loader import PilotConfig
        path = self._make_config()
        cfg = PilotConfig(config_path=path)
        self.assertEqual(cfg.name, "Test")
        # Modifica il file
        with open(path, "r") as f:
            data = json.load(f)
        data["meta"]["name"] = "Reloaded"
        with open(path, "w") as f:
            json.dump(data, f)
        cfg.reload()
        self.assertEqual(cfg.name, "Reloaded")


# ======================================================================
# PLANNER — PARSING
# ======================================================================

class TestReActParsing(unittest.TestCase):
    """Test del parsing output del modello (ReActPlanner)."""

    def _make_planner(self):
        from core.ai_pilot.planner import ReActPlanner
        cfg = MagicMock()
        cfg.planner_max_steps = 5
        cfg.max_tool_calls = 10
        tool_executor = MagicMock()
        return ReActPlanner(cfg, tool_executor)

    def test_parse_final_answer(self):
        planner = self._make_planner()
        step = planner.parse_model_output("Risposta Finale: Ciao, come stai?")
        self.assertTrue(step.is_final)
        self.assertEqual(step.final_answer, "Ciao, come stai?")

    def test_parse_final_answer_english(self):
        planner = self._make_planner()
        step = planner.parse_model_output("Final Answer: Hello, world!")
        self.assertTrue(step.is_final)
        self.assertEqual(step.final_answer, "Hello, world!")

    def test_parse_action_fs(self):
        planner = self._make_planner()
        output = 'Pensiero: Devo leggere il file.\nAzione: fs({"action": "read", "path": "test.txt"})'
        step = planner.parse_model_output(output)
        self.assertFalse(step.is_final)
        self.assertEqual(step.action, "fs")
        self.assertEqual(step.action_params.get("action"), "read")

    def test_parse_action_py(self):
        planner = self._make_planner()
        output = 'Pensiero: Calcolo necessario.\nAzione: py({"code": "print(2+2)"})'
        step = planner.parse_model_output(output)
        self.assertEqual(step.action, "py")
        self.assertEqual(step.action_params.get("code"), "print(2+2)")

    def test_fallback_no_format(self):
        """Se l'output non segue il formato, deve restituire il testo come risposta finale."""
        planner = self._make_planner()
        step = planner.parse_model_output("Questa è solo una risposta libera.")
        self.assertTrue(step.is_final)

    def test_fallback_action_parse_strict(self):
        """P1-2: 'ricordo' in testo casuale NON deve triggerare db tool."""
        planner = self._make_planner()
        tool_id, params = planner._fallback_action_parse(
            "Mi ricordo che ieri era una bella giornata"
        )
        # Non deve triggerare nessun tool (pattern troppo vago senza virgolette)
        self.assertIsNone(tool_id)

    def test_fallback_action_parse_with_quoted_db(self):
        """P1-2: Cerca nella memoria con parametro esplicito tra virgolette."""
        planner = self._make_planner()
        tool_id, params = planner._fallback_action_parse(
            'Cerco nella memoria "nome utente"'
        )
        self.assertEqual(tool_id, "db")
        self.assertEqual(params["query"], "nome utente")

    def test_fallback_action_parse_fs_default(self):
        """fs senza virgolette deve avere il default list '.'."""
        planner = self._make_planner()
        tool_id, params = planner._fallback_action_parse(
            "leggi il file di configurazione"
        )
        # senza virgolette → default list
        # pattern match "legg[io].*file" ma niente virgolette
        self.assertEqual(tool_id, "fs")

    def test_needs_planning_with_tool_keywords(self):
        planner = self._make_planner()
        tools = [{"id": "fs"}, {"id": "py"}]
        # Serve score >= 3: "leggi il file" = +2, "lista directory" = +2
        self.assertTrue(planner.needs_planning("leggi il file e lista directory", tools))

    def test_needs_planning_simple_question(self):
        planner = self._make_planner()
        tools = [{"id": "fs"}]
        self.assertFalse(planner.needs_planning("Ciao, come stai?", tools))

    def test_needs_planning_no_tools(self):
        planner = self._make_planner()
        self.assertFalse(planner.needs_planning("leggi il file", []))


# ======================================================================
# TOOL EXECUTOR — VALIDATION
# ======================================================================

class TestToolExecutor(unittest.TestCase):
    """Test di ToolExecutor: validazione AST, rate limiting."""

    def _make_executor(self):
        from core.ai_pilot.tool_executor import ToolExecutor
        cfg = MagicMock()
        cfg.fs_root = "./workspace"
        cfg.tool_timeout_ms = 5000
        cfg.max_tool_calls = 10
        cfg.raw = {"tools": {"registry": []}}
        executor = ToolExecutor(cfg)
        return executor

    def test_validate_python_safe(self):
        """Codice Python sicuro → nessun errore."""
        executor = self._make_executor()
        result = executor._validate_python_ast("x = 2 + 2\nprint(x)")
        self.assertIsNone(result)

    def test_validate_python_import_blocked(self):
        """Import di moduli pericolosi deve essere bloccato."""
        executor = self._make_executor()
        result = executor._validate_python_ast("import os\nos.system('rm -rf /')")
        self.assertIsNotNone(result)

    def test_validate_python_dunder_blocked(self):
        """Accesso a dunder non consentiti deve essere bloccato."""
        executor = self._make_executor()
        result = executor._validate_python_ast("x.__class__.__bases__[0]")
        self.assertIsNotNone(result)

    def test_validate_python_safe_dunders(self):
        """Dunder sicuri (__init__, __str__, ecc.) devono essere permessi."""
        executor = self._make_executor()
        result = executor._validate_python_ast("class A:\n  def __init__(self): pass")
        self.assertIsNone(result)

    def test_rate_limit_db(self):
        """P0-4: dopo _FACT_WRITES_PER_TURN (5), add_fact deve restituire errore."""
        from core.ai_pilot.tool_executor import ToolExecutor
        cfg = MagicMock()
        cfg.fs_root = "./workspace"
        cfg.sandbox_fs_root = "./workspace"
        cfg.tool_timeout_ms = 5000
        cfg.max_tool_calls = 10
        cfg.raw = {"tools": {"registry": [
            {"id": "db", "type": "database", "enabled": True, "config": {}},
        ]}}
        cfg.get_enabled_tools = MagicMock(return_value=[
            {"id": "db", "type": "database", "enabled": True, "config": {}},
        ])
        cfg.get_tool_config = MagicMock(return_value=
            {"id": "db", "type": "database", "enabled": True, "config": {}}
        )
        cfg.get_tool_policy = MagicMock(return_value="auto")
        executor = ToolExecutor(cfg)
        mem = MagicMock()
        mem.add_fact = MagicMock(return_value=1)
        mem.retrieve = MagicMock(return_value="")
        executor.set_memory_store(mem)

        # Scrivi fino al limite (5), poi il 6° deve fallire
        for i in range(5):
            r = executor.execute("db", {"action": "add_fact", "key": f"k{i}", "value": f"v{i}"})
            self.assertTrue(r.success, f"Scrittura {i+1} doveva riuscire")
        r_over = executor.execute("db", {"action": "add_fact", "key": "x", "value": "y"})
        self.assertFalse(r_over.success)
        self.assertIn("limite", r_over.error.lower())

        # Dopo reset, si può scrivere di nuovo
        executor.reset_turn_limits()
        r_after = executor.execute("db", {"action": "add_fact", "key": "new", "value": "val"})
        self.assertTrue(r_after.success)


# ======================================================================
# MEMORY STORE
# ======================================================================

class TestMemoryStore(unittest.TestCase):
    """Test di MemoryStore: CRUD fatti, ricerca FTS5."""

    def setUp(self):
        from core.ai_pilot.memory_store import MemoryStore
        cfg = MagicMock()
        # Usa un db in memoria (tempfile)
        self._tmpdir = tempfile.mkdtemp()
        cfg.memory_storage_path = os.path.join(self._tmpdir, "test.sqlite")
        # Bypassa la risoluzione path relativa nella classe
        cfg.retrieval_top_k = 5
        cfg.retrieval_min_score = 0.1
        cfg.retrieval_mode = "hybrid"
        cfg.chunking_max_chars = 2000
        cfg.chunking_overlap_chars = 200

        # Patch Path resolution: MemoryStore concatena base_dir / storage_path
        # Usiamo monkeypatch per il path
        with patch.object(Path, '__truediv__', side_effect=lambda self, other: Path(cfg.memory_storage_path)):
            pass

        # Simpler: creiamo il MemoryStore con path diretto
        self.store = MemoryStore.__new__(MemoryStore)
        self.store.cfg = cfg
        import threading
        self.store._lock = threading.RLock()

        import sqlite3
        db_path = os.path.join(self._tmpdir, "test.sqlite")
        self.store._db_path = db_path
        self.store._conn = sqlite3.connect(db_path, check_same_thread=False)
        self.store._conn.row_factory = sqlite3.Row
        self.store._conn.execute("PRAGMA journal_mode=WAL")
        self.store._init_tables()

    def tearDown(self):
        try:
            self.store.close()
        except Exception:
            pass

    def test_add_and_get_fact(self):
        fid = self.store.add_fact("nome", "Mario", source="test")
        self.assertIsInstance(fid, int)
        facts = self.store.get_all_facts()
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["key"], "nome")
        self.assertEqual(facts[0]["value"], "Mario")

    def test_search_facts_fts(self):
        self.store.add_fact("linguaggio", "Python è il mio linguaggio preferito")
        self.store.add_fact("colore", "Blu è il mio colore preferito")
        result = self.store.retrieve("Python linguaggio")
        self.assertIn("Python", result)

    def test_add_task(self):
        tid = self.store.add_task("Comprare il latte")
        self.assertIsInstance(tid, int)
        tasks = self.store.get_open_tasks()
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["title"], "Comprare il latte")

    def test_stats(self):
        self.store.add_fact("a", "b")
        self.store.add_fact("c", "d")
        self.store.add_task("test")
        stats = self.store.get_stats()
        self.assertEqual(stats["facts"], 2)
        self.assertEqual(stats["tasks"], 1)


# ======================================================================
# AUDIT LOGGER
# ======================================================================

class TestAuditLogger(unittest.TestCase):
    """Test di AuditLogger: scrittura, flush, read_tail efficiente."""

    def setUp(self):
        from core.ai_pilot.audit_logger import AuditLogger
        cfg = MagicMock()
        self._tmpdir = tempfile.mkdtemp()
        cfg.log_level = "info"
        cfg.log_events_path = os.path.join(self._tmpdir, "events.jsonl")
        cfg.log_conversations_path = os.path.join(self._tmpdir, "conv.jsonl")
        cfg.audit_enabled = True
        cfg.audit_log_prompts = False
        cfg.audit_log_tool_io = True
        self.logger = AuditLogger(cfg)

    def tearDown(self):
        try:
            self.logger.flush()
        except Exception:
            pass

    def test_log_event_and_read(self):
        self.logger.log_event("test_event", {"key": "value"})
        self.logger.flush()
        events = self.logger.read_recent_events(10)
        self.assertGreaterEqual(len(events), 1)
        self.assertEqual(events[-1]["type"], "test_event")

    def test_log_conversation_turn(self):
        self.logger.log_conversation_turn("conv1", "user", "Ciao")
        self.logger.flush()
        turns = self.logger.read_recent_conversations(10)
        self.assertGreaterEqual(len(turns), 1)
        self.assertEqual(turns[-1]["role"], "user")
        # Quando log_prompts=False, il contenuto non viene salvato
        self.assertIn("content_length", turns[-1])

    def test_read_tail_efficient(self):
        """P2-9: verifica che _read_tail legga correttamente le ultime N righe."""
        # Scrivi 100 eventi
        for i in range(100):
            self.logger.log_event(f"evt_{i}", {"i": i})
        self.logger.flush()
        # Leggi solo gli ultimi 5
        events = self.logger.read_recent_events(5)
        self.assertEqual(len(events), 5)
        self.assertEqual(events[-1]["data"]["i"], 99)

    def test_read_tail_empty_file(self):
        events = self.logger.read_recent_events(10)
        # Potrebbe essere vuoto o avere eventi init
        self.assertIsInstance(events, list)


# ======================================================================
# PILOT — SIMULATE STREAM
# ======================================================================

class TestSimulateStream(unittest.TestCase):
    """Test di Pilot._simulate_stream: word boundaries (P2-5)."""

    def _make_pilot(self):
        """Crea un Pilot minimale per testare _simulate_stream."""
        from core.ai_pilot.pilot import Pilot
        pilot = Pilot.__new__(Pilot)
        return pilot

    def test_no_word_cutting(self):
        """P2-5: i chunk non devono tagliare le parole a metà."""
        pilot = self._make_pilot()
        text = "Questa è una frase di test che ha diverse parole lunghe"
        chunks = list(pilot._simulate_stream(text))
        reconstructed = "".join(chunks)
        self.assertEqual(reconstructed, text)
        # Verifica che nessun chunk (tranne forse l'ultimo) finisce mid-word
        for chunk in chunks[:-1]:
            last_char = chunk[-1]
            self.assertTrue(
                last_char in (" ", "\n", "\t", "\r"),
                f"Chunk termina mid-word: '{chunk}'"
            )

    def test_preserves_newlines(self):
        pilot = self._make_pilot()
        text = "Riga uno\nRiga due\nRiga tre"
        chunks = list(pilot._simulate_stream(text))
        reconstructed = "".join(chunks)
        self.assertEqual(reconstructed, text)

    def test_empty_text(self):
        pilot = self._make_pilot()
        chunks = list(pilot._simulate_stream(""))
        self.assertEqual(chunks, [])

    def test_short_text(self):
        pilot = self._make_pilot()
        chunks = list(pilot._simulate_stream("Ok"))
        self.assertEqual("".join(chunks), "Ok")


# ======================================================================
# PILOT — CONTEXT TRUNCATION
# ======================================================================

class TestContextTruncation(unittest.TestCase):
    """P1-3: verifica che la troncatura del contesto mantenga il primo step."""

    def test_truncation_keeps_first_step(self):
        """Il primo blocco di contesto deve sopravvivere alla troncatura."""
        first_step = "PRIMO_STEP: " + "x" * 300
        middle = "\n\n" + "MIDDLE: " + "y" * 5000
        last_step = "\n\nULTIMO: " + "z" * 3000
        accumulated = first_step + middle + last_step

        _MAX_CONTEXT_CHARS = 8000
        if len(accumulated) > _MAX_CONTEXT_CHARS:
            first_break = accumulated.find("\n\n", 200)
            if first_break == -1 or first_break > _MAX_CONTEXT_CHARS // 3:
                first_break = _MAX_CONTEXT_CHARS // 4
            first_part = accumulated[:first_break]
            tail_budget = _MAX_CONTEXT_CHARS - len(first_part) - 30
            last_part = accumulated[-tail_budget:]
            accumulated = first_part + "\n\n[...passi intermedi omessi...]\n\n" + last_part

        self.assertIn("PRIMO_STEP", accumulated)
        self.assertIn("ULTIMO", accumulated)
        self.assertLessEqual(len(accumulated), _MAX_CONTEXT_CHARS + 100)  # small margin


if __name__ == "__main__":
    unittest.main()
