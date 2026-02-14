"""
Test suite per i 5 moduli core:
  ai_engine, memory, advanced_memory, document_processor, web_search
"""

import json
import os
import re
import tempfile
import threading
import types
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

# ── Patch config PRIMA di importare i moduli sotto test ────────────────
import config  # noqa: E402

_TMPDIR = tempfile.mkdtemp()
config.CONVERSATIONS_DIR = os.path.join(_TMPDIR, "conversations")
config.UPLOADS_DIR = os.path.join(_TMPDIR, "uploads")
config.DATA_DIR = os.path.join(_TMPDIR, "data")
os.makedirs(config.CONVERSATIONS_DIR, exist_ok=True)
os.makedirs(config.UPLOADS_DIR, exist_ok=True)
os.makedirs(config.DATA_DIR, exist_ok=True)

from core.ai_engine import AIEngine, _detect_repetition, _get_ollama_client  # noqa: E402
from core.memory import ConversationMemory, _validate_conv_id  # noqa: E402
from core.advanced_memory import (  # noqa: E402
    AdvancedMemory,
    ContextManager,
    EntityTracker,
    KnowledgeBase,
)
from core.document_processor import DocumentProcessor  # noqa: E402
from core import web_search as ws  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════
# ai_engine.py
# ═══════════════════════════════════════════════════════════════════════

class TestDetectRepetition(unittest.TestCase):
    def test_no_repetition(self):
        self.assertFalse(_detect_repetition("Questa è una frase normale senza ripetizioni."))

    def test_repetition_detected(self):
        phrase = "x" * 50
        buf = phrase * 5  # same phrase 5 times
        self.assertTrue(_detect_repetition(buf))

    def test_short_buffer_no_detection(self):
        self.assertFalse(_detect_repetition("ciao"))


class TestAIEngineBuildMessages(unittest.TestCase):
    """P1-1: verifica che _build_messages funzioni correttamente."""

    def test_basic(self):
        msgs = AIEngine._build_messages("Ciao")
        self.assertEqual(len(msgs), 2)  # system + user
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[1]["content"], "Ciao")

    def test_with_history(self):
        history = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        msgs = AIEngine._build_messages("c", conversation_history=history)
        self.assertEqual(len(msgs), 4)  # system + 2 history + user

    def test_with_images(self):
        msgs = AIEngine._build_messages("guarda", images=["base64data"])
        self.assertIn("images", msgs[-1])
        self.assertEqual(msgs[-1]["images"], ["base64data"])

    def test_custom_system_prompt(self):
        msgs = AIEngine._build_messages("q", system_prompt="Custom")
        self.assertEqual(msgs[0]["content"], "Custom")


class TestAIEngineBuildOpts(unittest.TestCase):
    def setUp(self):
        with patch("core.ai_engine._get_ollama_client"):
            self.eng = AIEngine.__new__(AIEngine)
            self.eng.client = MagicMock()
            self.eng._model_lock = threading.Lock()
            self.eng.model = "test"
            self.eng.temperature = 0.7
            self.eng.max_tokens = 2048
            self.eng.context_window = 8192

    def test_default_opts(self):
        opts = self.eng._build_opts()
        self.assertEqual(opts["temperature"], 0.7)
        self.assertEqual(opts["num_predict"], 2048)

    def test_vision_cap(self):
        opts = self.eng._build_opts(images=["img"])
        self.assertLessEqual(opts["num_predict"], AIEngine.VISION_MAX_TOKENS)

    def test_override_temperature(self):
        opts = self.eng._build_opts(temperature=0.1)
        self.assertEqual(opts["temperature"], 0.1)


class TestAIEngineAnalyzeDocTruncation(unittest.TestCase):
    """P2-2: il modello deve essere avvisato del troncamento."""

    def setUp(self):
        with patch("core.ai_engine._get_ollama_client"):
            self.eng = AIEngine.__new__(AIEngine)
            self.eng.client = MagicMock()
            self.eng._model_lock = threading.Lock()
            self.eng.model = "test"
            self.eng.temperature = 0.7
            self.eng.max_tokens = 2048
            self.eng.context_window = 8192
            self.eng.client.chat.return_value = {"message": {"content": "ok"}}

    def test_truncation_note_present(self):
        long_doc = "A" * 10000
        self.eng.analyze_document(long_doc)
        call_args = self.eng.client.chat.call_args
        prompt_content = call_args[1]["messages"][-1]["content"]
        self.assertIn("troncato", prompt_content)

    def test_no_truncation_note_when_short(self):
        short_doc = "A" * 100
        self.eng.analyze_document(short_doc)
        call_args = self.eng.client.chat.call_args
        prompt_content = call_args[1]["messages"][-1]["content"]
        self.assertNotIn("troncato", prompt_content)


class TestAIEngineChangeModelLock(unittest.TestCase):
    """P2-1: change_model deve essere thread-safe."""

    def setUp(self):
        with patch("core.ai_engine._get_ollama_client"):
            self.eng = AIEngine.__new__(AIEngine)
            self.eng.client = MagicMock()
            self.eng._model_lock = threading.Lock()
            self.eng.model = "original"
            self.eng.temperature = 0.7
            self.eng.max_tokens = 2048
            self.eng.context_window = 8192

    def test_rollback_on_unavailable(self):
        self.eng.client.list.side_effect = Exception("no model")
        result = self.eng.change_model("nonexistent")
        self.assertFalse(result)
        self.assertEqual(self.eng.model, "original")


class TestLazyOllamaClient(unittest.TestCase):
    """P1-2: il client Ollama deve essere creato lazy."""

    @patch("core.ai_engine.ollama.Client")
    def test_lazy_creation(self, mock_client_cls):
        import core.ai_engine as mod
        mod._ollama_client = None  # reset
        _get_ollama_client()
        mock_client_cls.assert_called_once()
        # Second call reuses
        _get_ollama_client()
        mock_client_cls.assert_called_once()
        mod._ollama_client = None  # cleanup


# ═══════════════════════════════════════════════════════════════════════
# memory.py
# ═══════════════════════════════════════════════════════════════════════

class TestConvIdValidation(unittest.TestCase):
    def test_valid_id(self):
        self.assertEqual(_validate_conv_id("abc_123-def"), "abc_123-def")

    def test_path_traversal(self):
        with self.assertRaises(ValueError):
            _validate_conv_id("../../etc/passwd")

    def test_empty_id(self):
        with self.assertRaises(ValueError):
            _validate_conv_id("")


class TestConversationMemory(unittest.TestCase):
    def setUp(self):
        ConversationMemory._conv_list_cache = None
        self.mem = ConversationMemory()

    def test_create_and_load(self):
        cid = self.mem.create_new_conversation("Test Conv")
        conv = self.mem.load_conversation(cid)
        self.assertIsNotNone(conv)
        self.assertEqual(conv["title"], "Test Conv")

    def test_add_message_auto_title(self):
        cid = self.mem.create_new_conversation()
        self.mem.add_message(cid, "user", "Prima domanda che faccio")
        conv = self.mem.load_conversation(cid)
        self.assertIn("Prima domanda", conv["title"])

    def test_delete(self):
        cid = self.mem.create_new_conversation("To Delete")
        self.assertTrue(self.mem.delete_conversation(cid))
        self.assertIsNone(self.mem.load_conversation(cid))

    def test_update_title(self):
        cid = self.mem.create_new_conversation("Old Title")
        self.mem.update_conversation_title(cid, "New Title")
        conv = self.mem.load_conversation(cid)
        self.assertEqual(conv["title"], "New Title")

    def test_max_messages_class_level(self):
        """P2-3: _MAX_MESSAGES deve essere attributo di classe."""
        self.assertTrue(hasattr(ConversationMemory, "_MAX_MESSAGES"))
        self.assertEqual(ConversationMemory._MAX_MESSAGES, 500)

    def test_search(self):
        cid = self.mem.create_new_conversation("Ricerca Test")
        self.mem.add_message(cid, "user", "parola chiave unica xyz123")
        results = self.mem.search_conversations("xyz123")
        self.assertTrue(len(results) >= 1)

    def test_list_all_conversations(self):
        self.mem.create_new_conversation("ListTest")
        convs = self.mem.list_all_conversations()
        self.assertTrue(len(convs) >= 1)

    def test_conv_to_meta_keys(self):
        cid = self.mem.create_new_conversation("Meta Test")
        conv = self.mem.load_conversation(cid)
        meta = ConversationMemory._conv_to_meta(conv)
        self.assertIn("id", meta)
        self.assertIn("message_count", meta)


# ═══════════════════════════════════════════════════════════════════════
# advanced_memory.py
# ═══════════════════════════════════════════════════════════════════════

class TestEstimateTokens(unittest.TestCase):
    """P1-5: verifica che il moltiplicatore 1.3 venga applicato."""

    def test_multiplier_applied(self):
        cm = ContextManager()
        # 10 parole → ~13 token, non 10
        text = "una due tre quattro cinque sei sette otto nove dieci"
        tokens = cm.estimate_tokens(text)
        self.assertEqual(tokens, int(10 * 1.3))

    def test_char_fallback_for_long_words(self):
        cm = ContextManager()
        text = "superlongword" * 10  # 1 word-like block
        tokens = cm.estimate_tokens(text)
        self.assertGreater(tokens, 0)


class TestContextManagerCompress(unittest.TestCase):
    def test_under_limit_returns_all(self):
        cm = ContextManager(max_context_tokens=99999)
        msgs = [{"role": "user", "content": "Ciao"}]
        result = cm.compress_context(msgs, None)
        self.assertEqual(result, msgs)

    def test_over_limit_creates_summary(self):
        cm = ContextManager(max_context_tokens=10)
        msgs = [{"role": "user", "content": f"Msg {i} " * 50} for i in range(20)]
        mock_engine = MagicMock()
        mock_engine.generate_response.return_value = "Riassunto"
        result = cm.compress_context(msgs, mock_engine)
        self.assertTrue(result[0].get("is_summary", False))


class TestEntityTracker(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(_TMPDIR, "data", "test_entities.json")
        if os.path.exists(self.path):
            os.remove(self.path)
        self.tracker = EntityTracker(self.path)

    def test_extract_name(self):
        self.tracker.extract_and_save("mi chiamo Marco e sono felice", "user")
        self.assertIn("Marco", self.tracker.entities["people"])
        self.assertIn("Marco", self.tracker.entities["people"])

    def test_stop_names_class_level_frozenset(self):
        """P1-4: STOP_NAMES deve essere frozenset class-level."""
        self.assertIsInstance(EntityTracker._STOP_NAMES, frozenset)
        self.assertIn("ciao", EntityTracker._STOP_NAMES)

    def test_extract_preferences(self):
        self.tracker.extract_and_save("Mi piace la pizza e preferisco il caffè", "user")
        prefs = self.tracker.entities["preferences"]
        self.assertTrue(len(prefs) > 0)

    def test_only_user_messages(self):
        self.tracker.extract_and_save("Mi chiamo Robot", "assistant")
        self.assertEqual(len(self.tracker.entities["people"]), 0)

    def test_get_relevant_entities_cap(self):
        """P2-4: i risultati devono avere un cap globale."""
        # Inserisci molti nomi
        for i in range(20):
            name = f"Nome{i}"
            self.tracker.entities["people"][name] = [{"context": "ctx", "timestamp": "ts"}]
        # Una query che contiene tutti i nomi
        query = " ".join(f"Nome{i}" for i in range(20))
        result = self.tracker.get_relevant_entities(query)
        # Conta le righe con "•"
        lines = [l for l in result.split("\n") if l.strip().startswith("•")]
        self.assertLessEqual(len(lines), EntityTracker._MAX_RELEVANT_ENTITIES)

    def test_persistence(self):
        self.tracker.extract_and_save("mi chiamo Luca e sono studente", "user")
        tracker2 = EntityTracker(self.path)
        self.assertIn("Luca", tracker2.entities["people"])


class TestGenerateSummaryLimit(unittest.TestCase):
    """B1: _generate_summary ora usa troncamento semplice (non più AI)."""

    def test_summary_input_capped(self):
        cm = ContextManager()
        mock_engine = MagicMock()
        # 200 messaggi lunghi
        msgs = [{"role": "user", "content": "X" * 500} for _ in range(200)]
        result = cm._generate_summary(msgs, mock_engine)
        # Non deve chiamare l'AI (B1 perf-fix)
        mock_engine.generate_response.assert_not_called()
        # Deve produrre un riassunto testuale
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


class TestKnowledgeBase(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(_TMPDIR, "data", "test_kb_dir")
        os.makedirs(self.path, exist_ok=True)
        self.kb = KnowledgeBase(self.path)

    def tearDown(self):
        self.kb.close()

    def test_extract_user_name(self):
        msgs = [{"role": "user", "content": "Mi chiamo Carlo e studio informatica"}]
        self.kb.update_from_conversation(msgs)
        self.assertEqual(self.kb.knowledge["user_profile"]["name"], "Carlo")

    def test_extract_interests_broader(self):
        """P2-5: deve gestire 'mi piace' e 'passione'."""
        self.kb._extract_interests("La mia passione è la musica elettronica.")
        interests = self.kb.knowledge["user_profile"]["interests"]
        self.assertTrue(len(interests) >= 1)
        # L'interesse estratto deve contenere 'musica elettronica'
        self.assertTrue(any("musica elettronica" in i for i in interests))

    def test_topic_counting(self):
        msgs = [
            {"role": "user", "content": "Parliamo di python e programmazione"},
            {"role": "user", "content": "Mi interessa lo sviluppo software"},
        ]
        self.kb.update_from_conversation(msgs)
        self.assertIn("programmazione", self.kb.knowledge["topics_discussed"])

    def test_persistence(self):
        self.kb._set_kv("name", "Test")
        self.kb.close()
        kb2 = KnowledgeBase(self.path)
        self.assertEqual(kb2.knowledge["user_profile"]["name"], "Test")
        kb2.close()


class TestAdvancedMemory(unittest.TestCase):
    def setUp(self):
        ConversationMemory._conv_list_cache = None
        self.amem = AdvancedMemory()

    def test_add_message_advanced(self):
        cid = self.amem.create_new_conversation("Adv Test")
        ok = self.amem.add_message_advanced(cid, "user", "Ciao, mi chiamo Giulia")
        self.assertTrue(ok)
        conv = self.amem.load_conversation(cid)
        self.assertTrue(len(conv["messages"]) >= 1)


# ═══════════════════════════════════════════════════════════════════════
# document_processor.py
# ═══════════════════════════════════════════════════════════════════════

class TestDocumentProcessor(unittest.TestCase):
    def setUp(self):
        self.dp = DocumentProcessor()

    def test_is_allowed_file(self):
        self.assertTrue(self.dp.is_allowed_file("doc.txt"))
        self.assertTrue(self.dp.is_allowed_file("code.py"))
        self.assertFalse(self.dp.is_allowed_file("virus.exe"))

    def test_process_txt(self):
        # Crea file temporaneo
        path = os.path.join(config.UPLOADS_DIR, "test_file.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("Contenuto di test")
        text, err = self.dp.process_file(path)
        self.assertIsNone(err)
        self.assertEqual(text, "Contenuto di test")

    def test_process_code_prefix(self):
        path = os.path.join(config.UPLOADS_DIR, "test_code.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write("print('hello')")
        text, err = self.dp.process_file(path)
        self.assertIsNone(err)
        self.assertIn("[File .py]", text)

    def test_process_missing_file(self):
        text, err = self.dp.process_file("/nonexistent/file.txt")
        self.assertIsNone(text)
        self.assertIn("non trovato", err)

    def test_safe_filename(self):
        safe = self.dp._make_safe_filename("../etc/passwd")
        self.assertNotIn("/", safe)
        self.assertNotIn("..", safe)
        self.assertEqual(self.dp._make_safe_filename("CON.txt"), "_CON.txt")
        self.assertNotEqual(self.dp._make_safe_filename(""), "")

    def test_save_upload_empty_data(self):
        """P2-7: file_data vuoto deve essere rifiutato."""
        path, err = self.dp.save_upload(b"", "empty.txt")
        self.assertIsNone(path)
        self.assertIn("vuoto", err.lower())

    def test_save_upload_valid(self):
        path, err = self.dp.save_upload(b"contenuto", "upload.txt")
        self.assertIsNone(err)
        self.assertIsNotNone(path)
        self.assertTrue(os.path.exists(path))

    def test_get_file_info(self):
        path = os.path.join(config.UPLOADS_DIR, "info_test.txt")
        with open(path, "w") as f:
            f.write("data")
        info = self.dp.get_file_info(path)
        self.assertIn("size", info)
        self.assertIn("extension", info)

    def test_get_file_info_missing(self):
        info = self.dp.get_file_info("/nonexistent")
        self.assertIsNone(info)


# ═══════════════════════════════════════════════════════════════════════
# web_search.py
# ═══════════════════════════════════════════════════════════════════════

class TestWebSearchPatterns(unittest.TestCase):
    def test_needs_web_search_italian(self):
        self.assertTrue(ws.needs_web_search("cercami informazioni su Python"))
        self.assertTrue(ws.needs_web_search("trovami il link di Wikipedia"))

    def test_needs_web_search_english(self):
        self.assertTrue(ws.needs_web_search("search for Python tutorials"))

    def test_no_search_needed(self):
        self.assertFalse(ws.needs_web_search("Ciao come stai?"))

    def test_needs_factual(self):
        self.assertTrue(ws.needs_factual_search("Chi è Albert Einstein?"))
        self.assertTrue(ws.needs_factual_search("Quanti abitanti ha Roma?"))

    def test_factual_not_triggered_on_explicit(self):
        """Se è ricerca esplicita, factual non si attiva."""
        self.assertFalse(ws.needs_factual_search("cercami chi è Einstein"))

    def test_youtube_detection(self):
        self.assertTrue(ws.is_youtube_query("youtube video di Python"))
        self.assertFalse(ws.is_youtube_query("Come si programma in Python?"))


class TestStripHallucinatedUrls(unittest.TestCase):
    """P1-6: verifica che il capture group funzioni."""

    def test_keeps_allowed_url(self):
        text = "[Google](https://google.com)"
        result = ws.strip_hallucinated_urls(text, {"https://google.com"})
        self.assertIn("https://google.com", result)

    def test_removes_fake_url(self):
        text = "[Fake](https://fake.example.com)"
        result = ws.strip_hallucinated_urls(text, set())
        self.assertNotIn("https://", result)
        self.assertIn("Fake", result)  # titolo preservato

    def test_removes_bare_url(self):
        text = "Visita https://fake.com per info."
        result = ws.strip_hallucinated_urls(text, set())
        self.assertNotIn("https://fake.com", result)

    def test_keeps_bare_allowed_url(self):
        text = "Visita https://real.com per info."
        result = ws.strip_hallucinated_urls(text, {"https://real.com"})
        self.assertIn("https://real.com", result)


class TestCleanQuery(unittest.TestCase):
    def test_filler_removal(self):
        result = ws._clean_query("cercami informazioni su Python")
        # "cercami", "su" removed
        self.assertIn("Python", result)
        self.assertNotIn("cercami", result.lower())

    def test_short_result_fallback(self):
        result = ws._clean_query("cerca di")
        self.assertTrue(len(result) >= 3)


class TestFormatResults(unittest.TestCase):
    def test_empty_results(self):
        result = ws.format_search_results([], query="test")
        self.assertIn("Nessun risultato", result)

    def test_format_user(self):
        results = [{"title": "Python", "url": "https://python.org", "snippet": "desc"}]
        text = ws.format_search_results_user(results, "python")
        self.assertIn("[Python](https://python.org)", text)

    def test_strip_trailing_references(self):
        text = "Risposta valida.\n\nFonte: Wikipedia\n---"
        result = ws._strip_trailing_references(text)
        self.assertNotIn("Fonte:", result)


class TestSearchAndFormat(unittest.TestCase):
    def test_returns_none_for_normal_message(self):
        result = ws.search_and_format("Ciao, come va?")
        self.assertIsNone(result)

    @patch("core.web_search.web_search")
    @patch("core.web_search._fetch_wikipedia_extract", return_value=None)
    def test_factual_mode_clean_query(self, mock_wiki, mock_ddg):
        """P2-8: DDG deve usare query pulita senza prefisso 'wikipedia'."""
        mock_ddg.return_value = [{"title": "T", "url": "https://x.com", "snippet": "S"}]
        ws.search_and_format("Chi è Albert Einstein?")
        # Verifica che la query DDG NON contenga "wikipedia"
        call_args = mock_ddg.call_args
        query_sent = call_args[0][0] if call_args[0] else call_args[1].get("query", "")
        self.assertNotIn("wikipedia", query_sent.lower())


# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main()
