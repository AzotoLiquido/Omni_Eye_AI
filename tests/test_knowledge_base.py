"""Tests for the SQLite FTS5-backed KnowledgeBase class."""

import json
import os
import sqlite3
import tempfile
import unittest

from core.advanced_memory import KnowledgeBase


class TestKnowledgeBaseInit(unittest.TestCase):
    """Test inizializzazione e schema SQLite."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.kb = KnowledgeBase(storage_path=self.tmpdir)

    def tearDown(self):
        self.kb.close()

    def test_db_created(self):
        path = os.path.join(self.tmpdir, "knowledge.db")
        self.assertTrue(os.path.exists(path))

    def test_tables_exist(self):
        tables = {r[0] for r in self.kb._conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'trigger')"
        ).fetchall()}
        self.assertIn("kv", tables)
        self.assertIn("topics", tables)
        self.assertIn("facts", tables)

    def test_fts_virtual_table_exists(self):
        rows = self.kb._conn.execute(
            "SELECT name FROM sqlite_master WHERE name='facts_fts'"
        ).fetchall()
        self.assertTrue(len(rows) > 0)

    def test_wal_mode(self):
        mode = self.kb._conn.execute("PRAGMA journal_mode").fetchone()[0]
        self.assertEqual(mode, "wal")


class TestKnowledgeBaseKV(unittest.TestCase):
    """Test operazioni key-value (profilo utente)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.kb = KnowledgeBase(storage_path=self.tmpdir)

    def tearDown(self):
        self.kb.close()

    def test_set_get_kv(self):
        self.kb._set_kv("name", "Marco")
        self.assertEqual(self.kb._get_kv("name"), "Marco")

    def test_get_kv_default(self):
        self.assertEqual(self.kb._get_kv("nonexist", "fallback"), "fallback")

    def test_get_kv_none_default(self):
        self.assertIsNone(self.kb._get_kv("nonexist"))

    def test_upsert_kv(self):
        self.kb._set_kv("name", "Marco")
        self.kb._set_kv("name", "Luca")
        self.assertEqual(self.kb._get_kv("name"), "Luca")


class TestKnowledgeBaseTopics(unittest.TestCase):
    """Test topics (argomenti discussi)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.kb = KnowledgeBase(storage_path=self.tmpdir)

    def tearDown(self):
        self.kb.close()

    def test_set_and_get_topics(self):
        self.kb._set_topic("python", 5)
        self.kb._set_topic("arte", 3)
        topics = self.kb._get_topics()
        self.assertEqual(topics["python"], 5)
        self.assertEqual(topics["arte"], 3)

    def test_upsert_topic(self):
        self.kb._set_topic("python", 2)
        self.kb._set_topic("python", 10)
        self.assertEqual(self.kb._get_topics()["python"], 10)

    def test_empty_topics(self):
        self.assertEqual(self.kb._get_topics(), {})

    def test_count_topics_from_text(self):
        self.kb._count_topics("sto imparando python e javascript")
        topics = self.kb._get_topics()
        self.assertIn("programmazione", topics)
        self.assertGreaterEqual(topics["programmazione"], 1)

    def test_count_topics_multiple_categories(self):
        self.kb._count_topics("mi piace l'ai e anche il calcio")
        topics = self.kb._get_topics()
        self.assertIn("tecnologia", topics)
        self.assertIn("sport", topics)


class TestKnowledgeBaseFacts(unittest.TestCase):
    """Test CRUD fatti con FTS5."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.kb = KnowledgeBase(storage_path=self.tmpdir)

    def tearDown(self):
        self.kb.close()

    def test_add_fact_returns_id(self):
        fid = self.kb.add_fact("Python è un linguaggio di programmazione")
        self.assertIsInstance(fid, int)
        self.assertGreater(fid, 0)

    def test_add_multiple_facts(self):
        id1 = self.kb.add_fact("Fatto uno")
        id2 = self.kb.add_fact("Fatto due")
        self.assertNotEqual(id1, id2)

    def test_facts_count(self):
        self.assertEqual(self.kb.get_facts_count(), 0)
        self.kb.add_fact("Primo")
        self.kb.add_fact("Secondo")
        self.assertEqual(self.kb.get_facts_count(), 2)

    def test_search_facts_fts5(self):
        self.kb.add_fact("Python è ottimo per il machine learning")
        self.kb.add_fact("JavaScript domina il frontend web")
        self.kb.add_fact("Rust è veloce e sicuro")

        results = self.kb.search_facts("Python")
        self.assertEqual(len(results), 1)
        self.assertIn("Python", results[0]["content"])

    def test_search_facts_no_match(self):
        self.kb.add_fact("Python è bello")
        results = self.kb.search_facts("dinosauri")
        self.assertEqual(results, [])

    def test_search_facts_with_limit(self):
        for i in range(20):
            self.kb.add_fact(f"Concetto numero {i} sulla programmazione")
        results = self.kb.search_facts("programmazione", limit=5)
        self.assertLessEqual(len(results), 5)

    def test_search_facts_returns_metadata(self):
        self.kb.add_fact("Informazione importante", source="chat")
        results = self.kb.search_facts("importante")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source"], "chat")
        self.assertIn("created_at", results[0])

    def test_add_fact_with_source(self):
        fid = self.kb.add_fact("Dato dal web", source="web_search")
        results = self.kb.search_facts("web")
        self.assertEqual(results[0]["source"], "web_search")


class TestKnowledgeBaseSanitizeFTS(unittest.TestCase):
    """Test sanitizzazione query FTS5."""

    def test_normal_query(self):
        result = KnowledgeBase._sanitize_fts("hello world")
        self.assertEqual(result, '"hello"* "world"*')

    def test_empty_query(self):
        result = KnowledgeBase._sanitize_fts("")
        self.assertEqual(result, '""')

    def test_none_query(self):
        result = KnowledgeBase._sanitize_fts(None)
        self.assertEqual(result, '""')

    def test_special_chars_stripped(self):
        result = KnowledgeBase._sanitize_fts('hello* +world -foo')
        # special chars are stripped from edges, but * is added for prefix
        self.assertNotIn("+", result)
        self.assertNotIn("-", result)
        # Words should be quoted with prefix
        self.assertIn('"hello"*', result)
        self.assertIn('"world"*', result)
        self.assertIn('"foo"*', result)

    def test_quotes_removed(self):
        result = KnowledgeBase._sanitize_fts('"hello" "world"')
        self.assertEqual(result, '"hello"* "world"*')


class TestKnowledgeBaseKnowledgeProperty(unittest.TestCase):
    """Test backward-compatible .knowledge property."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.kb = KnowledgeBase(storage_path=self.tmpdir)

    def tearDown(self):
        self.kb.close()

    def test_returns_dict(self):
        k = self.kb.knowledge
        self.assertIsInstance(k, dict)

    def test_has_required_keys(self):
        k = self.kb.knowledge
        self.assertIn("user_profile", k)
        self.assertIn("learned_facts", k)
        self.assertIn("topics_discussed", k)
        self.assertIn("last_updated", k)

    def test_profile_reflects_kv(self):
        self.kb._set_kv("name", "Alice")
        self.kb._set_kv("interests", json.dumps(["coding", "music"]))
        k = self.kb.knowledge
        self.assertEqual(k["user_profile"]["name"], "Alice")
        self.assertEqual(k["user_profile"]["interests"], ["coding", "music"])

    def test_topics_reflected(self):
        self.kb._set_topic("scienza", 7)
        k = self.kb.knowledge
        self.assertEqual(k["topics_discussed"]["scienza"], 7)


class TestKnowledgeBaseSearch(unittest.TestCase):
    """Test unified search() method."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.kb = KnowledgeBase(storage_path=self.tmpdir)

    def tearDown(self):
        self.kb.close()

    def test_search_returns_structure(self):
        results = self.kb.search("test")
        self.assertIn("user_profile", results)
        self.assertIn("relevant_topics", results)
        self.assertIn("related_facts", results)

    def test_search_finds_matching_topic(self):
        self.kb._set_topic("python", 5)
        results = self.kb.search("python")
        self.assertEqual(len(results["relevant_topics"]), 1)
        self.assertEqual(results["relevant_topics"][0]["topic"], "python")
        self.assertEqual(results["relevant_topics"][0]["mentions"], 5)

    def test_search_finds_matching_facts(self):
        self.kb.add_fact("Flask è un micro-framework web per Python")
        results = self.kb.search("Flask")
        self.assertEqual(len(results["related_facts"]), 1)

    def test_search_matches_user_profile(self):
        self.kb._set_kv("name", "Marco")
        results = self.kb.search("Marco")
        self.assertTrue(results["user_profile"])

    def test_search_no_match(self):
        results = self.kb.search("xyznotexist")
        self.assertEqual(results["user_profile"], {})
        self.assertEqual(results["relevant_topics"], [])
        self.assertEqual(results["related_facts"], [])


class TestKnowledgeBaseMigration(unittest.TestCase):
    """Test migrazione da JSON a SQLite."""

    def test_migrates_from_json(self):
        tmpdir = tempfile.mkdtemp()
        json_path = os.path.join(tmpdir, "knowledge_base.json")
        old_data = {
            "user_profile": {
                "name": "TestUser",
                "interests": ["coding", "chess"],
                "language": "italiano",
                "created_at": "2025-01-01T00:00:00",
            },
            "topics_discussed": {
                "programmazione": 10,
                "scienza": 3,
            },
            "learned_facts": [
                "Python supporta il duck typing",
                {"content": "Flask usa Jinja2", "source": "docs"},
            ],
            "last_updated": "2025-06-01T12:00:00",
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(old_data, f)

        kb = KnowledgeBase(storage_path=json_path)

        # Profile migrated
        self.assertEqual(kb._get_kv("name"), "TestUser")
        interests = json.loads(kb._get_kv("interests", "[]"))
        self.assertIn("coding", interests)

        # Topics migrated
        topics = kb._get_topics()
        self.assertEqual(topics["programmazione"], 10)

        # Facts migrated
        self.assertEqual(kb.get_facts_count(), 2)

        # JSON renamed
        self.assertTrue(os.path.exists(json_path + ".migrated"))
        self.assertFalse(os.path.exists(json_path))

        kb.close()

    def test_no_migration_without_json(self):
        tmpdir = tempfile.mkdtemp()
        kb = KnowledgeBase(storage_path=tmpdir)
        self.assertEqual(kb.get_facts_count(), 0)
        self.assertEqual(kb._get_topics(), {})
        kb.close()


class TestKnowledgeBaseUserContext(unittest.TestCase):
    """Test generazione contesto utente."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.kb = KnowledgeBase(storage_path=self.tmpdir)

    def tearDown(self):
        self.kb.close()

    def test_empty_context(self):
        self.assertEqual(self.kb.get_user_context(), "")

    def test_context_with_name(self):
        self.kb._set_kv("name", "Alice")
        ctx = self.kb.get_user_context()
        self.assertIn("Alice", ctx)
        self.assertIn("PROFILO UTENTE", ctx)

    def test_context_with_interests(self):
        self.kb._set_kv("interests", json.dumps(["AI", "robotica"]))
        ctx = self.kb.get_user_context()
        self.assertIn("AI", ctx)

    def test_context_with_topics(self):
        self.kb._set_topic("programmazione", 8)
        ctx = self.kb.get_user_context()
        self.assertIn("programmazione", ctx)


class TestKnowledgeBaseUpdateFromConv(unittest.TestCase):
    """Test update_from_conversation."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.kb = KnowledgeBase(storage_path=self.tmpdir)

    def tearDown(self):
        self.kb.close()

    def test_extracts_name(self):
        msgs = [{"role": "user", "content": "Ciao, mi chiamo Marco"}]
        self.kb.update_from_conversation(msgs)
        self.assertEqual(self.kb._get_kv("name"), "Marco")

    def test_extracts_interest(self):
        msgs = [{"role": "user",
                 "content": "Mi interessa la programmazione funzionale"}]
        self.kb.update_from_conversation(msgs)
        interests = json.loads(self.kb._get_kv("interests", "[]"))
        self.assertTrue(len(interests) > 0)

    def test_counts_topics(self):
        msgs = [{"role": "user", "content": "Parliamo di python e javascript"}]
        self.kb.update_from_conversation(msgs)
        topics = self.kb._get_topics()
        self.assertIn("programmazione", topics)

    def test_ignores_assistant_messages(self):
        msgs = [{"role": "assistant", "content": "Mi chiamo Bot"}]
        self.kb.update_from_conversation(msgs)
        self.assertIsNone(self.kb._get_kv("name"))


class TestKnowledgeBaseStats(unittest.TestCase):
    """Test get_stats()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.kb = KnowledgeBase(storage_path=self.tmpdir)

    def tearDown(self):
        self.kb.close()

    def test_stats_empty(self):
        stats = self.kb.get_stats()
        self.assertEqual(stats["facts"], 0)
        self.assertEqual(stats["topics"], 0)
        self.assertIn("db_path", stats)

    def test_stats_after_data(self):
        self.kb.add_fact("Fatto 1")
        self.kb.add_fact("Fatto 2")
        self.kb._set_topic("test", 1)
        stats = self.kb.get_stats()
        self.assertEqual(stats["facts"], 2)
        self.assertEqual(stats["topics"], 1)


class TestKnowledgeBaseClose(unittest.TestCase):
    """Test close() e riapertura."""

    def test_close_sets_conn_none(self):
        tmpdir = tempfile.mkdtemp()
        kb = KnowledgeBase(storage_path=tmpdir)
        kb.close()
        self.assertIsNone(kb._conn)

    def test_data_persists_after_close(self):
        tmpdir = tempfile.mkdtemp()
        kb = KnowledgeBase(storage_path=tmpdir)
        kb.add_fact("Dato persistente")
        kb._set_kv("name", "Test")
        kb.close()

        kb2 = KnowledgeBase(storage_path=tmpdir)
        self.assertEqual(kb2.get_facts_count(), 1)
        self.assertEqual(kb2._get_kv("name"), "Test")
        kb2.close()


if __name__ == "__main__":
    unittest.main()
