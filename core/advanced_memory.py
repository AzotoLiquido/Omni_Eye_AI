"""
Sistema di Memoria Conversazionale Avanzata
Gestisce contesto intelligente, summarization, entità e knowledge base
"""

import json
import os
import sqlite3
import threading
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging
import re
import config
from core.memory import ConversationMemory

logger = logging.getLogger(__name__)


class ContextManager:
    """Gestisce il contesto e la compressione automatica dei messaggi"""
    
    def __init__(self, max_context_tokens: int = 3000):
        """
        Args:
            max_context_tokens: Massimo numero di token da mantenere in contesto
        """
        self.max_context_tokens = max_context_tokens
        
    def estimate_tokens(self, text: str) -> int:
        """Stima approssimativa dei token.
        P3-13: Slightly better estimate: count words + punctuation overhead.
        Still rough but better than pure char/4.
        """
        # Average: ~1.3 tokens per word for English/Italian
        words = len(text.split())
        return max(int(words * 1.3), len(text) // 4)
    
    def calculate_messages_tokens(self, messages: List[Dict]) -> int:
        """Calcola il totale dei token nei messaggi"""
        total = 0
        for msg in messages:
            total += self.estimate_tokens(msg.get('content', ''))
        return total
    
    def compress_context(self, messages: List[Dict], ai_engine) -> List[Dict]:
        """
        Comprime i messaggi vecchi mantenendo quelli più recenti.

        B1 perf-fix: rimossa la summarizzazione sincrona con AI che bloccava
        il thread per 5-30 secondi.  Ora usa troncamento semplice:
        mantiene gli ultimi N messaggi entro il budget token.
        
        Args:
            messages: Lista completa dei messaggi
            ai_engine: Engine AI (non più usato, mantenuto per compatibilità)
            
        Returns:
            Lista di messaggi compressa
        """
        total_tokens = self.calculate_messages_tokens(messages)
        
        # Se sotto il limite, restituisci tutto
        if total_tokens <= self.max_context_tokens:
            return messages
        
        # ── Troncamento progressivo: mantieni quanti più messaggi recenti possibile ──
        # Parti da tutti i messaggi e rimuovi i più vecchi finché non rientri nel budget.
        # Mantieni SEMPRE almeno gli ultimi 2 messaggi (ultima coppia user/assistant).
        truncated = list(messages)
        while len(truncated) > 2:
            truncated = truncated[1:]
            if self.calculate_messages_tokens(truncated) <= self.max_context_tokens:
                break
        
        # Se abbiamo tagliato messaggi, aggiungi nota di contesto
        n_dropped = len(messages) - len(truncated)
        if n_dropped > 0:
            # Costruisci riassunto veloce: prima riga di ogni messaggio tagliato
            summary_lines = []
            for msg in messages[:min(n_dropped, 6)]:
                role = msg.get('role', '?').upper()
                first_line = msg.get('content', '')[:80].split('\n')[0]
                summary_lines.append(f"- {role}: {first_line}…")
            summary_text = "\n".join(summary_lines)
            if n_dropped > 6:
                summary_text += f"\n- … e altri {n_dropped - 6} messaggi"
            
            truncated.insert(0, {
                'role': 'system',
                'content': (
                    f"[CONTESTO: {n_dropped} messaggi precedenti omessi per limite contesto]\n"
                    f"{summary_text}"
                ),
                'timestamp': datetime.now().isoformat(),
                'is_summary': True,
            })
        
        return truncated
    
    def _generate_summary(self, messages: List[Dict], ai_engine) -> str:
        """Genera un riassunto dei messaggi usando l'AI.
        
        NOTA: Non più usato da compress_context() (B1 perf-fix).
        Mantenuto per compatibilità API.
        """
        parts = []
        for msg in messages[:6]:
            role = msg.get('role', '?').upper()
            first_line = msg.get('content', '')[:120].split('\n')[0]
            parts.append(f"- {role}: {first_line}")
        return "\n".join(parts) or f"Conversazione con {len(messages)} messaggi."


class EntityTracker:
    """Traccia entitÃ  menzionate nelle conversazioni (persone, luoghi, fatti)"""

    # P1-4: Parole comuni da escludere (frozenset class-level)
    _STOP_NAMES: frozenset = frozenset({
        'sono', 'come', 'cosa', 'tutto', 'questo', 'quella', 'quello',
        'ancora', 'sempre', 'grazie', 'ciao', 'buongiorno', 'buonasera',
        'molto', 'poco', 'bene', 'male', 'dopo', 'prima', 'oggi',
        'domani', 'ieri', 'perché', 'quando', 'dove', 'quindi',
    })

    _MAX_RELEVANT_ENTITIES = 10  # P2-4: cap globale su risultati di get_relevant_entities

    def __init__(self, storage_path: str):
        """
        Args:
            storage_path: Percorso al file JSON per salvare le entitÃ 
        """
        self.storage_path = storage_path
        self._lock = threading.RLock()  # RLock: extract_and_save → _save_entities nesting
        self.entities = self._load_entities()
    
    def _load_entities(self) -> Dict:
        """Carica le entitÃ  dal disco"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Errore caricamento entitÃ  da %s: %s", self.storage_path, e)
        
        return {
            'people': {},        # nome -> [contesto1, contesto2, ...]
            'places': {},        # luogo -> [contesto1, contesto2, ...]
            'facts': {},         # categoria -> [fatto1, fatto2, ...]
            'preferences': {},   # tipo -> valore
            'dates': []          # eventi con date
        }
    
    def _save_entities(self) -> bool:
        """Salva le entitÃ  sul disco (atomico con temp+replace)"""
        tmp_path = self.storage_path + ".tmp"
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with self._lock:
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    json.dump(self.entities, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, self.storage_path)
            return True
        except Exception as e:
            logger.error("Errore salvataggio entit\u00e0: %s", e)
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            return False
    
    def extract_and_save(self, message: str, role: str) -> None:
        """
        Estrae entitÃ  da un messaggio e le salva
        
        Args:
            message: Testo del messaggio
            role: 'user' o 'assistant'
        """
        # Estrai solo dai messaggi utente
        if role != 'user':
            return
        
        timestamp = datetime.now().isoformat()
        
        # Lock per thread safety (le _extract_* modificano self.entities)
        with self._lock:
            self._extract_names(message, timestamp)
            self._extract_preferences(message, timestamp)
            self._extract_dates(message, timestamp)
            self._save_entities()  # P2-5: save inside lock to avoid gap
    
    def _extract_names(self, text: str, timestamp: str) -> None:
        """Estrae nomi propri dal testo con pattern espliciti"""
        # Pattern espliciti per nomi (piÃ¹ precisi del semplice "parola maiuscola")
        name_patterns = [
            r'(?:mi chiamo|si chiama|chiamato|chiamata|conosco)\s+([A-Z][a-zÃ -Ãº]+)',
            r'(?:parlo con|parlato con|visto|incontrato)\s+([A-Z][a-zÃ -Ãº]+)',
        ]

        
        # P3-14: Removed re.IGNORECASE since patterns explicitly require uppercase first letter
        for pattern in name_patterns:
            matches = re.findall(pattern, text)
            for name in matches:
                name = name.strip('.,!?;:')
                if len(name) <= 2 or name.lower() in self._STOP_NAMES:
                    continue
                if not name[0].isupper():
                    continue
                
                if name not in self.entities['people']:
                    self.entities['people'][name] = []
                if len(self.entities['people'][name]) < 3:
                    self.entities['people'][name].append({
                        'context': text[:100],
                        'timestamp': timestamp
                    })
    
    def _extract_preferences(self, text: str, timestamp: str) -> None:
        """Estrae preferenze dell'utente"""
        _MAX_PREFERENCES = 200
        text_lower = text.lower()
        
        # Pattern per preferenze
        patterns = {
            'mi piace': r'mi piace (.*?)(?:\.|,|$)',
            'preferisco': r'preferisco (.*?)(?:\.|,|$)',
            'odio': r'odio (.*?)(?:\.|,|$)',
            'non mi piace': r'non mi piace (.*?)(?:\.|,|$)',
        }
        
        for sentiment, pattern in patterns.items():
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                preference = match.strip()
                if preference:
                    key = f"{sentiment}:{preference}"
                    self.entities['preferences'][key] = {
                        'value': preference,
                        'sentiment': sentiment,
                        'timestamp': timestamp
                    }
        
        # Tronca a _MAX_PREFERENCES piÃ¹ recenti per evitare crescita illimitata
        if len(self.entities['preferences']) > _MAX_PREFERENCES:
            sorted_prefs = sorted(
                self.entities['preferences'].items(),
                key=lambda x: x[1].get('timestamp', ''),
            )
            self.entities['preferences'] = dict(sorted_prefs[-_MAX_PREFERENCES:])
    
    def _extract_dates(self, text: str, timestamp: str) -> None:
        """Estrae riferimenti a date ed eventi"""
        date_patterns = [
            r'(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})',  # 12/05/2024
            r'(oggi|domani|ieri)',
            r'(lunedÃ¬|martedÃ¬|mercoledÃ¬|giovedÃ¬|venerdÃ¬|sabato|domenica)',
        ]
        
        _MAX_DATES = 100

        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                self.entities['dates'].append({
                    'date_ref': match,
                    'context': text[:100],
                    'timestamp': timestamp
                })

        # Tronca a _MAX_DATES piÃ¹ recenti per evitare crescita illimitata
        if len(self.entities['dates']) > _MAX_DATES:
            self.entities['dates'] = self.entities['dates'][-_MAX_DATES:]
    
    def get_relevant_entities(self, current_message: str) -> str:
        """
        Trova entitÃ  rilevanti per il messaggio corrente
        
        Returns:
            Stringa formattata con le entitÃ  rilevanti
        """
        relevant = []
        text_lower = current_message.lower()
        
        # Cerca persone menzionate
        for name, contexts in self.entities['people'].items():
            if name.lower() in text_lower:
                relevant.append(f"• {name}: menzionato precedentemente")
            if len(relevant) >= self._MAX_RELEVANT_ENTITIES:
                break
        
        # Cerca preferenze rilevanti per la query (non tutte)
        for key, pref in self.entities['preferences'].items():
            if pref['value'].lower() in text_lower or pref['sentiment'].lower() in text_lower:
                relevant.append(f"• Preferenza: {pref['sentiment']} {pref['value']}")
                if len(relevant) >= self._MAX_RELEVANT_ENTITIES:
                    break
        
        if relevant:
            return "[INFORMAZIONI CONTESTUALI]\n" + "\n".join(relevant) + "\n"
        
        return ""



class KnowledgeBase:
    """Memoria a lungo termine per informazioni cross-conversazione.

    Backend: SQLite con FTS5 per ricerca full-text (~1-5ms per query,
    scala a 100K+ entry senza degrado). Migra automaticamente i dati
    dal vecchio knowledge_base.json al primo avvio.
    """

    _DEFAULT_DB_NAME = "knowledge.db"

    def __init__(self, storage_path: str = None):
        """
        Args:
            storage_path: Percorso al vecchio file JSON (usato per migrazione)
                          oppure directory dove creare il DB SQLite.
                          Se None, usa config.DATA_DIR.
        """
        if storage_path and storage_path.endswith(".json"):
            self._json_path = storage_path
            db_dir = os.path.dirname(storage_path)
        else:
            self._json_path = None
            db_dir = storage_path or os.path.join(config.DATA_DIR)

        os.makedirs(db_dir, exist_ok=True)
        self._db_path = os.path.join(db_dir, self._DEFAULT_DB_NAME)
        self._lock = threading.RLock()
        self._conn = None

        self._connect()
        self._init_tables()
        self._maybe_migrate_json()

    # -- Connection & Schema -------------------------------------------------

    def _connect(self):
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")

    def _init_tables(self):
        c = self._conn
        c.execute("""
            CREATE TABLE IF NOT EXISTS kv (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS topics (
                name  TEXT PRIMARY KEY,
                count INTEGER NOT NULL DEFAULT 0
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                content    TEXT NOT NULL,
                source     TEXT DEFAULT '',
                created_at TEXT NOT NULL
            )
        """)
        c.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
            USING fts5(content, content=facts, content_rowid=id)
        """)
        c.executescript("""
            CREATE TRIGGER IF NOT EXISTS kb_facts_ai AFTER INSERT ON facts BEGIN
                INSERT INTO facts_fts(rowid, content)
                VALUES (new.id, new.content);
            END;
            CREATE TRIGGER IF NOT EXISTS kb_facts_ad AFTER DELETE ON facts BEGIN
                INSERT INTO facts_fts(facts_fts, rowid, content)
                VALUES ('delete', old.id, old.content);
            END;
            CREATE TRIGGER IF NOT EXISTS kb_facts_au AFTER UPDATE ON facts BEGIN
                INSERT INTO facts_fts(facts_fts, rowid, content)
                VALUES ('delete', old.id, old.content);
                INSERT INTO facts_fts(rowid, content)
                VALUES (new.id, new.content);
            END;
        """)
        c.commit()

    # -- JSON -> SQLite migration --------------------------------------------

    def _maybe_migrate_json(self):
        """Migra dati dal vecchio knowledge_base.json se presente."""
        if not self._json_path or not os.path.exists(self._json_path):
            return
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM kv").fetchone()
            if row[0] > 0:
                return
        try:
            with open(self._json_path, 'r', encoding='utf-8') as f:
                old = json.load(f)
            profile = old.get('user_profile', {})
            for key in ('name', 'language', 'created_at'):
                if profile.get(key):
                    self._set_kv(key, str(profile[key]))
            if profile.get('interests'):
                self._set_kv('interests', json.dumps(profile['interests'],
                                                     ensure_ascii=False))
            if profile.get('expertise'):
                self._set_kv('expertise', json.dumps(profile['expertise'],
                                                     ensure_ascii=False))
            for topic, count in old.get('topics_discussed', {}).items():
                self._set_topic(topic, count)
            for fact in old.get('learned_facts', []):
                if isinstance(fact, str):
                    self.add_fact(fact)
                elif isinstance(fact, dict):
                    self.add_fact(fact.get('content', str(fact)),
                                 fact.get('source', ''))
            if old.get('last_updated'):
                self._set_kv('last_updated', old['last_updated'])
            logger.info("KB: migrati dati da %s -> %s",
                        self._json_path, self._db_path)
            backup = self._json_path + ".migrated"
            try:
                os.rename(self._json_path, backup)
            except OSError:
                pass
        except Exception as e:
            logger.warning("KB: migrazione JSON fallita: %s", e)

    # -- Low-level helpers ---------------------------------------------------

    def _set_kv(self, key, value):
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)",
                (key, value))
            self._conn.commit()

    def _get_kv(self, key, default=None):
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def _set_topic(self, name, count):
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO topics (name, count) VALUES (?, ?)",
                (name, count))
            self._conn.commit()

    def _get_topics(self):
        with self._lock:
            rows = self._conn.execute(
                "SELECT name, count FROM topics").fetchall()
        return {r[0]: r[1] for r in rows}

    # -- Facts CRUD ----------------------------------------------------------

    def add_fact(self, content, source=""):
        """Aggiunge un fatto alla KB. Restituisce l'ID."""
        now = datetime.now().isoformat()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO facts (content, source, created_at) "
                "VALUES (?, ?, ?)",
                (content, source, now))
            self._conn.commit()
        return cur.lastrowid

    def search_facts(self, query, limit=10):
        """Ricerca full-text FTS5 nei fatti. O(log n), ~1-5ms."""
        safe_q = self._sanitize_fts(query)
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT f.id, f.content, f.source, f.created_at, rank "
                    "FROM facts_fts "
                    "JOIN facts f ON facts_fts.rowid = f.id "
                    "WHERE facts_fts MATCH ? "
                    "ORDER BY rank LIMIT ?",
                    (safe_q, limit)).fetchall()
                return [dict(r) for r in rows]
            except sqlite3.OperationalError:
                rows = self._conn.execute(
                    "SELECT * FROM facts WHERE content LIKE ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (f"%{query}%", limit)).fetchall()
                return [dict(r) for r in rows]

    def get_facts_count(self):
        """Numero di fatti nella KB."""
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM facts").fetchone()[0]

    @staticmethod
    def _sanitize_fts(query):
        """Escape FTS5 special chars per prevenire injection.

        Usa prefix matching (word*) per catturare anche forme
        derivate (es. 'async' trova 'asyncio', 'async/await').
        """
        if not query or not query.strip():
            return '""'
        words = query.split()
        safe = []
        for w in words:
            w = w.replace('"', '').strip('*+-~^')
            if w and len(w) >= 2:
                safe.append(f'"{w}"*')
            elif w:
                safe.append(f'"{w}"')
        return ' '.join(safe) if safe else '""'

    # -- Backward-compatible property ----------------------------------------

    @property
    def knowledge(self):
        """Dict compatibile col vecchio formato JSON.

        Usato da main.py /api/knowledge/summary e export_knowledge_summary().
        """
        profile = {
            'name': self._get_kv('name'),
            'interests': json.loads(self._get_kv('interests', '[]')),
            'expertise': json.loads(self._get_kv('expertise', '[]')),
            'language': self._get_kv('language', 'italiano'),
            'created_at': self._get_kv('created_at',
                                       datetime.now().isoformat()),
        }
        return {
            'user_profile': profile,
            'learned_facts': [],
            'topics_discussed': self._get_topics(),
            'last_updated': self._get_kv('last_updated',
                                         datetime.now().isoformat()),
        }

    # -- Public API (backward compatible) ------------------------------------

    def update_from_conversation(self, messages):
        """Aggiorna la KB dai messaggi della conversazione."""
        with self._lock:
            self._conn.execute("DELETE FROM topics")
            self._conn.commit()
        for msg in messages:
            if msg['role'] == 'user':
                content_lower = msg['content'].lower()
                if 'mi chiamo' in content_lower or 'sono' in content_lower:
                    self._extract_user_name(msg['content'])
                if any(w in content_lower
                       for w in ['interessa', 'piace', 'passione']):
                    self._extract_interests(msg['content'])
                self._count_topics(msg['content'])
        self._set_kv('last_updated', datetime.now().isoformat())

    def _extract_user_name(self, text):
        patterns = [r'mi chiamo (\w+)', r'il mio nome \u00e8 (\w+)']
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                if len(name) > 2 and name[0].isupper():
                    self._set_kv('name', name)
                    return

    def _extract_interests(self, text):
        text_lower = text.lower()
        triggers = [
            ('mi interessa', 'mi interessa'),
            ('mi piace', 'mi piace'),
            ('la mia passione', 'la mia passione'),
            ('sono appassionato di', 'sono appassionato di'),
        ]
        current = json.loads(self._get_kv('interests', '[]'))
        changed = False
        for trigger, split_on in triggers:
            if trigger in text_lower:
                after = text_lower.split(split_on, 1)[1].split('.')[0]
                interest = after.strip()
                if interest and interest not in current:
                    current.append(interest)
                    changed = True
        if changed:
            self._set_kv('interests',
                         json.dumps(current, ensure_ascii=False))

    def _count_topics(self, text):
        topic_map = {
            'programmazione': ['python', 'javascript', 'codice', 'programma',
                               'sviluppo'],
            'scienza': ['fisica', 'chimica', 'biologia', 'scienza'],
            'tecnologia': ['ai', 'intelligenza artificiale', 'computer',
                           'tecnologia'],
            'arte': ['arte', 'pittura', 'musica', 'letteratura'],
            'sport': ['calcio', 'basket', 'sport', 'tennis'],
        }
        text_lower = text.lower()
        for topic, keywords in topic_map.items():
            if any(kw in text_lower for kw in keywords):
                with self._lock:
                    self._conn.execute(
                        "INSERT INTO topics (name, count) VALUES (?, 1) "
                        "ON CONFLICT(name) DO UPDATE SET count = count + 1",
                        (topic,))
                    self._conn.commit()

    def get_user_context(self):
        """Genera un contesto sull'utente dalla knowledge base."""
        context_parts = []
        name = self._get_kv('name')
        if name:
            context_parts.append(f"Nome utente: {name}")
        interests = json.loads(self._get_kv('interests', '[]'))
        if interests:
            context_parts.append(f"Interessi: {', '.join(interests[:3])}")
        topics = self._get_topics()
        if topics:
            top = sorted(topics.items(), key=lambda x: x[1],
                         reverse=True)[:3]
            topics_str = ", ".join(f"{t[0]} ({t[1]}x)" for t in top)
            context_parts.append(f"Topic frequenti: {topics_str}")
        if context_parts:
            return "[PROFILO UTENTE]\n" + "\n".join(context_parts) + "\n"
        return ""

    def search(self, query, limit=10):
        """Ricerca unificata: profilo + topics + facts (FTS5)."""
        results = {
            'user_profile': {},
            'relevant_topics': [],
            'related_facts': [],
        }
        query_lower = query.lower()
        name = self._get_kv('name')
        if name and name.lower() in query_lower:
            results['user_profile'] = self.knowledge['user_profile']
        for topic, count in self._get_topics().items():
            if topic.lower() in query_lower or query_lower in topic.lower():
                results['relevant_topics'].append({
                    'topic': topic, 'mentions': count})
        results['related_facts'] = self.search_facts(query, limit=limit)
        return results

    def close(self):
        """Chiude la connessione al DB."""
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None

    def get_stats(self):
        """Statistiche sulla knowledge base."""
        with self._lock:
            facts_count = self._conn.execute(
                "SELECT COUNT(*) FROM facts").fetchone()[0]
            topics_count = self._conn.execute(
                "SELECT COUNT(*) FROM topics").fetchone()[0]
        return {
            'facts': facts_count,
            'topics': topics_count,
            'db_path': self._db_path,
        }

class AdvancedMemory(ConversationMemory):
    """Sistema di memoria avanzata con context management e knowledge base"""
    
    def __init__(self):
        """Inizializza il sistema di memoria avanzata"""
        super().__init__()
        
        # Inizializza i componenti avanzati
        self.context_manager = ContextManager(max_context_tokens=3000)
        
        # Path per i file di storage
        kb_path = os.path.join(config.DATA_DIR, 'knowledge_base.json')
        entities_path = os.path.join(config.DATA_DIR, 'entities.json')
        
        self.knowledge_base = KnowledgeBase(kb_path)
        self.entity_tracker = EntityTracker(entities_path)
    
    def add_message_advanced(
        self, 
        conv_id: str, 
        role: str, 
        content: str,
        extract_entities: bool = True
    ) -> bool:
        """
        Aggiunge un messaggio con elaborazione avanzata
        
        Args:
            conv_id: ID della conversazione
            role: 'user' o 'assistant'
            content: Contenuto del messaggio
            extract_entities: Se estrarre entitÃ  dal messaggio
            
        Returns:
            True se il salvataggio Ã¨ riuscito
        """
        # Salva il messaggio normalmente
        success = self.add_message(conv_id, role, content)
        
        if success and extract_entities:
            # Estrai entitÃ  dai messaggi utente
            if role == 'user':
                self.entity_tracker.extract_and_save(content, role)
            
            # Aggiorna knowledge base periodicamente
            conversation = self.load_conversation(conv_id)
            if conversation and len(conversation['messages']) % 5 == 0:
                # Ogni 5 messaggi aggiorna la knowledge base
                self.knowledge_base.update_from_conversation(conversation['messages'])
        
        return success
    
    def get_smart_context(
        self, 
        conv_id: str, 
        ai_engine,
        include_knowledge: bool = True
    ) -> Tuple[List[Dict], str]:
        """
        Ottiene il contesto ottimizzato per l'AI con compressione intelligente
        
        Args:
            conv_id: ID della conversazione
            ai_engine: Engine AI per generare riassunti
            include_knowledge: Se includere informazioni dalla knowledge base
            
        Returns:
            Tupla (messaggi_ottimizzati, contesto_addizionale)
        """
        # Carica la conversazione
        conversation = self.load_conversation(conv_id)
        if not conversation:
            return [], ""
        
        messages = conversation.get('messages', [])
        
        # Comprimi il contesto se necessario
        optimized_messages = self.context_manager.compress_context(messages, ai_engine)
        
        # Prepara contesto addizionale
        additional_context = ""
        
        if include_knowledge:
            # Aggiungi profilo utente dalla knowledge base
            user_context = self.knowledge_base.get_user_context()
            if user_context:
                additional_context += user_context + "\n"
            
            # Aggiungi entitÃ  rilevanti se c'Ã¨ un ultimo messaggio
            if messages:
                last_message = messages[-1]['content']
                entities_context = self.entity_tracker.get_relevant_entities(last_message)
                if entities_context:
                    additional_context += entities_context
        
        return optimized_messages, additional_context
    
    def get_conversation_stats(self, conv_id: str) -> Dict:
        """Ottiene statistiche sulla conversazione"""
        conversation = self.load_conversation(conv_id)
        if not conversation:
            return {}
        
        messages = conversation.get('messages', [])
        user_msgs = [m for m in messages if m['role'] == 'user']
        assistant_msgs = [m for m in messages if m['role'] == 'assistant']
        
        return {
            'total_messages': len(messages),
            'user_messages': len(user_msgs),
            'assistant_messages': len(assistant_msgs),
            'total_tokens_estimate': self.context_manager.calculate_messages_tokens(messages),
            'created_at': conversation.get('created_at'),
            'updated_at': conversation.get('updated_at'),
        }
    
    def search_in_knowledge(self, query: str) -> Dict:
        """
        Cerca informazioni nella knowledge base (FTS5).

        Args:
            query: Query di ricerca

        Returns:
            Dizionario con risultati rilevanti
        """
        return self.knowledge_base.search(query)
    
    def export_knowledge_summary(self) -> str:
        """Esporta un riassunto testuale della knowledge base"""
        stats = self.knowledge_base.get_stats()
        kb = self.knowledge_base.knowledge

        summary_lines = ["=== KNOWLEDGE BASE SUMMARY ===\n"]

        # User Profile
        profile = kb.get('user_profile', {})
        if profile.get('name'):
            summary_lines.append(f"\U0001f464 Utente: {profile['name']}")

        if profile.get('interests'):
            summary_lines.append(f"\U0001f4a1 Interessi: {', '.join(profile['interests'])}")

        # Topics
        topics = kb.get('topics_discussed', {})
        if topics:
            summary_lines.append("\n\U0001f4ca Topic discussi:")
            sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)
            for topic, count in sorted_topics[:5]:
                summary_lines.append(f"   \u2022 {topic}: {count} volte")

        # Facts
        facts_count = stats.get('facts_count', 0)
        if facts_count:
            summary_lines.append(f"\n\U0001f9e0 Fatti memorizzati: {facts_count}")

        # Entities
        if hasattr(self, 'entity_tracker'):
            people = self.entity_tracker.entities.get('people', {})
            if people:
                names = ', '.join(list(people.keys())[:5])
                summary_lines.append(f"\n\U0001f465 Persone menzionate: {names}")

        summary_lines.append(f"\n\u23f0 Ultimo aggiornamento: {kb.get('last_updated', 'N/A')}")

        return '\n'.join(summary_lines)
