"""
Sistema di Memoria - Gestisce lo storico delle conversazioni
"""

import json
import logging
import os
import re
import threading
import uuid
from datetime import datetime
import time as _time
from typing import List, Dict, Optional
import config

logger = logging.getLogger(__name__)

# Regex per validare conv_id (previene path traversal)
_SAFE_CONV_ID = re.compile(r'^[a-zA-Z0-9_\-]+$')


def _validate_conv_id(conv_id: str) -> str:
    """Validazione conv_id per prevenire path traversal."""
    if not conv_id or not _SAFE_CONV_ID.match(conv_id):
        raise ValueError(f"conv_id non valido: {conv_id!r}")
    return conv_id


# File locking cross-platform (riservato per uso futuro, attualmente
# _save_conversation usa la strategia atomica temp+replace)
# Se necessario, importare _lock_file/_unlock_file da qui.


class ConversationMemory:
    """Gestisce il salvataggio e il caricamento delle conversazioni"""
    
    # P2-9: Simple time-based cache for list_all_conversations
    _conv_list_cache = None
    _conv_list_cache_time = 0
    _CACHE_TTL = 2.0  # seconds
    _cache_lock = threading.Lock()  # Thread-safe cache access
    _MAX_MESSAGES = 500  # P2-3: cap messaggi per conversazione (class-level)

    # ── Cache conversazione attiva in-memory ──────────────────────────
    # Evita 2-3 letture disco ridondanti per ogni messaggio
    _active_conv_cache: Dict[str, Dict] = {}  # conv_id → dati completi
    _active_conv_lock = threading.Lock()
    _ACTIVE_CACHE_MAX = 5  # max conversazioni in cache
    
    def __init__(self):
        """Inizializza il sistema di memoria"""
        self.conversations_dir = config.CONVERSATIONS_DIR
        
    def create_new_conversation(self, title: str = None) -> str:
        """
        Crea una nuova conversazione
        
        Args:
            title: Titolo della conversazione (opzionale)
            
        Returns:
            ID della nuova conversazione
        """
        timestamp = datetime.now()
        conv_id = timestamp.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        
        conversation = {
            'id': conv_id,
            'title': title or f"Conversazione {timestamp.strftime('%d/%m/%Y %H:%M')}",
            'created_at': timestamp.isoformat(),
            'updated_at': timestamp.isoformat(),
            'messages': []
        }
        
        self._save_conversation(conv_id, conversation)
        return conv_id
    
    def add_message(self, conv_id: str, role: str, content: str) -> bool:
        """
        Aggiunge un messaggio alla conversazione
        
        Args:
            conv_id: ID della conversazione
            role: 'user' o 'assistant'
            content: Contenuto del messaggio
            
        Returns:
            True se il salvataggio Ã¨ riuscito
        """
        conversation = self.load_conversation(conv_id)
        if not conversation:
            return False
        
        # P3-11: Single datetime.now() call for consistency
        now_ts = datetime.now().isoformat()
        
        # P3-12: Limit messages per conversation to prevent unbounded growth
        if len(conversation.get('messages', [])) >= self._MAX_MESSAGES:
            # Keep the first message (for title) + last (_MAX_MESSAGES - 1)
            msgs = conversation['messages']
            conversation['messages'] = [msgs[0]] + msgs[-(self._MAX_MESSAGES - 2):]
        
        message = {
            'role': role,
            'content': content,
            'timestamp': now_ts
        }
        
        conversation['messages'].append(message)
        conversation['updated_at'] = now_ts
        
        # Aggiorna il titolo automaticamente dal primo messaggio utente
        if len(conversation['messages']) == 1 and role == 'user':
            # Usa le prime parole come titolo
            title_preview = content[:50].strip()
            if len(content) > 50:
                title_preview += "..."
            conversation['title'] = title_preview
        
        return self._save_conversation(conv_id, conversation)
    
    def load_conversation(self, conv_id: str) -> Optional[Dict]:
        """
        Carica una conversazione dal disco (con cache in-memory).
        
        La cache evita riletture ridondanti: in un singolo scambio
        la conversazione viene caricata 2-3 volte (add_message,
        get_smart_context, update_from_conversation).
        """
        _validate_conv_id(conv_id)

        # Check cache first
        with self._active_conv_lock:
            if conv_id in self._active_conv_cache:
                return self._active_conv_cache[conv_id]

        filepath = os.path.join(self.conversations_dir, f"{conv_id}.json")
        
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Popola cache
            self._cache_conversation(conv_id, data)
            return data
        except Exception as e:
            logger.error("Errore caricamento conversazione %s: %s", conv_id, e)
            return None

    def _cache_conversation(self, conv_id: str, data: Dict) -> None:
        """Aggiunge una conversazione alla cache in-memory (LRU semplice)."""
        with self._active_conv_lock:
            # Evict più vecchie se cache piena
            if conv_id not in self._active_conv_cache and len(self._active_conv_cache) >= self._ACTIVE_CACHE_MAX:
                oldest = next(iter(self._active_conv_cache))
                del self._active_conv_cache[oldest]
            self._active_conv_cache[conv_id] = data

    def _invalidate_conv_cache(self, conv_id: str) -> None:
        """Rimuove una conversazione dalla cache."""
        with self._active_conv_lock:
            self._active_conv_cache.pop(conv_id, None)
    
    def get_conversation_history(self, conv_id: str, limit: int = None) -> List[Dict]:
        """
        Ottiene lo storico dei messaggi per l'AI
        
        Args:
            conv_id: ID della conversazione
            limit: Numero massimo di messaggi da restituire
            
        Returns:
            Lista di messaggi nel formato per l'AI
        """
        conversation = self.load_conversation(conv_id)
        if not conversation:
            return []
        
        messages = conversation.get('messages', [])
        
        if limit:
            messages = messages[-limit:]
        
        # Restituisci nel formato richiesto dall'AI (solo role e content)
        return [
            {'role': msg['role'], 'content': msg['content']}
            for msg in messages
        ]
    
    def list_all_conversations(self) -> List[Dict]:
        """
        Restituisce la lista di tutte le conversazioni
        
        Returns:
            Lista di dizionari con metadata delle conversazioni
        """
        # P2-9: Return cached result if fresh enough
        now = _time.monotonic()
        with ConversationMemory._cache_lock:
            if (ConversationMemory._conv_list_cache is not None
                    and now - ConversationMemory._conv_list_cache_time < self._CACHE_TTL):
                return list(ConversationMemory._conv_list_cache)
        
        conversations = []
        
        try:
            for filename in os.listdir(self.conversations_dir):
                if not filename.endswith('.json'):
                    continue
                filepath = os.path.join(self.conversations_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    conversations.append(self._conv_to_meta(data))
                except Exception:
                    continue
            
            conversations.sort(key=lambda x: x['updated_at'], reverse=True)
            
        except Exception as e:
            logger.error("Errore lista conversazioni: %s", e)
        
        with ConversationMemory._cache_lock:
            ConversationMemory._conv_list_cache = conversations
            ConversationMemory._conv_list_cache_time = now
        return list(conversations)
    
    def delete_conversation(self, conv_id: str) -> bool:
        """
        Elimina una conversazione
        
        Args:
            conv_id: ID della conversazione
            
        Returns:
            True se l'eliminazione Ã¨ riuscita
        """
        _validate_conv_id(conv_id)
        filepath = os.path.join(self.conversations_dir, f"{conv_id}.json")
        
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                self._invalidate_conv_cache(conv_id)
                with ConversationMemory._cache_lock:
                    ConversationMemory._conv_list_cache = None  # Invalidate list cache
                return True
            return False
        except Exception as e:
            logger.error("Errore eliminazione conversazione %s: %s", conv_id, e)
            return False
    
    def update_conversation_title(self, conv_id: str, new_title: str) -> bool:
        """
        Aggiorna il titolo di una conversazione
        
        Args:
            conv_id: ID della conversazione
            new_title: Nuovo titolo
            
        Returns:
            True se l'aggiornamento Ã¨ riuscito
        """
        conversation = self.load_conversation(conv_id)
        if not conversation:
            return False
        
        conversation['title'] = new_title
        conversation['updated_at'] = datetime.now().isoformat()
        
        return self._save_conversation(conv_id, conversation)
    def clear_conversation(self, conv_id: str) -> bool:
        """
        Svuota i messaggi di una conversazione senza eliminarla.

        Mantiene id, titolo e data creazione; resetta messaggi e updated_at.

        Args:
            conv_id: ID della conversazione

        Returns:
            True se l'operazione è riuscita
        """
        conversation = self.load_conversation(conv_id)
        if not conversation:
            return False

        conversation['messages'] = []
        conversation['updated_at'] = datetime.now().isoformat()

        return self._save_conversation(conv_id, conversation)    
    def _save_conversation(self, conv_id: str, conversation: Dict) -> bool:
        """
        Salva una conversazione sul disco
        
        Args:
            conv_id: ID della conversazione
            conversation: Dati della conversazione
            
        Returns:
            True se il salvataggio Ã¨ riuscito
        """
        _validate_conv_id(conv_id)
        filepath = os.path.join(self.conversations_dir, f"{conv_id}.json")
        tmp_path = filepath + ".tmp"
        
        try:
            # Scrivi su file temporaneo poi rinomina (atomico)
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(conversation, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, filepath)
            # Aggiorna cache in-memory
            self._cache_conversation(conv_id, conversation)
            with ConversationMemory._cache_lock:
                ConversationMemory._conv_list_cache = None  # Invalidate list cache
            return True
        except Exception as e:
            logger.error("Errore salvataggio conversazione %s: %s", conv_id, e)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return False
    
    def search_conversations(self, query: str) -> List[Dict]:
        """
        Cerca nelle conversazioni.
        Carica ogni file una sola volta ed esce appena trova un match.
        
        Args:
            query: Testo da cercare
            
        Returns:
            Lista delle conversazioni che contengono il testo
        """
        results = []
        query_lower = query.lower()

        try:
            filenames = sorted(
                (f for f in os.listdir(self.conversations_dir) if f.endswith('.json')),
                reverse=True,
            )
        except OSError as e:
            logger.error("Errore lettura directory conversazioni: %s", e)
            return results

        for filename in filenames:
            conv_id = filename[:-5]
            conv = self.load_conversation(conv_id)
            if not conv:
                continue

            # Cerca nel titolo (veloce)
            if query_lower in conv.get('title', '').lower():
                results.append(self._conv_to_meta(conv))
                continue

            # Cerca nei messaggi (interrompi al primo match)
            for msg in conv.get('messages', []):
                if query_lower in msg.get('content', '').lower():
                    results.append(self._conv_to_meta(conv))
                    break

        return results

    @staticmethod
    def _conv_to_meta(conv: Dict) -> Dict:
        """Estrae i metadati essenziali da una conversazione caricata."""
        return {
            'id': conv['id'],
            'title': conv.get('title', ''),
            'created_at': conv.get('created_at', ''),
            'updated_at': conv.get('updated_at', ''),
            'message_count': len(conv.get('messages', [])),
        }

