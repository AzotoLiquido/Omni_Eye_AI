"""
Sistema di Memoria - Gestisce lo storico delle conversazioni
"""

import json
import logging
import os
import re
import uuid
from datetime import datetime
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


# File locking cross-platform
try:
    import msvcrt

    def _lock_file(f):
        """Blocca l'intero file (Windows)"""
        f.seek(0, 2)          # vai alla fine
        size = f.tell() or 1  # almeno 1 byte
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, size)

    def _unlock_file(f):
        f.seek(0, 2)
        size = f.tell() or 1
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, size)

except ImportError:
    try:
        import fcntl

        def _lock_file(f):
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        def _unlock_file(f):
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except ImportError:
        # Fallback: nessun locking (piattaforma sconosciuta)
        def _lock_file(f):
            pass

        def _unlock_file(f):
            pass


class ConversationMemory:
    """Gestisce il salvataggio e il caricamento delle conversazioni"""
    
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
        
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        
        conversation['messages'].append(message)
        conversation['updated_at'] = datetime.now().isoformat()
        
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
        Carica una conversazione dal disco
        
        Args:
            conv_id: ID della conversazione
            
        Returns:
            Dizionario con i dati della conversazione o None
        """
        _validate_conv_id(conv_id)
        filepath = os.path.join(self.conversations_dir, f"{conv_id}.json")
        
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error("Errore caricamento conversazione %s: %s", conv_id, e)
            return None
    
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
            
            # Ordina per data di aggiornamento (piÃ¹ recenti prima)
            conversations.sort(key=lambda x: x['updated_at'], reverse=True)
            
        except Exception as e:
            logger.error("Errore lista conversazioni: %s", e)
        
        return conversations
    
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

