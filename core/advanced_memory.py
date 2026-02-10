"""
Sistema di Memoria Conversazionale Avanzata
Gestisce contesto intelligente, summarization, entitÃ  e knowledge base
"""

import json
import os
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
        """Stima approssimativa dei token (1 token â‰ˆ 4 caratteri)"""
        return len(text) // 4
    
    def calculate_messages_tokens(self, messages: List[Dict]) -> int:
        """Calcola il totale dei token nei messaggi"""
        total = 0
        for msg in messages:
            total += self.estimate_tokens(msg.get('content', ''))
        return total
    
    def compress_context(self, messages: List[Dict], ai_engine) -> List[Dict]:
        """
        Comprime i messaggi vecchi mantenendo quelli piÃ¹ recenti
        
        Args:
            messages: Lista completa dei messaggi
            ai_engine: Engine AI per generare riassunti
            
        Returns:
            Lista di messaggi compressa con riassunto iniziale
        """
        total_tokens = self.calculate_messages_tokens(messages)
        
        # Se sotto il limite, restituisci tutto
        if total_tokens <= self.max_context_tokens:
            return messages
        
        # Mantieni sempre gli ultimi N messaggi
        recent_messages_count = 6  # Ultimi 3 scambi
        recent_messages = messages[-recent_messages_count:]
        old_messages = messages[:-recent_messages_count]
        
        # Crea un riassunto dei messaggi vecchi
        if old_messages:
            summary = self._generate_summary(old_messages, ai_engine)
            
            # Sostituisci i vecchi messaggi con il riassunto
            compressed = [
                {
                    'role': 'system',
                    'content': f"[RIASSUNTO CONVERSAZIONE PRECEDENTE]\n{summary}",
                    'timestamp': datetime.now().isoformat(),
                    'is_summary': True
                }
            ]
            compressed.extend(recent_messages)
            return compressed
        
        return recent_messages
    
    def _generate_summary(self, messages: List[Dict], ai_engine) -> str:
        """Genera un riassunto dei messaggi usando l'AI"""
        
        # Prepara il testo da riassumere
        conversation_text = "\n\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in messages
        ])
        
        summary_prompt = f"""Riassumi questa conversazione in modo conciso ma completo, 
mantenendo tutti i fatti importanti, nomi, date e informazioni chiave.
Scrivi solo il riassunto, senza introduzioni.

CONVERSAZIONE:
{conversation_text}

RIASSUNTO:"""
        
        try:
            # Genera il riassunto
            summary = ai_engine.generate_response(
                prompt=summary_prompt,
                conversation_history=None,
                system_prompt="Sei un esperto nel creare riassunti concisi e informativi."
            )
            return summary
        except Exception as e:
            # Fallback: riassunto basico
            return f"Conversazione con {len(messages)} messaggi scambiati su vari argomenti."


class EntityTracker:
    """Traccia entitÃ  menzionate nelle conversazioni (persone, luoghi, fatti)"""
    
    def __init__(self, storage_path: str):
        """
        Args:
            storage_path: Percorso al file JSON per salvare le entitÃ 
        """
        self.storage_path = storage_path
        self._lock = threading.Lock()
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
        
        self._save_entities()
    
    def _extract_names(self, text: str, timestamp: str) -> None:
        """Estrae nomi propri dal testo con pattern espliciti"""
        # Pattern espliciti per nomi (piÃ¹ precisi del semplice "parola maiuscola")
        name_patterns = [
            r'(?:mi chiamo|si chiama|chiamato|chiamata|conosco)\s+([A-Z][a-zÃ -Ãº]+)',
            r'(?:parlo con|parlato con|visto|incontrato)\s+([A-Z][a-zÃ -Ãº]+)',
        ]
        
        # Parole comuni da escludere (false positive frequenti)
        STOP_NAMES = {
            'sono', 'come', 'cosa', 'tutto', 'questo', 'quella', 'quello',
            'ancora', 'sempre', 'grazie', 'ciao', 'buongiorno', 'buonasera',
            'molto', 'poco', 'bene', 'male', 'dopo', 'prima', 'oggi',
            'domani', 'ieri', 'perchÃ©', 'quando', 'dove', 'quindi',
        }
        
        for pattern in name_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for name in matches:
                name = name.strip('.,!?;:')
                if len(name) <= 2 or name.lower() in STOP_NAMES:
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
                relevant.append(f"â€¢ {name}: menzionato precedentemente")
        
        # Cerca preferenze rilevanti per la query (non tutte)
        for key, pref in self.entities['preferences'].items():
            if pref['value'].lower() in text_lower or pref['sentiment'].lower() in text_lower:
                relevant.append(f"• Preferenza: {pref['sentiment']} {pref['value']}")
                if len(relevant) >= 8:
                    break
        
        if relevant:
            return "[INFORMAZIONI CONTESTUALI]\n" + "\n".join(relevant) + "\n"
        
        return ""


class KnowledgeBase:
    """Memoria a lungo termine per informazioni cross-conversazione"""
    
    def __init__(self, storage_path: str):
        """
        Args:
            storage_path: Percorso al file JSON per la knowledge base
        """
        self.storage_path = storage_path
        self._lock = threading.Lock()
        self.knowledge = self._load_knowledge()
    
    def _load_knowledge(self) -> Dict:
        """Carica la knowledge base dal disco"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Errore caricamento knowledge base: %s", e)
        
        return {
            'user_profile': {
                'name': None,
                'interests': [],
                'expertise': [],
                'language': 'italiano',
                'created_at': datetime.now().isoformat()
            },
            'learned_facts': [],  # Lista di fatti appresi
            'topics_discussed': {},  # topic -> conteggio
            'last_updated': datetime.now().isoformat()
        }
    
    def _save_knowledge(self) -> bool:
        """Salva la knowledge base sul disco (atomico con temp+replace)"""
        tmp_path = self.storage_path + ".tmp"
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            self.knowledge['last_updated'] = datetime.now().isoformat()
            with self._lock:
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    json.dump(self.knowledge, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, self.storage_path)
            return True
        except Exception as e:
            logger.error("Errore salvataggio knowledge base: %s", e)
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            return False
    
    def update_from_conversation(self, messages: List[Dict]) -> None:
        """Aggiorna la knowledge base dai messaggi della conversazione"""
        
        for msg in messages:
            if msg['role'] == 'user':
                content_lower = msg['content'].lower()
                
                # Estrai il nome dell'utente
                if 'mi chiamo' in content_lower or 'sono' in content_lower:
                    self._extract_user_name(msg['content'])
                
                # Estrai interessi
                if any(word in content_lower for word in ['interessa', 'piace', 'passione']):
                    self._extract_interests(msg['content'])
                
                # Conta i topic
                self._count_topics(msg['content'])
        
        self._save_knowledge()
    
    def _extract_user_name(self, text: str) -> None:
        """Estrae il nome dell'utente con pattern precisi"""
        # "sono X" rimosso: troppo generico (es. "sono stanco" â†’ falso positivo)
        patterns = [
            r'mi chiamo (\w+)',
            r'il mio nome Ã¨ (\w+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                if len(name) > 2 and name[0].isupper():
                    self.knowledge['user_profile']['name'] = name
                    return
    
    def _extract_interests(self, text: str) -> None:
        """Estrae interessi dell'utente"""
        text_lower = text.lower()
        
        # Pattern per interessi
        if 'mi interessa' in text_lower:
            after = text_lower.split('mi interessa')[1].split('.')[0]
            interest = after.strip()
            if interest and interest not in self.knowledge['user_profile']['interests']:
                self.knowledge['user_profile']['interests'].append(interest)
    
    def _count_topics(self, text: str) -> None:
        """Conta i topic discussi"""
        # Topic keywords comuni
        topics = {
            'programmazione': ['python', 'javascript', 'codice', 'programma', 'sviluppo'],
            'scienza': ['fisica', 'chimica', 'biologia', 'scienza'],
            'tecnologia': ['ai', 'intelligenza artificiale', 'computer', 'tecnologia'],
            'arte': ['arte', 'pittura', 'musica', 'letteratura'],
            'sport': ['calcio', 'basket', 'sport', 'tennis'],
        }
        
        text_lower = text.lower()
        for topic, keywords in topics.items():
            if any(keyword in text_lower for keyword in keywords):
                self.knowledge['topics_discussed'][topic] = \
                    self.knowledge['topics_discussed'].get(topic, 0) + 1
    
    def get_user_context(self) -> str:
        """Genera un contesto sull'utente dalla knowledge base"""
        profile = self.knowledge.get('user_profile', {})
        context_parts = []
        
        if profile.get('name'):
            context_parts.append(f"Nome utente: {profile['name']}")
        
        if profile.get('interests'):
            context_parts.append(f"Interessi: {', '.join(profile['interests'][:3])}")
        
        # Top 3 topic discussi
        topics = self.knowledge.get('topics_discussed', {})
        if topics:
            top_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:3]
            topics_str = ", ".join([f"{t[0]} ({t[1]}x)" for t in top_topics])
            context_parts.append(f"Topic frequenti: {topics_str}")
        
        if context_parts:
            return "[PROFILO UTENTE]\n" + "\n".join(context_parts) + "\n"
        
        return ""


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
        Cerca informazioni nella knowledge base
        
        Args:
            query: Query di ricerca
            
        Returns:
            Dizionario con risultati rilevanti
        """
        results = {
            'user_profile': {},
            'relevant_topics': [],
            'related_facts': []
        }
        
        query_lower = query.lower()
        
        # Cerca nel profilo utente
        profile = self.knowledge_base.knowledge.get('user_profile', {})
        if profile.get('name') and profile['name'].lower() in query_lower:
            results['user_profile'] = profile
        
        # Cerca nei topic
        topics = self.knowledge_base.knowledge.get('topics_discussed', {})
        for topic, count in topics.items():
            if topic.lower() in query_lower or query_lower in topic.lower():
                results['relevant_topics'].append({
                    'topic': topic,
                    'mentions': count
                })
        
        return results
    
    def export_knowledge_summary(self) -> str:
        """Esporta un riassunto testuale della knowledge base"""
        kb = self.knowledge_base.knowledge
        
        lines = ["=== KNOWLEDGE BASE SUMMARY ===\n"]
        
        # User Profile
        profile = kb.get('user_profile', {})
        if profile.get('name'):
            lines.append(f"ðŸ‘¤ Utente: {profile['name']}")
        
        if profile.get('interests'):
            lines.append(f"ðŸ’¡ Interessi: {', '.join(profile['interests'])}")
        
        # Topics
        topics = kb.get('topics_discussed', {})
        if topics:
            lines.append("\nðŸ“Š Topic discussi:")
            sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)
            for topic, count in sorted_topics[:5]:
                lines.append(f"   â€¢ {topic}: {count} volte")
        
        # Entities
        if hasattr(self, 'entity_tracker'):
            people = self.entity_tracker.entities.get('people', {})
            if people:
                lines.append(f"\nðŸ‘¥ Persone menzionate: {', '.join(list(people.keys())[:5])}")
        
        lines.append(f"\nâ° Ultimo aggiornamento: {kb.get('last_updated', 'N/A')}")
        
        return "\n".join(lines)

