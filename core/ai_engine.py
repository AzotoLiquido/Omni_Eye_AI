"""
Motore AI - Gestisce l'interazione con Ollama
"""

import logging
import ollama
from typing import List, Dict, Generator
import config

logger = logging.getLogger(__name__)


class AIEngine:
    """Gestisce la comunicazione con i modelli AI locali tramite Ollama"""
    
    def __init__(self, model: str = None):
        """
        Inizializza il motore AI
        
        Args:
            model: Nome del modello da usare (default da config)
        """
        self.model = model or config.AI_CONFIG['model']
        self.temperature = config.AI_CONFIG['temperature']
        self.max_tokens = config.AI_CONFIG['max_tokens']
        
    def check_ollama_available(self) -> bool:
        """Verifica se Ollama è disponibile e in esecuzione"""
        try:
            ollama.list()
            return True
        except Exception as e:
            logger.debug("Ollama non disponibile: %s", e)
            return False
    
    def check_model_available(self) -> bool:
        """Verifica se il modello configurato è scaricato"""
        try:
            models = ollama.list()
            model_names = [m.model for m in models.models]
            # Controlla se il modello o una sua variante è presente
            return any(self.model in name for name in model_names)
        except Exception as e:
            logger.debug("Errore verifica modello: %s", e)
            return False
    
    def list_available_models(self) -> List[str]:
        """Restituisce la lista dei modelli disponibili"""
        try:
            models = ollama.list()
            return [m.model for m in models.models]
        except Exception as e:
            logger.warning("Impossibile listare modelli Ollama: %s", e)
            return []
    
    def generate_response(
        self, 
        prompt: str, 
        conversation_history: List[Dict] = None,
        system_prompt: str = None,
        images: List[str] = None
    ) -> str:
        """
        Genera una risposta dal modello AI
        
        Args:
            prompt: Il messaggio dell'utente
            conversation_history: Storico della conversazione
            system_prompt: Prompt di sistema personalizzato
            images: Lista di immagini codificate in base64 (per modelli vision)
            
        Returns:
            La risposta del modello
        """
        try:
            # Prepara i messaggi
            messages = []
            
            # Aggiungi system prompt
            if system_prompt:
                messages.append({
                    'role': 'system',
                    'content': system_prompt
                })
            else:
                messages.append({
                    'role': 'system',
                    'content': config.SYSTEM_PROMPT
                })
            
            # Aggiungi storico conversazione
            if conversation_history:
                messages.extend(conversation_history)
            
            # Aggiungi il prompt corrente (con immagini se presenti)
            user_msg = {'role': 'user', 'content': prompt}
            if images:
                user_msg['images'] = images
            messages.append(user_msg)
            
            # Genera risposta
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    'temperature': self.temperature,
                    'num_predict': self.max_tokens,
                }
            )
            
            return response['message']['content']
            
        except Exception as e:
            logger.error("Errore generazione risposta: %s", e)
            raise RuntimeError(f"Errore nella generazione della risposta: {e}") from e
    
    def generate_response_stream(
        self, 
        prompt: str, 
        conversation_history: List[Dict] = None,
        system_prompt: str = None,
        images: List[str] = None
    ) -> Generator[str, None, None]:
        """
        Genera una risposta in streaming (per visualizzare parola per parola)
        
        Args:
            prompt: Il messaggio dell'utente
            conversation_history: Storico della conversazione
            system_prompt: Prompt di sistema personalizzato
            images: Lista di immagini codificate in base64 (per modelli vision)
            
        Yields:
            Parti della risposta man mano che vengono generate
        """
        try:
            # Prepara i messaggi
            messages = []
            
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            else:
                messages.append({'role': 'system', 'content': config.SYSTEM_PROMPT})
            
            if conversation_history:
                messages.extend(conversation_history)
            
            # Aggiungi messaggio utente (con immagini se presenti)
            user_msg = {'role': 'user', 'content': prompt}
            if images:
                user_msg['images'] = images
            messages.append(user_msg)
            
            # Genera risposta in streaming
            stream = ollama.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options={
                    'temperature': self.temperature,
                    'num_predict': self.max_tokens,
                }
            )
            
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
                    
        except Exception as e:
            logger.error("Errore streaming risposta: %s", e)
            yield "\n\n❌ Si è verificato un errore nella generazione della risposta."
    
    # Massimo di caratteri inviati al modello per analisi documento
    ANALYZE_DOC_MAX_CHARS = 3000

    def analyze_document(self, document_text: str, question: str = None) -> str:
        """
        Analizza un documento e risponde a domande su di esso
        
        Args:
            document_text: Il testo del documento
            question: Domanda specifica (opzionale)
            
        Returns:
            Analisi o risposta
        """
        max_c = self.ANALYZE_DOC_MAX_CHARS

        if question:
            prompt = f"""Ho questo documento:

---
{document_text[:max_c]}  
---

Domanda: {question}

Rispondi basandoti sul contenuto del documento."""
        else:
            prompt = f"""Analizza questo documento e fornisci un riassunto dettagliato:

---
{document_text[:max_c]}
---

Fornisci:
1. Riassunto principale
2. Punti chiave
3. Eventuali osservazioni importanti"""
        
        return self.generate_response(prompt)
    
    def change_model(self, new_model: str) -> bool:
        """
        Cambia il modello AI utilizzato
        
        Args:
            new_model: Nome del nuovo modello
            
        Returns:
            True se il cambio è avvenuto con successo
        """
        old_model = self.model
        self.model = new_model
        
        if self.check_model_available():
            return True
        else:
            self.model = old_model
            return False
