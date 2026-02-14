"""
Motore AI - Gestisce l'interazione con Ollama
"""

import logging
import threading
import time
import ollama
from typing import List, Dict, Generator
import config

logger = logging.getLogger(__name__)

# Client Ollama — lazy init (non crea connessione all'import time)
_ollama_lock = threading.Lock()
_ollama_client: ollama.Client | None = None


def _get_ollama_client() -> ollama.Client:
    """Restituisce il client Ollama, creandolo alla prima chiamata."""
    global _ollama_client
    if _ollama_client is None:
        with _ollama_lock:
            if _ollama_client is None:
                _ollama_client = ollama.Client(host=config.OLLAMA_HOST)
    return _ollama_client


# ── Rilevamento ripetizioni nello streaming ──────────────────────────

def _detect_repetition(buffer: str, *, min_phrase: int = 40, max_repeats: int = 2) -> bool:
    """Rileva se il testo in *buffer* contiene un pattern che si ripete.

    Scorre le ultime `min_phrase`..`len(buffer)//2` lunghezze di sotto-stringa
    dalla coda del buffer e verifica se la stessa frase compare più di
    *max_repeats* volte.  Ritorna True appena trova una ripetizione.
    """
    blen = len(buffer)
    if blen < min_phrase * (max_repeats + 1):
        return False
    # Controlla frasi di lunghezza crescente
    for phrase_len in (min_phrase, min_phrase * 2, min(blen // 3, 300)):
        if phrase_len > blen // 2:
            break
        tail = buffer[-phrase_len:]
        count = buffer.count(tail)
        if count > max_repeats:
            return True
    return False


class AIEngine:
    """Gestisce la comunicazione con i modelli AI locali tramite Ollama"""

    # Penalità ripetizione Ollama – valori > 1.0 scoraggiano i token già generati
    REPEAT_PENALTY = 1.3
    REPEAT_LAST_N = 128          # finestra di contesto per la penalità
    VISION_MAX_TOKENS = 1024     # cap dedicato per evitare loop nei modelli vision

    # ── Performance: keep_alive lungo per evitare cold-start (default Ollama = 5m) ──
    KEEP_ALIVE = "30m"           # mantieni modello in VRAM per 30 minuti

    def __init__(self, model: str = None):
        """
        Inizializza il motore AI
        
        Args:
            model: Nome del modello da usare (default da config)
        """
        self.client = _get_ollama_client()
        self._model_lock = threading.Lock()
        self.model = model or config.AI_CONFIG['model']
        self.temperature = config.AI_CONFIG['temperature']
        self.max_tokens = config.AI_CONFIG['max_tokens']
        self.context_window = config.AI_CONFIG['context_window']
        
    def check_ollama_available(self) -> bool:
        """Verifica se Ollama è disponibile e in esecuzione"""
        try:
            self.client.list()
            return True
        except Exception as e:
            logger.debug("Ollama non disponibile: %s", e)
            return False
    
    def check_model_available(self) -> bool:
        """Verifica se il modello configurato è scaricato"""
        try:
            models = self.client.list()
            model_names = [m.model for m in models.models]
            # Controlla se il modello o una sua variante è presente
            return any(self.model in name for name in model_names)
        except Exception as e:
            logger.debug("Errore verifica modello: %s", e)
            return False
    
    def list_available_models(self) -> List[str]:
        """Restituisce la lista dei modelli disponibili"""
        try:
            models = self.client.list()
            return [m.model for m in models.models]
        except Exception as e:
            logger.warning("Impossibile listare modelli Ollama: %s", e)
            return []
    
    def _build_opts(
        self,
        images: List[str] = None,
        *,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ) -> dict:
        """Costruisce le opzioni Ollama usando il profilo per-modello (thread-safe).

        Cerca in config.MODEL_PROFILES il profilo ottimizzato per il modello.
        Se non trovato, usa il profilo '_default'.
        """
        use_model = model or self.model
        profile = config.MODEL_PROFILES.get(
            use_model, config.MODEL_PROFILES.get('_default', {})
        )

        temp = temperature if temperature is not None else profile.get('temperature', self.temperature)
        tokens = max_tokens if max_tokens is not None else profile.get('num_predict', self.max_tokens)

        opts = {
            'temperature': temp,
            'num_predict': tokens,
            'num_ctx': profile.get('num_ctx', self.context_window),
            'repeat_penalty': profile.get('repeat_penalty', self.REPEAT_PENALTY),
            'repeat_last_n': profile.get('repeat_last_n', self.REPEAT_LAST_N),
            'num_thread': profile.get('num_thread', 6),
            'num_batch': profile.get('num_batch', 512),
            'num_gpu': profile.get('num_gpu', -1),
            'top_k': profile.get('top_k', 40),
            'top_p': profile.get('top_p', 0.9),
        }
        if images:
            opts['num_predict'] = min(opts['num_predict'], self.VISION_MAX_TOKENS)
        return opts

    @staticmethod
    def _build_messages(
        prompt: str,
        conversation_history: List[Dict] = None,
        system_prompt: str = None,
        images: List[str] = None,
        *,
        model: str = None,
    ) -> List[Dict]:
        """Costruisce la lista messaggi per Ollama (DRY, usato da sync e stream).

        Se il modello è in BAKED_PROMPT_MODELS, il system prompt viene omesso
        perché già integrato nel Modelfile (risparmio ~300 token/richiesta).
        """
        messages: List[Dict] = []
        # Skip system prompt per modelli con prompt integrato nel Modelfile
        if not (model and model in config.BAKED_PROMPT_MODELS):
            messages.append({
                'role': 'system',
                'content': system_prompt or config.SYSTEM_PROMPT,
            })
        if conversation_history:
            messages.extend(conversation_history)
        user_msg: Dict = {'role': 'user', 'content': prompt}
        if images:
            user_msg['images'] = images
        messages.append(user_msg)
        return messages

    def generate_response(
        self, 
        prompt: str, 
        conversation_history: List[Dict] = None,
        system_prompt: str = None,
        images: List[str] = None,
        *,
        model: str = None,
        temperature: float = None,
    ) -> str:
        """
        Genera una risposta dal modello AI
        
        Args:
            prompt: Il messaggio dell'utente
            conversation_history: Storico della conversazione
            system_prompt: Prompt di sistema personalizzato
            images: Lista di immagini codificate in base64 (per modelli vision)
            model: Override modello per questa richiesta (thread-safe)
            temperature: Override temperatura per questa richiesta (thread-safe)
            
        Returns:
            La risposta del modello
        """
        try:
            use_model = model or self.model
            messages = self._build_messages(
                prompt, conversation_history, system_prompt, images,
                model=use_model,
            )

            response = self.client.chat(
                model=use_model,
                messages=messages,
                options=self._build_opts(images, model=use_model, temperature=temperature),
                keep_alive=self.KEEP_ALIVE,
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
        images: List[str] = None,
        *,
        model: str = None,
        temperature: float = None,
    ) -> Generator[str, None, None]:
        """
        Genera una risposta in streaming (per visualizzare parola per parola)
        
        Args:
            prompt: Il messaggio dell'utente
            conversation_history: Storico della conversazione
            system_prompt: Prompt di sistema personalizzato
            images: Lista di immagini codificate in base64 (per modelli vision)
            model: Override modello per questa richiesta (thread-safe)
            temperature: Override temperatura per questa richiesta (thread-safe)
            
        Yields:
            Parti della risposta man mano che vengono generate
        """
        try:
            use_model = model or self.model
            messages = self._build_messages(
                prompt, conversation_history, system_prompt, images,
                model=use_model,
            )

            stream = self.client.chat(
                model=use_model,
                messages=messages,
                stream=True,
                options=self._build_opts(images, model=use_model, temperature=temperature),
                keep_alive=self.KEEP_ALIVE,
            )

            # Rilevamento ripetizioni in tempo reale (controlla ogni N token)
            running_text = ""
            _check_interval = 20   # controlla ogni N token, non ad ogni singolo
            _token_count = 0
            _stream_start = time.perf_counter()
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    token = chunk['message']['content']
                    running_text += token
                    _token_count += 1

                    # Guard anti-loop: controlla periodicamente
                    if (_token_count % _check_interval == 0
                            and len(running_text) > 200
                            and _detect_repetition(running_text)):
                        logger.warning(
                            "Ripetizione rilevata dopo %d chars, generazione interrotta",
                            len(running_text),
                        )
                        yield "\n\n⚠️ *Risposta troncata: il modello ha iniziato a ripetere.*"
                        return

                    yield token

            # Performance logging: tokens/sec
            _elapsed = time.perf_counter() - _stream_start
            if _elapsed > 0 and _token_count > 0:
                _tps = _token_count / _elapsed
                logger.info(
                    "[perf] %s: %d tok in %.1fs (%.1f tok/s)",
                    use_model, _token_count, _elapsed, _tps,
                )
                    
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
        truncated = len(document_text) > max_c
        doc_slice = document_text[:max_c]
        trunc_note = (
            f"\n[Nota: documento troncato a {max_c} caratteri su {len(document_text)} totali]"
            if truncated else ""
        )

        if question:
            prompt = f"""Ho questo documento:

---
{doc_slice}{trunc_note}
---

Domanda: {question}

Rispondi basandoti sul contenuto del documento."""
        else:
            prompt = f"""Analizza questo documento e fornisci un riassunto dettagliato:

---
{doc_slice}{trunc_note}
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
        with self._model_lock:
            old_model = self.model
            self.model = new_model
            if self.check_model_available():
                return True
            self.model = old_model
            return False
