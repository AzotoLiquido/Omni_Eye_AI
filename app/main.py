"""
Omni Eye AI - Backend Flask
"""

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf import CSRFProtect
import base64 as _b64
import logging
import os
import sys
import threading
import re as _re

logger = logging.getLogger(__name__)

# Aggiungi la directory root al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import AIEngine, DocumentProcessor
from core.advanced_memory import AdvancedMemory
from core.ai_pilot import Pilot
from app.vision import (
    VISION_ONLY_TAGS, MULTILINGUAL_VISION, VISION_PRIORITY,
    VISION_SYSTEM_PROMPT, user_visible_models, vision_model_priority,
    build_vision_prompt,
)
import config

# Regex per validazione ID conversazione (anti path-traversal)
_SAFE_CONV_ID = _re.compile(r'^[a-zA-Z0-9_-]+$')

# Lock per cambio modello thread-safe (P1-7)
_model_lock = threading.Lock()


def _sse_data(text: str) -> str:
    """Format text as SSE data, handling embedded newlines (P1-6)."""
    return "".join(f"data: {line}\n" for line in text.split('\n')) + "\n"

app = Flask(__name__)
app.secret_key = config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max (immagini base64 sono ~33% più grandi)

# Configurazione CORS sicura - Solo localhost
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:5000", "http://127.0.0.1:5000"],
        "methods": ["GET", "POST", "DELETE"],
        "allow_headers": ["Content-Type"]
    }
})

# Protezione CSRF DISABILITATA per API
# Gli endpoint API sono già protetti da CORS + Rate Limiting
# Il CSRF è rilevante solo per form HTML tradizionali
if getattr(config, 'CSRF_ENABLED', False):  # Cambiato default a False
    csrf = CSRFProtect(app)
    logger.info("CSRF Protection: ATTIVA")
else:
    logger.info("CSRF Protection: DISATTIVATA (API-only app)")

# Configurazione Rate Limiting (PERMISSIVO - non ostacola lavoro normale)
# Protegge solo da abusi estremi (loop infiniti, script massivi)
limiter = Limiter(
    get_remote_address,
    app=app,
    storage_uri=config.RATE_LIMIT_CONFIG['storage_uri'],
    default_limits=[config.RATE_LIMIT_CONFIG['limits']['default']],
    enabled=config.RATE_LIMIT_CONFIG['enabled']  # Disabilita con env var se serve
)

# ============================================================================
# INIZIALIZZAZIONE COMPONENTI
# ============================================================================

# Inizializza i componenti CON MEMORIA AVANZATA
ai_engine = AIEngine()
memory = AdvancedMemory()  # Sistema avanzato con context management
doc_processor = DocumentProcessor()

# Inizializza AI-Pilot (orchestratore avanzato)
try:
    pilot = Pilot()
    PILOT_ENABLED = True
    logger.info("AI-Pilot: ATTIVO (v%s, planner=%s)", pilot.cfg.version, pilot.cfg.planner_strategy)
except Exception as e:
    pilot = None
    PILOT_ENABLED = False
    logger.warning("AI-Pilot: DISATTIVATO (%s)", e)

# Note: Rimossa variabile globale current_conversation_id per evitare race conditions.
# L'ID conversazione viene ora gestito tramite richieste client o session Flask.

# Debug logging per tutte le richieste (solo in modalità debug)
@app.before_request
def log_request():
    if request.method == 'POST' and app.debug:
        logger.debug("POST Request: %s (Content-Type: %s)", request.path, request.content_type)


@app.route('/')
def index():
    """Pagina principale"""
    return render_template('index.html')


@app.route('/knowledge')
def knowledge_page():
    """Pagina Knowledge Base"""
    return render_template('knowledge.html')


@app.route('/api/status', methods=['GET'])
def api_status():
    """Verifica lo stato del sistema"""
    ollama_ok = ai_engine.check_ollama_available()
    model_ok = ai_engine.check_model_available() if ollama_ok else False
    
    return jsonify({
        'ollama_available': ollama_ok,
        'model_available': model_ok,
        'model_name': ai_engine.model,
        'available_models': user_visible_models(ai_engine) if ollama_ok else []
    })


@app.route('/api/chat', methods=['POST'])
@limiter.limit(config.RATE_LIMIT_CONFIG['limits']['chat'])
def api_chat():
    """Endpoint per chat normale (non streaming)
    
    Gestisce una singola richiesta chat e restituisce la risposta completa.
    Utilizza il sistema di memoria avanzata per contesto e learning.
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Nessun dato ricevuto'}), 400
    user_message = data.get('message', '').strip()
    conv_id = data.get('conversation_id')
    
    if not user_message:
        return jsonify({'error': 'Messaggio vuoto'}), 400
    
    # Validazione conv_id
    if conv_id and not _SAFE_CONV_ID.match(conv_id):
        return jsonify({'error': 'ID conversazione non valido'}), 400
    
    # Crea nuova conversazione se necessario
    if not conv_id:
        conv_id = memory.create_new_conversation()
    
    # Salva messaggio utente con estrazione entità
    memory.add_message_advanced(conv_id, 'user', user_message)
    
    # Ottieni contesto ottimizzato con compressione automatica
    history, additional_context = memory.get_smart_context(conv_id, ai_engine)
    
    # Rimuovi l'ultimo messaggio dalla history
    if history:
        history = history[:-1]
    
    # System prompt: Pilot (se attivo) oppure fallback config.py
    if PILOT_ENABLED and pilot:
        system_prompt = pilot.build_system_prompt(
            user_message=user_message,
            extra_instructions=additional_context,
        )
    else:
        system_prompt = config.SYSTEM_PROMPT
        if additional_context:
            system_prompt = additional_context + "\n" + system_prompt
    
    # Genera risposta
    try:
        if PILOT_ENABLED and pilot:
            response, meta = pilot.process(
                user_message,
                conversation_history=history,
                ai_engine=ai_engine,
                conv_id=conv_id,
            )
        else:
            response = ai_engine.generate_response(
                user_message, 
                conversation_history=history,
                system_prompt=system_prompt
            )
            meta = {}
        
        # Salva risposta (senza estrazione entità per l'assistant)
        memory.add_message_advanced(conv_id, 'assistant', response, extract_entities=False)
        
        return jsonify({
            'response': response,
            'conversation_id': conv_id,
            'success': True,
            'pilot_meta': meta if meta else None
        })
    
    except Exception as e:
        logger.error("Errore api_chat: %s", e, exc_info=True)
        return jsonify({
            'error': 'Errore interno del server',
            'success': False
        }), 500


@app.route('/api/chat/stream', methods=['POST'])
@limiter.limit(config.RATE_LIMIT_CONFIG['limits']['chat_stream'])
def api_chat_stream():
    """Endpoint per chat con streaming (risposta parola per parola)
    
    Utilizza Server-Sent Events (SSE) per inviare la risposta in chunks real-time.
    Più responsive per l'utente rispetto all'endpoint chat standard.
    """
    try:
        data = request.json
        if not data:
            logger.warning("Chat stream: nessun JSON ricevuto")
            return jsonify({'error': 'Nessun dato ricevuto'}), 400
            
        user_message = data.get('message', '').strip()
        conv_id = data.get('conversation_id')
        raw_images = data.get('images')  # Lista di stringhe base64 (per modelli vision)

        # Validazione conv_id
        if conv_id and not _SAFE_CONV_ID.match(conv_id):
            return jsonify({'error': 'ID conversazione non valido'}), 400

        # ── Validazione immagini (P0-2) ──
        images = None
        if raw_images:
            if not isinstance(raw_images, list):
                return jsonify({'error': 'images deve essere una lista'}), 400
            images = []
            for i, img in enumerate(raw_images[:5]):  # max 5 immagini
                if not isinstance(img, str):
                    return jsonify({'error': f'images[{i}]: deve essere una stringa base64'}), 400
                # Limita dimensione singola immagine (10MB decodificati ~= 13.3MB base64)
                if len(img) > 14_000_000:
                    return jsonify({'error': f'images[{i}]: troppo grande (max ~10MB)'}), 400
                # Controlla che sia base64 valido (campiona i primi bytes)
                try:
                    _sample = img[:1024]
                    _sample += '=' * (-len(_sample) % 4)
                    _b64.b64decode(_sample, validate=True)
                except Exception:
                    return jsonify({'error': f'images[{i}]: base64 non valido'}), 400
                images.append(img)

        logger.debug("Richiesta chat: message='%s...', conv_id=%s, images=%d",
                      user_message[:50] if user_message else '', conv_id, len(images) if images else 0)
        
        if not user_message and not images:
            logger.warning("Chat stream: messaggio vuoto")
            return jsonify({'error': 'Messaggio vuoto'}), 400
    except Exception as e:
        logger.error("Errore parsing richiesta chat stream: %s", e)
        return jsonify({'error': f'Errore parsing richiesta: {str(e)}'}), 400
    
    # Crea nuova conversazione se necessario
    if not conv_id:
        conv_id = memory.create_new_conversation()
    
    # Salva messaggio utente con estrazione entità
    memory.add_message_advanced(conv_id, 'user', user_message)
    
    # Ottieni contesto ottimizzato
    history, additional_context = memory.get_smart_context(conv_id, ai_engine)
    if history:
        history = history[:-1]
    
    # Estrai descrizioni immagini salvate nella conversazione (per follow-up)
    # P2-2: tieni solo le ultime 3 analisi per non esplodere il contesto
    image_context_parts = []
    clean_history = []
    for msg in history:
        if msg.get('role') == 'system' and '[ANALISI IMMAGINE]' in msg.get('content', ''):
            image_context_parts.append(msg['content'])
        else:
            clean_history.append(msg)
    image_context_parts = image_context_parts[-3:]  # cap a 3 analisi più recenti
    image_memory = "\n".join(image_context_parts) if image_context_parts else ""
    
    # System prompt: Pilot (se attivo) oppure fallback
    if PILOT_ENABLED and pilot:
        extra = additional_context
        if image_memory:
            extra = image_memory + "\n" + (extra or "")
        system_prompt = pilot.build_system_prompt(
            user_message=user_message,
            extra_instructions=extra,
        )
    else:
        system_prompt = config.SYSTEM_PROMPT
        if image_memory:
            system_prompt = image_memory + "\n" + system_prompt
        if additional_context:
            system_prompt = additional_context + "\n" + system_prompt
    
    def generate():
        """Generator per lo streaming"""
        full_response = ""
        response_saved = False
        
        try:
            # Auto-switch a modello vision se ci sono immagini
            # P0-1: NON mutiamo più ai_engine.model/.temperature
            #       usiamo variabili locali + parametri model=/temperature= nei metodi
            if images:
                all_models = ai_engine.list_available_models()
                vision_models = [m for m in all_models
                                 if any(v in m.lower() for v in VISION_PRIORITY)]
                # Ordina per priorità definita in VISION_PRIORITY
                vision_models.sort(key=vision_model_priority)

                current = ai_engine.model.lower()
                is_vision = any(v in current for v in VISION_PRIORITY)

                if not is_vision and vision_models:
                    vision_model = vision_models[0]
                    logger.info("Auto-switch a modello vision: %s", vision_model)
                elif not is_vision and not vision_models:
                    yield f"data: ⚠️ Nessun modello vision installato. Scaricane uno con: ollama pull minicpm-v\n\n"
                    yield f"event: end\ndata: {conv_id}\n\n"
                    return
                else:
                    vision_model = ai_engine.model

                text_model = config.AI_CONFIG['model']
                is_multilingual = any(tag in vision_model.lower()
                                      for tag in MULTILINGUAL_VISION)

                # --- Prompt adattivo in base alla domanda utente ---
                vision_prompt = build_vision_prompt(user_message, is_multilingual)

                if is_multilingual:
                    # ═══ PIPELINE SINGOLA — modello multilingue ═══
                    logger.info("Vision pipeline SINGOLA (multilingual): %s", vision_model)

                    direct_response = ""
                    for chunk in ai_engine.generate_response_stream(
                        vision_prompt,
                        conversation_history=clean_history[-4:],
                        system_prompt=VISION_SYSTEM_PROMPT,
                        images=images,
                        model=vision_model,
                        temperature=0.2,
                    ):
                        direct_response += chunk
                        full_response += chunk
                        yield _sse_data(chunk)

                    # Salva descrizione per follow-up
                    img_context = (
                        f"[ANALISI IMMAGINE] L'utente ha inviato un'immagine. "
                        f"Risposta dell'analisi visiva: {direct_response.strip()}"
                    )
                    memory.add_message_advanced(
                        conv_id, 'system', img_context, extract_entities=False
                    )

                else:
                    # ═══ PIPELINE 2 FASI — modello EN-only ═══
                    logger.info("Vision pipeline 2 FASI (EN-only): %s -> %s",
                                vision_model, text_model)

                    # P2-3: feedback utente durante fase 1
                    yield "data: 🔍 *Analisi immagine in corso...*\n\n"

                    image_description = ""
                    for chunk in ai_engine.generate_response_stream(
                        vision_prompt,
                        conversation_history=None,
                        # P1-3: system prompt anche per EN-only
                        system_prompt=VISION_SYSTEM_PROMPT,
                        images=images,
                        model=vision_model,
                        temperature=0.15,
                    ):
                        image_description += chunk

                    logger.info("Vision fase 1 completata: %d chars", len(image_description))

                    # Sostituisci il messaggio di progress
                    yield "data: \n\n"

                    # Salva la descrizione nella conversazione per follow-up
                    img_context = (
                        f"[ANALISI IMMAGINE] L'utente ha inviato un'immagine. "
                        f"Questa è la descrizione oggettiva di ciò che contiene: "
                        f"{image_description.strip()}"
                    )
                    memory.add_message_advanced(
                        conv_id, 'system', img_context, extract_entities=False
                    )

                    answer_prompt = (
                        f"# Contesto\n"
                        f"L'utente ha inviato un'immagine. Un modello visivo l'ha analizzata "
                        f"e ha prodotto questa descrizione:\n\n"
                        f"---\n{image_description.strip()}\n---\n\n"
                        f"# Domanda dell'utente\n\"{user_message}\"\n\n"
                        f"# Istruzioni\n"
                        f"1. Rispondi basandoti ESCLUSIVAMENTE sulla descrizione fornita.\n"
                        f"2. NON inventare dettagli assenti dalla descrizione.\n"
                        f"3. Se la descrizione non contiene informazioni sufficienti per "
                        f"rispondere, dichiaralo esplicitamente.\n"
                        f"4. Rispondi nella stessa lingua della domanda dell'utente.\n"
                        f"5. Sii preciso e strutturato nella risposta."
                    )
                    for chunk in ai_engine.generate_response_stream(
                        answer_prompt,
                        conversation_history=clean_history,
                        system_prompt=config.SYSTEM_PROMPT,
                        model=text_model,
                    ):
                        full_response += chunk
                        yield _sse_data(chunk)

            elif PILOT_ENABLED and pilot:
                # Streaming via Pilot (con eventuale pianificazione ReAct)
                for chunk in pilot.process_stream(
                    user_message,
                    conversation_history=clean_history,
                    ai_engine=ai_engine,
                    conv_id=conv_id,
                    images=images,
                ):
                    full_response += chunk
                    yield _sse_data(chunk)
            else:
                for chunk in ai_engine.generate_response_stream(
                    user_message, 
                    conversation_history=clean_history,
                    system_prompt=system_prompt,
                    images=images,
                ):
                    full_response += chunk
                    yield _sse_data(chunk)
            
            # Salva la risposta completa
            memory.add_message_advanced(conv_id, 'assistant', full_response, extract_entities=False)
            response_saved = True
            
            # Invia segnale di fine
            yield f"event: end\ndata: {conv_id}\n\n"
            
        except Exception as e:
            logger.error("Errore streaming: %s", e, exc_info=True)
            yield f"event: error\ndata: Errore interno del server\n\n"
        finally:
            # P1-8: salva risposta parziale anche in caso di errore
            if not response_saved and full_response.strip():
                try:
                    memory.add_message_advanced(
                        conv_id, 'assistant', full_response, extract_entities=False
                    )
                except Exception:
                    logger.warning("Impossibile salvare risposta parziale")
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        }
    )


@app.route('/api/conversations', methods=['GET'])
@limiter.limit(config.RATE_LIMIT_CONFIG['limits']['conversations'])
def api_list_conversations():
    """Lista tutte le conversazioni"""
    conversations = memory.list_all_conversations()
    return jsonify(conversations)


@app.route('/api/conversations/<conv_id>', methods=['GET'])
@limiter.limit(config.RATE_LIMIT_CONFIG['limits']['conversations'])
def api_get_conversation(conv_id):
    """Ottiene una conversazione specifica"""
    if not _SAFE_CONV_ID.match(conv_id):
        return jsonify({'error': 'ID conversazione non valido'}), 400
    conversation = memory.load_conversation(conv_id)
    
    if not conversation:
        return jsonify({'error': 'Conversazione non trovata'}), 404
    
    return jsonify(conversation)


@app.route('/api/conversations/<conv_id>', methods=['DELETE'])
@limiter.limit(config.RATE_LIMIT_CONFIG['limits']['conversations'])
def api_delete_conversation(conv_id):
    """Elimina una conversazione"""
    if not _SAFE_CONV_ID.match(conv_id):
        return jsonify({'error': 'ID conversazione non valido'}), 400
    success = memory.delete_conversation(conv_id)
    
    if success:
        return jsonify({'success': True})
    else:
        return jsonify({'error': 'Impossibile eliminare'}), 404


@app.route('/api/conversations/new', methods=['POST'])
@limiter.limit(config.RATE_LIMIT_CONFIG['limits']['conversations'])
def api_new_conversation():
    """Crea una nuova conversazione
    
    Inizializza una conversazione vuota nella memoria.
    Il client riceve l'ID e lo usa per le successive richieste.
    """
    data = request.get_json(silent=True) or {}
    title = data.get('title')
    
    conv_id = memory.create_new_conversation(title)
    
    return jsonify({
        'conversation_id': conv_id,
        'success': True
    })


@app.route('/api/upload', methods=['POST'])
@limiter.limit(config.RATE_LIMIT_CONFIG['limits']['upload'])
def api_upload_document():
    """Carica e analizza un documento
    
    Supporta PDF, DOCX, TXT e altri formati.
    Estrae il testo e lo rende disponibile per analisi AI.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Nessun file caricato'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Nome file vuoto'}), 400
    
    if not doc_processor.is_allowed_file(file.filename):
        return jsonify({'error': 'Tipo di file non supportato'}), 400
    
    # Salva il file
    file_data = file.read()
    filepath, error = doc_processor.save_upload(file_data, file.filename)
    
    if error:
        return jsonify({'error': error}), 400
    
    # Processa il file
    text, error = doc_processor.process_file(filepath)
    
    if error:
        return jsonify({'error': error}), 400
    
    # Ottieni info file
    file_info = doc_processor.get_file_info(filepath)
    
    # Indicizza il documento nella memoria Pilot (se attivo)
    if PILOT_ENABLED and pilot and text:
        try:
            pilot.add_document(filepath, text, tags=[file.filename])
        except Exception:
            pass  # Non critico, il documento rimane comunque accessibile
    
    return jsonify({
        'success': True,
        'text': text[:5000],  # Prime 5000 caratteri come preview
        'full_length': len(text),
        'file_info': file_info
    })


@app.route('/api/analyze-document', methods=['POST'])
@limiter.limit(config.RATE_LIMIT_CONFIG['limits']['default'])
def api_analyze_document():
    """Analizza un documento con l'AI
    
    Permette di fare domande specifiche su un documento caricato.
    L'AI genera una risposta basata sul contenuto del documento.
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Nessun dato ricevuto'}), 400
    document_text = data.get('text', '')
    question = data.get('question')
    
    if not document_text:
        return jsonify({'error': 'Nessun testo fornito'}), 400
    
    try:
        analysis = ai_engine.analyze_document(document_text, question)
        
        return jsonify({
            'analysis': analysis,
            'success': True
        })
    
    except Exception as e:
        logger.error("Errore analyze-document: %s", e, exc_info=True)
        return jsonify({
            'error': 'Errore interno del server',
            'success': False
        }), 500


# ============================================================================
# ENDPOINT MEMORIA AVANZATA
# Forniscono accesso a funzionalità avanzate del sistema di memoria
# ============================================================================

@app.route('/api/conversation/<conv_id>/stats', methods=['GET'])
@limiter.limit(config.RATE_LIMIT_CONFIG['limits']['default'])
def api_conversation_stats(conv_id):
    """Ottiene statistiche dettagliate sulla conversazione
    
    Restituisce metriche come numero messaggi, token count, topic estratti.
    """
    stats = memory.get_conversation_stats(conv_id)
    
    if not stats:
        return jsonify({'error': 'Conversazione non trovata'}), 404
    
    return jsonify({
        'success': True,
        'stats': stats
    })


@app.route('/api/knowledge/summary', methods=['GET'])
@limiter.limit(config.RATE_LIMIT_CONFIG['limits']['default'])
def api_knowledge_summary():
    """Esporta un riassunto della knowledge base"""
    summary = memory.export_knowledge_summary()
    
    return jsonify({
        'success': True,
        'summary': summary,
        'knowledge_base': {
            'user_profile': memory.knowledge_base.knowledge.get('user_profile', {}),
            'topics_discussed': dict(memory.knowledge_base.knowledge.get('topics_discussed', {})),
            'topics_count': len(memory.knowledge_base.knowledge.get('topics_discussed', {})),
            'last_updated': memory.knowledge_base.knowledge.get('last_updated')
        }
    })


@app.route('/api/knowledge/search', methods=['POST'])
@limiter.limit(config.RATE_LIMIT_CONFIG['limits']['knowledge'])
def api_knowledge_search():
    """Cerca informazioni nella knowledge base"""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Nessun dato ricevuto'}), 400
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({'error': 'Query vuota'}), 400
    
    results = memory.search_in_knowledge(query)
    
    return jsonify({
        'success': True,
        'query': query,
        'results': results
    })


@app.route('/api/entities', methods=['GET'])
@limiter.limit(config.RATE_LIMIT_CONFIG['limits']['default'])
def api_get_entities():
    """Ottiene tutte le entità tracciate"""
    entities = memory.entity_tracker.entities
    
    return jsonify({
        'success': True,
        'entities': {
            'people_count': len(entities.get('people', {})),
            'people': list(entities.get('people', {}).keys())[:10],  # Prime 10
            'preferences_count': len(entities.get('preferences', {})),
            'dates_count': len(entities.get('dates', []))
        }
    })


@app.route('/api/memory/optimize', methods=['POST'])
@limiter.limit(config.RATE_LIMIT_CONFIG['limits']['default'])
def api_optimize_memory():
    """Forza l'ottimizzazione della memoria per la conversazione corrente
    
    Utile per testare la compressione e il summarization della memoria.
    """
    data = request.get_json() or {}
    conv_id = data.get('conversation_id')
    
    if not conv_id:
        return jsonify({'error': 'Nessuna conversazione attiva'}), 400
    
    # Validazione conv_id
    if not _SAFE_CONV_ID.match(conv_id):
        return jsonify({'error': 'ID conversazione non valido'}), 400
    
    # Ottieni contesto ottimizzato
    optimized_messages, additional_context = memory.get_smart_context(conv_id, ai_engine)
    
    conv_data = memory.load_conversation(conv_id)
    if not conv_data:
        return jsonify({'error': 'Conversazione non trovata'}), 404
    
    return jsonify({
        'success': True,
        'original_messages_count': len(conv_data.get('messages', [])),
        'optimized_messages_count': len(optimized_messages),
        'has_summary': any(m.get('is_summary') for m in optimized_messages),
        'additional_context_length': len(additional_context)
    })


@app.route('/api/models', methods=['GET'])
@limiter.limit(config.RATE_LIMIT_CONFIG['limits']['default'])
def api_list_models():
    """Lista i modelli disponibili (esclusi quelli vision-only di backend)."""
    models = user_visible_models(ai_engine)
    return jsonify({
        'models': models,
        'current': ai_engine.model
    })


@app.route('/api/models/change', methods=['POST'])
@limiter.limit(config.RATE_LIMIT_CONFIG['limits']['models'])
def api_change_model():
    """Cambia il modello AI
    
    Switching dinamico tra modelli Ollama (es. mistral, llama2, ecc.)
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Nessun dato ricevuto'}), 400
    new_model = data.get('model')
    
    if not new_model:
        return jsonify({'error': 'Nessun modello specificato'}), 400
    
    with _model_lock:
        success = ai_engine.change_model(new_model)
    
    return jsonify({
        'success': success,
        'current_model': ai_engine.model
    })


# ============================================================================
# ENDPOINT AI-PILOT
# Accesso alle funzionalità avanzate del Pilot
# ============================================================================

@app.route('/api/pilot/status', methods=['GET'])
@limiter.limit(config.RATE_LIMIT_CONFIG['limits']['default'])
def api_pilot_status():
    """Stato del sistema AI-Pilot"""
    if not PILOT_ENABLED or not pilot:
        return jsonify({'enabled': False, 'reason': 'Pilot non inizializzato'})
    return jsonify({'enabled': True, **pilot.get_status()})


@app.route('/api/pilot/memory', methods=['GET'])
@limiter.limit(config.RATE_LIMIT_CONFIG['limits']['default'])
def api_pilot_memory():
    """Statistiche e fatti dalla memoria Pilot"""
    if not PILOT_ENABLED or not pilot:
        return jsonify({'error': 'Pilot non attivo'}), 503
    return jsonify({
        'stats': pilot.get_memory_stats(),
        'facts': pilot.get_all_facts(),
        'open_tasks': pilot.get_open_tasks(),
    })


@app.route('/api/pilot/memory/search', methods=['POST'])
@limiter.limit(config.RATE_LIMIT_CONFIG['limits']['default'])
def api_pilot_memory_search():
    """Cerca nella memoria strutturata del Pilot"""
    if not PILOT_ENABLED or not pilot:
        return jsonify({'error': 'Pilot non attivo'}), 503
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Nessun dato ricevuto'}), 400
    query = data.get('query', '').strip()
    if not query:
        return jsonify({'error': 'Query vuota'}), 400
    results = pilot.search_memory(query)
    return jsonify({'success': True, 'query': query, 'results': results})


@app.route('/api/pilot/memory/fact', methods=['POST'])
@limiter.limit(config.RATE_LIMIT_CONFIG['limits']['default'])
def api_pilot_add_fact():
    """Aggiunge un fatto manuale alla memoria Pilot"""
    if not PILOT_ENABLED or not pilot:
        return jsonify({'error': 'Pilot non attivo'}), 503
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Nessun dato ricevuto'}), 400
    key = data.get('key', '').strip()
    value = data.get('value', '').strip()
    if not key or not value:
        return jsonify({'error': 'Chiave e valore richiesti'}), 400
    fid = pilot.add_fact(key, value, source='api')
    return jsonify({'success': True, 'fact_id': fid})


@app.route('/api/pilot/task', methods=['POST'])
@limiter.limit(config.RATE_LIMIT_CONFIG['limits']['default'])
def api_pilot_add_task():
    """Crea un task nel Pilot"""
    if not PILOT_ENABLED or not pilot:
        return jsonify({'error': 'Pilot non attivo'}), 503
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Nessun dato ricevuto'}), 400
    title = data.get('title', '').strip()
    if not title:
        return jsonify({'error': 'Titolo richiesto'}), 400
    tid = pilot.add_task(title, due_at=data.get('due_at'))
    return jsonify({'success': True, 'task_id': tid})


@app.route('/api/pilot/reload', methods=['POST'])
@limiter.limit(config.RATE_LIMIT_CONFIG['limits']['default'])
def api_pilot_reload():
    """Ricarica la configurazione del Pilot da disco"""
    if not PILOT_ENABLED or not pilot:
        return jsonify({'error': 'Pilot non attivo'}), 503
    try:
        pilot.reload_config()
        return jsonify({'success': True, 'version': pilot.cfg.version})
    except Exception as e:
        logger.error("Errore reload config: %s", e, exc_info=True)
        return jsonify({'error': 'Errore nel reload della configurazione'}), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint non trovato'}), 404


@app.errorhandler(400)
def bad_request(e):
    """Handler per bad request"""
    logger.warning("BAD REQUEST 400: %s", e)
    return jsonify({
        'error': 'Richiesta non valida'
    }), 400


@app.errorhandler(429)
def ratelimit_handler(e):
    """Handler per rate limit exceeded
    
    Messaggio user-friendly che spiega il problema.
    In sviluppo, suggerisce come disabilitare il rate limiting.
    """
    return jsonify({
        'error': 'Troppe richieste. Rallenta un po\' 😊',
        'message': 'Hai superato il limite di richieste permesse. Attendi qualche secondo.',
        'retry_after': getattr(e.description, 'retry_after', 60),
        **({'dev_note': 'In sviluppo puoi disabilitare il rate limiting con: RATE_LIMIT_ENABLED=False'} if app.debug else {})
    }), 429


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Errore interno del server'}), 500


if __name__ == '__main__':
    # Configura logging per startup
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(name)s: %(message)s',
    )

    logger.info("OMNI EYE AI - Avvio Server...")
    
    # Stato AI-Pilot
    if PILOT_ENABLED and pilot:
        status = pilot.get_status()
        mem = status['memory']
        logger.info(
            "AI-Pilot v%s — Planner: %s | Tools: %s | Memoria: %d fatti, %d task, %d chunk",
            status['version'], status['planner'],
            ', '.join(status['tools']) or 'nessuno',
            mem['facts'], mem['tasks'], mem['document_chunks'],
        )
    
    # Verifica sistema
    if ai_engine.check_ollama_available():
        logger.info("Ollama disponibile")
        if ai_engine.check_model_available():
            logger.info("Modello '%s' pronto", ai_engine.model)
        else:
            logger.warning("Modello '%s' non trovato! Scaricalo con: ollama pull %s", ai_engine.model, ai_engine.model)
            models = ai_engine.list_available_models()
            if models:
                logger.info("Modelli disponibili: %s", ', '.join(models))
    else:
        logger.error("Ollama non disponibile! Installa da: https://ollama.ai/download")
    
    logger.info("Server in esecuzione su: http://localhost:%s", config.SERVER_CONFIG['port'])
    
    app.run(
        host=config.SERVER_CONFIG['host'],
        port=config.SERVER_CONFIG['port'],
        debug=config.SERVER_CONFIG['debug']
    )
