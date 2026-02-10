"""
Omni Eye AI - Backend Flask
"""

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf import CSRFProtect
import logging
import os
import sys

logger = logging.getLogger(__name__)

# Aggiungi la directory root al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import AIEngine, DocumentProcessor
from core.advanced_memory import AdvancedMemory
from core.ai_pilot import Pilot
import config

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


# Modelli usati solo come backend per vision; nascosti dalla selezione utente
_VISION_ONLY_TAGS = ('moondream', 'bakllava', 'minicpm-v')

# Modelli vision multilingue: possono rispondere direttamente in qualsiasi lingua
# → pipeline singola (no fase intermedia in inglese)
_MULTILINGUAL_VISION = ('minicpm-v', 'llava:13b', 'llava-llama3')

# Modelli vision EN-only: descrivono solo in inglese
# → pipeline a 2 fasi (descrizione EN + risposta IT via modello testo)
_EN_ONLY_VISION = ('moondream', 'bakllava', 'llava-phi3')

# Ordine di priorità modelli vision (il primo disponibile viene usato)
_VISION_PRIORITY = ('minicpm-v', 'llava:13b', 'llava-llama3', 'llava:7b',
                    'llava', 'moondream', 'bakllava', 'vision')

def _user_visible_models() -> list:
    """Restituisce solo i modelli selezionabili dall'utente (esclude vision-only)."""
    return [m for m in ai_engine.list_available_models()
            if not any(tag in m.lower() for tag in _VISION_ONLY_TAGS)]


def _build_vision_prompt(user_message: str, multilingual: bool) -> str:
    """Costruisce un prompt vision adattivo in base alla domanda dell'utente.

    Analizza il contenuto della domanda per capire cosa l'utente vuole sapere
    e genera un prompt specializzato (OCR, persone, scena, tecnico, ecc.).

    Args:
        user_message: La domanda dell'utente sull'immagine
        multilingual: Se True, il prompt include istruzioni di lingua
    """
    msg = (user_message or "").lower()

    # Rilevamento intento dalla domanda utente
    ocr_keywords = ('testo', 'scritto', 'scritta', 'leggi', 'leggere', 'parole',
                    'scrivi', 'text', 'read', 'ocr', 'lettere', 'titolo',
                    'etichetta', 'label', 'caption', 'didascalia')
    people_keywords = ('persona', 'persone', 'gente', 'viso', 'volto', 'faccia',
                       'chi', 'uomo', 'donna', 'bambino', 'people', 'person',
                       'face', 'who', 'espressione', 'emozione')
    tech_keywords = ('codice', 'code', 'programma', 'errore', 'error', 'bug',
                     'screenshot', 'schermo', 'screen', 'interfaccia', 'ui',
                     'terminale', 'terminal', 'console', 'log')
    food_keywords = ('cibo', 'piatto', 'mangiare', 'ricetta', 'ingredienti',
                     'food', 'dish', 'recipe', 'cucina')
    location_keywords = ('dove', 'luogo', 'posto', 'citt', 'paese', 'location',
                         'where', 'edificio', 'building', 'strada', 'via')

    has_ocr = any(k in msg for k in ocr_keywords)
    has_people = any(k in msg for k in people_keywords)
    has_tech = any(k in msg for k in tech_keywords)
    has_food = any(k in msg for k in food_keywords)
    has_location = any(k in msg for k in location_keywords)

    # Costruisci prompt specializzato
    parts = []

    if multilingual:
        # Per modelli multilingue: rispondi direttamente alla domanda dell'utente
        if has_ocr:
            parts.append(
                "Analizza attentamente l'immagine e trascrivi TUTTO il testo visibile, "
                "inclusi titoli, etichette, didascalie e qualsiasi scritta. "
                "Mantieni la formattazione originale dove possibile."
            )
        elif has_tech:
            parts.append(
                "Analizza questo screenshot/interfaccia tecnica. "
                "Identifica il tipo di applicazione, il linguaggio di programmazione se presente, "
                "eventuali errori visibili, e descrivi i dettagli tecnici rilevanti."
            )
        elif has_people:
            parts.append(
                "Descrivi le persone presenti nell'immagine: quante sono, "
                "il loro aspetto generale, espressioni, posizioni, "
                "abbigliamento e cosa stanno facendo."
            )
        elif has_food:
            parts.append(
                "Analizza il cibo/piatto nell'immagine. Identifica gli ingredienti visibili, "
                "il tipo di piatto, la presentazione e il contesto."
            )
        elif has_location:
            parts.append(
                "Descrivi il luogo mostrato nell'immagine: tipo di ambiente, "
                "elementi architettonici, segnali o indicazioni di localizzazione, "
                "atmosfera e dettagli rilevanti."
            )
        else:
            parts.append(
                "Descrivi tutto ciò che vedi nell'immagine nel modo più completo possibile."
            )

        if user_message and user_message.strip():
            parts.append(f"\nDomanda specifica dell'utente: \"{user_message}\"")
            parts.append("Rispondi nella stessa lingua della domanda.")
        else:
            parts.append("Rispondi in italiano.")

    else:
        # Per modelli EN-only: descrizione dettagliata in inglese
        if has_ocr:
            parts.append(
                "Carefully examine this image and transcribe ALL visible text, "
                "including titles, labels, captions, signs, and any writing. "
                "Preserve the original formatting where possible."
            )
        elif has_tech:
            parts.append(
                "Analyze this screenshot or technical interface in detail. "
                "Identify the application type, programming language if present, "
                "any visible errors, UI elements, and technical details."
            )
        elif has_people:
            parts.append(
                "Describe the people in this image: how many, their general appearance, "
                "expressions, positions, clothing, and what they are doing."
            )
        elif has_food:
            parts.append(
                "Analyze the food/dish in this image. Identify visible ingredients, "
                "the type of dish, presentation style, and context."
            )
        elif has_location:
            parts.append(
                "Describe the location shown in this image: environment type, "
                "architectural elements, signs or landmarks, atmosphere, and notable details."
            )
        else:
            parts.append(
                "Describe everything you see in this image in detail. "
                "Include objects, colors, text, people, animals, spatial layout, "
                "and any notable features."
            )

        # Aggiungi sempre focus su aspetti importanti per EN-only
        parts.append(
            "\nBe precise and thorough. If there is any text visible, transcribe it exactly."
        )

    return "\n".join(parts)


@app.route('/api/status', methods=['GET'])
def api_status():
    """Verifica lo stato del sistema"""
    ollama_ok = ai_engine.check_ollama_available()
    model_ok = ai_engine.check_model_available() if ollama_ok else False
    
    return jsonify({
        'ollama_available': ollama_ok,
        'model_available': model_ok,
        'model_name': ai_engine.model,
        'available_models': _user_visible_models() if ollama_ok else []
    })


@app.route('/api/chat', methods=['POST'])
@limiter.limit(config.RATE_LIMIT_CONFIG['limits']['chat'])
def api_chat():
    """Endpoint per chat normale (non streaming)
    
    Gestisce una singola richiesta chat e restituisce la risposta completa.
    Utilizza il sistema di memoria avanzata per contesto e learning.
    """
    data = request.json
    user_message = data.get('message', '').strip()
    conv_id = data.get('conversation_id')
    
    if not user_message:
        return jsonify({'error': 'Messaggio vuoto'}), 400
    
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
        return jsonify({
            'error': str(e),
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
        images = data.get('images')  # Lista di stringhe base64 (per modelli vision)
        
        logger.debug("Richiesta chat: message='%s...', conv_id=%s, images=%d",
                      user_message[:50], conv_id, len(images) if images else 0)
        
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
    image_context_parts = []
    clean_history = []
    for msg in history:
        if msg.get('role') == 'system' and '[ANALISI IMMAGINE]' in msg.get('content', ''):
            image_context_parts.append(msg['content'])
        else:
            clean_history.append(msg)
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
        saved_model = ai_engine.model  # Salva sempre per ripristino
        
        try:
            # Auto-switch a modello vision se ci sono immagini
            if images:
                all_models = ai_engine.list_available_models()
                vision_models = [m for m in all_models
                                 if any(v in m.lower() for v in _VISION_PRIORITY)]
                # Ordina per priorità definita in _VISION_PRIORITY
                def _priority(name):
                    low = name.lower()
                    for i, tag in enumerate(_VISION_PRIORITY):
                        if tag in low:
                            return i
                    return len(_VISION_PRIORITY)
                vision_models.sort(key=_priority)

                current = ai_engine.model.lower()
                is_vision = any(v in current for v in _VISION_PRIORITY)

                if not is_vision and vision_models:
                    ai_engine.model = vision_models[0]
                    logger.info("Auto-switch a modello vision: %s", ai_engine.model)
                elif not is_vision and not vision_models:
                    yield f"data: ⚠️ Nessun modello vision installato. Scaricane uno con: ollama pull minicpm-v\n\n"
                    yield f"event: end\ndata: {conv_id}\n\n"
                    return

                vision_model = ai_engine.model
                text_model = config.AI_CONFIG['model']
                is_multilingual = any(tag in vision_model.lower()
                                      for tag in _MULTILINGUAL_VISION)

                # --- Prompt adattivo in base alla domanda utente ---
                vision_prompt = _build_vision_prompt(user_message, is_multilingual)

                if is_multilingual:
                    # ═══ PIPELINE SINGOLA — modello multilingue ═══
                    # Il modello vision risponde direttamente nella lingua
                    # dell'utente, con l'immagine allegata. Niente fase 2.
                    logger.info("Vision pipeline SINGOLA (multilingual): %s", vision_model)

                    # Temperatura bassa per massima accuratezza visiva
                    saved_temp = ai_engine.temperature
                    ai_engine.temperature = 0.2

                    direct_response = ""
                    for chunk in ai_engine.generate_response_stream(
                        vision_prompt,
                        conversation_history=clean_history[-4:],  # ultimi 2 turni per contesto
                        system_prompt=(
                            "Sei un assistente visivo esperto. Analizza le immagini "
                            "con precisione e rispondi nella lingua dell'utente. "
                            "Sii dettagliato su testo visibile (OCR), colori, layout, "
                            "persone, oggetti e contesto."
                        ),
                        images=images,
                    ):
                        direct_response += chunk
                        full_response += chunk
                        yield f"data: {chunk}\n\n"

                    ai_engine.temperature = saved_temp

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
                    # Fase 1: descrizione inglese con modello vision
                    # Fase 2: risposta nella lingua dell'utente via modello testo
                    logger.info("Vision pipeline 2 FASI (EN-only): %s -> %s",
                                vision_model, text_model)

                    saved_temp = ai_engine.temperature
                    ai_engine.temperature = 0.15  # massima fedeltà per descrizione

                    image_description = ""
                    for chunk in ai_engine.generate_response_stream(
                        vision_prompt,
                        conversation_history=None,
                        system_prompt=None,
                        images=images,
                    ):
                        image_description += chunk

                    ai_engine.temperature = saved_temp
                    logger.info("Vision fase 1 completata: %d chars", len(image_description))

                    # Salva la descrizione nella conversazione per follow-up
                    img_context = (
                        f"[ANALISI IMMAGINE] L'utente ha inviato un'immagine. "
                        f"Questa è la descrizione oggettiva di ciò che contiene: "
                        f"{image_description.strip()}"
                    )
                    memory.add_message_advanced(
                        conv_id, 'system', img_context, extract_entities=False
                    )

                    # Fase 2 — risposta all'utente con modello testo
                    ai_engine.model = text_model

                    answer_prompt = (
                        f"Ho analizzato un'immagine e ottenuto questa descrizione dettagliata:\n\n"
                        f"---\n{image_description.strip()}\n---\n\n"
                        f"Domanda dell'utente sull'immagine: \"{user_message}\"\n\n"
                        f"Rispondi in modo dettagliato e preciso, basandoti SOLO sulla "
                        f"descrizione dell'immagine. Non inventare dettagli che non sono "
                        f"nella descrizione. Rispondi nella stessa lingua della domanda."
                    )
                    for chunk in ai_engine.generate_response_stream(
                        answer_prompt,
                        conversation_history=clean_history,
                        system_prompt=config.SYSTEM_PROMPT,
                    ):
                        full_response += chunk
                        yield f"data: {chunk}\n\n"

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
                    yield f"data: {chunk}\n\n"
            else:
                for chunk in ai_engine.generate_response_stream(
                    user_message, 
                    conversation_history=clean_history,
                    system_prompt=system_prompt,
                    images=images,
                ):
                    full_response += chunk
                    yield f"data: {chunk}\n\n"
            
            # Salva la risposta completa
            memory.add_message_advanced(conv_id, 'assistant', full_response, extract_entities=False)
            
            # Invia segnale di fine
            yield f"event: end\ndata: {conv_id}\n\n"
            
        except Exception as e:
            yield f"event: error\ndata: {str(e)}\n\n"
        finally:
            # Ripristina sempre il modello che l'utente aveva selezionato
            ai_engine.model = saved_model
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/api/conversations', methods=['GET'])
@limiter.limit(config.RATE_LIMIT_CONFIG['limits']['conversations'])
def api_list_conversations():
    """Lista tutte le conversazioni"""
    conversations = memory.list_all_conversations()
    return jsonify(conversations)


@app.route('/api/conversations/<conv_id>', methods=['GET'])
def api_get_conversation(conv_id):
    """Ottiene una conversazione specifica"""
    conversation = memory.load_conversation(conv_id)
    
    if not conversation:
        return jsonify({'error': 'Conversazione non trovata'}), 404
    
    return jsonify(conversation)


@app.route('/api/conversations/<conv_id>', methods=['DELETE'])
def api_delete_conversation(conv_id):
    """Elimina una conversazione"""
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
        'file_info': file_info,
        'filepath': filepath
    })


@app.route('/api/analyze-document', methods=['POST'])
def api_analyze_document():
    """Analizza un documento con l'AI
    
    Permette di fare domande specifiche su un documento caricato.
    L'AI genera una risposta basata sul contenuto del documento.
    """
    data = request.json
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
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


# ============================================================================
# ENDPOINT MEMORIA AVANZATA
# Forniscono accesso a funzionalità avanzate del sistema di memoria
# ============================================================================

@app.route('/api/conversation/<conv_id>/stats', methods=['GET'])
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
    data = request.json
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
def api_optimize_memory():
    """Forza l'ottimizzazione della memoria per la conversazione corrente
    
    Utile per testare la compressione e il summarization della memoria.
    """
    data = request.get_json() or {}
    conv_id = data.get('conversation_id')
    
    if not conv_id:
        return jsonify({'error': 'Nessuna conversazione attiva'}), 400
    
    # Ottieni contesto ottimizzato
    optimized_messages, additional_context = memory.get_smart_context(conv_id, ai_engine)
    
    return jsonify({
        'success': True,
        'original_messages_count': len(memory.load_conversation(conv_id).get('messages', [])),
        'optimized_messages_count': len(optimized_messages),
        'has_summary': any(m.get('is_summary') for m in optimized_messages),
        'additional_context_length': len(additional_context)
    })


@app.route('/api/models', methods=['GET'])
def api_list_models():
    """Lista i modelli disponibili (esclusi quelli vision-only di backend)."""
    models = _user_visible_models()
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
    data = request.json
    new_model = data.get('model')
    
    if not new_model:
        return jsonify({'error': 'Nessun modello specificato'}), 400
    
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
def api_pilot_status():
    """Stato del sistema AI-Pilot"""
    if not PILOT_ENABLED or not pilot:
        return jsonify({'enabled': False, 'reason': 'Pilot non inizializzato'})
    return jsonify({'enabled': True, **pilot.get_status()})


@app.route('/api/pilot/memory', methods=['GET'])
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
def api_pilot_memory_search():
    """Cerca nella memoria strutturata del Pilot"""
    if not PILOT_ENABLED or not pilot:
        return jsonify({'error': 'Pilot non attivo'}), 503
    data = request.json
    query = data.get('query', '').strip()
    if not query:
        return jsonify({'error': 'Query vuota'}), 400
    results = pilot.search_memory(query)
    return jsonify({'success': True, 'query': query, 'results': results})


@app.route('/api/pilot/memory/fact', methods=['POST'])
def api_pilot_add_fact():
    """Aggiunge un fatto manuale alla memoria Pilot"""
    if not PILOT_ENABLED or not pilot:
        return jsonify({'error': 'Pilot non attivo'}), 503
    data = request.json
    key = data.get('key', '').strip()
    value = data.get('value', '').strip()
    if not key or not value:
        return jsonify({'error': 'Chiave e valore richiesti'}), 400
    fid = pilot.add_fact(key, value, source='api')
    return jsonify({'success': True, 'fact_id': fid})


@app.route('/api/pilot/task', methods=['POST'])
def api_pilot_add_task():
    """Crea un task nel Pilot"""
    if not PILOT_ENABLED or not pilot:
        return jsonify({'error': 'Pilot non attivo'}), 503
    data = request.json
    title = data.get('title', '').strip()
    if not title:
        return jsonify({'error': 'Titolo richiesto'}), 400
    tid = pilot.add_task(title, due_at=data.get('due_at'))
    return jsonify({'success': True, 'task_id': tid})


@app.route('/api/pilot/reload', methods=['POST'])
def api_pilot_reload():
    """Ricarica la configurazione del Pilot da disco"""
    if not PILOT_ENABLED or not pilot:
        return jsonify({'error': 'Pilot non attivo'}), 503
    try:
        pilot.reload_config()
        return jsonify({'success': True, 'version': pilot.cfg.version})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint non trovato'}), 404


@app.errorhandler(400)
def bad_request(e):
    """Handler per bad request - debug"""
    logger.warning("BAD REQUEST 400: %s — %s", e, getattr(e, 'description', ''))
    return jsonify({
        'error': 'Richiesta non valida',
        'details': str(e.description) if hasattr(e, 'description') else str(e)
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
        'dev_note': 'In sviluppo puoi disabilitare il rate limiting con: RATE_LIMIT_ENABLED=False'
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
