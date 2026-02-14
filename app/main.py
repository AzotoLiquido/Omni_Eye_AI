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
from core.web_search import search_and_format
from core.model_router import ModelRouter, ModelMapping, Intent, classify_intent
from core.pipeline import (
    PipelineScheduler, build_maintenance_pipeline,
    build_document_pipeline, build_memory_pipeline,
)
from app.vision import (
    VISION_ONLY_TAGS, MULTILINGUAL_VISION, VISION_PRIORITY,
    VISION_SYSTEM_PROMPT, user_visible_models, vision_model_priority,
    build_vision_prompt,
)
import config

# Regex per validazione ID conversazione (anti path-traversal)
_SAFE_CONV_ID = _re.compile(r'^[a-zA-Z0-9_-]+$')

# Regex per domande sull'identità (compilato una volta)
_IDENTITY_RE = _re.compile(config.IDENTITY_PATTERNS, _re.IGNORECASE)

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

# Inizializza Model Router (instradamento intelligente multi-modello)
_router_cfg = getattr(config, 'MODEL_ROUTER_CONFIG', {})
if _router_cfg.get('enabled', False):
    _models = _router_cfg.get('models', {})
    model_router = ModelRouter(
        mapping=ModelMapping(
            general=_models.get('general', 'gemma3:4b'),
            code=_models.get('code', 'qwen2.5-coder:7b'),
            vision=_models.get('vision', 'minicpm-v'),
        ),
        fallback_model=_router_cfg.get('fallback', 'llama3.2'),
    )
    # Popola la cache dei modelli installati
    _installed = ai_engine.list_available_models()
    model_router.refresh_installed(_installed)
    # Imposta il warm model iniziale
    model_router.warm_model = ai_engine.model
    ROUTER_ENABLED = True
    logger.info("Model Router: ATTIVO (general=%s, code=%s, vision=%s)",
                _models.get('general'), _models.get('code'), _models.get('vision'))
else:
    model_router = None
    ROUTER_ENABLED = False
    logger.info("Model Router: DISATTIVATO (modello singolo: %s)", ai_engine.model)

# ── Warmup: pre-carica il modello default in VRAM per eliminare il cold start ──
def _warmup_model():
    """Genera un singolo token silenziosamente per forzare il caricamento in VRAM."""
    try:
        model = ai_engine.model
        logger.info("Warmup: caricamento %s in VRAM...", model)
        import time as _t
        _start = _t.perf_counter()
        ai_engine.client.chat(
            model=model,
            messages=[{"role": "user", "content": "test"}],
            options={"num_predict": 1},
            keep_alive=ai_engine.KEEP_ALIVE,
        )
        _elapsed = _t.perf_counter() - _start
        logger.info("Warmup completato: %s caricato in %.1fs", model, _elapsed)
    except Exception as e:
        logger.warning("Warmup fallito (non critico): %s", e)

# Avvia warmup in background per non bloccare l'avvio del server
threading.Thread(target=_warmup_model, daemon=True, name="model-warmup").start()

# ── Pipeline Scheduler: manutenzione periodica in background ──────────────────
pipeline_scheduler = PipelineScheduler()
pipeline_scheduler.register(
    "maintenance", build_maintenance_pipeline(),
    interval_seconds=6 * 3600,   # ogni 6 ore
    run_on_start=True,           # pulizia immediata all'avvio
)
pipeline_scheduler.register(
    "memory_refresh", build_memory_pipeline(),
    interval_seconds=30 * 60,    # ogni 30 minuti
)
pipeline_scheduler.start()
logger.info("Pipeline Scheduler: avviato (maintenance=6h, memory=30min)")

# Pipeline riutilizzabile per processing documenti
_document_pipeline = build_document_pipeline()

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
        'available_models': user_visible_models(ai_engine) if ollama_ok else [],
        'router': {
            'enabled': ROUTER_ENABLED,
            'models': model_router.mapping.__dict__ if ROUTER_ENABLED and model_router else None,
            'warm_model': model_router.warm_model if ROUTER_ENABLED and model_router else None,
        },
        'pipeline_scheduler': pipeline_scheduler.get_status(),
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
    
    # ── Avvia web search in background MENTRE si carica la conversazione ──
    # Salta web search per documenti caricati (il contenuto è locale)
    _has_document = user_message.startswith('[Documento caricato:')
    _ws_result = [None]
    def _bg_search():
        _ws_result[0] = search_and_format(user_message) if not _has_document else None
    _ws_thread = threading.Thread(target=_bg_search, daemon=True)
    _ws_thread.start()

    # Salva messaggio utente con estrazione entità
    memory.add_message_advanced(conv_id, 'user', user_message)
    
    # Ottieni contesto ottimizzato con compressione automatica
    history, additional_context = memory.get_smart_context(conv_id, ai_engine)
    
    # Rimuovi l'ultimo messaggio dalla history
    if history:
        history = history[:-1]
    
    # ── Aspetta risultato web search (ormai completata in parallelo) ──
    _ws_thread.join(timeout=10)
    web_search_data_sync = _ws_result[0]
    
    # Modalità "links": risultati diretti + frase di chiusura statica
    if web_search_data_sync and web_search_data_sync["mode"] == "links":
        user_results = web_search_data_sync["user"]
        response = user_results + "\nEcco i risultati trovati, fammi sapere se ti serve altro."
        memory.add_message_advanced(conv_id, 'assistant', response, extract_entities=False)
        return jsonify({
            'response': response,
            'conversation_id': conv_id,
            'success': True,
            'pilot_meta': None
        })
    
    # Model routing: seleziona modello ottimale PRIMA di costruire il prompt
    route_result = None
    routed_model = None
    _is_code_intent = False
    if ROUTER_ENABLED and model_router:
        route_result = model_router.route(user_message)
        routed_model = route_result.model
        _is_code_intent = (route_result.intent == Intent.CODE)
        if route_result.is_swap:
            logger.info("Router swap → %s (intent=%s)", routed_model, route_result.intent.value)

    # System prompt: Pilot (se attivo) oppure fallback config.py
    # Iniezione identità: se l'utente chiede "chi sei", aggiungi il prompt
    # identità. Altrimenti il nome non compare MAI nel system prompt.
    identity_inject = ""
    if _IDENTITY_RE.search(user_message):
        identity_inject = config.IDENTITY_PROMPT + "\n"

    if PILOT_ENABLED and pilot:
        pilot_extra = identity_inject
        if web_search_data_sync and web_search_data_sync["mode"] == "augmented":
            pilot_extra += web_search_data_sync["context"] + "\n\n"
        if additional_context:
            pilot_extra += additional_context
        system_prompt = None  # Pilot costruisce il suo internamente
    else:
        # Usa il prompt specializzato per codice se l'intent è CODE
        base_prompt = config.CODE_SYSTEM_PROMPT if _is_code_intent else config.SYSTEM_PROMPT
        system_prompt = base_prompt
        if identity_inject:
            system_prompt = identity_inject + system_prompt
        if additional_context:
            system_prompt = additional_context + "\n" + system_prompt
        # Modalità "augmented": inietta contesto web nel system prompt
        if web_search_data_sync and web_search_data_sync["mode"] == "augmented":
            system_prompt = web_search_data_sync["context"] + "\n\n" + system_prompt
    
    # Genera risposta
    try:

        if PILOT_ENABLED and pilot:
            response, meta = pilot.process(
                user_message,
                conversation_history=history,
                ai_engine=ai_engine,
                conv_id=conv_id,
                extra_instructions=pilot_extra if PILOT_ENABLED and pilot else "",
                model=routed_model,
            )
        else:
            response = ai_engine.generate_response(
                user_message, 
                conversation_history=history,
                system_prompt=system_prompt,
                model=routed_model,
            )
            meta = {}
        
        # Filtra URL inventati se la risposta è arricchita da ricerca web
        if web_search_data_sync and web_search_data_sync["mode"] == "augmented":
            from core.web_search import strip_hallucinated_urls
            response = strip_hallucinated_urls(response, web_search_data_sync["urls"])
        
        # Salva risposta (senza estrazione entità per l'assistant)
        memory.add_message_advanced(conv_id, 'assistant', response, extract_entities=False)
        
        return jsonify({
            'response': response,
            'conversation_id': conv_id,
            'success': True,
            'pilot_meta': meta if meta else None,
            'route': route_result.to_dict() if route_result else None,
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
    
    # ── Avvia web search in background MENTRE si carica la conversazione ──
    # Salta web search per documenti caricati (il contenuto è locale)
    _has_document = user_message.startswith('[Documento caricato:')
    _ws_stream_result = [None]
    def _bg_stream_search():
        _ws_stream_result[0] = search_and_format(user_message) if not images and not _has_document else None
    _ws_stream_thread = threading.Thread(target=_bg_stream_search, daemon=True)
    _ws_stream_thread.start()

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
    
    # ── Aspetta risultato web search (ormai completata in parallelo) ──
    _ws_stream_thread.join(timeout=10)
    web_search_data = _ws_stream_result[0]
    web_mode = web_search_data["mode"] if web_search_data else None
    
    # Model routing: seleziona modello ottimale PRIMA di costruire il prompt
    stream_route = None
    stream_routed_model = None
    _is_code_intent_stream = False
    if ROUTER_ENABLED and model_router:
        stream_route = model_router.route(user_message, has_images=bool(images))
        stream_routed_model = stream_route.model
        _is_code_intent_stream = (stream_route.intent == Intent.CODE)
        if stream_route.is_swap:
            logger.info("Router stream swap → %s (intent=%s)",
                        stream_routed_model, stream_route.intent.value)

    # System prompt + extra per Pilot
    # Iniezione identità: solo se l'utente chiede "chi sei"
    identity_inject_stream = ""
    if _IDENTITY_RE.search(user_message):
        identity_inject_stream = config.IDENTITY_PROMPT + "\n"

    # Seleziona prompt base: CODE_SYSTEM_PROMPT per codice, SYSTEM_PROMPT per il resto
    _base_stream = config.CODE_SYSTEM_PROMPT if _is_code_intent_stream else config.SYSTEM_PROMPT

    if PILOT_ENABLED and pilot and not web_search_data:
        pilot_extra_stream = identity_inject_stream
        if image_memory:
            pilot_extra_stream += image_memory + "\n"
        if additional_context:
            pilot_extra_stream += additional_context

        # Se c'è image_memory ma no immagini allegate → follow-up immagine,
        # serve un system_prompt perché bypassa il Pilot
        if image_memory and not images:
            system_prompt = _base_stream
            if identity_inject_stream:
                system_prompt += "\n" + identity_inject_stream
            system_prompt += "\n" + image_memory
            if additional_context:
                system_prompt += "\n" + additional_context
        else:
            system_prompt = None  # Pilot costruisce il suo internamente
    else:
        # KV cache: system_prompt fisso PRIMA, contesto dinamico DOPO
        system_prompt = _base_stream
        if identity_inject_stream:
            system_prompt += "\n" + identity_inject_stream
        if image_memory:
            system_prompt += "\n" + image_memory
        if additional_context:
            system_prompt += "\n" + additional_context

    # Modalità "augmented": inietta contesto web nel system prompt
    if web_mode == "augmented":
        system_prompt += "\n\n" + web_search_data["context"]
    
    def generate():
        """Generator per lo streaming"""
        full_response = ""
        response_saved = False
        
        try:
            # ── Web search: modalità "links" → risultati diretti, niente commento AI ──
            if web_mode == "links":
                user_results = web_search_data["user"]
                full_response = user_results + "\nEcco i risultati trovati, fammi sapere se ti serve altro."
                yield _sse_data(full_response)
                memory.add_message_advanced(conv_id, 'assistant', full_response, extract_entities=False)
                response_saved = True
                yield f"event: end\ndata: {conv_id}\n\n"
                return

            # ── Web search: modalità "augmented" → modello con contesto web ──
            # Buffer + filtro URL per togliere eventuali link inventati
            if web_mode == "augmented":
                model_buf = ""
                for chunk in ai_engine.generate_response_stream(
                    user_message,
                    conversation_history=clean_history,
                    system_prompt=system_prompt,
                    images=images,
                    model=stream_routed_model,
                ):
                    model_buf += chunk
                from core.web_search import strip_hallucinated_urls
                model_buf = strip_hallucinated_urls(model_buf, web_search_data["urls"])
                full_response += model_buf
                yield _sse_data(model_buf)

            # Auto-switch a modello vision se ci sono immagini
            elif images:
                # Se il router è attivo, usa il modello vision configurato
                if ROUTER_ENABLED and model_router and stream_routed_model:
                    vision_model = stream_routed_model
                    logger.info("Router vision → %s", vision_model)
                else:
                    all_models = ai_engine.list_available_models()
                    vision_models = [m for m in all_models
                                     if any(v in m.lower() for v in VISION_PRIORITY)]
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

                text_model = (model_router.mapping.general if ROUTER_ENABLED and model_router
                              else config.AI_CONFIG['model'])
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
                        f"L'utente ha inviato un'immagine. Un modello visivo l'ha analizzata "
                        f"e ha prodotto questa descrizione in inglese:\n\n"
                        f"---\n{image_description.strip()}\n---\n\n"
                        f"Domanda dell'utente: \"{user_message}\"\n\n"
                        f"Rispondi in italiano descrivendo ci\u00f2 che c'\u00e8 nell'immagine "
                        f"basandoti sulla descrizione. Traduci in modo naturale "
                        f"(es. 'tongue out' = 'lingua fuori', non 'linguaggio fuori'). "
                        f"Non inventare dettagli assenti. "
                        f"Non ripetere la domanda dell'utente. "
                        f"Sii descrittivo ma senza ripetizioni."
                    )
                    for chunk in ai_engine.generate_response_stream(
                        answer_prompt,
                        conversation_history=clean_history,
                        system_prompt=config.SYSTEM_PROMPT,
                        model=text_model,
                    ):
                        full_response += chunk
                        yield _sse_data(chunk)

            # ── Follow-up immagine: bypass Pilot per evitare knowledge base ──
            elif image_memory and not images:
                # L'utente sta chiedendo info su un'immagine analizzata in precedenza
                # (es. "descrivi", "dimmi di più", "cos'altro vedi")
                # Il contesto immagine è già nel system_prompt, usiamo il modello
                # testo direttamente per evitare che il Pilot cerchi nella KB
                logger.info("Image follow-up senza Pilot: '%s'", user_message[:50])
                for chunk in ai_engine.generate_response_stream(
                    user_message,
                    conversation_history=clean_history,
                    system_prompt=system_prompt,
                    model=stream_routed_model,
                ):
                    full_response += chunk
                    yield _sse_data(chunk)

            elif PILOT_ENABLED and pilot:
                for chunk in pilot.process_stream(
                    user_message,
                    conversation_history=clean_history,
                    ai_engine=ai_engine,
                    conv_id=conv_id,
                    images=images,
                    extra_instructions=pilot_extra_stream if PILOT_ENABLED and pilot else "",
                    model=stream_routed_model,
                ):
                    full_response += chunk
                    yield _sse_data(chunk)
            else:
                for chunk in ai_engine.generate_response_stream(
                    user_message, 
                    conversation_history=clean_history,
                    system_prompt=system_prompt,
                    images=images,
                    model=stream_routed_model,
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


@app.route('/api/conversations/<conv_id>/clear', methods=['POST'])
@limiter.limit(config.RATE_LIMIT_CONFIG['limits']['conversations'])
def api_clear_conversation(conv_id):
    """Svuota i messaggi di una conversazione senza eliminarla"""
    if not _SAFE_CONV_ID.match(conv_id):
        return jsonify({'error': 'ID conversazione non valido'}), 400
    success = memory.clear_conversation(conv_id)

    if success:
        return jsonify({'success': True})
    else:
        return jsonify({'error': 'Conversazione non trovata'}), 404


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
    
    # Indicizza il documento tramite pipeline asincrona (parse → chunk → index Pilot)
    if PILOT_ENABLED and pilot and text:
        def _run_doc_pipeline():
            try:
                _document_pipeline.run(
                    filepath=filepath,
                    filename=file.filename,
                    pilot=pilot,
                )
            except Exception as e:
                logger.warning("Document pipeline fallita: %s", e)
        threading.Thread(
            target=_run_doc_pipeline, daemon=True, name="doc-pipeline",
        ).start()
    
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


# ── Knowledge Packs API ────────────────────────────────────────────────────

@app.route('/api/knowledge/packs', methods=['GET'])
@limiter.limit(config.RATE_LIMIT_CONFIG['limits']['default'])
def api_knowledge_packs():
    """Lista pack disponibili e statistiche KB."""
    from core.knowledge_packs import get_available_packs
    packs = get_available_packs()
    return jsonify({
        'success': True,
        'packs': packs,
        'total_packs': len(packs),
        'total_facts_available': sum(p['facts_count'] for p in packs),
        'facts_in_db': memory.knowledge_base.get_facts_count(),
    })


@app.route('/api/knowledge/packs/install', methods=['POST'])
@limiter.limit(config.RATE_LIMIT_CONFIG['limits']['default'])
def api_install_pack():
    """Installa uno o tutti i pack. Body: {"pack": "nome"} o {"all": true}."""
    from core.knowledge_packs import install_pack, install_all_packs
    data = request.get_json(silent=True) or {}

    try:
        if data.get('all'):
            result = install_all_packs(memory.knowledge_base)
            return jsonify({'success': True, **result})
        elif data.get('pack'):
            result = install_pack(memory.knowledge_base, data['pack'])
            return jsonify({'success': True, 'pack': data['pack'], **result})
        else:
            return jsonify({'error': 'Specificare "pack" o "all": true'}), 400
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error("Errore installazione pack: %s", e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/knowledge/stats', methods=['GET'])
@limiter.limit(config.RATE_LIMIT_CONFIG['limits']['default'])
def api_knowledge_stats():
    """Statistiche dettagliate della KB."""
    kb = memory.knowledge_base
    count = kb.get_facts_count()
    with kb._lock:
        rows = kb._conn.execute(
            "SELECT source, COUNT(*) as cnt FROM facts "
            "GROUP BY source ORDER BY cnt DESC"
        ).fetchall()
    sources = {(r[0] or "(nessuna)"): r[1] for r in rows}
    return jsonify({
        'success': True,
        'total_facts': count,
        'sources': sources,
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
        # Aggiorna il router se il cambio è avvenuto
        if success and ROUTER_ENABLED and model_router:
            model_router.warm_model = new_model
            model_router.refresh_installed(ai_engine.list_available_models())
    
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
