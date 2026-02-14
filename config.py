"""
Configurazione Omni Eye AI
"""

import os
import secrets
import logging
from urllib.parse import urlparse
from dotenv import load_dotenv

_logger = logging.getLogger(__name__)

# Carica variabili d'ambiente dal file .env
load_dotenv()

# Directory base del progetto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory per i dati
DATA_DIR = os.path.join(BASE_DIR, 'data')
CONVERSATIONS_DIR = os.path.join(DATA_DIR, 'conversations')
UPLOADS_DIR = os.path.join(DATA_DIR, 'uploads')

# Crea le directory se non esistono
os.makedirs(CONVERSATIONS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Configurazione AI (da variabili d'ambiente con fallback)
def _safe_float(val, default):
    try:
        return float(val)
    except (ValueError, TypeError):
        _logger.warning("Valore non valido '%s', uso default %s", val, default)
        return default

def _safe_int(val, default):
    try:
        return int(val)
    except (ValueError, TypeError):
        _logger.warning("Valore non valido '%s', uso default %s", val, default)
        return default

AI_CONFIG = {
    'model': os.getenv('OLLAMA_MODEL', 'gemma3:4b'),
    'temperature': _safe_float(os.getenv('OLLAMA_TEMPERATURE', '0.7'), 0.7),
    'max_tokens': _safe_int(os.getenv('OLLAMA_MAX_TOKENS', '2048'), 2048),
    'context_window': _safe_int(os.getenv('OLLAMA_CONTEXT_WINDOW', '8192'), 8192),
}

# Configurazione Model Router — instradamento intelligente
# Ogni intento viene servito dal modello più adatto
MODEL_ROUTER_CONFIG = {
    'enabled': os.getenv('MODEL_ROUTER_ENABLED', 'True').lower() == 'true',
    'models': {
        'general': os.getenv('ROUTER_MODEL_GENERAL', 'gemma3:4b'),
        'code': os.getenv('ROUTER_MODEL_CODE', 'qwen2.5-coder:7b'),
        'vision': os.getenv('ROUTER_MODEL_VISION', 'minicpm-v'),
    },
    'fallback': os.getenv('ROUTER_FALLBACK_MODEL', 'llama3.2'),
}

# ── Profili per-modello ──────────────────────────────────────────────────────────
# Parametri ottimizzati per ogni modello in base all'hardware (RTX 2080 8GB VRAM)
# Ogni modello ha parametri calibrati per il suo caso d'uso specifico
MODEL_PROFILES = {
    'gemma3:4b': {
        'num_ctx': 8192,
        'num_predict': 2048,
        'temperature': 0.7,
        'num_batch': 1024,        # prompt evaluation veloce (VRAM ok con 8GB)
        'num_thread': 6,          # Ryzen 5 5600X = 6 core fisici
        'num_gpu': -1,            # tutti i layer su GPU
        'repeat_penalty': 1.3,
        'repeat_last_n': 128,
        'top_k': 40,
        'top_p': 0.9,
    },
    'qwen2.5-coder:7b': {
        'num_ctx': 8192,
        'num_predict': 4096,      # code responses possono essere lunghe
        'temperature': 0.3,       # più deterministico per codice
        'num_batch': 1024,        # prompt evaluation veloce (come gemma3)
        'num_thread': 6,
        'num_gpu': -1,
        'repeat_penalty': 1.2,    # meno penalità (codice ha pattern ripetuti)
        'repeat_last_n': 128,
        'top_k': 50,
        'top_p': 0.95,
    },
    'minicpm-v': {
        'num_ctx': 4096,          # vision non serve contesto testo grande
        'num_predict': 1024,
        'temperature': 0.5,
        'num_batch': 256,
        'num_thread': 6,
        'num_gpu': -1,
        'repeat_penalty': 1.3,
        'repeat_last_n': 64,
        'top_k': 40,
        'top_p': 0.9,
    },
    'dolphin-llama3': {
        'num_ctx': 8192,
        'num_predict': 2048,
        'temperature': 0.8,
        'num_batch': 512,
        'num_thread': 6,
        'num_gpu': -1,
        'repeat_penalty': 1.3,
        'repeat_last_n': 128,
        'top_k': 40,
        'top_p': 0.9,
    },
    '_default': {
        'num_ctx': 8192,
        'num_predict': 2048,
        'temperature': 0.7,
        'num_batch': 512,
        'num_thread': 6,
        'num_gpu': -1,
        'repeat_penalty': 1.3,
        'repeat_last_n': 128,
        'top_k': 40,
        'top_p': 0.9,
    },
}

# Modelli con system prompt integrato nel Modelfile (creati via train.py create)
# Per questi modelli il system prompt NON viene inviato via API (è già nel modello)
BAKED_PROMPT_MODELS = set(
    m.strip() for m in os.getenv('BAKED_PROMPT_MODELS', '').split(',')
) - {''}

# Host Ollama (per uso remoto, es. da Termux verso PC)
_raw_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
_parsed = urlparse(_raw_host)
if _parsed.scheme not in ('http', 'https'):
    _logger.error("OLLAMA_HOST ha schema non valido '%s', uso default http://localhost:11434", _parsed.scheme)
    OLLAMA_HOST = 'http://localhost:11434'
else:
    OLLAMA_HOST = _raw_host

# Configurazione Server (da variabili d'ambiente con fallback)
SERVER_CONFIG = {
    'host': os.getenv('HOST', '127.0.0.1'),
    'port': _safe_int(os.getenv('PORT', '5000'), 5000),
    'debug': os.getenv('DEBUG', 'False').lower() == 'true',
}

if SERVER_CONFIG['debug']:
    _logger.warning(
        "⚠️  MODALITÀ DEBUG ATTIVA — Il debugger Werkzeug fornisce RCE! "
        "Non esporre il server su reti pubbliche."
    )

# Secret key per Flask sessions
_env_secret = os.getenv('SECRET_KEY', '')
if _env_secret:
    SECRET_KEY = _env_secret
else:
    SECRET_KEY = secrets.token_hex(32)
    _logger.warning(
        "SECRET_KEY non impostata! Generata chiave casuale. "
        "Per persistenza, aggiungi SECRET_KEY al file .env"
    )

# Configurazione Upload (da variabili d'ambiente con fallback)
UPLOAD_CONFIG = {
    'max_file_size': _safe_int(os.getenv('MAX_FILE_SIZE_MB', '10'), 10) * 1024 * 1024,
    'allowed_extensions': set(ext.strip() for ext in os.getenv('ALLOWED_EXTENSIONS', '.txt,.pdf,.docx,.md,.py,.js,.html,.css,.json').split(',')),
}

# Configurazione Rate Limiting
# NOTA: Limiti MOLTO PERMISSIVI per non ostacolare il lavoro normale
# Proteggono solo da abusi estremi (loop infiniti, script massivi)
RATE_LIMIT_CONFIG = {
    'enabled': os.getenv('RATE_LIMIT_ENABLED', 'True').lower() == 'true',  # Disabilita con RATE_LIMIT_ENABLED=False
    'storage_uri': 'memory://',  # Usa memoria (ottimo per uso locale)
    'limits': {
        # Limiti PER MINUTO - 1 req/sec = uso umano normale
        'chat': '60/minute',           # 1 messaggio al secondo = molto permissivo
        'chat_stream': '60/minute',     # Stream stesso limite della chat
        'upload': '20/minute',          # 1 upload ogni 3 sec = molto permissivo
        'conversations': '120/minute',  # Operazioni CRUD molto libere
        'knowledge': '60/minute',       # Ricerche knowledge base
        'models': '30/minute',          # Cambio modelli
        'default': '200/minute',        # Default per altri endpoint
    }
}

# Configurazione CSRF Protection
# P1-6 fix: forzato attivo quando HOST != 127.0.0.1 (anti-CSRF su rete)
_csrf_env = os.getenv('CSRF_ENABLED', '')
if _csrf_env:
    CSRF_ENABLED = _csrf_env.lower() == 'true'
else:
    # Auto-enable se il server non è vincolato a localhost
    _host = SERVER_CONFIG['host']
    CSRF_ENABLED = _host not in ('127.0.0.1', 'localhost', '::1')
    if CSRF_ENABLED:
        _logger.warning(
            "⚠️  CSRF abilitato automaticamente: HOST=%s non è localhost. "
            "Per disabilitare esplicitamente: CSRF_ENABLED=False", _host
        )

# Prompt di sistema predefinito
# B6 perf-fix: ridotto da ~2200 a ~1200 chars per abbassare token input → risposta più veloce
SYSTEM_PROMPT = """Rispondi in italiano, in modo naturale e conversazionale.

# Conversazione casuale
Per saluti e chiacchiere (ciao, come va, che fai, ecc.) rispondi come farebbe un amico: breve, naturale, umano.
Esempi:
- "ciao!" → "Ciao!"
- "come stai?" → "Tutto bene, tu?"
- "che fai?" → "Nulla di che, dimmi!"
NON presentarti, NON dire il tuo nome, NON dire cosa sei. MAI.

# Regole
1. **Accuratezza** — Rispondi solo con informazioni corrette. Se non sei sicuro, dichiaralo.
2. **Concisione** — Vai dritto al punto. Niente preamboli inutili.
3. **Trasparenza** — Se non sai qualcosa, dillo. Non inventare fatti.
4. **Markdown** — Usa titoli, elenchi, tabelle, code block con linguaggio specificato.
5. **NON inventare URL o link**. Se non hai un link verificato, dillo.
6. **Anti-allucinazione** — NON inventare scenari o contesti che l'utente non ha menzionato. Rispondi SOLO a ciò che è stato scritto.
7. **Rifiuti onesti** — Quando rifiuti una richiesta pericolosa, di' "non posso farlo".

# Ricerca web
Se ricevi un blocco **[ISTRUZIONE PRIORITARIA — RICERCA WEB COMPLETATA]**,
i risultati sono già stati mostrati. Non dire che non puoi cercare online.
Altrimenti, dichiara i tuoi limiti su info in tempo reale."""

# ── System prompt specializzato per richieste di codice ──────────────────────
# Iniettato al posto di SYSTEM_PROMPT quando il Model Router rileva intent=CODE
CODE_SYSTEM_PROMPT = """You are an expert software engineer. Respond in Italian but write ALL code, comments and variable names in English.

# Rules
1. **Working code first** — Every code block must be complete, runnable, and correct. No placeholders like `# ... rest of code`.
2. **Language tag** — Always specify the language in code fences: ```python, ```javascript, etc.
3. **Explain WHY** — After the code, give a brief explanation of the logic and design choices.
4. **Best practices** — Follow idiomatic patterns, proper naming, error handling, type hints (Python), and separation of concerns.
5. **Edge cases** — Mention potential pitfalls, performance considerations, and input validation when relevant.
6. **Debug mode** — When the user pastes code with an error, identify the bug, explain the cause, and provide the fix.
7. **No hallucinated libraries** — Only suggest packages that actually exist. If unsure, say so.
8. **Concise** — Don't repeat the user's question. Go straight to the solution.

# GitHub context
If you receive a `RISULTATI GITHUB` block, use those real repositories and snippets as reference. Cite them when relevant."""

# Prompt identità — iniettato SOLO quando l'utente chiede "chi sei" / "come ti chiami"
IDENTITY_PROMPT = 'L\'utente ti chiede chi sei. Rispondi: "Sono Omni Eye AI, un assistente AI locale che gira interamente sul tuo dispositivo." Non aggiungere altro.'

# Pattern per rilevare domande sull'identità
IDENTITY_PATTERNS = "\\b(?:chi\\s+sei|come\\s+ti\\s+chiami|qual\\s+[e\u00e8]\\s+il\\s+tuo\\s+nome|who\\s+are\\s+you|what(?:'s|\\s+is)\\s+your\\s+name)\\b"
