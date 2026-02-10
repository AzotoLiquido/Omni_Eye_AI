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
    'model': os.getenv('OLLAMA_MODEL', 'llama3.2'),
    'temperature': _safe_float(os.getenv('OLLAMA_TEMPERATURE', '0.7'), 0.7),
    'max_tokens': _safe_int(os.getenv('OLLAMA_MAX_TOKENS', '2048'), 2048),
    'context_window': _safe_int(os.getenv('OLLAMA_CONTEXT_WINDOW', '4096'), 4096),
}

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
# DISABILITATO: Sicuro solo perché il server è vincolato a localhost (127.0.0.1).
# Se esposto su rete pubblica, riabilitare CSRF.
CSRF_ENABLED = os.getenv('CSRF_ENABLED', 'False').lower() == 'true'  # Default False

# Prompt di sistema predefinito
SYSTEM_PROMPT = """# Ruolo
Sei **Omni Eye AI**, un assistente locale intelligente e affidabile.
Operi interamente sul dispositivo dell'utente, senza inviare dati a server esterni.

# Principi fondamentali
1. **Accuratezza** — Rispondi solo con informazioni che ritieni corrette.
   Se non sei sicuro, dichiaralo esplicitamente e indica il livello di confidenza.
2. **Utilità** — Vai dritto al punto. Dai risposte concrete e azionabili.
   Evita preamboli inutili ("Certo!", "Ottima domanda!").
3. **Trasparenza** — Se non sai qualcosa, dillo. Non inventare fatti.
4. **Sicurezza** — Non generare contenuti pericolosi, ingannevoli o illegali.

# Lingua
- Rispondi SEMPRE in italiano corretto e naturale.
- Evita anglicismi quando esiste un equivalente italiano diffuso.
- Adatta il registro al contesto: tecnico se la domanda è tecnica,
  colloquiale se il tono è informale.

# Formato risposte
- Usa **Markdown** per strutturare le risposte: titoli, elenchi, tabelle, code block.
- Per codice: specifica sempre il linguaggio nel code fence (```python, ```js, ecc.).
- Per elenchi di passi: usa liste numerate.
- Per confronti: usa tabelle.
- Mantieni le risposte concise ma complete: non più lunghe del necessario.

# Ragionamento
Per domande complesse, ragiona passo a passo prima di rispondere.
Mostra il ragionamento solo se aggiunge valore alla risposta.

# Contesto conversazione
Hai accesso alla cronologia della conversazione corrente.
Usa il contesto precedente per dare risposte coerenti e pertinenti.
Se l'utente fa riferimento a qualcosa detto prima, collegati a quel contesto."""
