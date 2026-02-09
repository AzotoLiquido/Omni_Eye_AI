"""
Configurazione Omni Eye AI
"""

import os
import secrets
import logging
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
AI_CONFIG = {
    'model': os.getenv('OLLAMA_MODEL', 'llama3.2'),
    'temperature': float(os.getenv('OLLAMA_TEMPERATURE', '0.7')),
    'max_tokens': int(os.getenv('OLLAMA_MAX_TOKENS', '2048')),
    'context_window': int(os.getenv('OLLAMA_CONTEXT_WINDOW', '4096')),
}

# Host Ollama (per uso remoto, es. da Termux verso PC)
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')

# Configurazione Server (da variabili d'ambiente con fallback)
SERVER_CONFIG = {
    'host': os.getenv('HOST', '127.0.0.1'),
    'port': int(os.getenv('PORT', '5000')),
    'debug': os.getenv('DEBUG', 'False').lower() == 'true',
}

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
    'max_file_size': int(os.getenv('MAX_FILE_SIZE_MB', '10')) * 1024 * 1024,
    'allowed_extensions': set(os.getenv('ALLOWED_EXTENSIONS', '.txt,.pdf,.docx,.md,.py,.js,.html,.css,.json').split(',')),
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
# DISABILITATO: Non serve per API REST (già protette da CORS)
CSRF_ENABLED = os.getenv('CSRF_ENABLED', 'False').lower() == 'true'  # Default False

# Prompt di sistema predefinito
SYSTEM_PROMPT = """Sei un assistente AI versatile e competente.

IMPORTANTE - LINGUA:
- Rispondi SEMPRE e SOLO in italiano corretto
- NON mescolare mai italiano e inglese
- NON usare parole inglesi se esiste l'equivalente italiano
- Mantieni coerenza linguistica in tutte le risposte

IMPORTANTE - COMPORTAMENTO:
- Rispondi a qualsiasi domanda in modo diretto e onesto
- Sii chiaro, preciso e utile nelle tue risposte
- Fornisci informazioni accurate e ben strutturate
- Se non sei sicuro di qualcosa, dillo chiaramente"""
