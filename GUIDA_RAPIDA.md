# 🚀 Guida Rapida - Omni Eye AI

## Installazione Prime Volte

### 1. Installa Ollama
```powershell
# Metodo 1: Con winget
winget install Ollama.Ollama

# Metodo 2: Download manuale
# Vai su https://ollama.ai/download
```

### 2. Installa Dipendenze Python
```powershell
pip install -r requirements.txt
```

### 3. Scarica un Modello AI
```powershell
# Modello leggero (consigliato per iniziare)
ollama pull llama3.2

# Oppure uno più potente
ollama pull mistral
```

### 4. Avvia l'Applicazione
```powershell
python start.py
```

Poi apri il browser su: **http://localhost:5000**

---

## Uso Quotidiano

Ogni volta che vuoi usare Omni Eye AI:

1. Assicurati che Ollama sia in esecuzione (si avvia automaticamente di solito)
2. Esegui: `python start.py`
3. Apri il browser su http://localhost:5000

---

## Comandi Utili

### Gestione Modelli

```powershell
# Lista modelli installati
ollama list

# Scarica un nuovo modello
ollama pull <nome-modello>

# Rimuovi un modello
ollama rm <nome-modello>

# Verifica che Ollama funzioni
ollama run llama3.2 "Ciao!"
```

### Modelli Consigliati

| Modello | Dimensione | RAM | Velocità | Qualità |
|---------|-----------|-----|----------|---------|
| llama3.2:1b | 1.3GB | 4GB | ⚡⚡⚡ | ⭐⭐ |
| llama3.2 | 2GB | 8GB | ⚡⚡⚡ | ⭐⭐⭐ |
| mistral | 4.1GB | 8GB | ⚡⚡ | ⭐⭐⭐⭐ |
| llama3.1 | 4.7GB | 16GB | ⚡⚡ | ⭐⭐⭐⭐ |
| codellama | 3.8GB | 8GB | ⚡⚡ | ⭐⭐⭐⭐ (per codice) |

### Test Rapido
```powershell
# Test AI engine
python core/ai_engine.py

# Test memoria
python core/memory.py

# Test documenti
python core/document_processor.py
```

---

## Problemi Comuni

### ❌ "Ollama non disponibile"

**Soluzione:**
1. Verifica che Ollama sia installato: `ollama --version`
2. Se non è installato: `winget install Ollama.Ollama`
3. Riavvia il terminale

### ❌ "Modello non trovato"

**Soluzione:**
```powershell
ollama pull llama3.2
```

### ❌ "Errore memoria" / Computer lento

**Soluzione:**
- Usa un modello più piccolo: `ollama pull llama3.2:1b`
- Modifica `config.py` per impostare il modello più piccolo
- Chiudi altre applicazioni

### ❌ "Porta 5000 già in uso"

**Soluzione:**
1. Apri `config.py`
2. Cambia `'port': 5000` in `'port': 5001` (o altro numero)
3. Riavvia l'app

---

## Configurazione Avanzata

### Cambia Modello AI

Modifica `config.py`:
```python
AI_CONFIG = {
    'model': 'mistral',  # Cambia qui
    'temperature': 0.7,
    'max_tokens': 2048,
}
```

### Personalizza Comportamento AI

In `config.py`, modifica `SYSTEM_PROMPT`:
```python
SYSTEM_PROMPT = """Il tuo prompt personalizzato qui..."""
```

### Cambia Porta del Server

In `config.py`:
```python
SERVER_CONFIG = {
    'host': '0.0.0.0',
    'port': 8080,  # Cambia qui
    'debug': False,  # False in produzione
}
```

---

## Backup dei Dati

I tuoi dati sono salvati in:
- **Conversazioni:** `data/conversations/`
- **Upload:** `data/uploads/`

Per fare backup:
```powershell
# Copia l'intera cartella data
Copy-Item -Path "data" -Destination "backup/data_$(Get-Date -Format 'yyyyMMdd')" -Recurse
```

---

## Caratteristiche

✅ **Chat Intelligente** - Conversazioni naturali con memoria  
✅ **Analisi Documenti** - Carica PDF, DOCX, TXT e analizzali  
✅ **100% Locale** - Tutto resta sul tuo PC, zero cloud  
✅ **Illimitato** - Nessun costo o limite di utilizzo  
✅ **Modelli Potenti** - LLaMA, Mistral e altri modelli open source  
✅ **Interfaccia Moderna** - GUI web bella e intuitiva  

---

## Aggiornamenti

Per aggiornare Omni Eye AI:
```powershell
git pull origin main
pip install -r requirements.txt --upgrade
```

Per aggiornare i modelli:
```powershell
ollama pull <nome-modello>
```

---

## Supporto

- **Documentazione:** Leggi [README.md](README.md)
- **Problemi:** Controlla i log nel terminale
- **Test:** Esegui gli script di test in `core/`

---

## Licenza

MIT License - Usa liberamente!

Creato con ❤️ per la privacy e la libertà dell'AI
