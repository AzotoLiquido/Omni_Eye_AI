# Omni Eye AI - Assistente AI Locale

🤖 Assistente AI completamente locale e privato, senza limiti di utilizzo.

## 🌟 Caratteristiche

- ✅ **Completamente Locale**: Nessun dato esce dal tuo PC
- ✅ **Gratuito e Illimitato**: Nessun token o costo di utilizzo
- ✅ **Memoria Conversazioni**: L'AI ricorda le chat precedenti
- ✅ **Analisi Documenti**: Carica e analizza file PDF, TXT, DOCX
- ✅ **Interfaccia Web Modern**: GUI intuitiva e bella
- ✅ **Modelli Potenti**: LLaMA, Mistral, e altri modelli open source

## 📋 Requisiti

- Python 3.8+
- Ollama (per i modelli AI)
- 8GB+ RAM (16GB consigliato)
- 10GB+ spazio disco per i modelli

## 🚀 Installazione Rapida

1. **Installa Ollama** (se non l'hai già):
   - Scarica da: https://ollama.ai/download
   - Oppure esegui: `winget install Ollama.Ollama`

2. **Installa dipendenze Python**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Scarica un modello AI** (prima volta):
   ```bash
   ollama pull llama3.2
   ```

4. **Avvia l'applicazione**:
   ```bash
   python start.py
   ```

5. **Apri il browser** e vai su: http://localhost:5000

## 🎯 Modelli Consigliati

- `llama3.2` (3B) - Veloce, buono per PC normali
- `mistral` (7B) - Ottimo bilanciamento qualità/velocità
- `llama3.1` (8B) - Molto intelligente
- `codellama` - Specializzato per codice

Per cambiare modello, modifica `config.py`

## 📁 Struttura Progetto

```
Omni_Eye_AI/
├── app/
│   ├── static/      # CSS, JS, immagini
│   ├── templates/   # HTML templates
│   └── main.py      # Backend Flask
├── core/
│   ├── ai_engine.py    # Motore AI Ollama
│   ├── memory.py       # Sistema memoria
│   └── document_processor.py  # Analisi documenti
├── data/
│   ├── conversations/  # Chat salvate
│   └── uploads/       # File caricati
├── config.py          # Configurazione
├── requirements.txt   # Dipendenze
└── start.py          # Script avvio

```

## 💡 Utilizzo

1. **Chat Normale**: Scrivi messaggi come in ChatGPT
2. **Carica Documenti**: Click su 📎 per analizzare file
3. **Cronologia**: Accedi alle conversazioni passate
4. **Nuova Chat**: Inizia conversazione fresca

## 🔒 Privacy

Tutto rimane sul tuo PC:
- Modelli AI scaricati localmente
- Conversazioni salvate in `data/conversations/`
- Documenti processati in `data/uploads/`
- Nessuna connessione a server esterni (dopo download modelli)

## ⚙️ Configurazione

Modifica `config.py` per:
- Cambiare modello AI
- Regolare temperatura (creatività)
- Cambiare porta server
- Personalizzare prompt di sistema

## 🆘 Risoluzione Problemi

**Ollama non trovato?**
- Assicurati che Ollama sia installato e in esecuzione
- Controlla: `ollama list`

**Modello non trovato?**
- Scarica: `ollama pull llama3.2`

**Errori di memoria?**
- Usa un modello più piccolo (llama3.2:1b)
- Chiudi altre applicazioni

## 📝 Licenza

MIT - Usa liberamente!
