# Audit Completo — Core Modules

**File analizzati**: `ai_engine.py` (310 LOC), `memory.py` (329 LOC), `advanced_memory.py` (644 LOC), `document_processor.py` (232 LOC), `web_search.py` (543 LOC)  
**Totale**: 5 file, ~2058 righe  
**Data**: 2025-02-10  
**Punteggio iniziale**: 7.8 / 10

---

## Riepilogo Severità

| Sev. | # | Descrizione breve |
|------|---|-------------------|
| P0   | 0 | — |
| P1   | 6 | DRY, eager init, context overflow, per-call alloc, stima token, regex fragile |
| P2   | 8 | Race condition, troncamento silenzioso, costanti locali, cap mancante, interessi limitati, PyPDF2 deprecato, empty data, query biased |

---

## Dettaglio Findings

### P1 — Alta Priorità

| ID | File | Riga | Descrizione |
|----|------|------|-------------|
| P1-1 | ai_engine.py | 113-175, 185-250 | **DRY violation** — La costruzione `messages[]` (system prompt → history → user msg + images) è duplicata identica tra `generate_response` e `generate_response_stream`. ~20 righe. → Estrarre `_build_messages()`. |
| P1-2 | ai_engine.py | 14 | **Eager init module-level** — `_ollama_client = ollama.Client(host=...)` viene creato all'import time. Se `OLLAMA_HOST` cambia o Ollama è spento, la config è congelata. → Lazy factory con cache. |
| P1-3 | advanced_memory.py | 97-113 | **Context overflow in `_generate_summary`** — `conversation_text` concatena TUTTI i vecchi messaggi senza limite. Con 200+ messaggi il prompt supera la context window del modello. → Troncare a `max_context_tokens` chars. |
| P1-4 | advanced_memory.py | 185-195 | **`STOP_NAMES` set ricostruito ad ogni chiamata** a `_extract_names`. Alloca un `set` di 28 stringhe per ogni messaggio utente. → `frozenset` class-level. |
| P1-5 | advanced_memory.py | 30-36 | **`estimate_tokens` incoerente** — Il commento dice "~1.3 tokens per word" ma il codice non applica il moltiplicatore: ritorna `max(words, len(text)//4)`. Per testi italiani lunghi, sottostima i token. → Applicare `int(words * 1.3)`. |
| P1-6 | web_search.py | 271-279 | **`_replace_md_link` — doppio parsing regex** fragile. L'URL viene ri-estratto con `re.search()` dentro il callback. Se `_MD_LINK_RE` cambia formato, il callback si rompe silenziosamente. → Aggiungere capture group per URL direttamente nella regex. |

### P2 — Media Priorità

| ID | File | Riga | Descrizione |
|----|------|------|-------------|
| P2-1 | ai_engine.py | 297-310 | **`change_model` race condition** — Read-set-check-rollback su `self.model` senza lock. Due thread concorrenti possono corrompersi. → `threading.Lock`. |
| P2-2 | ai_engine.py | 256-275 | **`analyze_document` troncamento silenzioso** — Quando `document_text > ANALYZE_DOC_MAX_CHARS`, il modello riceve il testo tagliato senza alcuna indicazione. → Aggiungere nota `[documento troncato a X caratteri]`. |
| P2-3 | memory.py | 90 | **`_MAX_MESSAGES = 500` costante locale** nel body di `add_message`. Ricompilata ad ogni chiamata. → Costante class-level. |
| P2-4 | advanced_memory.py | 290-307 | **`get_relevant_entities` nessun cap globale** — Il cap a 8 si applica solo alle preferenze. Se molti nomi matchano, la lista cresce senza limite. → Cap globale (es. 10). |
| P2-5 | advanced_memory.py | 408-413 | **`_extract_interests` troppo ristretto** — Gestisce solo "mi interessa" ma il caller controlla anche "piace", "passione". Questi ultimi non vengono estratti. → Aggiungere pattern per "mi piace" e "passione". |
| P2-6 | document_processor.py | 18-19 | **`PyPDF2` deprecato** — Il pacchetto è stato rinominato in `pypdf`. PyPDF2 non riceve più aggiornamenti di sicurezza. → Migrare a `pypdf` (API identica). |
| P2-7 | document_processor.py | 156-157 | **`save_upload` accetta `file_data` vuoto** — `len(file_data) == 0` non viene controllato. Crea un file vuoto su disco. → Validare lunghezza > 0. |
| P2-8 | web_search.py | 342-347 | **Factual mode DDG query biased** — `web_search("wikipedia " + clean_q)` aggiunge "wikipedia" alla query, biasando i risultati. Se l'API Wikipedia ha già restituito contenuto, questa ricerca è ridondante. → Usare query pulita; fare DDG solo come fallback se wiki_extract è None. |

---

## Punteggio post-fix stimato: 9.0 / 10
