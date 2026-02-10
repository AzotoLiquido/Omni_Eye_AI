# Audit Completo Omni Eye AI — v2

**Data**: 2025-02-09  
**Scope**: Tutte le aree del progetto  
**Totale issues**: 127 (12 P0, 19 P1, 54 P2, 42 P3)

---

## Riepilogo per Severità

| Sev | Count | Descrizione |
|-----|-------|-------------|
| **P0** | 12 | Crash, sicurezza, RCE, XSS, path traversal |
| **P1** | 19 | Funzionalità rotte, data loss, parsing errato |
| **P2** | 54 | Qualità, robustezza, race condition, info disclosure |
| **P3** | 42 | Polish, UX, accessibilità, codice morto |

---

## 1. Core AI Engine (`ai_engine.py`)

✅ **Già fixato nelle sessioni precedenti.** Un solo issue residuo:

| # | Sev | Issue |
|---|-----|-------|
| 1 | P3 | Import `deque` inutilizzato (residuo di refactor precedente) |

---

## 2. Memory & Context (`memory.py`, `advanced_memory.py`, `document_processor.py`, `__init__.py`)

### P1 — Critico
| # | File | Linea | Issue |
|---|------|-------|-------|
| 1 | advanced_memory.py | ~L180 | `update_from_conversation` double-count dei topic: chiamata ogni 5 msg con TUTTI i messaggi, i contatori si incrementano senza reset |
| 2 | advanced_memory.py | ~L220 | `compress_context` viola silenziosamente il token limit quando ci sono ≤6 messaggi |

### P2 — Qualità
| # | File | Linea | Issue |
|---|------|-------|-------|
| 3 | advanced_memory.py | ~L310 | `last_updated` mutato fuori dal lock |
| 4 | advanced_memory.py | ~L180 | `update_from_conversation` modifica knowledge sub-dict senza lock |
| 5 | advanced_memory.py | — | `extract_and_save` gap di lock tra estrazione e salvataggio |
| 6 | document_processor.py | — | `save_upload` nessun check dimensione file prima della scrittura su disco |
| 7 | document_processor.py | — | TOCTOU race nella logica anti-overwrite |
| 8 | document_processor.py | — | `clean_old_uploads` può eliminare file in lettura |
| 9 | memory.py | — | `list_all_conversations` nessun caching (O(n) I/O) |
| 10 | memory.py | — | `search_conversations` carica tutti i file |

### P3 — Polish
| # | File | Issue |
|---|------|-------|
| 11 | memory.py | Doppia `datetime.now()` |
| 12 | memory.py | Nessun limite messaggi per conversazione |
| 13 | advanced_memory.py | Stima token grezza (1:4 chars) |
| 14 | advanced_memory.py | `_extract_names` regex con IGNORECASE inutile |
| 15 | document_processor.py | `errors='ignore'` scarta byte silenziosamente |
| 16 | document_processor.py | `st_ctime` vs `st_mtime` inconsistenza |
| 17 | document_processor.py | `max_name_len` negativo edge case |
| 18 | `__init__.py` | Import eager di `AIEngine` |
| 19 | `__init__.py` | `__all__` non include `AdvancedMemory` |

---

## 3. AI-Pilot Subsystem

### P0 — Sicurezza Critica
| # | File | Linea | Issue |
|---|------|-------|-------|
| 1 | prompt_builder.py | L205 | **Prompt injection**: `user_message` iniettato raw nel prompt di estrazione fatti |
| 2 | prompt_builder.py | L175 | **Prompt injection**: fatti memorizzati (potenzialmente attacker-controlled) iniettati nel system prompt |
| 3 | tool_executor.py | L87 | **Sandbox bypass**: blocklist regex per Python trivialmente aggirabile (`__builtins__`, `compile()`) |
| 4 | tool_executor.py | L94 | **Env leak**: `PYTHONPATH`/`VIRTUAL_ENV` passati al subprocess sandboxed |
| — | Cross-cutting | — | **Pipeline completa**: utente → fact extraction → memory → system prompt = injection persistente |

### P1 — Funzionalità Rotte
| # | File | Linea | Issue |
|---|------|-------|-------|
| 5 | pilot.py | L197 | `process_stream()`: codice post-yield non eseguito se generatore abbandonato |
| 6 | pilot.py | L192 | Doppia estrazione fatti (`process_stream` thread + inline) |
| 7 | planner.py | L36 | Regex `RE_ACTION`: parentesi nei params JSON troncano il match |
| 8 | planner.py | L128 | `execute_step` converte `ToolResult` a stringa, perde `.success` |
| 9 | tool_executor.py | L61 | `_resolve_safe_path` vulnerabile a symlink + case-insensitive bypass (Windows) |
| 10 | tool_executor.py | L126 | `shlex.split` in modalità POSIX su Windows — backslash errati |
| 11 | config_loader.py | L14 | Fallback percorso config senza check esistenza file |
| 12 | audit_logger.py | L108 | **Deadlock**: `_flush_buffer` richiamato con lock non rientrante |
| 13 | memory_store.py | L30 | SQLite `check_same_thread=False` ma read senza lock |
| 14 | memory_store.py | L117 | FTS5 query injection (operatori `*`, `OR`, `NOT` causano crash) |
| 15 | memory_store.py | L196 | Stessa FTS5 injection in `search_documents` |
| — | Cross-cutting | — | `max_tool_calls` definito in config ma mai verificato nel ReAct loop |

### P2 — Qualità/Robustezza (18 issues)
Principali: race condition `ai_engine` condiviso, `accumulated_context` senza limite, errore detection via stringa "ERRORE", `_parse_params` corrompe apostrofi JSON, reload config non propagato ai sub-system.

### P3 — Polish (9 issues)
Principali: `metadata["memory_retrieved"]` always True, `_simulate_stream` perde whitespace, `PlanStep` non esportato, `_STR_MAP` riallocato ad ogni accesso proprietà.

---

## 4. Flask Backend (`main.py`)

### P0 — Sicurezza
| # | Linea | Issue |
|---|-------|-------|
| 1 | L583 | **Info disclosure**: SSE error event invia `str(e)` raw al client |
| 2 | L296, L739 | **Info disclosure**: error handler 500 invia eccezione raw |
| 3 | L605, L614 | **Path traversal**: `conv_id` non validato, usato per costruire path file |
| 4 | L940 | **Dev note leak**: `dev_note` esposto a tutti, non solo in debug |

### P1 — Funzionalità Rotte
| # | Linea | Issue |
|---|-------|-------|
| 5 | L261+ | `request.json` può essere `None` → `AttributeError` su molti endpoint |
| 6 | L488+ | SSE newline injection: `\n\n` nel chunk termina l'evento prematuramente |
| 7 | L843 | Race condition: `change_model()` muta stato globale condiviso |
| 8 | L585 | Eccezione mid-stream lascia conversazione in stato inconsistente |
| 9 | L810 | `api_optimize_memory` crash se conversazione non trovata |
| 10 | L263 | `conv_id` dal JSON body non validato (path traversal) |

### P2 — Qualità (13 issues)
Principali: filepath disclosure in upload, base64 validation incompleta, nessun rate limit su endpoint distruttivi, MAX_CONTENT_LENGTH inconsistente con upload config, file.read() carica tutto in memoria.

### P3 — Polish (4 issues)
Principali: import base64 nel loop, Connection: keep-alive mancante.

---

## 5. Frontend (`script.js`, `style.css`, `index.html`, `knowledge.html`)

### P0 — XSS
| # | File | Linea | Issue |
|---|------|-------|-------|
| 1 | knowledge.html | L~497 | `JSON.stringify(user_profile)` dentro `innerHTML` di un `<pre>` — XSS stored |
| 2 | knowledge.html | L414 | Entity counts interpolati in template literal + `innerHTML` |

### P1 — Funzionalità Rotte
| # | File | Linea | Issue |
|---|------|-------|-------|
| 3 | script.js | L519 | SSE parser: chunk boundary spezza le righe → caratteri persi |
| 4 | script.js | L518 | `TextDecoder` senza `stream: true` → caratteri multi-byte corrotti |
| 5 | script.js | L477 | `quickPromptsWrapper` ID inesistente — hide/show è codice morto |
| 6 | knowledge.html | L9 | CSS variables (`--primary`, `--bg-primary`...) non definite in style.css → pagina non stilata |
| 7 | script.js | L882 | `currentUploadedFile` settato ma mai consumato da `sendMessage()` |
| 8 | script.js | L308 | Fetch calls non verificano `response.ok` prima di `.json()` |

### P2 — Qualità (16 issues)
Principali: memory leak blob URL, nessun guard concorrenza send, nessun AbortController, ARIA non sincronizzato, CSS duplicato, DOM churn del grid orb, keyboard navigation rotta.

### P3 — Polish (11 issues)
Principali: nessun `prefers-reduced-motion`, CSS inutilizzato, toast senza `aria-live`, favicon mancante.

---

## 6. Config & Security (`config.py`, `start.py`, `requirements.txt`)

### P0 — Sicurezza
| # | File | Linea | Issue |
|---|------|-------|-------|
| 1 | config.py | L39 | **SSRF**: `OLLAMA_HOST` non validato — URL arbitrario passato al client Ollama |
| 2 | config.py | L45 | **RCE**: `DEBUG=true` abilita Werkzeug debugger interattivo (RCE) |

### P1 — Funzionalità Rotte
| # | File | Linea | Issue |
|---|------|-------|-------|
| 3 | config.py | L31 | `float()`/`int()` su env var crash su input invalido |
| 4 | start.py | L42 | Check dipendenze incompleto — mancano flask_limiter, flask_wtf, dotenv |
| 5 | start.py | L139 | `suggest_model_download` usa `ollama` bare, non il path scoperto |

### P2 — Qualità (6 issues)
Principali: dipendenze parzialmente pinnate, allowed_extensions non strippa whitespace, rate limit storage in-memory, SECRET_KEY transitoria.

### P3 — Polish (3 issues)
Principali: commento CSRF impreciso, test commentati, nessun `.env.example`.

---

## Piano Fix

### Fase 1 — P0 (12 issues critici)
1. XSS knowledge.html (→ `textContent`)
2. SSE error info disclosure (→ messaggio generico)
3. Error handler info disclosure (→ messaggio generico)
4. Path traversal conv_id (→ regex validation)
5. OLLAMA_HOST SSRF (→ URL scheme validation)
6. DEBUG mode warning
7. Rate limit dev_note (→ solo in debug)
8. Prompt injection fence (→ delimiter + escape)
9. Sandbox Python security (→ AST check + blocklist estesa)

### Fase 2 — P1 (19 issues)
Fix SSE parser, TextDecoder, request.json guard, env var parsing, ecc.

### Fase 3 — P2 selezionati
Le 54 P2 verranno priorizzate per impatto.
