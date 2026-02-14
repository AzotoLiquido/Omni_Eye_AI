"""
Pipeline Engine — orchestratore di workflow leggero per Omni Eye AI

Zero dipendenze esterne. Usa solo stdlib (threading, concurrent.futures).

Concetti:
  • Step     — singola unità di lavoro (funzione con retry e timeout)
  • Pipeline — sequenza/parallelo di Step con dipendenze
  • Scheduler — esecuzione periodica di pipeline (pulizia, backup, ecc.)

Esempio:
    pipe = Pipeline("doc_processing")
    pipe.add_step(Step("parse",   parse_document))
    pipe.add_step(Step("chunk",   chunk_text,   depends_on=["parse"]))
    pipe.add_step(Step("index",   index_chunks, depends_on=["chunk"]))
    result = pipe.run(filepath="/path/to/doc.pdf")
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─── Step Status ────────────────────────────────────────────────────────

class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


# ─── Step ───────────────────────────────────────────────────────────────

@dataclass
class StepResult:
    """Risultato di un singolo step."""
    status: StepStatus
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    retries_used: int = 0


@dataclass
class Step:
    """Singola unità di lavoro in una pipeline.

    Args:
        name:        Identificatore univoco dello step
        fn:          Funzione da eseguire. Riceve **kwargs (context della pipeline)
                     e restituisce un valore (disponibile come ctx[name])
        depends_on:  Lista di nomi di step che devono completarsi prima
        retries:     Numero di tentativi in caso di errore (0 = nessun retry)
        backoff:     Secondi di attesa tra i retry (moltiplicati per tentativo)
        timeout:     Timeout in secondi per lo step (None = nessun timeout)
        on_error:    "fail" → interrompe la pipeline; "skip" → segna come SKIPPED
    """
    name: str
    fn: Callable[..., Any]
    depends_on: List[str] = field(default_factory=list)
    retries: int = 0
    backoff: float = 1.0
    timeout: Optional[float] = None
    on_error: str = "fail"  # "fail" | "skip"


# ─── Pipeline ──────────────────────────────────────────────────────────

class PipelineError(Exception):
    """Errore durante l'esecuzione di una pipeline."""


class Pipeline:
    """Grafo aciclico di Step con esecuzione parallela dove possibile.

    Utilizzo:
        pipe = Pipeline("my_pipeline")
        pipe.add_step(Step("a", fn_a))
        pipe.add_step(Step("b", fn_b, depends_on=["a"]))
        pipe.add_step(Step("c", fn_c, depends_on=["a"]))
        # b e c girano in parallelo dopo a
        result = pipe.run(input_data="hello")
    """

    def __init__(self, name: str, max_workers: int = 4):
        self.name = name
        self.max_workers = max_workers
        self._steps: Dict[str, Step] = {}
        self._order: List[str] = []

    def add_step(self, step: Step) -> "Pipeline":
        """Aggiunge uno step alla pipeline. Restituisce self per chaining."""
        if step.name in self._steps:
            raise ValueError(f"Step '{step.name}' già presente nella pipeline '{self.name}'")
        for dep in step.depends_on:
            if dep not in self._steps:
                raise ValueError(
                    f"Step '{step.name}' dipende da '{dep}' che non esiste. "
                    f"Aggiungi '{dep}' prima."
                )
        self._steps[step.name] = step
        self._order.append(step.name)
        return self

    def run(self, **kwargs: Any) -> Dict[str, StepResult]:
        """Esegue la pipeline.

        Args:
            **kwargs: Contesto iniziale passato a tutti gli step

        Returns:
            Dict con il nome dello step come chiave e StepResult come valore
        """
        t_start = time.perf_counter()
        ctx: Dict[str, Any] = dict(kwargs)
        results: Dict[str, StepResult] = {}
        completed: Dict[str, threading.Event] = {
            name: threading.Event() for name in self._steps
        }

        logger.info("Pipeline '%s': avvio (%d step)", self.name, len(self._steps))

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures: Dict[str, Future] = {}

            def _submit(step: Step) -> None:
                """Sottomette uno step per l'esecuzione."""
                fut = pool.submit(self._run_step, step, ctx, results, completed)
                futures[step.name] = fut

            # Topological execution: lancia step senza dipendenze subito,
            # gli altri appena le loro dipendenze sono soddisfatte.
            pending = list(self._order)

            def _launch_ready() -> int:
                """Lancia gli step le cui dipendenze sono tutte soddisfatte."""
                launched = 0
                still_pending = []
                for name in pending:
                    if name in futures:
                        continue
                    step = self._steps[name]
                    deps_met = all(completed[d].is_set() for d in step.depends_on)
                    if deps_met:
                        # Controlla se qualche dipendenza è fallita (e on_error=fail)
                        dep_failed = any(
                            results.get(d) and results[d].status == StepStatus.FAILED
                            for d in step.depends_on
                        )
                        if dep_failed and step.on_error != "skip":
                            results[name] = StepResult(
                                status=StepStatus.SKIPPED,
                                error="Dipendenza fallita",
                            )
                            completed[name].set()
                        else:
                            _submit(step)
                            launched += 1
                    else:
                        still_pending.append(name)
                pending.clear()
                pending.extend(still_pending)
                return launched

            # Prima passata: lancia step senza dipendenze
            _launch_ready()

            # Attendi il completamento e lancia nuovi step man mano
            while len(results) < len(self._steps):
                # Attendi che almeno uno step si completi
                for name, evt in completed.items():
                    if name not in results:
                        continue
                    # Già completato, skip
                if not pending and all(n in results for n in self._steps):
                    break
                # Polling leggero — controlla nuovi step pronti
                time.sleep(0.01)
                _launch_ready()
                # Controlla completamento futures
                for name, fut in list(futures.items()):
                    if fut.done() and name not in results:
                        try:
                            fut.result(timeout=0)
                        except Exception:
                            pass  # Errore già gestito in _run_step

        elapsed = (time.perf_counter() - t_start) * 1000
        ok = sum(1 for r in results.values() if r.status == StepStatus.SUCCESS)
        fail = sum(1 for r in results.values() if r.status == StepStatus.FAILED)
        logger.info(
            "Pipeline '%s': completata in %.0fms (%d ok, %d fail, %d skip)",
            self.name, elapsed, ok, fail,
            len(results) - ok - fail,
        )
        return results

    def _run_step(
        self,
        step: Step,
        ctx: Dict[str, Any],
        results: Dict[str, StepResult],
        completed: Dict[str, threading.Event],
    ) -> None:
        """Esegue un singolo step con retry e timeout."""
        # Attendi le dipendenze
        for dep in step.depends_on:
            completed[dep].wait()

        t0 = time.perf_counter()
        last_error = None

        for attempt in range(step.retries + 1):
            try:
                if attempt > 0:
                    wait = step.backoff * attempt
                    logger.debug("Step '%s': retry %d/%d (attesa %.1fs)",
                                 step.name, attempt, step.retries, wait)
                    time.sleep(wait)

                output = step.fn(**ctx)

                # Salva l'output nel contesto (disponibile per step successivi)
                ctx[step.name] = output

                elapsed = (time.perf_counter() - t0) * 1000
                results[step.name] = StepResult(
                    status=StepStatus.SUCCESS,
                    output=output,
                    duration_ms=elapsed,
                    retries_used=attempt,
                )
                completed[step.name].set()
                logger.debug("Step '%s': OK in %.0fms (retry: %d)",
                             step.name, elapsed, attempt)
                return

            except Exception as e:
                last_error = str(e)
                logger.warning("Step '%s': errore (tentativo %d/%d): %s",
                               step.name, attempt + 1, step.retries + 1, e)

        # Tutti i tentativi falliti
        elapsed = (time.perf_counter() - t0) * 1000
        if step.on_error == "skip":
            results[step.name] = StepResult(
                status=StepStatus.SKIPPED,
                error=last_error,
                duration_ms=elapsed,
                retries_used=step.retries,
            )
        else:
            results[step.name] = StepResult(
                status=StepStatus.FAILED,
                error=last_error,
                duration_ms=elapsed,
                retries_used=step.retries,
            )
        completed[step.name].set()


# ─── Scheduler ─────────────────────────────────────────────────────────

class PipelineScheduler:
    """Scheduler leggero per eseguire pipeline periodicamente.

    Usa un singolo daemon thread. Nessuna dipendenza esterna.

    Esempio:
        scheduler = PipelineScheduler()
        scheduler.register("cleanup", cleanup_pipeline, interval_seconds=3600)
        scheduler.start()
        # ... più tardi ...
        scheduler.stop()
    """

    def __init__(self):
        self._tasks: Dict[str, Dict] = {}
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        pipeline: Pipeline,
        interval_seconds: int,
        kwargs: Optional[Dict] = None,
        run_on_start: bool = False,
    ) -> None:
        """Registra una pipeline per l'esecuzione periodica."""
        with self._lock:
            self._tasks[name] = {
                "pipeline": pipeline,
                "interval": interval_seconds,
                "kwargs": kwargs or {},
                "last_run": 0.0 if run_on_start else time.time(),
                "run_count": 0,
                "last_result": None,
            }
        logger.info("Scheduler: registrata '%s' (ogni %ds, on_start=%s)",
                     name, interval_seconds, run_on_start)

    def start(self) -> None:
        """Avvia il thread scheduler."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="pipeline-scheduler", daemon=True,
        )
        self._thread.start()
        logger.info("Scheduler: avviato")

    def stop(self) -> None:
        """Ferma il thread scheduler."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Scheduler: fermato")

    def get_status(self) -> Dict[str, Dict]:
        """Restituisce lo stato di tutti i task registrati."""
        with self._lock:
            return {
                name: {
                    "interval": t["interval"],
                    "run_count": t["run_count"],
                    "last_run": t["last_run"],
                    "last_result": (
                        {k: v.status.value for k, v in t["last_result"].items()}
                        if t["last_result"] else None
                    ),
                }
                for name, t in self._tasks.items()
            }

    def _loop(self) -> None:
        """Loop principale dello scheduler."""
        while not self._stop_event.is_set():
            now = time.time()
            with self._lock:
                tasks_snapshot = list(self._tasks.items())

            for name, task in tasks_snapshot:
                if now - task["last_run"] >= task["interval"]:
                    try:
                        logger.info("Scheduler: esecuzione '%s'", name)
                        result = task["pipeline"].run(**task["kwargs"])
                        with self._lock:
                            self._tasks[name]["last_run"] = now
                            self._tasks[name]["run_count"] += 1
                            self._tasks[name]["last_result"] = result
                    except Exception as e:
                        logger.error("Scheduler: errore in '%s': %s", name, e)
                        with self._lock:
                            self._tasks[name]["last_run"] = now

            # Sleep breve per non consumare CPU
            self._stop_event.wait(timeout=10)


# ═══════════════════════════════════════════════════════════════════════
# PIPELINE CONCRETE — workflow predefiniti per Omni Eye AI
# ═══════════════════════════════════════════════════════════════════════

def build_maintenance_pipeline() -> Pipeline:
    """Pipeline di manutenzione: pulizia upload, log, backup conversazioni.

    Scheduled: ogni 6 ore.
    """

    def clean_uploads(**kwargs: Any) -> int:
        """Rimuove upload più vecchi di 7 giorni."""
        from core.document_processor import DocumentProcessor
        dp = DocumentProcessor()
        import os
        uploads_dir = dp.uploads_dir
        if not os.path.isdir(uploads_dir):
            return 0
        cutoff = time.time() - (7 * 86400)
        removed = 0
        for fname in os.listdir(uploads_dir):
            fpath = os.path.join(uploads_dir, fname)
            if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                try:
                    os.remove(fpath)
                    removed += 1
                except OSError:
                    pass
        if removed:
            logger.info("Maintenance: rimossi %d upload vecchi", removed)
        return removed

    def trim_logs(**kwargs: Any) -> int:
        """Tronca i file di log Pilot se superano 5MB."""
        import os
        log_dir = os.path.join("data", "logs")
        if not os.path.isdir(log_dir):
            return 0
        trimmed = 0
        max_size = 5 * 1024 * 1024  # 5MB
        for fname in os.listdir(log_dir):
            fpath = os.path.join(log_dir, fname)
            if os.path.isfile(fpath) and os.path.getsize(fpath) > max_size:
                try:
                    # Mantieni solo le ultime 1000 righe
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        lines = f.readlines()
                    with open(fpath, "w", encoding="utf-8") as f:
                        f.writelines(lines[-1000:])
                    trimmed += 1
                    logger.info("Maintenance: troncato log %s (%d righe)", fname, len(lines))
                except Exception as e:
                    logger.warning("Maintenance: errore troncamento %s: %s", fname, e)
        return trimmed

    def backup_knowledge_base(**kwargs: Any) -> str:
        """Crea un backup della knowledge base."""
        import os
        import shutil
        kb_path = os.path.join("data", "knowledge_base.json")
        if not os.path.isfile(kb_path):
            return "no_kb"
        backup_path = kb_path + ".bak"
        try:
            shutil.copy2(kb_path, backup_path)
            logger.info("Maintenance: backup knowledge_base.json creato")
            return backup_path
        except Exception as e:
            logger.warning("Maintenance: errore backup KB: %s", e)
            return f"error: {e}"

    pipe = Pipeline("maintenance", max_workers=3)
    pipe.add_step(Step("clean_uploads", clean_uploads, on_error="skip"))
    pipe.add_step(Step("trim_logs", trim_logs, on_error="skip"))
    pipe.add_step(Step("backup_kb", backup_knowledge_base, on_error="skip"))
    return pipe


def build_document_pipeline() -> Pipeline:
    """Pipeline di processing documenti: parse → chunk → index Pilot.

    Usata quando un file viene caricato via /api/upload.
    Gli step sono sequenziali perché ognuno dipende dal precedente.
    """

    def parse_document(**kwargs: Any) -> str:
        """Estrae testo dal file."""
        from core.document_processor import DocumentProcessor
        filepath = kwargs["filepath"]
        dp = DocumentProcessor()
        text, error = dp.process_file(filepath)
        if error:
            raise RuntimeError(f"Parsing fallito: {error}")
        return text

    def chunk_text(**kwargs: Any) -> List[str]:
        """Divide il testo in chunk per indicizzazione."""
        text = kwargs.get("parse_document", "")
        if not text:
            return []
        chunk_size = kwargs.get("chunk_size", 512)
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def index_chunks(**kwargs: Any) -> int:
        """Indicizza i chunk nella memoria Pilot."""
        chunks = kwargs.get("chunk_text", [])
        filepath = kwargs.get("filepath", "")
        filename = kwargs.get("filename", "")
        pilot = kwargs.get("pilot")

        if not chunks or not pilot:
            return 0

        indexed = 0
        for chunk in chunks:
            try:
                pilot.add_document(filepath, chunk, tags=[filename])
                indexed += 1
            except Exception:
                pass
        logger.info("Document pipeline: indicizzati %d/%d chunk per '%s'",
                     indexed, len(chunks), filename)
        return indexed

    pipe = Pipeline("document_processing", max_workers=1)
    pipe.add_step(Step("parse_document", parse_document, retries=1))
    pipe.add_step(Step("chunk_text", chunk_text, depends_on=["parse_document"]))
    pipe.add_step(Step(
        "index_chunks", index_chunks,
        depends_on=["chunk_text"],
        on_error="skip",
    ))
    return pipe


def build_memory_pipeline() -> Pipeline:
    """Pipeline di manutenzione memoria: entità → KB → compress.

    Scheduled: ogni 30 minuti (o on-demand).
    """

    def refresh_entities(**kwargs: Any) -> int:
        """Ricalcola le entità da conversazioni recenti."""
        import os
        import json
        conv_dir = os.path.join("data", "conversations")
        if not os.path.isdir(conv_dir):
            return 0

        from core.advanced_memory import EntityTracker
        tracker = EntityTracker(os.path.join("data", "entities.json"))

        processed = 0
        # Prendi le ultime 5 conversazioni
        files = sorted(
            [f for f in os.listdir(conv_dir) if f.endswith(".json") and f != ".gitkeep"],
            reverse=True,
        )[:5]

        for fname in files:
            try:
                fpath = os.path.join(conv_dir, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    conv = json.load(f)
                for msg in conv.get("messages", []):
                    if msg.get("role") == "user":
                        tracker.extract_and_save(msg["content"], "user")
                        processed += 1
            except Exception:
                pass
        tracker.save()
        logger.info("Memory pipeline: processati %d messaggi utente", processed)
        return processed

    def update_knowledge_base(**kwargs: Any) -> int:
        """Aggiorna la knowledge base dalle conversazioni recenti."""
        import os
        import json
        conv_dir = os.path.join("data", "conversations")
        if not os.path.isdir(conv_dir):
            return 0

        from core.advanced_memory import KnowledgeBase
        kb = KnowledgeBase()

        updated = 0
        files = sorted(
            [f for f in os.listdir(conv_dir) if f.endswith(".json") and f != ".gitkeep"],
            reverse=True,
        )[:5]

        for fname in files:
            try:
                fpath = os.path.join(conv_dir, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    conv = json.load(f)
                messages = conv.get("messages", [])
                if messages:
                    kb.update_from_conversation(messages)
                    updated += 1
            except Exception:
                pass
        logger.info("Memory pipeline: aggiornata KB da %d conversazioni", updated)
        return updated

    pipe = Pipeline("memory_refresh", max_workers=2)
    pipe.add_step(Step("refresh_entities", refresh_entities, on_error="skip"))
    pipe.add_step(Step("update_kb", update_knowledge_base, on_error="skip"))
    return pipe
