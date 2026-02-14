"""
Test per il Pipeline Engine (core/pipeline.py)
"""

import time
import pytest
from core.pipeline import (
    Step, StepResult, StepStatus, Pipeline, PipelineError,
    PipelineScheduler,
    build_maintenance_pipeline, build_document_pipeline, build_memory_pipeline,
)


# ── Step basics ──────────────────────────────────────────────────────────

class TestStep:
    def test_step_creation(self):
        s = Step("test", lambda **kw: 42)
        assert s.name == "test"
        assert s.depends_on == []
        assert s.retries == 0

    def test_step_with_deps(self):
        s = Step("b", lambda **kw: 1, depends_on=["a"])
        assert s.depends_on == ["a"]


# ── Pipeline basics ──────────────────────────────────────────────────────

class TestPipeline:
    def test_single_step(self):
        pipe = Pipeline("test")
        pipe.add_step(Step("hello", lambda **kw: "world"))
        results = pipe.run()
        assert results["hello"].status == StepStatus.SUCCESS
        assert results["hello"].output == "world"

    def test_sequential_steps(self):
        """Step B dipende da A → esecuzione sequenziale."""
        pipe = Pipeline("seq")
        pipe.add_step(Step("a", lambda **kw: 10))
        pipe.add_step(Step("b", lambda **kw: kw["a"] + 5, depends_on=["a"]))
        results = pipe.run()
        assert results["a"].status == StepStatus.SUCCESS
        assert results["a"].output == 10
        assert results["b"].status == StepStatus.SUCCESS
        assert results["b"].output == 15

    def test_parallel_steps(self):
        """Step B e C dipendono da A → esecuzione parallela."""
        order = []

        def step_a(**kw):
            order.append("a")
            return 1

        def step_b(**kw):
            time.sleep(0.05)
            order.append("b")
            return kw["a"] + 10

        def step_c(**kw):
            time.sleep(0.05)
            order.append("c")
            return kw["a"] + 20

        pipe = Pipeline("par")
        pipe.add_step(Step("a", step_a))
        pipe.add_step(Step("b", step_b, depends_on=["a"]))
        pipe.add_step(Step("c", step_c, depends_on=["a"]))
        results = pipe.run()

        assert results["a"].output == 1
        assert results["b"].output == 11
        assert results["c"].output == 21
        assert order[0] == "a"  # A eseguito per primo

    def test_diamond_dag(self):
        """A → B, C → D (diamante)."""
        pipe = Pipeline("diamond")
        pipe.add_step(Step("a", lambda **kw: 1))
        pipe.add_step(Step("b", lambda **kw: kw["a"] * 2, depends_on=["a"]))
        pipe.add_step(Step("c", lambda **kw: kw["a"] * 3, depends_on=["a"]))
        pipe.add_step(Step("d", lambda **kw: kw["b"] + kw["c"], depends_on=["b", "c"]))
        results = pipe.run()
        assert results["d"].output == 5  # 2 + 3

    def test_kwargs_passthrough(self):
        """I kwargs iniziali sono accessibili a tutti gli step."""
        pipe = Pipeline("kw")
        pipe.add_step(Step("greet", lambda **kw: f"hello {kw['name']}"))
        results = pipe.run(name="world")
        assert results["greet"].output == "hello world"

    def test_duplicate_step_raises(self):
        pipe = Pipeline("dup")
        pipe.add_step(Step("x", lambda **kw: 1))
        with pytest.raises(ValueError, match="già presente"):
            pipe.add_step(Step("x", lambda **kw: 2))

    def test_missing_dependency_raises(self):
        pipe = Pipeline("dep")
        with pytest.raises(ValueError, match="non esiste"):
            pipe.add_step(Step("b", lambda **kw: 1, depends_on=["a"]))

    def test_chaining(self):
        """add_step restituisce self per chaining."""
        pipe = Pipeline("chain")
        result = pipe.add_step(Step("a", lambda **kw: 1))
        assert result is pipe


# ── Retry & Error handling ────────────────────────────────────────────

class TestRetryAndErrors:
    def test_retry_success(self):
        """Step fallisce al primo tentativo, riesce al secondo."""
        call_count = [0]

        def flaky(**kw):
            call_count[0] += 1
            if call_count[0] < 2:
                raise RuntimeError("transient")
            return "ok"

        pipe = Pipeline("retry")
        pipe.add_step(Step("flaky", flaky, retries=2, backoff=0.01))
        results = pipe.run()
        assert results["flaky"].status == StepStatus.SUCCESS
        assert results["flaky"].retries_used == 1

    def test_retry_exhausted(self):
        """Step fallisce dopo tutti i retry."""
        def always_fail(**kw):
            raise RuntimeError("permanent")

        pipe = Pipeline("fail")
        pipe.add_step(Step("bad", always_fail, retries=1, backoff=0.01))
        results = pipe.run()
        assert results["bad"].status == StepStatus.FAILED
        assert "permanent" in results["bad"].error

    def test_on_error_skip(self):
        """Step con on_error='skip' viene marcato SKIPPED, non FAILED."""
        def fail(**kw):
            raise RuntimeError("skip me")

        pipe = Pipeline("skip")
        pipe.add_step(Step("soft", fail, on_error="skip"))
        results = pipe.run()
        assert results["soft"].status == StepStatus.SKIPPED

    def test_failed_dependency_skips_child(self):
        """Se A fallisce, B (che dipende da A) viene skippato."""
        pipe = Pipeline("cascade")
        pipe.add_step(Step("a", lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))))
        pipe.add_step(Step("b", lambda **kw: "never", depends_on=["a"]))
        results = pipe.run()
        assert results["a"].status == StepStatus.FAILED
        assert results["b"].status == StepStatus.SKIPPED


# ── Duration tracking ─────────────────────────────────────────────────

class TestDuration:
    def test_duration_recorded(self):
        def slow(**kw):
            time.sleep(0.05)
            return "done"

        pipe = Pipeline("dur")
        pipe.add_step(Step("slow", slow))
        results = pipe.run()
        assert results["slow"].duration_ms >= 40  # almeno ~50ms


# ── Scheduler ──────────────────────────────────────────────────────────

class TestScheduler:
    def test_register_and_status(self):
        scheduler = PipelineScheduler()
        pipe = Pipeline("test")
        pipe.add_step(Step("noop", lambda **kw: None))
        scheduler.register("test_task", pipe, interval_seconds=3600)
        status = scheduler.get_status()
        assert "test_task" in status
        assert status["test_task"]["interval"] == 3600
        assert status["test_task"]["run_count"] == 0

    def test_start_stop(self):
        scheduler = PipelineScheduler()
        pipe = Pipeline("test")
        pipe.add_step(Step("noop", lambda **kw: None))
        scheduler.register("quick", pipe, interval_seconds=3600)
        scheduler.start()
        time.sleep(0.1)
        scheduler.stop()

    def test_run_on_start(self):
        scheduler = PipelineScheduler()
        pipe = Pipeline("immediate")
        pipe.add_step(Step("fast", lambda **kw: 42))
        scheduler.register("now", pipe, interval_seconds=3600, run_on_start=True)
        scheduler.start()
        time.sleep(0.5)  # Lascia tempo allo scheduler di eseguire
        scheduler.stop()
        status = scheduler.get_status()
        assert status["now"]["run_count"] >= 1


# ── Build functions ────────────────────────────────────────────────────

class TestPipelineBuilders:
    def test_maintenance_pipeline_builds(self):
        pipe = build_maintenance_pipeline()
        assert pipe.name == "maintenance"
        assert "clean_uploads" in pipe._steps
        assert "trim_logs" in pipe._steps
        assert "backup_kb" in pipe._steps

    def test_document_pipeline_builds(self):
        pipe = build_document_pipeline()
        assert pipe.name == "document_processing"
        assert "parse_document" in pipe._steps
        assert "chunk_text" in pipe._steps
        assert "index_chunks" in pipe._steps
        # Verifica dipendenze
        assert pipe._steps["chunk_text"].depends_on == ["parse_document"]
        assert pipe._steps["index_chunks"].depends_on == ["chunk_text"]

    def test_memory_pipeline_builds(self):
        pipe = build_memory_pipeline()
        assert pipe.name == "memory_refresh"
        assert "refresh_entities" in pipe._steps
        assert "update_kb" in pipe._steps

    def test_maintenance_pipeline_runs(self):
        """La maintenance pipeline deve girare senza errori (anche se non c'è nulla da pulire)."""
        pipe = build_maintenance_pipeline()
        results = pipe.run()
        # Tutti gli step devono essere SUCCESS o SKIPPED (mai FAILED)
        for name, r in results.items():
            assert r.status in (StepStatus.SUCCESS, StepStatus.SKIPPED), \
                f"Step '{name}' fallito: {r.error}"

    def test_memory_pipeline_runs(self):
        """La memory pipeline deve girare senza errori."""
        pipe = build_memory_pipeline()
        results = pipe.run()
        for name, r in results.items():
            assert r.status in (StepStatus.SUCCESS, StepStatus.SKIPPED), \
                f"Step '{name}' fallito: {r.error}"
