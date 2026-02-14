"""Tests per core.model_router — classificatore di intento e routing multi-modello."""

import unittest
from core.model_router import (
    Intent, classify_intent,
    ModelMapping, ModelRouter, RouteResult,
)


class TestClassifyIntent(unittest.TestCase):
    """Verifica il classificatore di intento basato su regole."""

    # ── VISION (immagini) ──────────────────────────────────────────────

    def test_images_override_everything(self):
        """Se ci sono immagini, l'intento è sempre VISION."""
        self.assertEqual(classify_intent("scrivi un programma python", has_images=True), Intent.VISION)
        self.assertEqual(classify_intent("ciao", has_images=True), Intent.VISION)
        self.assertEqual(classify_intent("", has_images=True), Intent.VISION)

    # ── CODE ───────────────────────────────────────────────────────────

    def test_code_fence(self):
        self.assertEqual(classify_intent("ecco il codice:\n```python\nprint('hi')\n```"), Intent.CODE)

    def test_inline_code(self):
        self.assertEqual(classify_intent("cosa fa `sorted()`?"), Intent.CODE)

    def test_code_keywords_explicit(self):
        self.assertEqual(classify_intent("scrivi un codice python per ordinare una lista"), Intent.CODE)
        self.assertEqual(classify_intent("genera una funzione di sorting"), Intent.CODE)
        self.assertEqual(classify_intent("implementa un binary search"), Intent.CODE)

    def test_code_keywords_languages(self):
        self.assertEqual(classify_intent("come si fa un loop in javascript?"), Intent.CODE)
        self.assertEqual(classify_intent("differenza tra rust e golang"), Intent.CODE)

    def test_code_keywords_technical(self):
        self.assertEqual(classify_intent("ho un bug nel mio script"), Intent.CODE)
        self.assertEqual(classify_intent("come faccio un git commit?"), Intent.CODE)
        self.assertEqual(classify_intent("scrivi una query SQL"), Intent.CODE)
        self.assertEqual(classify_intent("debugga questo errore nel codice"), Intent.CODE)

    def test_code_indented(self):
        msg = "guarda questo:\n  def hello():\n    print('world')"
        self.assertEqual(classify_intent(msg), Intent.CODE)

    # ── GENERAL ────────────────────────────────────────────────────────

    def test_greeting(self):
        self.assertEqual(classify_intent("ciao"), Intent.GENERAL)
        self.assertEqual(classify_intent("buongiorno!"), Intent.GENERAL)

    def test_general_question(self):
        self.assertEqual(classify_intent("qual è la capitale della Francia?"), Intent.GENERAL)
        self.assertEqual(classify_intent("spiegami la teoria della relatività"), Intent.GENERAL)

    def test_empty_message(self):
        self.assertEqual(classify_intent(""), Intent.GENERAL)
        self.assertEqual(classify_intent("  "), Intent.GENERAL)

    def test_general_italian(self):
        self.assertEqual(classify_intent("come stai?"), Intent.GENERAL)
        self.assertEqual(classify_intent("raccontami una storia"), Intent.GENERAL)


class TestModelMapping(unittest.TestCase):
    """Verifica la mappa intento→modello."""

    def test_default_mapping(self):
        m = ModelMapping()
        self.assertEqual(m.get(Intent.GENERAL), "gemma2:9b")
        self.assertEqual(m.get(Intent.CODE), "qwen2.5-coder:7b")
        self.assertEqual(m.get(Intent.VISION), "minicpm-v")

    def test_custom_mapping(self):
        m = ModelMapping(general="llama3.1", code="codellama:7b", vision="minicpm-v")
        self.assertEqual(m.get(Intent.GENERAL), "llama3.1")
        self.assertEqual(m.get(Intent.CODE), "codellama:7b")

    def test_all_models_unique(self):
        m = ModelMapping()
        models = m.all_models()
        self.assertEqual(len(models), len(set(models)))

    def test_all_models_dedup(self):
        """Se general e code sono lo stesso modello, non duplicare."""
        m = ModelMapping(general="llama3.1", code="llama3.1", vision="minicpm-v")
        self.assertEqual(len(m.all_models()), 2)


class TestModelRouter(unittest.TestCase):
    """Verifica il router completo con tracking del warm model."""

    def setUp(self):
        self.router = ModelRouter(
            mapping=ModelMapping(
                general="gemma2:9b",
                code="qwen2.5-coder:7b",
                vision="minicpm-v",
            ),
            fallback_model="llama3.2",
        )
        # Simula modelli installati
        self.router.refresh_installed([
            "gemma2:9b", "qwen2.5-coder:7b", "minicpm-v", "llama3.2",
        ])

    def test_route_general(self):
        result = self.router.route("ciao, come stai?")
        self.assertEqual(result.model, "gemma2:9b")
        self.assertEqual(result.intent, Intent.GENERAL)
        self.assertFalse(result.fallback_used)

    def test_route_code(self):
        result = self.router.route("scrivi un codice python per fibonacci")
        self.assertEqual(result.model, "qwen2.5-coder:7b")
        self.assertEqual(result.intent, Intent.CODE)

    def test_route_vision(self):
        result = self.router.route("cosa c'è in questa foto?", has_images=True)
        self.assertEqual(result.model, "minicpm-v")
        self.assertEqual(result.intent, Intent.VISION)

    def test_fallback_when_not_installed(self):
        """Se il modello preferito non è installato, usa il fallback."""
        router = ModelRouter(
            mapping=ModelMapping(general="phi3:14b", code="phi3:14b", vision="phi3:14b"),
            fallback_model="llama3.2",
        )
        router.refresh_installed(["llama3.2"])
        result = router.route("ciao")
        self.assertEqual(result.model, "llama3.2")
        self.assertTrue(result.fallback_used)

    def test_swap_detection(self):
        """Rileva correttamente quando serve uno swap."""
        r1 = self.router.route("ciao")  # gemma2:9b → warm
        self.assertFalse(r1.is_swap)  # primo caricamento

        r2 = self.router.route("ciao ancora")  # gemma2:9b → stesso
        self.assertFalse(r2.is_swap)

        r3 = self.router.route("scrivi codice python")  # swap a qwen
        self.assertTrue(r3.is_swap)

    def test_warm_model_tracking(self):
        self.assertIsNone(self.router.warm_model)
        self.router.route("test")
        self.assertEqual(self.router.warm_model, "gemma2:9b")

    def test_warm_model_manual_set(self):
        self.router.warm_model = "custom:model"
        self.assertEqual(self.router.warm_model, "custom:model")

    def test_force_intent(self):
        result = self.router.route("ciao", force_intent=Intent.CODE)
        self.assertEqual(result.intent, Intent.CODE)
        self.assertEqual(result.model, "qwen2.5-coder:7b")

    def test_route_result_to_dict(self):
        result = self.router.route("test")
        d = result.to_dict()
        self.assertIn("model", d)
        self.assertIn("intent", d)
        self.assertIn("is_swap", d)
        self.assertEqual(d["intent"], "general")


class TestRouteResult(unittest.TestCase):
    def test_to_dict(self):
        r = RouteResult(model="gemma2:9b", intent=Intent.GENERAL, is_swap=True, fallback_used=False)
        d = r.to_dict()
        self.assertEqual(d["model"], "gemma2:9b")
        self.assertEqual(d["intent"], "general")
        self.assertTrue(d["is_swap"])
        self.assertFalse(d["fallback_used"])


if __name__ == "__main__":
    unittest.main()
