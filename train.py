#!/usr/bin/env python3
"""
Omni Eye AI — Training & Optimization Toolkit

Strumenti per:
1. Creare modelli personalizzati (Modelfile con system prompt + parametri ottimizzati)
2. Esportare dati di conversazione per fine-tuning esterno
3. Benchmark delle prestazioni di inferenza
4. Ottimizzare le variabili d'ambiente Ollama

Uso:
    python train.py create              Crea modelli personalizzati omni-chat e omni-coder
    python train.py create --model X    Crea solo il modello X
    python train.py export              Esporta conversazioni in JSONL per fine-tuning
    python train.py export -f alpaca    Esporta in formato Alpaca
    python train.py benchmark           Benchmark di tutti i modelli
    python train.py benchmark -m gemma2:9b  Benchmark di un modello specifico
    python train.py optimize            Imposta variabili di ottimizzazione Ollama
    python train.py all                 Esegui tutto
"""

import json
import os
import sys
import time
import glob
import argparse
import subprocess
import platform
import logging

# Aggiungi il percorso del progetto
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

try:
    import ollama
except ImportError:
    print("❌ Pacchetto 'ollama' non installato. Esegui: pip install ollama")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class OmniTrainer:
    """Toolkit per training, ottimizzazione e benchmark dei modelli Omni Eye AI."""

    # ── Definizione modelli personalizzati ────────────────────────────
    MODELFILES = {
        'omni-chat': {
            'base': 'gemma3:4b',
            'system': config.SYSTEM_PROMPT,
            'params': {
                'temperature': 0.7,
                'num_ctx': 8192,
                'num_predict': 2048,
                'repeat_penalty': 1.3,
                'repeat_last_n': 128,
                'top_k': 40,
                'top_p': 0.9,
            },
        },
        'omni-coder': {
            'base': 'qwen2.5-coder:7b',
            'system': (
                config.SYSTEM_PROMPT
                + "\n\n# Modalità Codice\n"
                "Sei specializzato in programmazione. Rispondi con codice pulito, "
                "commentato e funzionante. Usa sempre code block Markdown con il "
                "linguaggio specificato. Preferisci soluzioni idiomatiche."
            ),
            'params': {
                'temperature': 0.3,
                'num_ctx': 8192,
                'num_predict': 4096,
                'repeat_penalty': 1.2,
                'repeat_last_n': 128,
                'top_k': 50,
                'top_p': 0.95,
            },
        },
    }

    # ── Prompt di benchmark ──────────────────────────────────────────
    BENCH_PROMPTS = {
        'general': (
            "Spiega in modo conciso cos'è l'intelligenza artificiale "
            "e le sue applicazioni principali."
        ),
        'code': (
            "Scrivi una funzione Python che implementa il merge sort "
            "con commenti esplicativi."
        ),
        'reasoning': (
            "Un treno parte da Milano alle 8:00 a 120 km/h. Un altro parte "
            "da Roma alle 8:30 a 150 km/h in direzione Milano. La distanza "
            "è 600 km. Quando si incontrano?"
        ),
    }

    def __init__(self):
        self.client = ollama.Client(host=config.OLLAMA_HOST)
        self.conversations_dir = config.CONVERSATIONS_DIR
        self.data_dir = config.DATA_DIR

    # ══════════════════════════════════════════════════════════════════
    # 1. CREAZIONE MODELLI PERSONALIZZATI
    # ══════════════════════════════════════════════════════════════════

    def _generate_modelfile(self, name: str) -> str:
        """Genera il contenuto del Modelfile per un modello personalizzato."""
        spec = self.MODELFILES[name]
        lines = [f"FROM {spec['base']}", ""]

        # System prompt (triple-quoted per gestire caratteri speciali)
        lines.append(f'SYSTEM """{spec["system"]}"""')
        lines.append("")

        # Parametri ottimizzati
        for key, value in spec['params'].items():
            lines.append(f"PARAMETER {key} {value}")

        return "\n".join(lines)

    def create_model(self, name: str) -> bool:
        """Crea un singolo modello personalizzato in Ollama."""
        if name not in self.MODELFILES:
            logger.error(
                "❌ Modello '%s' non definito. Disponibili: %s",
                name, list(self.MODELFILES.keys()),
            )
            return False

        spec = self.MODELFILES[name]
        logger.info("\n🔧 Creazione modello '%s' (base: %s)...", name, spec['base'])

        modelfile_content = self._generate_modelfile(name)

        # Salva il Modelfile per riferimento
        modelfile_dir = os.path.join(config.BASE_DIR, 'Ollama')
        os.makedirs(modelfile_dir, exist_ok=True)
        modelfile_path = os.path.join(modelfile_dir, f'Modelfile.{name}')
        with open(modelfile_path, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)
        logger.info("   📄 Modelfile salvato: %s", modelfile_path)

        try:
            start = time.perf_counter()
            # Crea il modello in Ollama — usa il contenuto come stringa
            self.client.create(model=name, modelfile=modelfile_content)
            elapsed = time.perf_counter() - start
            logger.info(
                "   ✅ Modello '%s' creato con successo! (%.1fs)", name, elapsed,
            )
            return True
        except Exception as e:
            logger.error("   ❌ Errore creazione modello '%s': %s", name, e)
            return False

    def create_all_models(self) -> dict:
        """Crea tutti i modelli personalizzati definiti."""
        logger.info("=" * 60)
        logger.info("🏗️  CREAZIONE MODELLI PERSONALIZZATI")
        logger.info("=" * 60)

        results = {}
        for name in self.MODELFILES:
            results[name] = self.create_model(name)

        # Riepilogo
        logger.info("\n" + "=" * 60)
        logger.info("📊 RISULTATI:")
        for name, success in results.items():
            status = "✅" if success else "❌"
            logger.info("   %s %s", status, name)

        created = [n for n, s in results.items() if s]
        if created:
            logger.info("\n💡 Per attivare i modelli personalizzati:")
            logger.info("   1. Aggiungi in .env:")
            logger.info("      BAKED_PROMPT_MODELS=%s", ",".join(created))
            logger.info("      ROUTER_MODEL_GENERAL=omni-chat")
            logger.info("      ROUTER_MODEL_CODE=omni-coder")
            logger.info("   2. Riavvia il server Flask")

        return results

    # ══════════════════════════════════════════════════════════════════
    # 2. EXPORT DATI DI TRAINING
    # ══════════════════════════════════════════════════════════════════

    def export_training_data(
        self,
        output: str = None,
        min_turns: int = 2,
        format: str = 'chatml',
    ) -> int:
        """Esporta conversazioni salvate in JSONL per fine-tuning.

        Formati supportati:
        - 'chatml': formato ChatML / OpenAI (Unsloth, Axolotl, LLaMA-Factory)
        - 'alpaca': formato Alpaca (instruction / input / output)

        Returns:
            Numero di esempi esportati
        """
        output = output or os.path.join(self.data_dir, 'training_data.jsonl')

        logger.info("\n" + "=" * 60)
        logger.info("📤 EXPORT DATI DI TRAINING")
        logger.info("=" * 60)
        logger.info("   Formato:    %s", format)
        logger.info("   Min turni:  %d", min_turns)
        logger.info("   Output:     %s", output)

        conv_files = glob.glob(os.path.join(self.conversations_dir, '*.json'))
        logger.info("   Conversazioni trovate: %d", len(conv_files))

        examples = []
        skipped = 0

        for conv_file in conv_files:
            try:
                with open(conv_file, 'r', encoding='utf-8') as f:
                    conv = json.load(f)

                messages = conv.get('messages', [])
                # Filtra solo user/assistant (escludi system/tool)
                clean = [
                    m for m in messages
                    if m.get('role') in ('user', 'assistant')
                ]

                if len(clean) < min_turns * 2:
                    skipped += 1
                    continue

                if format == 'chatml':
                    # ChatML: {"messages": [{"role":..., "content":...}]}
                    example = {
                        "messages": [
                            {"role": "system", "content": config.SYSTEM_PROMPT},
                            *[
                                {"role": m['role'], "content": m['content']}
                                for m in clean
                            ],
                        ]
                    }
                    examples.append(example)

                elif format == 'alpaca':
                    # Alpaca: ogni coppia user/assistant → un esempio
                    for i in range(0, len(clean) - 1, 2):
                        if (clean[i]['role'] == 'user'
                                and clean[i + 1]['role'] == 'assistant'):
                            example = {
                                "instruction": clean[i]['content'],
                                "input": "",
                                "output": clean[i + 1]['content'],
                            }
                            examples.append(example)

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("   ⚠️ Errore parsing %s: %s", conv_file, e)
                skipped += 1

        # Scrivi JSONL
        os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
        with open(output, 'w', encoding='utf-8') as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')

        logger.info(
            "\n   ✅ Esportati %d esempi (%d conversazioni saltate)",
            len(examples), skipped,
        )
        logger.info("   📁 File: %s", output)

        if examples:
            logger.info("\n💡 Per fine-tuning, usa uno di questi strumenti:")
            logger.info(
                "   • Unsloth (consigliato, veloce): "
                "https://github.com/unslothai/unsloth"
            )
            logger.info(
                "   • Axolotl: "
                "https://github.com/OpenAccess-AI-Collective/axolotl"
            )
            logger.info(
                "   • LLaMA-Factory: "
                "https://github.com/hiyouga/LLaMA-Factory"
            )
            logger.info("\n   Dopo il fine-tuning, converti in GGUF e importa:")
            logger.info("   ollama create mio-modello -f Modelfile")

        return len(examples)

    # ══════════════════════════════════════════════════════════════════
    # 3. BENCHMARK INFERENZA
    # ══════════════════════════════════════════════════════════════════

    def benchmark_model(
        self, model: str, prompts: dict = None, runs: int = 3,
    ) -> dict:
        """Esegue benchmark di inferenza su un modello.

        Returns:
            Dict con metriche: tokens/sec, TTFT, tempo totale
        """
        prompts = prompts or self.BENCH_PROMPTS
        results = {}

        logger.info("\n   📊 Benchmark: %s", model)
        logger.info("   " + "-" * 50)

        for prompt_type, prompt in prompts.items():
            times = []
            token_counts = []
            first_token_times = []

            for run_i in range(runs):
                try:
                    start = time.perf_counter()
                    first_token_time = None
                    tokens = 0

                    stream = self.client.chat(
                        model=model,
                        messages=[
                            {
                                "role": "system",
                                "content": "Rispondi in italiano, in modo conciso.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        stream=True,
                        options={
                            'num_predict': 256,
                            'num_ctx': 4096,
                            'temperature': 0.7,
                        },
                    )

                    for chunk in stream:
                        if 'message' in chunk and chunk['message'].get('content'):
                            tokens += 1
                            if first_token_time is None:
                                first_token_time = time.perf_counter() - start

                    elapsed = time.perf_counter() - start
                    times.append(elapsed)
                    token_counts.append(tokens)
                    if first_token_time:
                        first_token_times.append(first_token_time)

                except Exception as e:
                    logger.warning("      ⚠️ Run %d fallito: %s", run_i + 1, e)

            if times:
                avg_time = sum(times) / len(times)
                avg_tokens = sum(token_counts) / len(token_counts)
                avg_tps = avg_tokens / avg_time if avg_time > 0 else 0
                avg_ttft = (
                    sum(first_token_times) / len(first_token_times)
                    if first_token_times else 0
                )

                results[prompt_type] = {
                    'avg_time': round(avg_time, 2),
                    'avg_tokens': round(avg_tokens),
                    'tokens_per_sec': round(avg_tps, 1),
                    'time_to_first_token_ms': round(avg_ttft * 1000),
                }

                logger.info(
                    "      %-10s  %.1f tok/s  |  TTFT: %4dms  |  %d tok in %.1fs",
                    prompt_type, avg_tps, avg_ttft * 1000,
                    int(avg_tokens), avg_time,
                )

        return results

    def benchmark_all(self, runs: int = 3) -> dict:
        """Benchmark di tutti i modelli configurati."""
        logger.info("\n" + "=" * 60)
        logger.info("⚡ BENCHMARK INFERENZA")
        logger.info("=" * 60)
        logger.info("   Run per prompt: %d", runs)

        # Raccogli modelli da testare
        models = set()
        models.add(config.AI_CONFIG['model'])
        for m in config.MODEL_ROUTER_CONFIG.get('models', {}).values():
            if m:
                models.add(m)

        # Verifica modelli installati
        try:
            installed = [m.model for m in self.client.list().models]
        except Exception:
            installed = []

        all_results = {}
        for model in sorted(models):
            model_base = model.split(':')[0]
            if installed and not any(
                model_base == m.split(':')[0] for m in installed
            ):
                logger.info("\n   ⏭️  %s — non installato, skip", model)
                continue

            all_results[model] = self.benchmark_model(model, runs=runs)

        # Riepilogo finale
        logger.info("\n" + "=" * 60)
        logger.info("📊 RIEPILOGO BENCHMARK")
        logger.info("=" * 60)
        logger.info("  %-25s  %8s  %8s", "Modello", "tok/s", "TTFT")
        logger.info("  " + "-" * 48)
        for model, results in all_results.items():
            if results:
                avg_tps = (
                    sum(r['tokens_per_sec'] for r in results.values())
                    / len(results)
                )
                avg_ttft = (
                    sum(r['time_to_first_token_ms'] for r in results.values())
                    / len(results)
                )
                logger.info(
                    "  %-25s  %7.1f  %6.0fms", model, avg_tps, avg_ttft,
                )

        return all_results

    # ══════════════════════════════════════════════════════════════════
    # 4. OTTIMIZZAZIONE OLLAMA
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def optimize_ollama() -> None:
        """Imposta variabili d'ambiente Ollama per prestazioni ottimali.

        Variabili configurate:
        - OLLAMA_FLASH_ATTENTION=1   Flash Attention (RTX 20xx+, compute >= 7.0)
        - OLLAMA_KV_CACHE_TYPE=q8_0  KV cache quantizzata (~50% meno VRAM)
        - OLLAMA_MAX_LOADED_MODELS=1 Un solo modello in VRAM (per 8GB)
        - OLLAMA_NUM_PARALLEL=1      Una richiesta alla volta (max velocità)
        """
        logger.info("\n" + "=" * 60)
        logger.info("⚙️  OTTIMIZZAZIONE OLLAMA")
        logger.info("=" * 60)

        env_vars = {
            'OLLAMA_FLASH_ATTENTION': '1',
            'OLLAMA_KV_CACHE_TYPE': 'q8_0',
            'OLLAMA_MAX_LOADED_MODELS': '1',
            'OLLAMA_NUM_PARALLEL': '1',
        }

        descriptions = {
            'OLLAMA_FLASH_ATTENTION': 'Flash Attention (30-50% più veloce)',
            'OLLAMA_KV_CACHE_TYPE': 'KV cache quantizzata (50% meno VRAM)',
            'OLLAMA_MAX_LOADED_MODELS': '1 modello alla volta (8GB VRAM)',
            'OLLAMA_NUM_PARALLEL': '1 richiesta parallela (max velocità)',
        }

        if platform.system() != 'Windows':
            logger.info("   ℹ️  Sistema non-Windows. Aggiungi al tuo .bashrc:")
            for key, val in env_vars.items():
                logger.info("   export %s=%s  # %s", key, val, descriptions[key])
            return

        # Windows: usa setx per persistenza
        for key, val in env_vars.items():
            current = os.environ.get(key)
            try:
                subprocess.run(
                    ['setx', key, val],
                    capture_output=True, text=True, check=True,
                )
                status = "aggiornato" if current != val else "confermato"
                logger.info(
                    "   ✅ %s = %s  (%s — %s)",
                    key, val, status, descriptions[key],
                )
            except subprocess.CalledProcessError as e:
                logger.error("   ❌ %s: %s", key, e.stderr.strip())

        logger.info("\n   ⚠️  Riavvia Ollama per applicare le modifiche:")
        logger.info("   → Chiudi Ollama dalla system tray, poi riaprilo")

    # ══════════════════════════════════════════════════════════════════
    # 5. INFO SISTEMA
    # ══════════════════════════════════════════════════════════════════

    def show_info(self) -> None:
        """Mostra info su hardware, modelli e configurazione corrente."""
        logger.info("\n" + "=" * 60)
        logger.info("ℹ️  INFO SISTEMA")
        logger.info("=" * 60)

        # Modelli installati
        try:
            models = self.client.list().models
            logger.info("\n   Modelli installati:")
            for m in models:
                size_gb = getattr(m, 'size', 0) / (1024 ** 3)
                logger.info("      • %-25s  %.1f GB", m.model, size_gb)
        except Exception as e:
            logger.warning("   ⚠️ Impossibile listare modelli: %s", e)

        # Configurazione corrente
        logger.info("\n   Configurazione Model Router:")
        logger.info("      Abilitato: %s", config.MODEL_ROUTER_CONFIG['enabled'])
        for intent, model in config.MODEL_ROUTER_CONFIG.get('models', {}).items():
            logger.info("      %-10s → %s", intent, model)

        # Profili per-modello
        logger.info("\n   Profili per-modello:")
        for model, profile in config.MODEL_PROFILES.items():
            if model == '_default':
                continue
            logger.info(
                "      %-25s  ctx=%d  batch=%d  temp=%.1f  top_k=%d",
                model,
                profile.get('num_ctx', 0),
                profile.get('num_batch', 0),
                profile.get('temperature', 0),
                profile.get('top_k', 0),
            )

        # Env vars Ollama
        logger.info("\n   Variabili Ollama:")
        for key in ('OLLAMA_FLASH_ATTENTION', 'OLLAMA_KV_CACHE_TYPE',
                     'OLLAMA_MAX_LOADED_MODELS', 'OLLAMA_NUM_PARALLEL'):
            val = os.environ.get(key, '(non impostata)')
            logger.info("      %s = %s", key, val)

        # Conversazioni
        conv_files = glob.glob(os.path.join(self.conversations_dir, '*.json'))
        logger.info("\n   Conversazioni salvate: %d", len(conv_files))

        # Baked models
        if config.BAKED_PROMPT_MODELS:
            logger.info("   Modelli con prompt integrato: %s",
                         ", ".join(config.BAKED_PROMPT_MODELS))
        else:
            logger.info("   Modelli con prompt integrato: nessuno")
            logger.info("   → Esegui 'python train.py create' per crearli")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Omni Eye AI — Training & Optimization Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python train.py info                  Mostra info sistema e configurazione
  python train.py create               Crea modelli personalizzati
  python train.py create --model X     Crea solo il modello X
  python train.py export               Esporta dati per fine-tuning (ChatML)
  python train.py export -f alpaca     Esporta in formato Alpaca
  python train.py benchmark            Benchmark di tutti i modelli
  python train.py benchmark -m gemma2  Benchmark di un modello specifico
  python train.py optimize             Configura env vars Ollama (flash attn)
  python train.py all                  Optimize + Create + Export + Benchmark
        """,
    )
    subparsers = parser.add_subparsers(dest='command', help='Comando da eseguire')

    # info
    subparsers.add_parser('info', help='Mostra info sistema e configurazione')

    # create
    create_p = subparsers.add_parser('create', help='Crea modelli personalizzati')
    create_p.add_argument(
        '--model', help='Nome modello specifico (omni-chat, omni-coder)',
    )

    # export
    export_p = subparsers.add_parser('export', help='Esporta dati di training')
    export_p.add_argument('--output', '-o', help='File output JSONL')
    export_p.add_argument(
        '--format', '-f', default='chatml', choices=['chatml', 'alpaca'],
        help='Formato output (default: chatml)',
    )
    export_p.add_argument(
        '--min-turns', type=int, default=2,
        help='Minimo turni per conversazione (default: 2)',
    )

    # benchmark
    bench_p = subparsers.add_parser('benchmark', help='Benchmark inferenza')
    bench_p.add_argument(
        '--runs', '-r', type=int, default=3,
        help='Iterazioni per prompt (default: 3)',
    )
    bench_p.add_argument(
        '--model', '-m', help='Benchmark di un modello specifico',
    )

    # optimize
    subparsers.add_parser(
        'optimize', help='Imposta variabili di ottimizzazione Ollama',
    )

    # all
    subparsers.add_parser(
        'all', help='Esegui tutto (optimize + create + export + benchmark)',
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    trainer = OmniTrainer()

    if args.command == 'info':
        trainer.show_info()

    elif args.command == 'create':
        if args.model:
            trainer.create_model(args.model)
        else:
            trainer.create_all_models()

    elif args.command == 'export':
        trainer.export_training_data(
            output=args.output,
            format=args.format,
            min_turns=args.min_turns,
        )

    elif args.command == 'benchmark':
        if args.model:
            trainer.benchmark_model(args.model, runs=args.runs)
        else:
            trainer.benchmark_all(runs=args.runs)

    elif args.command == 'optimize':
        trainer.optimize_ollama()

    elif args.command == 'all':
        trainer.show_info()
        trainer.optimize_ollama()
        trainer.create_all_models()
        trainer.export_training_data()
        trainer.benchmark_all()


if __name__ == '__main__':
    main()
