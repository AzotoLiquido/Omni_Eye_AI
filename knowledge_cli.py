#!/usr/bin/env python3
"""
CLI per la gestione della Knowledge Base di Omni Eye AI.

Comandi:
    install <pack>      Installa un singolo pack
    install --all        Installa tutti i pack
    list                 Lista pack disponibili
    search <query>       Cerca nella KB
    stats                Statistiche KB
    export <file>        Esporta tutti i fatti in JSON
    import <file>        Importa fatti da file (JSON/CSV/TXT)

Uso:
    python knowledge_cli.py list
    python knowledge_cli.py install --all
    python knowledge_cli.py install programming
    python knowledge_cli.py search "Python asyncio"
    python knowledge_cli.py stats
    python knowledge_cli.py export backup.json
    python knowledge_cli.py import miei_fatti.json
"""

import argparse
import json
import os
import sys

# Aggiungi root al path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.advanced_memory import KnowledgeBase
from core.knowledge_packs import (
    KNOWLEDGE_PACKS,
    get_available_packs,
    install_pack,
    install_all_packs,
    import_file,
)


DATA_DIR = os.path.join(ROOT, "data")


def get_kb():
    """Crea e restituisce un'istanza KnowledgeBase."""
    return KnowledgeBase(DATA_DIR)


# ── Comandi ─────────────────────────────────────────────────────────────────

def cmd_list(args):
    """Lista i pack disponibili."""
    packs = get_available_packs()
    total = sum(p["facts_count"] for p in packs)
    print(f"\n{'='*60}")
    print(f"  KNOWLEDGE PACKS DISPONIBILI ({len(packs)} pack, {total} fatti)")
    print(f"{'='*60}\n")
    for p in packs:
        print(f"  {p['name']:20s}  {p['facts_count']:3d} fatti  │ {p['description']}")
    print(f"\n{'─'*60}")
    print(f"  Totale: {total} fatti in {len(packs)} pack\n")


def cmd_install(args):
    """Installa uno o più pack."""
    kb = get_kb()
    try:
        if args.all:
            print("\nInstallazione di tutti i pack...")
            result = install_all_packs(kb)
            print(f"\n{'='*60}")
            print(f"  INSTALLAZIONE COMPLETATA")
            print(f"{'='*60}\n")
            for name, r in result["packs"].items():
                status = "✓" if r["skipped"] == 0 else "⚠"
                print(f"  {status} {name:20s}  {r['installed']:3d} installati"
                      + (f"  ({r['skipped']} saltati)" if r["skipped"] else ""))
            print(f"\n{'─'*60}")
            print(f"  Totale: {result['total_installed']} installati, "
                  f"{result['total_skipped']} saltati\n")
        elif args.pack_name:
            print(f"\nInstallazione pack '{args.pack_name}'...")
            r = install_pack(kb, args.pack_name)
            print(f"  ✓ {r['installed']} fatti installati"
                  + (f", {r['skipped']} saltati" if r["skipped"] else ""))
        else:
            print("Specificare un nome pack o --all")
            sys.exit(1)
    finally:
        kb.close()


def cmd_search(args):
    """Cerca nella KB."""
    kb = get_kb()
    try:
        results = kb.search_facts(args.query, limit=args.limit)
        print(f"\n  Risultati per '{args.query}' ({len(results)} trovati):\n")
        for i, r in enumerate(results, 1):
            source = f" [{r['source']}]" if r.get("source") else ""
            print(f"  {i:2d}. {r['content']}{source}")
        if not results:
            print("  Nessun risultato trovato.")
        print()
    finally:
        kb.close()


def cmd_stats(args):
    """Mostra statistiche della KB."""
    kb = get_kb()
    try:
        count = kb.get_facts_count()
        topics = kb._get_topics()
        print(f"\n{'='*60}")
        print(f"  STATISTICHE KNOWLEDGE BASE")
        print(f"{'='*60}\n")
        print(f"  Fatti totali:      {count}")
        print(f"  Topic tracciati:   {len(topics)}")
        print(f"  Database:          {kb._db_path}")

        # Conta fatti per source
        with kb._lock:
            rows = kb._conn.execute(
                "SELECT source, COUNT(*) as cnt FROM facts "
                "GROUP BY source ORDER BY cnt DESC"
            ).fetchall()
        if rows:
            print(f"\n  Fatti per sorgente:")
            for r in rows:
                src = r[0] if r[0] else "(nessuna)"
                print(f"    {src:30s} {r[1]:5d}")
        print()
    finally:
        kb.close()


def cmd_export(args):
    """Esporta tutti i fatti in un file JSON."""
    kb = get_kb()
    try:
        with kb._lock:
            rows = kb._conn.execute(
                "SELECT id, content, source, created_at FROM facts "
                "ORDER BY id"
            ).fetchall()
        facts = [dict(r) for r in rows]
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(facts, f, ensure_ascii=False, indent=2)
        print(f"\n  ✓ Esportati {len(facts)} fatti in '{args.output}'\n")
    finally:
        kb.close()


def cmd_import(args):
    """Importa fatti da un file esterno."""
    kb = get_kb()
    try:
        count = import_file(kb, args.input, source=args.source)
        print(f"\n  ✓ Importati {count} fatti da '{args.input}'\n")
    finally:
        kb.close()


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Gestione Knowledge Base – Omni Eye AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", help="Comando da eseguire")

    # list
    sub.add_parser("list", help="Lista pack disponibili")

    # install
    p_install = sub.add_parser("install", help="Installa pack")
    p_install.add_argument("pack_name", nargs="?", help="Nome del pack")
    p_install.add_argument("--all", action="store_true", help="Installa tutti i pack")

    # search
    p_search = sub.add_parser("search", help="Cerca nella KB")
    p_search.add_argument("query", help="Testo da cercare")
    p_search.add_argument("-n", "--limit", type=int, default=10, help="Max risultati")

    # stats
    sub.add_parser("stats", help="Statistiche KB")

    # export
    p_export = sub.add_parser("export", help="Esporta fatti in JSON")
    p_export.add_argument("output", help="File di output")

    # import
    p_import = sub.add_parser("import", help="Importa fatti da file")
    p_import.add_argument("input", help="File da importare (JSON/CSV/TXT)")
    p_import.add_argument("--source", help="Sorgente personalizzata")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    commands = {
        "list": cmd_list,
        "install": cmd_install,
        "search": cmd_search,
        "stats": cmd_stats,
        "export": cmd_export,
        "import": cmd_import,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
