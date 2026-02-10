"""
Web Search — ricerca DuckDuckGo per Omni Eye AI

Fornisce ricerche web reali per evitare che il modello inventi link/URL.
Usa DuckDuckGo: gratuito, senza API key, privacy-friendly.
"""

import logging
import re
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# ── Pattern per rilevare richieste di ricerca ──────────────────────────
# Parole chiave italiane e inglesi che indicano una ricerca web
_SEARCH_PATTERNS = re.compile(
    r"""
    (?:
        # Italiano
        cerca(?:mi|re)?
        | trovar?(?:mi|e)?
        | (?:dammi|mostra(?:mi)?)\s+(?:il\s+)?link
        | link\s+(?:di|per|a|della?|dello?)
        | url\s+(?:di|per|della?|dello?)
        | su\s+(?:youtube|google|internet|web|wikipedia|amazon|spotify)
        | apri(?:mi)?\s+
        # Inglese
        | search(?:\s+for)?
        | find\s+(?:me\s+)?
        | look(?:\s+up)?
        | google
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Pattern per richieste specifiche YouTube
_YOUTUBE_PATTERN = re.compile(
    r"youtube|video\s+(?:di|della?|su)|canzone|musica|brano|song|music|watch",
    re.IGNORECASE,
)


def needs_web_search(message: str) -> bool:
    """Indica se il messaggio dell'utente richiede probabilmente una ricerca web."""
    return bool(_SEARCH_PATTERNS.search(message))


def is_youtube_query(message: str) -> bool:
    """Indica se la ricerca è specifica per YouTube."""
    return bool(_YOUTUBE_PATTERN.search(message))


# Parole filler da rimuovere per pulire la query di ricerca
_FILLER_WORDS = re.compile(
    r"\b(?:trovami|cercami|cercamelo|dammi|mostrami|apri|aprimi|"
    r"trova|cerca|cerca(?:re)?|per favore|please|puoi|potresti|"
    r"il link|un link|il video|della canzone|canzone|brano|musica|song|music|"
    r"di|del|della|dello|"
    r"su|mi|me|la|le|lo|gli|un|una|dei|delle|degli)\b",
    re.IGNORECASE,
)


def _clean_query(message: str) -> str:
    """Rimuove parole filler dal messaggio per ottenere una query di ricerca pulita."""
    cleaned = _FILLER_WORDS.sub(" ", message)
    # Collassa spazi multipli
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Se troppo corta dopo il cleaning, usa l'originale
    if len(cleaned) < 3:
        return message.strip()
    return cleaned


def web_search(
    query: str,
    *,
    max_results: int = 5,
    region: str = "it-it",
    youtube: bool = False,
) -> List[Dict[str, str]]:
    """
    Esegue una ricerca web via DuckDuckGo.

    Args:
        query: Testo da cercare
        max_results: Numero massimo di risultati (default 5)
        region: Regione per i risultati (default it-it)
        youtube: Se True, aggiunge "site:youtube.com" alla query

    Returns:
        Lista di dict con chiavi: title, url, snippet
        Lista vuota se la ricerca fallisce.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        logger.warning("ddgs non installato. Installa con: pip install ddgs")
        return []

    # Per YouTube usare regione mondiale: evita che DuckDuckGo
    # restituisca cover/traduzioni italiane invece dell'originale.
    if youtube:
        region = "wt-wt"

    search_query = query
    if youtube:
        search_query = f"site:youtube.com {query}"

    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(search_query, region=region, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
        logger.info("Web search: %d risultati per '%s'", len(results), search_query[:60])
        return results
    except Exception as e:
        logger.error("Errore web search: %s", e)
        return []


def format_search_results(results: List[Dict[str, str]], query: str = "") -> str:
    """
    Formatta i risultati per il contesto AI (il modello NON deve riprodurre gli URL).
    """
    if not results:
        return (
            "[RICERCA WEB] Nessun risultato trovato"
            + (f" per: {query}" if query else "")
            + ". Informa l'utente che la ricerca non ha prodotto risultati."
        )

    lines = [
        "[ISTRUZIONE PRIORITARIA — RICERCA WEB COMPLETATA]",
        f"Una ricerca web per \"{query}\" è stata GIÀ ESEGUITA con successo." if query else "Una ricerca web è stata GIÀ ESEGUITA con successo.",
        f"I {len(results)} risultati con link reali sono GIÀ STATI MOSTRATI all'utente sopra questo messaggio.",
        "",
        "Il tuo compito ORA:",
        "- Rispondi con un BREVE commento utile sui risultati (es. quale link è più rilevante, un consiglio).",
        "- NON dire che non puoi cercare online: la ricerca È GIÀ STATA FATTA.",
        "- NON ripetere i link già mostrati.",
        "- NON inventare link o URL aggiuntivi.",
        "- NON dire 'mi dispiace' o 'non posso': i risultati ci sono già.",
    ]
    return "\n".join(lines)


def format_search_results_user(results: List[Dict[str, str]], query: str = "") -> str:
    """
    Formatta i risultati come Markdown da mostrare direttamente all'utente.
    Gli URL sono reali e cliccabili.
    """
    if not results:
        return f"Nessun risultato trovato per: *{query}*" if query else "Nessun risultato trovato."

    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "Senza titolo")
        url = r.get("url", "")
        snippet = r.get("snippet", "")
        if url:
            lines.append(f"{i}. [{title}]({url})")
        else:
            lines.append(f"{i}. {title}")
        if snippet:
            # Tronca snippet e rimuovi newline
            clean_snippet = snippet[:180].replace("\n", " ").strip()
            if clean_snippet:
                lines.append(f"   {clean_snippet}")
    return "\n".join(lines)


def search_and_format(message: str, max_results: int = 5) -> Optional[str]:
    """
    Punto di ingresso principale: rileva se serve una ricerca,
    la esegue e restituisce il contesto formattato.

    Args:
        message: Messaggio dell'utente

    Returns:
        Stringa con risultati formattati, o None se non serve ricerca
    """
    if not needs_web_search(message):
        return None

    youtube = is_youtube_query(message)
    clean_q = _clean_query(message)
    results = web_search(clean_q, max_results=max_results, youtube=youtube)
    if not results:
        return None
    return {
        "user": format_search_results_user(results, query=message),
        "model": format_search_results(results, query=message),
        "empty": len(results) == 0,
    }
