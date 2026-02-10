"""
Web Search — ricerca DuckDuckGo per Omni Eye AI

Fornisce ricerche web reali per evitare che il modello inventi link/URL.
Usa DuckDuckGo: gratuito, senza API key, privacy-friendly.

Due modalità:
  • "links"     — ricerca esplicita ("cercami X") → link diretti, modello saltato
  • "augmented" — domanda fattuale ("chi è X?")   → ricerca in background,
                   i dati vengono passati al modello che risponde in modo naturale
"""

import html
import logging
import re
from typing import List, Dict, Optional, Set

logger = logging.getLogger(__name__)

# ── Pattern per rilevare richieste di ricerca ESPLICITA ────────────────
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

# ── Pattern per domande FATTUALI che beneficiano di dati aggiornati ────
# Nota: [eè] matcha sia "e" che "è" per gestire input senza accenti.
#       ['\u2019']? matcha apostrofo dritto, curvo o nessuno.
_FACTUAL_PATTERNS = re.compile(
    r"""
    (?:
        # Domande dirette (italiano)
        chi\s+(?:[eè]|era|sono|erano|ha\s+(?:vinto|inventato|scoperto|fondato|creato|scritto))
        | cos['\u2019']?\s*[eè]
        | cosa\s+(?:[eè]|sono|significa)
        | quand[oi]\s+(?:[eè]|sono|nasce|muore|esce|usc[iì]|[eè]\s+(?:nato|morto|uscito|successo))
        | quant[oiae]\s+(?:cost[ai]|vale|pesa|misura|dista|abitanti|[eè]\s+(?:alto|grande|lungo|vecchio))
        | dov['\u2019']?\s*[eè]
        | dove\s+(?:si\s+trova|nasce|vive|abita)
        | perch[eé]\s+
        | qual\s+[eè]
        | com['\u2019']\s*[eè]
        | (?:[eè]|sono)\s+(?:vero|vera|veri|vere)\s+che
        # Aggiornamenti / attualità
        | (?:ultime\s+)?notizi[ae]\s+(?:su|di|riguardo)
        | (?:chi|cosa)\s+ha\s+vinto
        | risultat[oi]\s+(?:di|della?|dello?)
        | classifica\s+(?:di|della?|dello?)
        | prezz[oi]\s+(?:di|della?|dello?)
        | meteo\s+(?:a|di|in)
        | dat[ae]\s+(?:di|della?|dello?)
        # Domande dirette (inglese)
        | who\s+(?:is|was|are|were|won|invented|discovered|founded|created|wrote)
        | what\s+(?:is|are|was|were|does|did|happened)
        | when\s+(?:is|was|did|does|will)
        | where\s+(?:is|are|was|were|does|did)
        | how\s+(?:much|many|old|tall|long|far|big)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def needs_web_search(message: str) -> bool:
    """Indica se il messaggio dell'utente richiede probabilmente una ricerca web."""
    return bool(_SEARCH_PATTERNS.search(message))


def needs_factual_search(message: str) -> bool:
    """Indica se il messaggio è una domanda fattuale che beneficerebbe di dati web."""
    # Non attivare se è già una ricerca esplicita (gestita da needs_web_search)
    if needs_web_search(message):
        return False
    return bool(_FACTUAL_PATTERNS.search(message))


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

# Filler per domande fattuali (meno aggressivo, toglie solo la parte interrogativa)
_FACTUAL_FILLER = re.compile(
    r"\b(?:chi|cos|cosa|quand[oi]|quant[oiae]|dov|dove|qual|com|perch[eé])"
    r"(?:['’])?\s*(?:[eè]|era|sono|erano|ha|hanno|costa|costano|vale|pesa|si\s+trova)?\b",
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
                    "title": html.unescape(r.get("title", "")),
                    "url": r.get("href", ""),
                    "snippet": html.unescape(r.get("body", "")),
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


# ── Post-filtro: rimuove URL inventati dal modello ─────────────────────
_MD_LINK_RE = re.compile(r"\[([^\]]*?)\]\(https?://[^)]+\)")
_BARE_URL_RE = re.compile(r"(?<!\()https?://[^\s)]+")  # Lookbehind: skip URLs inside markdown (


def strip_hallucinated_urls(text: str, allowed_urls: Set[str]) -> str:
    """
    Rimuove dal testo ogni URL non presente in allowed_urls.

    - Link Markdown [titolo](url) → mantiene solo il titolo
    - URL bare (https://...) → rimosso
    """
    def _replace_md_link(m: re.Match) -> str:
        # Estrai l'URL dal link markdown
        full = m.group(0)
        url_match = re.search(r"\((https?://[^)]+)\)", full)
        if url_match and url_match.group(1) in allowed_urls:
            return full  # URL reale, mantieni
        return m.group(1)  # Solo il titolo, senza link

    text = _MD_LINK_RE.sub(_replace_md_link, text)
    text = _BARE_URL_RE.sub(
        lambda m: m.group(0) if m.group(0) in allowed_urls else "",
        text,
    )
    # Pulisci spazi doppi residui
    text = re.sub(r"  +", " ", text)
    return text


def search_and_format(message: str, max_results: int = 5) -> Optional[dict]:
    """
    Punto di ingresso principale: rileva se serve una ricerca,
    la esegue e restituisce il contesto formattato.

    Returns:
        dict con chiavi:
            mode   — "links" (ricerca esplicita) o "augmented" (domanda fattuale)
            user   — Markdown per l'utente (solo mode=links)
            context— Testo contesto per il system prompt (solo mode=augmented)
            urls   — set di URL reali per il filtro post-risposta
        oppure None se non serve ricerca
    """
    explicit = needs_web_search(message)
    factual = needs_factual_search(message)

    if not explicit and not factual:
        return None

    youtube = is_youtube_query(message) if explicit else False
    clean_q = _clean_query(message) if explicit else _clean_factual_query(message)

    # Per domande fattuali: cerca preferibilmente su Wikipedia
    if factual and not youtube:
        results = web_search(
            f"wikipedia {clean_q}", max_results=max_results, region="it-it",
        )
        # Fallback: se Wikipedia non dà risultati, cerca normalmente
        if not results:
            results = web_search(clean_q, max_results=max_results)
    else:
        results = web_search(clean_q, max_results=max_results, youtube=youtube)
    if not results:
        return None

    urls = {r["url"] for r in results if r.get("url")}

    if explicit:
        # Modalità "links": risultati diretti, modello saltato
        return {
            "mode": "links",
            "user": format_search_results_user(results, query=message),
            "urls": urls,
        }
    else:
        # Modalità "augmented": contesto per il modello
        return {
            "mode": "augmented",
            "context": _format_augmented_context(results, query=message),
            "urls": urls,
        }


def _clean_factual_query(message: str) -> str:
    """Pulisce una domanda fattuale per ottenere una query di ricerca efficace."""
    # Rimuovi la parte interrogativa (italiano)
    cleaned = _FACTUAL_FILLER.sub(" ", message)
    # Rimuovi pattern inglesi
    cleaned = re.sub(
        r"\b(?:who|what|when|where|how)\s+(?:is|are|was|were|does|did|do|much|many|old|tall|long|far|big)"
        r"(?:\s+(?:a|an|the|does|did|do|is|it|cost))?\b",
        " ", cleaned, flags=re.IGNORECASE,
    )
    # Rimuovi "notizie su/di", "ultime notizie"
    cleaned = re.sub(r"\b(?:ultime\s+)?notizi[ae]\s+(?:su|di|riguardo)\b", " ", cleaned, flags=re.IGNORECASE)
    # Rimuovi punteggiatura interrogativa e apostrofi isolati
    cleaned = re.sub(r"[?!]", "", cleaned)
    cleaned = re.sub(r"(?:^|\s)['\u2019](?:\s|$)", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) < 3:
        return message.strip()
    return cleaned


def _format_augmented_context(results: List[Dict[str, str]], query: str = "") -> str:
    """
    Formatta i risultati come contesto fattuale per il system prompt.
    Il modello usa queste informazioni per arricchire la risposta.
    """
    lines = [
        "[DATI DA RICERCA WEB — usa queste informazioni per rispondere]",
        f"Domanda dell'utente: \"{query}\"" if query else "",
        "",
    ]
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        url = r.get("url", "")
        lines.append(f"Fonte {i}: {title}")
        if snippet:
            lines.append(f"  {snippet[:500]}")
        if url:
            lines.append(f"  URL: {url}")
        lines.append("")

    lines.extend([
        "ISTRUZIONI:",
        "- Rispondi in modo COMPLETO e DETTAGLIATO usando i dati sopra.",
        "- Struttura la risposta con paragrafi, elenchi puntati o sezioni Markdown.",
        "- Integra le informazioni da pi\u00f9 fonti per dare una risposta ricca.",
        "- NON citare le fonti per numero (l'utente non le vede).",
        "- NON inventare URL. NON inserire link nella risposta.",
        "- Se i dati non bastano a rispondere completamente, dichiaralo.",
    ])
    return "\n".join(lines)
