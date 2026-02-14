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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Set
from urllib.parse import quote_plus

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
    r"youtube|video\s+(?:di|della?|su)",
    re.IGNORECASE,
)

# ── Pattern per richieste MUSICALI ──────────────────────────────────────
_MUSIC_PATTERN = re.compile(
    r"(?:"
    r"canzon[ei]|brano|brani|musica|song|music|track|album|testo|lyric[s]?"
    r"|ascolta(?:re|mi)?|play|senti(?:re|mi)?"
    r"|spotify|soundcloud|apple\s*music|deezer|tidal|shazam"
    r"|concert[oi]|tour|live"
    r")",
    re.IGNORECASE,
)

# Pattern secondario: "[titolo] dei/di/degli [artista]" — indica una canzone
_SONG_ARTIST_HINT = re.compile(
    r"\b(?:dei|degli|delle|di|by)\s+\w+(?:\s+\w+)*\s*$",
    re.IGNORECASE,
)

# Termini che escludono una query musicale (notizie, meteo, informazioni, ecc.)
_NON_MUSIC_TERMS = re.compile(
    r"\b(?:notizi[ae]|meteo|prezz[oi]|informazion[ei]|politic[ao]|economi[ao]|"
    r"sport|ricett[ae]|orari[oi]|risultat[oi]|classifica|dove|come\s+(?:si|fare)|"
    r"perch[eé]|quand[oi]|quant[oiae]|cos['\u2019]|cosa|chi\s+(?:[eè]|ha)|"
    r"libro|film|serie\s+tv|telefilm|ristorante|hotel|volo|treno|negozio)\b",
    re.IGNORECASE,
)

# Piattaforme musicali per risultati multi-piattaforma
_MUSIC_PLATFORMS = [
    "youtube.com", "spotify.com", "genius.com", "apple.com/music",
    "soundcloud.com", "deezer.com", "shazam.com",
]

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


def is_music_query(message: str) -> bool:
    """Indica se la ricerca riguarda musica/canzoni.

    Due livelli di rilevamento:
    1. Parole chiave musicali esplicite (canzone, brano, ascolta, spotify, ecc.)
    2. Euristica strutturale: "[titolo] dei/di [artista]" in query di ricerca,
       purché non contenga termini chiaramente non musicali (notizie, meteo, ecc.)
    """
    if _MUSIC_PATTERN.search(message):
        return True
    # Euristica: ricerca esplicita + pattern "[titolo] di/dei [artista]"
    if _SEARCH_PATTERNS.search(message) and not _NON_MUSIC_TERMS.search(message):
        if _SONG_ARTIST_HINT.search(message):
            return True
    return False


# Parole filler da rimuovere per pulire la query di ricerca
_FILLER_WORDS = re.compile(
    r"\b(?:trovami|cercami|cercamelo|dammi|mostrami|apri|aprimi|"
    r"trova|cerca|cerca(?:re)?|per favore|please|puoi|potresti|"
    r"il link|un link|il video|"
    r"su|mi|me)\.?\b",
    re.IGNORECASE,
)

# Filler aggiuntivi SOLO per query NON musicali (non rimuovere 'canzone', 'brano' ecc. dalle query musicali)
_FILLER_NON_MUSIC = re.compile(
    r"\b(?:della canzone|canzone|brano|musica|song|music|"
    r"di|del|della|dello|la|le|lo|gli|un|una|dei|delle|degli)\b",
    re.IGNORECASE,
)

# Filler per domande fattuali (meno aggressivo, toglie solo la parte interrogativa)
_FACTUAL_FILLER = re.compile(
    r"\b(?:chi|cos|cosa|quand[oi]|quant[oiae]|dov|dove|qual|com|perch[eé])"
    r"(?:['’])?\s*(?:[eè]|era|sono|erano|ha|hanno|costa|costano|vale|pesa|si\s+trova)?\b",
    re.IGNORECASE,
)


def _clean_query(message: str, *, is_music: bool = False) -> str:
    """Rimuove parole filler dal messaggio per ottenere una query di ricerca pulita."""
    cleaned = _FILLER_WORDS.sub(" ", message)
    # Per query non musicali, rimuovi anche articoli e parole generiche
    if not is_music:
        cleaned = _FILLER_NON_MUSIC.sub(" ", cleaned)
    # Collassa spazi multipli
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Se troppo corta dopo il cleaning, usa l'originale
    if len(cleaned) < 3:
        return message.strip()
    return cleaned


# Pre-compiled regex for music query cleaning (avoid re-compilation per call)
_MUSIC_FILLER = re.compile(
    r"\b(?:trovami|cercami|cercamelo|dammi|mostrami|apri|aprimi|"
    r"trova|cerca|cercami|per favore|please|puoi|potresti|ascoltami|ascolta|play)\.?\b",
    re.IGNORECASE,
)
_MUSIC_PLATFORM_FILLER = re.compile(
    r"\bsu\s+(?:youtube|google|internet|web|spotify|deezer|soundcloud)\b",
    re.IGNORECASE,
)
_MUSIC_LEADING_ARTICLES = re.compile(
    r"^\s*(?:la|il|lo|una?|del(?:la)?|di)\s+",
    re.IGNORECASE,
)


def _clean_music_query(message: str) -> str:
    """Pulisce una query musicale preservando artista + titolo.

    Rimuove solo le parole di ricerca (cercami, trovami...) e
    piattaforme (su youtube, su spotify) ma mantiene tutto ciò
    che identifica la canzone: titolo, artista, 'canzone', 'brano'.
    """
    cleaned = _MUSIC_FILLER.sub(" ", message)
    cleaned = _MUSIC_PLATFORM_FILLER.sub(" ", cleaned)
    cleaned = _MUSIC_LEADING_ARTICLES.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) < 3:
        return message.strip()
    return cleaned


# ── Filtro snippet spazzatura (pagine generiche YouTube, placeholder) ──
_GARBAGE_PATTERNS = re.compile(
    r"О сервисе|Авторские права|Связаться с нами|Рекламодателям"
    r"|Enjoy the videos and music you love, upload original content"
    r"|Share your videos with friends, family, and the world"
    r"|Sign in to like videos, comment, and subscribe"
    r"|О сервисе Прессе",
    re.IGNORECASE,
)


def _is_garbage_snippet(snippet: str) -> bool:
    """Rileva snippet generici/placeholder (es. pagine YouTube boilerplate)."""
    if not snippet or len(snippet.strip()) < 15:
        return True
    return bool(_GARBAGE_PATTERNS.search(snippet))


def web_search(
    query: str,
    *,
    max_results: int = 5,
    region: str = "it-it",
    youtube: bool = False,
    music: bool = False,
) -> List[Dict[str, str]]:
    """
    Esegue una ricerca web via DuckDuckGo.

    Args:
        query: Testo da cercare
        max_results: Numero massimo di risultati (default 5)
        region: Regione per i risultati (default it-it)
        youtube: Se True, aggiunge "site:youtube.com" alla query
        music: Se True, ottimizza per risultati musicali multi-piattaforma

    Returns:
        Lista di dict con chiavi: title, url, snippet
        Lista vuota se la ricerca fallisce.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        logger.warning("ddgs non installato. Installa con: pip install ddgs")
        return []

    # Per YouTube e musica usare regione mondiale per risultati originali
    if youtube or music:
        region = "wt-wt"

    search_query = query
    if youtube:
        search_query = f"site:youtube.com {query}"
    elif music:
        # Per musica: cerca senza site:youtube.com e filtra YouTube in Python
        # (DuckDuckGo non supporta -site: in modo affidabile)
        q_lower = query.lower()
        has_music_term = any(
            w in q_lower for w in ("canzone", "brano", "song", "music", "album", "testo", "lyrics")
        )
        search_query = f"{query} song" if not has_music_term else query

    try:
        results = []
        with DDGS() as ddgs:
            # Per musica, richiedi più risultati per poi selezionare i migliori
            fetch_count = max_results * 2 if music else max_results
            for r in ddgs.text(search_query, region=region, max_results=fetch_count):
                snippet = html.unescape(r.get("body", ""))
                # Filtra risultati con snippet spazzatura (pagine generiche YouTube, ecc.)
                if _is_garbage_snippet(snippet):
                    continue
                results.append({
                    "title": html.unescape(r.get("title", "")),
                    "url": r.get("href", ""),
                    "snippet": snippet,
                })

        # Per musica: filtra YouTube in Python (già coperto dalla ricerca dedicata)
        if music and results:
            results = [r for r in results
                       if not any(yt in r.get("url", "").lower()
                                  for yt in ("youtube.com", "youtu.be"))]

        # Per musica: ordina privilegiando piattaforme musicali e risultati rilevanti
        if music and results:
            results = _sort_music_results(results)
            results = results[:max_results]

        logger.info("Web search: %d risultati per '%s'", len(results), search_query[:60])
        return results
    except Exception as e:
        logger.error("Errore web search: %s", e)
        return []


def _sort_music_results(results: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Ordina i risultati musicali privilegiando piattaforme di streaming/musica.

    Priorità: YouTube (video ufficiale) > Spotify > Genius (testi) > altre piattaforme > resto.
    """
    def _score(r: Dict[str, str]) -> int:
        url = r.get("url", "").lower()
        title = r.get("title", "").lower()
        score = 0
        # Piattaforme musicali (priorità massima)
        if "youtube.com" in url or "youtu.be" in url:
            score += 100
            if "official" in title or "ufficiale" in title:
                score += 50  # Video ufficiale in cima
            if "music video" in title or "video musicale" in title:
                score += 40
        elif "spotify.com" in url:
            score += 90
        elif "genius.com" in url:
            score += 80  # Testi
        elif "apple.com/music" in url or "music.apple.com" in url:
            score += 70
        elif "soundcloud.com" in url:
            score += 60
        elif "shazam.com" in url:
            score += 55
        elif "deezer.com" in url:
            score += 50
        # Penalizza pagine wiki e fan pages
        if "wikipedia.org" in url:
            score -= 20
        if "fandom.com" in url:
            score -= 30
        return score

    return sorted(results, key=_score, reverse=True)


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


def format_search_results_user(
    results: List[Dict[str, str]],
    query: str = "",
    music: bool = False,
) -> str:
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

        # Icona per piattaforma (solo per ricerche musicali)
        icon = ""
        if music and url:
            icon = _music_platform_icon(url)

        if url:
            lines.append(f"{i}. {icon}[{title}]({url})")
        else:
            lines.append(f"{i}. {icon}{title}")
        if snippet:
            # Tronca snippet e rimuovi newline
            clean_snippet = snippet[:180].replace("\n", " ").strip()
            if clean_snippet:
                lines.append(f"   {clean_snippet}")
    return "\n".join(lines)


def _music_platform_icon(url: str) -> str:
    """Restituisce un'icona emoji per la piattaforma musicale."""
    u = url.lower()
    if "youtube.com" in u or "youtu.be" in u:
        return "▶️ "
    if "spotify.com" in u:
        return "🎧 "
    if "genius.com" in u:
        return "📝 "
    if "apple.com/music" in u or "music.apple.com" in u:
        return "🍎 "
    if "soundcloud.com" in u:
        return "☁️ "
    if "deezer.com" in u:
        return "🎵 "
    if "shazam.com" in u:
        return "🔍 "
    return "🎵 "


# ── Post-filtro: rimuove URL inventati e riferimenti falsi ─────────────
_MD_LINK_RE = re.compile(r"\[([^\]]*?)\]\((https?://[^)]+)\)")
_BARE_URL_RE = re.compile(r"(?<!\()https?://[^\s)]+")  # Lookbehind: skip URLs inside markdown (

# Pattern per righe che sembrano riferimenti/bibliografia falsi
_FAKE_REF_PATTERNS = re.compile(
    r"^\s*(?:"
    r"(?:[-•*]\s*)?(?:Fonte|Fonti|Source|Sources|Riferiment[oi]|Reference|Link|Per saperne di più|Per maggiori informazioni|Ulteriori informazioni)\s*:?.*"
    r"|(?:[-•*]\s*)?\S+.*-\s*Wikipedia.*"
    r"|(?:[-•*]\s*)?@\w+.*"
    r"|(?:[-•*]\s*)?\[?(?:Fonte|Source)\s*\d+\]?.*"
    r"|(?:[-•*]\s*)?(?:Vedi anche|See also|Leggi anche|Read more)\s*:?.*"
    r")\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def strip_hallucinated_urls(text: str, allowed_urls: Set[str]) -> str:
    """
    Rimuove dal testo ogni URL non presente in allowed_urls.

    - Link Markdown [titolo](url) → mantiene solo il titolo
    - URL bare (https://...) → rimosso
    """
    def _replace_md_link(m: re.Match) -> str:
        title, url = m.group(1), m.group(2)
        if url in allowed_urls:
            return m.group(0)  # URL reale, mantieni
        return title  # Solo il titolo, senza link

    text = _MD_LINK_RE.sub(_replace_md_link, text)
    text = _BARE_URL_RE.sub(
        lambda m: m.group(0) if m.group(0) in allowed_urls else "",
        text,
    )
    # Rimuovi righe che sembrano riferimenti/bibliografia inventati
    text = _strip_trailing_references(text)
    # Pulisci spazi doppi residui
    text = re.sub(r"  +", " ", text)
    # Rimuovi righe vuote multiple consecutive
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_trailing_references(text: str) -> str:
    """
    Rimuove dall'output del modello le righe finali che sembrano
    riferimenti/fonti/link inventati (es. 'Donald Trump - Wikipedia',
    '@handle', 'Fonti:', etc.).
    """
    lines = text.rstrip().split("\n")
    # Scorri dal basso: rimuovi righe vuote e righe che matchano i pattern
    while lines:
        last = lines[-1].strip()
        if not last:
            lines.pop()
            continue
        if _FAKE_REF_PATTERNS.match(last):
            lines.pop()
            continue
        # Riga con solo un separatore (---, ===, ***)
        if re.match(r"^[-=*_]{3,}\s*$", last):
            lines.pop()
            continue
        # Riga che introduce una lista di link (es. "trovate nei seguenti link:")
        if re.search(
            r"(?:seguenti|questi|queste)\s+(?:link|fonti|riferimenti|risorse)"
            r"|(?:following|these)\s+(?:links|sources|references)"
            r"|(?:per\s+(?:ulteriori|maggiori|pi[uù])\s+(?:informazioni|dettagli|approfondimenti))"
            r"|(?:(?:puoi|potete)\s+(?:consultare|visitare|trovare|leggere))",
            last, re.IGNORECASE,
        ):
            lines.pop()
            continue
        break
    return "\n".join(lines)


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
    from core.github_search import (
        is_code_query, search_repositories, search_code,
        format_github_context, format_github_user,
        clean_code_query, detect_language,
    )

    explicit = needs_web_search(message)
    factual = needs_factual_search(message)
    code = is_code_query(message)

    if not explicit and not factual and not code:
        return None

    youtube = is_youtube_query(message) if explicit else False
    music = is_music_query(message)

    # Pulizia query: musica usa pulizia dedicata, altrimenti standard
    if music and explicit:
        clean_q = _clean_music_query(message)
    elif explicit:
        clean_q = _clean_query(message, is_music=music)
    else:
        clean_q = _clean_factual_query(message)

    # ── Lancia tutte le ricerche in parallelo ──────────────────────────
    # Ogni sorgente (GitHub, Wikipedia, DuckDuckGo) è indipendente.
    # ThreadPoolExecutor elimina l'attesa seriale (da ~4-6s a ~1-2s).
    github_data = None
    wiki_extract = None
    results = []

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {}

        # GitHub (repos + code in parallelo)
        if code and not music:
            code_q = clean_code_query(message)
            lang = detect_language(message)
            futures["gh_repos"] = pool.submit(
                search_repositories, code_q, max_results=max_results, language=lang,
            )
            futures["gh_code"] = pool.submit(
                search_code, code_q, max_results=3, language=lang,
            )

        # Wikipedia + DuckDuckGo in parallelo per domande fattuali
        if factual and not youtube and not music:
            futures["wiki"] = pool.submit(_fetch_wikipedia_extract, clean_q)
            futures["ddg"] = pool.submit(
                web_search, clean_q, max_results=max_results, region="it-it",
            )
        elif music:
            # YouTube e generale in parallelo
            futures["yt"] = pool.submit(
                web_search, clean_q, max_results=2, youtube=True,
            )
            futures["music_general"] = pool.submit(
                web_search, clean_q, max_results=max_results, music=True,
            )
        elif explicit or factual:
            futures["ddg"] = pool.submit(
                web_search, clean_q, max_results=max_results, youtube=youtube,
            )

        # Raccogli risultati
        for key, fut in futures.items():
            try:
                val = fut.result(timeout=12)
            except Exception as e:
                logger.warning("Ricerca '%s' fallita: %s", key, e)
                val = None

            if key == "gh_repos":
                if github_data is None:
                    github_data = {"repos": [], "code": []}
                github_data["repos"] = val or []
            elif key == "gh_code":
                if github_data is None:
                    github_data = {"repos": [], "code": []}
                github_data["code"] = val or []
            elif key == "wiki":
                wiki_extract = val
            elif key == "ddg":
                results = val or []
            elif key == "yt":
                results.extend(val or [])
            elif key == "music_general":
                # Merge senza duplicati
                seen = {r.get("url") for r in results if r.get("url")}
                for r in (val or []):
                    url = r.get("url", "")
                    if url and url not in seen:
                        seen.add(url)
                        results.append(r)

    # Fallback DuckDuckGo se prima ricerca vuota (regione diversa)
    if factual and not youtube and not music and not results:
        results = web_search(clean_q, max_results=max_results)

    # Post-processing musicale
    if music and results:
        results = _sort_music_results(results)
        results = results[:max_results]

    if github_data:
        logger.info("GitHub search: repos=%d code=%d per '%s'",
                     len(github_data.get("repos", [])),
                     len(github_data.get("code", [])),
                     clean_q[:60])

    # Se né web né GitHub hanno prodotto risultati, ritorna None
    has_github = github_data and (github_data.get("repos") or github_data.get("code"))
    if not results and not has_github:
        return None

    urls = {r["url"] for r in results if r.get("url")}
    # Aggiungi URL GitHub al set di URL consentiti
    if has_github:
        for r in github_data.get("repos", []):
            if r.get("url"):
                urls.add(r["url"])
        for c in github_data.get("code", []):
            if c.get("url"):
                urls.add(c["url"])

    if explicit:
        # Modalità "links": risultati diretti, modello saltato
        user_parts = []
        if has_github:
            user_parts.append(format_github_user(github_data, query=message))
        if results:
            web_fmt = format_search_results_user(results, query=message, music=music)
            if has_github:
                user_parts.append("\n**🌐 Web:**\n" + web_fmt)
            else:
                user_parts.append(web_fmt)
        return {
            "mode": "links",
            "user": "\n".join(user_parts),
            "urls": urls,
        }
    else:
        # Modalità "augmented": contesto per il modello
        context_parts = []
        if has_github:
            context_parts.append(format_github_context(github_data, query=message))
        context_parts.append(
            _format_augmented_context(
                results, query=message, wiki_extract=wiki_extract,
            )
        )
        return {
            "mode": "augmented",
            "context": "\n\n".join(context_parts),
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


def _format_augmented_context(
    results: List[Dict[str, str]],
    query: str = "",
    wiki_extract: Optional[str] = None,
) -> str:
    """
    Formatta i risultati come contesto fattuale per il system prompt.
    Se disponibile, include l'estratto Wikipedia come fonte principale.
    """
    lines = [
        "[DATI DA RICERCA WEB — usa queste informazioni per rispondere]",
        f"Domanda dell'utente: \"{query}\"" if query else "",
        "",
    ]

    # Wikipedia come fonte principale (contenuto ricco, ~1000-3000 chars)
    if wiki_extract:
        lines.append("═══ CONTENUTO WIKIPEDIA (fonte principale) ═══")
        lines.append(wiki_extract)
        lines.append("")
        lines.append("═══ ALTRE FONTI WEB ═══")

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
        "",
        "═══ ISTRUZIONI OBBLIGATORIE ═══",
        "LINGUA: Rispondi in italiano corretto e naturale.",
        "CONTENUTO:",
        "- PARAFRASA fedelmente i dati forniti sopra. NON inventare fatti.",
        "- Se c'è contenuto Wikipedia, quello è la tua fonte principale.",
        "- Copia date, nomi, numeri ESATTAMENTE come appaiono nei dati.",
        "- NON aggiungere informazioni che non sono nei dati forniti.",
        "- Se una informazione non è nei dati, puoi integrarla SOLO se sei sicuro.",
        "FORMATO:",
        "- Struttura con paragrafi, elenchi puntati, sezioni Markdown.",
        "- Copri: biografia/definizione, fatti chiave, contesto storico, impatto.",
        "DIVIETI ASSOLUTI:",
        "- NON aggiungere sezioni 'Fonti', 'Riferimenti', 'Link utili' alla fine.",
        "- NON citare Wikipedia, fonti web o nomi di siti.",
        "- NON inventare URL. NON inserire link nella risposta.",
        "- NON scrivere handle social (@...) o nomi di account.",
        "- NON aggiungere una lista di link/fonti alla fine della risposta.",
    ])
    return "\n".join(lines)


# ── Wikipedia API — contenuto ricco per domande fattuali ───────────────

_WIKI_API = "https://{lang}.wikipedia.org/w/api.php"


def _fetch_wikipedia_extract(
    query: str,
    *,
    lang: str = "it",
    max_chars: int = 3000,
) -> Optional[str]:
    """
    Cerca su Wikipedia e restituisce l'introduzione dell'articolo più rilevante.

    Usa l'API MediaWiki (opensearch → query/extracts). Nessuna dipendenza
    aggiuntiva: solo requests (già usato da ddgs).

    Returns:
        Testo dell'introduzione (fino a max_chars), o None se non trovato.
    """
    try:
        import requests
    except ImportError:
        logger.warning("requests non installato — skip Wikipedia")
        return None

    try:
        headers = {"User-Agent": "OmniEyeAI/1.0 (local assistant; Python/requests)"}

        # Step 1: OpenSearch per trovare il titolo esatto
        search_url = _WIKI_API.format(lang=lang)
        search_resp = requests.get(
            search_url,
            params={
                "action": "opensearch",
                "search": query,
                "limit": 1,
                "format": "json",
            },
            headers=headers,
            timeout=5,
        )
        search_resp.raise_for_status()
        search_data = search_resp.json()

        # opensearch ritorna: [query, [titles], [descriptions], [urls]]
        if not search_data[1]:
            if lang != "en":
                logger.info("Wikipedia IT: nessun risultato per '%s', provo EN", query)
                return _fetch_wikipedia_extract(query, lang="en", max_chars=max_chars)
            return None

        title = search_data[1][0]

        # Step 2: Ottieni l'estratto completo dell'introduzione
        extract_resp = requests.get(
            search_url,
            headers=headers,
            params={
                "action": "query",
                "prop": "extracts",
                "exintro": "1",
                "explaintext": "1",
                "titles": title,
                "format": "json",
                "exsectionformat": "plain",
            },
            timeout=5,
        )
        extract_resp.raise_for_status()
        pages = extract_resp.json().get("query", {}).get("pages", {})

        for page in pages.values():
            extract = page.get("extract", "")
            if not extract:
                continue
            # Tronca al confine di frase se troppo lungo
            if len(extract) > max_chars:
                cut = extract[:max_chars].rfind(". ")
                if cut > max_chars // 2:
                    extract = extract[: cut + 1]
                else:
                    extract = extract[:max_chars] + "..."
            logger.info(
                "Wikipedia [%s]: %d chars per '%s'", lang, len(extract), title,
            )
            return extract

        return None

    except Exception as e:
        logger.warning("Wikipedia API error: %s", e)
        return None
