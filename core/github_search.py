"""
GitHub Search — ricerca codice e repository per Omni Eye AI

Usa le GitHub REST API (no auth per ricerche base, 10 req/min).
Fornisce contesto di codice reale al modello per migliorare le risposte di programmazione.

Due tipi di ricerca:
  • Repository — trova progetti rilevanti (nome, description, stars, linguaggio)
  • Code       — trova snippet di codice reali con contesto
"""

import logging
import os
import re
from typing import List, Dict, Optional
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

_GITHUB_API = "https://api.github.com"
_GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "").strip()

_HEADERS = {
    "Accept": "application/vnd.github.v3+json",
    "User-Agent": "OmniEyeAI/1.0 (local assistant)",
}
if _GITHUB_TOKEN:
    _HEADERS["Authorization"] = f"token {_GITHUB_TOKEN}"
    logger.info("GitHub token trovato — rate limit: 5000 req/ora")
else:
    logger.info("Nessun GitHub token — rate limit: 60 req/ora")

# ── Pattern per rilevare query legate a codice/programmazione ───────────
_CODE_QUERY_PATTERN = re.compile(
    r"\b(?:"
    # Richieste di codice esplicite
    r"scrivi\s+(?:un\s+)?(?:codice|script|programma|funzione|classe|metodo)"
    r"|genera\s+(?:un\s+)?(?:codice|script|programma|funzione|classe)"
    r"|implementa|refactor|debug(?:ga)?|codice"
    # Concetti tecnici
    r"|come\s+(?:si\s+)?(?:fa|implementa|crea|usa|scrive)\s"
    r"|esempio\s+(?:di\s+)?(?:codice|implementazione|uso)"
    r"|libreria|framework|package|modulo"
    r"|api\b|endpoint|database|query|regex"
    r"|algoritmo|struttura\s+dati"
    # Linguaggi
    r"|python|javascript|typescript|java\b|c\+\+|c#|rust|golang|go\b|ruby"
    r"|html|css|sql|bash|shell|powershell|kotlin|swift|dart|php"
    r"|react|angular|vue|django|flask|fastapi|express|nextjs|node\.?js"
    # GitHub esplicito
    r"|github|repository|repo\b|open\s*source"
    r"|git\b|commit|branch|merge|pull\s*request"
    r"|pip\s+install|npm\s+install|cargo\s+add"
    r")\b",
    re.IGNORECASE,
)

# Parole che indicano una ricerca informativa (non codice)
_NON_CODE_TERMS = re.compile(
    r"\b(?:chi\s+[eè]|cos['\u2019]\s*[eè]|quand[oi]|dov['\u2019]\s*[eè]|perch[eé]"
    r"|notizi[ae]|meteo|prezz[oi]|canzon[ei]|brano|film|serie\s+tv"
    r"|ristorante|hotel|volo|ricett[ae])\b",
    re.IGNORECASE,
)


def is_code_query(message: str) -> bool:
    """Rileva se il messaggio è una domanda di programmazione/codice."""
    if _NON_CODE_TERMS.search(message):
        return False
    return bool(_CODE_QUERY_PATTERN.search(message))


def search_repositories(
    query: str,
    *,
    max_results: int = 5,
    language: str = "",
) -> List[Dict[str, str]]:
    """
    Cerca repository su GitHub.

    Returns:
        Lista di dict con: name, url, description, stars, language, topics
    """
    try:
        import requests
    except ImportError:
        logger.warning("requests non installato — skip GitHub search")
        return []

    search_q = query
    if language:
        search_q += f" language:{language}"

    try:
        resp = requests.get(
            f"{_GITHUB_API}/search/repositories",
            headers=_HEADERS,
            params={
                "q": search_q,
                "sort": "stars",
                "order": "desc",
                "per_page": max_results,
            },
            timeout=8,
        )

        # Rate limit check
        if resp.status_code == 403:
            logger.warning("GitHub API rate limit raggiunto")
            return []
        resp.raise_for_status()

        data = resp.json()
        results = []
        for item in data.get("items", [])[:max_results]:
            results.append({
                "name": item.get("full_name", ""),
                "url": item.get("html_url", ""),
                "description": (item.get("description") or "")[:200],
                "stars": _format_stars(item.get("stargazers_count", 0)),
                "language": item.get("language") or "",
                "topics": ", ".join(item.get("topics", [])[:5]),
            })

        logger.info("GitHub repos: %d risultati per '%s'", len(results), query[:60])
        return results

    except Exception as e:
        logger.error("GitHub repo search error: %s", e)
        return []


def search_code(
    query: str,
    *,
    max_results: int = 3,
    language: str = "",
) -> List[Dict[str, str]]:
    """
    Cerca snippet di codice su GitHub.

    Note: l'API code search richiede almeno un qualificatore (language, repo, ecc.)
    e NON funziona senza autenticazione per risultati di codice.
    Usiamo DuckDuckGo con site:github.com come fallback affidabile.

    Returns:
        Lista di dict con: title, url, snippet
    """
    try:
        from ddgs import DDGS
    except ImportError:
        logger.warning("ddgs non installato — skip GitHub code search")
        return []

    search_query = f"site:github.com {query}"
    if language:
        search_query += f" {language}"

    try:
        import html as html_mod
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(search_query, region="wt-wt", max_results=max_results * 2):
                url = r.get("href", "")
                # Filtra solo risultati github.com reali
                if "github.com" not in url.lower():
                    continue
                snippet = html_mod.unescape(r.get("body", ""))
                if not snippet or len(snippet.strip()) < 15:
                    continue
                results.append({
                    "title": html_mod.unescape(r.get("title", "")),
                    "url": url,
                    "snippet": snippet[:300],
                })
                if len(results) >= max_results:
                    break

        logger.info("GitHub code: %d risultati per '%s'", len(results), query[:60])
        return results

    except Exception as e:
        logger.error("GitHub code search error: %s", e)
        return []


def search_github(
    query: str,
    *,
    max_results: int = 5,
    language: str = "",
) -> Dict[str, list]:
    """
    Ricerca combinata GitHub: repository + snippet di codice.

    Returns:
        Dict con chiavi 'repos' e 'code', ciascuna una lista di risultati.
    """
    repos = search_repositories(query, max_results=max_results, language=language)
    code = search_code(query, max_results=3, language=language)
    return {"repos": repos, "code": code}


def format_github_context(
    github_data: Dict[str, list],
    query: str = "",
) -> str:
    """
    Formatta i risultati GitHub come contesto per il system prompt.
    Iniettato in modalità augmented per domande di codice.
    """
    repos = github_data.get("repos", [])
    code = github_data.get("code", [])

    if not repos and not code:
        return ""

    lines = [
        "═══ RISULTATI GITHUB ═══",
        "",
    ]

    if repos:
        lines.append("📦 **Repository rilevanti:**")
        for i, r in enumerate(repos, 1):
            name = r.get("name", "")
            url = r.get("url", "")
            desc = r.get("description", "")
            stars = r.get("stars", "")
            lang = r.get("language", "")
            info_parts = []
            if stars:
                info_parts.append(f"⭐ {stars}")
            if lang:
                info_parts.append(lang)
            info = " | ".join(info_parts)
            lines.append(f"  {i}. **{name}** ({info})")
            if desc:
                lines.append(f"     {desc}")
            lines.append(f"     {url}")
        lines.append("")

    if code:
        lines.append("💻 **Snippet di codice:**")
        for i, c in enumerate(code, 1):
            title = c.get("title", "")
            url = c.get("url", "")
            snippet = c.get("snippet", "")
            lines.append(f"  {i}. {title}")
            if snippet:
                lines.append(f"     {snippet[:250]}")
            lines.append(f"     {url}")
        lines.append("")

    return "\n".join(lines)


def format_github_user(
    github_data: Dict[str, list],
    query: str = "",
) -> str:
    """
    Formatta i risultati GitHub come Markdown per l'utente (mode=links).
    """
    repos = github_data.get("repos", [])
    code = github_data.get("code", [])

    if not repos and not code:
        return f"Nessun risultato GitHub per: *{query}*" if query else "Nessun risultato trovato."

    lines = []
    i = 1

    if repos:
        lines.append("**📦 Repository:**")
        for r in repos:
            name = r.get("name", "")
            url = r.get("url", "")
            desc = r.get("description", "")
            stars = r.get("stars", "")
            lang = r.get("language", "")
            info_parts = []
            if stars:
                info_parts.append(f"⭐ {stars}")
            if lang:
                info_parts.append(lang)
            info = " · ".join(info_parts)
            lines.append(f"{i}. [{name}]({url}) — {info}")
            if desc:
                lines.append(f"   {desc[:150]}")
            i += 1

    if code:
        if repos:
            lines.append("")
        lines.append("**💻 Codice:**")
        for c in code:
            title = c.get("title", "")
            url = c.get("url", "")
            snippet = c.get("snippet", "")
            lines.append(f"{i}. [{title}]({url})")
            if snippet:
                lines.append(f"   {snippet[:150]}")
            i += 1

    return "\n".join(lines)


def _format_stars(count: int) -> str:
    """Formatta il conteggio stelle in modo leggibile (es. 12.5k)."""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}k"
    return str(count)


# ── Pulizia query per GitHub ───────────────────────────────────────────

_CODE_FILLER = re.compile(
    r"\b(?:trovami|cercami|mostrami|dammi|scrivi(?:mi)?|genera(?:mi)?|"
    r"cerca|trova|per favore|please|puoi|potresti|"
    r"come\s+(?:si\s+)?(?:fa|implementa|crea|usa)|"
    r"esempio\s+(?:di|per)|codice\s+(?:per|di|che)|"
    r"su\s+github|repository\s+(?:per|di)|"
    r"un|una|il|lo|la|le|gli|dei|delle|degli|di|del|della|dello)\b",
    re.IGNORECASE,
)


def clean_code_query(message: str) -> str:
    """Pulisce una query di codice per la ricerca GitHub."""
    cleaned = _CODE_FILLER.sub(" ", message)
    cleaned = re.sub(r"[?!.,;:]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) < 3:
        return message.strip()
    return cleaned


def detect_language(message: str) -> str:
    """Rileva il linguaggio di programmazione menzionato nella query."""
    lang_map = {
        r"\bpython\b": "Python",
        r"\bjavascript\b|\bjs\b": "JavaScript",
        r"\btypescript\b|\bts\b": "TypeScript",
        r"\bjava\b": "Java",
        r"\bc\+\+|\bcpp\b": "C++",
        r"\bc#\b|\bcsharp\b": "C#",
        r"\brust\b": "Rust",
        r"\bgolang\b|\bgo\b": "Go",
        r"\bruby\b": "Ruby",
        r"\bphp\b": "PHP",
        r"\bswift\b": "Swift",
        r"\bkotlin\b": "Kotlin",
        r"\bdart\b": "Dart",
        r"\bsql\b": "SQL",
        r"\bbash\b|\bshell\b": "Shell",
        r"\bhtml\b": "HTML",
        r"\bcss\b": "CSS",
    }
    for pattern, lang in lang_map.items():
        if re.search(pattern, message, re.IGNORECASE):
            return lang
    return ""
