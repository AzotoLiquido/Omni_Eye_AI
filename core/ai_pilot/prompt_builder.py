"""
Prompt Builder - Genera system prompt dinamici dalla configurazione Pilot
"""

from typing import Dict, List
from .config_loader import PilotConfig


# =========================================================================
# Mappe stile → istruzioni per il modello
# =========================================================================

_TONE_INSTRUCTIONS = {
    "terminal": (
        "Rispondi in stile terminale: conciso, diretto, tecnico. "
        "Usa il prefisso configurato per ogni blocco di output. "
        "Niente fronzoli, niente emoji, solo dati."
    ),
    "neutro": (
        "Rispondi in modo neutro e professionale. "
        "Mantieni un tono oggettivo e informativo."
    ),
    "amichevole": (
        "Rispondi in modo cordiale e accessibile. "
        "Usa un linguaggio semplice senza essere banale."
    ),
    "formale": (
        "Rispondi in modo formale e accademico. "
        "Usa un linguaggio ricercato e strutturato."
    ),
}

_VERBOSITY_MAP = {
    0: "Risposte minime: solo il dato richiesto, niente altro.",
    1: "Risposte estremamente brevi: una o due frasi al massimo.",
    2: "Risposte brevi e concise: poche frasi mirate.",
    3: "Risposte moderate: brevi paragrafi con i punti chiave.",
    5: "Risposte bilanciate: spiegazioni chiare con esempi quando utile.",
    7: "Risposte dettagliate: approfondimenti, esempi, alternative.",
    10: "Risposte esaustive: copri ogni aspetto, con esempi completi.",
}


def _nearest_verbosity(level: int) -> str:
    """Trova il livello di verbosità più vicino nella mappa"""
    keys = sorted(_VERBOSITY_MAP.keys())
    closest = min(keys, key=lambda k: abs(k - level))
    return _VERBOSITY_MAP[closest]


# =========================================================================
# Builder principale
# =========================================================================

class PromptBuilder:
    """Costruisce system prompt completi dalla configurazione Pilot"""

    def __init__(self, cfg: PilotConfig):
        self.cfg = cfg

    def build_system_prompt(
        self,
        memory_context: str = "",
        available_tools: List[Dict] = None,
        extra_instructions: str = "",
    ) -> str:
        """
        Genera il system prompt completo.

        Args:
            memory_context:     Fatti/documenti rilevanti recuperati dalla memoria
            available_tools:    Lista tool disponibili per questa richiesta
            extra_instructions: Istruzioni aggiuntive ad-hoc

        Returns:
            System prompt assemblato
        """
        sections: List[str] = []

        # 1. Identità
        sections.append(self._section_identity())

        # 2. Stile e formattazione
        sections.append(self._section_style())

        # 3. Lingua
        sections.append(self._section_language())

        # 4. Politiche di sicurezza
        sections.append(self._section_safety())

        # 5. Strumenti disponibili
        if available_tools:
            sections.append(self._section_tools(available_tools))

        # 6. Contesto dalla memoria
        if memory_context:
            sections.append(self._section_memory(memory_context))

        # 7. Istruzioni extra
        if extra_instructions:
            sections.append(f"[ISTRUZIONI AGGIUNTIVE]\n{extra_instructions}")

        # 8. Formato output
        sections.append(self._section_output())

        return "\n\n".join(s for s in sections if s)

    # ------------------------------------------------------------------
    # Sezioni del prompt
    # ------------------------------------------------------------------

    def _section_identity(self) -> str:
        desc = self.cfg.raw["meta"].get("description", "")
        lines = [
            f"[IDENTITÀ]",
            f"Nome: {self.cfg.name}",
            f"Versione: {self.cfg.version}",
        ]
        if desc:
            lines.append(f"Descrizione: {desc}")
        return "\n".join(lines)

    def _section_style(self) -> str:
        tone_instr = _TONE_INSTRUCTIONS.get(self.cfg.tone, _TONE_INSTRUCTIONS["neutro"])
        verb_instr = _nearest_verbosity(self.cfg.verbosity)

        lines = [
            "[STILE]",
            f"Tono: {tone_instr}",
            f"Verbosità: {verb_instr}",
        ]
        fmt = self.cfg.formatting
        if fmt.get("use_lists"):
            lines.append("- Usa elenchi puntati quando elenchi informazioni.")
        if fmt.get("use_tables"):
            lines.append("- Usa tabelle per dati strutturati.")
        if fmt.get("code_fences"):
            lines.append("- Racchiudi il codice in blocchi ``` con il linguaggio indicato.")

        return "\n".join(lines)

    def _section_language(self) -> str:
        lines = [
            "[LINGUA]",
            f"Lingua principale: {self.cfg.primary_language}",
            "Rispondi SEMPRE nella lingua principale."
        ]
        if self.cfg.avoid_english:
            lines.append(
                "Evita termini inglesi se esiste un equivalente nella lingua principale."
            )
            glossary = self.cfg.glossary
            if glossary:
                lines.append("Glossario sostitutivo:")
                for eng, ita in glossary.items():
                    lines.append(f"  '{eng}' → '{ita}'")
        return "\n".join(lines)

    def _section_safety(self) -> str:
        cats = self.cfg.refuse_categories
        lines = [
            "[SICUREZZA]",
            f"Rifiuta richieste relative a: {', '.join(cats)}.",
        ]
        if self.cfg.redact_secrets:
            lines.append(
                "Non mostrare mai credenziali, chiavi API, password o token nei tuoi output."
            )
        if self.cfg.pii_handling == "strict_redaction":
            lines.append("Oscura qualsiasi dato personale identificabile (PII) negli output.")
        elif self.cfg.pii_handling == "minimize":
            lines.append("Minimizza l'uso di dati personali nelle risposte.")
        return "\n".join(lines)

    def _section_tools(self, tools: List[Dict]) -> str:
        """Istruzioni sul formato ReAct per l'uso dei tool"""
        lines = [
            "[STRUMENTI DISPONIBILI]",
            "Puoi usare questi strumenti per rispondere. "
            "Per invocare uno strumento, utilizza ESATTAMENTE questo formato:",
            "",
            "Pensiero: <ragionamento su cosa fare>",
            "Azione: <nome_tool>(<parametri JSON>)",
            "---",
            "",
            "Dopo aver ricevuto il risultato (Osservazione), "
            "puoi fare un'altra Azione oppure dare la Risposta Finale.",
            "",
            "Rispondi con 'Risposta Finale: <testo>' quando hai abbastanza informazioni.",
            "",
            "Strumenti:",
        ]
        for t in tools:
            tid = t["id"]
            name = t.get("name", tid)
            desc = t.get("description", "")
            lines.append(f"  - {tid} ({name}): {desc}")

        return "\n".join(lines)

    def _section_memory(self, memory_context: str) -> str:
        return (
            "[CONTESTO DALLA MEMORIA]\n"
            "Informazioni rilevanti recuperate dalla memoria:\n"
            f"{memory_context}"
        )

    def _section_output(self) -> str:
        fmt = self.cfg.output_format
        prefix = self.cfg.terminal_prefix
        lines = [
            "[FORMATO OUTPUT]",
            f"Formato predefinito: {fmt}",
        ]
        if self.cfg.tone == "terminal" and prefix:
            lines.append(f"Prefisso output: '{prefix}'")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Prompt specializzati
    # ------------------------------------------------------------------

    def build_summarization_prompt(self, text: str) -> str:
        """Prompt per riassumere testo (contesto conversazione)"""
        return (
            "Riassumi il seguente testo in modo conciso ma completo. "
            "Mantieni fatti, nomi, date e informazioni chiave. "
            "Scrivi solo il riassunto:\n\n"
            f"{text}"
        )

    def build_entity_extraction_prompt(self, message: str) -> str:
        """Prompt per estrarre entità/fatti da un messaggio utente"""
        return (
            "Analizza questo messaggio ed estrai fatti memorizzabili. "
            "Restituisci SOLO un JSON con questa struttura (niente altro testo):\n"
            '{"facts": [{"key": "breve_etichetta", "value": "informazione"}]}\n'
            "Se non ci sono fatti nuovi, restituisci: {\"facts\": []}\n\n"
            f"Messaggio: {message}"
        )

    def build_tool_decision_prompt(self, user_message: str, tools: List[Dict]) -> str:
        """Prompt per decidere se usare tool per una richiesta"""
        tool_names = [t["id"] for t in tools]
        return (
            f"L'utente chiede: \"{user_message}\"\n\n"
            f"Strumenti disponibili: {', '.join(tool_names)}\n\n"
            "Questa richiesta necessita di uno strumento? "
            "Rispondi SOLO 'SI' o 'NO' e il nome dello strumento se SI.\n"
            "Formato: SI:nome_tool oppure NO"
        )
