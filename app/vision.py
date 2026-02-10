"""
Vision Pipeline - Gestione analisi immagini con modelli vision

Estratto da main.py per mantenere la separazione delle responsabilità.
Contiene: costanti vision, _build_vision_prompt, _user_visible_models, VISION_SYSTEM_PROMPT.
"""

import logging

logger = logging.getLogger(__name__)

# Modelli usati solo come backend per vision; nascosti dalla selezione utente
VISION_ONLY_TAGS = ('moondream', 'bakllava', 'minicpm-v')

# Modelli vision multilingue: possono rispondere direttamente in qualsiasi lingua
# → pipeline singola (no fase intermedia in inglese)
MULTILINGUAL_VISION = ('minicpm-v', 'llava:13b', 'llava-llama3')

# Modelli vision EN-only: descrivono solo in inglese
# → pipeline a 2 fasi (descrizione EN + risposta IT via modello testo)
EN_ONLY_VISION = ('moondream', 'bakllava', 'llava-phi3', 'llava:7b', 'llava')

# Ordine di priorità modelli vision (il primo disponibile viene usato)
VISION_PRIORITY = ('minicpm-v', 'llava:13b', 'llava-llama3', 'llava:7b',
                   'llava', 'moondream', 'bakllava', 'vision')

# System prompt condiviso per analisi visiva (module-level, non ricostruito ad ogni request)
VISION_SYSTEM_PROMPT = (
    "# Ruolo\n"
    "Sei un analista visivo esperto. Il tuo compito è analizzare immagini "
    "con la massima precisione possibile.\n\n"
    "# Istruzioni\n"
    "1. Descrivi ESATTAMENTE ciò che vedi, senza inventare dettagli.\n"
    "2. Se c'è testo visibile, trascrivilo fedelmente.\n"
    "3. Specifica posizioni spaziali (in alto, a sinistra, sullo sfondo...).\n"
    "4. Distingui ciò che è certo da ciò che è incerto (\"sembra\", \"potrebbe\").\n"
    "5. Rispondi nella lingua dell'utente.\n"
    "6. NON aggiungere informazioni che non derivano dall'immagine.\n"
    "7. Sii conciso ma completo. Non ripetere le stesse informazioni."
)


def user_visible_models(ai_engine) -> list:
    """Restituisce solo i modelli selezionabili dall'utente (esclude vision-only)."""
    return [m for m in ai_engine.list_available_models()
            if not any(tag in m.lower() for tag in VISION_ONLY_TAGS)]


def vision_model_priority(name: str) -> int:
    """Chiave di ordinamento per priorità modelli vision."""
    low = name.lower()
    for i, tag in enumerate(VISION_PRIORITY):
        if tag in low:
            return i
    return len(VISION_PRIORITY)


def build_vision_prompt(user_message: str, multilingual: bool) -> str:
    """Costruisce un prompt vision adattivo in base alla domanda dell'utente.

    Analizza il contenuto della domanda per capire cosa l'utente vuole sapere
    e genera un prompt specializzato (OCR, persone, scena, tecnico, ecc.).

    Args:
        user_message: La domanda dell'utente sull'immagine
        multilingual: Se True, il prompt include istruzioni di lingua
    """
    msg = (user_message or "").lower()

    # Rilevamento intento dalla domanda utente
    ocr_keywords = ('testo', 'scritto', 'scritta', 'leggi', 'leggere', 'parole',
                    'scrivi', 'text', 'read', 'ocr', 'lettere', 'titolo',
                    'etichetta', 'label', 'caption', 'didascalia')
    people_keywords = ('persona', 'persone', 'gente', 'viso', 'volto', 'faccia',
                       'chi', 'uomo', 'donna', 'bambino', 'people', 'person',
                       'face', 'who', 'espressione', 'emozione')
    tech_keywords = ('codice', 'code', 'programma', 'errore', 'error', 'bug',
                     'screenshot', 'schermo', 'screen', 'interfaccia', 'ui',
                     'terminale', 'terminal', 'console', 'log')
    food_keywords = ('cibo', 'piatto', 'mangiare', 'ricetta', 'ingredienti',
                     'food', 'dish', 'recipe', 'cucina')
    location_keywords = ('dove', 'luogo', 'posto', 'citt', 'paese', 'location',
                         'where', 'edificio', 'building', 'strada', 'via')

    has_ocr = any(k in msg for k in ocr_keywords)
    has_people = any(k in msg for k in people_keywords)
    has_tech = any(k in msg for k in tech_keywords)
    has_food = any(k in msg for k in food_keywords)
    has_location = any(k in msg for k in location_keywords)

    # Costruisci prompt specializzato
    parts = []

    if multilingual:
        if has_ocr:
            parts.append(
                "Analizza attentamente l'immagine e trascrivi TUTTO il testo visibile, "
                "inclusi titoli, etichette, didascalie e qualsiasi scritta. "
                "Mantieni la formattazione originale dove possibile."
            )
        elif has_tech:
            parts.append(
                "Analizza questo screenshot/interfaccia tecnica. "
                "Identifica il tipo di applicazione, il linguaggio di programmazione se presente, "
                "eventuali errori visibili, e descrivi i dettagli tecnici rilevanti."
            )
        elif has_people:
            parts.append(
                "Descrivi le persone presenti nell'immagine: quante sono, "
                "il loro aspetto generale, espressioni, posizioni, "
                "abbigliamento e cosa stanno facendo."
            )
        elif has_food:
            parts.append(
                "Analizza il cibo/piatto nell'immagine. Identifica gli ingredienti visibili, "
                "il tipo di piatto, la presentazione e il contesto."
            )
        elif has_location:
            parts.append(
                "Descrivi il luogo mostrato nell'immagine: tipo di ambiente, "
                "elementi architettonici, segnali o indicazioni di localizzazione, "
                "atmosfera e dettagli rilevanti."
            )
        else:
            parts.append(
                "Descrivi tutto ciò che vedi nell'immagine nel modo più completo possibile."
            )

        if user_message and user_message.strip():
            parts.append(f"\nDomanda specifica dell'utente: \"{user_message}\"")
            parts.append("Rispondi nella stessa lingua della domanda.")
        else:
            parts.append("Rispondi in italiano.")

    else:
        if has_ocr:
            parts.append(
                "Carefully examine this image and transcribe ALL visible text, "
                "including titles, labels, captions, signs, and any writing. "
                "Preserve the original formatting where possible."
            )
        elif has_tech:
            parts.append(
                "Analyze this screenshot or technical interface in detail. "
                "Identify the application type, programming language if present, "
                "any visible errors, UI elements, and technical details."
            )
        elif has_people:
            parts.append(
                "Describe the people in this image: how many, their general appearance, "
                "expressions, positions, clothing, and what they are doing."
            )
        elif has_food:
            parts.append(
                "Analyze the food/dish in this image. Identify visible ingredients, "
                "the type of dish, presentation style, and context."
            )
        elif has_location:
            parts.append(
                "Describe the location shown in this image: environment type, "
                "architectural elements, signs or landmarks, atmosphere, and notable details."
            )
        else:
            parts.append(
                "Describe everything you see in this image in detail. "
                "Include objects, colors, text, people, animals, spatial layout, "
                "and any notable features."
            )

        parts.append(
            "\nBe precise and thorough. If there is any text visible, transcribe it exactly."
        )

    return "\n".join(parts)
