"""
Knowledge Packs – modulo per popolare la KnowledgeBase con pacchetti
tematici predefiniti.

Ogni pack è un dizionario con:
  - name:        identificatore univoco
  - description: breve descrizione
  - facts:       lista di dict con chiave "fact"

Uso diretto:
    from core.knowledge_packs import install_all_packs
    from core.advanced_memory import KnowledgeBase
    kb = KnowledgeBase("data")
    stats = install_all_packs(kb)
    kb.close()
"""

from __future__ import annotations

import csv
import json
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
#  DEFINIZIONE PACK
# ════════════════════════════════════════════════════════════════════════════

KNOWLEDGE_PACKS: Dict[str, dict] = {}


def _reg(name: str, description: str, facts: List[str]):
    """Helper per registrare un pack."""
    KNOWLEDGE_PACKS[name] = {
        "name": name,
        "description": description,
        "facts": [{"fact": f} for f in facts],
    }


# ── 1. Programming ─────────────────────────────────────────────────────────
_reg("programming", "Linguaggi di programmazione, paradigmi e best practice", [
    # Python
    "Python usa tipizzazione dinamica e duck typing: 'If it walks like a duck and quacks like a duck, it's a duck'.",
    "Le list comprehension in Python sono fino a 2x più veloci dei cicli for equivalenti grazie all'ottimizzazione del bytecode.",
    "Il GIL (Global Interpreter Lock) di Python limita l'esecuzione a un thread alla volta per il bytecode; per il parallelismo CPU si usa multiprocessing.",
    "I decoratori Python sono funzioni che wrappano altre funzioni: @decorator equivale a func = decorator(func).",
    "Python supporta multiple inheritance con MRO (Method Resolution Order) basato sull'algoritmo C3 linearization.",
    "asyncio in Python usa un event loop single-threaded per gestire migliaia di connessioni I/O-bound concorrenti.",
    "I context manager (with statement) garantiscono cleanup delle risorse anche in caso di eccezioni tramite __enter__/__exit__.",
    "I dataclass (Python 3.7+) generano automaticamente __init__, __repr__, __eq__ e possono essere frozen (immutabili).",
    "typing.Protocol (Python 3.8+) permette structural subtyping senza ereditarietà esplicita.",
    "match/case (Python 3.10+) è il pattern matching strutturale, supporta guard clauses e destructuring.",
    "Python f-string (3.6+) supporta espressioni, formattazione e debug con = suffix: f'{var=}'.",
    "Le named tuple e i dataclass frozen sono hashable e possono essere usati come chiavi di dizionario.",
    "Il modulo itertools fornisce combinazioni, permutazioni, prodotti cartesiani e iteratori infiniti.",
    "functools.lru_cache memorizza i risultati delle chiamate per velocizzare funzioni ricorsive o costose.",
    # JavaScript / TypeScript
    "JavaScript è single-threaded ma supporta concorrenza tramite l'event loop e le callback queue (microtask e macrotask).",
    "Le Promise in JavaScript rappresentano valori futuri; async/await è zucchero sintattico sulle Promise.",
    "Il prototype chain è il meccanismo di ereditarietà di JavaScript: ogni oggetto ha un __proto__ che punta al prototipo.",
    "TypeScript aggiunge tipizzazione statica a JavaScript con type inference, generics, union types e mapped types.",
    "Il garbage collector V8 usa generational GC: Young Generation (Scavenge) e Old Generation (Mark-Sweep-Compact).",
    "ES Modules (import/export) sono staticamente analizzabili e supportano tree-shaking per ridurre il bundle size.",
    "WeakMap e WeakSet in JavaScript non impediscono la garbage collection delle chiavi, utili per metadati temporanei.",
    "Optional chaining (?.) e nullish coalescing (??) riducono i null check verbosi in JavaScript/TypeScript.",
    # Rust
    "Rust garantisce memory safety senza garbage collector tramite ownership, borrowing e lifetimes.",
    "Il borrow checker di Rust verifica a compile-time che non esistano data race: o N riferimenti immutabili o 1 mutabile.",
    "I trait in Rust sono simili alle interfacce ma supportano default implementations e possono essere usati come bounds generici.",
    "enum in Rust supporta variant con dati (algebraic data types) e pattern matching esaustivo con match.",
    "Rust compila in binari nativi senza runtime overhead; le zero-cost abstractions non aggiungono costo a runtime.",
    "async/await in Rust usa un modello a futures lazy: non fanno nulla finché non vengono .await-ati.",
    "Il macro system di Rust opera a livello AST e può generare codice arbitrario a compile-time (proc macros).",
    # Go
    "Go usa goroutine (green threads) con stack dinamico iniziale di 2-8 KB, gestite dal runtime scheduler M:N.",
    "I channel in Go implementano CSP (Communicating Sequential Processes) per comunicazione sicura tra goroutine.",
    "Go non ha ereditarietà né generics tradizionali (fino a 1.18); usa composizione tramite embedding di struct.",
    "defer in Go esegue funzioni in ordine LIFO alla fine della funzione corrente, utile per cleanup risorse.",
    "Go compila in un singolo binario statico, semplificando il deployment senza dipendenze esterne.",
    # Java / Kotlin
    "La JVM usa JIT compilation: interpreta il bytecode e compila in codice nativo le sezioni hot path.",
    "Java Streams API (Java 8+) supporta operazioni funzionali lazy su collezioni con parallelStream() opzionale.",
    "Kotlin usa coroutine strutturate con scope (CoroutineScope) per gestire la concorrenza in modo sicuro.",
    "I sealed class/interface (Java 17+, Kotlin) limitano le sottoclassi a un set predefinito per pattern matching esaustivo.",
    "Kotlin null safety è gestita nel type system: String (non-null) vs String? (nullable) con operatori ?. e !!.",
    # C / C++
    "C è il linguaggio più vicino all'hardware senza assembly; puntatori e aritmetica dei puntatori danno controllo diretto sulla memoria.",
    "RAII (Resource Acquisition Is Initialization) in C++ lega il ciclo di vita delle risorse agli oggetti scope-based.",
    "I smart pointer C++ (unique_ptr, shared_ptr, weak_ptr) gestiscono la memoria automaticamente senza GC.",
    "Template metaprogramming in C++ è Turing-complete: si possono calcolare valori a compile-time.",
    "move semantics (C++11) evita copie inutili trasferendo la proprietà delle risorse tramite && (rvalue reference).",
    # Paradigmi generali
    "I principi SOLID sono: Single responsibility, Open-closed, Liskov substitution, Interface segregation, Dependency inversion.",
    "Il pattern Observer (pub/sub) disaccoppia produttori e consumatori di eventi ed è alla base di React, RxJS, Redis Pub/Sub.",
    "Dependency Injection inverte il controllo: le dipendenze sono fornite dall'esterno anziché create internamente.",
    "Il pattern Strategy permette di cambiare algoritmo a runtime incapsulandoli in classi intercambiabili.",
    "Immutabilità e funzioni pure riducono i bug: nessun side effect, output determinato solo dagli input.",
    "Big O descrive la complessità asintotica: O(1) costante, O(log n) logaritmico, O(n) lineare, O(n log n), O(n²) quadratico.",
])


# ── 2. Mathematics ─────────────────────────────────────────────────────────
_reg("mathematics", "Matematica pura e applicata", [
    "Il teorema fondamentale dell'algebra: ogni polinomio di grado n ha esattamente n radici complesse (contate con molteplicità).",
    "e^(iπ) + 1 = 0 (identità di Eulero) collega le 5 costanti fondamentali: e, i, π, 1, 0.",
    "La serie di Fibonacci F(n) = F(n-1) + F(n-2) converge al rapporto aureo φ = (1+√5)/2 ≈ 1.618.",
    "Il teorema di Bayes: P(A|B) = P(B|A)·P(A) / P(B). È la base dell'inferenza bayesiana e del machine learning.",
    "La trasformata di Fourier decompone un segnale in frequenze sinusoidali; la FFT ha complessità O(n log n).",
    "Il teorema di Gödel: in ogni sistema formale coerente e sufficientemente potente esistono enunciati veri ma indimostrabili.",
    "I numeri primi sono infiniti (dimostrato da Euclide ~300 a.C.); la loro distribuzione segue il Teorema dei Numeri Primi.",
    "La congettura di Riemann (1859, ancora irrisolta) riguarda gli zeri della funzione zeta e la distribuzione dei primi.",
    "Il calcolo matriciale è alla base della computer graphics, del machine learning e della meccanica quantistica.",
    "La derivata misura il tasso di cambiamento istantaneo; l'integrale è l'operazione inversa (Teorema Fondamentale del Calcolo).",
    "La probabilità condizionata P(A|B) = P(A∩B)/P(B) è alla base del ragionamento probabilistico e dell'AI.",
    "Il metodo di Newton-Raphson x_{n+1} = x_n - f(x_n)/f'(x_n) converge quadraticamente per trovare radici di equazioni.",
    "I numeri complessi z = a + bi estendono i reali e sono essenziali in analisi, elettrotecnica e meccanica quantistica.",
    "La teoria dei grafi studia relazioni tra nodi; è alla base di social network, routing e ottimizzazione combinatoria.",
    "L'algebra lineare (vettori, matrici, spazi vettoriali) è il linguaggio matematico del machine learning e della fisica.",
])


# ── 3. Physics ──────────────────────────────────────────────────────────────
_reg("physics", "Fisica classica, moderna e quantistica", [
    "Le tre leggi di Newton: (1) inerzia, (2) F = ma, (3) azione-reazione. Fondamento della meccanica classica.",
    "E = mc² (Einstein 1905): l'energia è equivalente alla massa per il quadrato della velocità della luce.",
    "La velocità della luce nel vuoto è c ≈ 299.792.458 m/s, il limite massimo per informazioni e materia.",
    "Il principio di indeterminazione di Heisenberg: Δx·Δp ≥ ℏ/2. Non si possono conoscere simultaneamente posizione e quantità di moto.",
    "L'entanglement quantistico lega due particelle: misurare una determina istantaneamente lo stato dell'altra, a qualsiasi distanza.",
    "Il modello standard descrive 17 particelle fondamentali: 6 quark, 6 leptoni, 4 bosoni di gauge e il bosone di Higgs.",
    "La termodinamica ha 4 leggi: (0) equilibrio termico, (1) conservazione energia, (2) entropia cresce, (3) zero assoluto irraggiungibile.",
    "Le onde gravitazionali, predette da Einstein nel 1916, sono state rilevate per la prima volta da LIGO nel 2015.",
    "La relatività generale descrive la gravità come curvatura dello spaziotempo causata da massa ed energia.",
    "I buchi neri hanno un orizzonte degli eventi oltre il quale nulla può sfuggire; Hawking predisse che emettono radiazione.",
    "La teoria delle stringhe propone che le particelle fondamentali siano vibrazioni di stringhe 1D in 10-11 dimensioni.",
    "L'effetto fotoelettrico (spiegato da Einstein 1905) dimostra la natura corpuscolare della luce: E = hν.",
    "La superconduttività è resistenza elettrica zero sotto una temperatura critica; viene usata nei magneti MRI e nel CERN.",
    "Il principio di sovrapposizione quantistica: un sistema esiste in tutti gli stati possibili simultaneamente fino alla misura.",
])


# ── 4. Chemistry ────────────────────────────────────────────────────────────
_reg("chemistry", "Chimica generale, organica e biochimica", [
    "La tavola periodica organizza 118 elementi per numero atomico; le proprietà chimiche seguono pattern periodici.",
    "Il legame covalente condivide elettroni tra atomi; il legame ionico trasferisce elettroni; il legame metallico delocalizza elettroni.",
    "Il pH misura l'acidità: pH < 7 acido, pH = 7 neutro, pH > 7 basico. È il logaritmo negativo della concentrazione H+.",
    "L'acqua (H₂O) è il solvente universale grazie alla sua polarità e ai legami a idrogeno.",
    "La legge di Avogadro: una mole contiene 6.022 × 10²³ particelle (numero di Avogadro).",
    "Le reazioni redox implicano trasferimento di elettroni: ossidazione = perdita, riduzione = acquisizione.",
    "I catalizzatori accelerano le reazioni abbassando l'energia di attivazione senza essere consumati.",
    "Il DNA è un polimero di nucleotidi con doppia elica; i legami A-T e G-C codificano l'informazione genetica.",
    "I polimeri (plastica, gomma, proteine) sono macromolecole formate da monomeri ripetuti.",
    "L'equilibrio chimico è dinamico: le reazioni diretta e inversa procedono alla stessa velocità (principio di Le Chatelier).",
    "Gli isotopi sono atomi dello stesso elemento con diverso numero di neutroni; alcuni sono radioattivi (C-14 per datazione).",
    "La chimica organica studia i composti del carbonio; la sua versatilità (4 legami) genera milioni di molecole diverse.",
])


# ── 5. Biology ──────────────────────────────────────────────────────────────
_reg("biology", "Biologia cellulare, genetica ed evoluzione", [
    "La cellula è l'unità fondamentale della vita: procarioti (senza nucleo) ed eucarioti (con nucleo).",
    "Il DNA contiene il codice genetico; l'RNA messaggero lo trascrive e i ribosomi lo traducono in proteine.",
    "La fotosintesi converte CO₂ + H₂O + luce → glucosio + O₂ nei cloroplasti delle cellule vegetali.",
    "La respirazione cellulare produce ATP: glucosio + O₂ → CO₂ + H₂O + 36-38 ATP nei mitocondri.",
    "CRISPR-Cas9 è un sistema di editing genetico che taglia il DNA in punti specifici per modificare geni.",
    "L'evoluzione per selezione naturale (Darwin 1859): gli organismi con tratti vantaggiosi si riproducono di più.",
    "Il genoma umano contiene circa 3.2 miliardi di coppie di basi e circa 20.000-25.000 geni codificanti proteine.",
    "I virus non sono cellule: sono acidi nucleici (DNA o RNA) avvolti in un capside proteico; si replicano usando cellule ospiti.",
    "L'omeostasi è la capacità degli organismi di mantenere condizioni interne stabili (temperatura, pH, glucosio).",
    "Il sistema immunitario umano ha due linee: innato (rapido, aspecifico) e adattativo (lento, specifico, con memoria).",
    "I mitocondri hanno il proprio DNA, supportando la teoria endosimbiotica: erano batteri inglobati da cellule ancestrali.",
    "L'epigenetica studia modifiche ereditabili dell'espressione genica senza cambiamenti nella sequenza del DNA.",
])


# ── 6. Astronomy ────────────────────────────────────────────────────────────
_reg("astronomy", "Astronomia, astrofisica e cosmologia", [
    "L'universo ha circa 13.8 miliardi di anni; il Big Bang non fu un'esplosione ma un'espansione dello spaziotempo stesso.",
    "La Via Lattea contiene 100-400 miliardi di stelle; il Sistema Solare orbita a 26.000 anni luce dal centro galattico.",
    "La luce del Sole impiega circa 8 minuti e 20 secondi per raggiungere la Terra (distanza media: 150 milioni di km).",
    "Giove è il pianeta più grande del Sistema Solare; la sua Grande Macchia Rossa è una tempesta attiva da almeno 350 anni.",
    "Le stelle di neutroni sono resti di supernove con densità estrema: un cucchiaino pesa circa 6 miliardi di tonnellate.",
    "L'universo è composto per il 68% da energia oscura, 27% materia oscura e solo 5% materia ordinaria (barionica).",
    "Un anno luce è la distanza percorsa dalla luce in un anno: circa 9.461 miliardi di km.",
    "La stella più vicina al Sole è Proxima Centauri, a 4.24 anni luce; ha almeno un pianeta nella zona abitabile.",
    "Il telescopio James Webb (JWST, 2021) osserva nell'infrarosso per studiare le prime galassie formatesi dopo il Big Bang.",
    "Le onde gravitazionali sono prodotte da eventi cosmici estremi: fusione di buchi neri e stelle di neutroni.",
    "Marte ha il vulcano più alto del Sistema Solare (Olympus Mons, 21.9 km) e il canyon più lungo (Valles Marineris, 4000 km).",
    "I buchi neri supermassicci al centro delle galassie possono avere masse di miliardi di masse solari.",
])


# ── 7. History ──────────────────────────────────────────────────────────────
_reg("history", "Storia mondiale e momenti chiave", [
    "La rivoluzione agricola (circa 10.000 a.C.) permise la sedentarizzazione e la nascita delle prime civiltà.",
    "La scrittura fu inventata dai Sumeri in Mesopotamia intorno al 3500 a.C. (cuneiforme su tavolette d'argilla).",
    "La democrazia ateniese (508 a.C.) fu il primo sistema di governo democratico diretto nella storia documentata.",
    "L'Impero Romano (27 a.C. – 476 d.C.) al suo apice governava 70 milioni di persone su 5 milioni di km².",
    "La caduta di Costantinopoli (1453) segnò la fine dell'Impero Bizantino e stimolò le esplorazioni oceaniche europee.",
    "La stampa a caratteri mobili di Gutenberg (1440) rivoluzionò la diffusione della conoscenza in Europa.",
    "La Rivoluzione Francese (1789) abolì la monarchia assoluta e introdusse i concetti di libertà, uguaglianza, fraternità.",
    "La Rivoluzione Industriale (1760-1840) trasformò economia e società con macchine a vapore e produzione in fabbrica.",
    "La Seconda Guerra Mondiale (1939-1945) fu il conflitto più devastante della storia: 70-85 milioni di vittime.",
    "La caduta del Muro di Berlino (9 novembre 1989) simboleggiò la fine della Guerra Fredda e la riunificazione tedesca.",
    "L'invenzione di Internet (ARPANET 1969, WWW 1991) ha rivoluzionato comunicazione, commercio ed educazione globali.",
    "La Dichiarazione Universale dei Diritti Umani (ONU, 1948) stabilì per la prima volta diritti fondamentali universali.",
])


# ── 8. Philosophy ───────────────────────────────────────────────────────────
_reg("philosophy", "Filosofia occidentale e orientale", [
    "Socrate (470-399 a.C.) non scrisse nulla; il suo metodo dialettico (maieutica) usa domande per far emergere la verità.",
    "Platone distinse il mondo delle Idee (forme perfette) dal mondo sensibile (copie imperfette) nell'allegoria della caverna.",
    "Aristotele fondò la logica formale (sillogismo), classificò le scienze e influenzò il pensiero occidentale per 2000 anni.",
    "Il cogito di Cartesio 'Penso, dunque sono' (1637) è il fondamento del razionalismo e della filosofia moderna.",
    "Kant nella Critica della Ragion Pura (1781) distinse conoscenza a priori (innata) e a posteriori (esperienziale).",
    "Nietzsche proclamò la 'morte di Dio' come crisi dei valori tradizionali e propose l'Übermensch come ideale umano.",
    "L'esistenzialismo (Sartre, Camus) afferma che l'esistenza precede l'essenza: siamo condannati a essere liberi.",
    "Il pragmatismo (James, Dewey) valuta la verità delle idee in base alle loro conseguenze pratiche.",
    "Confucio (551-479 a.C.) insegnò l'etica delle relazioni sociali, il rispetto gerarchico e la coltivazione morale.",
    "Il buddismo insegna le Quattro Nobili Verità: sofferenza, origine della sofferenza, cessazione, via verso la cessazione.",
    "Wittgenstein nel Tractatus: 'Di ciò di cui non si può parlare, si deve tacere' — i limiti del linguaggio sono i limiti del mondo.",
    "L'utilitarismo (Bentham, Mill) misura la moralità delle azioni dalla quantità di felicità/benessere che producono.",
])


# ── 9. Economics ────────────────────────────────────────────────────────────
_reg("economics", "Economia, finanza e teorie economiche", [
    "La legge della domanda e dell'offerta: il prezzo di equilibrio si trova dove la quantità domandata uguaglia quella offerta.",
    "Il PIL (Prodotto Interno Lordo) misura il valore totale di beni e servizi prodotti in un paese in un anno.",
    "L'inflazione è l'aumento generalizzato dei prezzi; le banche centrali la controllano con tassi di interesse e politica monetaria.",
    "Adam Smith (1776): la 'mano invisibile' del libero mercato guida l'interesse individuale verso il benessere collettivo.",
    "Keynes (1936) propose l'intervento statale nella spesa pubblica per contrastare recessioni e disoccupazione ciclica.",
    "Il costo opportunità è il valore della migliore alternativa sacrificata quando si sceglie; è un concetto chiave in economia.",
    "La teoria dei giochi (Nash) studia le decisioni strategiche: l'equilibrio di Nash è quando nessun giocatore può migliorare cambiando strategia.",
    "Il debito pubblico è il totale dei prestiti di uno Stato; il rapporto debito/PIL indica la sostenibilità fiscale.",
    "La blockchain e le criptovalute (Bitcoin 2009) propongono sistemi finanziari decentralizzati senza intermediari.",
    "L'effetto moltiplicatore: un aumento della spesa pubblica genera un aumento del PIL superiore all'investimento iniziale.",
    "Il paradosso della parsimonia: se tutti risparmiano di più contemporaneamente, la domanda aggregata cala e l'economia si contrae.",
    "La curva di Phillips suggerisce una relazione inversa tra inflazione e disoccupazione, almeno nel breve periodo.",
])


# ── 10. Psychology ──────────────────────────────────────────────────────────
_reg("psychology", "Psicologia cognitiva, sociale e clinica", [
    "La piramide di Maslow ordina i bisogni: fisiologici → sicurezza → appartenenza → stima → autorealizzazione.",
    "Il condizionamento classico (Pavlov): uno stimolo neutro associato ripetutamente a uno incondizionato produce una risposta condizionata.",
    "Il condizionamento operante (Skinner): i comportamenti rinforzati positivamente si ripetono; quelli puniti diminuiscono.",
    "Il bias di conferma: tendiamo a cercare e interpretare informazioni che confermano le nostre credenze preesistenti.",
    "L'effetto Dunning-Kruger: chi ha poca competenza tende a sovrastimare le proprie abilità; gli esperti tendono a sottostimarle.",
    "La dissonanza cognitiva (Festinger): la tensione psicologica quando le azioni contraddicono le credenze porta a cambiare una delle due.",
    "La memoria di lavoro ha capacità limitata: circa 4±1 chunk (aggiornamento del 7±2 di Miller) per brevi periodi.",
    "L'intelligenza emotiva (Goleman) include autoconsapevolezza, autoregolazione, motivazione, empatia e abilità sociali.",
    "L'effetto alone: un'impressione positiva/negativa in un'area influenza il giudizio su altre aree non correlate.",
    "La terapia cognitivo-comportamentale (CBT) modifica pensieri disfunzionali per cambiare emozioni e comportamenti.",
    "L'esperimento di Milgram (1963) dimostrò che le persone obbediscono all'autorità anche quando devono infliggere sofferenza.",
    "Il flow (Csíkszentmihályi) è uno stato di concentrazione ottimale quando sfida e abilità sono in equilibrio.",
])


# ── 11. Medicine ────────────────────────────────────────────────────────────
_reg("medicine", "Medicina, anatomia e salute", [
    "Il cuore umano batte circa 100.000 volte al giorno, pompando circa 7.500 litri di sangue.",
    "Gli antibiotici combattono i batteri ma NON i virus; l'uso eccessivo causa resistenza antimicrobica (AMR).",
    "I vaccini stimolano il sistema immunitario a riconoscere e combattere patogeni specifici senza causare la malattia.",
    "Il cervello umano ha circa 86 miliardi di neuroni connessi da circa 100 trilioni di sinapsi.",
    "Il DNA umano in ogni cellula, se srotolato, misurerebbe circa 2 metri; tutto il DNA del corpo coprirebbe 600 volte Sole-Plutone.",
    "L'effetto placebo è un miglioramento reale dei sintomi causato dalla convinzione di ricevere un trattamento efficace.",
    "Il microbioma intestinale contiene 30-40 trilioni di batteri che influenzano digestione, immunità e persino l'umore.",
    "L'mRNA (usato nei vaccini Pfizer/Moderna COVID-19) istruisce le cellule a produrre una proteina per attivare la risposta immunitaria.",
    "Il sonno è essenziale: il cervello elimina proteine tossiche (beta-amiloide) durante il sonno profondo tramite il sistema glinfatico.",
    "Le malattie cardiovascolari sono la prima causa di morte globale; prevenzione: esercizio, dieta sana, no fumo.",
    "L'editing genetico con CRISPR potrebbe in futuro correggere malattie genetiche come l'anemia falciforme e la distrofia muscolare.",
    "Lo stress cronico aumenta il cortisolo, causando infiammazione, indebolimento immunitario e rischio cardiovascolare.",
])


# ── 12. Art ─────────────────────────────────────────────────────────────────
_reg("art", "Arte, storia dell'arte e movimenti artistici", [
    "Il Rinascimento (XIV-XVI sec.) riportò l'arte alla prospettiva, anatomia e naturalismo; Leonardo, Michelangelo, Raffaello.",
    "La prospettiva lineare fu formalizzata da Brunelleschi (1415) e permise rappresentazioni tridimensionali su superfici piane.",
    "L'Impressionismo (1870s) catturò la luce e il momento: Monet, Renoir, Degas dipingevano en plein air.",
    "Il Cubismo (Picasso, Braque, 1907+) scompose le forme in geometrie multiple viste simultaneamente.",
    "L'espressionismo astratto (Pollock, Rothko, 1940s) fu il primo movimento americano a influenzare il mondo dell'arte.",
    "La Gioconda di Leonardo da Vinci (1503-1519) è il dipinto più famoso al mondo, conservato al Louvre di Parigi.",
    "L'arte bizantina (V-XV sec.) usava mosaici dorati e figure frontali stilizzate per esprimere trascendenza spirituale.",
    "Il Barocco (XVII sec.) enfatizzò dramma, contrasto e grandiosità: Caravaggio, Bernini, Vermeer.",
    "Andy Warhol e la Pop Art (1960s) elevarono oggetti di consumo (lattine Campbell, Marilyn) ad arte alta.",
    "L'arte digitale e i NFT (2020s) hanno creato nuovi mercati per opere digitali certificate su blockchain.",
    "Il rapporto aureo (φ ≈ 1.618) è usato in composizione artistica e architettura per proporzioni esteticamente armoniche.",
    "Street art e graffiti (Banksy, Keith Haring) sono forme d'arte urbana che commentano temi sociali e politici.",
])


# ── 13. Music ───────────────────────────────────────────────────────────────
_reg("music", "Musica, teoria musicale e storia della musica", [
    "La scala cromatica ha 12 semitoni; le scale maggiori e minori selezionano 7 note con pattern specifici di toni e semitoni.",
    "Bach (1685-1750) perfezionò il contrappunto e il temperamento equabile; le sue fughe sono capolavori di struttura matematica.",
    "Beethoven (1770-1827) rivoluzionò la musica classica; compose la Nona Sinfonia quasi completamente sordo.",
    "Il blues (Delta del Mississippi, fine 1800) è la radice di jazz, rock, R&B, soul; usa la scala blues con la 'blue note'.",
    "Il jazz (New Orleans, 1900s) si basa su improvvisazione, swing e armonie complesse; Miles Davis, Coltrane.",
    "Il rock and roll (1950s) fuse blues, country e R&B; Elvis, Chuck Berry; portò alla rivoluzione culturale degli anni '60.",
    "La musica elettronica usa sintetizzatori e campionatori; generi: techno (Detroit), house (Chicago), ambient (Eno).",
    "Il formato MP3 (1993) comprime l'audio rimuovendo frequenze che l'orecchio umano non percepisce (trasformata di Fourier).",
    "La frequenza del La centrale (A4) è standardizzata a 440 Hz; ogni ottava raddoppia la frequenza.",
    "Il tempo in musica si misura in BPM (battiti per minuto): 60-70 adagio, 120-130 allegro, 140+ presto.",
    "Pro Tools, Ableton Live e Logic Pro sono le DAW (Digital Audio Workstation) più usate nella produzione musicale.",
    "Lo streaming (Spotify 2008, Apple Music 2015) ha trasformato l'industria musicale: oltre 100 milioni di brani disponibili.",
])


# ── 14. Cooking ─────────────────────────────────────────────────────────────
_reg("cooking", "Cucina, tecniche culinarie e scienza del cibo", [
    "La reazione di Maillard (120-180°C) crea sapore e colore nella carne arrostita, pane tostato e caffè: è diversa dalla caramellizzazione.",
    "Le 5 salse madri della cucina classica (Escoffier): besciamella, velouté, spagnola, pomodoro, olandese.",
    "La pasta va cotta in acqua abbondante e salata (10g sale/litro); al dente = amido ancora parzialmente cristallino al centro.",
    "Il glutine si forma quando acqua e proteine del grano (glutenina e gliadina) si legano; impastare sviluppa la rete glutinica.",
    "La fermentazione produce pane (lieviti → CO₂), birra (lieviti → alcol), yogurt (batteri → acido lattico).",
    "Umami è il quinto gusto fondamentale (Ikeda, 1908): glutammato presente in parmigiano, funghi, salsa di soia, pomodori maturi.",
    "La cottura sous vide (in sacchetto sottovuoto, bagnomaria a temperatura precisa) garantisce cottura uniforme e succhi preservati.",
    "L'emulsione è la dispersione di grassi in acqua (maionese, vinaigrette); la lecitina del tuorlo fa da emulsionante naturale.",
    "Il sale esalta i sapori non perché li 'aggiunge' ma perché sopprime l'amaro e amplifica dolce e umami.",
    "La cucina molecolare (Adrià, Blumenthal) applica principi scientifici: sferificazione, gelificazione, azoto liquido.",
    "La temperatura interna sicura per il pollo è 74°C; per la carne di manzo al sangue 52°C, medium 63°C, ben cotta 77°C.",
    "L'acido (limone, aceto) bilancia i piatti grassi; il grasso (olio, burro) attenua il piccante e la durezza dei sapori.",
])


# ── 15. Languages ───────────────────────────────────────────────────────────
_reg("languages", "Linguistica, lingue del mondo e filologia", [
    "Esistono circa 7.000 lingue al mondo; il 40% è in pericolo di estinzione con meno di 1.000 parlanti nativi.",
    "Il mandarino è la lingua con più parlanti nativi (~920 milioni); l'inglese è la più diffusa come seconda lingua (~1.5 miliardi).",
    "Le famiglie linguistiche principali: indoeuropea (3 miliardi), sino-tibetana (1.5 miliardi), afroasiatica, austronesiana.",
    "L'italiano deriva dal latino volgare; Dante, con la Divina Commedia (1320), stabilì il toscano come base dell'italiano letterario.",
    "La grammatica universale (Chomsky) ipotizza che tutti gli esseri umani nascano con una capacità innata per il linguaggio.",
    "L'ordine delle parole varia: SVO (inglese, italiano), SOV (giapponese, hindi, turco), VSO (arabo, irlandese).",
    "Il giapponese usa tre sistemi di scrittura: hiragana (sillabe native), katakana (prestiti), kanji (caratteri cinesi).",
    "L'esperanto (Zamenhof, 1887) è la lingua artificiale più diffusa al mondo con circa 2 milioni di parlanti.",
    "La lingua dei segni non è universale: ASL (americana), BSL (britannica), LIS (italiana) sono lingue distinte.",
    "Il sanscrito (circa 1500 a.C.) è una delle lingue indoeuropee più antiche documentate, fondamentale per la linguistica comparata.",
    "Il bilinguismo migliora le funzioni esecutive del cervello: maggiore flessibilità cognitiva e ritardo del declino cognitivo.",
    "L'alfabeto latino, usato da più lingue al mondo, deriva dall'alfabeto etrusco che a sua volta derivava dal greco.",
])


# ── 16. Geography ───────────────────────────────────────────────────────────
_reg("geography", "Geografia fisica, politica e ambientale", [
    "La Terra ha una superficie di 510 milioni di km²: 71% acqua (361 milioni km²) e 29% terra emersa (149 milioni km²).",
    "L'Everest (8.849 m) è la montagna più alta sul livello del mare; il Mauna Kea (Hawaii) è il più alto dalla base oceanica (10.203 m).",
    "La Fossa delle Marianne nel Pacifico raggiunge 10.994 m di profondità: il punto più profondo degli oceani terrestri.",
    "Il Nilo (6.650 km) e l'Amazzoni (6.400 km) si contendono il titolo di fiume più lungo; l'Amazzoni ha la portata maggiore.",
    "La tettonica a placche spiega terremoti, vulcanismo e formazione delle montagne tramite il movimento delle placche litosferiche.",
    "Il permafrost artico contiene 1.500 miliardi di tonnellate di carbonio; il suo scioglimento accelererebbe il riscaldamento globale.",
    "La Russia è il paese più grande (17.1 milioni km²); il Vaticano il più piccolo (0.44 km²).",
    "La corrente del Golfo trasporta calore dall'equatore all'Europa nord-occidentale, rendendola più calda di regioni alla stessa latitudine.",
    "Il deserto del Sahara (9.2 milioni km²) è grande quasi quanto gli Stati Uniti e sta espandendosi per la desertificazione.",
    "Le barriere coralline coprono lo 0.1% del fondale oceanico ma ospitano il 25% delle specie marine; sono minacciate dal riscaldamento.",
    "L'effetto serra naturale mantiene la Terra a ~15°C anziché -18°C; CO₂ e metano antropici lo intensificano.",
    "Il 55% della popolazione mondiale vive in aree urbane (2023); si prevede il 68% entro il 2050 (ONU).",
])


# ── 17. Computer Science & AI ──────────────────────────────────────────────
_reg("cs_ai", "Informatica teorica, algoritmi e intelligenza artificiale", [
    "L'algoritmo di Dijkstra trova il cammino più breve in un grafo pesato con pesi non negativi in O((V+E)log V).",
    "Le reti neurali sono composte da layer di neuroni artificiali; ogni neurone calcola una somma pesata + attivazione non lineare.",
    "Il deep learning usa reti con molti layer nascosti; la backpropagation calcola i gradienti per ottimizzare i pesi.",
    "I Transformer (Vaswani 2017) usano self-attention per modellare dipendenze a lungo raggio: base di GPT, BERT, LLaMA.",
    "GPT (Generative Pre-trained Transformer) genera testo token-by-token predicendo il prossimo token più probabile.",
    "La complessità P vs NP è il problema aperto più importante dell'informatica: P = problemi risolvibili in tempo polinomiale.",
    "Il machine learning si divide in: supervised (etichette), unsupervised (pattern), reinforcement (ricompense).",
    "Le CNN (Convolutional Neural Networks) eccellono nel riconoscimento immagini grazie a filtri convoluzionali gerarchici.",
    "Il problema dell'halting (Turing 1936): non esiste un algoritmo che determini se un programma arbitrario terminerà.",
    "La legge di Moore (1965): il numero di transistor nei chip raddoppia circa ogni 2 anni; tendenza in rallentamento dal 2015.",
    "RAG (Retrieval-Augmented Generation) migliora gli LLM fornendo documenti rilevanti come contesto aggiuntivo.",
    "I database relazionali (SQL) usano tabelle con relazioni; i NoSQL (MongoDB, Redis) sacrificano la struttura per la scalabilità.",
    "Git è un sistema di version control distribuito (Linus Torvalds 2005); ogni clone è un backup completo del repository.",
    "Docker containerizza applicazioni con le loro dipendenze; Kubernetes orchestra container su cluster di macchine.",
    "L'algoritmo A* combina Dijkstra con un'euristica per trovare il cammino ottimo più velocemente nei grafi.",
])


# ── 18. Law ─────────────────────────────────────────────────────────────────
_reg("law", "Diritto, sistemi giuridici e legislazione", [
    "I due grandi sistemi giuridici sono: civil law (codificato, Europa continentale) e common law (basato su precedenti, paesi anglosassoni).",
    "La Costituzione italiana (1948) ha 139 articoli + disposizioni transitorie; è rigida e richiede procedura aggravata per le modifiche.",
    "Il GDPR (2018) è il regolamento europeo sulla protezione dei dati personali: consenso, diritto all'oblio, data portability.",
    "Il diritto penale punisce reati (delitti e contravvenzioni); il diritto civile regola rapporti tra privati (contratti, proprietà).",
    "La presunzione di innocenza: l'accusato è innocente fino a prova contraria oltre ogni ragionevole dubbio.",
    "Il copyright protegge opere creative automaticamente alla creazione; i brevetti proteggono invenzioni per 20 anni.",
    "Le licenze open source (MIT, GPL, Apache) definiscono come il software può essere usato, modificato e redistribuito.",
    "Il diritto internazionale si basa su trattati, consuetudini e principi generali; la Corte dell'Aia giudica le controversie tra Stati.",
    "L'habeas corpus (1679) garantisce che nessuno possa essere detenuto senza essere portato davanti a un giudice.",
    "Il principio di legalità: nulla poena sine lege — nessuna pena senza una legge preesistente che definisca il reato.",
    "La Corte Europea dei Diritti dell'Uomo (CEDU, Strasburgo) tutela i diritti fondamentali dei cittadini dei 46 Stati membri.",
    "Il diritto del lavoro tutela i lavoratori: contratti collettivi, salario minimo, limiti orario, tutele contro il licenziamento.",
])


# ── 19. Linux & DevOps ─────────────────────────────────────────────────────
_reg("linux", "Linux, Unix, system administration e DevOps", [
    "Linux è un kernel monolitico creato da Linus Torvalds nel 1991; le distribuzioni (Ubuntu, Fedora, Arch) aggiungono userspace.",
    "Il filesystem Linux è gerarchico con root (/); /etc per config, /var per dati variabili, /home per utenti, /tmp per temporanei.",
    "I permessi Unix sono rwx (read, write, execute) per owner, group, others; chmod 755 = rwxr-xr-x.",
    "systemd è il sistema init moderno di Linux: gestisce servizi (unit files), journal logging e target di avvio.",
    "I container (Docker, Podman) usano cgroups e namespaces del kernel Linux per isolare processi senza overhead di VM.",
    "CI/CD (Continuous Integration/Delivery) automatizza build, test e deploy: GitHub Actions, GitLab CI, Jenkins.",
    "Nginx e Apache sono i web server più diffusi; Nginx eccelle come reverse proxy e load balancer ad alto throughput.",
    "SSH (Secure Shell) fornisce accesso remoto crittografato; le chiavi pubbliche/private evitano l'uso di password.",
    "Il kernel Linux supporta centinaia di filesystem: ext4 (default), XFS (grandi file), Btrfs (snapshot), ZFS (integrità dati).",
    "iptables/nftables gestiscono il firewall a livello kernel; regole per INPUT, OUTPUT, FORWARD chain.",
    "Ansible, Terraform e Puppet sono strumenti IaC (Infrastructure as Code) per gestire server in modo dichiarativo e riproducibile.",
    "I log di sistema in Linux si consultano con journalctl (systemd) o nei file /var/log/syslog, /var/log/auth.log.",
])


# ── 20. Networking ──────────────────────────────────────────────────────────
_reg("networking", "Reti informatiche, protocolli e sicurezza", [
    "Il modello OSI ha 7 livelli: Physical, Data Link, Network, Transport, Session, Presentation, Application.",
    "TCP garantisce consegna ordinata e affidabile (handshake, ACK, retransmission); UDP è più veloce ma senza garanzie.",
    "HTTP/2 usa multiplexing (più richieste su una connessione), header compression e server push; HTTP/3 usa QUIC su UDP.",
    "DNS traduce nomi di dominio in indirizzi IP; usa una gerarchia: root servers → TLD → authoritative → resolver.",
    "TLS/SSL crittografa le comunicazioni web (HTTPS); usa certificati X.509 per autenticazione e chiavi simmetriche per i dati.",
    "IPv4 ha 4.3 miliardi di indirizzi (32 bit); IPv6 ne ha 340 undecilioni (128 bit): 3.4 × 10³⁸.",
    "Una VPN crea un tunnel crittografato su rete pubblica; protocolli: WireGuard, OpenVPN, IPsec.",
    "Il NAT (Network Address Translation) permette a più dispositivi di condividere un IP pubblico, ritardando l'esaurimento IPv4.",
    "I firewall filtrano il traffico per IP, porta e protocollo; i WAF (Web Application Firewall) proteggono da attacchi applicativi.",
    "BGP (Border Gateway Protocol) è il protocollo che interconnette gli ISP e gestisce il routing globale di Internet.",
    "Le WebSocket (RFC 6455) permettono comunicazione bidirezionale persistente tra client e server su una singola connessione TCP.",
    "Zero Trust Security: 'never trust, always verify' — ogni richiesta deve essere autenticata indipendentemente dalla posizione in rete.",
])


# ════════════════════════════════════════════════════════════════════════════
#  FUNZIONI DI INSTALLAZIONE
# ════════════════════════════════════════════════════════════════════════════

def get_available_packs() -> List[dict]:
    """Restituisce la lista dei pack disponibili con nome, descrizione e conteggio."""
    return [
        {
            "name": p["name"],
            "description": p["description"],
            "facts_count": len(p["facts"]),
        }
        for p in KNOWLEDGE_PACKS.values()
    ]


def install_pack(kb, pack_name: str) -> dict:
    """Installa un singolo pack nella KnowledgeBase.

    Args:
        kb: istanza di KnowledgeBase (core.advanced_memory)
        pack_name: nome del pack da installare

    Returns:
        dict con 'installed' (int) e 'skipped' (int)
    """
    pack = KNOWLEDGE_PACKS.get(pack_name)
    if not pack:
        raise ValueError(f"Pack '{pack_name}' non trovato. "
                         f"Disponibili: {list(KNOWLEDGE_PACKS.keys())}")

    installed = 0
    skipped = 0
    source = f"pack:{pack_name}"

    for entry in pack["facts"]:
        try:
            kb.add_fact(entry["fact"], source=source)
            installed += 1
        except Exception as e:
            logger.warning("Errore inserimento fatto: %s", e)
            skipped += 1

    logger.info("Pack '%s': %d installati, %d saltati", pack_name, installed, skipped)
    return {"installed": installed, "skipped": skipped}


def install_all_packs(kb) -> dict:
    """Installa tutti i pack disponibili.

    Returns:
        dict con statistiche per pack e totali.
    """
    results = {}
    total_installed = 0
    total_skipped = 0

    for name in KNOWLEDGE_PACKS:
        r = install_pack(kb, name)
        results[name] = r
        total_installed += r["installed"]
        total_skipped += r["skipped"]

    logger.info("Tutti i pack installati: %d fatti totali, %d saltati",
                total_installed, total_skipped)
    return {
        "packs": results,
        "total_installed": total_installed,
        "total_skipped": total_skipped,
    }


# ════════════════════════════════════════════════════════════════════════════
#  IMPORTAZIONE DA FILE ESTERNI
# ════════════════════════════════════════════════════════════════════════════

def import_from_json(kb, filepath: str, source: Optional[str] = None) -> int:
    """Importa fatti da un file JSON (lista di stringhe o lista di dict con 'fact')."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    src = source or f"file:{os.path.basename(filepath)}"
    count = 0

    if isinstance(data, list):
        for item in data:
            text = item if isinstance(item, str) else item.get("fact", str(item))
            kb.add_fact(text, source=src)
            count += 1
    elif isinstance(data, dict) and "facts" in data:
        for item in data["facts"]:
            text = item if isinstance(item, str) else item.get("fact", str(item))
            kb.add_fact(text, source=src)
            count += 1

    logger.info("Importati %d fatti da %s", count, filepath)
    return count


def import_from_csv(kb, filepath: str, column: str = "fact",
                    source: Optional[str] = None) -> int:
    """Importa fatti da un file CSV con header."""
    src = source or f"file:{os.path.basename(filepath)}"
    count = 0

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get(column, "").strip()
            if text:
                kb.add_fact(text, source=src)
                count += 1

    logger.info("Importati %d fatti da %s", count, filepath)
    return count


def import_from_txt(kb, filepath: str, source: Optional[str] = None) -> int:
    """Importa fatti da un file di testo (una riga = un fatto)."""
    src = source or f"file:{os.path.basename(filepath)}"
    count = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                kb.add_fact(line, source=src)
                count += 1

    logger.info("Importati %d fatti da %s", count, filepath)
    return count


def import_file(kb, filepath: str, source: Optional[str] = None) -> int:
    """Auto-detect del formato e importazione."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".json":
        return import_from_json(kb, filepath, source)
    elif ext == ".csv":
        return import_from_csv(kb, filepath, source)
    elif ext in (".txt", ".md"):
        return import_from_txt(kb, filepath, source)
    else:
        raise ValueError(f"Formato non supportato: {ext} (supportati: .json, .csv, .txt, .md)")
