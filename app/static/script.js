// ============================================================================
// OMNI EYE AI - FRONTEND APPLICATION
// ============================================================================
// Sistema di chat AI locale con gestione conversazioni, upload documenti,
// memoria avanzata e streaming real-time delle risposte.
// ============================================================================

const API_BASE = '/api';

// ============================================================================
// STATE MANAGEMENT
// Oggetto centrale per gestire stato applicazione (elimina variabili globali)
// ============================================================================
const App = {
    state: {
        currentConversationId: null,
        currentUploadedFile: null,
        pendingImages: [],       // Array di {base64, name, dataUrl} per immagini in attesa
        isProcessing: false,
        currentAbortController: null  // AbortController per fetch streaming (P2)
    },
    
    gridOrb: {
        el: null,
        timer: null,
        resizeTimer: null,
        initialized: false,
        col: 0,
        row: 0,
        dir: 0,
        cols: 0,
        rows: 0,
        offsetY: 0
    }
};

const GRID_SIZE = 42;  // Dimensione cella griglia (pixel)
const GRID_ORB_SIZE = 14;  // Dimensione orb animato (pixel)
const GRID_ORB_SPEED = 80;  // Velocità movimento orb (px/sec)
const GRID_ORB_TURN_CHANCE = 0.30;  // Probabilità cambio direzione agli incroci
const DIRS = [
    { dc: 1, dr: 0 },   // Destra
    { dc: 0, dr: 1 },   // Giù
    { dc: -1, dr: 0 },  // Sinistra
    { dc: 0, dr: -1 },  // Su
];

// ============================================================================
// DOM ELEMENTS CACHE
// Riferimenti agli elementi DOM usati frequentemente
// ============================================================================
const elements = {
    messageInput: document.getElementById('messageInput'),
    sendBtn: document.getElementById('sendBtn'),
    messagesContainer: document.getElementById('messagesContainer'),
    conversationsList: document.getElementById('conversationsList'),
    newChatBtn: document.getElementById('newChatBtn'),
    chatTitle: document.getElementById('chatTitle'),
    statusIndicator: document.getElementById('statusIndicator'),
    modelSelect: document.getElementById('modelSelect'),
    uploadBtn: document.getElementById('uploadBtn'),
    fileInput: document.getElementById('fileInput'),
    filePreview: document.getElementById('filePreview'),
    fileName: document.getElementById('fileName'),
    fileRemove: document.getElementById('fileRemove'),
    clearBtn: document.getElementById('clearBtn'),
    imageBtn: document.getElementById('imageBtn'),
    imageInput: document.getElementById('imageInput'),
    imagePreview: document.getElementById('imagePreview'),
    imageThumb: document.getElementById('imageThumb'),
    imageName: document.getElementById('imageName'),
    imageRemove: document.getElementById('imageRemove'),
};

// ============================================================================
// INITIALIZATION
// Setup iniziale dell'applicazione
// ============================================================================
async function init() {
    
    // Event listeners
    elements.sendBtn.addEventListener('click', sendMessage);
    elements.messageInput.addEventListener('input', handleInputChange);
    elements.messageInput.addEventListener('keydown', handleKeyDown);
    elements.newChatBtn.addEventListener('click', createNewConversation);
    elements.uploadBtn.addEventListener('click', () => elements.fileInput.click());
    elements.fileInput.addEventListener('change', handleFileUpload);
    elements.fileRemove.addEventListener('click', removeFile);
    elements.imageBtn.addEventListener('click', () => elements.imageInput.click());
    elements.imageInput.addEventListener('change', handleImageUpload);
    elements.imageRemove.addEventListener('click', removeImage);
    elements.clearBtn.addEventListener('click', clearCurrentChat);
    elements.modelSelect.addEventListener('change', handleModelChange);

    // Sidebar toggle
    const sidebarToggle = document.getElementById('sidebarToggle');
    const sidebar = document.getElementById('sidebar');
    if (sidebarToggle && sidebar) {
        sidebarToggle.addEventListener('click', () => {
            sidebar.classList.toggle('collapsed');
            sidebarToggle.textContent = sidebar.classList.contains('collapsed') ? '▶' : '◀';
            sidebarToggle.title = sidebar.classList.contains('collapsed') ? 'Apri sidebar' : 'Chiudi sidebar';
            // Ricalcola griglia orb dopo la transizione
            setTimeout(() => startGridOrb(), 350);
        });
    }

    // Quick prompts toggle
    const qpToggle = document.getElementById('quickPromptsToggle');
    const qpPanel = document.getElementById('quickPrompts');
    if (qpToggle && qpPanel) {
        qpToggle.addEventListener('click', () => {
            qpPanel.classList.toggle('expanded');
            qpToggle.classList.toggle('active');
            qpToggle.setAttribute('aria-expanded', qpPanel.classList.contains('expanded'));
        });
    }
    
    // Quick prompts
    document.querySelectorAll('.quick-prompt').forEach(btn => {
        btn.addEventListener('click', () => {
            elements.messageInput.value = btn.dataset.prompt;
            handleInputChange();
            sendMessage();
        });
    });
    
    // Verifica stato sistema
    await checkSystemStatus();
    
    // Carica conversazioni
    await loadConversations();
    
    // Carica modelli disponibili
    await loadAvailableModels();

    // Avvia orb sulla griglia del message container
    startGridOrb();
    
    // Focus sull'input
    elements.messageInput.focus();
}

/* ── Grid-Orb Screensaver ── */

function canOrbMove(col, row, dir) {
    const d = DIRS[dir];
    const nc = col + d.dc;
    const nr = row + d.dr;
    return nc >= 0 && nc < App.gridOrb.cols && nr >= 0 && nr < App.gridOrb.rows;
}

function orbStepMs() {
    return (GRID_SIZE / GRID_ORB_SPEED) * 1000;
}

function spawnTrailDot(px, py) {
    const chatArea = document.querySelector('.chat-area');
    if (!chatArea) return;
    const trail = document.createElement('div');
    trail.className = 'grid-orb-trail';
    
    // Random variations for organic look
    const size = 6 + Math.random() * 6; // 6-12px
    const offsetX = (Math.random() - 0.5) * 4; // -2 to +2px
    const offsetY = (Math.random() - 0.5) * 4;
    const duration = 0.6 + Math.random() * 0.4; // 0.6-1s
    const blur = 2 + Math.random() * 1; // 2-3px
    
    trail.style.cssText = `
        position: absolute;
        left: ${px + offsetX}px;
        top: ${py + App.gridOrb.offsetY + offsetY}px;
        width: ${size}px;
        height: ${size}px;
        margin-left: ${-size/2}px;
        margin-top: ${-size/2}px;
        border-radius: 50%;
        background: radial-gradient(circle, 
            rgba(92, 255, 138, 1) 0%, 
            rgba(92, 255, 138, 0.8) 20%, 
            rgba(92, 255, 138, 0.4) 50%, 
            rgba(92, 255, 138, 0.1) 80%, 
            transparent 100%);
        box-shadow: 
            0 0 ${size * 1.5}px rgba(92, 255, 138, 0.9), 
            0 0 ${size * 2.5}px rgba(92, 255, 138, 0.6),
            0 0 ${size * 4}px rgba(92, 255, 138, 0.3);
        filter: blur(${blur}px);
        pointer-events: none;
        z-index: 499;
        opacity: 1;
        animation: trailFade ${duration}s ease-out forwards;
    `;
    
    chatArea.appendChild(trail);
    
    setTimeout(() => {
        if (trail.parentNode) {
            trail.remove();
        }
    }, duration * 1000);
}

function moveOrbStep() {
    // At each intersection: maybe turn, or must turn if wall ahead
    const wantTurn = Math.random() < GRID_ORB_TURN_CHANCE;
    const mustTurn = !canOrbMove(App.gridOrb.col, App.gridOrb.row, App.gridOrb.dir);

    if (wantTurn || mustTurn) {
        // Prefer perpendicular directions (no U-turn unless forced)
        const candidates = [0, 1, 2, 3].filter(d =>
            d !== App.gridOrb.dir && d !== (App.gridOrb.dir + 2) % 4 && canOrbMove(App.gridOrb.col, App.gridOrb.row, d)
        );
        if (candidates.length > 0) {
            App.gridOrb.dir = candidates[Math.floor(Math.random() * candidates.length)];
        } else {
            // Allow U-turn as last resort
            const any = [0, 1, 2, 3].filter(d => canOrbMove(App.gridOrb.col, App.gridOrb.row, d));
            if (any.length === 0) { App.gridOrb.timer = setTimeout(moveOrbStep, 500); return; }
            App.gridOrb.dir = any[Math.floor(Math.random() * any.length)];
        }
    }

    // Current position
    const startPx = App.gridOrb.col * GRID_SIZE;
    const startPy = App.gridOrb.row * GRID_SIZE;

    // Advance one cell
    const d = DIRS[App.gridOrb.dir];
    
    App.gridOrb.col += d.dc;
    App.gridOrb.row += d.dr;

    // New position
    const endPx = App.gridOrb.col * GRID_SIZE;
    const endPy = App.gridOrb.row * GRID_SIZE;

    // Move orb to new position with CSS transition
    App.gridOrb.el.style.transform = `translate(${endPx}px, ${endPy + App.gridOrb.offsetY}px)`;

    // Create trail particles along the path
    const steps = 8;
    for (let i = 0; i < steps; i++) {
        setTimeout(() => {
            const progress = i / steps;
            const px = startPx + (endPx - startPx) * progress;
            const py = startPy + (endPy - startPy) * progress;
            spawnTrailDot(px, py);
        }, i * (orbStepMs() / steps));
    }

    // Schedule next
    App.gridOrb.timer = setTimeout(moveOrbStep, orbStepMs());
}

function startGridOrb() {
    // Attach orb to .chat-area (not messages-container) so it stays in visible viewport
    const chatArea = document.querySelector('.chat-area');
    if (!chatArea) {
        return;
    }

    // Crea griglia come elemento DOM invece di pseudo-element
    let gridBg = document.getElementById('grid-background');
    if (!gridBg) {
        gridBg = document.createElement('div');
        gridBg.id = 'grid-background';
        gridBg.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
                linear-gradient(rgba(128, 128, 128, 0.25) 1px, transparent 1px),
                linear-gradient(90deg, rgba(128, 128, 128, 0.25) 1px, transparent 1px);
            background-size: 42px 42px;
            background-position: 0px 0px;
            pointer-events: none;
            z-index: 0;
            opacity: 1 !important;
        `;
        chatArea.insertBefore(gridBg, chatArea.firstChild);
    }

    // Create / re-attach element
    if (!App.gridOrb.el || !App.gridOrb.el.parentNode) {
        App.gridOrb.el = document.createElement('div');
        App.gridOrb.el.className = 'grid-orb';
        App.gridOrb.el.style.cssText = `
            position: absolute;
            width: 14px;
            height: 14px;
            margin-top: -7px;
            margin-left: -7px;
            border-radius: 50%;
            background: radial-gradient(circle, 
                rgba(92, 255, 138, 1) 0%, 
                rgba(92, 255, 138, 0.9) 30%, 
                rgba(92, 255, 138, 0.6) 50%, 
                rgba(92, 255, 138, 0.3) 70%, 
                transparent 100%);
            box-shadow: 
                0 0 20px rgba(92, 255, 138, 1), 
                0 0 35px rgba(92, 255, 138, 0.8),
                0 0 50px rgba(92, 255, 138, 0.5);
            filter: blur(1.5px);
            pointer-events: none;
            z-index: 500;
            opacity: 1 !important;
            transition: transform ${orbStepMs() / 1000}s linear;
            animation: orbPulse 2s ease-in-out infinite;
        `;
        chatArea.appendChild(App.gridOrb.el);
    }

    // Measure visible grid area of chat-area, excluding header and input
    const chatHeader = chatArea.querySelector('.chat-header');
    const inputArea = chatArea.querySelector('.input-area');
    const headerH = chatHeader ? chatHeader.offsetHeight : 0;
    const inputH = inputArea ? inputArea.offsetHeight : 0;
    const cw = chatArea.clientWidth;
    const ch = chatArea.clientHeight - headerH - inputH;

    // How many grid cells fit inside the visible area
    App.gridOrb.cols = Math.max(2, Math.floor(cw / GRID_SIZE));
    App.gridOrb.rows = Math.max(2, Math.floor(ch / GRID_SIZE));

    // Store the Y pixel offset so the orb starts below the header
    App.gridOrb.offsetY = headerH;

    // Align CSS grid background with the orb's offset
    chatArea.style.setProperty('--grid-offset-y', headerH + 'px');
    gridBg.style.backgroundPosition = `0px ${headerH}px`;

    // Random start position & direction
    App.gridOrb.col = Math.floor(Math.random() * App.gridOrb.cols);
    App.gridOrb.row = Math.floor(Math.random() * App.gridOrb.rows);
    App.gridOrb.dir = Math.floor(Math.random() * 4);

    // Place instantly (no transition)
    if (App.gridOrb.timer) clearTimeout(App.gridOrb.timer);
    App.gridOrb.el.style.transition = 'none';
    App.gridOrb.el.style.transform = `translate(${App.gridOrb.col * GRID_SIZE}px, ${App.gridOrb.row * GRID_SIZE + App.gridOrb.offsetY}px)`;
    App.gridOrb.el.style.opacity = '1';

    // After a frame, enable smooth transition and start loop
    requestAnimationFrame(() => {
        const ms = orbStepMs();
        App.gridOrb.el.style.transition = `transform ${ms}ms linear, opacity 0.6s ease`;
        App.gridOrb.timer = setTimeout(moveOrbStep, ms);
    });

    // Resize handler (register once)
    if (!App.gridOrb.initialized) {
        window.addEventListener('resize', () => {
            if (App.gridOrb.resizeTimer) clearTimeout(App.gridOrb.resizeTimer);
            App.gridOrb.resizeTimer = setTimeout(startGridOrb, 250);
        });
        App.gridOrb.initialized = true;
    }
}

// ============================================================================
// SYSTEM STATUS
// Verifica disponibilità Ollama e modelli AI
// ============================================================================

// Verifica stato del sistema
async function checkSystemStatus() {
    try {
        const response = await fetch(`${API_BASE}/status`);
        const data = await response.json();
        
        if (data.ollama_available && data.model_available) {
            setStatus('online', `${data.model_name} pronto`);
        } else if (data.ollama_available) {
            setStatus('offline', `Modello non disponibile`);
        } else {
            setStatus('offline', 'Ollama non disponibile');
        }
        
        return data;
    } catch (error) {
        console.error('❌ Errore verifica stato:', error);
        setStatus('offline', 'Errore connessione');
        return null;
    }
}

// Imposta stato
function setStatus(status, text) {
    elements.statusIndicator.className = `status-indicator ${status}`;
    elements.statusIndicator.querySelector('.status-text').textContent = text;
}

// ============================================================================
// MODEL MANAGEMENT
// Gestione modelli AI disponibili
// ============================================================================

// Carica modelli disponibili
async function loadAvailableModels() {
    try {
        const response = await fetch(`${API_BASE}/models`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        
        elements.modelSelect.innerHTML = '';
        
        if (data.models.length === 0) {
            elements.modelSelect.innerHTML = '<option>Nessun modello</option>';
            return;
        }
        
        data.models.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            if (model === data.current) {
                option.selected = true;
            }
            elements.modelSelect.appendChild(option);
        });
        
    } catch (error) {
        console.error('❌ Errore caricamento modelli:', error);
    }
}

// Cambia modello
async function handleModelChange(e) {
    const newModel = e.target.value;
    
    try {
        const response = await fetch(`${API_BASE}/models/change`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: newModel })
        });
        
        const data = await response.json();
        
        if (data.success) {
            await checkSystemStatus();
            showNotification('✅ Modello cambiato con successo', 'success');
        } else {
            showNotification('❌ Impossibile cambiare modello', 'error');
        }
    } catch (error) {
        console.error('❌ Errore cambio modello:', error);
        showNotification('❌ Errore cambio modello', 'error');
    }
}

// ============================================================================
// INPUT HANDLING
// Gestione textarea e validazione input
// ============================================================================

// Gestione input
function handleInputChange() {
    const hasText = elements.messageInput.value.trim().length > 0;
    const hasImages = App.state.pendingImages.length > 0;
    elements.sendBtn.disabled = !(hasText || hasImages);
    elements.sendBtn.setAttribute('aria-disabled', String(elements.sendBtn.disabled));
    
    // Auto-resize textarea
    elements.messageInput.style.height = 'auto';
    elements.messageInput.style.height = elements.messageInput.scrollHeight + 'px';
}

function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

// ============================================================================
// MESSAGE SENDING - HELPERS
// Funzioni ausiliarie per gestire invio messaggi (scompone logica complessa)
// ============================================================================

/**
 * Prepara l'UI per l'invio di un messaggio
 */
function prepareUIForSending() {
    elements.messageInput.disabled = true;
    elements.sendBtn.disabled = true;
    
    // Nascondi welcome message
    const welcomeMsg = elements.messagesContainer.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.style.display = 'none';
    }

    // Nascondi quick prompts panel (P1-5: was quickPromptsWrapper)
    const qpw = document.getElementById('quickPrompts');
    if (qpw) qpw.style.display = 'none';
}

/**
 * Ripristina l'UI dopo invio messaggio
 */
function restoreUIAfterSending() {
    elements.messageInput.disabled = false;
    elements.sendBtn.disabled = false;
    elements.messageInput.focus();
}

/**
 * Gestisce lo streaming della risposta SSE
 * @param {Response} response - Response object dal fetch
 * @param {HTMLElement} contentDiv - Div dove inserire il contenuto
 * @returns {Promise<void>}
 */
async function handleStreamResponse(response, contentDiv) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let fullResponse = '';
    let currentEvent = null;
    let remainder = '';  // Buffer per linee spezzate tra chunk
    
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = remainder + decoder.decode(value, {stream: true});
        remainder = '';
        const lines = chunk.split('\n');
        
        // L'ultima "linea" potrebbe essere incompleta — la teniamo per il prossimo chunk
        if (!chunk.endsWith('\n')) {
            remainder = lines.pop();
        }
        
        for (const line of lines) {
            if (line.startsWith('event: ')) {
                currentEvent = line.slice(7).trim();
            } else if (line.startsWith('data: ')) {
                const data = line.slice(6);
                
                if (!currentEvent || currentEvent === 'message') {
                    fullResponse += data;
                    contentDiv.textContent = fullResponse;
                    scrollToBottom();
                } else if (currentEvent === 'end') {
                    App.state.currentConversationId = data;
                } else if (currentEvent === 'error') {
                    throw new Error(data);
                }
                
                currentEvent = null;
            }
        }
    }
    // Flush finale del decoder per caratteri multi-byte residui
    const final = decoder.decode();
    if (final) remainder += final;
}

// ============================================================================
// MESSAGE SENDING - MAIN
// Funzione principale per invio messaggi (scomposta in step più piccoli)
// ============================================================================
async function sendMessage() {
    const message = elements.messageInput.value.trim();
    const images = App.state.pendingImages.slice(); // copia
    
    // Step 1: Validazione - serve almeno testo o immagine
    if (!message && images.length === 0) return;
    
    // Step 2: Prepara UI
    prepareUIForSending();
    
    // Step 3: Aggiungi messaggio utente (con eventuale thumbnail)
    addMessage('user', message, images);
    
    // Step 4: Pulisci input e immagini pendenti
    elements.messageInput.value = '';
    elements.messageInput.style.height = 'auto';
    removeImage();
    
    // Step 5: Mostra typing indicator
    const typingId = addTypingIndicator();
    
    try {
        // Step 6: Prepara body richiesta
        const requestBody = {
            message: message || 'Descrivi questa immagine.',
            conversation_id: App.state.currentConversationId
        };
        
        // Aggiungi immagini base64 se presenti
        if (images.length > 0) {
            requestBody.images = images.map(img => img.base64);
        }
        
        // P1-7: Includi contesto documento se presente
        if (App.state.currentUploadedFile && App.state.currentUploadedFile.text) {
            requestBody.message = `[Documento caricato: "${App.state.currentUploadedFile.info?.name || 'file'}"]

Contenuto (anteprima):
${App.state.currentUploadedFile.text.slice(0, 3000)}

---
Domanda utente: ${requestBody.message}`;
            App.state.currentUploadedFile = null;
            removeFile();
        }
        
        // P2: Abort previous streaming request if any
        if (App.state.currentAbortController) {
            App.state.currentAbortController.abort();
        }
        App.state.currentAbortController = new AbortController();
        
        // Step 7: Invia richiesta con streaming
        const response = await fetch(`${API_BASE}/chat/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody),
            signal: App.state.currentAbortController.signal
        });
        
        if (!response.ok) {
            let errorMessage = `Errore HTTP ${response.status}`;
            try {
                const errorData = await response.json();
                errorMessage = errorData.error || errorData.message || errorMessage;
            } catch {
                errorMessage = await response.text() || errorMessage;
            }
            throw new Error(errorMessage);
        }
        
        // Step 7: Rimuovi typing indicator
        removeTypingIndicator(typingId);
        
        // Step 8: Crea messaggio assistant vuoto
        const assistantMessageDiv = createMessageElement('assistant', '');
        elements.messagesContainer.appendChild(assistantMessageDiv);
        const contentDiv = assistantMessageDiv.querySelector('.message-content');
        
        // Step 9: Gestisci stream di risposta
        await handleStreamResponse(response, contentDiv);
        
        // Step 10: Aggiorna lista conversazioni
        await loadConversations();
        
    } catch (error) {
        console.error('❌ Errore invio messaggio:', error);
        removeTypingIndicator(typingId);
        addMessage('assistant', `❌ Errore: ${error.message}`);
        showNotification('❌ Errore nell\'invio del messaggio', 'error');
    } finally {
        // Step 11: Ripristina UI
        App.state.currentAbortController = null;
        restoreUIAfterSending();
    }
}

// ============================================================================
// MESSAGE UI
// Rendering e gestione UI messaggi
// ============================================================================

// Aggiungi messaggio alla UI
function addMessage(role, content, images = []) {
    const messageDiv = createMessageElement(role, content, images);
    elements.messagesContainer.appendChild(messageDiv);
    scrollToBottom();
}

// Crea elemento messaggio
function createMessageElement(role, content, images = []) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? '👤' : '🤖';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // Mostra thumbnail immagini se presenti (solo messaggi utente)
    if (images && images.length > 0) {
        const imgContainer = document.createElement('div');
        imgContainer.className = 'message-images';
        for (const img of images) {
            const thumb = document.createElement('img');
            thumb.src = img.dataUrl;
            thumb.alt = img.name || 'Immagine allegata';
            thumb.className = 'message-image-thumb';
            thumb.addEventListener('click', () => {
                window.open(img.dataUrl, '_blank');
            });
            imgContainer.appendChild(thumb);
        }
        contentDiv.appendChild(imgContainer);
    }
    
    if (content) {
        const textNode = document.createElement('span');
        textNode.textContent = content;
        contentDiv.appendChild(textNode);
    }
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    
    return messageDiv;
}

// Typing indicator
function addTypingIndicator() {
    const id = 'typing-' + Date.now();
    const typingDiv = document.createElement('div');
    typingDiv.id = id;
    typingDiv.className = 'message assistant';
    typingDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    elements.messagesContainer.appendChild(typingDiv);
    scrollToBottom();
    return id;
}

function removeTypingIndicator(id) {
    const element = document.getElementById(id);
    if (element) {
        element.remove();
    }
}

// Scroll automatico
function scrollToBottom() {
    elements.messagesContainer.scrollTop = elements.messagesContainer.scrollHeight;
}

// ============================================================================
// CONVERSATION MANAGEMENT
// Caricamento e gestione conversazioni
// ============================================================================

// Carica conversazioni
async function loadConversations() {
    try {
        const response = await fetch(`${API_BASE}/conversations`, { cache: 'no-store' });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const conversations = await response.json();
        
        elements.conversationsList.innerHTML = '';
        
        if (conversations.length === 0) {
            elements.conversationsList.innerHTML = `
                <div style="padding: 1rem; text-align: center; color: #94a3b8;">
                    Nessuna conversazione
                </div>
            `;
            return;
        }
        
        conversations.forEach(conv => {
            const item = document.createElement('div');
            item.className = 'conversation-item';
            if (conv.id === App.state.currentConversationId) {
                item.classList.add('active');
            }
            
            const date = new Date(conv.updated_at);
            const dateStr = date.toLocaleDateString('it-IT', {
                day: '2-digit',
                month: '2-digit',
                hour: '2-digit',
                minute: '2-digit'
            });
            
            // Sanitizza i dati per prevenire XSS
            const titleDiv = document.createElement('div');
            titleDiv.className = 'title';
            titleDiv.textContent = conv.title;
            
            const metaDiv = document.createElement('div');
            metaDiv.className = 'meta';
            metaDiv.textContent = `${dateStr} · ${conv.message_count} msg`;
            
            const deleteBtn = document.createElement('button');
            deleteBtn.className = 'delete-btn';
            deleteBtn.textContent = '✕';
            deleteBtn.setAttribute('aria-label', 'Elimina conversazione');
            deleteBtn.onclick = (e) => deleteConversation(conv.id, e);
            
            item.appendChild(titleDiv);
            item.appendChild(metaDiv);
            item.appendChild(deleteBtn);
            
            item.addEventListener('click', () => loadConversation(conv.id));
            elements.conversationsList.appendChild(item);
        });
        
    } catch (error) {
        console.error('❌ Errore caricamento conversazioni:', error);
    }
}

// Carica conversazione specifica
async function loadConversation(convId) {
    try {
        const response = await fetch(`${API_BASE}/conversations/${convId}`, { cache: 'no-store' });
        if (!response.ok) {
            let errorMessage = `Errore HTTP ${response.status}`;
            try {
                const errorData = await response.json();
                errorMessage = errorData.error || errorData.message || errorMessage;
            } catch {
                errorMessage = await response.text() || errorMessage;
            }
            throw new Error(errorMessage);
        }
        const conversation = await response.json();
        
        App.state.currentConversationId = convId;
        elements.chatTitle.textContent = conversation.title || 'Conversazione';
        
        // Pulisci messaggi
        elements.messagesContainer.innerHTML = '';
        
        // Mostra messaggi
        const messages = Array.isArray(conversation.messages) ? conversation.messages : [];
        messages.forEach(msg => {
            addMessage(msg.role, msg.content);
        });
        if (messages.length === 0) {
            elements.messagesContainer.innerHTML = `
                <div class="welcome-message">
                    <h1>💬 Conversazione vuota</h1>
                    <p>Nessun messaggio salvato in questa chat</p>
                </div>
            `;
            const qpw = document.getElementById('quickPrompts');
            if (qpw) qpw.style.display = '';
        } else {
            const qpw = document.getElementById('quickPrompts');
            if (qpw) qpw.style.display = 'none';
        }

        startGridOrb();
        
        // Aggiorna UI
        await loadConversations();
        
    } catch (error) {
        console.error('❌ Errore caricamento conversazione:', error);
        showNotification('❌ Errore caricamento conversazione', 'error');
    }
}

// Elimina conversazione
async function deleteConversation(convId, event) {
    event.stopPropagation();
    
    if (!confirm('Sei sicuro di voler eliminare questa conversazione?')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/conversations/${convId}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            if (App.state.currentConversationId === convId) {
                createNewConversation();
            }
            await loadConversations();
            showNotification('✅ Conversazione eliminata', 'success');
        }
    } catch (error) {
        console.error('❌ Errore eliminazione:', error);
        showNotification('❌ Errore eliminazione conversazione', 'error');
    }
}

// Nuova conversazione
async function createNewConversation() {
    try {
        const response = await fetch(`${API_BASE}/conversations/new`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });
        
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        
        if (data.success) {
            App.state.currentConversationId = data.conversation_id;
            elements.messagesContainer.innerHTML = `
                <div class="welcome-message">
                    <h1>💬 Nuova Conversazione</h1>
                    <p>Inizia a chattare con il tuo assistente AI locale</p>
                </div>
            `;
            elements.chatTitle.textContent = 'Nuova Conversazione';
            elements.messageInput.focus();
            const qpw = document.getElementById('quickPrompts');
            if (qpw) qpw.style.display = '';
            startGridOrb();
            await loadConversations();
        }
    } catch (error) {
        console.error('❌ Errore creazione conversazione:', error);
        showNotification('❌ Errore creazione conversazione', 'error');
    }
}

// Pulisci chat corrente
function clearCurrentChat() {
    if (!App.state.currentConversationId) return;
    
    if (confirm('Vuoi iniziare una nuova conversazione?')) {
        createNewConversation();
    }
}

// ============================================================================
// FILE UPLOAD
// Gestione upload e preview documenti
// ============================================================================

// Configurazione upload
const FILE_UPLOAD_CONFIG = {
    maxSize: 10 * 1024 * 1024, // 10MB
    allowedTypes: [
        'application/pdf',
        'text/plain',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/msword',
        'text/markdown'
    ],
    allowedExtensions: ['.pdf', '.txt', '.docx', '.doc', '.md']
};

// Upload file
async function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    // Validazione dimensione
    if (file.size > FILE_UPLOAD_CONFIG.maxSize) {
        const sizeMB = (FILE_UPLOAD_CONFIG.maxSize / (1024 * 1024)).toFixed(0);
        showNotification(`❌ File troppo grande. Limite: ${sizeMB}MB`, 'error');
        elements.fileInput.value = '';
        return;
    }
    
    // Validazione tipo file
    const fileExt = '.' + file.name.split('.').pop().toLowerCase();
    const isValidType = FILE_UPLOAD_CONFIG.allowedTypes.includes(file.type) || 
                       FILE_UPLOAD_CONFIG.allowedExtensions.includes(fileExt);
    
    if (!isValidType) {
        showNotification('❌ Tipo di file non supportato. Usa: PDF, TXT, DOCX, MD', 'error');
        elements.fileInput.value = '';
        return;
    }
    
    // Mostra preview
    elements.fileName.textContent = `📄 ${file.name}`;
    elements.filePreview.style.display = 'block';
    
    // Leggi e carica file
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        showNotification('⏳ Caricamento file...', 'info');
        
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            App.state.currentUploadedFile = {
                text: data.text,
                info: data.file_info,
                filepath: data.filepath
            };
            
            showNotification('✅ File caricato con successo', 'success');
            
            // Pre-compila domanda
            elements.messageInput.value = `Ho caricato il file "${data.file_info.name}". Puoi analizzarlo?`;
            handleInputChange();
            
        } else {
            throw new Error(data.error);
        }
        
    } catch (error) {
        console.error('❌ Errore upload:', error);
        showNotification(`❌ Errore: ${error.message}`, 'error');
        removeFile();
    }
}

// Rimuovi file
function removeFile() {
    App.state.currentUploadedFile = null;
    elements.filePreview.style.display = 'none';
    elements.fileInput.value = '';
}

// ============================================================================
// IMAGE UPLOAD
// Gestione upload e preview immagini per analisi visiva (modelli vision)
// ============================================================================

const IMAGE_UPLOAD_CONFIG = {
    maxSize: 20 * 1024 * 1024,  // 20MB
    allowedTypes: ['image/jpeg', 'image/png', 'image/webp'],
    maxDimension: 1024,         // ridimensiona a max 1024px (lato lungo)
    outputQuality: 0.85,        // qualità JPEG/WebP dopo resize
};

/**
 * Gestisce la selezione di un'immagine dall'input file.
 * Converte in base64 e mostra anteprima.
 */
function handleImageUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    // Validazione dimensione
    if (file.size > IMAGE_UPLOAD_CONFIG.maxSize) {
        const sizeMB = (IMAGE_UPLOAD_CONFIG.maxSize / (1024 * 1024)).toFixed(0);
        showNotification(`❌ Immagine troppo grande. Limite: ${sizeMB}MB`, 'error');
        elements.imageInput.value = '';
        return;
    }

    // Validazione tipo
    if (!IMAGE_UPLOAD_CONFIG.allowedTypes.includes(file.type)) {
        showNotification('❌ Formato non supportato. Usa: JPG, PNG, WebP', 'error');
        elements.imageInput.value = '';
        return;
    }

    // P1-1: Ridimensiona via canvas prima di convertire in base64
    _resizeAndEncodeImage(file).then(({ base64, dataUrl }) => {
        App.state.pendingImages.push({
            base64: base64,
            name: file.name,
            dataUrl: dataUrl
        });

        // Mostra anteprima (ultima immagine aggiunta)
        elements.imageThumb.src = dataUrl;
        const names = App.state.pendingImages.map(i => i.name).join(', ');
        elements.imageName.textContent = names;
        elements.imagePreview.style.display = 'block';

        // Abilita invio anche senza testo
        handleInputChange();

        // Suggerisci prompt se input vuoto
        if (!elements.messageInput.value.trim()) {
            elements.messageInput.setAttribute(
                'placeholder', 
                'Cosa vuoi sapere su questa immagine? [ENTER per inviare]'
            );
        }

        const count = App.state.pendingImages.length;
        const label = count > 1 ? `${count} immagini pronte` : 'Immagine pronta';
        showNotification(`🖼️ ${label} per l'analisi`, 'success');
    }).catch(() => {
        showNotification('❌ Errore elaborazione immagine', 'error');
    });
}

/**
 * Ridimensiona un'immagine a max IMAGE_UPLOAD_CONFIG.maxDimension px
 * e la codifica in base64 JPEG/WebP per ridurre il payload.
 * @returns {Promise<{base64: string, dataUrl: string}>}
 */
function _resizeAndEncodeImage(file) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        const objectUrl = URL.createObjectURL(file);
        img.onerror = () => { URL.revokeObjectURL(objectUrl); reject(new Error('Image load failed')); };
        img.onload = function() {
            // P2: Revoke blob URL to avoid memory leak
            URL.revokeObjectURL(objectUrl);

            const max = IMAGE_UPLOAD_CONFIG.maxDimension;
            let w = img.naturalWidth;
            let h = img.naturalHeight;

            // Ridimensiona solo se supera il limite
            if (w > max || h > max) {
                if (w >= h) {
                    h = Math.round(h * (max / w));
                    w = max;
                } else {
                    w = Math.round(w * (max / h));
                    h = max;
                }
            }

            const canvas = document.createElement('canvas');
            canvas.width = w;
            canvas.height = h;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0, w, h);

            // Codifica come JPEG (più compatto per vision)
            const quality = IMAGE_UPLOAD_CONFIG.outputQuality;
            const dataUrl = canvas.toDataURL('image/jpeg', quality);
            const base64 = dataUrl.split(',')[1];
            resolve({ base64, dataUrl });
        };
        img.src = objectUrl;
    });
}
/**
 * Rimuove l'immagine pendente e ripristina lo stato.
 */
function removeImage() {
    App.state.pendingImages = [];
    elements.imagePreview.style.display = 'none';
    elements.imageThumb.src = '';
    elements.imageName.textContent = '';
    elements.imageInput.value = '';
    elements.messageInput.setAttribute('placeholder', 'enter command... [ENTER to execute]');
    handleInputChange();
}

// ============================================================================
// UTILITIES
// Funzioni di utilità generali
// ============================================================================

// Sanitizza HTML per prevenire XSS
function escapeHTML(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// Sistema di notifiche toast
function showNotification(message, type = 'info') {
    console.log(`[${type.toUpperCase()}] ${message}`);
    
    // Crea container se non esiste
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-container';
        document.body.appendChild(container);
    }
    
    // Crea toast
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    // Icona in base al tipo
    const icons = {
        success: '✅',
        error: '❌',
        warning: '⚠️',
        info: 'ℹ️'
    };
    
    const icon = document.createElement('span');
    icon.className = 'toast-icon';
    icon.textContent = icons[type] || icons.info;
    
    const text = document.createElement('span');
    text.className = 'toast-text';
    text.textContent = message;
    
    toast.appendChild(icon);
    toast.appendChild(text);
    container.appendChild(toast);
    
    // Animazione entrata
    setTimeout(() => toast.classList.add('show'), 10);
    
    // Rimozione automatica
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ============================================================================
// APP INITIALIZATION
// Avvio applicazione quando DOM è pronto
// ============================================================================

// Inizializza app quando DOM è pronto
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

// Export per uso globale
window.deleteConversation = deleteConversation;
