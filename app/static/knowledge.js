/**
 * Knowledge Base - Frontend Logic
 * Gestisce visualizzazione dati, ricerca e rendering entità.
 */

// Shared XSS-safe escape helper (single definition, no duplication)
function esc(str) {
    const d = document.createElement('div');
    d.textContent = String(str);
    return d.innerHTML;
}

// Event listeners
document.getElementById('searchBtn').addEventListener('click', searchKnowledge);

document.addEventListener('DOMContentLoaded', async () => {
    await loadKnowledgeBase();
    await loadKnowledgePacks();

    document.getElementById('searchInput')?.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') searchKnowledge();
    });

    document.getElementById('installAllPacksBtn')?.addEventListener('click', installAllPacks);
});

async function loadKnowledgeBase() {
    try {
        const [kbResponse, entitiesResponse, statsResponse] = await Promise.all([
            fetch('/api/knowledge/summary'),
            fetch('/api/entities'),
            fetch('/api/knowledge/stats'),
        ]);

        if (!kbResponse.ok) throw new Error(`Knowledge API ${kbResponse.status}`);
        if (!entitiesResponse.ok) throw new Error(`Entities API ${entitiesResponse.status}`);

        const kbData = await kbResponse.json();
        const entitiesData = await entitiesResponse.json();
        const statsData = statsResponse.ok ? await statsResponse.json() : null;

        if (kbData.success) {
            displayUserProfile(kbData.knowledge_base.user_profile);
            displaySummary(kbData.summary);
            displayTopics(kbData.knowledge_base.topics_discussed);
        }

        if (entitiesData.success) {
            displayEntities(entitiesData.entities);
        }

        if (statsData && statsData.success) {
            displayFactsStats(statsData);
        }

        document.getElementById('loading').style.display = 'none';
        document.getElementById('content').style.display = 'block';
    } catch (error) {
        console.error('Errore caricamento knowledge base:', error);
        document.getElementById('loading').innerHTML =
            '<p style="color: var(--danger);">❌ Errore nel caricamento dei dati</p>';
    }
}

function displayUserProfile(profile) {
    const container = document.getElementById('userProfile');

    const hasData = profile.name || profile.age || profile.birthday || profile.gender ||
                    profile.job || profile.family_status ||
                    (profile.interests && profile.interests.length > 0) ||
                    (profile.passions && profile.passions.length > 0) ||
                    (profile.personality && profile.personality.length > 0) ||
                    (profile.health && profile.health.length > 0) ||
                    (profile.physical && profile.physical.length > 0) ||
                    (profile.goals && profile.goals.length > 0);

    if (!hasData) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">🤷</div>
                <p>Nessun profilo utente ancora.<br>Inizia a chattare per costruirlo!</p>
            </div>`;
        return;
    }

    let html = '';

    // --- Campi singoli ---
    const singleFields = [
        { key: 'name', label: 'Nome', icon: '👤' },
        { key: 'age', label: 'Età', icon: '🎂', suffix: ' anni' },
        { key: 'birthday', label: 'Data di nascita', icon: '📅' },
        { key: 'gender', label: 'Sesso', icon: '⚧' },
        { key: 'job', label: 'Lavoro', icon: '💼' },
        { key: 'family_status', label: 'Stato famiglia', icon: '👨‍👩‍👧' },
        { key: 'children', label: 'Figli', icon: '👶' },
        { key: 'language', label: 'Lingua', icon: '🌐' },
    ];

    for (const f of singleFields) {
        const val = profile[f.key];
        if (val) {
            html += `<div class="kb-stat">
                <span class="kb-stat-label">${f.icon} ${f.label}</span>
                <span class="kb-stat-value">${esc(val)}${f.suffix || ''}</span>
            </div>`;
        }
    }

    // --- Campi lista (tag) ---
    const listFields = [
        { key: 'passions', label: 'Passioni', icon: '🔥' },
        { key: 'interests', label: 'Interessi', icon: '💡' },
        { key: 'personality', label: 'Personalità', icon: '🧩' },
        { key: 'physical', label: 'Caratteristiche fisiche', icon: '📏' },
        { key: 'health', label: 'Salute', icon: '🏥' },
        { key: 'family_details', label: 'Familiari', icon: '👪' },
        { key: 'goals', label: 'Obiettivi', icon: '🎯' },
    ];

    for (const f of listFields) {
        const arr = profile[f.key];
        if (arr && arr.length > 0) {
            html += `<div class="kb-stat">
                <span class="kb-stat-label">${f.icon} ${f.label}</span>
                <span class="kb-stat-value">${arr.length}</span>
            </div>
            <div class="entity-list">
                ${arr.map(i => `<span class="entity-tag">${esc(i)}</span>`).join('')}
            </div>`;
        }
    }

    container.innerHTML = html;
}

function displayTopics(topicsDiscussed) {
    const container = document.getElementById('topics');

    if (!topicsDiscussed || Object.keys(topicsDiscussed).length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">📊</div>
                <p>Nessun topic discusso ancora.<br>Parla di vari argomenti!</p>
            </div>`;
        return;
    }

    const sorted = Object.entries(topicsDiscussed).sort((a, b) => b[1] - a[1]);

    container.innerHTML = sorted.map(([name, count]) => `
        <div class="topic-item">
            <span class="topic-name">${esc(name)}</span>
            <span class="topic-count">${Number(count)}x</span>
        </div>`).join('');
}

function displayEntities(entities) {
    const container = document.getElementById('entities');

    if (entities.people_count === 0 && entities.preferences_count === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">🏷️</div>
                <p>Nessuna entità tracciata.<br>Menziona nomi o preferenze!</p>
            </div>`;
        return;
    }

    let html = `
        <div class="kb-stat">
            <span class="kb-stat-label">Persone menzionate</span>
            <span class="kb-stat-value">${Number(entities.people_count) || 0}</span>
        </div>
        <div class="kb-stat">
            <span class="kb-stat-label">Preferenze salvate</span>
            <span class="kb-stat-value">${Number(entities.preferences_count) || 0}</span>
        </div>
        <div class="kb-stat">
            <span class="kb-stat-label">Date/Eventi</span>
            <span class="kb-stat-value">${Number(entities.dates_count) || 0}</span>
        </div>`;

    if (entities.people && entities.people.length > 0) {
        html += `<div class="entity-list">
            ${entities.people.slice(0, 10).map(p => `<span class="entity-tag">👤 ${esc(p)}</span>`).join('')}
        </div>`;
    }

    container.innerHTML = html;
}

function displaySummary(summary) {
    document.getElementById('summary').textContent = summary;
}

function displayFactsStats(stats) {
    const container = document.getElementById('factsStats');
    if (!container) return;

    const total = stats.total_facts || 0;
    const sources = stats.sources || {};
    const sortedSources = Object.entries(sources).sort((a, b) => b[1] - a[1]);

    // Raggruppa per categoria (pack:programming -> programming)
    const categories = {};
    for (const [src, count] of sortedSources) {
        const cat = src.startsWith('pack:') ? src.slice(5) : src;
        categories[cat] = (categories[cat] || 0) + count;
    }
    const sortedCats = Object.entries(categories).sort((a, b) => b[1] - a[1]);

    let html = `
        <div class="kb-stat">
            <span class="kb-stat-label">Fatti totali nel database</span>
            <span class="kb-stat-value">${total}</span>
        </div>
        <div class="kb-stat">
            <span class="kb-stat-label">Categorie</span>
            <span class="kb-stat-value">${sortedCats.length}</span>
        </div>`;

    if (sortedCats.length > 0) {
        html += '<div class="facts-categories" style="margin-top: 0.8rem;">';
        for (const [cat, count] of sortedCats.slice(0, 25)) {
            html += `
                <div class="topic-item">
                    <span class="topic-name">${esc(cat)}</span>
                    <span class="topic-count">${count}</span>
                </div>`;
        }
        html += '</div>';
    }

    container.innerHTML = html;
}

async function searchKnowledge() {
    const query = document.getElementById('searchInput').value.trim();
    const resultsContainer = document.getElementById('searchResults');

    if (!query) {
        resultsContainer.innerHTML = '<p style="color: var(--warning);">⚠️ Inserisci una query di ricerca</p>';
        return;
    }

    resultsContainer.innerHTML = '<p>🔄 Ricerca in corso...</p>';

    try {
        const response = await fetch('/api/knowledge/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const data = await response.json();

        if (data.success) {
            displaySearchResults(data.results);
        } else {
            resultsContainer.innerHTML = '<p style="color: var(--danger);">❌ Errore nella ricerca</p>';
        }
    } catch (error) {
        console.error('Errore ricerca:', error);
        resultsContainer.innerHTML = '<p style="color: var(--danger);">❌ Errore di connessione</p>';
    }
}

function displaySearchResults(results) {
    const container = document.getElementById('searchResults');

    let html = '<div style="margin-top: 1rem;">';

    // FTS5 facts (related_facts)
    const facts = results.related_facts || [];
    if (facts.length > 0) {
        html += `<h3>🧠 Fatti trovati (${facts.length}):</h3>`;
        facts.forEach((fact, i) => {
            const source = fact.source ? `<span class="topic-count">${esc(fact.source)}</span>` : '';
            html += `<div class="topic-item" style="align-items: flex-start;">
                <span class="topic-name" style="white-space: normal;">${esc(fact.content)}</span>
                ${source}
            </div>`;
        });
    }

    // Relevant topics
    if (results.relevant_topics && results.relevant_topics.length > 0) {
        html += '<h3>📊 Topic Rilevanti:</h3>';
        results.relevant_topics.forEach(topic => {
            html += `<div class="topic-item">
                <span class="topic-name">${esc(topic.topic)}</span>
                <span class="topic-count">${Number(topic.mentions)}x</span>
            </div>`;
        });
    }

    // User profile match
    if (results.user_profile && Object.keys(results.user_profile).length > 0) {
        html += '<h3>👤 Profilo Utente:</h3>';
        html += '<pre id="_profilePre"></pre>';
    }

    if (!facts.length && !results.relevant_topics?.length && !Object.keys(results.user_profile || {}).length) {
        html += '<p>Nessun risultato trovato per questa query.</p>';
    }

    html += '</div>';
    container.innerHTML = html;

    const profilePre = document.getElementById('_profilePre');
    if (profilePre) {
        profilePre.textContent = JSON.stringify(results.user_profile, null, 2);
    }
}

// ============================================================================
// KNOWLEDGE PACKS
// ============================================================================

async function loadKnowledgePacks() {
    const container = document.getElementById('packsContainer');
    if (!container) return;

    try {
        const response = await fetch('/api/knowledge/packs');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();

        if (!data.success || !data.packs || data.packs.length === 0) {
            container.innerHTML = '<p style="color: var(--text-secondary);">Nessun pack disponibile.</p>';
            return;
        }

        let html = `<p style="margin-bottom: 0.5rem; color: var(--text-secondary);">
            ${data.total_packs} pack disponibili · ${data.total_facts_available} fatti totali · ${data.facts_in_db} nel DB
        </p>`;
        html += '<div class="packs-list">';

        for (const pack of data.packs) {
            html += `
                <div class="topic-item" style="flex-wrap: wrap; gap: 0.5rem;">
                    <div style="flex: 1; min-width: 150px;">
                        <span class="topic-name">${esc(pack.name)}</span>
                        <span style="color: var(--text-secondary); font-size: 0.85rem; margin-left: 0.5rem;">${pack.facts_count} fatti</span>
                    </div>
                    <button class="btn btn-secondary btn-sm" onclick="installPack('${esc(pack.name)}')" style="padding: 0.3rem 0.8rem; font-size: 0.85rem;">
                        📥 Installa
                    </button>
                </div>`;
        }

        html += '</div>';
        container.innerHTML = html;

    } catch (error) {
        console.error('Errore caricamento packs:', error);
        container.innerHTML = '<p style="color: var(--danger);">❌ Errore caricamento packs</p>';
    }
}

async function installPack(packName) {
    try {
        const response = await fetch('/api/knowledge/packs/install', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pack: packName })
        });

        const data = await response.json();
        if (data.success) {
            alert(`✅ Pack "${packName}" installato: ${data.added || 0} nuovi, ${data.skipped || 0} già presenti`);
            await loadKnowledgePacks();
            await loadKnowledgeBase();  // refresh stats
        } else {
            alert(`❌ Errore: ${data.error}`);
        }
    } catch (error) {
        console.error('Errore installazione pack:', error);
        alert('❌ Errore di connessione');
    }
}

async function installAllPacks() {
    if (!confirm('Installare tutti i knowledge packs?')) return;

    const btn = document.getElementById('installAllPacksBtn');
    if (btn) {
        btn.disabled = true;
        btn.textContent = '⏳ Installazione...';
    }

    try {
        const response = await fetch('/api/knowledge/packs/install', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ all: true })
        });

        const data = await response.json();
        if (data.success) {
            alert(`✅ Tutti i pack installati: ${data.added || 0} nuovi, ${data.skipped || 0} già presenti`);
            await loadKnowledgePacks();
            await loadKnowledgeBase();
        } else {
            alert(`❌ Errore: ${data.error}`);
        }
    } catch (error) {
        console.error('Errore installazione packs:', error);
        alert('❌ Errore di connessione');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.textContent = '📥 Installa Tutti';
        }
    }
}
