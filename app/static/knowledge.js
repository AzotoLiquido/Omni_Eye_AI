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

    document.getElementById('searchInput')?.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') searchKnowledge();
    });
});

async function loadKnowledgeBase() {
    try {
        const [kbResponse, entitiesResponse] = await Promise.all([
            fetch('/api/knowledge/summary'),
            fetch('/api/entities'),
        ]);

        if (!kbResponse.ok) throw new Error(`Knowledge API ${kbResponse.status}`);
        if (!entitiesResponse.ok) throw new Error(`Entities API ${entitiesResponse.status}`);

        const kbData = await kbResponse.json();
        const entitiesData = await entitiesResponse.json();

        if (kbData.success) {
            displayUserProfile(kbData.knowledge_base.user_profile);
            displaySummary(kbData.summary);
            displayTopics(kbData.knowledge_base.topics_discussed);
        }

        if (entitiesData.success) {
            displayEntities(entitiesData.entities);
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

    if (!profile.name && (!profile.interests || profile.interests.length === 0)) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">🤷</div>
                <p>Nessun profilo utente ancora.<br>Inizia a chattare per costruirlo!</p>
            </div>`;
        return;
    }

    let html = '';

    if (profile.name) {
        html += `<div class="kb-stat">
            <span class="kb-stat-label">Nome</span>
            <span class="kb-stat-value">${esc(profile.name)}</span>
        </div>`;
    }

    if (profile.language) {
        html += `<div class="kb-stat">
            <span class="kb-stat-label">Lingua</span>
            <span class="kb-stat-value">${esc(profile.language)}</span>
        </div>`;
    }

    if (profile.interests && profile.interests.length > 0) {
        html += `<div class="kb-stat">
            <span class="kb-stat-label">Interessi</span>
            <span class="kb-stat-value">${profile.interests.length}</span>
        </div>
        <div class="entity-list">
            ${profile.interests.map(i => `<span class="entity-tag">${esc(i)}</span>`).join('')}
        </div>`;
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

    if (results.relevant_topics && results.relevant_topics.length > 0) {
        html += '<h3>Topic Rilevanti:</h3>';
        results.relevant_topics.forEach(topic => {
            html += `<div class="topic-item">
                <span class="topic-name">${esc(topic.topic)}</span>
                <span class="topic-count">${Number(topic.mentions)}x</span>
            </div>`;
        });
    }

    if (Object.keys(results.user_profile).length > 0) {
        html += '<h3>Profilo Utente:</h3>';
        html += '<pre id="_profilePre"></pre>';
    }

    if (!results.relevant_topics?.length && Object.keys(results.user_profile).length === 0) {
        html += '<p>Nessun risultato trovato per questa query.</p>';
    }

    html += '</div>';
    container.innerHTML = html;

    // Inserisci profilo utente in modo sicuro (textContent evita XSS)
    const profilePre = document.getElementById('_profilePre');
    if (profilePre) {
        profilePre.textContent = JSON.stringify(results.user_profile, null, 2);
    }
}
