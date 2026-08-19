document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chat-messages');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const typingIndicator = document.getElementById('typing-indicator');
    const topKSlider = document.getElementById('top-k-slider');
    const topKVal = document.getElementById('top-k-val');
    const clearChatBtn = document.getElementById('clear-chat-btn');
    const systemStatusDot = document.getElementById('system-status-dot');
    const systemStatusText = document.getElementById('system-status-text');
    const modelNameEl = document.getElementById('model-name');
    const totalChunksEl = document.getElementById('total-chunks');
    const promptButtons = document.querySelectorAll('.prompt-btn');

    // ─── Status Check ─────────────────────────────
    async function checkServerStatus() {
        try {
            const res = await fetch('/api/status');
            const data = await res.json();
            if (data.status === 'ok' && data.ready) {
                systemStatusDot.classList.add('active');
                systemStatusText.innerText = 'system ready';
                if (modelNameEl) modelNameEl.innerText = (data.model || 'e5-base-v2').replace('intfloat/', '');
                if (totalChunksEl) totalChunksEl.innerText = `${data.total_chunks.toLocaleString()} chunks`;

                const retrievalEl = document.getElementById('retrieval-mode');
                if (retrievalEl) retrievalEl.innerText = data.bm25_enabled ? 'Hybrid BM25+Dense' : 'Dense only';

                const llmEl = document.getElementById('llm-model-name');
                if (llmEl) {
                    if (data.llm_connected) {
                        llmEl.innerText = `${data.llm_provider} // ${(data.llm_model || '').split('/').pop()}`;
                        llmEl.style.color = '#4ade80';
                    } else {
                        llmEl.innerText = 'fallback (no key)';
                        llmEl.style.color = '#facc15';
                    }
                }
            } else {
                systemStatusDot.classList.remove('active');
                systemStatusText.innerText = 'index not found';
            }
        } catch (err) {
            console.error('Status error:', err);
            systemStatusDot.classList.remove('active');
            systemStatusText.innerText = 'server unreachable';
        }
    }

    checkServerStatus();

    // ─── Slider ───────────────────────────────────
    topKSlider.addEventListener('input', (e) => {
        topKVal.innerText = e.target.value;
    });

    // ─── Quick Prompts ────────────────────────────
    promptButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const query = btn.getAttribute('data-query');
            if (query) {
                userInput.value = query;
                sendMessage(query);
            }
        });
    });

    // ─── Clear Chat ───────────────────────────────
    clearChatBtn.addEventListener('click', () => {
        chatMessages.innerHTML = `
            <div class="message bot-message">
                <div class="avatar"></div>
                <div class="message-content">
                    <div class="message-header">
                        <span class="author">AutoRAG</span>
                        <span class="time">${getTime()}</span>
                    </div>
                    <div class="message-body">
                        <p>Session cleared. Ready for new queries.</p>
                    </div>
                </div>
            </div>
        `;
    });

    // ─── Form Submit ──────────────────────────────
    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const q = userInput.value.trim();
        if (q) sendMessage(q);
    });

    // ─── Core Send ────────────────────────────────
    async function sendMessage(queryText) {
        appendUser(queryText);
        userInput.value = '';
        showTyping(true);
        scrollDown();

        const topK = parseInt(topKSlider.value, 10) || 1;

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: queryText, top_k: topK })
            });

            const data = await response.json();
            showTyping(false);

            if (data.status === 'success') {
                appendBotResponse(data);
            } else {
                appendError(data.message || 'Retrieval failed');
            }
        } catch (error) {
            console.error('Chat error:', error);
            showTyping(false);
            appendError('Connection to RAG server lost');
        }

        scrollDown();
    }

    // ─── Render: User Message ─────────────────────
    function appendUser(text) {
        const div = document.createElement('div');
        div.className = 'message user-message';
        div.innerHTML = `
            <div class="avatar"></div>
            <div class="message-content">
                <div class="message-header">
                    <span class="author">you</span>
                    <span class="time">${getTime()}</span>
                </div>
                <div class="message-body">
                    <p>${esc(text)}</p>
                </div>
            </div>
        `;
        chatMessages.appendChild(div);
    }

    // ─── Render: Bot Response (LM answer + sources) ──
    function appendBotResponse(data) {
        const div = document.createElement('div');
        div.className = 'message bot-message';

        const mainAnswer = data.answer || (data.results && data.results[0] ? data.results[0].answer : 'No relevant data found in knowledge base.');
        const results = data.results || [];

        let sourcesBlock = '';
        if (results.length > 0) {
            let sourceItems = '';
            results.forEach((item, idx) => {
                const rrf = item.score ? item.score.toFixed(4) : '--';
                const bm25 = item.bm25_score != null ? item.bm25_score.toFixed(2) : '--';
                const dense = item.dense_score != null ? item.dense_score.toFixed(4) : '--';

                sourceItems += `
                    <div class="result-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                            <span style="font-family:var(--font-mono); font-size:0.68rem; color:var(--accent); font-weight:600;">#${item.rank || idx + 1} rrf:${rrf}</span>
                            <span style="font-family:var(--font-mono); font-size:0.62rem; color:var(--text-dim);">bm25:${bm25} dense:${dense}</span>
                        </div>
                        <div class="result-answer">${esc(item.answer)}</div>
                    </div>
                `;
            });

            sourcesBlock = `
                <div style="margin-top:12px; border-top:1px solid var(--border-dim); padding-top:8px;">
                    <details>
                        <summary style="font-family:var(--font-mono); font-size:0.72rem; color:var(--text-dim); font-weight:500; cursor:pointer;">
                            retrieved sources (${results.length})
                        </summary>
                        <div style="margin-top:8px; display:flex; flex-direction:column; gap:6px;">
                            ${sourceItems}
                        </div>
                    </details>
                </div>
            `;
        }

        const formatted = esc(mainAnswer).replace(/\n/g, '<br>');

        div.innerHTML = `
            <div class="avatar"></div>
            <div class="message-content">
                <div class="message-header">
                    <span class="author">AutoRAG</span>
                    <span class="time">${getTime()}</span>
                </div>
                <div class="message-body">
                    <div class="result-answer" style="line-height:1.65;">${formatted}</div>
                    ${sourcesBlock}
                </div>
            </div>
        `;
        chatMessages.appendChild(div);
    }

    // ─── Render: Error ────────────────────────────
    function appendError(text) {
        const div = document.createElement('div');
        div.className = 'message bot-message';
        div.innerHTML = `
            <div class="avatar"></div>
            <div class="message-content">
                <div class="message-header">
                    <span class="author" style="color:var(--red);">error</span>
                    <span class="time">${getTime()}</span>
                </div>
                <div class="message-body" style="border-color:rgba(248,113,113,0.3);">
                    <p style="color:var(--red); font-family:var(--font-mono); font-size:0.82rem;">${esc(text)}</p>
                </div>
            </div>
        `;
        chatMessages.appendChild(div);
    }

    // ─── Utilities ────────────────────────────────
    function showTyping(show) {
        typingIndicator.style.display = show ? 'flex' : 'none';
    }

    function scrollDown() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function getTime() {
        return new Date().toLocaleTimeString('th-TH', { hour: '2-digit', minute: '2-digit' });
    }

    function esc(text) {
        const d = document.createElement('div');
        d.innerText = text;
        return d.innerHTML;
    }

    // ─── Copy handler ─────────────────────────────
    window.copyText = function(btn) {
        const card = btn.closest('.result-card');
        const answer = card.querySelector('.result-answer').innerText;
        navigator.clipboard.writeText(answer).then(() => {
            btn.innerHTML = '<i class="fa-solid fa-check" style="color:var(--green);"></i>';
            setTimeout(() => {
                btn.innerHTML = '<i class="fa-regular fa-copy"></i>';
            }, 1500);
        });
    };
});
