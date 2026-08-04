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

    // Fetch initial status from server
    async function checkServerStatus() {
        try {
            const res = await fetch('/api/status');
            const data = await res.json();
            if (data.status === 'ok' && data.ready) {
                systemStatusDot.classList.add('active');
                systemStatusText.innerText = 'ระบบ RAG พร้อมใช้งาน';
                modelNameEl.innerText = data.model || 'MiniLM-L12-v2';
                totalChunksEl.innerText = `${data.total_chunks} Chunks`;
            } else {
                systemStatusDot.classList.remove('active');
                systemStatusText.innerText = 'ระบบเวกเตอร์ฐานข้อมูลไม่พร้อม';
            }
        } catch (err) {
            console.error('Status check error:', err);
            systemStatusDot.classList.remove('active');
            systemStatusText.innerText = 'ไม่สามารถเชื่อมต่อกับเซิร์ฟเวอร์';
        }
    }

    checkServerStatus();

    // Top-K Slider listener
    topKSlider.addEventListener('input', (e) => {
        topKVal.innerText = e.target.value;
    });

    // Quick Prompt Buttons listener
    promptButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const query = btn.getAttribute('data-query');
            if (query) {
                userInput.value = query;
                sendMessage(query);
            }
        });
    });

    // Clear Chat history
    clearChatBtn.addEventListener('click', () => {
        chatMessages.innerHTML = `
            <div class="message bot-message">
                <div class="avatar">🌿</div>
                <div class="message-content">
                    <div class="message-header">
                        <span class="author">กะเพรา RAG Bot</span>
                        <span class="time">${getCurrentTime()}</span>
                    </div>
                    <div class="message-body">
                        <p>ล้างประวัติการสนทนาเรียบร้อยครับ อยากถามเกี่ยวกับผัดกะเพราเมนูไหนอีกสั่งมาได้เลย! 🌿🍳🌶️</p>
                    </div>
                </div>
            </div>
        `;
    });

    // Form Submit Event
    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const query = userInput.value.trim();
        if (query) {
            sendMessage(query);
        }
    });

    // Core function to send chat query
    async function sendMessage(queryText) {
        // Append user message bubble
        appendUserMessage(queryText);
        userInput.value = '';

        // Show typing indicator
        showTyping(true);
        scrollToBottom();

        const topK = parseInt(topKSlider.value, 10) || 1;

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: queryText,
                    top_k: topK
                })
            });

            const data = await response.json();
            showTyping(false);

            if (data.status === 'success') {
                appendBotResults(data.results);
            } else {
                appendErrorMessage(data.message || 'เกิดข้อผิดพลาดในการดึงข้อมูล');
            }
        } catch (error) {
            console.error('Chat error:', error);
            showTyping(false);
            appendErrorMessage('ไม่สามารถติดต่อเซิร์ฟเวอร์ RAG ได้ กรุณาตรวจสอบว่าเซิร์ฟเวอร์ทำงานอยู่');
        }

        scrollToBottom();
    }

    function appendUserMessage(text) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message user-message';
        msgDiv.innerHTML = `
            <div class="avatar">👤</div>
            <div class="message-content">
                <div class="message-header">
                    <span class="author">คุณ</span>
                    <span class="time">${getCurrentTime()}</span>
                </div>
                <div class="message-body">
                    <p>${escapeHtml(text)}</p>
                </div>
            </div>
        `;
        chatMessages.appendChild(msgDiv);
    }

    function appendBotResults(results) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message bot-message';

        let resultsHtml = '';
        if (!results || results.length === 0) {
            resultsHtml = '<p>ขออภัยครับ ไม่พบคำตอบที่เกี่ยวข้องในคลังความรู้ผัดกะเพรา 🌿</p>';
        } else {
            results.forEach((item) => {
                const scoreClass = getScoreBadgeClass(item.score);
                const scorePercent = (item.score * 100).toFixed(1);
                
                resultsHtml += `
                    <div class="result-card">
                        <div class="result-card-header">
                            <span class="score-badge ${scoreClass}">
                                <i class="fa-solid fa-chart-line"></i> ความคล้ายคลึง: ${item.score} (${scorePercent}%)
                            </span>
                            <button class="copy-btn" onclick="copyText(this)" title="คัดลอกคำตอบ">
                                <i class="fa-regular fa-copy"></i>
                            </button>
                        </div>
                        <div class="result-answer">${escapeHtml(item.answer)}</div>
                    </div>
                `;
            });
        }

        msgDiv.innerHTML = `
            <div class="avatar">🌿</div>
            <div class="message-content">
                <div class="message-header">
                    <span class="author">กะเพรา RAG Bot</span>
                    <span class="time">${getCurrentTime()}</span>
                </div>
                <div class="message-body">
                    ${resultsHtml}
                </div>
            </div>
        `;
        chatMessages.appendChild(msgDiv);
    }

    function appendErrorMessage(errorText) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message bot-message';
        msgDiv.innerHTML = `
            <div class="avatar">⚠️</div>
            <div class="message-content">
                <div class="message-header">
                    <span class="author">ระบบแจ้งเตือน</span>
                    <span class="time">${getCurrentTime()}</span>
                </div>
                <div class="message-body" style="border-color: rgba(239, 68, 68, 0.4);">
                    <p style="color: #ef4444;"><i class="fa-solid fa-circle-exclamation"></i> ${escapeHtml(errorText)}</p>
                </div>
            </div>
        `;
        chatMessages.appendChild(msgDiv);
    }

    function showTyping(show) {
        typingIndicator.style.display = show ? 'flex' : 'none';
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function getCurrentTime() {
        const now = new Date();
        return now.toLocaleTimeString('th-TH', { hour: '2-digit', minute: '2-digit' });
    }

    function getScoreBadgeClass(score) {
        if (score >= 0.7) return 'score-high';
        if (score >= 0.4) return 'score-medium';
        return 'score-low';
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.innerText = text;
        return div.innerHTML;
    }

    // Global copy handler
    window.copyText = function(btn) {
        const card = btn.closest('.result-card');
        const answer = card.querySelector('.result-answer').innerText;
        navigator.clipboard.writeText(answer).then(() => {
            btn.innerHTML = '<i class="fa-solid fa-check" style="color: #10b981;"></i>';
            setTimeout(() => {
                btn.innerHTML = '<i class="fa-regular fa-copy"></i>';
            }, 2000);
        });
    };
});
