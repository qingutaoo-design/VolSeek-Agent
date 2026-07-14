// VolSeek Agent 前端应用
class SuperBizAgentApp {
    constructor() {
        this.apiBaseUrl = '';
        this.currentMode = 'stream'; // 默认流式模式，用户可实时看到 Agent 的思考和输出过程
        this.sessionId = this.generateSessionId();
        this.isStreaming = false;
        this.currentChatHistory = [];
        this.chatHistories = this.loadChatHistories();
        this.isCurrentChatFromHistory = false;
        
        this.initializeElements();
        this.bindEvents();
        this.updateUI();
        this.initMarkdown();
        this.checkAndSetCentered();
        this.renderChatHistory();
        this.checkHealth();
    }

    // ---- 健康检查 ----
    async checkHealth() {
        try {
            const res = await fetch(this.apiBaseUrl + '/api/health');
            const data = await res.json();
            console.log('后端连接成功:', data);
        } catch (e) {
            console.error('后端连接失败:', e);
            this.showNotification('后端服务连接失败，请确认服务已启动', 'error');
        }
    }

    initMarkdown() {
        const checkMarked = () => {
            if (typeof marked !== 'undefined') {
                try {
                    marked.setOptions({ breaks: true, gfm: true, headerIds: false, mangle: false });
                    if (typeof hljs !== 'undefined') {
                        marked.setOptions({
                            highlight: function(code, lang) {
                                if (lang && hljs.getLanguage(lang)) {
                                    try { return hljs.highlight(code, { language: lang }).value; } catch (err) {}
                                }
                                return code;
                            }
                        });
                    }
                    console.log('Markdown 渲染库初始化成功');
                } catch (e) {}
            } else {
                setTimeout(checkMarked, 100);
            }
        };
        checkMarked();
    }

    renderMarkdown(content) {
        if (!content) return '';
        if (typeof marked === 'undefined') {
            return this.escapeHtml(content);
        }
        try {
            return marked.parse(content);
        } catch (e) {
            return this.escapeHtml(content);
        }
    }

    highlightCodeBlocks(container) {
        if (typeof hljs !== 'undefined' && container) {
            try {
                container.querySelectorAll('pre code').forEach((block) => {
                    if (!block.classList.contains('hljs')) hljs.highlightElement(block);
                });
            } catch (e) {}
        }
    }

    initializeElements() {
        this.sidebar = document.querySelector('.sidebar');
        this.newChatBtn = document.getElementById('newChatBtn');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.toolsBtn = document.getElementById('toolsBtn');
        this.toolsMenu = document.getElementById('toolsMenu');
        this.uploadFileItem = document.getElementById('uploadFileItem');
        this.modeSelectorBtn = document.getElementById('modeSelectorBtn');
        this.modeDropdown = document.getElementById('modeDropdown');
        this.currentModeText = document.getElementById('currentModeText');
        this.fileInput = document.getElementById('fileInput');
        this.chatMessages = document.getElementById('chatMessages');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.chatContainer = document.querySelector('.chat-container');
        this.welcomeGreeting = document.getElementById('welcomeGreeting');
        this.chatHistoryList = document.getElementById('chatHistoryList');
        
        this.checkAndSetCentered();
    }

    bindEvents() {
        if (this.newChatBtn) {
            this.newChatBtn.addEventListener('click', () => this.newChat());
        }
        
        if (this.modeSelectorBtn) {
            this.modeSelectorBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.toggleModeDropdown();
            });
        }
        
        document.querySelectorAll('.dropdown-item').forEach(item => {
            item.addEventListener('click', (e) => {
                const mode = item.getAttribute('data-mode');
                this.selectMode(mode);
                this.closeModeDropdown();
            });
        });
        
        document.addEventListener('click', (e) => {
            if (!this.modeSelectorBtn.contains(e.target) && !this.modeDropdown.contains(e.target)) {
                this.closeModeDropdown();
            }
        });
        
        if (this.sendButton) {
            this.sendButton.addEventListener('click', () => this.sendMessage());
        }
        
        if (this.messageInput) {
            this.messageInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });
        }
        
        if (this.toolsBtn) {
            this.toolsBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.toggleToolsMenu();
            });
        }
        
        if (this.uploadFileItem) {
            this.uploadFileItem.addEventListener('click', () => {
                if (this.fileInput) this.fileInput.click();
                this.closeToolsMenu();
            });
        }
        
        document.addEventListener('click', (e) => {
            if (this.toolsBtn && this.toolsMenu && !this.toolsBtn.contains(e.target) && !this.toolsMenu.contains(e.target)) {
                this.closeToolsMenu();
            }
        });
        
        if (this.fileInput) {
            this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        }
    }

    toggleToolsMenu() {
        if (this.toolsMenu && this.toolsBtn) {
            const wrapper = this.toolsBtn.closest('.tools-btn-wrapper');
            if (wrapper) wrapper.classList.toggle('active');
        }
    }

    closeToolsMenu() {
        if (this.toolsMenu && this.toolsBtn) {
            const wrapper = this.toolsBtn.closest('.tools-btn-wrapper');
            if (wrapper) wrapper.classList.remove('active');
        }
    }

    // ---- 新建对话 ----
    newChat() {
        if (this.isStreaming) {
            this.showNotification('请等待当前对话完成后再新建对话', 'warning');
            return;
        }
        
        if (this.currentChatHistory.length > 0) {
            if (this.isCurrentChatFromHistory) {
                this.updateCurrentChatHistory();
            } else {
                this.saveCurrentChat();
            }
        }
        
        this.isStreaming = false;
        if (this.messageInput) this.messageInput.value = '';
        this.currentChatHistory = [];
        this.isCurrentChatFromHistory = false;
        if (this.chatMessages) this.chatMessages.innerHTML = '';
        this.sessionId = this.generateSessionId();
        this.currentMode = 'stream';
        this.updateUI();
        this.checkAndSetCentered();
        if (this.chatContainer) this.chatContainer.style.transition = 'all 0.5s ease';
        this.renderChatHistory();
    }
    
    saveCurrentChat() {
        if (this.currentChatHistory.length === 0) return;
        const existingIndex = this.chatHistories.findIndex(h => h.id === this.sessionId);
        if (existingIndex !== -1) { this.updateCurrentChatHistory(); return; }
        const firstUserMessage = this.currentChatHistory.find(msg => msg.type === 'user');
        const title = firstUserMessage ? (firstUserMessage.content.substring(0, 30) + (firstUserMessage.content.length > 30 ? '...' : '')) : '新对话';
        const chatHistory = {
            id: this.sessionId, title: title,
            messages: [...this.currentChatHistory],
            createdAt: new Date().toISOString(), updatedAt: new Date().toISOString()
        };
        this.chatHistories.unshift(chatHistory);
        if (this.chatHistories.length > 50) this.chatHistories = this.chatHistories.slice(0, 50);
        this.saveChatHistories();
    }
    
    updateCurrentChatHistory() {
        if (this.currentChatHistory.length === 0) return;
        const existingIndex = this.chatHistories.findIndex(h => h.id === this.sessionId);
        if (existingIndex === -1) { this.saveCurrentChat(); return; }
        const history = this.chatHistories[existingIndex];
        history.messages = [...this.currentChatHistory];
        history.updatedAt = new Date().toISOString();
        const firstUserMessage = this.currentChatHistory.find(msg => msg.type === 'user');
        if (firstUserMessage) {
            const newTitle = firstUserMessage.content.substring(0, 30) + (firstUserMessage.content.length > 30 ? '...' : '');
            if (history.title !== newTitle) history.title = newTitle;
        }
        this.saveChatHistories();
    }
    
    loadChatHistories() {
        try { const stored = localStorage.getItem('chatHistories'); return stored ? JSON.parse(stored) : []; } catch (e) { return []; }
    }
    
    saveChatHistories() {
        try { localStorage.setItem('chatHistories', JSON.stringify(this.chatHistories)); } catch (e) {}
    }
    
    renderChatHistory() {
        if (!this.chatHistoryList) return;
        this.chatHistoryList.innerHTML = '';
        if (this.chatHistories.length === 0) return;
        this.chatHistories.forEach((history) => {
            const historyItem = document.createElement('div');
            historyItem.className = 'history-item';
            historyItem.dataset.historyId = history.id;
            historyItem.innerHTML = `
                <div class="history-item-content">
                    <span class="history-item-title">${this.escapeHtml(history.title)}</span>
                </div>
                <button class="history-item-delete" data-history-id="${history.id}" title="删除">
                    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M18 6L6 18M6 6L18 18" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                    </svg>
                </button>
            `;
            historyItem.addEventListener('click', (e) => {
                if (!e.target.closest('.history-item-delete')) this.loadChatHistory(history.id);
            });
            historyItem.querySelector('.history-item-delete').addEventListener('click', (e) => {
                e.stopPropagation();
                this.deleteChatHistory(history.id);
            });
            this.chatHistoryList.appendChild(historyItem);
        });
    }
    
    loadChatHistory(historyId) {
        const history = this.chatHistories.find(h => h.id === historyId);
        if (!history) return;
        if (this.currentChatHistory.length > 0 && this.sessionId !== historyId) {
            if (this.isCurrentChatFromHistory) this.updateCurrentChatHistory();
            else this.saveCurrentChat();
        }
        this.sessionId = history.id;
        this.currentChatHistory = [...history.messages];
        this.isCurrentChatFromHistory = true;
        if (this.chatMessages) {
            this.chatMessages.innerHTML = '';
            history.messages.forEach(msg => this.addMessage(msg.type, msg.content, false, false));
        }
        this.checkAndSetCentered();
        this.renderChatHistory();
    }
    
    deleteChatHistory(historyId) {
        this.chatHistories = this.chatHistories.filter(h => h.id !== historyId);
        this.saveChatHistories();
        this.renderChatHistory();
        if (this.sessionId === historyId) {
            this.currentChatHistory = [];
            if (this.chatMessages) this.chatMessages.innerHTML = '';
            this.sessionId = this.generateSessionId();
            this.checkAndSetCentered();
        }
    }

    toggleModeDropdown() {
        if (this.modeSelectorBtn && this.modeDropdown) {
            const wrapper = this.modeSelectorBtn.closest('.mode-selector-wrapper');
            if (wrapper) wrapper.classList.toggle('active');
        }
    }

    closeModeDropdown() {
        if (this.modeSelectorBtn && this.modeDropdown) {
            const wrapper = this.modeSelectorBtn.closest('.mode-selector-wrapper');
            if (wrapper) wrapper.classList.remove('active');
        }
    }

    selectMode(mode) {
        if (this.isStreaming) { this.showNotification('请等待当前对话完成后再切换模式', 'warning'); return; }
        this.currentMode = mode;
        this.updateUI();
        const modeNames = { 'stream': '流式', 'quick': '快速' };
        this.showNotification(`已切换到${modeNames[mode]}模式`, 'info');
    }

    updateUI() {
        const modeNames = { 'stream': '流式', 'quick': '快速' };
        if (this.currentModeText) this.currentModeText.textContent = modeNames[this.currentMode] || '快速';
        document.querySelectorAll('.dropdown-item').forEach(item => {
            item.classList.toggle('active', item.getAttribute('data-mode') === this.currentMode);
        });
        if (this.sendButton) this.sendButton.disabled = this.isStreaming;
        if (this.messageInput) {
            this.messageInput.disabled = this.isStreaming;
            this.messageInput.placeholder = '问问 VolSeek Agent';
        }
    }

    generateSessionId() {
        return 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
    }

    // ---- 发送消息 ----
    async sendMessage() {
        let message = '';
        if (this.messageInput) message = this.messageInput.value.trim();
        
        if (!message) {
            this.showNotification('请输入消息内容', 'warning');
            return;
        }

        if (this.isStreaming) {
            this.showNotification('请等待当前对话完成', 'warning');
            return;
        }

        console.log('发送消息:', message, '模式:', this.currentMode);
        this.addMessage('user', message);
        if (this.messageInput) this.messageInput.value = '';

        this.isStreaming = true;
        this.updateUI();

        try {
            if (this.currentMode === 'quick') {
                await this.sendQuickMessage(message);
            } else if (this.currentMode === 'stream') {
                await this.sendStreamMessage(message);
            }
        } catch (error) {
            console.error('发送消息失败:', error);
            this.addMessage('assistant', '抱歉，发送消息时出现错误：' + error.message);
        } finally {
            this.isStreaming = false;
            this.updateUI();
            if (this.isCurrentChatFromHistory && this.currentChatHistory.length > 0) {
                this.updateCurrentChatHistory();
                this.renderChatHistory();
            }
        }
    }

    // ---- 快速模式 (POST /api/query/sync) ----
    async sendQuickMessage(message) {
        console.log('快速模式: 请求 /api/query/sync');
        const response = await fetch(`${this.apiBaseUrl}/api/query/sync`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: message })
        });

        if (!response.ok) {
            throw new Error(`HTTP错误: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        console.log('快速模式 响应:', data);

        if (data.answer) {
            this.addMessage('assistant', data.answer);
        } else {
            throw new Error('未生成回答');
        }
    }

    // ---- 流式模式 (POST /api/query → SSE 实时流) ----
    async sendStreamMessage(message) {
        console.log('流式模式: 请求 /api/query');

        // 先创建一个空的 assistant 消息占位
        const msgEl = this.addMessage('assistant', '', true);
        let fullAnswer = '';
        let prevLen = 0;
        let streamFinished = false;

        // 创建一个状态展示元素
        const statusEl = document.createElement('div');
        statusEl.className = 'thinking-status';
        const wrapper = msgEl?.querySelector('.message-content-wrapper');
        if (wrapper) {
            wrapper.insertBefore(statusEl, wrapper.firstChild);
        }

        // 缓存消息内容元素
        const mc = msgEl?.querySelector('.message-content');

        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            xhr.open('POST', `${this.apiBaseUrl}/api/query`);
            xhr.setRequestHeader('Content-Type', 'application/json');

            // 增量解析 SSE 数据（定义在 Promise 内，可以访问 xhr/resolve）
            const processStreamData = () => {
                if (streamFinished) return;
                const all = xhr.responseText;
                const newText = all.substring(prevLen);
                prevLen = all.length;
                if (!newText) return;

                for (const line of newText.split('\n')) {
                    const t = line.trim();
                    if (!t.startsWith('data: ')) continue;
                    try {
                        const ev = JSON.parse(t.substring(6));

                        if (ev.type === 'plan' && ev.content) {
                            this.updateThinkingStatus(statusEl, 'plan', ev.content);
                        } else if (ev.type === 'think' && ev.content) {
                            this.updateThinkingStatus(statusEl, 'think', ev.content);
                        } else if (ev.type === 'tool_call' && ev.content) {
                            this.updateThinkingStatus(statusEl, 'tool_call', ev.content);
                        } else if (ev.type === 'tool_result' && ev.content) {
                            this.updateThinkingStatus(statusEl, 'tool_result', ev.content);
                        } else if (ev.type === 'reflect' && ev.content) {
                            this.updateThinkingStatus(statusEl, 'reflect', ev.content);
                        } else if (ev.type === 'error' && ev.content) {
                            this.updateThinkingStatus(statusEl, 'error', ev.content);
                        } else if (ev.type === 'answer' && ev.content) {
                            fullAnswer += ev.content;
                            if (mc) mc.textContent = fullAnswer;
                            this.scrollToBottom();
                        }
                    } catch (e) {}
                }
            };

            // 完成处理
            const finishStream = () => {
                if (streamFinished) return;
                streamFinished = true;

                // 兜底：从完整响应中提取 answer
                if (!fullAnswer && xhr.responseText) {
                    for (const line of xhr.responseText.split('\n')) {
                        const t = line.trim();
                        if (!t.startsWith('data: ')) continue;
                        try {
                            const ev = JSON.parse(t.substring(6));
                            if (ev.type === 'answer' && ev.content) {
                                fullAnswer += ev.content;
                            } else if (ev.type === 'done' && ev.content && !fullAnswer) {
                                fullAnswer = ev.content;
                            }
                        } catch (e) {}
                    }
                }

                console.log('流式完成, 总长度:', fullAnswer.length);
                if (statusEl && statusEl.parentNode) statusEl.remove();
                if (msgEl) {
                    msgEl.classList.remove('streaming');
                    const mcFinal = msgEl.querySelector('.message-content');
                    if (mcFinal && fullAnswer) {
                        mcFinal.innerHTML = this.renderMarkdown(fullAnswer);
                        this.highlightCodeBlocks(mcFinal);
                    } else if (mcFinal) {
                        mcFinal.textContent = '（未生成回答）';
                    }
                }
                if (fullAnswer) {
                    this.currentChatHistory.push({
                        type: 'assistant', content: fullAnswer, timestamp: new Date().toISOString()
                    });
                    if (this.isCurrentChatFromHistory) {
                        this.updateCurrentChatHistory();
                        this.renderChatHistory();
                    }
                }
                resolve();
            };

            // onreadystatechange 比 onprogress 更可靠
            xhr.onreadystatechange = () => {
                if (xhr.readyState === 3) {
                    processStreamData();
                } else if (xhr.readyState === 4) {
                    processStreamData();
                    finishStream();
                }
            };

            xhr.onerror = () => reject(new Error('网络错误'));
            xhr.send(JSON.stringify({ query: message }));
        });
    }

    // ---- 更新思考状态展示 ----
    updateThinkingStatus(el, type, content) {
        if (!el) return;
        const icons = {
            plan: '🧠',
            think: '💭',
            tool_call: '🔧',
            tool_result: '✅',
            reflect: '🔄',
            error: '❌',
        };
        const icon = icons[type] || '•';
        // 对于 tool_call 和 tool_result 等较长的内容，只展示第一行摘要
        let displayText = content;
        if (type === 'tool_result') {
            const firstLine = content.split('\n')[0];
            displayText = firstLine.length > 60 ? firstLine.substring(0, 60) + '...' : firstLine;
        } else if (type === 'plan') {
            const lines = content.split('\n');
            displayText = lines.length > 3 ? lines.slice(0, 3).join('\n') + '...' : content;
        }
        el.innerHTML = `<div class="status-line status-${type}">${icon} ${this.escapeHtml(displayText)}</div>`;
        this.scrollToBottom();
    }

    // ---- 添加消息 ----
    addMessage(type, content, isStreaming = false, saveToHistory = true) {
        const isFirst = this.chatMessages && this.chatMessages.querySelectorAll('.message').length === 0;
        
        if (!isStreaming && saveToHistory && content) {
            this.currentChatHistory.push({ type, content, timestamp: new Date().toISOString() });
        }
        
        const div = document.createElement('div');
        div.className = `message ${type}${isStreaming ? ' streaming' : ''}`;

        if (type === 'assistant') {
            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none"><path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" fill="white"/></svg>`;
            div.appendChild(avatar);
        }

        const wrapper = document.createElement('div');
        wrapper.className = 'message-content-wrapper';
        const mc = document.createElement('div');
        mc.className = 'message-content';

        if (type === 'assistant' && !isStreaming) {
            mc.innerHTML = this.renderMarkdown(content);
            this.highlightCodeBlocks(mc);
        } else {
            mc.textContent = content;
        }

        wrapper.appendChild(mc);
        div.appendChild(wrapper);

        if (this.chatMessages) {
            this.chatMessages.appendChild(div);
            if (isFirst && this.chatContainer) {
                this.chatContainer.classList.remove('centered');
                this.chatContainer.style.transition = 'all 0.5s ease';
            }
            this.scrollToBottom();
        }
        return div;
    }

    checkAndSetCentered() {
        if (this.chatMessages && this.chatContainer) {
            const has = this.chatMessages.querySelectorAll('.message').length > 0;
            this.chatContainer.classList.toggle('centered', !has);
        }
    }

    scrollToBottom() {
        if (this.chatMessages) this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    showNotification(message, type = 'info') {
        const n = document.createElement('div');
        n.className = `notification ${type}`;
        n.textContent = message;
        n.style.cssText = `position:fixed;top:20px;right:20px;padding:15px 20px;border-radius:8px;color:#fff;font-weight:500;z-index:10000;animation:slideIn .3s ease;max-width:300px;`;
        const colors = { info: '#1a73e8', success: '#34a853', warning: '#fbbc04', error: '#ea4335' };
        n.style.backgroundColor = colors[type] || colors.info;
        document.body.appendChild(n);
        setTimeout(() => {
            n.style.animation = 'slideOut .3s ease';
            setTimeout(() => n.remove(), 300);
        }, 3000);
    }

    // ---- 文件上传 ----
    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            if (!this.validateFileType(file)) {
                this.showNotification('只支持上传 .txt 或 .md 格式的文件', 'error');
                this.fileInput.value = '';
                return;
            }
            this.uploadFile(file);
        }
    }

    validateFileType(file) {
        return ['.txt', '.md'].some(ext => file.name.toLowerCase().endsWith(ext));
    }

    async uploadFile(file) {
        if (!this.validateFileType(file)) {
            this.showNotification('只支持上传 .txt 或 .md 格式的文件', 'error');
            return;
        }

        const maxSize = 50 * 1024 * 1024;
        if (file.size > maxSize) {
            this.showNotification('文件大小不能超过50MB', 'error');
            return;
        }

        this.isStreaming = true;
        this.updateUI();
        this.showUploadOverlay(true, file.name);

        try {
            const fd = new FormData();
            fd.append('file', file);
            const res = await fetch(`${this.apiBaseUrl}/api/knowledge/upload`, { method: 'POST', body: fd });
            if (!res.ok) throw new Error(`HTTP错误: ${res.status}`);
            const data = await res.json();
            if (data.uuid) {
                this.addMessage('assistant', `${file.name} 上传到知识库成功`, false, true);
            } else {
                throw new Error(data.message || '上传失败');
            }
        } catch (error) {
            console.error('文件上传失败:', error);
            this.showNotification('文件上传失败: ' + error.message, 'error');
        } finally {
            if (this.fileInput) this.fileInput.value = '';
            this.isStreaming = false;
            this.showUploadOverlay(false);
            this.updateUI();
        }
    }

    showUploadOverlay(show, fileName = '') {
        if (this.loadingOverlay) {
            this.loadingOverlay.style.display = show ? 'flex' : 'none';
            if (show) {
                const t = this.loadingOverlay.querySelector('.loading-text');
                const s = this.loadingOverlay.querySelector('.loading-subtext');
                if (t) t.textContent = '正在上传文件...';
                if (s) s.textContent = fileName ? `上传: ${fileName}` : '请稍候';
                document.body.style.overflow = 'hidden';
            } else {
                document.body.style.overflow = '';
            }
        }
    }

    escapeHtml(text) {
        const d = document.createElement('div');
        d.textContent = text;
        return d.innerHTML;
    }

    stripHtml(text) {
        const d = document.createElement('div');
        d.innerHTML = text;
        return d.textContent;
    }
}

// CSS动画
const style = document.createElement('style');
style.textContent = `
@keyframes slideIn { from { transform: translateX(100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
@keyframes slideOut { from { transform: translateX(0); opacity: 1; } to { transform: translateX(100%); opacity: 0; } }
`;
document.head.appendChild(style);

document.addEventListener('DOMContentLoaded', () => {
    new SuperBizAgentApp();
});
