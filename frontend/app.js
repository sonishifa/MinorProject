/* ═══════════════════════════════════════════════════════════
   Brain-First Model Tuning Toolkit — Frontend Logic
   Keystroke capture, chat, EEG WebSocket, UI updates
   ═══════════════════════════════════════════════════════════ */

const API = window.location.origin;
const WS_URL = `ws://${window.location.host}/ws/eeg`;

// ── State ──
let keystrokeBuffer = [];
let eegWs = null;
let eegRunning = false;

// ── DOM Refs ──
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const chatMessages = document.getElementById('chat-messages');
const btnSend = document.getElementById('btn-send');
const btnReset = document.getElementById('btn-reset');
const connectionStatus = document.getElementById('connection-status');

// Emotion panel
const vaDot = document.getElementById('va-dot');
const valValence = document.getElementById('val-valence');
const valArousal = document.getElementById('val-arousal');
const valZone = document.getElementById('val-zone');
const emotionBadge = document.getElementById('chat-emotion-badge');
const emotionPanel = document.getElementById('emotion-panel');

// Signal cards
const sigTextEmotion = document.getElementById('sig-text-emotion');
const sigKsEmotion = document.getElementById('sig-ks-emotion');
const sigEegEmotion = document.getElementById('sig-eeg-emotion');
const sigTextConf = document.getElementById('sig-text-conf');
const sigKsConf = document.getElementById('sig-ks-conf');
const sigEegConf = document.getElementById('sig-eeg-conf');

// Adaptation
const adaptTone = document.getElementById('adapt-tone');
const adaptTempBar = document.getElementById('adapt-temp-bar');
const adaptTemp = document.getElementById('adapt-temp');
const adaptMemory = document.getElementById('adapt-memory');
const adaptLatency = document.getElementById('adapt-latency');

// EEG
const btnEegToggle = document.getElementById('btn-eeg-toggle');
const eegEmotionSelect = document.getElementById('eeg-emotion-select');
const eegCanvas = document.getElementById('eeg-canvas');
const eegCtx = eegCanvas.getContext('2d');
const eegPredicted = document.getElementById('eeg-predicted');
const eegConf = document.getElementById('eeg-conf');

// ═══════════════════════════════════════════════════════════
// KEYSTROKE CAPTURE
// Silently records keydown/keyup timestamps while the user types
// ═══════════════════════════════════════════════════════════

chatInput.addEventListener('keydown', (e) => {
    keystrokeBuffer.push({
        keyCode: e.keyCode || e.which,
        keyDown: performance.now(),
        keyUp: null,
        key: e.key,
    });
});

chatInput.addEventListener('keyup', (e) => {
    // Find the most recent keydown for this keyCode without a keyUp
    for (let i = keystrokeBuffer.length - 1; i >= 0; i--) {
        if (keystrokeBuffer[i].keyCode === (e.keyCode || e.which) && keystrokeBuffer[i].keyUp === null) {
            keystrokeBuffer[i].keyUp = performance.now();
            break;
        }
    }
});

// Auto-resize textarea
chatInput.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
});

// ═══════════════════════════════════════════════════════════
// CHAT
// ═══════════════════════════════════════════════════════════

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const message = chatInput.value.trim();
    if (!message) return;

    // Add user message to chat
    addMessage(message, 'user');
    chatInput.value = '';
    chatInput.style.height = 'auto';
    btnSend.disabled = true;

    // Prepare keystroke events (filter out incomplete ones)
    // Convert performance.now() ms timestamps to relative seconds
    // (training data uses seconds, browser gives ms)
    const validEvents = keystrokeBuffer.filter(k => k.keyUp !== null);
    const t0 = validEvents.length > 0 ? validEvents[0].keyDown : 0;
    const events = validEvents.map(k => ({
        keyCode: k.keyCode,
        keyDown: (k.keyDown - t0) / 1000,
        keyUp: (k.keyUp - t0) / 1000,
    }));

    // Clear buffer
    keystrokeBuffer = [];

    // Show typing indicator
    const typingEl = showTyping();

    try {
        const res = await fetch(`${API}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, keystroke_events: events }),
        });

        const data = await res.json();

        // Remove typing indicator
        typingEl.remove();

        // Add assistant response
        const meta = data.adaptation;
        const metaText = `zone: ${meta.zone} | temp: ${meta.temperature} | memory: ${meta.memory_depth} | latency: ${meta.actual_latency_ms}ms`;
        addMessage(data.response, 'assistant', metaText);

        // Update all panels
        updateEmotionPanel(data.emotional_state);
        updateSignals(data.signals);
        updateAdaptation(data.adaptation);

        setConnected(true);
    } catch (err) {
        typingEl.remove();
        addMessage('Connection error. Is the backend running?', 'system');
        setConnected(false);
    }

    btnSend.disabled = false;
    chatInput.focus();
});

function addMessage(text, type, meta = null) {
    const div = document.createElement('div');
    div.className = `message ${type}-message`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    if (type === 'assistant') {
        // Render markdown to HTML for assistant messages
        contentDiv.innerHTML = renderMarkdown(text);
    } else {
        const p = document.createElement('p');
        p.textContent = text;
        contentDiv.appendChild(p);
    }
    div.appendChild(contentDiv);

    if (meta) {
        const metaDiv = document.createElement('div');
        metaDiv.className = 'adaptation-meta';
        metaDiv.textContent = meta;
        div.appendChild(metaDiv);
    }

    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function renderMarkdown(text) {
    // Lightweight markdown → HTML converter
    let html = text
        // Escape HTML first
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        // Code blocks (```) — must come before inline code
        .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code class="lang-$1">$2</code></pre>')
        // Inline code (`text`)
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        // Bold (**text**)
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        // Italic (*text*)
        .replace(/\*([^*]+)\*/g, '<em>$1</em>')
        // Headings (### text)
        .replace(/^### (.+)$/gm, '<h4>$1</h4>')
        .replace(/^## (.+)$/gm, '<h3>$1</h3>')
        .replace(/^# (.+)$/gm, '<h2>$1</h2>')
        // Horizontal rule
        .replace(/^---$/gm, '<hr>')
        // Unordered lists (- item or * item)
        .replace(/^[\-\*] (.+)$/gm, '<li>$1</li>')
        // Numbered lists (1. item)
        .replace(/^\d+\. (.+)$/gm, '<li>$1</li>')
        // Wrap consecutive <li> in <ul>
        .replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>')
        // Paragraphs: double newline → new paragraph
        .replace(/\n\n/g, '</p><p>')
        // Single newline → line break
        .replace(/\n/g, '<br>');

    return '<p>' + html + '</p>';
}

function showTyping() {
    const div = document.createElement('div');
    div.className = 'typing-indicator';
    div.innerHTML = '<span></span><span></span><span></span>';
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return div;
}

// ═══════════════════════════════════════════════════════════
// EMOTION PANEL UPDATES
// ═══════════════════════════════════════════════════════════

function updateEmotionPanel(state) {
    if (!state) return;

    const v = state.valence || 0;
    const a = state.arousal || 0;
    const zone = state.zone || 'neutral';
    const emotion = state.emotion || 'neutral';

    // Update VA dot position (map -1..1 to 0..100%)
    const dotX = 50 + v * 45; // center + offset
    const dotY = 50 - a * 45; // center - offset (Y is inverted)
    vaDot.style.left = `${dotX}%`;
    vaDot.style.top = `${dotY}%`;

    // Update values
    valValence.textContent = v.toFixed(3);
    valArousal.textContent = a.toFixed(3);
    valZone.textContent = zone.replace('_', ' ');

    // Color the valence based on sign
    valValence.style.color = v > 0.1 ? '#4ade80' : v < -0.1 ? '#f87171' : '#94a3b8';
    valArousal.style.color = a > 0.1 ? '#fbbf24' : a < -0.1 ? '#60a5fa' : '#94a3b8';

    // Update badge
    emotionBadge.textContent = emotion;
    emotionBadge.className = 'badge';
    if (zone.startsWith('positive')) emotionBadge.classList.add('positive');
    else if (zone.startsWith('negative')) emotionBadge.classList.add('negative');

    // Update panel glow
    emotionPanel.className = 'panel zone-' + zone;
}

function updateSignals(signals) {
    if (!signals) return;

    // Text signal
    if (signals.text) {
        sigTextEmotion.textContent = signals.text.emotion || '—';
        sigTextConf.style.width = ((signals.text.confidence || 0) * 100) + '%';
        document.getElementById('sig-text').classList.add('active');
    }

    // Keystroke signal
    if (signals.keystroke) {
        sigKsEmotion.textContent = signals.keystroke.emotion || '—';
        sigKsConf.style.width = ((signals.keystroke.confidence || 0) * 100) + '%';
        document.getElementById('sig-keystroke').classList.add('active');
    } else {
        sigKsEmotion.textContent = '—';
        sigKsConf.style.width = '0%';
        document.getElementById('sig-keystroke').classList.remove('active');
    }

    // EEG signal
    if (signals.eeg) {
        sigEegEmotion.textContent = signals.eeg.emotion || '—';
        sigEegConf.style.width = ((signals.eeg.confidence || 0) * 100) + '%';
        document.getElementById('sig-eeg').classList.add('active');
    }
}

function updateAdaptation(adapt) {
    if (!adapt) return;

    // Tone — truncate long strings
    const tone = adapt.system_prompt_tone || 'balanced';
    adaptTone.textContent = tone.length > 50 ? tone.substring(0, 50) + '…' : tone;

    // Temperature bar
    const temp = adapt.temperature || 0.7;
    adaptTempBar.style.width = (temp * 100) + '%';
    adaptTemp.textContent = temp.toFixed(1);

    // Memory
    adaptMemory.textContent = (adapt.memory_depth || 7) + ' msgs';

    // Latency
    adaptLatency.textContent = (adapt.latency_ms || 0) + ' ms';
}

// ═══════════════════════════════════════════════════════════
// EEG SIMULATOR
// ═══════════════════════════════════════════════════════════

btnEegToggle.addEventListener('click', async () => {
    if (!eegRunning) {
        await startEEG();
    } else {
        await stopEEG();
    }
});

eegEmotionSelect.addEventListener('change', async () => {
    if (!eegRunning) return;
    try {
        await fetch(`${API}/api/simulate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'set_emotion', emotion: eegEmotionSelect.value }),
        });
    } catch (e) { /* silent */ }
});

async function startEEG() {
    try {
        await fetch(`${API}/api/simulate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'start', speed: 1.0 }),
        });

        // Set emotion from dropdown
        await fetch(`${API}/api/simulate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'set_emotion', emotion: eegEmotionSelect.value }),
        });

        eegRunning = true;
        btnEegToggle.textContent = 'Stop';
        btnEegToggle.classList.add('running');

        connectEEGWebSocket();
    } catch (e) {
        addMessage('Failed to start EEG simulator', 'system');
    }
}

async function stopEEG() {
    try {
        await fetch(`${API}/api/simulate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'stop' }),
        });
    } catch (e) { /* silent */ }

    eegRunning = false;
    btnEegToggle.textContent = 'Start';
    btnEegToggle.classList.remove('running');

    if (eegWs) {
        eegWs.close();
        eegWs = null;
    }
}

// ── EEG WebSocket ──
let eegWaveHistory = [];

function connectEEGWebSocket() {
    if (eegWs) eegWs.close();

    eegWs = new WebSocket(WS_URL);

    eegWs.onopen = () => {
        console.log('[EEG WS] Connected');
    };

    eegWs.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'eeg_frame') {
            drawEEGWaves(data.raw_channels, data.channel_names);
            eegPredicted.textContent = data.predicted_emotion;
            eegConf.textContent = `(${(data.prediction_confidence * 100).toFixed(0)}%)`;

            // Update EEG signal card
            sigEegEmotion.textContent = data.predicted_emotion;
            sigEegConf.style.width = (data.prediction_confidence * 100) + '%';
            document.getElementById('sig-eeg').classList.add('active');
        }
    };

    eegWs.onclose = () => {
        console.log('[EEG WS] Disconnected');
    };

    eegWs.onerror = (err) => {
        console.error('[EEG WS] Error:', err);
    };
}

// ── EEG Waveform Drawing ──
const WAVE_COLORS = [
    '#f0a500', '#4ade80', '#60a5fa', '#f87171',
    '#a78bfa', '#34d399', '#fbbf24', '#fb923c'
];

function drawEEGWaves(channels, names) {
    const dpr = window.devicePixelRatio || 1;
    const rect = eegCanvas.getBoundingClientRect();
    eegCanvas.width = rect.width * dpr;
    eegCanvas.height = rect.height * dpr;
    eegCtx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;

    // Clear
    eegCtx.fillStyle = '#12121c';
    eegCtx.fillRect(0, 0, w, h);

    if (!channels || channels.length === 0) return;

    const numChannels = channels.length;
    const channelHeight = h / numChannels;

    for (let ch = 0; ch < numChannels; ch++) {
        const data = channels[ch];
        if (!data || data.length === 0) continue;

        const yOffset = ch * channelHeight + channelHeight / 2;

        // Channel label
        eegCtx.fillStyle = '#55556b';
        eegCtx.font = '9px Inter';
        eegCtx.fillText(names?.[ch] || `Ch${ch}`, 4, yOffset - channelHeight / 2 + 12);

        // Waveform
        eegCtx.strokeStyle = WAVE_COLORS[ch % WAVE_COLORS.length];
        eegCtx.lineWidth = 1;
        eegCtx.globalAlpha = 0.8;
        eegCtx.beginPath();

        const step = w / data.length;
        for (let i = 0; i < data.length; i++) {
            const x = i * step;
            const y = yOffset + data[i] * (channelHeight * 0.3);
            if (i === 0) eegCtx.moveTo(x, y);
            else eegCtx.lineTo(x, y);
        }
        eegCtx.stroke();
        eegCtx.globalAlpha = 1;

        // Separator line
        if (ch < numChannels - 1) {
            eegCtx.strokeStyle = '#2a2a40';
            eegCtx.lineWidth = 0.5;
            eegCtx.beginPath();
            eegCtx.moveTo(0, (ch + 1) * channelHeight);
            eegCtx.lineTo(w, (ch + 1) * channelHeight);
            eegCtx.stroke();
        }
    }
}

// ═══════════════════════════════════════════════════════════
// RESET
// ═══════════════════════════════════════════════════════════

btnReset.addEventListener('click', async () => {
    try {
        await fetch(`${API}/api/reset`, { method: 'POST' });
    } catch (e) { /* silent */ }

    // Reset UI
    chatMessages.innerHTML = `
        <div class="message system-message fade-in">
            <p>Session reset. All state cleared.</p>
        </div>`;

    keystrokeBuffer = [];
    sigTextEmotion.textContent = '—';
    sigKsEmotion.textContent = '—';
    sigEegEmotion.textContent = '—';
    sigTextConf.style.width = '0%';
    sigKsConf.style.width = '0%';
    sigEegConf.style.width = '0%';
    valValence.textContent = '0.000';
    valArousal.textContent = '0.000';
    valZone.textContent = 'neutral';
    vaDot.style.left = '50%';
    vaDot.style.top = '50%';
    emotionBadge.textContent = 'neutral';
    emotionBadge.className = 'badge';
    emotionPanel.className = 'panel';
    eegPredicted.textContent = '—';
    eegConf.textContent = '—';

    await stopEEG();
});

// ═══════════════════════════════════════════════════════════
// CONNECTION STATUS
// ═══════════════════════════════════════════════════════════

function setConnected(connected) {
    const pill = connectionStatus;
    if (connected) {
        pill.className = 'status-pill connected';
        pill.querySelector('.label').textContent = 'Connected';
    } else {
        pill.className = 'status-pill error';
        pill.querySelector('.label').textContent = 'Disconnected';
    }
}

// Check backend status on load
async function checkStatus() {
    try {
        const res = await fetch(`${API}/api/status`);
        if (res.ok) {
            setConnected(true);
            const data = await res.json();
            if (data.eeg_simulator?.running) {
                eegRunning = true;
                btnEegToggle.textContent = 'Stop';
                btnEegToggle.classList.add('running');
                connectEEGWebSocket();
            }
        } else {
            setConnected(false);
        }
    } catch (e) {
        setConnected(false);
    }
}

// Submit on Enter (Shift+Enter for newline)
chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        chatForm.dispatchEvent(new Event('submit'));
    }
});

// ── Initialize ──
checkStatus();
chatInput.focus();

// Draw empty EEG canvas
eegCtx.fillStyle = '#12121c';
eegCtx.fillRect(0, 0, eegCanvas.width, eegCanvas.height);
eegCtx.fillStyle = '#55556b';
eegCtx.font = '12px Inter';
eegCtx.textAlign = 'center';
eegCtx.fillText('Start simulator to see EEG waveforms', eegCanvas.width / 2, eegCanvas.height / 2);
