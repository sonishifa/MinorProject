/* ═══════════════════════════════════════════════════════════
   Brain-First Model Tuning Toolkit — Frontend Logic
   Keystroke capture, facial capture, chat
   ═══════════════════════════════════════════════════════════ */

const API = window.location.origin;

// ── State ──
let keystrokeBuffer = [];

// ── Camera State ──
let cameraEnabled   = false;
let videoStream     = null;
let capturedFrames  = [];
let captureInterval = null;

// ── DOM Refs ──
const chatForm         = document.getElementById('chat-form');
const chatInput        = document.getElementById('chat-input');
const chatMessages     = document.getElementById('chat-messages');
const btnSend          = document.getElementById('btn-send');
const btnReset         = document.getElementById('btn-reset');
const connectionStatus = document.getElementById('connection-status');

// Emotion panel
const vaDot        = document.getElementById('va-dot');
const valValence   = document.getElementById('val-valence');
const valArousal   = document.getElementById('val-arousal');
const valZone      = document.getElementById('val-zone');
const emotionBadge = document.getElementById('chat-emotion-badge');
const emotionPanel = document.getElementById('emotion-panel');

// Signal cards
const sigTextEmotion   = document.getElementById('sig-text-emotion');
const sigKsEmotion     = document.getElementById('sig-ks-emotion');
const sigFacialEmotion = document.getElementById('sig-facial-emotion');
const sigTextConf      = document.getElementById('sig-text-conf');
const sigKsConf        = document.getElementById('sig-ks-conf');
const sigFacialConf    = document.getElementById('sig-facial-conf');

// Adaptation
const adaptTone      = document.getElementById('adapt-tone');
const adaptTempBar   = document.getElementById('adapt-temp-bar');
const adaptTemp      = document.getElementById('adapt-temp');
const adaptMemory    = document.getElementById('adapt-memory');
const adaptSources   = document.getElementById('adapt-sources');
const adaptModeBadge = document.getElementById('adaptation-mode-badge');

// Camera
const btnCamera      = document.getElementById('btn-camera');
const cameraStatusEl = document.getElementById('camera-status');


// ═══════════════════════════════════════════════════════════
// CAMERA — permission → capture → send
// ═══════════════════════════════════════════════════════════

window.toggleCamera = async function() {
    if (cameraEnabled) {
        stopCamera();
    } else {
        await startCamera();
    }
};

async function startCamera() {
    try {
        videoStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 320, height: 240, facingMode: 'user' }
        });
        const video   = document.getElementById('videoFeed');
        video.srcObject = videoStream;
        cameraEnabled   = true;

       // Change these two lines:
        btnCamera.textContent      = 'Disable Camera Detection';
        btnCamera.classList.add('camera-active');
        cameraStatusEl.textContent = ' Camera active — facial emotion being detected';
        sigFacialEmotion.textContent = 'detecting...';
        document.getElementById('sig-facial').classList.add('active');

    } catch (err) {
        cameraStatusEl.textContent = 'Camera permission denied';
        sigFacialEmotion.textContent = 'denied';
        cameraEnabled = false;
    }
}

function stopCamera() {
    if (videoStream) {
        videoStream.getTracks().forEach(t => t.stop());
        videoStream = null;
    }
    clearInterval(captureInterval);
    captureInterval = null;
    cameraEnabled   = false;

    btnCamera.textContent = 'Enable Camera Detection';
    btnCamera.classList.remove('sensor-active');
    cameraStatusEl.textContent = 'Camera off — using text + keystroke only';
    sigFacialEmotion.textContent = 'off';
    sigFacialConf.style.width = '0%';
    document.getElementById('sig-facial').classList.remove('active');
}

function captureFrame() {
    if (!cameraEnabled || !videoStream) return null;
    const video  = document.getElementById('videoFeed');
    if (video.readyState < 2) return null;
    const canvas = document.createElement('canvas');
    canvas.width  = 320;
    canvas.height = 240;
    canvas.getContext('2d').drawImage(video, 0, 0);
    return canvas.toDataURL('image/jpeg', 0.7);
}


// ═══════════════════════════════════════════════════════════
// KEYSTROKE CAPTURE
// ═══════════════════════════════════════════════════════════

let keystrokeTrackingEnabled = true;

window.toggleKeystroke = function() {
    keystrokeTrackingEnabled = !keystrokeTrackingEnabled;
    const btn = document.getElementById('btn-keystroke');
    const status = document.getElementById('keystroke-status');
    
    if (keystrokeTrackingEnabled) {
        btn.textContent = 'Disable Keystroke Tracking';
        btn.classList.add('sensor-active');
        status.textContent = 'Tracking active — learning typing pattern';
    } else {
        btn.textContent = 'Enable Keystroke Tracking';
        btn.classList.remove('sensor-active');
        status.textContent = 'Keystroke tracking disabled';
        keystrokeBuffer = [];
        document.getElementById('sig-keystroke').classList.remove('active');
        sigKsEmotion.textContent = 'off';
        sigKsConf.style.width = '0%';
    }
};

chatInput.addEventListener('keydown', (e) => {
    // Start frame capture when user begins typing
    if (cameraEnabled && !captureInterval) {
        capturedFrames  = [];
        captureInterval = setInterval(() => {
            const frame = captureFrame();
            if (frame) capturedFrames.push(frame);
        }, 2000);
    }

    if (!keystrokeTrackingEnabled) return;

    keystrokeBuffer.push({
        keyCode: e.keyCode || e.which,
        keyDown: performance.now(),
        keyUp:   null,
        key:     e.key,
    });
});

chatInput.addEventListener('keyup', (e) => {
    if (!keystrokeTrackingEnabled) return;

    let dwellTime = 0;
    for (let i = keystrokeBuffer.length - 1; i >= 0; i--) {
        if (keystrokeBuffer[i].keyCode === (e.keyCode || e.which) && keystrokeBuffer[i].keyUp === null) {
            keystrokeBuffer[i].keyUp = performance.now();
            dwellTime = Math.round(keystrokeBuffer[i].keyUp - keystrokeBuffer[i].keyDown);
            break;
        }
    }
    
    // Live UI metric update
    const status = document.getElementById('keystroke-status');
    if (keystrokeTrackingEnabled && keystrokeBuffer.length > 0) {
        status.textContent = `Tracking active — buffered: ${keystrokeBuffer.length} keys | last dwell: ${dwellTime}ms`;
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

    addMessage(message, 'user');
    chatInput.value = '';
    chatInput.style.height = 'auto';
    btnSend.disabled = true;

    // Stop frame capture, grab one final frame
    clearInterval(captureInterval);
    captureInterval = null;
    if (cameraEnabled) {
        const finalFrame = captureFrame();
        if (finalFrame) capturedFrames.push(finalFrame);
    }

    // Prepare keystroke events — normalise to seconds from first keydown
    const validEvents = keystrokeBuffer.filter(k => k.keyUp !== null);
    const t0          = validEvents.length > 0 ? validEvents[0].keyDown : 0;
    const events      = validEvents.map(k => ({
        keyCode: k.keyCode,
        keyDown: (k.keyDown - t0) / 1000,   // ms → seconds
        keyUp:   (k.keyUp   - t0) / 1000,
    }));

    const framesToSend = [...capturedFrames];
    keystrokeBuffer = [];
    capturedFrames  = [];
    
    if (keystrokeTrackingEnabled) {
        document.getElementById('keystroke-status').textContent = ' Tracking active — learning typing pattern';
    }

    const typingEl = showTyping();

    try {
        const res = await fetch(`${API}/api/chat`, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                keystroke_events: events,
                facial_frames:    framesToSend,
            }),
        });

        const data = await res.json();
        typingEl.remove();

        const meta     = data.adaptation;
        const metaText = `zone: ${meta.zone} | temp: ${meta.temperature} | tone: ${meta.system_prompt_tone}`;
        addMessage(data.response, 'assistant', metaText);

        updateEmotionPanel(data.emotional_state);
        updateSignals(data.signals);
        updateAdaptation(data.adaptation);   // now includes effective_weights
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
        contentDiv.innerHTML = renderMarkdown(text);
    } else {
        const p = document.createElement('p');
        p.textContent = text;
        contentDiv.appendChild(p);
    }
    div.appendChild(contentDiv);

    if (meta) {
        const metaDiv = document.createElement('div');
        metaDiv.className   = 'adaptation-meta';
        metaDiv.textContent = meta;
        div.appendChild(metaDiv);
    }

    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}


function renderMarkdown(text) {
    let html = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code class="lang-$1">$2</code></pre>')
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        .replace(/\*([^*]+)\*/g, '<em>$1</em>')
        .replace(/^### (.+)$/gm, '<h4>$1</h4>')
        .replace(/^## (.+)$/gm,  '<h3>$1</h3>')
        .replace(/^# (.+)$/gm,   '<h2>$1</h2>')
        .replace(/^---$/gm, '<hr>')
        .replace(/^[\-\*] (.+)$/gm, '<li>$1</li>')
        .replace(/^\d+\. (.+)$/gm,  '<li>$1</li>')
        .replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>')
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g,   '<br>');
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
    const v    = state.valence || 0;
    const a    = state.arousal || 0;
    const zone = state.zone    || 'neutral';

    vaDot.style.left = `${50 + v * 45}%`;
    vaDot.style.top  = `${50 - a * 45}%`;

    valValence.textContent = v.toFixed(3);
    valArousal.textContent = a.toFixed(3);
    valZone.textContent    = zone.replace('_', ' ');

    valValence.style.color = v > 0.1 ? '#4ade80' : v < -0.1 ? '#f87171' : '#94a3b8';
    valArousal.style.color = a > 0.1 ? '#fbbf24' : a < -0.1 ? '#60a5fa' : '#94a3b8';

    const emotion = state.emotion || 'neutral';
    emotionBadge.textContent = emotion;
    emotionBadge.className   = 'badge';
    if (zone.startsWith('positive')) emotionBadge.classList.add('positive');
    else if (zone.startsWith('negative')) emotionBadge.classList.add('negative');

    emotionPanel.className = 'panel zone-' + zone;
}


function updateSignals(signals) {
    if (!signals) return;

    // Text
    if (signals.text) {
        sigTextEmotion.textContent = signals.text.emotion || '—';
        sigTextConf.style.width    = ((signals.text.confidence || 0) * 100) + '%';
        document.getElementById('sig-text').classList.add('active');
    }

    // Keystroke
    if (signals.keystroke) {
        sigKsEmotion.textContent = signals.keystroke.emotion || '—';
        sigKsConf.style.width    = ((signals.keystroke.confidence || 0) * 100) + '%';
        document.getElementById('sig-keystroke').classList.add('active');
    } else {
        sigKsEmotion.textContent = '—';
        sigKsConf.style.width    = '0%';
        document.getElementById('sig-keystroke').classList.remove('active');
    }

    // Facial
    if (signals.facial) {
        const f = signals.facial;
        if (f.camera_active && f.confidence > 0) {
            sigFacialEmotion.textContent = f.emotion || '—';
            sigFacialConf.style.width    = (f.confidence * 100) + '%';
            document.getElementById('sig-facial').classList.add('active');
        } else if (f.camera_active) {
            sigFacialEmotion.textContent = 'no face';
            sigFacialConf.style.width    = '0%';
        }
    }
}


function updateAdaptation(adapt) {
    if (!adapt) return;

    adaptTone.textContent    = adapt.system_prompt_tone || 'balanced';
    const temp               = adapt.temperature || 0.6;
    adaptTempBar.style.width = (temp * 100) + '%';
    adaptTemp.textContent    = temp.toFixed(1);
    adaptMemory.textContent  = (adapt.memory_depth || 7) + ' msgs';

    // effective_weights is now passed through from emotional_state
    // via llm_adapter.py — shows which modalities actually contributed
    const weights = adapt.effective_weights || {};
    const active  = Object.keys(weights).filter(k => weights[k] > 0);
    adaptSources.textContent = active.length > 0 ? active.join(' + ') : 'text';

    const zone = adapt.zone || 'neutral';
    const BADGE_LABELS = {
        'negative_high': ' Calm Mode',
        'negative_low':  ' Empathy Mode',
        'positive_high': ' Energy Mode',
        'positive_low':  ' Friendly Mode',
        'neutral':       ' Standard',
    };
    adaptModeBadge.textContent = BADGE_LABELS[zone] || 'Standard';
    adaptModeBadge.className   = `adaptation-badge ${zone.replace('_', '-')}-badge`;
}


// ═══════════════════════════════════════════════════════════
// RESET
// ═══════════════════════════════════════════════════════════

btnReset.addEventListener('click', async () => {
    try { await fetch(`${API}/api/reset`, { method: 'POST' }); } catch (e) { /* silent */ }

    chatMessages.innerHTML = `
        <div class="message system-message fade-in">
            <p>Session reset. All state cleared.</p>
        </div>`;

    keystrokeBuffer = [];
    capturedFrames  = [];

    // Reset VA display
    valValence.textContent = '0.000';
    valArousal.textContent = '0.000';
    valZone.textContent    = 'neutral';
    vaDot.style.left       = '50%';
    vaDot.style.top        = '50%';
    emotionBadge.textContent = 'neutral';
    emotionBadge.className   = 'badge';
    emotionPanel.className   = 'panel';

    // Reset signal cards — clear stale emotion labels and confidence bars
    sigTextEmotion.textContent = '—';
    sigKsEmotion.textContent   = '—';
    sigTextConf.style.width    = '0%';
    sigKsConf.style.width      = '0%';
    document.getElementById('sig-text').classList.remove('active');
    document.getElementById('sig-keystroke').classList.remove('active');

    // Facial card: reset to 'off' only if camera is not currently active
    if (!cameraEnabled) {
        sigFacialEmotion.textContent = 'off';
        sigFacialConf.style.width    = '0%';
        document.getElementById('sig-facial').classList.remove('active');
    } else {
        sigFacialEmotion.textContent = 'detecting...';
        sigFacialConf.style.width    = '0%';
    }

    // Reset adaptation panel
    adaptModeBadge.textContent = 'Standard';
    adaptModeBadge.className   = 'adaptation-badge neutral-badge';
    adaptTone.textContent      = 'balanced';
    adaptTempBar.style.width   = '60%';
    adaptTemp.textContent      = '0.6';
    adaptMemory.textContent    = '7 msgs';
    adaptSources.textContent   = 'text';
    
    if (keystrokeTrackingEnabled) {
        document.getElementById('keystroke-status').textContent = 'Tracking active — learning typing pattern';
    }
});


// ═══════════════════════════════════════════════════════════
// CONNECTION STATUS
// ═══════════════════════════════════════════════════════════

function setConnected(connected) {
    if (connected) {
        connectionStatus.className = 'status-pill connected';
        connectionStatus.querySelector('.label').textContent = 'Connected';
    } else {
        connectionStatus.className = 'status-pill error';
        connectionStatus.querySelector('.label').textContent = 'Disconnected';
    }
}

async function checkStatus() {
    try {
        const res = await fetch(`${API}/api/status`);
        setConnected(res.ok);
    } catch (e) {
        setConnected(false);
    }
}

// Pressing Enter submits (Shift+Enter = newline)
chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        chatForm.dispatchEvent(new Event('submit'));
    }
});

// ── Init ──
checkStatus();
chatInput.focus();