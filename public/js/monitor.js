// ══════════════════════════════════════════════════════════════════════════
//  GUARDIAN NET — monitor.js
//  Clicking "Start Detector" spawns guardian_all.py via the Node server.
//  Clicking "Stop Detector" kills that process.
//  All detection events arrive over WebSocket and update the UI live.
// ══════════════════════════════════════════════════════════════════════════

// ── Auth guard ────────────────────────────────────────────────────────────
const token = localStorage.getItem('token');
if (!token) window.location.href = '/login.html';

const user = JSON.parse(localStorage.getItem('user') || '{}');
const userNameEl = document.getElementById('userName');
if (userNameEl) userNameEl.textContent = user.full_name || 'User';

const logoutBtn = document.getElementById('logoutBtn');
if (logoutBtn) {
    logoutBtn.addEventListener('click', (e) => {
        e.preventDefault();
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        window.location.href = '/login.html';
    });
}

// ── State ─────────────────────────────────────────────────────────────────
let ws                  = null;
let reconnectAttempts   = 0;
const MAX_RECONNECT     = 10;
let detectorRunning     = false;
let startTime           = Date.now();
let uptimeTimer         = null;

let stats = {
    fall:   { count: 0, lastTime: null, confidence: 0 },
    voice:  { count: 0, lastTime: null, keywords:   [] },
    alerts: 0,
    state:  'MONITORING'
};

// ── Safe element helper ───────────────────────────────────────────────────
function el(id) { return document.getElementById(id); }

// ══════════════════════════════════════════════════════════════════════════
//  WEBSOCKET
// ══════════════════════════════════════════════════════════════════════════
function connectWebSocket() {
    try {
        ws = new WebSocket('ws://localhost:3000');

        ws.onopen = () => {
            reconnectAttempts = 0;
            setConnectionUI(true);
            addLog('System', 'Connected to Guardian Net server');
        };

        ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                switch (msg.type) {
                    case 'detection':      handleDetection(msg);        break;
                    case 'status':         handleStatusUpdate(msg.data); break;
                    case 'detector_log':   addLog('Detector', msg.text); break;
                    case 'detector_ready': onDetectorReady();            break;
                    case 'detector_stopped': onDetectorStopped(msg.code); break;
                    case 'error':          addLog('Error', msg.message); break;
                }
            } catch(e) { console.error('WS parse error:', e); }
        };

        ws.onclose = () => {
            setConnectionUI(false);
            if (reconnectAttempts < MAX_RECONNECT) {
                reconnectAttempts++;
                setTimeout(connectWebSocket, 3000);
            } else {
                addLog('System', '❌ Could not reconnect to server');
            }
        };

        ws.onerror = () => setConnectionUI(false);

    } catch(e) {
        setTimeout(connectWebSocket, 3000);
    }
}

function setConnectionUI(connected) {
    const led  = el('connectionLed');
    const txt  = el('connectionText');
    const sLed = el('systemLed');
    const sTxt = el('systemStatus');
    if (led)  led.className  = connected ? 'status-led active' : 'status-led offline';
    if (txt)  txt.textContent  = connected ? 'Connected'    : 'Disconnected';
    if (sLed) sLed.className = connected ? 'status-led active' : 'status-led offline';
    if (sTxt) sTxt.textContent = connected ? 'Connected'    : 'Disconnected';
}

// ══════════════════════════════════════════════════════════════════════════
//  DETECTOR START / STOP  — called by the buttons in monitor.html
// ══════════════════════════════════════════════════════════════════════════
function startDetector() {
    if (detectorRunning) return;

    // Get selected patient ID from the dropdown
    const patientId = parseInt(el('patientSelect')?.value || '1', 10);

    // Show loading overlay
    showLoading(`Starting detector for Patient ${patientId}…`);
    addLog('System', `Starting detector — patient ID ${patientId}`);

    fetch('/api/detector/start', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ patient_id: patientId })
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            detectorRunning = true;
            setDetectorUI(true);
            addLog('System', '⏳ Detector process launched — waiting for camera and mic…');
            // Loading overlay stays until detector_ready arrives (max 30s safety)
            setTimeout(hideLoading, 30000);
        } else {
            hideLoading();
            addLog('Error', data.error || 'Failed to start detector');
        }
    })
    .catch(err => {
        hideLoading();
        addLog('Error', `Start failed: ${err.message}`);
    });
}

function stopDetector() {
    if (!detectorRunning) return;

    fetch('/api/detector/stop', { method: 'POST' })
    .then(r => r.json())
    .then(() => {
        detectorRunning = false;
        setDetectorUI(false);
        addLog('System', 'Detector stopped');
        stopUptimeTimer();
    })
    .catch(err => addLog('Error', `Stop failed: ${err.message}`));
}

function resetCounters() {
    if (!confirm('Reset all counters?')) return;
    stats = { fall: { count:0, lastTime:null, confidence:0 },
               voice:{ count:0, lastTime:null, keywords:[] },
               alerts:0, state:'MONITORING' };
    updateAllCounters();
    if (ws?.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type:'command', command:'reset_counts' }));
    }
    addLog('System', 'Counters reset');
}

function changePatient() {
    const pid = el('patientSelect')?.value;
    const pidSpan = el('patientId');
    if (pid && pidSpan) pidSpan.textContent = pid;
    addLog('System', `Patient switched to ID ${pid}`);
}

// ══════════════════════════════════════════════════════════════════════════
//  CALLBACKS FROM SERVER
// ══════════════════════════════════════════════════════════════════════════
function onDetectorReady() {
    hideLoading();
    addLog('System', '✅ All detectors ready — monitoring active');
    // Start the uptime clock
    startTime = Date.now();
    startUptimeTimer();
}

function onDetectorStopped(code) {
    detectorRunning = false;
    setDetectorUI(false);
    stopUptimeTimer();
    hideLoading();
    addLog('System', `Detector process ended (exit code ${code})`);
}

// ══════════════════════════════════════════════════════════════════════════
//  DETECTION EVENT HANDLING
// ══════════════════════════════════════════════════════════════════════════
function handleDetection(data) {
    const type       = data.event_type;
    const message    = data.details   || `${type} detected`;
    const confidence = data.confidence || 0;
    const keywords   = data.keywords  || [];
    const ts         = new Date(data.timestamp || Date.now());

    if (type === 'fall') {
        stats.fall.count++;
        stats.fall.lastTime    = ts;
        stats.fall.confidence  = confidence;
        updateFallUI(confidence);
        triggerEmergency('🚨 FALL DETECTED!', message, 'fall');
        playAlarm('fall');
        addLog('FALL', message, confidence);
        flashCard('fallStatCard');

    } else if (type === 'voice') {
        stats.voice.count++;
        stats.voice.lastTime   = ts;
        stats.voice.keywords   = keywords;
        updateVoiceUI(keywords);
        triggerEmergency('🔊 VOICE EMERGENCY!', message, 'voice');
        playAlarm('voice');
        addLog('VOICE', message, confidence, keywords);
        flashCard('voiceStatCard');

    } else if (type === 'gesture') {
        triggerEmergency('🤚 GESTURE ALERT!', message, 'gesture');
        playAlarm('voice');
        addLog('GESTURE', message, confidence);
    }

    stats.alerts = stats.fall.count + stats.voice.count;
    updateAlertUI();
    addToDetectionsTable(type, message, confidence, keywords);

    // Set detection badge
    const badge = el('detectionBadge');
    const dText = el('detectionText');
    const dLed  = el('detectionLed');
    if (badge) badge.style.background = type === 'fall'
        ? 'rgba(239,68,68,0.9)' : 'rgba(16,185,129,0.9)';
    if (dText) dText.textContent = type === 'fall' ? 'FALL DETECTED' : 'VOICE EMERGENCY';
    if (dLed)  dLed.className = 'status-led emergency';

    // Auto-reset badge after 6s
    setTimeout(() => {
        if (badge) badge.style.background = 'rgba(0,0,0,0.7)';
        if (dText) dText.textContent = 'MONITORING';
        if (dLed)  dLed.className = detectorRunning ? 'status-led active' : 'status-led offline';
    }, 6000);
}

function handleStatusUpdate(status) {
    if (!status) return;

    if (status.fall) {
        const fLed = el('fallLed');
        const fTxt = el('fallStatus');
        if (fLed) fLed.className = status.fall.active ? 'status-led active' : 'status-led idle';
        if (fTxt) fTxt.textContent = status.fall.active ? 'Active' : 'Inactive';
        if (typeof status.fall.total === 'number') {
            stats.fall.count = status.fall.total;
            const fc = el('fallCount');
            if (fc) fc.textContent = stats.fall.count;
        }
        if (status.fall.confidence) {
            const fp = el('fallProgress');
            if (fp) fp.style.width = `${status.fall.confidence * 100}%`;
            const cb = el('confidenceBar');
            if (cb) cb.style.width = `${status.fall.confidence * 100}%`;
            const cv = el('confidenceValue');
            if (cv) cv.textContent = `${Math.round(status.fall.confidence * 100)}%`;
        }
    }

    if (status.voice) {
        const vLed = el('voiceLed');
        const vTxt = el('voiceStatus');
        if (vLed) vLed.className = status.voice.active ? 'status-led active' : 'status-led idle';
        if (vTxt) vTxt.textContent = status.voice.active ? 'Active' : 'Inactive';
        if (typeof status.voice.total === 'number') {
            stats.voice.count = status.voice.total;
            const vc = el('voiceCount');
            if (vc) vc.textContent = stats.voice.count;
        }
    }

    if (status.currentState) {
        stats.state = status.currentState;
        const cs = el('currentState');
        if (cs) cs.textContent = `State: ${status.currentState}`;
    }

    if (typeof status.connected === 'boolean') {
        if (status.connected !== detectorRunning) {
            detectorRunning = status.connected;
            setDetectorUI(detectorRunning);
        }
    }

    stats.alerts = (status.fall?.total || 0) + (status.voice?.total || 0);
    const ac = el('alertCount');
    if (ac) ac.textContent = stats.alerts;
}

// ══════════════════════════════════════════════════════════════════════════
//  UI HELPERS
// ══════════════════════════════════════════════════════════════════════════
function setDetectorUI(running) {
    const startBtn = el('startDetectorBtn');
    const stopBtn  = el('stopDetectorBtn');
    const camLed   = el('cameraLed');
    const camTxt   = el('cameraStatusText');
    const noFeed   = el('noFeedMessage');
    const overlay  = el('detectionOverlay');
    const fallLed  = el('fallLed');
    const fallTxt  = el('fallStatus');
    const voiceLed = el('voiceLed');
    const voiceTxt = el('voiceStatus');
    const detTxt   = el('detectionText');

    if (startBtn) startBtn.disabled = running;
    if (stopBtn)  stopBtn.disabled  = !running;

    if (camLed)   camLed.className  = running ? 'status-led active' : 'status-led offline';
    if (camTxt)   camTxt.textContent  = running ? 'Camera Active'   : 'Camera Offline';
    if (noFeed)   noFeed.style.display  = running ? 'none' : 'flex';
    if (overlay)  overlay.style.display = running ? 'block' : 'none';

    if (!running) {
        if (fallLed)  fallLed.className  = 'status-led idle';
        if (fallTxt)  fallTxt.textContent  = 'Inactive';
        if (voiceLed) voiceLed.className = 'status-led idle';
        if (voiceTxt) voiceTxt.textContent = 'Inactive';
        if (detTxt)   detTxt.textContent   = 'OFFLINE';
    } else {
        if (detTxt)   detTxt.textContent   = 'MONITORING';
    }
}

function updateFallUI(confidence) {
    const fc = el('fallCount');
    const ft = el('fallLastTime');
    const fcf = el('fallConfidence');
    const fp  = el('fallProgress');
    const cb  = el('confidenceBar');
    const cv  = el('confidenceValue');

    if (fc)  fc.textContent  = stats.fall.count;
    if (ft)  ft.textContent  = `Last: ${stats.fall.lastTime?.toLocaleTimeString() || '-'}`;
    if (fcf) fcf.textContent = `Confidence: ${Math.round(confidence * 100)}%`;
    if (fp)  fp.style.width  = `${confidence * 100}%`;
    if (cb)  cb.style.width  = `${confidence * 100}%`;
    if (cv)  cv.textContent  = `${Math.round(confidence * 100)}%`;
}

function updateVoiceUI(keywords) {
    const vc  = el('voiceCount');
    const vt  = el('voiceLastTime');
    const vk  = el('voiceKeywords');
    const lh  = el('lastHeard');

    if (vc) vc.textContent = stats.voice.count;
    if (vt) vt.textContent = `Last: ${stats.voice.lastTime?.toLocaleTimeString() || '-'}`;
    if (keywords.length) {
        if (vk) vk.textContent = `Keywords: ${keywords.join(', ')}`;
        if (lh) lh.textContent = keywords.join(', ');
    }
}

function updateAlertUI() {
    const ac = el('alertCount');
    const cs = el('currentState');
    const la = el('lastAlertTime');
    if (ac) ac.textContent = stats.alerts;
    if (cs) cs.textContent = `State: ${stats.state}`;
    const lastTime = stats.fall.lastTime || stats.voice.lastTime;
    if (la && lastTime) la.textContent = `Last: ${lastTime.toLocaleTimeString()}`;
}

function updateAllCounters() {
    const fields = {
        fallCount:    stats.fall.count,
        voiceCount:   stats.voice.count,
        alertCount:   stats.alerts,
        fallLastTime: 'Last: -',
        fallConfidence: 'Confidence: -',
        voiceLastTime: 'Last: -',
        voiceKeywords: 'Keywords: -',
        currentState:  'State: MONITORING',
        lastAlertTime: 'Last: -'
    };
    Object.entries(fields).forEach(([id, val]) => {
        const e = el(id); if (e) e.textContent = val;
    });
    ['fallProgress','confidenceBar'].forEach(id => {
        const e = el(id); if (e) e.style.width = '0%';
    });
    const cv = el('confidenceValue'); if (cv) cv.textContent = '0%';
}

function flashCard(id) {
    const card = el(id);
    if (!card) return;
    card.style.animation = 'pulse 0.8s ease';
    card.classList.add('emergency');
    setTimeout(() => {
        card.style.animation = '';
        card.classList.remove('emergency');
    }, 2000);
}

// ── Emergency popup ───────────────────────────────────────────────────────
function triggerEmergency(title, message, type) {
    const popup  = el('emergencyPopup');
    const etitle = el('emergencyTitle');
    const emsg   = el('emergencyMessage');
    const etime  = el('emergencyTime');

    if (etitle) etitle.textContent = title;
    if (emsg)   emsg.textContent   = message;
    if (etime)  etime.textContent  = new Date().toLocaleString();

    if (popup) {
        popup.style.borderColor = type === 'fall' ? '#ef4444' : '#10b981';
        popup.classList.add('show');
        setTimeout(() => popup.classList.remove('show'), 10000);
    }
}

function acknowledgeEmergency() {
    const popup = el('emergencyPopup');
    if (popup) popup.classList.remove('show');
    addLog('System', 'Emergency acknowledged by caretaker');
}

// ── Alarm (Web Audio API — no external files needed) ─────────────────────
function playAlarm(type) {
    try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const pattern = type === 'fall'
            ? [[1000,150],[700,150],[1000,150],[700,150],[1400,400]]
            : [[600,150],[900,150],[1200,150],[900,150],[1200,300]];
        let delay = 0;
        pattern.forEach(([freq, dur]) => {
            const osc  = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.frequency.value = freq;
            gain.gain.value     = 0.35;
            osc.start(ctx.currentTime + delay / 1000);
            osc.stop(ctx.currentTime  + delay / 1000 + dur / 1000);
            delay += dur + 40;
        });
    } catch(e) {
        const a = el('alarmSound');
        if (a) a.play().catch(() => {});
    }
}

// ── Detection log ─────────────────────────────────────────────────────────
function addLog(type, message, confidence, keywords) {
    const container = el('logContainer');
    if (!container) return;

    const entry = document.createElement('div');
    entry.className = `log-entry ${type === 'System' ? 'system' : ''}`;

    const time    = new Date().toLocaleTimeString();
    const confTxt = (confidence && confidence > 0)
        ? ` <strong>${Math.round(confidence*100)}%</strong>` : '';
    const kwTxt   = (keywords && keywords.length)
        ? ` <span class="log-keywords">${keywords.join(', ')}</span>` : '';

    entry.innerHTML = `
        <span class="log-time">${time}</span>
        <span class="log-type ${type.toLowerCase()}">${type}</span>
        <span class="log-message">${message}${confTxt}${kwTxt}</span>
    `;

    container.insertBefore(entry, container.firstChild);
    while (container.children.length > 80) container.removeChild(container.lastChild);
}

function clearLog() {
    const c = el('logContainer');
    if (c) c.innerHTML = '<div class="log-entry system"><span class="log-time">System</span><span class="log-message">Log cleared</span></div>';
}

function exportLog() {
    const lines = [];
    el('logContainer')?.querySelectorAll('.log-entry').forEach(e => {
        lines.push([
            e.querySelector('.log-time')?.textContent,
            e.querySelector('.log-type')?.textContent,
            e.querySelector('.log-message')?.textContent
        ].filter(Boolean).join(' | '));
    });
    const blob = new Blob([lines.join('\n')], { type: 'text/plain' });
    const a    = document.createElement('a');
    a.href     = URL.createObjectURL(blob);
    a.download = `guardian_log_${new Date().toISOString().slice(0,10)}.txt`;
    a.click();
    URL.revokeObjectURL(a.href);
    addLog('System', 'Log exported');
}

// ── Detections table ──────────────────────────────────────────────────────
function addToDetectionsTable(type, message, confidence, keywords) {
    const tbody = el('detectionsList');
    if (!tbody) return;
    if (tbody.children.length === 1 && tbody.children[0].querySelector('[colspan]'))
        tbody.innerHTML = '';

    const row = tbody.insertRow(0);
    row.innerHTML = `
        <td>${new Date().toLocaleString()}</td>
        <td><span class="badge badge-${type}">${type}</span></td>
        <td>${message}</td>
        <td>${confidence ? Math.round(confidence*100)+'%' : '-'}</td>
        <td>${keywords?.length ? keywords.join(', ') : '-'}</td>
    `;
    while (tbody.children.length > 25) tbody.deleteRow(-1);
}

// ── Loading overlay ───────────────────────────────────────────────────────
function showLoading(msg) {
    const ov = el('loadingOverlay');
    const lm = el('loadingMessage');
    if (lm) lm.textContent = msg || 'Loading…';
    if (ov) ov.classList.add('show');
}
function hideLoading() {
    el('loadingOverlay')?.classList.remove('show');
}

// ── Uptime timer ──────────────────────────────────────────────────────────
function startUptimeTimer() {
    stopUptimeTimer();
    uptimeTimer = setInterval(() => {
        if (!detectorRunning) return;
        const s = Math.floor((Date.now() - startTime) / 1000);
        const h = Math.floor(s / 3600);
        const m = Math.floor((s % 3600) / 60);
        const sec = s % 60;
        const ut = el('uptime');
        if (ut) ut.textContent = `${h}h ${m}m ${sec}s`;
    }, 1000);
}
function stopUptimeTimer() {
    if (uptimeTimer) { clearInterval(uptimeTimer); uptimeTimer = null; }
    const ut = el('uptime'); if (ut) ut.textContent = '0s';
}

// ══════════════════════════════════════════════════════════════════════════
//  INITIALISATION
// ══════════════════════════════════════════════════════════════════════════

// Load patient list into the dropdown
fetch('/api/patients', { headers: { 'Authorization': `Bearer ${token}` } })
    .then(r => r.json())
    .then(patients => {
        const sel = el('patientSelect');
        if (sel && Array.isArray(patients) && patients.length) {
            sel.innerHTML = patients.map(p =>
                `<option value="${p.id}">${p.full_name} (Room ${p.room_number || 'N/A'})</option>`
            ).join('');
        }
    })
    .catch(() => {});

// Load current detector status
fetch('/api/detector/status')
    .then(r => r.json())
    .then(data => {
        handleStatusUpdate(data);
        detectorRunning = data.connected || false;
        setDetectorUI(detectorRunning);
        if (detectorRunning) startUptimeTimer();
    })
    .catch(() => {});

// Connect WebSocket after short delay so page renders first
setTimeout(connectWebSocket, 400);

// Re-connect when tab becomes visible again
document.addEventListener('visibilitychange', () => {
    if (!document.hidden && (!ws || ws.readyState !== WebSocket.OPEN))
        connectWebSocket();
});

window.addEventListener('beforeunload', () => ws?.close());