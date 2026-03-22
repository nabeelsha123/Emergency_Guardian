const express = require('express');
const cors = require('cors');
const path = require('path');
const db = require('./database');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const http = require('http');
const WebSocket = require('ws');
const { spawn } = require('child_process');
const fs = require('fs');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

const PORT = 3000;
const JWT_SECRET = 'guardian_net_secret_key_2024';

// ── Connected WebSocket clients ────────────────────────────────────────────────
const clients = new Set();
let detectorProcess = null;

// ── Live detector status (sent to dashboard via WebSocket) ─────────────────────
let detectorStatus = {
    fall:    { active: false, total: 0, lastAlert: null, confidence: 0 },
    voice:   { active: false, total: 0, lastAlert: null, keywords: [] },
    gesture: { active: false, total: 0, lastAlert: null, gestureType: '' },
    connected: false,
    currentState: 'MONITORING'
};

// ══════════════════════════════════════════════════════════════════════════════
//  WEBSOCKET
// ══════════════════════════════════════════════════════════════════════════════
wss.on('connection', (ws) => {
    console.log('🟢 WebSocket client connected');
    clients.add(ws);

    // Send current status immediately on connect
    ws.send(JSON.stringify({ type: 'status', data: detectorStatus }));

    ws.on('message', (message) => {
        try {
            const data = JSON.parse(message);
            if (data.type === 'command') handleCommand(data.command, data.payload || {});
        } catch (error) {
            console.error('WebSocket message error:', error);
        }
    });

    ws.on('close', () => {
        console.log('🔴 WebSocket client disconnected');
        clients.delete(ws);
    });
});

function broadcastToAll(message) {
    const str = JSON.stringify(message);
    clients.forEach(client => {
        if (client.readyState === WebSocket.OPEN) client.send(str);
    });
}

function handleCommand(command, payload) {
    if      (command === 'start_detector')  startPythonDetector(payload?.patient_id || 1);
    else if (command === 'stop_detector')   stopPythonDetector();
    else if (command === 'reset_counts') {
        detectorStatus.fall.total    = 0;
        detectorStatus.voice.total   = 0;
        detectorStatus.gesture.total = 0;
        broadcastToAll({ type: 'status', data: detectorStatus });
    }
}

// ══════════════════════════════════════════════════════════════════════════════
//  DETECTOR ALERT ENDPOINT  ← called by guardian_integration.py
// ══════════════════════════════════════════════════════════════════════════════
app.post('/api/detector/alert', express.json(), (req, res) => {
    const { patient_id, alert_type, message, confidence, keywords, timestamp } = req.body;

    console.log(`\n📨 Alert received  type=${alert_type}  conf=${confidence}  patient=${patient_id}`);

    const now = new Date().toISOString();

    // ── Update in-memory status ────────────────────────────────────────────
    if (alert_type === 'fall') {
        detectorStatus.fall.total++;
        detectorStatus.fall.lastAlert   = now;
        detectorStatus.fall.active      = true;
        detectorStatus.fall.confidence  = confidence || 0;
        detectorStatus.currentState     = 'FALL_DETECTED';
        console.log(`🚨 FALL  confidence=${(confidence*100).toFixed(1)}%`);

    } else if (alert_type === 'voice') {
        detectorStatus.voice.total++;
        detectorStatus.voice.lastAlert  = now;
        detectorStatus.voice.active     = true;
        detectorStatus.voice.keywords   = keywords || [];
        detectorStatus.currentState     = 'VOICE_DETECTED';
        console.log(`🔊 VOICE  keywords=${(keywords||[]).join(', ')}`);

    } else if (alert_type === 'gesture') {
        detectorStatus.gesture.total++;
        detectorStatus.gesture.lastAlert     = now;
        detectorStatus.gesture.active        = true;
        detectorStatus.gesture.gestureType   = message || '';
        detectorStatus.currentState          = 'GESTURE_DETECTED';
        console.log(`🤚 GESTURE  ${message}`);

    } else {
        console.warn(`⚠️  Unknown alert_type: ${alert_type}`);
    }

    // ── Persist to DB ──────────────────────────────────────────────────────
    db.run(
        `INSERT INTO detection_events (patient_id, event_type, confidence, details)
         VALUES (?, ?, ?, ?)`,
        [patient_id || 1, alert_type, confidence || 0.0, message || ''],
        err => { if (err) console.error('DB detection_events error:', err); }
    );

    db.run(
        `INSERT INTO emergency_alerts (patient_id, alert_type, message, confidence, status)
         VALUES (?, ?, ?, ?, ?)`,
        [patient_id || 1, alert_type,
         message || `${alert_type} detected`,
         confidence || 0.0, 'pending'],
        err => { if (err) console.error('DB emergency_alerts error:', err); }
    );

    // ── Push to all dashboard clients ──────────────────────────────────────
    broadcastToAll({
        type:        'detection',
        event_type:  alert_type,
        confidence:  confidence,
        details:     message,
        keywords:    keywords || [],
        patient_id:  patient_id || 1,
        timestamp:   timestamp || now,
    });

    broadcastToAll({ type: 'status', data: detectorStatus });

    res.json({ success: true, message: 'Alert received' });
});

// ── State-change update (MONITORING → FALL_DETECTED etc.) ─────────────────────
app.post('/api/detector/status-update', express.json(), (req, res) => {
    const { state, fall_active, voice_active } = req.body;

    if (state)                        detectorStatus.currentState   = state;
    if (fall_active  !== undefined)   detectorStatus.fall.active    = fall_active;
    if (voice_active !== undefined)   detectorStatus.voice.active   = voice_active;

    // Auto-clear active flags when back to MONITORING
    if (state === 'MONITORING') {
        detectorStatus.fall.active    = false;
        detectorStatus.voice.active   = false;
        detectorStatus.gesture.active = false;
    }

    broadcastToAll({ type: 'status', data: detectorStatus });
    res.json({ success: true });
});

// ── Simple status ping (used by GuardianAlertSender.test_connection) ───────────
app.get('/api/detector/status', (req, res) => {
    res.json(detectorStatus);
});

// ══════════════════════════════════════════════════════════════════════════════
//  MIDDLEWARE
// ══════════════════════════════════════════════════════════════════════════════
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, '../public')));
app.use('/css', express.static(path.join(__dirname, '../public/css')));
app.use('/js',  express.static(path.join(__dirname, '../public/js')));

// ══════════════════════════════════════════════════════════════════════════════
//  AUTH MIDDLEWARE
// ══════════════════════════════════════════════════════════════════════════════
const authenticateToken = (req, res, next) => {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];
    if (!token) return res.status(401).json({ error: 'Access denied' });

    jwt.verify(token, JWT_SECRET, (err, user) => {
        if (err) return res.status(403).json({ error: 'Invalid token' });
        req.user = user;
        next();
    });
};

// ══════════════════════════════════════════════════════════════════════════════
//  AUTH ROUTES
// ══════════════════════════════════════════════════════════════════════════════
app.post('/api/register', async (req, res) => {
    const { username, password, full_name, email, phone } = req.body;
    if (!username || !password || !full_name || !email)
        return res.status(400).json({ error: 'All fields are required' });

    try {
        const hashedPassword = await bcrypt.hash(password, 10);
        db.run(
            'INSERT INTO caretakers (username, password, full_name, email, phone) VALUES (?, ?, ?, ?, ?)',
            [username, hashedPassword, full_name, email, phone],
            function(err) {
                if (err) {
                    if (err.message.includes('UNIQUE'))
                        return res.status(400).json({ error: 'Username or email already exists' });
                    return res.status(500).json({ error: err.message });
                }
                const token = jwt.sign(
                    { id: this.lastID, username, full_name }, JWT_SECRET, { expiresIn: '24h' });
                res.json({ message: 'Registration successful', token,
                           user: { id: this.lastID, username, full_name, email, phone } });
            });
    } catch (error) { res.status(500).json({ error: error.message }); }
});

app.post('/api/login', (req, res) => {
    const { username, password } = req.body;
    db.get('SELECT * FROM caretakers WHERE username = ?', [username], async (err, user) => {
        if (err || !user) return res.status(401).json({ error: 'Invalid credentials' });
        const validPassword = await bcrypt.compare(password, user.password);
        if (!validPassword) return res.status(401).json({ error: 'Invalid credentials' });

        const token = jwt.sign(
            { id: user.id, username: user.username, full_name: user.full_name },
            JWT_SECRET, { expiresIn: '24h' });
        res.json({ token, user: {
            id: user.id, username: user.username,
            full_name: user.full_name, email: user.email, phone: user.phone
        }});
    });
});

// ══════════════════════════════════════════════════════════════════════════════
//  PATIENT ROUTES
// ══════════════════════════════════════════════════════════════════════════════
app.get('/api/patients', authenticateToken, (req, res) => {
    db.all('SELECT * FROM patients WHERE caretaker_id = ? AND is_active = 1 ORDER BY created_at DESC',
        [req.user.id], (err, rows) => {
            if (err) return res.status(500).json({ error: err.message });
            res.json(rows);
        });
});

app.get('/api/patients/:id', authenticateToken, (req, res) => {
    db.get('SELECT * FROM patients WHERE id = ? AND caretaker_id = ?',
        [req.params.id, req.user.id], (err, row) => {
            if (err) return res.status(500).json({ error: err.message });
            if (!row) return res.status(404).json({ error: 'Patient not found' });
            res.json(row);
        });
});

app.post('/api/patients', authenticateToken, (req, res) => {
    const { full_name, age, gender, medical_conditions,
            emergency_contact_name, emergency_contact_phone,
            emergency_contact_relation, room_number } = req.body;

    if (!full_name || !emergency_contact_name || !emergency_contact_phone)
        return res.status(400).json({ error: 'Required fields missing' });

    db.run(`INSERT INTO patients
        (caretaker_id, full_name, age, gender, medical_conditions,
         emergency_contact_name, emergency_contact_phone, emergency_contact_relation, room_number)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        [req.user.id, full_name, age, gender, medical_conditions,
         emergency_contact_name, emergency_contact_phone, emergency_contact_relation, room_number],
        function(err) {
            if (err) return res.status(500).json({ error: err.message });
            res.json({ id: this.lastID, message: 'Patient added successfully' });
        });
});

app.put('/api/patients/:id', authenticateToken, (req, res) => {
    const { full_name, age, gender, medical_conditions,
            emergency_contact_name, emergency_contact_phone,
            emergency_contact_relation, room_number } = req.body;

    db.run(`UPDATE patients SET
        full_name=?, age=?, gender=?, medical_conditions=?,
        emergency_contact_name=?, emergency_contact_phone=?,
        emergency_contact_relation=?, room_number=?
        WHERE id=? AND caretaker_id=?`,
        [full_name, age, gender, medical_conditions,
         emergency_contact_name, emergency_contact_phone,
         emergency_contact_relation, room_number,
         req.params.id, req.user.id],
        function(err) {
            if (err) return res.status(500).json({ error: err.message });
            if (this.changes === 0) return res.status(404).json({ error: 'Patient not found' });
            res.json({ message: 'Patient updated successfully' });
        });
});

app.delete('/api/patients/:id', authenticateToken, (req, res) => {
    db.run('UPDATE patients SET is_active=0 WHERE id=? AND caretaker_id=?',
        [req.params.id, req.user.id],
        function(err) {
            if (err) return res.status(500).json({ error: err.message });
            if (this.changes === 0) return res.status(404).json({ error: 'Patient not found' });
            res.json({ message: 'Patient deleted successfully' });
        });
});

// ══════════════════════════════════════════════════════════════════════════════
//  ALERTS & STATS
// ══════════════════════════════════════════════════════════════════════════════
app.get('/api/alerts', authenticateToken, (req, res) => {
    db.all(`SELECT a.*, p.full_name as patient_name, p.room_number
            FROM emergency_alerts a
            JOIN patients p ON a.patient_id = p.id
            WHERE p.caretaker_id = ?
            ORDER BY a.created_at DESC LIMIT 50`,
        [req.user.id], (err, rows) => {
            if (err) return res.status(500).json({ error: err.message });
            res.json(rows);
        });
});

app.get('/api/alerts/recent', authenticateToken, (req, res) => {
    db.all(`SELECT a.*, p.full_name as patient_name, p.room_number
            FROM emergency_alerts a
            JOIN patients p ON a.patient_id = p.id
            WHERE p.caretaker_id = ? AND a.created_at > datetime('now', '-1 hour')
            ORDER BY a.created_at DESC`,
        [req.user.id], (err, rows) => {
            if (err) return res.status(500).json({ error: err.message });
            res.json(rows);
        });
});

app.post('/api/alerts/:id/resolve', authenticateToken, (req, res) => {
    db.run('UPDATE emergency_alerts SET status=?, resolved_at=CURRENT_TIMESTAMP, resolved_by=? WHERE id=?',
        ['resolved', req.user.id, req.params.id],
        function(err) {
            if (err) return res.status(500).json({ error: err.message });
            res.json({ message: 'Alert resolved' });
        });
});

app.get('/api/detections', authenticateToken, (req, res) => {
    db.all(`SELECT d.*, p.full_name as patient_name
            FROM detection_events d
            JOIN patients p ON d.patient_id = p.id
            WHERE p.caretaker_id = ?
            ORDER BY d.created_at DESC LIMIT 50`,
        [req.user.id], (err, rows) => {
            if (err) return res.status(500).json({ error: err.message });
            res.json(rows);
        });
});

app.get('/api/stats', authenticateToken, (req, res) => {
    let stats = {};
    db.get('SELECT COUNT(*) as total FROM patients WHERE caretaker_id=? AND is_active=1',
        [req.user.id], (err, patientRow) => {
            stats.totalPatients = patientRow?.total || 0;
            db.get(`SELECT COUNT(*) as total FROM emergency_alerts a
                    JOIN patients p ON a.patient_id = p.id WHERE p.caretaker_id=?`,
                [req.user.id], (err, alertRow) => {
                    stats.totalAlerts = alertRow?.total || 0;
                    db.get(`SELECT COUNT(*) as total FROM emergency_alerts a
                            JOIN patients p ON a.patient_id = p.id
                            WHERE p.caretaker_id=? AND date(a.created_at)=date('now')`,
                        [req.user.id], (err, todayRow) => {
                            stats.todayAlerts = todayRow?.total || 0;
                            db.get(`SELECT COUNT(*) as total FROM detection_events d
                                    JOIN patients p ON d.patient_id = p.id
                                    WHERE p.caretaker_id=? AND event_type='fall'`,
                                [req.user.id], (err, fallRow) => {
                                    stats.totalFalls = fallRow?.total || 0;
                                    db.get(`SELECT COUNT(*) as total FROM detection_events d
                                            JOIN patients p ON d.patient_id = p.id
                                            WHERE p.caretaker_id=? AND event_type='voice'`,
                                        [req.user.id], (err, voiceRow) => {
                                            stats.totalVoice = voiceRow?.total || 0;
                                            db.get(`SELECT COUNT(*) as total FROM detection_events d
                                                    JOIN patients p ON d.patient_id = p.id
                                                    WHERE p.caretaker_id=? AND event_type='gesture'`,
                                                [req.user.id], (err, gestureRow) => {
                                                    stats.totalGesture = gestureRow?.total || 0;
                                                    db.get(`SELECT COUNT(*) as total FROM emergency_alerts a
                                                            JOIN patients p ON a.patient_id = p.id
                                                            WHERE p.caretaker_id=? AND a.status='pending'`,
                                                        [req.user.id], (err, pendingRow) => {
                                                            stats.pendingAlerts = pendingRow?.total || 0;
                                                            res.json(stats);
                                                        });
                                                });
                                        });
                                });
                        });
                });
        });
});

// ══════════════════════════════════════════════════════════════════════════════
//  PYTHON DETECTOR PROCESS CONTROL
// ══════════════════════════════════════════════════════════════════════════════
function startPythonDetector(patientId = 1) {
    if (detectorProcess) {
        console.log('⚠️  Detector already running');
        broadcastToAll({ type: 'status', data: detectorStatus });
        return;
    }

    // Search order: guardian_all.py → guardian_net_full.py → integrated_detector.py
    const candidates = [
        path.join(__dirname, '../detector', 'guardian_all.py'),
        path.join(__dirname, '../detector', 'guardian_net_full.py'),
        path.join(__dirname, '../detector', 'integrated_detector.py'),
    ];

    const detectorPath = candidates.find(p => fs.existsSync(p));
    if (!detectorPath) {
        console.error('❌ No detector Python file found. Searched:');
        candidates.forEach(p => console.error('   ✗', p));
        broadcastToAll({ type: 'error', message: 'Detector file not found on server' });
        broadcastToAll({ type: 'detector_log', text: '❌ detector file not found' });
        return;
    }

    console.log(`\n🐍 Starting: ${path.basename(detectorPath)}  patient_id=${patientId}`);
    broadcastToAll({ type: 'detector_log', text: `Starting detector for patient ${patientId}...` });

    detectorProcess = spawn('python', [detectorPath, '--patient_id', String(patientId)], {
        cwd:   path.dirname(detectorPath),
        stdio: 'pipe',
        env:   { ...process.env, PYTHONUNBUFFERED: '1' }
    });

    detectorProcess.stdout.on('data', (data) => {
        const lines = data.toString().split('\n');
        lines.forEach(line => {
            line = line.trim();
            if (!line) return;
            console.log(`📹 Detector: ${line}`);

            // Forward all detector output to dashboard log panel
            broadcastToAll({ type: 'detector_log', text: line });

            // Parse specific status lines to update live indicators
            if (line.includes('Listening for voice') || line.includes('Voice Detection Ready')) {
                detectorStatus.voice.active = true;
                broadcastToAll({ type: 'status', data: detectorStatus });
            }
            if (line.includes('Fall camera') || line.includes('Fall Detection Ready')) {
                detectorStatus.fall.active = true;
                broadcastToAll({ type: 'status', data: detectorStatus });
            }
            if (line.includes('ALL DETECTORS READY') || line.includes('DETECTORS READY')) {
                detectorStatus.fall.active  = true;
                detectorStatus.voice.active = true;
                detectorStatus.connected    = true;
                broadcastToAll({ type: 'status', data: detectorStatus });
                broadcastToAll({ type: 'detector_ready' });
            }
            if (line.includes('Mic test PASSED') || line.includes('Microphone calibrated')) {
                broadcastToAll({ type: 'detector_log', text: '✅ Microphone ready' });
            }
            if (line.includes('Custom model') || line.includes('POSE')) {
                broadcastToAll({ type: 'detector_log', text: line });
            }
        });
    });

    detectorProcess.stderr.on('data', (data) => {
        // Python prints many non-error things to stderr (e.g. model loading progress)
        // Only surface lines that look like actual errors
        const lines = data.toString().split('\n');
        lines.forEach(line => {
            line = line.trim();
            if (!line) return;
            // Suppress common noisy non-errors from ultralytics/torch/mediapipe
            const ignore = ['UserWarning', 'FutureWarning', 'DeprecationWarning',
                            'YOLO', 'Ultralytics', 'torch', 'tensorflow',
                            'WARNING', 'INFO', 'DEBUG'];
            if (!ignore.some(k => line.includes(k))) {
                console.error(`⚠️  Detector: ${line}`);
            }
        });
    });

    detectorProcess.on('close', (code) => {
        console.log(`\n🔴 Detector exited (code ${code})`);
        detectorProcess = null;
        detectorStatus.connected      = false;
        detectorStatus.fall.active    = false;
        detectorStatus.voice.active   = false;
        detectorStatus.gesture.active = false;
        detectorStatus.currentState   = 'MONITORING';
        broadcastToAll({ type: 'status', data: detectorStatus });
        broadcastToAll({ type: 'detector_stopped', code });
        broadcastToAll({ type: 'detector_log', text: `Detector stopped (exit code ${code})` });
    });

    detectorStatus.connected = true;
    broadcastToAll({ type: 'status', data: detectorStatus });
}

function stopPythonDetector() {
    if (detectorProcess) {
        detectorProcess.kill();
        detectorProcess = null;
    }
    detectorStatus.connected      = false;
    detectorStatus.fall.active    = false;
    detectorStatus.voice.active   = false;
    detectorStatus.gesture.active = false;
    detectorStatus.currentState   = 'MONITORING';
    broadcastToAll({ type: 'status', data: detectorStatus });
    console.log('🛑 Detector stopped');
}

app.post('/api/detector/start', express.json(), (req, res) => {
    const patientId = req.body?.patient_id || 1;
    startPythonDetector(patientId);
    res.json({ success: true, message: `Detector starting for patient ${patientId}` });
});

app.post('/api/detector/stop', (req, res) => {
    stopPythonDetector();
    res.json({ success: true, message: 'Detector stopped' });
});

// ══════════════════════════════════════════════════════════════════════════════
//  SERVE FRONTEND
// ══════════════════════════════════════════════════════════════════════════════
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '../public', 'index.html'));
});

app.get('/:page', (req, res) => {
    const filePath = path.join(__dirname, '../public', req.params.page);
    if (fs.existsSync(filePath) && filePath.endsWith('.html')) {
        res.sendFile(filePath);
    } else {
        res.sendFile(path.join(__dirname, '../public', 'index.html'));
    }
});

// ══════════════════════════════════════════════════════════════════════════════
//  START
// ══════════════════════════════════════════════════════════════════════════════
server.listen(PORT, () => {
    console.log('\n' + '='.repeat(70));
    console.log('🚀 GUARDIAN NET SERVER STARTED');
    console.log('='.repeat(70));
    console.log(`📱 Website  : http://localhost:${PORT}`);
    console.log(`🔑 Login    : admin / admin123`);
    console.log(`🔌 WebSocket: ws://localhost:${PORT}`);
    console.log(`📡 Alert API: POST http://localhost:${PORT}/api/detector/alert`);
    console.log('='.repeat(70) + '\n');
});