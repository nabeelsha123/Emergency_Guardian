const sqlite3 = require('sqlite3').verbose();
const path = require('path');
const bcrypt = require('bcryptjs');
const fs = require('fs');

// Ensure data directory exists
const dataDir = path.join(__dirname, 'data');
if (!fs.existsSync(dataDir)) {
    fs.mkdirSync(dataDir, { recursive: true });
}

const dbPath = path.join(dataDir, 'guardian_net.db');
const db = new sqlite3.Database(dbPath);

// Initialize database
db.serialize(() => {
    // Caretakers table
    db.run(`CREATE TABLE IF NOT EXISTS caretakers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        full_name TEXT,
        email TEXT UNIQUE,
        phone TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )`);

    // Patients table
    db.run(`CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        caretaker_id INTEGER,
        full_name TEXT,
        age INTEGER,
        gender TEXT,
        medical_conditions TEXT,
        emergency_contact_name TEXT,
        emergency_contact_phone TEXT,
        emergency_contact_relation TEXT,
        room_number TEXT,
        is_active INTEGER DEFAULT 1,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (caretaker_id) REFERENCES caretakers(id)
    )`);

    // Emergency alerts table
    db.run(`CREATE TABLE IF NOT EXISTS emergency_alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        alert_type TEXT,
        message TEXT,
        confidence REAL,
        status TEXT DEFAULT 'pending',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        resolved_at DATETIME,
        resolved_by INTEGER,
        FOREIGN KEY (patient_id) REFERENCES patients(id),
        FOREIGN KEY (resolved_by) REFERENCES caretakers(id)
    )`);

    // Detection events table
    db.run(`CREATE TABLE IF NOT EXISTS detection_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        event_type TEXT,
        confidence REAL,
        details TEXT,
        audio_file TEXT,
        video_file TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (patient_id) REFERENCES patients(id)
    )`);

    // Create default admin if not exists
    db.get('SELECT * FROM caretakers WHERE username = ?', ['admin'], (err, row) => {
        if (!row) {
            const hashedPassword = bcrypt.hashSync('admin123', 10);
            db.run('INSERT INTO caretakers (username, password, full_name, email) VALUES (?, ?, ?, ?)',
                ['admin', hashedPassword, 'System Administrator', 'admin@guardian.net']);
            console.log('✅ Default admin created');
        }
    });

    // Create sample patient if none exists
    db.get('SELECT COUNT(*) as count FROM patients', (err, row) => {
        if (row && row.count === 0) {
            db.get('SELECT id FROM caretakers WHERE username = ?', ['admin'], (err, caretaker) => {
                if (caretaker) {
                    db.run(`INSERT INTO patients 
                        (caretaker_id, full_name, age, emergency_contact_name, emergency_contact_phone, room_number) 
                        VALUES (?, ?, ?, ?, ?, ?)`,
                        [caretaker.id, 'John Doe', 75, 'Jane Doe', '+1234567890', '101']);
                    console.log('✅ Sample patient created');
                }
            });
        }
    });
});

console.log('✅ Database initialized');
module.exports = db;