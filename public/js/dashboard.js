// Dashboard functionality
const token = localStorage.getItem('token');
if (!token) window.location.href = '/login.html';

const user = JSON.parse(localStorage.getItem('user') || '{}');
document.getElementById('userName').textContent = user.full_name || 'User';

// Logout handler
document.getElementById('logoutBtn').addEventListener('click', (e) => {
    e.preventDefault();
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    window.location.href = '/login.html';
});

// Load dashboard stats
async function loadStats() {
    try {
        const response = await fetch('/api/stats', {
            headers: { 'Authorization': `Bearer ${token}` }
        });
        const stats = await response.json();
        
        document.getElementById('totalPatients').textContent = stats.totalPatients || 0;
        document.getElementById('totalFalls').textContent = stats.totalFalls || 0;
        document.getElementById('totalVoice').textContent = stats.totalVoice || 0;
        document.getElementById('todayAlerts').textContent = stats.todayAlerts || 0;
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

// Load recent patients
async function loadRecentPatients() {
    try {
        const response = await fetch('/api/patients', {
            headers: { 'Authorization': `Bearer ${token}` }
        });
        const patients = await response.json();
        
        const tbody = document.getElementById('patientsList');
        if (patients.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="text-center">No patients added</td></tr>';
            return;
        }
        
        tbody.innerHTML = patients.slice(0, 5).map(patient => `
            <tr>
                <td>${patient.full_name}</td>
                <td>${patient.age || '-'}</td>
                <td>${patient.emergency_contact_name || '-'}</td>
                <td>${patient.room_number || '-'}</td>
                <td>
                    <button class="btn-icon" onclick="viewPatient(${patient.id})" title="View">
                        <i class="bi bi-eye"></i>
                    </button>
                </td>
            </tr>
        `).join('');
    } catch (error) {
        console.error('Error loading patients:', error);
    }
}

// Load recent alerts
async function loadRecentAlerts() {
    try {
        const response = await fetch('/api/alerts', {
            headers: { 'Authorization': `Bearer ${token}` }
        });
        const alerts = await response.json();
        
        const tbody = document.getElementById('alertsList');
        if (alerts.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="text-center">No alerts</td></tr>';
            return;
        }
        
        tbody.innerHTML = alerts.slice(0, 5).map(alert => `
            <tr>
                <td>${new Date(alert.created_at).toLocaleString()}</td>
                <td>${alert.patient_name}</td>
                <td><span class="badge badge-${alert.alert_type}">${alert.alert_type}</span></td>
                <td>${alert.message}</td>
                <td><span class="badge badge-${alert.status}">${alert.status}</span></td>
            </tr>
        `).join('');
    } catch (error) {
        console.error('Error loading alerts:', error);
    }
}

function viewPatient(id) {
    window.location.href = `/patients.html?view=${id}`;
}

// Auto refresh every 30 seconds
loadStats();
loadRecentPatients();
loadRecentAlerts();
setInterval(loadStats, 30000);
setInterval(loadRecentAlerts, 30000);