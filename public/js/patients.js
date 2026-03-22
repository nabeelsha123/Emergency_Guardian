// Patients management functionality
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

// Load all patients
async function loadPatients() {
    try {
        const response = await fetch('/api/patients', {
            headers: { 'Authorization': `Bearer ${token}` }
        });
        const patients = await response.json();
        
        const tbody = document.getElementById('patientsList');
        if (patients.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="text-center">No patients added</td></tr>';
            return;
        }
        
        tbody.innerHTML = patients.map(patient => `
            <tr>
                <td>${patient.full_name}</td>
                <td>${patient.age || '-'}</td>
                <td>${patient.gender || '-'}</td>
                <td>${patient.emergency_contact_name || '-'}</td>
                <td>${patient.emergency_contact_phone || '-'}</td>
                <td>${patient.room_number || '-'}</td>
                <td class="actions">
                    <button class="btn-icon" onclick="editPatient(${patient.id})" title="Edit">
                        <i class="bi bi-pencil"></i>
                    </button>
                    <button class="btn-icon" onclick="deletePatient(${patient.id})" title="Delete">
                        <i class="bi bi-trash"></i>
                    </button>
                </td>
            </tr>
        `).join('');
    } catch (error) {
        console.error('Error loading patients:', error);
    }
}

// Search functionality
document.getElementById('searchPatient')?.addEventListener('input', (e) => {
    const term = e.target.value.toLowerCase();
    const rows = document.querySelectorAll('#patientsList tr');
    rows.forEach(row => {
        const text = row.textContent.toLowerCase();
        row.style.display = text.includes(term) ? '' : 'none';
    });
});

// Modal functions
function openAddModal() {
    document.getElementById('modalTitle').textContent = 'Add Patient';
    document.getElementById('patientForm').reset();
    document.getElementById('patientId').value = '';
    document.getElementById('patientModal').classList.add('show');
}

async function editPatient(id) {
    try {
        const response = await fetch(`/api/patients/${id}`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });
        const patient = await response.json();
        
        document.getElementById('modalTitle').textContent = 'Edit Patient';
        document.getElementById('patientId').value = patient.id;
        document.getElementById('fullName').value = patient.full_name || '';
        document.getElementById('age').value = patient.age || '';
        document.getElementById('gender').value = patient.gender || '';
        document.getElementById('medical_conditions').value = patient.medical_conditions || '';
        document.getElementById('emergencyName').value = patient.emergency_contact_name || '';
        document.getElementById('emergencyPhone').value = patient.emergency_contact_phone || '';
        document.getElementById('emergencyRelation').value = patient.emergency_contact_relation || '';
        document.getElementById('room_number').value = patient.room_number || '';
        
        document.getElementById('patientModal').classList.add('show');
    } catch (error) {
        console.error('Error loading patient:', error);
        alert('Failed to load patient details');
    }
}

async function savePatient() {
    const patientId = document.getElementById('patientId').value;
    const patientData = {
        full_name: document.getElementById('fullName').value,
        age: document.getElementById('age').value,
        gender: document.getElementById('gender').value,
        medical_conditions: document.getElementById('medical_conditions').value,
        emergency_contact_name: document.getElementById('emergencyName').value,
        emergency_contact_phone: document.getElementById('emergencyPhone').value,
        emergency_contact_relation: document.getElementById('emergencyRelation').value,
        room_number: document.getElementById('room_number').value
    };

    // Validate required fields
    if (!patientData.full_name || !patientData.emergency_contact_name || !patientData.emergency_contact_phone) {
        alert('Please fill in all required fields');
        return;
    }

    try {
        const url = patientId ? `/api/patients/${patientId}` : '/api/patients';
        const method = patientId ? 'PUT' : 'POST';
        
        const response = await fetch(url, {
            method: method,
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify(patientData)
        });
        
        if (response.ok) {
            closeModal();
            loadPatients();
        } else {
            const error = await response.json();
            alert(error.error || 'Failed to save patient');
        }
    } catch (error) {
        console.error('Error saving patient:', error);
        alert('Failed to save patient');
    }
}

async function deletePatient(id) {
    if (!confirm('Are you sure you want to delete this patient?')) return;
    
    try {
        const response = await fetch(`/api/patients/${id}`, {
            method: 'DELETE',
            headers: { 'Authorization': `Bearer ${token}` }
        });
        
        if (response.ok) {
            loadPatients();
        } else {
            alert('Failed to delete patient');
        }
    } catch (error) {
        console.error('Error deleting patient:', error);
        alert('Failed to delete patient');
    }
}

function closeModal() {
    document.getElementById('patientModal').classList.remove('show');
}

// Close modal when clicking outside
window.onclick = function(event) {
    const modal = document.getElementById('patientModal');
    if (event.target === modal) {
        closeModal();
    }
};

// Check for patient ID in URL (view mode)
const urlParams = new URLSearchParams(window.location.search);
const viewId = urlParams.get('view');
if (viewId) {
    // Highlight or show details of specific patient
    console.log('Viewing patient:', viewId);
}

// Load patients on page load
loadPatients();