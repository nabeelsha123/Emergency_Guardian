// Authentication handling for Login and Register pages

// Check if already logged in
const token = localStorage.getItem('token');
if (token && window.location.pathname.includes('login.html')) {
    window.location.href = '/dashboard.html';
}

// Login form handler
const loginForm = document.getElementById('loginForm');
if (loginForm) {
    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const username = document.getElementById('username').value;
        const password = document.getElementById('password').value;
        const errorDiv = document.getElementById('errorMessage');

        try {
            const response = await fetch('/api/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
            });

            const data = await response.json();

            if (response.ok) {
                localStorage.setItem('token', data.token);
                localStorage.setItem('user', JSON.stringify(data.user));
                window.location.href = '/dashboard.html';
            } else {
                errorDiv.textContent = data.error || 'Login failed';
                errorDiv.classList.add('show');
                setTimeout(() => errorDiv.classList.remove('show'), 3000);
            }
        } catch (error) {
            errorDiv.textContent = 'Connection error';
            errorDiv.classList.add('show');
            setTimeout(() => errorDiv.classList.remove('show'), 3000);
        }
    });
}

// Register form handler
const registerForm = document.getElementById('registerForm');
if (registerForm) {
    registerForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const full_name = document.getElementById('full_name').value;
        const username = document.getElementById('username').value;
        const email = document.getElementById('email').value;
        const phone = document.getElementById('phone').value;
        const password = document.getElementById('password').value;
        const confirm_password = document.getElementById('confirm_password').value;
        const errorDiv = document.getElementById('errorMessage');
        const successDiv = document.getElementById('successMessage');

        // Validation
        if (password !== confirm_password) {
            errorDiv.textContent = 'Passwords do not match';
            errorDiv.classList.add('show');
            setTimeout(() => errorDiv.classList.remove('show'), 3000);
            return;
        }

        if (password.length < 6) {
            errorDiv.textContent = 'Password must be at least 6 characters';
            errorDiv.classList.add('show');
            setTimeout(() => errorDiv.classList.remove('show'), 3000);
            return;
        }

        try {
            const response = await fetch('/api/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    full_name, 
                    username, 
                    email, 
                    phone, 
                    password 
                })
            });

            const data = await response.json();

            if (response.ok) {
                successDiv.textContent = 'Registration successful! Redirecting to login...';
                successDiv.classList.add('show');
                
                // Auto fill login form
                setTimeout(() => {
                    window.location.href = '/login.html';
                }, 2000);
            } else {
                errorDiv.textContent = data.error || 'Registration failed';
                errorDiv.classList.add('show');
                setTimeout(() => errorDiv.classList.remove('show'), 3000);
            }
        } catch (error) {
            errorDiv.textContent = 'Connection error';
            errorDiv.classList.add('show');
            setTimeout(() => errorDiv.classList.remove('show'), 3000);
        }
    });
}

// Password strength indicator (optional)
const passwordInput = document.getElementById('password');
if (passwordInput) {
    passwordInput.addEventListener('input', function() {
        const strength = checkPasswordStrength(this.value);
        // You can add visual indicator here if wanted
    });
}

function checkPasswordStrength(password) {
    let strength = 0;
    if (password.length >= 6) strength++;
    if (password.match(/[a-z]+/)) strength++;
    if (password.match(/[A-Z]+/)) strength++;
    if (password.match(/[0-9]+/)) strength++;
    if (password.match(/[$@#&!]+/)) strength++;
    return strength;
}