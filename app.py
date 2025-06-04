from flask import Flask, render_template, request, redirect, url_for, session, flash

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

# Simulated user database
users = {}

@app.route('/')
def overview():
    return render_template('overview.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # ✅ Check if password is at least 6 characters
        if len(password) < 6:
            flash("Password must be at least 6 characters long.")
            return redirect(url_for('register'))

        if username in users:
            flash("Username already exists! Please log in.")
            return redirect(url_for('overview'))  # Redirect back to Overview
        
        users[username] = password  # Save user
        session['registered'] = True  # Set session to indicate registration
        flash("Registration successful! Now click Login.")
        return redirect(url_for('overview'))  # Redirect back to Overview

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username not in users:
            flash("❌ User not found! Please register first.", "error")
            return redirect(url_for('login'))  # Redirect back to login

        if users[username] != password:
            flash("⚠️ Incorrect password. Try again.", "error")  
            return redirect(url_for('login'))  # Redirect back to login

        session['user'] = username  # Store session for logged-in user
        flash("✅ Login successful!", "success")

        return redirect(url_for('dashboard'))

    return render_template('login.html')


@app.route('/dashboard')
def dashboard():
    if 'user' not in session:  # Check if user is logged in
        flash("You must log in first!")
        return redirect(url_for('overview'))
    
    return render_template('dashboard.html')  # Assuming you have a 'dashboard.html' template

@app.route('/analysis')
def analysis():
    if 'user' not in session:  # Ensure user is logged in
        flash("You must log in first!")
        return redirect(url_for('overview'))

    return render_template('analysis.html')



if __name__ == '__main__':
    app.run(debug=True)
