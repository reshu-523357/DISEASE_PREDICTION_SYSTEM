import streamlit as st
import sqlite3
from passlib.hash import pbkdf2_sha256
import subprocess

# Function to create a SQLite database and table for users
def create_user_table():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, password TEXT)''')
    conn.commit()
    conn.close()

# Function to insert a new user into the database
def insert_user(username, password):
    hashed_password = pbkdf2_sha256.hash(password)
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
    conn.commit()
    conn.close()

# Function to authenticate user
def authenticate_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username=?", (username,))
    stored_password = c.fetchone()
    conn.close()
    if stored_password:
        return pbkdf2_sha256.verify(password, stored_password[0])
    return False

# Function to display home page
def home():
    st.title("Home Page")
    st.image('medical.png', use_container_width=True)  # Replace 'it_medical.jpg' with your image file path
    st.write("""
        ### Welcome to Our Application
        Discovering early signs of diseases is crucial for effective treatment and management. Our Disease Prediction System leverages advanced machine learning algorithms to analyze symptoms and predict potential diseases with high accuracy. By inputting symptoms into our intuitive interface, users can receive rapid insights, enabling proactive healthcare decisions and improving patient outcomes. Whether you're a healthcare professional or an individual concerned about your health, our system provides a reliable tool for early detection and personalized healthcare planning. Join us in advancing healthcare through innovative technology and proactive disease management.
    """)

# Main function for login page
def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate_user(username, password):
            st.success("Login successful!")
            # Redirect to another Streamlit app file
            st.write("Redirecting to app.py...")
            subprocess.Popen(["streamlit", "run", "app.py"])
        else:
            st.error("Invalid username or password.")

# Main function for signup page
def signup():
    st.title("Signup")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    if st.button("Signup"):
        if new_password == confirm_password:
            insert_user(new_username, new_password)
            st.success("Signup successful! You can now login.")
        else:
            st.error("Passwords do not match.")

# Main application function
def main():
    # Create user table if not exists
    create_user_table()
    
    # Sidebar layout
    st.sidebar.image('DPS_icon.png', width=200)
    st.sidebar.markdown("---")  # Divider line
    
    # Navigation links with increased font size and styled as buttons
    st.sidebar.markdown("<h3 style='text-align: left;'>Navigation</h3>", unsafe_allow_html=True)
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "Home"
    
    # Handle navigation buttons
    if st.sidebar.button("Home", key="nav_home"):
        st.session_state.page = "Home"
    if st.sidebar.button("Signup", key="nav_signup"):
        st.session_state.page = "Signup"
    if st.sidebar.button("Login", key="nav_login"):
        st.session_state.page = "Login"
    
    # Display content based on selected page
    if st.session_state.page == "Home":
        home()
    elif st.session_state.page == "Signup":
        signup()
    elif st.session_state.page == "Login":
        login()

if __name__ == "__main__":
    main()

