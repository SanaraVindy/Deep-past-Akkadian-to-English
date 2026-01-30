import streamlit as st
import os
import re
import sqlite3
import hashlib
import kagglehub
import base64
import smtplib
import random
import string
from email.mime.text import MIMEText
from transformers import AutoTokenizer, T5ForConditionalGeneration

# --- 1. DATABASE & SECURITY ---
def init_db():
    conn = sqlite3.connect('archives_access.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT, email TEXT)''')
    c.execute("PRAGMA table_info(users)")
    columns = [column[1] for column in c.fetchall()]
    if 'email' not in columns:
        c.execute("ALTER TABLE users ADD COLUMN email TEXT")
    conn.commit()
    conn.close()

def hash_pw(password):
    return hashlib.sha256(password.encode()).hexdigest()

def send_recovery_email(receiver_email, new_password):
    sender_email = "thevindimuhandiramge@gmail.com" 
    sender_password = "kdchtfxjnpqutmuj" 
    msg = MIMEText(f"Your temporary access key: {new_password}")
    msg['Subject'] = 'Deep Past Recovery'
    msg['From'] = sender_email
    msg['To'] = receiver_email
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        return True
    except Exception as e: return e

init_db()

# --- 2. MINIMALIST THEME ENGINE ---
st.set_page_config(page_title="Deep Past Initiative", page_icon="📜", layout="wide")

def set_professional_theme():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(script_dir, 'image_182e7c.jpg') 
    
    bg_css = "background-color: #1a1a1a;"
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        bg_css = f'background-image: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), url("data:image/jpg;base64,{b64}");'

    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Source+Code+Pro&display=swap');
        .stApp {{ {bg_css} background-size: cover; background-attachment: fixed; }}
        
        .auth-box {{
            background: transparent !important;
            padding: 0px; 
            border: none !important;
            margin: auto;
            max-width: 600px;
            text-align: center;
        }}

        .tablet-box {{
            background-color: rgba(30, 30, 30, 0.85);
            border: 1px solid #d4af37;
            border-radius: 12px;
            padding: 25px;
            color: #d4af37 !important;
            min-height: 400px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .stTextArea textarea {{
            background-color: rgba(10, 10, 10, 0.8) !important;
            color: #d4af37 !important;
            border: 1px solid #d4af37 !important;
        }}

        .stButton>button {{
            background-color: #d4af37 !important;
            color: black !important;
            font-weight: bold;
            width: 100%;
        }}
        
        .stTabs [data-baseweb="tab-list"] {{ justify-content: center; background: transparent; }}
        .stTabs [data-baseweb="tab"] {{ color: #d4af37 !important; font-weight: bold !important; }}
        
        .about-text {{
            color: #bfa57d;
            text-align: justify;
            font-size: 1rem;
            line-height: 1.6;
        }}
        </style>
        """, unsafe_allow_html=True)

set_professional_theme()

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_byt5():
    try:
        os.environ['KAGGLE_USERNAME'] = "thevindimuhandiramge" 
        os.environ['KAGGLE_KEY'] = "KGAT_2296bcbd3286473e9ae39386af74560e" 
        path = kagglehub.model_download("thevindimuhandiramge/akkadian-byt5-final/pytorch/default")
        
        model_path = path
        for root, dirs, files in os.walk(path):
            if "config.json" in files:
                model_path = root
                break
                
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        return tokenizer, model
    except Exception as e:
        st.error(f"Inference Engine Error: {e}")
        return None, None

# --- 4. AUTHENTICATION LOGIC ---
if "auth" not in st.session_state:
    st.markdown('<br><br><br><h1 style="text-align:center; color:#d4af37; font-size:3.5rem; font-family:Playfair Display;">WELCOME TO THE DEEP PAST</h1>', unsafe_allow_html=True)
    _, col_mid, _ = st.columns([1, 2, 1])
    with col_mid:
        st.markdown('<div class="auth-box">', unsafe_allow_html=True)
        t1, t2, t3, t4 = st.tabs(["Login", "Register", "Forgot Password", "About"])
        
        with t1:
            u = st.text_input("User ID", key="l_u")
            p = st.text_input("Access Key", type="password", key="l_p")
            if st.button("Unlock Archives", key="btn_login"):
                conn = sqlite3.connect('archives_access.db')
                c = conn.cursor()
                c.execute("SELECT * FROM users WHERE username=? AND password=?", (u, hash_pw(p)))
                if c.fetchone():
                    st.session_state["auth"] = True
                    st.session_state["user"] = u
                    st.rerun()
                else: st.error("Invalid Credentials")
                conn.close()
        
        with t2:
            nu = st.text_input("New Username", key="r_u")
            ne = st.text_input("User Email", key="r_e")
            np = st.text_input("New Password", type="password", key="r_p")
            if st.button("Create Account", key="btn_reg"):
                if nu and np and ne:
                    try:
                        conn = sqlite3.connect('archives_access.db')
                        c = conn.cursor()
                        c.execute("INSERT INTO users (username, password, email) VALUES (?,?,?)", (nu, hash_pw(np), ne))
                        conn.commit()
                        st.success("User Registered! Please Login.")
                        conn.close()
                    except: st.error("Username Taken.")
                else: st.warning("All fields required.")
        
        with t3:
            ru = st.text_input("User ID for recovery", key="f_u")
            if st.button("Recover Access"):
                conn = sqlite3.connect('archives_access.db')
                c = conn.cursor()
                c.execute("SELECT email FROM users WHERE username=?", (ru,))
                res = c.fetchone()
                if res:
                    temp_pw = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
                    if send_recovery_email(res[0], temp_pw) == True:
                        c.execute("UPDATE users SET password=? WHERE username=?", (hash_pw(temp_pw), ru))
                        conn.commit()
                        st.success(f"Recovery email sent.")
                    else: st.error("Email failed.")
                else: st.error("Not found.")
                conn.close()

        with t4:
            st.markdown("""
            <div class="about-text">
            <h3 style='color:#d4af37;'>The Deep Past Initiative: Neural Decipherment</h3>
            <p>The <b>Deep Past Initiative</b> is a specialized computational tool designed for the reconstruction of the <b>Old Assyrian</b> dialect.</p>
            <h4 style='color:#d4af37; font-size:1.1rem;'>Technical Framework</h4>
            <p>Utilizes <b>ByT5</b> Byte-level Transformers to handle noisy transliterations without fixed vocabularies.</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- 5. WORKSPACE (LOGGED IN) ---
else:
    tokenizer, model = load_byt5()
    st.markdown('<h1 style="text-align:center; color:#d4af37; font-size:3rem;">OLD ASSYRIAN TRANSLATOR</h1>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(f"### User: {st.session_state['user']}")
        with st.expander("ℹ️ About System"):
            st.info("**Version:** 1.0.4\\n**Model:** ByT5-Base\\n**Project:** CIS 6005")

        with st.expander("🔑 Change Access Key"):
            old_p = st.text_input("Current Key", type="password")
            new_p = st.text_input("New Key", type="password")
            if st.button("Update Key"):
                conn = sqlite3.connect('archives_access.db')
                c = conn.cursor()
                c.execute("SELECT * FROM users WHERE username=? AND password=?", (st.session_state["user"], hash_pw(old_p)))
                if c.fetchone():
                    c.execute("UPDATE users SET password=? WHERE username=?", (hash_pw(new_p), st.session_state["user"]))
                    conn.commit()
                    st.success("Access Key Updated!")
                else: st.error("Incorrect Current Key")
                conn.close()
        
        st.write("---")
        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()

    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        st.markdown("### 📜 Tablet Transliteration")
        src_text = st.text_area("Input transliteration...", height=400, key="main_input")
        if st.button("⚜️ GENERATE"):
            if src_text.strip() and model:
                with st.spinner("Decoding Ancient Bytes..."):
                    prompt = "translate Old Assyrian to English: " + src_text
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
                    outputs = model.generate(inputs["input_ids"], num_beams=12, max_length=150)
                    res = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    st.session_state['res'] = re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', res)
                    st.session_state['last_in'] = src_text
                    st.rerun() # Critical: Forces refresh to show output

    with col_r:
        st.markdown("### 🏛️ Deciphered English")
        if 'res' in st.session_state:
            st.markdown(f'<div class="tablet-box">{st.session_state["res"]}</div>', unsafe_allow_html=True)
            st.download_button("💾 DOWNLOAD", f"Input: {st.session_state.get('last_in')}\\nOutput: {st.session_state['res']}", "decipherment.txt")
        else:
            st.markdown('<div class="tablet-box" style="color:#666;">Awaiting input...</div>', unsafe_allow_html=True)

    st.markdown("<br><hr><p style='text-align:center; color:#888;'>© 2026 Deep Past Initiative | Developed for CIS 6005 project by Thevindi Muhandiramge</p>", unsafe_allow_html=True)