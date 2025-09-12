# -*- coding: utf-8 -*-
"""WaterGuard Full E-Learning App with Modules, Quizzes, Rewards, Badges & Certificate"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
import random
import base64
from streamlit.components.v1 import html
import plotly.express as px

# ----------------------------
# CONFIG
# ----------------------------
sns.set_style("whitegrid")
st.set_page_config(page_title="WaterGuard", layout="wide")

# ----------------------------
# SESSION STATE INIT
# ----------------------------
if "lang" not in st.session_state: st.session_state.lang = "en"
if "current_module" not in st.session_state: st.session_state.current_module = 0
if "quiz_scores" not in st.session_state: st.session_state.quiz_scores = {}
if "rewards" not in st.session_state: st.session_state.rewards = 0
if "modules_completed" not in st.session_state: st.session_state.modules_completed = []

# ----------------------------
# LANGUAGE TOGGLE
# ----------------------------
st.sidebar.title("Settings" if st.session_state.lang == "en" else "الإعدادات")
lang_choice = st.sidebar.radio("🌐 Language / اللغة", ["English", "العربية"])
st.session_state.lang = "ar" if lang_choice == "العربية" else "en"
lang = st.session_state.lang

# ----------------------------
# SCREEN READER BUTTON
# ----------------------------
def screen_reader_button(lang_local):
    lang_code = "en-US" if lang_local == "en" else "ar-SA"
    html(f"""
    <button onclick="setTimeout(readPage, 500);" style="
    background-color:#023e8a;
    color:white;
    border:none;
    padding:10px 20px;
    border-radius:10px;
    cursor:pointer;
    font-size:1rem;
    margin-top: 1rem;
    display: block;
    {'margin-left: auto;' if lang_local == 'en' else 'margin-right: auto;'}
    ">
    🔊 {'Activate Screen Reader' if lang_local == 'en' else 'تشغيل قارئ الشاشة'}
    </button>
    <script>
    function readPage() {{
        const synth = window.speechSynthesis;
        if (synth.speaking) {{ synth.cancel(); }}
        const app = document.querySelector('.main') || document.querySelector('.stApp');
        let text = '';
        if (app) {{
            const walker = document.createTreeWalker(app, NodeFilter.SHOW_TEXT, null, false);
            let node;
            while(node = walker.nextNode()) {{
                if(node.textContent.trim() !== '') {{ text += node.textContent.trim() + '. '; }}
            }}
        }} else {{ text = "Content not found."; }}
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = '{lang_code}';
        synth.speak(utterance);
    }}
    </script>
    """, height=80)

with st.sidebar:
    screen_reader_button(lang)

# ----------------------------
# BACKGROUND IMAGE
# ----------------------------
def set_background(image_path):
    try:
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
                position: relative;
                color: #f0f0f0;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                min-height: 100vh;
            }}
            .stApp::before {{
                content: "";
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background: rgba(0, 0, 0, 0.45);
                z-index: -1;
            }}
            [data-testid="stSidebar"] {{
                background: rgba(255, 255, 255, 0.95);
                border-radius: 0 15px 15px 0;
                padding: 1rem 1.5rem;
                box-shadow: 2px 0 12px rgba(0, 0, 0, 0.1);
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    except FileNotFoundError:
        pass

set_background("water_bg.jpg")

# ----------------------------
# COURSE MODULES
# ----------------------------
REWARD_FILS_PER_QUIZ = 500
REWARD_BHD_PER_QUIZ = REWARD_FILS_PER_QUIZ / 1000.0

COURSE = [
    {
        "title_en":"Intro: Why Water Monitoring Matters (5 min)",
        "title_ar":"مقدمة: لماذا تهم مراقبة المياه (5 دقائق)",
        "minutes":5,
        "content_en":"Why household water monitoring is important: cost savings, leak prevention, and sustainability...",
        "content_ar":"لماذا تُعد مراقبة المياه المنزلية مهمة: توفير التكاليف، منع التسرب، والاستدامة...",
        "quiz":[
            {
                "q_en":"Which is a direct benefit of early leak detection?",
                "q_ar":"ما هي فائدة الكشف المبكر عن التسرب؟",
                "options":["Higher bills","Increased water waste","Lower repair costs","More humid air"],
                "answer":2
            }
        ]
    },
    {
        "title_en":"How WaterGuard Detects Anomalies (8 min)",
        "title_ar":"كيف يكتشف ووتر جارد الأنماط الشاذة (8 دقائق)",
        "minutes":8,
        "content_en":"Overview of sensors, hourly data, anomaly detection models (e.g., IsolationForest)...",
        "content_ar":"نظرة عامة على الحساسات، البيانات الساعية، نماذج اكتشاف الخلل (مثل IsolationForest)...",
        "quiz":[
            {
                "q_en":"Which model is used in this prototype for anomaly detection?",
                "q_ar":"أي نموذج تم استخدامه في هذا النموذج لاكتشاف الخلل؟",
                "options":["KMeans","IsolationForest","Linear Regression","PCA"],
                "answer":1
            },
            {
                "q_en":"A severity labeled 'High' likely indicates:",
                "q_ar":"ماذا تعني شدة 'عالية' عادةً؟",
                "options":["Very low usage","Normal usage","Very high usage","No data"],
                "answer":2
            }
        ]
    }
]

BAHRAIN_HISTORY_EN = "Bahrain's relationship with water is ancient and multifaceted..."
BAHRAIN_HISTORY_AR = "لطالما كانت علاقة البحرين بالمياه قديمة ومعقّدة..."

# ----------------------------
# WATER DATA SIMULATION
# ----------------------------
@st.cache_data
def simulate_data():
    np.random.seed(42)
    hours = 365*24
    date_range = pd.date_range(start='2024-01-01', periods=hours, freq='H')
    usage_main = np.random.normal(12,3,hours).clip(0,50)
    usage_garden = np.random.normal(5,2,hours).clip(0,20)
    usage_kitchen = np.random.normal(3,1,hours).clip(0,10)
    usage_bathroom = np.random.normal(4,1.5,hours).clip(0,15)
    df_local = pd.DataFrame({
        'timestamp':date_range,
        'usage_main_liters':usage_main,
        'usage_garden_liters':usage_garden,
        'usage_kitchen_liters':usage_kitchen,
        'usage_bathroom_liters':usage_bathroom,
    })
    df_local['usage_liters'] = df_local[['usage_main_liters','usage_garden_liters','usage_kitchen_liters','usage_bathroom_liters']].sum(axis=1)
    df_local['date'] = df_local['timestamp'].dt.date
    num_anomalies = int(0.05*len(df_local))
    anomaly_indices = random.sample(range(len(df_local)), num_anomalies)
    for i in anomaly_indices:
        df_local.loc[i,['usage_main_liters','usage_garden_liters','usage_kitchen_liters','usage_bathroom_liters']] *= np.random.uniform(2,5)
    df_local['usage_liters'] = df_local[['usage_main_liters','usage_garden_liters','usage_kitchen_liters','usage_bathroom_liters']].sum(axis=1)
    return df_local

df = simulate_data()

# ----------------------------
# ANOMALY DETECTION
# ----------------------------
model = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = model.fit_predict(df[['usage_liters']])
df['anomaly'] = df['anomaly'].map({1:'Normal', -1:'Anomaly'})
df['severity'] = pd.cut(df['usage_liters'], bins=[-np.inf,20,40,np.inf], labels=['Low','Medium','High'])

# ----------------------------
# BADGES AND CERTIFICATE FUNCTIONS
# ----------------------------
def check_module_completion(module_idx):
    completed = True
    for i, q in enumerate(COURSE[module_idx]['quiz']):
        key = f"{module_idx}_{i}"
        if key not in st.session_state.quiz_scores or not st.session_state.quiz_scores[key]:
            completed = False
    if completed and module_idx not in st.session_state.modules_completed:
        st.session_state.modules_completed.append(module_idx)
    return completed

def show_certificate():
    st.balloons()
    st.success("🎓 Congratulations! You completed all modules!" if lang=="en" else "🎓 تهانينا! لقد أكملت جميع الوحدات!")
    st.markdown(f"""
    <div style='border:5px solid #023e8a; padding:30px; border-radius:20px; text-align:center; background:#f0f0f0;'>
        <h2>{'WaterGuard Certificate of Completion' if lang=='en' else 'شهادة إتمام ووتر جارد'}</h2>
        <p>{'This certifies that the learner has successfully completed all WaterGuard modules.' if lang=='en' else 'تشهد هذه الشهادة أن المتعلم قد أكمل جميع وحدات ووتر جارد بنجاح.'}</p>
        <p>💰 {st.session_state.rewards:.3f} BHD earned!</p>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# MAIN APP
# ----------------------------
st.title("WaterGuard" if lang=="en" else "ووتر جارد")

tab1, tab2, tab3 = st.tabs(["Dashboard","Course","Bahrain History"] if lang=="en" else ["لوحة البيانات","الدورة","تاريخ البحرين"])

with tab1:
    st.subheader("Water Usage Overview" if lang=="en" else "نظرة عامة على استهلاك المياه")
    fig = px.line(df.sample(2000), x='timestamp', y='usage_liters', color='anomaly', title="Hourly Usage & Anomalies" if lang=="en" else "الاستهلاك الساعي والأنماط الشاذة")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    module_idx = st.session_state.current_module
    module = COURSE[module_idx]
    st.header(module['title_en'] if lang=="en" else module['title_ar'])
    st.write(module['content_en'] if lang=="en" else module['content_ar'])

    st.subheader("Quiz" if lang=="en" else "الاختبار")
    for i, q in enumerate(module['quiz']):
        key = f"{module_idx}_{i}"
        ans = st.radio(q['q_en'] if lang=="en" else q['q_ar'], q['options'], key=key)
        if st.button("Submit Answer" if lang=="en" else "إرسال الإجابة", key=f"submit_{key}"):
            correct = q['options'][q['answer']]
            if ans == correct:
                st.success("Correct! 🎉" if lang=="en" else "صحيح! 🎉")
                st.session_state.rewards += REWARD_BHD_PER_QUIZ
                st.session_state.quiz_scores[key] = True
            else:
                st.error("Incorrect! ❌" if lang=="en" else "خاطئ! ❌")
                st.session_state.quiz_scores[key] = False

    completed = check_module_completion(module_idx)
    if completed:
        st.success("✅ Module Completed!" if lang=="en" else "✅ تم إكمال الوحدة!")

    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous Module" if lang=="en" else "الوحدة السابقة"):
            if st.session_state.current_module > 0:
                st.session_state.current_module -= 1
    with col2:
        if st.button("Next Module" if lang=="en" else "الوحدة التالية"):
            if st.session_state.current_module < len(COURSE)-1:
                st.session_state.current_module += 1

    # Show progress
    progress = len(st.session_state.modules_completed)/len(COURSE)*100
    st.info(f"📊 Progress: {progress:.0f}%" if lang=="en" else f"📊 التقدم: {progress:.0f}%")
    st.info(f"💰 Rewards: {st.session_state.rewards:.3f} BHD" if lang=="en" else f"💰 رصيدك: {st.session_state.rewards:.3f} دينار")

    if len(st.session_state.modules_completed) == len(COURSE):
        show_certificate()

with tab3:
    st.header("Bahrain Water History" if lang=="en" else "تاريخ المياه في البحرين")
    st.write(BAHRAIN_HISTORY_EN if lang=="en" else BAHRAIN_HISTORY_AR)
# ----------------------------
# PDF CERTIFICATE GENERATION
# ----------------------------
from fpdf import FPDF
import io

def generate_pdf_certificate(learner_name, rewards, lang="en"):
    pdf = FPDF('L','mm','A4')
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Background color
    pdf.set_fill_color(2, 62, 138)
    pdf.rect(0,0,297,210, 'F')
    
    # Title
    pdf.set_font("Arial", 'B', 36)
    pdf.set_text_color(255,255,255)
    pdf.ln(40)
    pdf.cell(0,10, "WaterGuard Certificate of Completion" if lang=="en" else "شهادة إتمام ووتر جارد", ln=True, align='C')
    
    # Spacer
    pdf.ln(20)
    
    # Body text
    pdf.set_font("Arial", '', 24)
    pdf.set_text_color(255,255,255)
    text = f"This certifies that {learner_name}" if lang=="en" else f"تشهد هذه الشهادة أن {learner_name}"
    pdf.cell(0,10,text, ln=True, align='C')
    
    pdf.ln(10)
    text2 = "has successfully completed all WaterGuard modules." if lang=="en" else "قد أكمل جميع وحدات ووتر جارد بنجاح."
    pdf.cell(0,10,text2, ln=True, align='C')
    
    pdf.ln(15)
    pdf.set_font("Arial", 'B', 20)
    reward_text = f"💰 Rewards Earned: {rewards:.3f} BHD" if lang=="en" else f"💰 الرصيد المكتسب: {rewards:.3f} دينار"
    pdf.cell(0,10,reward_text, ln=True, align='C')
    
    pdf.ln(20)
    pdf.set_font("Arial", 'I', 16)
    pdf.cell(0,10,"WaterGuard Training Team" if lang=="en" else "فريق تدريب ووتر جارد", ln=True, align='C')
    
    # Save PDF to bytes
    pdf_bytes = io.BytesIO()
    pdf.output(pdf_bytes)
    pdf_bytes.seek(0)
    return pdf_bytes

# ----------------------------
# SHOW CERTIFICATE & DOWNLOAD
# ----------------------------
def show_certificate():
    st.balloons()
    st.success("🎓 Congratulations! You completed all modules!" if lang=="en" else "🎓 تهانينا! لقد أكملت جميع الوحدات!")
    st.markdown(f"""
    <div style='border:5px solid #023e8a; padding:30px; border-radius:20px; text-align:center; background:#f0f0f0;'>
        <h2>{'WaterGuard Certificate of Completion' if lang=='en' else 'شهادة إتمام ووتر جارد'}</h2>
        <p>{'This certifies that the learner has successfully completed all WaterGuard modules.' if lang=='en' else 'تشهد هذه الشهادة أن المتعلم قد أكمل جميع وحدات ووتر جارد بنجاح.'}</p>
        <p>💰 {st.session_state.rewards:.3f} BHD earned!</p>
    </div>
    """, unsafe_allow_html=True)
    
    learner_name = st.text_input("Enter your name for the certificate:" if lang=="en" else "أدخل اسمك للشهادة")
    if learner_name:
        pdf_bytes = generate_pdf_certificate(learner_name, st.session_state.rewards, lang)
        st.download_button(
            label="Download Certificate as PDF" if lang=="en" else "تحميل الشهادة كملف PDF",
            data=pdf_bytes,
            file_name=f"WaterGuard_Certificate_{learner_name}.pdf",
            mime="application/pdf"
        )
