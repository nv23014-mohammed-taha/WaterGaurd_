# -*- coding: utf-8 -*-
"""WaterGuard App - Full corrected version with Course, Rewards, and Bahrain History integrated"""

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
import time
import json

sns.set_style("whitegrid")
st.set_page_config(page_title="WaterGuard", layout="wide")

# ----------------------------
# Session state initial setup
# ----------------------------
if "lang" not in st.session_state:
st.session_state.lang = "en" # Default to English

# Course / reward session state
if "course_progress" not in st.session_state:
st.session_state.course_progress = 0
if "current_module" not in st.session_state:
st.session_state.current_module = 0
if "quiz_scores" not in st.session_state:
st.session_state.quiz_scores = {}
if "reward_claimed" not in st.session_state:
st.session_state.reward_claimed = {}
if "rewards" not in st.session_state:
st.session_state.rewards = 0
if "completed_quizzes" not in st.session_state:
st.session_state.completed_quizzes = []

# ----------------------------
# LANGUAGE TOGGLE (sidebar)
# ----------------------------
st.sidebar.title("Settings" if st.session_state.lang == "en" else "الإعدادات")
language_selection = st.sidebar.radio("🌐 Language / اللغة", ["English", "العربية"])
if language_selection == "العربية":
st.session_state.lang = "ar"
else:
st.session_state.lang = "en"

lang = st.session_state.lang # convenience variable

# ----------------------------
# SCREEN READER BUTTON (fixed)
# ----------------------------
def screen_reader_button(lang_local):
lang_code = "en-US" if lang_local == "en" else "ar-SA"
button_html = f"""
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
if (synth.speaking) {{
synth.cancel();
}}
const app = document.querySelector('.main') || document.querySelector('.stApp');
let text = '';
if (app) {{
const walker = document.createTreeWalker(app, NodeFilter.SHOW_TEXT, null, false);
let node;
while(node = walker.nextNode()) {{
if(node.textContent.trim() !== '') {{
text += node.textContent.trim() + '. ';
}}
}}
}} else {{
text = "Content not found.";
}}

const utterance = new SpeechSynthesisUtterance(text);
utterance.lang = '{lang_code}';
synth.speak(utterance);
}}
</script>
"""
html(button_html, height=80)

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
# If image not found, ignore and continue (keeps app usable)
pass

set_background("water_bg.jpg")

# ----------------------------
# COURSE & BAHRAIN HISTORY CONTENT (to be shown on top)
# ----------------------------
# Reward config
REWARD_FILS_PER_QUIZ = 500 # 500 fils
REWARD_BHD_PER_QUIZ = REWARD_FILS_PER_QUIZ / 1000.0

# Course definition (short modules totalling ~30 min)
COURSE = [
{
"title_en": "Intro: Why Water Monitoring Matters (5 min)",
"title_ar": "مقدمة: لماذا تهم مراقبة المياه (5 دقائق)",
"minutes": 5,
"content_en": (
"Why household water monitoring is important: cost savings, leak prevention, "
"and sustainability. How small behavioral changes lead to significant savings."
),
"content_ar": (
"لماذا تُعد مراقبة المياه المنزلية مهمة: توفير التكاليف، منع التسرب، "
"والاستدامة. كيف تؤدي التغييرات الصغيرة في السلوك إلى وفورات كبيرة."
),
"quiz": [
{
"q_en": "Which is a direct benefit of early leak detection?",
"q_ar": "ما هي فائدة الكشف المبكر عن التسرب؟",
"options": ["Higher bills", "Increased water waste", "Lower repair costs", "More humid air"],
"answer": 2
}
]
},
{
"title_en": "How WaterGuard Detects Anomalies (8 min)",
"title_ar": "كيف يكتشف ووتر جارد الأنماط الشاذة (8 دقائق)",
"minutes": 8,
"content_en": (
"Overview of sensors, hourly data, anomaly detection models (e.g., IsolationForest), "
"and how thresholds & severity are set."
),
"content_ar": (
"نظرة عامة على الحساسات، البيانات الساعية، نماذج اكتشاف الخلل (مثل IsolationForest)، "
"وكيف يتم ضبط العتبات وحدود الشدة."
),
"quiz": [
{
"q_en": "Which model is used in this prototype for anomaly detection?",
"q_ar": "أي نموذج تم استخدامه في هذا النموذج لاكتشاف الخلل؟",
"options": ["KMeans", "IsolationForest", "Linear Regression", "PCA"],
"answer": 1
},
{
"q_en": "A severity labeled 'High' likely indicates:",
"q_ar": "ماذا تعني شدة 'عالية' عادةً؟",
"options": ["Very low usage", "Normal usage", "Very high usage", "No data"],
"answer": 2
}
]
},
{
"title_en": "Practical Tips & Fixes (7 min)",
"title_ar": "نصائح عملية وإصلاحات (7 دقائق)",
"minutes": 7,
"content_en": (
"Simple checks: fixture inspections, irrigation schedules, fixture replacement recommendations, "
"and behavioral tips to minimize waste."
),
"content_ar": (
"فحوصات بسيطة: التحقق من التركيبات، جداول الري، توصيات استبدال التركيبات، "
"ونصائح سلوكية لتقليل الهدر."
),
"quiz": [
{
"q_en": "Which action helps most to reduce garden overwatering?",
"q_ar": "أي إجراء يساعد أكثر على تقليل الري الزائد للحديقة؟",
"options": ["Run sprinklers more often", "Shorten irrigation intervals", "Schedule irrigation early morning", "Water during hottest hour"],
"answer": 2
}
]
},
{
"title_en": "Reading Reports & Using Insights (5 min)",
"title_ar": "قراءة التقارير واستخدام الرؤى (5 دقائق)",
"minutes": 5,
"content_en": (
"How to read hourly/daily/monthly visualizations, export CSV, and act on detected trends."
),
"content_ar": (
"كيفية قراءة الرسوم البيانية الساعية/اليومية/الشهرية، تصدير CSV، واتخاذ إجراءات بناءً على الاتجاهات المكتشفة."
),
"quiz": [
{
"q_en": "If daily usage spikes repeatedly at night, what is the first thing to check?",
"q_ar": "إذا تكررت زيادات الاستهلاك اليومية ليلاً، ما هو أول شيء يجب التحقق منه؟",
"options": ["Kitchen sink", "Garden irrigation / sprinkler", "Cooking routines", "Battery level"],
"answer": 1
}
]
}
]

# Bahrain water history content (~300+ words)
BAHRAIN_HISTORY_EN = """
Bahrain's relationship with water is ancient and multifaceted. Historically, freshwater
in the archipelago was scarce; communities relied on shallow groundwater lenses, seasonal
wadis on the larger islands of the Gulf, and simple rain-capture techniques. Over centuries,
Bahrain's small area and limited freshwater resources shaped settlement patterns, agriculture,
and trade. Traditional systems—such as hand-dug wells and small networks for date palm
irrigation—were central to village life. During the mid-20th century, rising population and
urbanization placed heavier demands on limited groundwater reserves, and salinization from
over-pumping became an increasing concern.

By the later decades of the 20th century, Bahrain adopted large-scale technological
responses: desalination and modern water distribution infrastructure. Desalination plants
enabled urban growth and industrial development by providing a reliable supply of potable water.
However, desalination introduces challenges: energy intensity, brine disposal, and long-term
costs. Bahrain's small size means national strategies can be targeted and implemented quickly,
but must balance costs with sustainable resource use.

Looking forward, Bahrain's water future will be shaped by efficiency, diversification, and
technology. Water conservation programs, improvements in leak detection and metering—
exactly the benefits that WaterGuard targets—are critical. Investing in renewables to power
desalination or employing more energy-efficient desalination technologies can reduce the
environmental footprint. Treated wastewater reuse for irrigation and industry can lower
freshwater demand, while smart-city initiatives and advanced monitoring will help optimize
distribution networks. Climate change and regional groundwater pressures make integrated
water resource management essential; policies that combine demand reduction, reuse, and
innovative supply solutions will be decisive. Community engagement and household-level
solutions—such as smart leak detection, efficient appliances, and behavioral change—remain
among the most cost-effective and immediate measures to secure Bahrain's water resilience.
""".strip()

BAHRAIN_HISTORY_AR = """
لطالما كانت علاقة البحرين بالمياه قديمة ومعقّدة. تاريخيًا، كانت المياه العذبة نادرة في الأرخبيل؛
اعتمدت المجتمعات على عدسات المياه الجوفية الضحلة، وتقنيات تجميع المطر والآبار اليدوية لري النخيل.
مع تزايد السكان والتحضر في القرن العشرين، زادت الضغوط على احتياطات المياه الجوفية وظهرت مشكلات
تمليح المياه نتيجة الضخ الجائر.

خلال العقود الأخيرة، اعتمدت البحرين على تحلية المياه والبنية التحتية الحديثة لتوزيع المياه،
ما سهل النمو الحضري والصناعي. ومع ذلك، تُعدّ التحلية مكلفة وتحتاج طاقة كبيرة، كما تؤدي مخلفات
الملح إلى تحديات بيئية.

في المستقبل، يتعين على البحرين التركيز على الكفاءة والتنوّع والتقنيات الحديثة. تتضمن الحلول
تحسين الكشف عن التسريبات والقياس الدقيق للمستهلكين (مثل الحلول التي يقدمها ووتر جارد)،
إعادة استخدام المياه المعالجة للري والصناعة، واستخدام مصادر طاقة متجددة لتقليل بصمة التحلية.
مع تبعات تغير المناخ وضغوط المياه الإقليمية، يصبح التوازن بين تقليل الطلب وإدارة الموارد
بشكل متكامل أمرًا حاسمًا للحفاظ على الأمن المائي.
""".strip()

# ----------------------------
# Core app content (existing) - Data simulation + analysis
# ----------------------------
@st.cache_data
def simulate_data():
np.random.seed(42)
hours = 365 * 24
date_range = pd.date_range(start='2024-01-01', periods=hours, freq='H')

usage_main = np.random.normal(12, 3, hours).clip(0, 50)
usage_garden = np.random.normal(5, 2, hours).clip(0, 20)
usage_kitchen = np.random.normal(3, 1, hours).clip(0, 10)
usage_bathroom = np.random.normal(4, 1.5, hours).clip(0, 15)

df_local = pd.DataFrame({
'timestamp': date_range,
'usage_main_liters': usage_main,
'usage_garden_liters': usage_garden,
'usage_kitchen_liters': usage_kitchen,
'usage_bathroom_liters': usage_bathroom,
})

# Add total usage
df_local['usage_liters'] = df_local[[
'usage_main_liters', 'usage_garden_liters', 'usage_kitchen_liters', 'usage_bathroom_liters'
]].sum(axis=1)

df_local['date'] = df_local['timestamp'].dt.date

# Inject synthetic anomaly spikes (~5%)
num_anomalies = int(0.05 * len(df_local))
anomaly_indices = random.sample(range(len(df_local)), num_anomalies)
for i in anomaly_indices:
df_local.loc[i, ['usage_main_liters', 'usage_garden_liters', 'usage_kitchen_liters', 'usage_bathroom_liters']] *= np.random.uniform(2, 5)
# Recalculate total after anomalies
df_local['usage_liters'] = df_local[[
'usage_main_liters', 'usage_garden_liters', 'usage_kitchen_liters', 'usage_bathroom_liters'
]].sum(axis=1)

return df_local

df = simulate_data()

# Anomaly detection
model = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = model.fit_predict(df[['usage_liters']])
df['anomaly'] = df['anomaly'].map({1: 'Normal', -1: 'Anomaly'})

# Severity
df['severity'] = pd.cut(df['usage_liters'],
bins=[-np.inf, 20, 40, np.inf],
labels=['Low', 'Medium', 'High'])

# ----------------------------
# Top tabs: Course, Bahrain History, Dashboard
# ----------------------------
top_tabs = st.tabs([
"Course" if lang == "en" else "الدورة التدريبية",
"Bahrain Water" if lang == "en" else "تاريخ المياه في البحرين",
"Dashboard" if lang == "en" else "لوحة التحكم"
])

# ----------------------------
# Course Tab
# ----------------------------
with top_tabs[0]:
st.header("💡 WaterGuard — 30 Minute Course" if lang == "en" else "💡 ووتر جارد — دورة 30 دقيقة")
# Progress indicator
progress_fraction = st.session_state.course_progress / len(COURSE) if len(COURSE) > 0 else 0
st.progress(min(max(progress_fraction, 0.0), 1.0))

# Display modules list
st.markdown("### Modules" if lang == "en" else "### الوحدات")
module_titles = [(m["title_en"] if lang == "en" else m["title_ar"]) for m in COURSE]
for idx, t in enumerate(module_titles):
status = ""
if idx < st.session_state.course_progress:
status = "✅ Completed" if lang == "en" else "✅ مكتملة"
elif idx == st.session_state.current_module:
status = "▶ Current" if lang == "en" else "▶ الحالية"
st.write(f"{idx+1}. {t} {status}")

module_idx = st.session_state.current_module
module = COURSE[module_idx]
st.subheader(module["title_en"] if lang == "en" else module["title_ar"])
st.write(module["content_en"] if lang == "en" else module["content_ar"])
st.write(f"*Estimated time: {module['minutes']} min*" if lang == "en" else f"*الوقت المقدر: {module['minutes']} دقيقة*")

# Mark module complete button (progress only)
if st.button("Mark module complete" if lang == "en" else "تحديد الوحدة كمكتملة"):
st.session_state.course_progress = max(st.session_state.course_progress, module_idx + 1)
st.success("Module marked complete." if lang == "en" else "تم تحديد الوحدة كمكتملة.")
st.rerun()


# Quiz UI for current module
if module.get("quiz"):
st.markdown("### Quiz" if lang == "en" else "### الاختبار")
answers = {}
for qi, q in enumerate(module["quiz"]):
question_text = q["q_en"] if lang == "en" else q["q_ar"]
opts = q["options"]
# Use unique key per question
choice = st.radio(f"{qi+1}. {question_text}", opts, key=f"quiz_{module_idx}_{qi}")
answers[qi] = opts.index(choice)

if st.button("Submit Quiz" if lang == "en" else "إرسال الاختبار"):
# Grade
total = len(module["quiz"])
correct = 0
for i_q, q_def in enumerate(module["quiz"]):
if answers.get(i_q) == q_def["answer"]:
correct += 1
score_pct = (correct / total) * 100 if total > 0 else 0
st.session_state.quiz_scores[module_idx] = {"correct": correct, "total": total, "pct": score_pct}
passed = score_pct >= 80 # pass threshold 80%
if passed:
st.success((f"Passed — Score: {score_pct:.0f}% — Reward earned: {REWARD_FILS_PER_QUIZ} fils (BHD {REWARD_BHD_PER_QUIZ:.3f})") if lang == "en" else (f"ناجح — النسبة: {score_pct:.0f}% — جائزة: {REWARD_FILS_PER_QUIZ} فلس (ب.د {REWARD_BHD_PER_QUIZ:.3f})"))
# Give reward if not already claimed for this module
quiz_name = f"module_{module_idx}"
if quiz_name not in st.session_state.completed_quizzes:
st.session_state.completed_quizzes.append(quiz_name)
st.session_state.rewards += REWARD_FILS_PER_QUIZ
st.session_state.reward_claimed[module_idx] = True
else:
st.info("Reward already claimed for this quiz." if lang == "en" else "تم المطالبة بالجائزة لهذا الاختبار بالفعل.")
else:
st.warning((f"Not passed — Score: {score_pct:.0f}%. Try again!" if lang == "en" else f"لم تجتز — النسبة: {score_pct:.0f}%. حاول مرة أخرى!"))
st.experimental_rerun()

# Navigation buttons
col1, col2 = st.columns(2)
with col1:
if st.button("Previous module" if lang == "en" else "الوحدة السابقة"):
if st.session_state.current_module > 0:
st.session_state.current_module -= 1
st.experimental_rerun()
with col2:
if st.button("Next module" if lang == "en" else "الوحدة التالية"):
if st.session_state.current_module < len(COURSE) - 1:
st.session_state.current_module += 1
st.experimental_rerun()

# Certificate + rewards summary when complete
if st.session_state.course_progress >= len(COURSE):
st.balloons()
st.success("Course complete!" if lang == "en" else "تم إكمال الدورة!")
# Show reward summary
st.markdown("### Rewards Summary" if lang == "en" else "### ملخص الجوائز")
st.write((f"Total earned: {st.session_state.rewards} fils (BHD {st.session_state.rewards/1000.0:.3f})") if lang == "en" else (f"المجموع المكتسب: {st.session_state.rewards} فلس (ب.د {st.session_state.rewards/1000.0:.3f})"))
# Download certificate
cert_text = f"WaterGuard Course Certificate\nUser: demo_user@example.com\nCompleted: YES\nScore Summary: {json.dumps(st.session_state.quiz_scores)}\nRewards (fils): {st.session_state.rewards}"
st.download_button("Download Certificate (TXT)" if lang == "en" else "تحميل الشهادة (TXT)", data=cert_text, file_name="waterguard_certificate.txt")

# ----------------------------
# Bahrain History Tab
# ----------------------------
with top_tabs[1]:
st.header("Bahrain Water: History & Future" if lang == "en" else "تاريخ المياه في البحرين ومستقبلها")
if lang == "en":
st.markdown(BAHRAIN_HISTORY_EN)
else:
st.markdown(f"<div dir='rtl' style='text-align: right'>{BAHRAIN_HISTORY_AR}</div>", unsafe_allow_html=True)

# ----------------------------
# Dashboard Tab (main app content)
# ----------------------------
with top_tabs[2]:
# ---------- INTRO SECTION ----------
if lang == "en":
st.markdown("""
<div style="background: rgba(255, 255, 255, 0.9); padding: 2rem; border-radius: 15px; max-width: 900px; margin: 1.5rem auto; color: #111; box-shadow: 0 8px 20px rgba(0,0,0,0.15); font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
<h1 style="color: #023e8a; font-weight: 700;">💧 WaterGuard Prototype</h1>
<p style="font-size: 1.05rem; line-height: 1.5;">
WaterGuard is a smart AI-powered water monitoring prototype built for a residential home in Saar. It tracks daily water usage, detects abnormal spikes, and provides real-time alerts to help homeowners save water and reduce costs.
</p>
</div>
""", unsafe_allow_html=True)
else:
st.markdown("""
<div style="background: rgba(255, 255, 255, 0.9); padding: 2rem; border-radius: 15px; max-width: 900px; margin: 1.5rem auto; color: #111; box-shadow: 0 8px 20px rgba(0,0,0,0.15); font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; direction: rtl; text-align: right;">
<h1 style="color: #023e8a; font-weight: 700;">💧 نموذج ووتر جارد</h1>
<p style="font-size: 1.05rem; line-height: 1.5;">
ووتر جارد هو نموذج ذكي لمراقبة استهلاك المياه في منزل سكني بمنطقة سار. يستخدم الذكاء الاصطناعي لتحليل البيانات وكشف أي استهلاك غير طبيعي، مما يساعد على تقليل الهدر وخفض الفواتير.
</p>
</div>
""", unsafe_allow_html=True)

# ---------- SIDEBAR SUMMARY (recreate sidebar controls inside Dashboard tab for clarity) ----------
selected_day = st.sidebar.date_input(
"📅 Select a day to view usage" if lang == "en" else "📅 اختر اليوم لعرض الاستهلاك",
value=df['date'].max(),
min_value=df['date'].min(),
max_value=df['date'].max()
)
df_day = df[df['date'] == selected_day]
day_usage = df_day['usage_liters'].sum()
daily_quota = 1500
remaining = max(daily_quota - day_usage, 0)
usage_ratio = day_usage / daily_quota

cost_per_liter = 0.000193
daily_cost = day_usage * cost_per_liter

if lang == 'en':
st.sidebar.markdown(f"""
## 💧 Daily Water Usage Summary 
**Date:** {selected_day} 
**Used:** {day_usage:,.0f} liters 
**Remaining:** {remaining:,.0f} liters 
**Quota:** {daily_quota} liters 
**Estimated Cost:** BHD {daily_cost:.3f} 
""")
else:
st.sidebar.markdown(f"""
## 💧 ملخص استهلاك المياه اليومي 
**التاريخ:** {selected_day} 
**المستهلك:** {day_usage:,.0f} لتر 
**المتبقي:** {remaining:,.0f} لتر 
**الحصة اليومية:** {daily_quota} لتر 
**التكلفة التقديرية:** {daily_cost:.3f} دينار بحريني 
""")

st.sidebar.progress(min(usage_ratio, 1.0))

# Alerts
high_usage_threshold = daily_quota * 0.9
if day_usage > high_usage_threshold:
if lang == 'en':
st.sidebar.warning("🚨 High water consumption detected today!")
else:
st.sidebar.warning("🚨 تم الكشف عن استهلاك مياه مرتفع اليوم!")

# Anomalies table
df_anomalies = df[df['anomaly'] == 'Anomaly']
if lang == 'en':
st.markdown("## 🔍 Detected Anomalies (Possible Leaks or Spikes)")
else:
st.markdown("## 🔍 الأنماط الشاذة المكتشفة (تسريبات أو زيادات محتملة)")

with st.expander(f"{'Show' if lang == 'en' else 'إظهار'} الأنماط الشاذة / Anomalies"):
anomaly_display = df_anomalies[['timestamp', 'usage_liters', 'severity']].copy()
anomaly_display['usage_liters'] = anomaly_display['usage_liters'].map(lambda x: f"{x:.2f}")
anomaly_display['severity'] = anomaly_display['severity'].astype(str)
st.dataframe(anomaly_display)
csv_anomaly = anomaly_display.to_csv(index=False)
st.download_button(
label="Download Anomalies CSV" if lang == 'en' else "تحميل الأنماط الشاذة CSV",
data=csv_anomaly,
file_name='waterguard_anomalies.csv',
mime='text/csv'
)

# Usage visualization - hourly for selected day
df['time_str'] = df['timestamp'].dt.strftime('%H:%M')
df_day_hourly = df[df['date'] == selected_day]

if lang == 'en':
st.markdown(f"## 📊 Hourly Water Usage for {selected_day}")
else:
st.markdown(f"## 📊 استهلاك المياه الساعي ليوم {selected_day}")

fig1, ax1 = plt.subplots(figsize=(14, 6))
sns.lineplot(data=df_day_hourly, x='time_str', y='usage_liters', ax=ax1, label='Usage' if lang == 'en' else 'الاستهلاك')
sns.scatterplot(data=df_day_hourly[df_day_hourly['anomaly'] == 'Anomaly'],
x='time_str', y='usage_liters',
color='red', marker='X', s=60, label='Anomaly' if lang == 'en' else 'خلل', ax=ax1)
ax1.set_xlabel('Time (HH:MM)' if lang == 'en' else 'الوقت (ساعة:دقيقة)')
ax1.set_ylabel('Liters' if lang == 'en' else 'لتر')
ax1.set_title(f"Hourly Water Usage for {selected_day}" if lang == 'en' else f"استهلاك المياه الساعي ليوم {selected_day}")
ax1.tick_params(axis='x', rotation=45)
ax1.legend()
st.pyplot(fig1)

# Daily data for last year
df_daily = df.set_index('timestamp').resample('D')['usage_liters'].sum().reset_index()
if lang == 'en':
st.markdown("## 📈 Daily Water Usage (Past Year)")
else:
st.markdown("## 📈 استهلاك المياه اليومي (السنة الماضية)")

fig2, ax2 = plt.subplots(figsize=(14, 5))
sns.lineplot(data=df_daily, x='timestamp', y='usage_liters', ax=ax2)
ax2.set_xlabel('Date' if lang == 'en' else 'التاريخ')
ax2.set_ylabel('Liters' if lang == 'en' else 'لتر')
ax2.set_title('Daily Water Usage' if lang == 'en' else 'استهلاك المياه اليومي')
ax2.tick_params(axis='x', rotation=45)
st.pyplot(fig2)

# Monthly data
df_monthly = df.set_index('timestamp').resample('M')['usage_liters'].sum().reset_index()
if lang == 'en':
st.markdown("## 📉 Monthly Water Usage (Past Year)")
else:
st.markdown("## 📉 استهلاك المياه الشهري (السنة الماضية)")

fig3, ax3 = plt.subplots(figsize=(14, 5))
sns.lineplot(data=df_monthly, x='timestamp', y='usage_liters', ax=ax3)
ax3.set_xlabel('Month' if lang == 'en' else 'الشهر')
ax3.set_ylabel('Liters' if lang == 'en' else 'لتر')
ax3.set_title('Monthly Water Usage' if lang == 'en' else 'استهلاك المياه الشهري')
ax3.tick_params(axis='x', rotation=45)
st.pyplot(fig3)

# Daily report download
if lang == 'en':
st.markdown("## 📥 Download Daily Usage Report")
else:
st.markdown("## 📥 تحميل تقرير الاستهلاك اليومي")

daily_report_csv = df_day.to_csv(index=False)
st.download_button(
label="Download Daily Report CSV" if lang == 'en' else "تحميل تقرير الاستهلاك اليومي CSV",
data=daily_report_csv,
file_name=f'daily_usage_{selected_day}.csv',
mime='text/csv'
)

# Real-time notification if anomaly present today
if "Anomaly" in df_day["anomaly"].values:
if lang == 'en':
st.warning("🚨 High water consumption anomaly detected today!")
else:
st.warning("🚨 تم الكشف عن خلل استهلاك المياه اليوم!")

# Water conservation tips
if lang == 'en':
st.markdown("### 💡 Water Conservation Tips")
st.markdown("""
- Fix leaks promptly to save water and money.
- Use water-efficient appliances and fixtures.
- Collect rainwater for irrigation.
- Turn off taps when not in use.
- Monitor your usage regularly to detect changes.
""")
else:
st.markdown("### 💡 نصائح للحفاظ على المياه")
st.markdown("""
- أصلح التسريبات بسرعة لتوفير المياه والمال.
- استخدم الأجهزة والتركيبات الموفرة للمياه.
- اجمع مياه الأمطار للري.
- أغلق الصنابير عند عدم الاستخدام.
- راقب استهلاكك للكشف عن التغيرات.
""")

# FAQ
if lang == "en":
st.markdown("""
<div style="
background: rgba(255, 255, 255, 0.9);
padding: 1rem 1.5rem;
border-radius: 12px;
margin-top: 1rem;
color: #111;
box-shadow: 0 2px 8px rgba(0,0,0,0.05);
">
<h2 style="color: #023e8a;">💧 WaterGuard FAQ</h2>
</div>
""", unsafe_allow_html=True)

faqs_en = {
"How can I detect a water leak early?":
"Use WaterGuard's anomaly detection alerts to spot unusual spikes.",
"What should I do if an anomaly is detected?":
"Check for leaks or unusual water usage immediately.",
"Can WaterGuard monitor multiple locations?":
"Yes, it supports tracking usage across various branches or sites.",
"How accurate is the anomaly detection?":
"The system uses AI to detect 95% of irregular water usage patterns.",
"Is WaterGuard suitable for factories with large consumption?":
"Yes, it manages high-volume water use and alerts for excess.",
"How often is water usage data updated?":
"Data is updated hourly for precise monitoring and alerts.",
"Can I download daily usage reports?":
"Yes, downloadable CSV reports are available for any selected day.",
"What cost savings can I expect?":
"Early leak detection and usage optimization significantly reduce bills.",
"Does WaterGuard support multiple languages?":
"Currently supports English and Arabic interfaces.",
"Who do I contact for technical support?":
"Contact support@waterguard.bh for all maintenance and help queries."
}

for q, a in faqs_en.items():
st.markdown(f"""
<div style="
background: rgba(255, 255, 255, 0.85);
padding: 0.75rem 1rem;
border-radius: 10px;
margin-bottom: 0.8rem;
">
<strong style="color: #0077b6;">{q}</strong>
<p style="margin-top: 0.4rem;">{a}</p>
</div>
""", unsafe_allow_html=True)

else:
st.markdown("""
<div style="
background: rgba(255, 255, 255, 0.9);
padding: 1rem 1.5rem;
border-radius: 12px;
margin-top: 1rem;
color: #111;
box-shadow: 0 2px 8px rgba(0,0,0,0.05);
direction: rtl;
text-align: right;
">
<h2 style="color: #023e8a;">💧 الأسئلة المتكررة - ووتر جارد</h2>
</div>
""", unsafe_allow_html=True)

faqs_ar = {
"كيف يمكنني اكتشاف تسريب المياه مبكرًا؟":
"استخدم تنبيهات كشف الخلل من ووتر جارد لرصد الزيادات غير المعتادة.",
"ماذا أفعل إذا تم اكتشاف خلل؟":
"تحقق فورًا من وجود تسريبات أو استهلاك غير طبيعي للمياه.",
"هل يمكن لووتر جارد مراقبة مواقع متعددة؟":
"نعم، يدعم تتبع الاستهلاك عبر فروع أو مواقع مختلفة.",
"ما مدى دقة كشف الخلل؟":
"يستخدم النظام الذكاء الاصطناعي لاكتشاف 95٪ من أنماط الاستهلاك غير الطبيعية.",
"هل ووتر جارد مناسب للمصانع ذات الاستهلاك الكبير؟":
"نعم، يدير استهلاك المياه العالي ويرسل تنبيهات عند الزيادة.",
"كم مرة يتم تحديث بيانات استهلاك المياه؟":
"يتم تحديث البيانات كل ساعة لمراقبة دقيقة وتنبيهات فورية.",
"هل يمكنني تحميل تقارير الاستهلاك اليومية؟":
"نعم، تتوفر تقارير CSV قابلة للتحميل لأي يوم محدد.",
"ما مقدار التوفير المتوقع في التكاليف؟":
"الكشف المبكر عن التسريبات وتحسين الاستخدام يقلل الفواتير بشكل كبير.",
"هل يدعم ووتر جارد لغات متعددة؟":
"يدعم حاليًا واجهات باللغتين الإنجليزية والعربية.",
"من أتصل به للدعم الفني؟":
"تواصل مع support@waterguard.bh لجميع استفسارات الصيانة والمساعدة."
}

for q, a in faqs_ar.items():
st.markdown(f"""
<div style="
background: rgba(255, 255, 255, 0.85);
padding: 0.75rem 1rem;
border-radius: 10px;
margin-bottom: 0.8rem;
direction: rtl;
text-align: right;
">
<strong style="color: #0077b6;">{q}</strong>
<p style="margin-top: 0.4rem;">{a}</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# Testimonials (English & Arabic)
# ----------------------------
testimonial_data = [
"💡 WaterGuard helped me discover a hidden leak — saved me BHD 12 this month!",
"✅ The alerts are super accurate. I got notified before a serious leak became worse.",
"📈 I love the usage graphs. Makes me aware of our daily water behavior.",
"💧 We found our garden sprinkler system was overwatering — now fixed!",
"🏡 Great for homes with large families — helps avoid high bills.",
"📊 Downloaded a report and shared it with my landlord. Very professional!",
"📱 The dashboard is clean and easy to use. Even my kids get it!",
"🔔 Real-time alerts helped me stop water waste while traveling.",
"🧠 I never knew how much the kitchen used until WaterGuard showed me.",
"🌱 We’re now more eco-conscious thanks to WaterGuard’s tips and insights."
]

profiles = [
("👨‍💼", "Khalid", "khalid_madan76@outlook.com"),
("👨‍💼", "Yousef", "yousef_albahbhani76@gmail.com"),
("👨‍💼", "Omar", "omar_abdullah36555@yahoo.com"),
("👨‍💼", "Adel", "adel_doseri55@yahoo.com"),
("👨‍💼", "Hassan", "hassan_al_anazi82@gmail.com"),
("👩‍💼", "Noor", "noor_01_altwash98@yahoo.com"),
("👩‍💼", "Mariam", "mariam_11_alekrawi@yahoo.com"),
("👩‍💼", "Rana", "rana_al_shammri93@outlook.com"),
("👩‍💼", "Zahra", "zahra_almtari31@outlook.com"),
("👩‍💼", "Aisha", "aisha_buqais2306@gmail.com"),
]

if lang == "en":
st.markdown("""
<div role="region" aria-label="User Testimonials" style="
background: rgba(255, 255, 255, 0.9);
padding: 1rem 1.5rem;
border-radius: 12px;
margin-top: 1rem;
color: #111;
box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
<h3 style="color: #023e8a;">💬 User Testimonials</h3>
</div>
""", unsafe_allow_html=True)

for i in range(len(testimonial_data)):
emoji, name, email = profiles[i]
testimonial = testimonial_data[i]
st.markdown(f"""
<div style="background: rgba(255, 255, 255, 0.85);
padding: 0.75rem 1rem;
border-radius: 8px;
margin-bottom: 0.8rem;
color: #111;
box-shadow: 0 1px 6px rgba(0,0,0,0.04);">
<strong>{emoji} {name} — <span style="color: #666;">{email}</span></strong>
<p style="margin-top: 0.4rem;">{testimonial}</p>
</div>
""", unsafe_allow_html=True)
else:
testimonial_data_ar = [
"💡 ساعدني ووتر جارد في اكتشاف تسريب مخفي — وفرت 12 دينار بحريني هذا الشهر!",
"✅ التنبيهات دقيقة للغاية. تم إعلامي قبل أن يصبح التسريب خطيرًا.",
"📈 أحب رسوم البيانية للاستهلاك. تجعلني على دراية بسلوكنا اليومي للمياه.",
"💧 اكتشفنا أن نظام رشاشات الحديقة كان يروي أكثر من اللازم — تم إصلاحه الآن!",
"🏡 رائع للمنازل التي تضم عائلات كبيرة — يساعد على تجنب الفواتير المرتفعة.",
"📊 حملت تقريرًا وشاركته مع مالك العقار. احترافي للغاية!",
"📱 لوحة التحكم نظيفة وسهلة الاستخدام. حتى أطفالي يفهمونها!",
"🔔 التنبيهات في الوقت الفعلي ساعدتني على وقف هدر المياه أثناء السفر.",
"🧠 لم أكن أعرف كم تستهلك المطبخ حتى أظهر لي ووتر جارد.",
"🌱 أصبحنا الآن أكثر وعيًا بيئيًا بفضل نصائح ورؤى ووتر جارد."
]

st.markdown("""
<div role="region" aria-label="شهادات المستخدمين" style="
background: rgba(255, 255, 255, 0.9);
padding: 1rem 1.5rem;
border-radius: 12px;
margin-top: 1rem;
color: #111;
box-shadow: 0 2px 8px rgba(0,0,0,0.05);
direction: rtl;
text-align: right;">
<h3 style="color: #023e8a;">💬 شهادات المستخدمين</h3>
</div>
""", unsafe_allow_html=True)

for i in range(len(testimonial_data_ar)):
emoji, name, email = profiles[i]
testimonial = testimonial_data_ar[i]
st.markdown(f"""
<div style="background: rgba(255, 255, 255, 0.85);
padding: 0.75rem 1rem;
border-radius: 8px;
margin-bottom: 0.8rem;
color: #111;
box-shadow: 0 1px 6px rgba(0,0,0,0.04);
direction: rtl;
text-align: right;">
<strong>{emoji} {name} — <span style="color: #666;">{email}</span></strong>
<p style="margin-top: 0.4rem;">{testimonial}</p>
</div>
""", unsafe_allow_html=True)

# End of app

