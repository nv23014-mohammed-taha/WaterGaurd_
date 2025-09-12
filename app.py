# -*- coding: utf-8 -*-
"""WaterGuard App"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
import random
import base64
from streamlit.components.v1 import html

sns.set_style("whitegrid")
st.set_page_config(page_title="WaterGuard", layout="wide")

# ---------- LANGUAGE TOGGLE ---------- #
language = st.sidebar.radio("🌐 Language / اللغة", ["English", "العربية"])
lang = "ar" if language == "العربية" else "en"

# ---------- SCREEN READER BUTTON ---------- #
def screen_reader_button(lang):
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
        {'margin-left: auto;' if lang == 'en' else 'margin-right: auto;'}
    ">
        🔊 {'Activate Screen Reader' if lang == 'en' else 'تشغيل قارئ الشاشة'}
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
        utterance.lang = '{'en-US' if lang == 'en' else 'ar-SA'}';
        synth.speak(utterance);
    }}
    </script>
    """
    html(button_html, height=60)

# Show the button in the sidebar
with st.sidebar:
    screen_reader_button(lang)

# ---------- BACKGROUND IMAGE ---------- #
def set_background(image_path):
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

set_background("water_bg.jpg")

# ---------- INTRO SECTION ---------- #
if lang == "en":
    st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.9); padding: 2rem; border-radius: 15px; max-width: 900px; margin: 3rem auto; color: #111; box-shadow: 0 8px 20px rgba(0,0,0,0.15); font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
            <h1 style="color: #023e8a; font-weight: 700;">💧 WaterGuard Prototype</h1>
            <p style="font-size: 1.1rem; line-height: 1.5;">
                WaterGuard is a smart AI-powered water monitoring prototype built for a residential home in Saar. It tracks daily water usage, detects abnormal spikes, and provides real-time alerts to help homeowners save water and reduce costs.
            </p>
            <h3 style="color: #023e8a; font-weight: 700;">Key Features:</h3>
            <ul style="font-size: 1rem; line-height: 1.6;">
                <li><strong>Leak Detection:</strong> Automatically identifies unusual usage that may indicate a leak.</li>
                <li><strong>Real-Time Alerts:</strong> Warns users when consumption exceeds normal levels.</li>
                <li><strong>Usage Reports:</strong> Visualizes daily and monthly usage to support smart water habits.</li>
            </ul>
            <h3 style="color: #023e8a; font-weight: 700;">Why WaterGuard?</h3>
            <ul style="font-size: 1rem; line-height: 1.6;">
                <li><strong>Smart Monitoring:</strong> Gain full insight into your household's water behavior.</li>
                <li><strong>Cost Savings:</strong> Reduce your monthly water bill through early detection and optimization.</li>
                <li><strong>Eco-Friendly:</strong> Support sustainability by preventing waste.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.9); padding: 2rem; border-radius: 15px; max-width: 900px; margin: 3rem auto; color: #111; box-shadow: 0 8px 20px rgba(0,0,0,0.15); font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; direction: rtl; text-align: right;">
            <h1 style="color: #023e8a; font-weight: 700;">💧 نموذج ووتر جارد</h1>
            <p style="font-size: 1.1rem; line-height: 1.5;">
                ووتر جارد هو نموذج ذكي لمراقبة استهلاك المياه في منزل سكني بمنطقة سار. يستخدم الذكاء الاصطناعي لتحليل البيانات وكشف أي استهلاك غير طبيعي، مما يساعد على تقليل الهدر وخفض الفواتير.
            </p>
            <h3 style="color: #023e8a; font-weight: 700;">الميزات الرئيسية:</h3>
            <ul style="font-size: 1rem; line-height: 1.6;">
                <li><strong>كشف التسريبات:</strong> يحدد تلقائيًا أي زيادات غير طبيعية قد تشير إلى وجود تسريب.</li>
                <li><strong>تنبيهات فورية:</strong> يحذرك عندما يتجاوز الاستهلاك المستويات الطبيعية.</li>
                <li><strong>تقارير استهلاك:</strong> يعرض الاستخدام اليومي والشهري بطريقة مرئية وسهلة الفهم.</li>
            </ul>
            <h3 style="color: #023e8a; font-weight: 700;">لماذا ووتر جارد؟</h3>
            <ul style="font-size: 1rem; line-height: 1.6;">
                <li><strong>مراقبة ذكية:</strong> احصل على رؤية شاملة لسلوك استهلاك المياه في منزلك.</li>
                <li><strong>توفير في التكاليف:</strong> خفّض فواتيرك من خلال الكشف المبكر والتحسين المستمر.</li>
                <li><strong>صديق للبيئة:</strong> ساهم في الاستدامة من خلال تقليل الهدر.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# ... other imports at the top ...

def screen_reader_button(lang):
    button_html = f"""
    <button onclick="readPage()" style="
        background-color:#023e8a; 
        color:white; 
        border:none; 
        padding:10px 20px; 
        border-radius:10px; 
        cursor:pointer;
        font-size:1rem;
        margin: 1rem 0;">
        🔊 {'Activate Screen Reader' if lang == 'en' else 'تشغيل قارئ الشاشة'}
    </button>
    <script>
    function readPage() {{
        const synth = window.speechSynthesis;
        if (synth.speaking) {{
            synth.cancel();
        }}
        const app = document.querySelector('.stApp');
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
        utterance.lang = '{ "en-US" if lang == "en" else "ar-SA" }';
        synth.speak(utterance);
    }}
    </script>
    """
    html(button_html, height=60)

# ---------- DATA SIMULATION ---------- #
@st.cache_data
def simulate_data():
    np.random.seed(42)
    hours = 365 * 24
    date_range = pd.date_range(start='2024-01-01', periods=hours, freq='H')

    usage_main = np.random.normal(12, 3, hours).clip(0, 50)
    usage_garden = np.random.normal(5, 2, hours).clip(0, 20)
    usage_kitchen = np.random.normal(3, 1, hours).clip(0, 10)
    usage_bathroom = np.random.normal(4, 1.5, hours).clip(0, 15)

    df = pd.DataFrame({
        'timestamp': date_range,
        'usage_main_liters': usage_main,
        'usage_garden_liters': usage_garden,
        'usage_kitchen_liters': usage_kitchen,
        'usage_bathroom_liters': usage_bathroom,
    })

    # Add total usage
    df['usage_liters'] = df[[
        'usage_main_liters', 'usage_garden_liters', 'usage_kitchen_liters', 'usage_bathroom_liters'
    ]].sum(axis=1)

    df['date'] = df['timestamp'].dt.date

    # Inject synthetic anomaly spikes (~5%)
    num_anomalies = int(0.05 * len(df))
    anomaly_indices = random.sample(range(len(df)), num_anomalies)
    for i in anomaly_indices:
        df.loc[i, ['usage_main_liters', 'usage_garden_liters', 'usage_kitchen_liters', 'usage_bathroom_liters']] *= np.random.uniform(2, 5)
    # Recalculate total after anomalies
    df['usage_liters'] = df[[
        'usage_main_liters', 'usage_garden_liters', 'usage_kitchen_liters', 'usage_bathroom_liters'
    ]].sum(axis=1)

    return df

df = simulate_data()

# ---------- ANOMALY DETECTION ---------- #
model = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = model.fit_predict(df[['usage_liters']])
df['anomaly'] = df['anomaly'].map({1: 'Normal', -1: 'Anomaly'})

# Add severity column based on usage liters
df['severity'] = pd.cut(df['usage_liters'],
                        bins=[-np.inf, 20, 40, np.inf],
                        labels=['Low', 'Medium', 'High'])

# ---------- SIDEBAR SUMMARY ---------- #
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

# Calculate daily cost estimate (Bahraini Dinars)
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


# ---------- ALERTS ---------- #
high_usage_threshold = daily_quota * 0.9

if day_usage > high_usage_threshold:
    alert_text_en = "🚨 High water consumption detected today!"
    alert_text_ar = "🚨 تم الكشف عن استهلاك مياه مرتفع اليوم!"
    if lang == 'en':
        st.sidebar.warning(alert_text_en)
    else:
        st.sidebar.warning(alert_text_ar)

# ---------- ANOMALIES TABLE ---------- #
df_anomalies = df[df['anomaly'] == 'Anomaly']

if lang == 'en':
    st.markdown("## 🔍 Detected Anomalies (Possible Leaks or Spikes)")
else:
    st.markdown("## 🔍 anomalies المكتشفة (تسريبات أو زيادات محتملة)")

with st.expander(f"{'Show' if lang == 'en' else 'إظهار'} anomalies / anomalies"):
    anomaly_display = df_anomalies[['timestamp', 'usage_liters', 'severity']].copy()
    anomaly_display['usage_liters'] = anomaly_display['usage_liters'].map(lambda x: f"{x:.2f}")
    anomaly_display['severity'] = anomaly_display['severity'].astype(str)
    st.dataframe(anomaly_display)

    # Export anomaly data CSV
    csv_anomaly = anomaly_display.to_csv(index=False)
    st.download_button(
        label="Download Anomalies CSV" if lang == 'en' else "تحميل anomalies CSV",
        data=csv_anomaly,
        file_name='waterguard_anomalies.csv',
        mime='text/csv'
    )

# ---------- USAGE VISUALIZATION ---------- #

# Prepare hourly data for selected day
df['time_str'] = df['timestamp'].dt.strftime('%H:%M')
df_day_hourly = df[df['date'] == selected_day]

if lang == 'en':
    st.markdown(f"## 📊 Hourly Water Usage for {selected_day}")
else:
    st.markdown(f"## 📊 استهلاك المياه الساعي ليوم {selected_day}")

fig1, ax1 = plt.subplots(figsize=(14,6))
sns.lineplot(data=df_day_hourly, x='time_str', y='usage_liters', ax=ax1, label='Usage')
sns.scatterplot(data=df_day_hourly[df_day_hourly['anomaly']=='Anomaly'],
                x='time_str', y='usage_liters',
                color='red', marker='X', s=60, label='Anomaly', ax=ax1)
ax1.set_xlabel('Time (HH:MM)' if lang == 'en' else 'الوقت (ساعة:دقيقة)')
ax1.set_ylabel('Liters' if lang == 'en' else 'لتر')
ax1.set_title(f"Hourly Water Usage for {selected_day}" if lang == 'en' else f"استهلاك المياه الساعي ليوم {selected_day}")
ax1.tick_params(axis='x', rotation=45)
ax1.legend()
st.pyplot(fig1)

# Prepare daily data for last year
df_daily = df.set_index('timestamp').resample('D')['usage_liters'].sum().reset_index()
if lang == 'en':
    st.markdown("## 📈 Daily Water Usage (Past Year)")
else:
    st.markdown("## 📈 استهلاك المياه اليومي (السنة الماضية)")

fig2, ax2 = plt.subplots(figsize=(14,5))
sns.lineplot(data=df_daily, x='timestamp', y='usage_liters', ax=ax2)
ax2.set_xlabel('Date' if lang == 'en' else 'التاريخ')
ax2.set_ylabel('Liters' if lang == 'en' else 'لتر')
ax2.set_title('Daily Water Usage' if lang == 'en' else 'استهلاك المياه اليومي')
ax2.tick_params(axis='x', rotation=45)
st.pyplot(fig2)

# Prepare monthly data for last year
df_monthly = df.set_index('timestamp').resample('M')['usage_liters'].sum().reset_index()
if lang == 'en':
    st.markdown("## 📉 Monthly Water Usage (Past Year)")
else:
    st.markdown("## 📉 استهلاك المياه الشهري (السنة الماضية)")

fig3, ax3 = plt.subplots(figsize=(14,5))
sns.lineplot(data=df_monthly, x='timestamp', y='usage_liters', ax=ax3)
ax3.set_xlabel('Month' if lang == 'en' else 'الشهر')
ax3.set_ylabel('Liters' if lang == 'en' else 'لتر')
ax3.set_title('Monthly Water Usage' if lang == 'en' else 'استهلاك المياه الشهري')
ax3.tick_params(axis='x', rotation=45)
st.pyplot(fig3)

# ---------- DAILY REPORT DOWNLOAD ---------- #
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

# --------- Real-Time Notifications & Alerts --------- #
if "Anomaly" in df_day["anomaly"].values:
    if lang == 'en':
        st.warning("🚨 High water consumption anomaly detected today!")
    else:
        st.warning("🚨 تم الكشف عن خلل استهلاك المياه اليوم!")

# --------- Water Conservation Tips --------- #
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

# --------- FAQ Section (Translucent white block + expanders) --------- #

# ---------- FAQ SECTION AT END ---------- #
if lang == "en":
    st.markdown("""
    <div style="
        background: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 15px;
        max-width: 900px;
        margin: 3rem auto 2rem auto;
        color: #111;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    ">
        <h2 style="color: #023e8a; font-weight: 700;">💧 WaterGuard FAQ</h2>
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
            padding: 1rem 1.5rem;
            border-radius: 12px;
            margin-bottom: 1.2rem;
            color: #111;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        ">
            <strong style="color: #0077b6;">{q}</strong>
            <p style="margin-top: 0.5rem;">{a}</p>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="
        background: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 15px;
        max-width: 900px;
        margin: 3rem auto 2rem auto;
        color: #111;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        direction: rtl;
        text-align: right;
    ">
        <h2 style="color: #023e8a; font-weight: 700;">💧 الأسئلة المتكررة - ووتر جارد</h2>
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
            padding: 1rem 1.5rem;
            border-radius: 12px;
            margin-bottom: 1.2rem;
            color: #111;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            direction: rtl;
            text-align: right;
        ">
            <strong style="color: #0077b6;">{q}</strong>
            <p style="margin-top: 0.5rem;">{a}</p>
        </div>
        """, unsafe_allow_html=True)
       # --------- USER TESTIMONIALS SECTION WITH NAME, EMAIL, EMOJI --------- #
from random import choice

testimonial_data = [
    ("💡 WaterGuard helped me discover a hidden leak — saved me BHD 12 this month!"),
    ("✅ The alerts are super accurate. I got notified before a serious leak became worse."),
    ("📈 I love the usage graphs. Makes me aware of our daily water behavior."),
    ("💧 We found our garden sprinkler system was overwatering — now fixed!"),
    ("🏡 Great for homes with large families — helps avoid high bills."),
    ("📊 Downloaded a report and shared it with my landlord. Very professional!"),
    ("📱 The dashboard is clean and easy to use. Even my kids get it!"),
    ("🔔 Real-time alerts helped me stop water waste while traveling."),
    ("🧠 I never knew how much the kitchen used until WaterGuard showed me."),
    ("🌱 We’re now more eco-conscious thanks to WaterGuard’s tips and insights.")
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
        padding: 2rem;
        border-radius: 15px;
        max-width: 900px;
        margin: 3rem auto 2rem auto;
        color: #111;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        <h2 style="color: #023e8a; font-weight: 700;">💬 User Testimonials</h2>
    </div>
    """, unsafe_allow_html=True)

    for i in range(len(testimonial_data)):
        emoji, name, email = profiles[i]
        testimonial = testimonial_data[i]
        st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.85);
                    padding: 1rem 1.5rem;
                    border-radius: 12px;
                    margin-bottom: 1.2rem;
                    color: #111;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
            <strong>{emoji} {name} — <span style="color: #666;">{email}</span></strong>
            <p style="margin-top: 0.5rem;">{testimonial}</p>
        </div>
        """, unsafe_allow_html=True)

import streamlit as st
import time

# Example alert simulation
if "leak_detected" not in st.session_state:
    st.session_state.leak_detected = False

st.title("🚨 WaterGuard Leak Alerts")

if st.button("Simulate Leak Detection"):
    st.session_state.leak_detected = True

if st.session_state.leak_detected:
    placeholder = st.empty()
    for i in range(6):  # Flash 3 times
        placeholder.error("⚠️ Leak Detected in Kitchen Pipe! Risk Level: HIGH")
        time.sleep(0.5)
        placeholder.empty()
        time.sleep(0.5)
import plotly.express as px
import pandas as pd

st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to:", ["Dashboard", "Reports", "Robot Status"])

if page == "Reports":
    st.title("📊 Water Usage Reports")

    # Example dataset
    df = pd.DataFrame({
        "Day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        "Usage (L)": [120, 135, 110, 150, 200, 170, 140]
    })

    # Plotly line chart
    fig = px.line(df, x="Day", y="Usage (L)", markers=True,
                  title="Weekly Water Usage")
    st.plotly_chart(fig, use_container_width=True)

    # Prediction (dummy example)
    st.success("🤖 AI Prediction: Medium risk of leak in Bathroom pipe within 2 weeks.")

import time

if page == "Robot Status":
    st.title("🤖 Pipe Inspection & Cleaning")

    progress = st.progress(0)
    status_text = st.empty()

    for i in range(101):
        progress.progress(i)
        if i < 30:
            status_text.text("🔍 Inspecting pipes...")
        elif i < 70:
            status_text.text("🧽 Cleaning buildup...")
        else:
            status_text.text("✅ Inspection & cleaning complete.")
        time.sleep(0.05)

    st.success("Pipes are healthy! ✅ No critical damage detected.")

import streamlit as st

st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to:", ["Dashboard", "Reports", "Robot Status", "Education"])

if page == "Education":
    st.title("📘 Water Conservation Education")

    # Intro context
    st.markdown("""
    🌍 **Bahrain is one of the driest countries in the world.**  
    Despite this, it also has one of the **highest water usage rates per resident**.  
    Conserving water is not only important for the environment but also for the future of Bahrain’s people and economy.  
    """)

    # Audience selection
    audience = st.radio("👤 Who are you?", ["Adult", "Kid"])

    # Language selector
    lang = st.selectbox("🌍 Choose Language:", ["English", "العربية", "Français"])

import streamlit as st

# Initialize session state
if "module" not in st.session_state:
    st.session_state.module = 1
if "audience" not in st.session_state:
    st.session_state.audience = "Adult"
if "video_completed" not in st.session_state:
    st.session_state.video_completed = False

st.title("📘 WaterGuard Education Course")

# Select path
audience = st.radio("👤 Who are you?", ["Adult", "Kid"])
st.session_state.audience = audience



# Initialize session state
if "module" not in st.session_state:
    st.session_state.module = 1
if "audience" not in st.session_state:
    st.session_state.audience = "Adult"
if "video_completed" not in st.session_state:
    st.session_state.video_completed = False
if "lang" not in st.session_state:
    st.session_state.lang = "English"

st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to:", ["Dashboard", "Reports", "Robot Status", "Education"])

if page == "Education":
    st.title("📘 WaterGuard Education Course")

    # Intro context
    st.markdown("""
    🌍 **Bahrain is one of the driest countries in the world.**  
    Despite this, it also has one of the **highest water usage rates per resident**.  
    Conserving water is important for the environment and for the future of Bahrain’s people and economy.  
    """)

    # Audience selection
    audience = st.radio("👤 Who are you?", ["Adult", "Kid"])
    st.session_state.audience = audience

    # Language selector
    lang = st.selectbox("🌍 Choose Language:", ["English", "العربية", "Français"])
    st.session_state.lang = lang

    # Adult modules (>300 words each)
adult_modules = {
    1: {
        "title": {
            "English": "💧 Module 1: The Water Challenge in Bahrain",
            "العربية": "💧 الوحدة 1: تحدي المياه في البحرين",
            "Français": "💧 Module 1 : Le défi de l’eau à Bahreïn"
        },
        "content": {
            "English": """
Bahrain is one of the most water-stressed countries in the world. On average, the country receives less than **80 mm of rainfall annually**, which is not nearly enough to meet the needs of its citizens. For decades, Bahrain has depended on **groundwater aquifers** and **desalination plants** as its main sources of drinking water. Groundwater has been heavily over-extracted, leading to **salinity intrusion from the sea**, making much of it unsuitable for human use. As a result, Bahrain now relies on desalination for **over 90% of its freshwater supply**.

Desalination, while effective, is both **energy-intensive and environmentally costly**. Powering desalination plants requires large amounts of fossil fuels, which contributes to greenhouse gas emissions. Additionally, the process creates **brine waste**, a salty byproduct that is often discharged back into the sea, harming marine ecosystems such as coral reefs, which are already under stress due to warming waters.

Another critical challenge is **consumption behavior**. A typical resident of Bahrain uses **250–300 liters of water per day**, which is nearly double the international average of 150 liters. Much of this consumption is wasted through overuse in household activities, inefficient appliances, and undetected leaks. With a population of around 1.5 million, this means Bahrainis are using hundreds of millions of liters every single day — a pace that is unsustainable given limited natural resources.

Experts project that if current trends continue, Bahrain could face **serious water shortages by 2050**, even with desalination. Rising energy costs, climate change, and higher demand due to population growth will only worsen the crisis. This module highlights why **behavioral change and technological adoption** — like smart leak detection, efficient appliances, and water-conscious habits — are essential to secure Bahrain’s water future.
""",
            "العربية": """
تُعد البحرين من أكثر الدول عرضة للإجهاد المائي في العالم. حيث تتلقى البلاد أقل من **80 ملم من الأمطار سنويًا**، وهو ما لا يكفي لتلبية احتياجات سكانها. لعقود طويلة، اعتمدت البحرين على **طبقات المياه الجوفية** ومحطات **تحلية المياه** كمصادر رئيسية للشرب. تم استخراج المياه الجوفية بشكل مفرط، مما أدى إلى **تسرب الملوحة من البحر** وجعل الكثير منها غير صالح للاستخدام البشري. لذلك تعتمد البحرين الآن على التحلية لتوفير **أكثر من 90% من إمدادات المياه العذبة**.

تعد التحلية فعالة، لكنها **تستهلك الطاقة كثيرًا وتكلف البيئة**. تشغيل محطات التحلية يتطلب كميات كبيرة من الوقود الأحفوري، مما يزيد من انبعاثات الغازات الدفيئة. بالإضافة إلى ذلك، تنتج عملية التحلية **نفايات مالحة** غالبًا ما تُصرف في البحر، مسببة ضررًا للنظم البيئية البحرية مثل الشعاب المرجانية، والتي تتعرض أصلاً للضغط نتيجة ارتفاع درجات حرارة المياه.

تحدٍ آخر مهم هو **سلوك الاستهلاك**. يستخدم المواطن العادي في البحرين **250–300 لتر من المياه يوميًا**، أي ضعف المتوسط العالمي تقريبًا البالغ 150 لترًا. ويهدر جزء كبير من هذا الاستهلاك في الاستخدام المفرط في الأنشطة المنزلية، والأجهزة غير الفعالة، والتسربات غير المكتشفة. مع عدد سكان يقارب 1.5 مليون، هذا يعني أن البحرينيين يستخدمون مئات الملايين من اللترات يوميًا — وهو معدل غير مستدام بالنظر إلى الموارد المحدودة.

يتوقع الخبراء أنه إذا استمرت الاتجاهات الحالية، فقد تواجه البحرين **نقصًا شديدًا في المياه بحلول عام 2050**، حتى مع الاعتماد على التحلية. وستزيد تكاليف الطاقة وارتفاع الطلب بسبب النمو السكاني وتغير المناخ من حدة الأزمة. تؤكد هذه الوحدة أهمية **التغيير السلوكي واعتماد التكنولوجيا** مثل اكتشاف التسربات الذكي، والأجهزة الموفرة، والعادات المائية الواعية لضمان مستقبل آمن للمياه في البحرين.
""",
            "Français": """
Bahreïn est l’un des pays les plus stressés par le manque d’eau dans le monde. En moyenne, le pays reçoit moins de **80 mm de pluie par an**, ce qui est insuffisant pour répondre aux besoins de ses habitants. Depuis des décennies, Bahreïn dépend des **nappes phréatiques** et des **stations de dessalement** comme principales sources d’eau potable. Les nappes phréatiques ont été fortement surexploitées, entraînant une **intrusion de la salinité de la mer**, rendant une grande partie de l’eau inutilisable pour la consommation humaine. Par conséquent, Bahreïn s’appuie désormais sur le dessalement pour **plus de 90 % de son approvisionnement en eau douce**.

Le dessalement, bien qu’efficace, est **très énergivore et coûteux pour l’environnement**. Alimenter les stations de dessalement nécessite de grandes quantités de combustibles fossiles, contribuant aux émissions de gaz à effet de serre. De plus, le processus génère des **déchets salins**, souvent rejetés en mer, ce qui nuit aux écosystèmes marins comme les récifs coralliens déjà sous pression due au réchauffement des eaux.

Un autre défi majeur est **le comportement de consommation**. Un résident typique de Bahreïn utilise **250–300 litres d’eau par jour**, presque le double de la moyenne internationale de 150 litres. Une grande partie de cette consommation est gaspillée par la surutilisation dans les activités domestiques, les appareils inefficaces et les fuites non détectées. Avec une population d’environ 1,5 million, cela signifie que les Bahreïnis utilisent des centaines de millions de litres chaque jour — un rythme insoutenable compte tenu des ressources limitées.

Les experts prévoient que si les tendances actuelles se poursuivent, Bahreïn pourrait faire face à des **pénuries d’eau graves d’ici 2050**, même avec le dessalement. L’augmentation des coûts énergétiques, le changement climatique et la croissance de la demande aggraveraient encore la crise. Ce module souligne pourquoi **le changement de comportement et l’adoption technologique** — comme la détection intelligente des fuites, les appareils efficaces et les habitudes de consommation responsables — sont essentiels pour garantir l’avenir de l’eau à Bahreïn.
"""
        },
        "video": "https://www.youtube.com/watch?v=YFt3ONM7eH0"
    }
}

        2: {
            "title": {
                "English": "♻️ Module 2: Smart Daily Practices",
                "العربية": "♻️ الوحدة 2: ممارسات يومية ذكية",
                "Français": "♻️ Module 2 : Pratiques quotidiennes intelligentes"
            },
            "content": {
                "English": """Daily water-saving practices in Bahrain must go beyond simple awareness campaigns...""",
                "العربية": """يجب أن تتجاوز ممارسات توفير المياه اليومية في البحرين الحملات التوعوية البسيطة...""",
                "Français": """Les pratiques quotidiennes d’économie d’eau à Bahreïn doivent aller au-delà de simples campagnes de sensibilisation..."""
            },
            "video": {
                "English": "https://www.youtube.com/watch?v=U6pAB4yQ58U",
                "العربية": "https://www.youtube.com/watch?v=mi_K7eLNz_M",
                "Français": "https://www.youtube.com/watch?v=zVZ2iK2dJdM"
            }
        },
        3: {
            "title": {
                "English": "🔧 Module 3: Leak Prevention & Detection",
                "العربية": "🔧 الوحدة 3: منع واكتشاف التسربات",
                "Français": "🔧 Module 3 : Prévention et détection des fuites"
            },
            "content": {
                "English": """One of the most overlooked yet impactful areas of water conservation in Bahrain is leak detection...""",
                "العربية": """أحد أكثر المجالات التي يتم تجاهلها رغم تأثيرها الكبير في الحفاظ على المياه في البحرين هو اكتشاف التسربات...""",
                "Français": """L’un des aspects les plus négligés mais pourtant impactants de la conservation de l’eau à Bahreïn est la détection des fuites..."""
            },
            "video": {
                "English": "https://www.youtube.com/watch?v=HMblNYq69fg",
                "العربية": "https://www.youtube.com/watch?v=mi_K7eLNz_M",
                "Français": "https://www.youtube.com/watch?v=zVZ2iK2dJdM"
            }
        },
        4: {
            "title": {
                "English": "🏢 Module 4: Industry & Community",
                "العربية": "🏢 الوحدة 4: الصناعة والمجتمع",
                "Français": "🏢 Module 4 : Industrie et communauté"
            },
            "content": {
                "English": """While households play a critical role in conservation, industries, businesses, and communities in Bahrain are equally important...""",
                "العربية": """بينما تلعب الأسر دورًا حيويًا في الحفاظ على المياه، فإن الصناعات والشركات والمجتمعات في البحرين لها أهمية مماثلة...""",
                "Français": """Bien que les ménages jouent un rôle crucial dans la conservation, les industries, entreprises et communautés à Bahreïn sont tout aussi importantes..."""
            },
            "video": {
                "English": "https://www.youtube.com/watch?v=zVZ2iK2dJdM",
                "العربية": "https://www.youtube.com/watch?v=mi_K7eLNz_M",
                "Français": "https://www.youtube.com/watch?v=zVZ2iK2dJdM"
            }
        },
        5: {
            "title": {
                "English": "🚀 Module 5: The Future of Water in Bahrain",
                "العربية": "🚀 الوحدة 5: مستقبل المياه في البحرين",
                "Français": "🚀 Module 5 : L’avenir de l’eau à Bahreïn"
            },
            "content": {
                "English": """Looking ahead, the future of water management in Bahrain will be shaped by technological innovation, policy reforms, and behavioral change...""",
                "العربية": """بالنظر إلى المستقبل، فإن إدارة المياه في البحرين ستتأثر بالابتكار التكنولوجي والإصلاحات السياسية وتغيير السلوكيات...""",
                "Français": """À l’avenir, la gestion de l’eau à Bahreïn sera façonnée par l’innovation technologique, les réformes politiques et le changement de comportement..."""
            },
            "video": {
                "English": "https://www.youtube.com/watch?v=4rO4pYlQH5M",
                "العربية": "https://www.youtube.com/watch?v=mi_K7eLNz_M",
                "Français": "https://www.youtube.com/watch?v=zVZ2iK2dJdM"
            }
        }
    }

    total_modules = len(adult_modules)

    if audience == "Adult":
        current = adult_modules[st.session_state.module]
        st.header(current["title"][lang])
        st.write(current["content"][lang])
        st.video(current["video"][lang])

        # Mark module as completed
        if st.button("✅ I finished this module"):
            st.session_state.video_completed = True

        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Back") and st.session_state.module > 1:
                st.session_state.module -= 1
                st.session_state.video_completed = False
        with col2:
            if st.button("➡️ Next") and st.session_state.video_completed and st.session_state.module < total_modules:
                st.session_state.module += 1
                st.session_state.video_completed = False

        # Completion message
        if st.session_state.module == total_modules and st.session_state.video_completed:
            st.balloons()
            st.success("🎉 Congratulations! You completed the WaterGuard Adult Course.")

    else:  # Kid audience
        st.subheader("🌟 Fun Water Saving Tips for Kids")
        st.write("""
        - Don’t leave the tap running when washing your hands.  
        - Take short showers instead of baths.  
        - Remind parents to fix leaks quickly.  
        - Use a bucket to water plants instead of a hose.  
        """)
        st.video("https://www.youtube.com/watch?v=5J3cw4biWWo")
        st.video("https://www.youtube.com/watch?v=nTcFXJT0Fsc")

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                


