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
import plotly.express as px
import time

sns.set_style("whitegrid")
st.set_page_config(page_title="WaterGuard", layout="wide")

# Initialize session state for language if not already set
if "lang" not in st.session_state:
    st.session_state.lang = "en" # Default to English

# ---------- LANGUAGE TOGGLE ---------- #
st.sidebar.title("Settings" if st.session_state.lang == "en" else "الإعدادات")
language_selection = st.sidebar.radio("🌐 Language / اللغة", ["English", "العربية"])
if language_selection == "العربية":
    st.session_state.lang = "ar"
else:
    st.session_state.lang = "en"

lang = st.session_state.lang # Use the session state for lang

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
intro_text_en = """
💧 WaterGuard Prototype
WaterGuard is a smart AI-powered water monitoring prototype built for a residential home in Saar. It tracks daily water usage, detects abnormal spikes, and provides real-time alerts to help homeowners save water and reduce costs. Additionally, it provides predictive insights on water usage, alerts for potential leaks before they occur, detailed visualizations for every hour, and practical water conservation tips tailored for your household. With these features, residents can actively reduce waste, optimize water consumption, and contribute to a more sustainable environment.
"""
intro_text_ar = """
💧 نموذج ووتر جارد
ووتر جارد هو نموذج ذكي لمراقبة استهلاك المياه في منزل سكني بمنطقة سار. يتتبع الاستهلاك اليومي، ويكشف الزيادات غير الطبيعية، ويقدم تنبيهات فورية لمساعدة السكان على حفظ المياه وخفض التكاليف. كما يوفر رؤى تنبؤية عن الاستهلاك، وتنبيهات للتسربات المحتملة قبل حدوثها، ورسوم بيانية مفصلة لكل ساعة، ونصائح عملية للحفاظ على المياه مصممة خصيصًا لمنزلك. مع هذه الميزات، يمكن للمقيمين تقليل الهدر، وتحسين استهلاك المياه، والمساهمة في بيئة أكثر استدامة.
"""

if lang == "en":
    st.markdown(f"""
    <div style="
        background: rgba(255,255,255,0.85);
        padding: 2rem;
        border-radius: 15px;
        max-width: 900px;
        margin: 3rem auto;
        color: #111;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    ">
        <h1 style="color: #023e8a; font-weight: 700;">{intro_text_en}</h1>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div style="
        background: rgba(255,255,255,0.85);
        padding: 2rem;
        border-radius: 15px;
        max-width: 900px;
        margin: 3rem auto;
        color: #111;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        direction: rtl;
        text-align: right;
    ">
        <h1 style="color: #023e8a; font-weight: 700;">{intro_text_ar}</h1>
    </div>
    """, unsafe_allow_html=True)

# ---------- BAHRAIN WATER HISTORY & FUTURE ---------- #
bahrain_water_text_en = """
Bahrain's relationship with water is ancient and multifaceted. Historically, freshwater in the archipelago was scarce; communities relied on shallow groundwater lenses, seasonal wadis on the larger islands of the Gulf, and simple rain-capture techniques. Over centuries, Bahrain's small area and limited freshwater resources shaped settlement patterns, agriculture, and trade. Traditional systems—such as hand-dug wells and small networks for date palm irrigation—were central to village life. During the mid-20th century, rising population and urbanization placed heavier demands on limited groundwater reserves, and salinization from over-pumping became an increasing concern.

By the later decades of the 20th century, Bahrain adopted large-scale technological responses: desalination and modern water distribution infrastructure. Desalination plants enabled urban growth and industrial development by providing a reliable supply of potable water. However, desalination introduces challenges: energy intensity, brine disposal, and long-term costs. Bahrain's small size means national strategies can be targeted and implemented quickly, but must balance costs with sustainable resource use.

Looking forward, Bahrain's water future will be shaped by efficiency, diversification, and technology. Water conservation programs, improvements in leak detection and metering—exactly the benefits that WaterGuard targets—are critical. Investing in renewables to power desalination or employing more energy-efficient desalination technologies can reduce the environmental footprint. Treated wastewater reuse for irrigation and industry can lower freshwater demand, while smart-city initiatives and advanced monitoring will help optimize distribution networks. Climate change and regional groundwater pressures make integrated water resource management essential; policies that combine demand reduction, reuse, and innovative supply solutions will be decisive. Community engagement and household-level solutions—such as smart leak detection, efficient appliances, and behavioral change—remain among the most cost-effective and immediate measures to secure Bahrain's water resilience.
"""

bahrain_water_text_ar = """
العلاقة بين البحرين والمياه قديمة ومتعددة الأوجه. تاريخيًا، كانت المياه العذبة نادرة؛ اعتمدت المجتمعات على المياه الجوفية الضحلة، والأودية الموسمية على الجزر الأكبر في الخليج، وتقنيات جمع المطر البسيطة. على مدى قرون، شكلت مساحة البحرين الصغيرة وموارد المياه المحدودة أنماط الاستيطان والزراعة والتجارة. كانت الأنظمة التقليدية—مثل الآبار المحفورة يدويًا والشبكات الصغيرة لري نخيل التمر—أساسية في حياة القرى. خلال منتصف القرن العشرين، أدى النمو السكاني والتحضر إلى زيادة الضغط على المخزونات الجوفية المحدودة، وأصبح ملوحة المياه من الضخ المفرط مصدر قلق متزايد.

بحلول العقود الأخيرة من القرن العشرين، اعتمدت البحرين استجابات تكنولوجية واسعة النطاق: التحلية وبنية تحتية حديثة لتوزيع المياه. مكّنت محطات التحلية النمو العمراني والتطوير الصناعي من خلال توفير إمدادات موثوقة من المياه الصالحة للشرب. ومع ذلك، فإن التحلية تطرح تحديات: كثافة الطاقة، والتخلص من الملوحة، والتكاليف طويلة المدى. تعني صغر مساحة البحرين أن الاستراتيجيات الوطنية يمكن تنفيذها بسرعة، ولكن يجب موازنة التكاليف مع الاستخدام المستدام للموارد.

مستقبلاً، ستتحدد مياه البحرين بالكفاءة، والتنويع، والتكنولوجيا. برامج ترشيد المياه، وتحسين كشف التسرب والعدادات—وهي نفس الفوائد التي يستهدفها ووتر جارد—ضرورية. يمكن أن يقلل الاستثمار في الطاقة المتجددة لتشغيل التحلية أو استخدام تقنيات تحلية أكثر كفاءة في الطاقة من البصمة البيئية. يمكن أن يقلل إعادة استخدام مياه الصرف المعالجة للري والصناعة الطلب على المياه العذبة، بينما تساعد مبادرات المدن الذكية والمراقبة المتقدمة في تحسين شبكات التوزيع. يجعل تغير المناخ وضغوط المياه الجوفية الإقليمية إدارة الموارد المائية المتكاملة أمرًا أساسيًا؛ وستكون السياسات التي تجمع بين الحد من الطلب، وإعادة الاستخدام، وحلول الإمداد المبتكرة حاسمة. يبقى انخراط المجتمع وحلول مستوى المنزل—مثل الكشف الذكي عن التسرب، والأجهزة الفعالة، وتغيير السلوك—من بين أكثر الإجراءات فعالية من حيث التكلفة وفورية لضمان مرونة مياه البحرين.
"""

if lang == "en":
    st.markdown(f"""
    <div style="
        background: rgba(255,255,255,0.85);
        padding: 2rem;
        border-radius: 15px;
        max-width: 900px;
        margin: 2rem auto;
        color: #111;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    ">
        <h2 style="color:#023e8a; font-weight:700;">💧 Bahrain Water: History & Future</h2>
        <p style="font-size:1rem; line-height:1.6; color:#111;">{bahrain_water_text_en}</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div style="
        background: rgba(255,255,255,0.85);
        padding: 2rem;
        border-radius: 15px;
        max-width: 900px;
        margin: 2rem auto;
        color: #111;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        direction: rtl;
        text-align: right;
    ">
        <h2 style="color:#023e8a; font-weight:700;">💧 البحرين والمياه: الماضي والمستقبل</h2>
        <p style="font-size:1rem; line-height:1.6; color:#111;">{bahrain_water_text_ar}</p>
    </div>
    """, unsafe_allow_html=True)

# ---------- FAQ SECTION ---------- #
faq_en = [
    ("How can I detect a water leak early?", "Use WaterGuard's anomaly detection alerts to spot unusual spikes."),
    ("What should I do if an anomaly is detected?", "Check for leaks or unusual water usage immediately."),
    ("Can WaterGuard monitor multiple locations?", "Yes, it supports tracking usage across various branches or sites."),
    ("How accurate is the anomaly detection?", "The system uses AI to detect 95% of irregular water usage patterns."),
    ("Is WaterGuard suitable for factories with large consumption?", "Yes, it manages high-volume water use and alerts for excess."),
    ("How often is water usage data updated?", "Data is updated hourly for precise monitoring and alerts."),
    ("Can I download daily usage reports?", "Yes, downloadable CSV reports are available for any selected day."),
    ("What cost savings can I expect?", "Early leak detection and usage optimization significantly reduce bills."),
    ("Does WaterGuard support multiple languages?", "Currently supports English and Arabic interfaces."),
    ("Who do I contact for technical support?", "Contact support@waterguard.bh for all maintenance and help queries.")
]

faq_ar = [
    ("كيف يمكنني اكتشاف تسرب المياه مبكرًا؟", "استخدم تنبيهات اكتشاف الشذوذ في ووتر جارد لملاحظة الزيادات غير الطبيعية."),
    ("ماذا أفعل إذا تم اكتشاف شذوذ؟", "تحقق فورًا من التسربات أو الاستهلاك غير الطبيعي للمياه."),
    ("هل يمكن لووتر جارد مراقبة مواقع متعددة؟", "نعم، يدعم تتبع الاستخدام عبر فروع أو مواقع مختلفة."),
    ("ما مدى دقة اكتشاف الشذوذ؟", "يستخدم النظام الذكاء الاصطناعي لاكتشاف 95% من أنماط استخدام المياه غير الطبيعية."),
    ("هل ووتر جارد مناسب للمصانع ذات الاستهلاك الكبير؟", "نعم، يدير الاستخدام العالي للمياه وينبه عند الزيادة."),
    ("كم مرة يتم تحديث بيانات استهلاك المياه؟", "يتم تحديث البيانات كل ساعة لمراقبة دقيقة وتنبيهات."),
    ("هل يمكنني تنزيل تقارير الاستخدام اليومية؟", "نعم، تتوفر تقارير CSV لأي يوم محدد."),
    ("ما مقدار التوفير المتوقع؟", "يساعد الاكتشاف المبكر والتحسين على تقليل الفواتير بشكل كبير."),
    ("هل يدعم ووتر جارد لغات متعددة؟", "يدعم حاليًا الواجهات الإنجليزية والعربية."),
    ("من يمكنني الاتصال به للدعم الفني؟", "اتصل بـ support@waterguard.bh لجميع الاستفسارات والصيانة.")
]

faq_data = faq_en if lang == "en" else faq_ar

faq_html = ""
for q, a in faq_data:
    faq_html += f"<p><strong>{q}</strong><br>{a}</p>"

st.markdown(f"""
<div style="
    background: rgba(255,255,255,0.9);
    padding: 2rem;
    border-radius: 15px;
    max-width: 900px;
    margin: 2rem auto;
    color: #111;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    line-height:1.6;
">
    <h2 style="color:#023e8a; font-weight:700;">💧 {'WaterGuard FAQ' if lang=='en' else 'الأسئلة الشائعة'}</h2>
    {faq_html}
</div>
""", unsafe_allow_html=True)
