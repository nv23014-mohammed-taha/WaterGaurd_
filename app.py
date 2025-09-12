# -*- coding: utf-8 -*-

"""
WaterGuard App - A comprehensive water monitoring and educational prototype.

This app features:
- A multi-tabbed interface for a dashboard, a course, and historical context.
- Language switching between English, Arabic, and French.
- Simulated real-time water usage data.
- Anomaly detection using the IsolationForest machine learning model.
- An educational course with quizzes and a reward system.
- Historical context about water in Bahrain.
- Responsive design and accessibility features.
"""

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
import json

# Set a consistent style for plots and the page config
sns.set_style("whitegrid")
st.set_page_config(page_title="WaterGuard", layout="wide")

# Custom CSS for button styling and other visual enhancements
st.markdown("""
<style>
.stApp {
    color: #f0f0f0;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.stButton>button {
    background-color: black;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: bold;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: all 0.2s ease;
}
.stButton>button:hover {
    background-color: #333;
    color: #fff;
    transform: translateY(-2px);
    box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
}
.testimonial-card {
    background: rgba(255, 255, 255, 0.9);
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    color: #000;
}
.testimonial-profile {
    display: flex;
    align-items: center;
    margin-top: 10px;
}
.testimonial-profile .emoji {
    font-size: 2rem;
    margin-right: 10px;
}
.faq-answer {
    color: #000;
}
.anomaly-alert {
    background-color: #fcebeb;
    color: #9f2a2a;
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid #f5c6cb;
    margin-top: 1rem;
    font-weight: bold;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ----------------------------
# Session state initial setup
# ----------------------------

# Set default language to English
if "lang" not in st.session_state:
    st.session_state.lang = "en"

# Initialize course and rewards state variables
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

title_text = {
    "en": "Settings",
    "ar": "الإعدادات",
    "fr": "Paramètres"
}
st.sidebar.title(title_text[st.session_state.lang])

language_selection = st.sidebar.radio(
    "🌐 Language / اللغة / Langue",
    ["English", "العربية", "Français"]
)

if language_selection == "العربية":
    st.session_state.lang = "ar"
elif language_selection == "Français":
    st.session_state.lang = "fr"
else:
    st.session_state.lang = "en"

lang = st.session_state.lang # Convenience variable for current language

# ----------------------------
# SCREEN READER BUTTON (fixed)
# ----------------------------
def screen_reader_button(lang_local):
    """Generates a button to activate a basic screen reader."""
    lang_codes = {
        "en": "en-US",
        "ar": "ar-SA",
        "fr": "fr-FR"
    }
    button_texts = {
        "en": "🔊 Activate Screen Reader",
        "ar": "🔊 تشغيل قارئ الشاشة",
        "fr": "🔊 Activer le lecteur d'écran"
    }
    lang_code = lang_codes.get(lang_local, "en-US")
    button_text = button_texts.get(lang_local, "🔊 Activate Screen Reader")

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
        {'margin-left: auto;' if lang_local in ['en', 'fr'] else 'margin-right: auto;'}
    ">
    {button_text}
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
    """Sets a full-screen background image."""
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
        pass # Ignore if image file is not found

# Make sure you have 'water_bg.jpg' in the same directory as this script.
set_background("water_bg.jpg")

# ----------------------------
# COURSE & BAHRAIN HISTORY CONTENT
# ----------------------------

REWARD_FILS_PER_QUIZ = 500  # 500 fils
REWARD_BHD_PER_QUIZ = REWARD_FILS_PER_QUIZ / 1000.0

# Course definition
COURSE = [
    {
        "title_en": "Intro: Why Water Monitoring Matters (5 min)",
        "title_ar": "مقدمة: لماذا تهم مراقبة المياه (5 دقائق)",
        "title_fr": "Intro: Pourquoi la surveillance de l'eau est importante (5 min)",
        "minutes": 5,
        "content_en": ("Why household water monitoring is important: cost savings, leak "
                       "prevention, and sustainability. How small behavioral changes lead to significant savings."),
        "content_ar": ("لماذا تُعد مراقبة المياه المنزلية مهمة: توفير التكاليف، منع التسرب، "
                       "والاستدامة. كيف تؤدي التغييرات الصغيرة في السلوك إلى وفورات كبيرة."),
        "content_fr": ("Pourquoi la surveillance de l'eau à domicile est importante : économies, "
                       "prévention des fuites et durabilité. Comment de petits changements de comportement "
                       "peuvent entraîner des économies importantes."),
        "quiz": [
            {
                "q_en": "Which is a direct benefit of early leak detection?",
                "q_ar": "ما هي فائدة الكشف المبكر عن التسرب؟",
                "q_fr": "Quel est un avantage direct de la détection précoce des fuites ?",
                "options": ["Higher bills", "Increased water waste", "Lower repair costs", "More humid air"],
                "options_ar": ["فواتير أعلى", "زيادة هدر المياه", "تكاليف إصلاح أقل", "هواء أكثر رطوبة"],
                "options_fr": ["Factures plus élevées", "Gaspillage d'eau accru", "Coûts de réparation réduits", "Air plus humide"],
                "answer": 2
            }
        ]
    },
    {
        "title_en": "How WaterGuard Detects Anomalies (8 min)",
        "title_ar": "كيف يكتشف ووتر جارد الأنماط الشاذة (8 دقائق)",
        "title_fr": "Comment WaterGuard détecte les anomalies (8 min)",
        "minutes": 8,
        "content_en": ("Overview of sensors, hourly data, anomaly detection models (e.g., IsolationForest), "
                       "and how thresholds & severity are set."),
        "content_ar": ("نظرة عامة على الحساسات، البيانات الساعية، نماذج اكتشاف الخلل (مثل IsolationForest)، "
                       "وكيف يتم ضبط العتبات وحدود الشدة."),
        "content_fr": ("Aperçu des capteurs, des données horaires, des modèles de détection d'anomalies (par ex., IsolationForest), "
                       "et comment les seuils et la gravité sont définis."),
        "quiz": [
            {
                "q_en": "Which model is used in this prototype for anomaly detection?",
                "q_ar": "أي نموذج تم استخدامه في هذا النموذج لاكتشاف الخلل؟",
                "q_fr": "Quel modèle est utilisé dans ce prototype pour la détection des anomalies ?",
                "options": ["KMeans", "IsolationForest", "Linear Regression", "PCA"],
                "options_ar": ["KMeans", "IsolationForest", "الانحدار الخطي", "PCA"],
                "options_fr": ["KMeans", "IsolationForest", "Régression linéaire", "ACP"],
                "answer": 1
            },
            {
                "q_en": "A severity labeled 'High' likely indicates:",
                "q_ar": "ماذا تعني شدة 'عالية' عادةً؟",
                "q_fr": "Une gravité étiquetée 'Élevée' indique probablement :",
                "options": ["Very low usage", "Normal usage", "Very high usage", "No data"],
                "options_ar": ["استهلاك منخفض جدًا", "استهلاك طبيعي", "استهلاك مرتفع جدًا", "لا توجد بيانات"],
                "options_fr": ["Consommation très faible", "Consommation normale", "Consommation très élevée", "Pas de données"],
                "answer": 2
            }
        ]
    },
    {
        "title_en": "Practical Tips & Fixes (7 min)",
        "title_ar": "نصائح عملية وإصلاحات (7 دقائق)",
        "title_fr": "Conseils pratiques et réparations (7 min)",
        "minutes": 7,
        "content_en": ("Simple checks: fixture inspections, irrigation schedules, fixture replacement "
                       "recommendations, and behavioral tips to minimize waste."),
        "content_ar": ("فحوصات بسيطة: التحقق من التركيبات، جداول الري، توصيات استبدال التركيبات، "
                       "ونصائح سلوكية لتقليل الهدر."),
        "content_fr": ("Vérifications simples : inspection des installations, calendriers d'irrigation, "
                       "recommandations de remplacement d'appareils, et conseils de comportement pour "
                       "minimiser le gaspillage."),
        "quiz": [
            {
                "q_en": "Which action helps most to reduce garden overwatering?",
                "q_ar": "أي إجراء يساعد أكثر على تقليل الري الزائد للحديقة؟",
                "q_fr": "Quelle action aide le plus à réduire l'excès d'arrosage du jardin ?",
                "options": ["Run sprinklers more often", "Shorten irrigation intervals", "Schedule irrigation early morning", "Water during hottest hour"],
                "options_ar": ["تشغيل الرشاشات بشكل متكرر", "تقصير فترات الري", "جدولة الري في الصباح الباكر", "الري في أشد ساعات الحر"],
                "options_fr": ["Arroser plus souvent", "Raccourcir les intervalles d'irrigation", "Prévoir l'arrosage tôt le matin", "Arroser pendant l'heure la plus chaude"],
                "answer": 2
            }
        ]
    },
    {
        "title_en": "Reading Reports & Using Insights (5 min)",
        "title_ar": "قراءة التقارير واستخدام الرؤى (5 دقائق)",
        "title_fr": "Lecture des rapports et utilisation des informations (5 min)",
        "minutes": 5,
        "content_en": ("How to read hourly/daily/monthly visualizations, export CSV, and act on detected trends."),
        "content_ar": ("كيفية قراءة الرسوم البيانية الساعية/اليومية/الشهرية، تصدير CSV، واتخاذ إجراءات بناءً على الاتجاهات المكتشفة."),
        "content_fr": ("Comment lire les visualisations horaires/quotidiennes/mensuelles, exporter au format CSV, "
                       "et agir sur les tendances détectées."),
        "quiz": [
            {
                "q_en": "If daily usage spikes repeatedly at night, what is the first thing to check?",
                "q_ar": "إذا تكررت زيادات الاستهلاك اليومية ليلاً، ما هو أول شيء يجب التحقق منه؟",
                "q_fr": "Si la consommation quotidienne augmente à plusieurs reprises la nuit, que faut-il vérifier en premier lieu ?",
                "options": ["Kitchen sink", "Garden irrigation / sprinkler", "Cooking routines", "Battery level"],
                "options_ar": ["حوض المطبخ", "ري الحديقة / الرشاش", "روتين الطبخ", "مستوى البطارية"],
                "options_fr": ["L'évier de la cuisine", "L'irrigation du jardin / l'arroseur", "Les routines de cuisine", "Le niveau de batterie"],
                "answer": 1
            }
        ]
    }
]

# Bahrain water history content
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

في المستقبل، يتعين على البحرين التركيز على الكفاءة والتنويع والتقنيات الحديثة. تتضمن الحلول
تحسين الكشف عن التسريبات والقياس الدقيق للمستهلكين (مثل الحلول التي يقدمها ووتر جارد)،
إعادة استخدام المياه المعالجة للري والصناعة، واستخدام مصادر طاقة متجددة لتقليل بصمة التحلية.
مع تبعات تغير المناخ وضغوط المياه الإقليمية، يصبح التوازن بين تقليل الطلب وإدارة الموارد
بشكل متكامل أمرًا حاسمًا للحفاظ على الأمن المائي.
""".strip()

BAHRAIN_HISTORY_FR = """
La relation de Bahreïn avec l'eau est ancienne et multiforme. Historiquement, l'eau douce
était rare dans l'archipel ; les communautés dépendaient des nappes phréatiques peu profondes,
des oueds saisonniers sur les plus grandes îles du Golfe et de techniques simples de collecte
de la pluie. Au fil des siècles, la petite superficie de Bahreïn et ses ressources en eau
douce limitées ont façonné les modèles de peuplement, l'agriculture et le commerce. Les
systèmes traditionnels — tels que les puits creusés à la main et les petits réseaux pour
l'irrigation des palmiers-dattiers — étaient essentiels à la vie des villages. Au milieu du
XXe siècle, la croissance démographique et l'urbanisation ont exercé une pression accrue sur
les réserves limitées d'eau souterraine, et la salinisation due au pompage excessif est
devenue une préoccupation croissante.

Au cours des dernières décennies du XXe siècle, Bahreïn a adopté des réponses technologiques
à grande échelle : le dessalement et une infrastructure de distribution d'eau moderne. Les
usines de dessalement ont permis la croissance urbaine et le développement industriel en
fournissant un approvisionnement fiable en eau potable. Cependant, le dessalement présente
des défis : forte consommation d'énergie, élimination de la saumure et coûts à long terme.
La petite taille de Bahreïn signifie que les stratégies nationales peuvent être ciblées et
mises en œuvre rapidement, mais doivent équilibrer les coûts et l'utilisation durable des
ressources.

À l'avenir, le futur de l'eau à Bahreïn sera façonné par l'efficacité, la diversification et
la technologie. Les programmes de conservation de l'eau, les améliorations de la détection
des fuites et du comptage — exactement les avantages ciblés par WaterGuard — sont essentiels.
Investir dans les énergies renouvelables pour alimenter le dessalement ou employer des
technologies de dessalement plus écoénergétiques peut réduire l'empreinte environnementale.
La réutilisation des eaux usées traitées pour l'irrigation et l'industrie peut réduire la
demande en eau douce, tandis que les initiatives de villes intelligentes et la surveillance
avancée aideront à optimiser les réseaux de distribution. Le changement climatique et les
pressions régionales sur les eaux souterraines rendent la gestion intégrée des ressources
en eau indispensable ; les politiques qui combinent la réduction de la demande, la
réutilisation et des solutions d'approvisionnement innovantes seront décisives.
L'engagement communautaire et les solutions au niveau domestique — telles que la détection
intelligente des fuites, les appareils efficaces et les changements de comportement —
restent parmi les mesures les plus rentables et les plus immédiates pour assurer la
résilience de Bahreïn en matière d'eau.
""".strip()

# Testimonial data
testimonial_data = {
    "en": [
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
    ],
    "ar": [
        "💡 ووتر جارد ساعدني في اكتشاف تسرب مخفي — ووفّر لي 12 دينارًا هذا الشهر!",
        "✅ التنبيهات دقيقة للغاية. تم إخطاري قبل أن يتفاقم التسرب الخطير.",
        "📈 أحب الرسوم البيانية للاستهلاك. تجعلني على دراية بسلوكنا اليومي في استخدام المياه.",
        "💧 وجدنا أن نظام رشاش الحديقة كان يروي أكثر من اللازم — وتم إصلاحه الآن!",
        "🏡 رائع للمنازل ذات العائلات الكبيرة — يساعد على تجنب الفواتير المرتفعة.",
        "📊 قمت بتحميل تقرير وشاركته مع مالك العقار. احترافي جداً!",
        "📱 لوحة التحكم نظيفة وسهلة الاستخدام. حتى أطفالي يفهمونها!",
        "🔔 ساعدتني التنبيهات الفورية على إيقاف هدر المياه أثناء السفر.",
        "🧠 لم أكن أعلم أبدًا كمية المياه التي يستهلكها المطبخ حتى أظهر لي ووتر جارد.",
        "🌱 نحن الآن أكثر وعيًا بيئيًا بفضل نصائح ورؤى ووتر جارد."
    ],
    "fr": [
        "💡 WaterGuard m'a aidé à découvrir une fuite cachée — j'ai économisé 12 BHD ce mois-ci !",
        "✅ Les alertes sont très précises. J'ai été prévenu avant qu'une fuite sérieuse ne s'aggrave.",
        "📈 J'adore les graphiques de consommation. Cela me rend conscient de notre comportement quotidien vis-à-vis de l'eau.",
        "💧 Nous avons découvert que notre système d'arrosage de jardin arrosait trop — c'est maintenant réparé !",
        "🏡 Idéal pour les familles nombreuses — cela aide à éviter les factures élevées.",
        "📊 J'ai téléchargé un rapport et je l'ai partagé avec mon propriétaire. Très professionnel !",
        "📱 Le tableau de bord est propre et facile à utiliser. Même mes enfants le comprennent !",
        "🔔 Les alertes en temps réel m'ont aidé à arrêter le gaspillage d'eau pendant un voyage.",
        "🧠 Je n'ai jamais su à quel point la cuisine consommait jusqu'à ce que WaterGuard me le montre.",
        "🌱 Nous sommes maintenant plus éco-conscients grâce aux conseils et aux informations de WaterGuard."
    ]
}

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


# ----------------------------
# Core app content (existing) - Data simulation + analysis
# ----------------------------

@st.cache_data
def simulate_data():
    """Generates a year of simulated hourly water usage data with anomalies."""
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
        df_local.loc[i, ['usage_main_liters', 'usage_garden_liters',
                         'usage_kitchen_liters', 'usage_bathroom_liters']] *= np.random.uniform(2, 5)

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

# Severity classification
df['severity'] = pd.cut(df['usage_liters'],
                        bins=[-np.inf, 20, 40, np.inf],
                        labels=['Low', 'Medium', 'High'])

# ----------------------------
# Top tabs: Course, Bahrain History, Dashboard
# ----------------------------
tab_labels = {
    "en": ["Course", "Bahrain Water", "Dashboard"],
    "ar": ["الدورة التدريبية", "تاريخ المياه في البحرين", "لوحة التحكم"],
    "fr": ["Cours", "Eau à Bahreïn", "Tableau de bord"]
}

top_tabs = st.tabs(tab_labels[lang])

# ----------------------------
# Course Tab
# ----------------------------
with top_tabs[0]:
    header_text = {
        "en": "💡 WaterGuard — 30 Minute Course",
        "ar": "💡ووتر جارد — دورة 30 دقيقة",
        "fr": "💡 WaterGuard — Cours de 30 minutes"
    }
    st.header(header_text[lang])

    # Progress indicator
    progress_fraction = st.session_state.course_progress / len(COURSE) if len(COURSE) > 0 else 0
    st.progress(min(max(progress_fraction, 0.0), 1.0))

    # Display modules list
    modules_heading = {
        "en": "### Modules",
        "ar": "### الوحدات",
        "fr": "### Modules"
    }
    st.markdown(modules_heading[lang])
    module_titles = [(m[f"title_{lang}"] if lang in m else m["title_en"]) for m in COURSE]
    status_texts = {
        "en": {"completed": "✅ Completed", "current": "▶ Current"},
        "ar": {"completed": "✅ مكتملة", "current": "▶الحالية"},
        "fr": {"completed": "✅ Terminé", "current": "▶ Actuel"}
    }
    for idx, t in enumerate(module_titles):
        status = ""
        if idx < st.session_state.course_progress:
            status = status_texts[lang]["completed"]
        elif idx == st.session_state.current_module:
            status = status_texts[lang]["current"]
        st.write(f"{idx+1}. {t} {status}")

    module_idx = st.session_state.current_module
    module = COURSE[module_idx]

    st.subheader(module[f"title_{lang}"])
    st.write(module[f"content_{lang}"])
    
    estimated_time_text = {
        "en": f"*Estimated time: {module['minutes']} min*",
        "ar": f"*الوقت المقدر: {module['minutes']} دقيقة*",
        "fr": f"*Temps estimé : {module['minutes']} min*"
    }
    st.write(estimated_time_text[lang])

    # Mark module complete button (progress only)
    mark_button_text = {
        "en": "Mark module complete",
        "ar": "تحديد الوحدة كمكتملة",
        "fr": "Marquer le module comme terminé"
    }
    success_message = {
        "en": "Module marked complete.",
        "ar": "تم تحديد الوحدة كمكتملة.",
        "fr": "Module marqué comme terminé."
    }
    if st.button(mark_button_text[lang]):
        st.session_state.course_progress = max(st.session_state.course_progress, module_idx + 1)
        st.success(success_message[lang])
        st.rerun()

    # Quiz UI for current module
    if module.get("quiz"):
        quiz_heading = {
            "en": "### Quiz",
            "ar": "### الاختبار",
            "fr": "### Quiz"
        }
        st.markdown(quiz_heading[lang])
        answers = {}
        for qi, q in enumerate(module["quiz"]):
            question_text = q[f"q_{lang}"] if lang in q else q["q_en"]
            opts = q.get(f"options_{lang}", q["options"])
            choice = st.radio(f"{qi+1}. {question_text}", opts, key=f"quiz_{module_idx}_{qi}")
            answers[qi] = opts.index(choice)

        submit_quiz_text = {
            "en": "Submit Quiz",
            "ar": "إرسال الاختبار",
            "fr": "Soumettre le quiz"
        }
        if st.button(submit_quiz_text[lang]):
            total = len(module["quiz"])
            correct = 0
            for i_q, q_def in enumerate(module["quiz"]):
                if answers.get(i_q) == q_def["answer"]:
                    correct += 1
            score_pct = (correct / total) * 100 if total > 0 else 0
            st.session_state.quiz_scores[module_idx] = {"correct": correct, "total": total, "pct": score_pct}
            passed = score_pct >= 80  # Pass threshold 80%

            if passed:
                success_message = {
                    "en": f"Passed — Score: {score_pct:.0f}% — Reward earned: {REWARD_FILS_PER_QUIZ} fils (BHD {REWARD_BHD_PER_QUIZ:.3f})",
                    "ar": f"ناجح — النسبة: {score_pct:.0f}% — جائزة: {REWARD_FILS_PER_QUIZ} فلس (ب.د {REWARD_BHD_PER_QUIZ:.3f})",
                    "fr": f"Réussi — Score : {score_pct:.0f}% — Récompense gagnée : {REWARD_FILS_PER_QUIZ} fils (BHD {REWARD_BHD_PER_QUIZ:.3f})"
                }
                st.success(success_message[lang])
                quiz_name = f"module_{module_idx}"
                if quiz_name not in st.session_state.completed_quizzes:
                    st.session_state.completed_quizzes.append(quiz_name)
                    st.session_state.rewards += REWARD_FILS_PER_QUIZ
                    st.session_state.reward_claimed[module_idx] = True
            else:
                warning_message = {
                    "en": f"Not passed — Score: {score_pct:.0f}%. Try again!",
                    "ar": f"لم تجتز — النسبة: {score_pct:.0f}%. حاول مرة أخرى!",
                    "fr": f"Échoué — Score : {score_pct:.0f}%. Essayez encore !"
                }
                st.warning(warning_message[lang])

    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        prev_button_text = {
            "en": "Previous module",
            "ar": "الوحدة السابقة",
            "fr": "Module précédent"
        }
        if st.button(prev_button_text[lang]):
            if st.session_state.current_module > 0:
                st.session_state.current_module -= 1
                st.rerun()
    with col2:
        next_button_text = {
            "en": "Next module",
            "ar": "الوحدة التالية",
            "fr": "Module suivant"
        }
        if st.button(next_button_text[lang]):
            if st.session_state.current_module < len(COURSE) - 1:
                st.session_state.current_module += 1
                st.rerun()

    # Certificate + rewards summary when complete
    if st.session_state.course_progress >= len(COURSE):
        st.balloons()
        course_complete_text = {
            "en": "Congratulations! You have successfully completed the WaterGuard Course.",
            "ar": "تهانينا! لقد أكملت دورة ووتر جارد بنجاح.",
            "fr": "Félicitations ! Vous avez terminé avec succès le cours WaterGuard."
        }
        st.success(course_complete_text[lang])

        rewards_summary_heading = {
            "en": "### Rewards Summary",
            "ar": "### ملخص الجوائز",
            "fr": "### Résumé des récompenses"
        }
        st.markdown(rewards_summary_heading[lang])

        total_earned_text = {
            "en": f"Total earned: {st.session_state.rewards} fils (BHD {st.session_state.rewards/1000.0:.3f})",
            "ar": f"المجموع المكتسب: {st.session_state.rewards} فلس (ب.د {st.session_state.rewards/1000.0:.3f})",
            "fr": f"Total gagné : {st.session_state.rewards} fils (BHD {st.session_state.rewards/1000.0:.3f})"
        }
        st.write(total_earned_text[lang])

        cert_text = f"WaterGuard Course Certificate\nUser: demo_user@example.com\nCompleted: YES\nScore Summary: {json.dumps(st.session_state.quiz_scores)}\nRewards (fils): {st.session_state.rewards}"
        download_cert_text = {
            "en": "Download Certificate (TXT)",
            "ar": "تحميل الشهادة (TXT)",
            "fr": "Télécharger le certificat (TXT)"
        }
        st.download_button(download_cert_text[lang], data=cert_text, file_name="waterguard_certificate.txt")

# ----------------------------
# Bahrain History Tab
# ----------------------------
with top_tabs[1]:
    header_text = {
        "en": "Bahrain Water: History & Future",
        "ar": "تاريخ المياه في البحرين ومستقبلها",
        "fr": "L'eau à Bahreïn : Histoire et Avenir"
    }
    st.header(header_text[lang])

    if lang == "en":
        st.markdown(BAHRAIN_HISTORY_EN)
    elif lang == "ar":
        st.markdown(f"<div dir='rtl' style='text-align: right'>{BAHRAIN_HISTORY_AR}</div>", unsafe_allow_html=True)
    else: # French
        st.markdown(BAHRAIN_HISTORY_FR)

# ----------------------------
# Dashboard Tab (main app content)
# ----------------------------
with top_tabs[2]:
    # ---------- INTRO SECTION ----------
    intro_html = {
        "en": """
        <div style="background: rgba(255, 255, 255, 0.9); padding: 2rem;
        border-radius: 15px; max-width: 900px; margin: 1.5rem auto; color: #111;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15); font-family: 'Segoe UI', Tahoma,
        Geneva, Verdana, sans-serif;">
        <h1 style="color: #023e8a; font-weight: 700;">💧 WaterGuard Prototype</h1>
        <p style="font-size: 1.05rem; line-height: 1.5;">
        WaterGuard is a smart AI-powered water monitoring prototype built for a residential home in Saar. It tracks daily water usage, detects abnormal spikes, and provides real-time alerts to help homeowners save water and reduce costs. By analyzing consumption habits, the system can identify subtle anomalies that might indicate a hidden leak or a faulty appliance. The intuitive dashboard offers a comprehensive view of your usage, allowing you to make informed decisions and adopt more sustainable behaviors. WaterGuard is more than just a monitor; it is a partner in responsible water management, contributing to both your budget and the preservation of this vital resource.
        </p>
        </div>
        """,
        "ar": """
        <div style="background: rgba(255, 255, 255, 0.9); padding: 2rem;
        border-radius: 15px; max-width: 900px; margin: 1.5rem auto; color: #111; box-shadow:
        0 8px 20px rgba(0,0,0,0.15); font-family: 'Segoe UI', Tahoma, Geneva, Verdana,
        sans-serif; direction: rtl; text-align: right;">
        <h1 style="color: #023e8a; font-weight: 700;">💧نموذج ووتر جارد</h1>
        <p style="font-size: 1.05rem; line-height: 1.5;">
        ووتر جارد هو نموذج ذكي لمراقبة استهلاك المياه في منزل سكني بمنطقة سار. يستخدم الذكاء الاصطناعي لتحليل البيانات وكشف أي استهلاك غير طبيعي، مما يساعد على تقليل الهدر وخفض الفواتير. من خلال تحليل عادات الاستهلاك، يمكن للنظام تحديد الأنماط الشاذة الدقيقة التي قد تشير إلى تسرب مخفي أو جهاز معطل. توفر لوحة التحكم سهلة الاستخدام نظرة عامة شاملة على استهلاكك، مما يتيح لك اتخاذ قرارات مستنيرة واتباع سلوكيات أكثر استدامة. ووتر جارد هو أكثر من مجرد جهاز مراقبة؛ إنه شريك في إدارة المياه بمسؤولية، مما يساهم في ميزانيتك وفي الحفاظ على هذا المورد الحيوي.
        </p>
        </div>
        """,
        "fr": """
        <div style="background: rgba(255, 255, 255, 0.9); padding: 2rem;
        border-radius: 15px; max-width: 900px; margin: 1.5rem auto; color: #111;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15); font-family: 'Segoe UI', Tahoma,
        Geneva, Verdana, sans-serif;">
        <h1 style="color: #023e8a; font-weight: 700;">💧 Prototype WaterGuard</h1>
        <p style="font-size: 1.05rem; line-height: 1.5;">
        WaterGuard est un prototype de surveillance de l'eau intelligent alimenté par l'IA,
        conçu pour une maison résidentielle à Saar. Il suit la consommation quotidienne
        d'eau, détecte les pics anormaux et fournit des alertes en temps réel pour aider
        les propriétaires à économiser l'eau et à réduire les coûts. En analysant les habitudes de consommation, le système peut identifier des anomalies subtiles qui pourraient indiquer une fuite cachée ou un appareil défectueux. Le tableau de bord intuitif offre une vue d'ensemble de votre consommation, vous permettant de prendre des décisions éclairées et d'adopter des comportements plus durables. WaterGuard est plus qu'un simple moniteur ; c'est un partenaire dans la gestion responsable de l'eau, contribuant à la fois à votre budget et à la préservation de cette ressource vitale.
        </p>
        </div>
        """
    }
    st.markdown(intro_html[lang], unsafe_allow_html=True)

    # ---------- SIDEBAR SUMMARY ----------
    sidebar_texts = {
        "en": "📅 Select a day to view usage",
        "ar": "📅 اختر اليوم لعرض الاستهلاك",
        "fr": "📅 Sélectionnez un jour pour voir la consommation"
    }
    selected_day = st.sidebar.date_input(
        sidebar_texts[lang],
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

    sidebar_summary = {
        "en": f"""
        ## 💧 Daily Water Usage Summary
        **Date:** {selected_day}
        **Used:** {day_usage:,.0f} liters
        **Remaining:** {remaining:,.0f} liters
        **Quota:** {daily_quota} liters
        **Estimated Cost:** BHD {daily_cost:.3f}
        """,
        "ar": f"""
        ## 💧 ملخص استهلاك المياه اليومي
        **التاريخ:** {selected_day}
        **المستهلك:** {day_usage:,.0f} لتر
        **المتبقي:** {remaining:,.0f} لتر
        **الحصة اليومية:** {daily_quota} لتر
        **التكلفة التقديرية:** {daily_cost:.3f} دينار بحريني
        """,
        "fr": f"""
        ## 💧 Résumé de la consommation d'eau quotidienne
        **Date :** {selected_day}
        **Utilisé :** {day_usage:,.0f} litres
        **Restant :** {remaining:,.0f} litres
        **Quota :** {daily_quota} litres
        **Coût estimé :** BHD {daily_cost:.3f}
        """
    }
    st.sidebar.markdown(sidebar_summary[lang])

    st.sidebar.progress(min(usage_ratio, 1.0))

    # Alerts
    high_usage_threshold = daily_quota * 0.9
    if day_usage > high_usage_threshold:
        alert_message = {
            "en": "🚨 High water consumption detected today!",
            "ar": "🚨 تم الكشف عن استهلاك مياه مرتفع اليوم!",
            "fr": "🚨 Consommation d'eau élevée détectée aujourd'hui !"
        }
        st.sidebar.warning(alert_message[lang])

    # Anomalies table
    anomaly_heading = {
        "en": "## 🔍 Detected Anomalies (Possible Leaks or Spikes)",
        "ar": "## 🔍 الأنماط الشاذة المكتشفة (تسريبات أو زيادات محتملة)",
        "fr": "## 🔍 Anomalies détectées (fuites ou pics possibles)"
    }
    st.markdown(anomaly_heading[lang])

    expander_label = {
        "en": "Show Anomalies",
        "ar": "إظهار الأنماط الشاذة",
        "fr": "Afficher les anomalies"
    }
    with st.expander(expander_label[lang]):
        df_anomalies = df[df['anomaly'] == 'Anomaly']
        anomaly_display = df_anomalies[['timestamp', 'usage_liters', 'severity']].copy()
        anomaly_display['usage_liters'] = anomaly_display['usage_liters'].map(lambda x: f"{x:.2f}")
        anomaly_display['severity'] = anomaly_display['severity'].astype(str)
        st.dataframe(anomaly_display)
        csv_anomaly = anomaly_display.to_csv(index=False)
        download_button_label = {
            "en": "Download Anomalies CSV",
            "ar": "تحميل الأنماط الشاذة CSV",
            "fr": "Télécharger les anomalies CSV"
        }
        st.download_button(
            label=download_button_label[lang],
            data=csv_anomaly,
            file_name='waterguard_anomalies.csv',
            mime='text/csv'
        )

    # Usage visualization - hourly for selected day
    df['time_str'] = df['timestamp'].dt.strftime('%H:%M')
    df_day_hourly = df[df['date'] == selected_day]

    hourly_heading = {
        "en": f"## 📊 Hourly Water Usage for {selected_day}",
        "ar": f"## 📊 استهلاك المياه الساعي ليوم {selected_day}",
        "fr": f"## 📊 Consommation d'eau horaire pour le {selected_day}"
    }
    st.markdown(hourly_heading[lang])

    fig1, ax1 = plt.subplots(figsize=(14, 6))
    sns.lineplot(data=df_day_hourly, x='time_str', y='usage_liters', ax=ax1, label='Usage' if lang in ['en', 'fr'] else 'الاستهلاك')
    sns.scatterplot(data=df_day_hourly[df_day_hourly['anomaly'] == 'Anomaly'],
                    x='time_str', y='usage_liters',
                    color='red', marker='X', s=60, label='Anomaly' if lang in ['en', 'fr'] else 'خلل', ax=ax1)
    
    xlabel_text = {
        "en": "Time (HH:MM)",
        "ar": "الوقت (ساعة:دقيقة)",
        "fr": "Heure (HH:MM)"
    }
    ylabel_text = {
        "en": "Liters",
        "ar": "لتر",
        "fr": "Litres"
    }
    title_text_plot1 = {
        "en": f"Hourly Water Usage for {selected_day}",
        "ar": f"استهلاك المياه الساعي ليوم {selected_day}",
        "fr": f"Consommation d'eau horaire pour le {selected_day}"
    }

    ax1.set_xlabel(xlabel_text[lang])
    ax1.set_ylabel(ylabel_text[lang])
    ax1.set_title(title_text_plot1[lang])
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()
    st.pyplot(fig1)

    # Daily data for last year
    df_daily = df.set_index('timestamp').resample('D')['usage_liters'].sum().reset_index()
    daily_heading = {
        "en": "## 📈 Daily Water Usage (Past Year)",
        "ar": "## 📈 استهلاك المياه اليومي (السنة الماضية)",
        "fr": "## 📈 Consommation d'eau quotidienne (Année passée)"
    }
    st.markdown(daily_heading[lang])

    fig2, ax2 = plt.subplots(figsize=(14, 5))
    sns.lineplot(data=df_daily, x='timestamp', y='usage_liters', ax=ax2)

    xlabel_text2 = {
        "en": "Date",
        "ar": "التاريخ",
        "fr": "Date"
    }
    ylabel_text2 = {
        "en": "Liters",
        "ar": "لتر",
        "fr": "Litres"
    }
    title_text_plot2 = {
        "en": "Daily Water Usage",
        "ar": "استهلاك المياه اليومي",
        "fr": "Consommation d'eau quotidienne"
    }

    ax2.set_xlabel(xlabel_text2[lang])
    ax2.set_ylabel(ylabel_text2[lang])
    ax2.set_title(title_text_plot2[lang])
    ax2.tick_params(axis='x', rotation=45)
    st.pyplot(fig2)

    # Monthly data
    df_monthly = df.set_index('timestamp').resample('M')['usage_liters'].sum().reset_index()
    monthly_heading = {
        "en": "## 📉 Monthly Water Usage (Past Year)",
        "ar": "## 📉 استهلاك المياه الشهري (السنة الماضية)",
        "fr": "## 📉 Consommation d'eau mensuelle (Année passée)"
    }
    st.markdown(monthly_heading[lang])

    fig3, ax3 = plt.subplots(figsize=(14, 5))
    sns.lineplot(data=df_monthly, x='timestamp', y='usage_liters', ax=ax3)

    xlabel_text3 = {
        "en": "Month",
        "ar": "الشهر",
        "fr": "Mois"
    }
    ylabel_text3 = {
        "en": "Liters",
        "ar": "لتر",
        "fr": "Litres"
    }
    title_text_plot3 = {
        "en": "Monthly Water Usage",
        "ar": "استهلاك المياه الشهري",
        "fr": "Consommation d'eau mensuelle"
    }
    ax3.set_xlabel(xlabel_text3[lang])
    ax3.set_ylabel(ylabel_text3[lang])
    ax3.set_title(title_text_plot3[lang])
    ax3.tick_params(axis='x', rotation=45)
    st.pyplot(fig3)

    # Daily report download
    download_report_heading = {
        "en": "## 📥 Download Daily Usage Report",
        "ar": "## 📥 تحميل تقرير الاستهلاك اليومي",
        "fr": "## 📥 Télécharger le rapport de consommation quotidienne"
    }
    st.markdown(download_report_heading[lang])

    daily_report_csv = df_day.to_csv(index=False)
    download_report_button_label = {
        "en": "Download Daily Report CSV",
        "ar": "تحميل تقرير الاستهلاك اليومي CSV",
        "fr": "Télécharger le rapport quotidien CSV"
    }
    st.download_button(
        label=download_report_button_label[lang],
        data=daily_report_csv,
        file_name=f'daily_usage_{selected_day}.csv',
        mime='text/csv'
    )

    # Real-time notification if anomaly present today
    if "Anomaly" in df_day["anomaly"].values:
        anomaly_warning_text = {
            "en": "🚨 High water consumption anomaly detected today!",
            "ar": "🚨 تم الكشف عن خلل استهلاك المياه اليوم!",
            "fr": "🚨 Une anomalie de consommation d'eau élevée a été détectée aujourd'hui !"
        }
        st.markdown(f'<div class="anomaly-alert">{anomaly_warning_text[lang]}</div>', unsafe_allow_html=True)

    # Water conservation tips
    tips_heading = {
        "en": "### 💡 Water Conservation Tips",
        "ar": "### 💡 نصائح للحفاظ على المياه",
        "fr": "### 💡 Conseils pour la conservation de l'eau"
    }
    st.markdown(tips_heading[lang])
    tips_content = {
        "en": """
        - Fix leaks promptly to save water and money.
        - Use water-efficient appliances and fixtures.
        - Collect rainwater for irrigation.
        - Turn off taps when not in use.
        - Monitor your usage regularly to detect changes.
        """,
        "ar": """
        - أصلح التسريبات بسرعة لتوفير المياه والمال.
        - استخدم الأجهزة والتركيبات الموفرة للمياه.
        - اجمع مياه الأمطار للري.
        - أغلق الصنابير عند عدم الاستخدام.
        - راقب استهلاكك للكشف عن التغيرات.
        """,
        "fr": """
        - Réparez rapidement les fuites pour économiser de l'eau et de l'argent.
        - Utilisez des appareils et des installations économes en eau.
        - Récupérez l'eau de pluie pour l'irrigation.
        - Fermez les robinets lorsqu'ils ne sont pas utilisés.
        - Surveillez régulièrement votre consommation pour détecter les changements.
        """
    }
    st.markdown(tips_content[lang])

    # Testimonials section
    st.markdown("---")
    testimonials_heading = {
        "en": "### What Our Users Say",
        "ar": "### ماذا يقول مستخدمونا",
        "fr": "### Ce que disent nos utilisateurs"
    }
    st.markdown(testimonials_heading[lang])

    cols = st.columns(2)
    for i, testimonial in enumerate(testimonial_data[lang]):
        with cols[i % 2]:
            profile_emoji, profile_name, profile_email = profiles[i]
            st.markdown(f"""
            <div class="testimonial-card">
                <p>"{testimonial}"</p>
                <div class="testimonial-profile">
                    <span class="emoji">{profile_emoji}</span>
                    <div>
                        <strong>{profile_name}</strong>
                        <p style="font-size: 0.8em; margin: 0; color: #555;">{profile_email}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


    # FAQ
    st.markdown("---")
    faq_heading_html = {
        "en": """
        <div style="background: rgba(255, 255, 255, 0.9); padding: 1rem 1.5rem;
        border-radius: 12px; margin-top: 1rem; color: #111;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
        <h2 style="color: #023e8a;">💧 WaterGuard FAQ</h2>
        </div>
        """,
        "ar": """
        <div style="background: rgba(255, 255, 255, 0.9); padding: 1rem 1.5rem;
        border-radius: 12px; margin-top: 1rem; color: #111;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05); direction: rtl; text-align: right;">
        <h2 style="color: #023e8a;">💧 الأسئلة المتكررة - ووتر جارد</h2>
        </div>
        """,
        "fr": """
        <div style="background: rgba(255, 255, 255, 0.9); padding: 1rem 1.5rem;
        border-radius: 12px; margin-top: 1rem; color: #111;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
        <h2 style="color: #023e8a;">💧 FAQ WaterGuard</h2>
        </div>
        """
    }

    faqs = {
        "en": {
            "How can I detect a water leak early?": "Use WaterGuard's anomaly detection alerts to spot unusual spikes.",
            "What should I do if an anomaly is detected?": "Check for leaks or unusual water usage immediately.",
            "Can WaterGuard monitor multiple locations?": "Yes, it supports tracking usage across various branches or sites.",
            "How accurate is the anomaly detection?": "The system uses AI to detect 95% of irregular water usage patterns.",
            "Is WaterGuard suitable for factories with large consumption?": "Yes, it manages high-volume water use and alerts for excess.",
            "How often is water usage data updated?": "Data is updated hourly for precise monitoring and alerts.",
            "Can I download daily usage reports?": "Yes, downloadable CSV reports are available for any selected day.",
            "What cost savings can I expect?": "Early leak detection and usage optimization significantly reduce bills.",
            "Does WaterGuard support multiple languages?": "Currently supports English, Arabic, and French interfaces.",
            "Who do I contact for technical support?": "Contact support@waterguard.bh for all maintenance and help queries."
        },
        "ar": {
            "كيف يمكنني اكتشاف تسريب المياه مبكرًا؟": "استخدم تنبيهات كشف الخلل من ووتر جارد لرصد الزيادات غير المعتادة.",
            "ماذا أفعل إذا تم اكتشاف خلل؟": "تحقق فورًا من وجود تسريبات أو استهلاك غير طبيعي للمياه.",
            "هل يمكن لووتر جارد مراقبة مواقع متعددة؟": "نعم، يدعم تتبع الاستهلاك عبر فروع أو مواقع مختلفة.",
            "ما مدى دقة كشف الخلل؟": "يستخدم النظام الذكاء الاصطناعي لاكتشاف 95٪ من أنماط الاستهلاك غير الطبيعية.",
            "هل ووتر جارد مناسب للمصانع ذات الاستهلاك الكبير؟": "نعم، يدير استهلاك المياه العالي ويرسل تنبيهات عند الزيادة.",
            "كم مرة يتم تحديث بيانات استهلاك المياه؟": "يتم تحديث البيانات كل ساعة لمراقبة دقيقة وتنبيهات فورية.",
            "هل يمكنني تحميل تقارير الاستهلاك اليومية؟": "نعم، تتوفر تقارير CSV قابلة للتحميل لأي يوم محدد.",
            "ما مقدار التوفير المتوقع في التكاليف؟": "الكشف المبكر عن التسريبات وتحسين الاستخدام يقلل الفواتير بشكل كبير.",
            "هل يدعم ووتر جارد لغات متعددة؟": "يدعم حاليًا واجهات باللغات الإنجليزية والعربية والفرنسية.",
            "من أتصل به للدعم الفني؟": "تواصل مع support@waterguard.bh لجميع استفسارات الصيانة والمساعدة."
        },
        "fr": {
            "Comment puis-je détecter une fuite d'eau tôt ?": "Utilisez les alertes de détection d'anomalies de WaterGuard pour repérer les pics inhabituels.",
            "Que dois-je faire si une anomalie est détectée ?": "Vérifiez immédiatement les fuites ou la consommation d'eau inhabituelle.",
            "WaterGuard peut-il surveiller plusieurs emplacements ?": "Oui, il prend en charge le suivi de la consommation sur plusieurs succursales ou sites.",
            "Quelle est la précision de la détection des anomalies ?": "Le système utilise l'IA pour détecter 95 % des modèles de consommation d'eau irréguliers.",
            "WaterGuard est-il adapté aux usines à forte consommation ?": "Oui, il gère la consommation d'eau à haut volume et alerte en cas d'excès.",
            "À quelle fréquence les données de consommation d'eau sont-elles mises à jour ?": "Les données sont mises à jour toutes les heures pour une surveillance et des alertes précises.",
            "Puis-je télécharger des rapports de consommation quotidiens ?": "Oui, des rapports CSV téléchargeables sont disponibles pour n'importe quel jour sélectionné.",
            "À quelles économies de coûts puis-je m'attendre ?": "La détection précoce des fuites et l'optimisation de la consommation réduisent considérablement les factures.",
            "WaterGuard prend-il en charge plusieurs langues ?": "Actuellement, il prend en charge les interfaces en anglais, arabe et français.",
            "Qui dois-je contacter pour le support technique ?": "Contactez support@waterguard.bh pour toutes les questions de maintenance et d'assistance."
        }
    }

    st.markdown(faq_heading_html[lang], unsafe_allow_html=True)

    for q, a in faqs[lang].items():
        st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.85);
        padding: 0.75rem 1rem; border-radius: 10px; margin-bottom: 0.8rem;">
        <strong style="color: #0077b6;">{q}</strong>
        <p class="faq-answer" style="margin-top: 0.4rem;">{a}</p>
        </div>
        """, unsafe_allow_html=True)
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
import json
from prophet import Prophet  # NEW for forecasting

# ----------------------------
# Page setup
# ----------------------------
sns.set_style("whitegrid")
st.set_page_config(page_title="WaterGuard", layout="wide")

# ----------------------------
# CSS Styles
# ----------------------------
st.markdown("""<style>
.stApp { color: #f0f0f0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
.stButton>button {
    background-color: black; color: white; border-radius: 10px; padding: 10px 20px;
    font-weight: bold; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: all 0.2s ease;
}
.stButton>button:hover { background-color: #333; transform: translateY(-2px);
    box-shadow: 0 6px 8px rgba(0,0,0,0.15); }
.testimonial-card {
    background: rgba(255, 255, 255, 0.9); padding: 1rem; border-radius: 10px;
    margin-bottom: 1rem; box-shadow: 0 2px 5px rgba(0,0,0,0.1); color: #000;
}
.testimonial-profile { display: flex; align-items: center; margin-top: 10px; }
.testimonial-profile .emoji { font-size: 2rem; margin-right: 10px; }
.faq-answer { color: #000; }
.anomaly-alert {
    background-color: #fcebeb; color: #9f2a2a; padding: 1rem;
    border-radius: 10px; border: 1px solid #f5c6cb; margin-top: 1rem;
    font-weight: bold; text-align: center;
}
</style>""", unsafe_allow_html=True)

# ----------------------------
# Session state setup
# ----------------------------
if "lang" not in st.session_state: st.session_state.lang = "en"
if "course_progress" not in st.session_state: st.session_state.course_progress = 0
if "current_module" not in st.session_state: st.session_state.current_module = 0
if "quiz_scores" not in st.session_state: st.session_state.quiz_scores = {}
if "reward_claimed" not in st.session_state: st.session_state.reward_claimed = {}
if "rewards" not in st.session_state: st.session_state.rewards = 0
if "completed_quizzes" not in st.session_state: st.session_state.completed_quizzes = []

# ----------------------------
# Language toggle
# ----------------------------
lang_map = {"English": "en", "العربية": "ar", "Français": "fr"}
language_selection = st.sidebar.radio("🌐 Language", list(lang_map.keys()))
st.session_state.lang = lang_map[language_selection]
lang = st.session_state.lang

# ----------------------------
# Background image
# ----------------------------
def set_background(image_path):
    # NOTE: You will need to provide a water_bg.jpg image in the same directory for this to work.
    # The try/except block will prevent the app from crashing if the image is not found.
    try:
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
            st.markdown(f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{encoded}");
                background-size: cover; background-position: center;
                background-repeat: no-repeat; background-attachment: fixed;
            }}
            .stApp::before {{
                content: ""; position: fixed; top: 0; left: 0;
                width: 100vw; height: 100vh; background: rgba(0,0,0,0.45);
                z-index: -1;
            }}
            </style>""", unsafe_allow_html=True)
    except FileNotFoundError: pass
set_background("water_bg.jpg")

# ----------------------------
# Data simulation + anomalies
# ----------------------------
@st.cache_data
def simulate_data():
    np.random.seed(42)
    hours = 365 * 24
    date_range = pd.date_range(start="2024-01-01", periods=hours, freq="H")
    usage_main = np.random.normal(12, 3, hours).clip(0, 50)
    usage_garden = np.random.normal(5, 2, hours).clip(0, 20)
    usage_kitchen = np.random.normal(3, 1, hours).clip(0, 10)
    usage_bathroom = np.random.normal(4, 1.5, hours).clip(0, 15)

    df_local = pd.DataFrame({
        "timestamp": date_range,
        "usage_main_liters": usage_main,
        "usage_garden_liters": usage_garden,
        "usage_kitchen_liters": usage_kitchen,
        "usage_bathroom_liters": usage_bathroom,
    })
    df_local["usage_liters"] = df_local[
        ["usage_main_liters","usage_garden_liters","usage_kitchen_liters","usage_bathroom_liters"]
    ].sum(axis=1)
    df_local["date"] = df_local["timestamp"].dt.date

    # Inject anomalies
    anomaly_indices = random.sample(range(len(df_local)), int(0.05 * len(df_local)))
    for i in anomaly_indices:
        df_local.loc[i, ["usage_main_liters","usage_garden_liters",
                         "usage_kitchen_liters","usage_bathroom_liters"]] *= np.random.uniform(2, 5)
    df_local["usage_liters"] = df_local[
        ["usage_main_liters","usage_garden_liters","usage_kitchen_liters","usage_bathroom_liters"]
    ].sum(axis=1)
    return df_local

df = simulate_data()
model = IsolationForest(contamination=0.05, random_state=42)
df["anomaly"] = model.fit_predict(df[["usage_liters"]])
df["anomaly"] = df["anomaly"].map({1:"Normal",-1:"Anomaly"})

# ----------------------------
# Tabs
# ----------------------------
tab_labels = {
    "en": ["Course", "Bahrain Water", "Dashboard", "Forecasting", "Robotics", "Sustainability"],
    "ar": ["الدورة التدريبية", "تاريخ المياه في البحرين", "لوحة التحكم", "التنبؤ", "الروبوتات", "الاستدامة"],
    "fr": ["Cours", "Eau à Bahreïn", "Tableau de bord", "Prévisions", "Robotique", "Durabilité"]
}
top_tabs = st.tabs(tab_labels[lang])

# ----------------------------
# Course Tab (Placeholder)
# ----------------------------
with top_tabs[0]:
    st.header("📝 WaterGuard Course")
    st.write("This is a placeholder for the course content. You can add your modules, quizzes, and rewards system here.")

# ----------------------------
# Bahrain Water Tab (Placeholder)
# ----------------------------
with top_tabs[1]:
    st.header("💧 The History of Water in Bahrain")
    st.write("This is a placeholder for the historical content and information about water in Bahrain.")

# ----------------------------
# Dashboard Tab (Placeholder)
# ----------------------------
with top_tabs[2]:
    st.header("📊 Your Personal Dashboard")
    st.write("This is a placeholder for the user's water usage dashboard. You can display graphs and anomaly alerts here.")

# ----------------------------
# Forecasting Tab (Added from user code)
# ----------------------------
with top_tabs[3]:
    st.header("📈 Predictive Forecasting")
    st.write("This model predicts daily water usage for the next 30 days.")

    df_daily = df.set_index("timestamp").resample("D")["usage_liters"].sum().reset_index()
    df_daily.columns = ["ds", "y"]

    model = Prophet()
    model.fit(df_daily)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    fig_forecast = px.line(forecast, x="ds", y="yhat", title="Predicted Daily Usage")
    st.plotly_chart(fig_forecast, use_container_width=True)

    daily_quota = 1500
    high_risk = forecast[forecast["yhat"] > daily_quota * 1.1]
    if not high_risk.empty:
        st.warning(f"⚠ High-risk days detected: {len(high_risk)} upcoming")

# ----------------------------
# Robotics Tab (Added from user code)
# ----------------------------
with top_tabs[4]:
    st.header("🤖 Robotic Pipe Inspection Simulation")

    lang_texts = {
        "en": {
            "intro": "The WaterGuard robot inspects and cleans pipes to prevent leaks, blockages, and contamination.",
            "steps": [
                ("🚪 Step 1: Entry", "The robot enters the pipe network through an access point."),
                ("🔦 Step 2: Scanning", "360° cameras and sensors detect cracks, rust, or buildup."),
                ("🧹 Step 3: Cleaning", "Brushes and ultrasonic tools remove sediment and buildup."),
                ("💡 Step 4: Repair Assistance", "Laser mapping highlights weak spots for technicians."),
                ("📡 Step 5: Reporting", "A full health report is sent to the dashboard.")
            ]
        },
        "ar": {
            "intro": "يقوم روبوت ووتر جارد بفحص وتنظيف الأنابيب لمنع التسربات والانسداد والتلوث.",
            "steps": [
                ("🚪 الخطوة 1: الدخول", "يدخل الروبوت شبكة الأنابيب عبر نقطة وصول."),
                ("🔦 الخطوة 2: الفحص", "كاميرات وأجهزة استشعار بزاوية 360° تكشف الشقوق أو الصدأ أو الترسبات."),
                ("🧹 الخطوة 3: التنظيف", "تزيل الفرشاة والأدوات بالموجات فوق الصوتية الرواسب."),
                ("💡 الخطوة 4: المساعدة في الإصلاح", "يحدد الليزر النقاط الضعيفة لفنيي الصيانة."),
                ("📡 الخطوة 5: التقرير", "يتم إرسال تقرير كامل إلى لوحة التحكم.")
            ]
        },
        "fr": {
            "intro": "Le robot WaterGuard inspecte et nettoie les tuyaux pour prévenir fuites, blocages et contaminations.",
            "steps": [
                ("🚪 Étape 1 : Entrée", "Le robot entre dans le réseau par un point d'accès."),
                ("🔦 Étape 2 : Inspection", "Caméras 360° et capteurs détectent fissures, rouille ou dépôts."),
                ("🧹 Étape 3 : Nettoyage", "Brosses et outils ultrasoniques éliminent les dépôts."),
                ("💡 Étape 4 : Réparations", "Cartographie laser des points faibles."),
                ("📡 Étape 5 : Rapport", "Un rapport complet est envoyé au tableau de bord.")
            ]
        }
    }

    st.write(lang_texts[lang]["intro"])

    step = st.slider("Choose step", 1, len(lang_texts[lang]["steps"]), 1)
    title, desc = lang_texts[lang]["steps"][step-1]
    st.subheader(title)
    st.write(desc)

    # NOTE: You will need to provide images named robotic_step_1.png through robotic_step_5.png
    st.image(f"robotic_step_{step}.png", caption="Concept simulation", use_container_width=True)
    st.info("Simulation only. Real-time robot feed would appear here.")

# ----------------------------
# Sustainability Tab (Added from user code)
# ----------------------------
with top_tabs[5]:
    st.header("🌍 Sustainability & SDGs")
    st.write("WaterGuard contributes to global sustainability goals:")

    sdgs = {
        "SDG 6": "💧 Clean Water & Sanitation — Smart leak detection ensures sustainable supply.",
        "SDG 13": "🌍 Climate Action — Reduces desalination energy demand and emissions.",
        "SDG 12": "♻ Responsible Consumption — Users monitor and reduce usage in real time.",
        "SDG 9": "🏗 Innovation & Infrastructure — AI + IoT + robotics for resilient systems.",
        "SDG 3": "❤️ Good Health — Prevents leaks/contamination, ensures safe water."
    }
    for goal, desc in sdgs.items():
        st.markdown(f"**{goal}** — {desc}")

    st.subheader("🚀 Future of WaterGuard")
    st.write("""
    - Expansion to municipal & industrial water systems  
    - Integration with smart city infrastructure  
    - AI-driven predictive leak prevention  
    - Green energy powered desalination  
    - Advanced robotics for automated maintenance  
    """)
