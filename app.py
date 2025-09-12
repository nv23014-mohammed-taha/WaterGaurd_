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
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import json
import warnings
warnings.filterwarnings('ignore')

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
# New "Predictive AI" function
# ----------------------------
@st.cache_data
def run_prediction(data, forecast_hours=7 * 24):
    """Trains an ARIMA model and generates a forecast."""
    # Use the last 30 days of "normal" data for training the model
    # We select a slice of the data that doesn't contain the randomly injected anomalies.
    training_data = data.iloc[-30*24:-forecast_hours]
    training_series = training_data['usage_liters']
    
    # Fit the ARIMA model (p=1, d=1, q=1 is a common starting point)
    model = ARIMA(training_series, order=(1, 1, 1))
    model_fit = model.fit()

    # Generate the forecast
    forecast = model_fit.forecast(steps=forecast_hours)

    # Prepare a DataFrame for plotting
    forecast_index = pd.date_range(
        start=training_data['timestamp'].iloc[-1] + pd.Timedelta(hours=1),
        periods=forecast_hours,
        freq='H'
    )
    forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['predicted_usage'])
    
    return training_data, forecast_df

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
            "ar": "### ملخص المكافآت",
            "fr": "### Résumé des récompenses"
        }
        st.markdown(rewards_summary_heading[lang])
        rewards_summary_text = {
            "en": f"You have earned a total of **{st.session_state.rewards} fils** (BHD {st.session_state.rewards/1000.0:.3f}) from completing the quizzes. Keep saving water and earning rewards!",
            "ar": f"لقد حصلت على ما مجموعه **{st.session_state.rewards} فلس** (ب.د {st.session_state.rewards/1000.0:.3f}) من إكمال الاختبارات. استمر في توفير المياه وكسب المكافآت!",
            "fr": f"Vous avez gagné un total de **{st.session_state.rewards} fils** (BHD {st.session_state.rewards/1000.0:.3f}) en terminant les quiz. Continuez à économiser l'eau et à gagner des récompenses !"
        }
        st.info(rewards_summary_text[lang])


# ----------------------------
# Bahrain Water History Tab
# ----------------------------
with top_tabs[1]:
    header_text = {
        "en": "💧 Bahrain's Water History",
        "ar": "💧 تاريخ المياه في البحرين",
        "fr": "💧 Histoire de l'eau à Bahreïn"
    }
    st.header(header_text[lang])
    content = {
        "en": BAHRAIN_HISTORY_EN,
        "ar": BAHRAIN_HISTORY_AR,
        "fr": BAHRAIN_HISTORY_FR
    }
    st.markdown(f'<div style="background: rgba(0,0,0,0.4); padding: 15px; border-radius: 10px; color: #fff; font-size: 1.1rem; line-height: 1.6;">{content[lang]}</div>', unsafe_allow_html=True)


# ----------------------------
# Dashboard Tab
# ----------------------------
with top_tabs[2]:
    dashboard_header = {
        "en": "📊 Your WaterGuard Dashboard",
        "ar": "📊 لوحة تحكم ووتر جارد الخاصة بك",
        "fr": "📊 Votre Tableau de bord WaterGuard"
    }
    st.header(dashboard_header[lang])
    
    # Summary Cards
    col1, col2, col3 = st.columns(3)
    
    daily_total = df['usage_liters'].iloc[-24:].sum()
    monthly_total = df['usage_liters'].iloc[-30*24:].sum()
    total_anomalies = (df['anomaly'] == 'Anomaly').sum()

    with col1:
        card_daily_title = {"en": "Today's Usage", "ar": "استهلاك اليوم", "fr": "Consommation du jour"}
        st.metric(card_daily_title[lang], f"{daily_total:.2f} L")
    
    with col2:
        card_monthly_title = {"en": "Monthly Usage", "ar": "الاستهلاك الشهري", "fr": "Consommation mensuelle"}
        st.metric(card_monthly_title[lang], f"{monthly_total:.2f} L")

    with col3:
        card_alerts_title = {"en": "Total Leak Alerts", "ar": "إجمالي تنبيهات التسرب", "fr": "Total des alertes de fuite"}
        st.metric(card_alerts_title[lang], f"{total_anomalies} alerts")

    # Anomaly Alert Section
    recent_anomalies = df[df['anomaly'] == 'Anomaly'].iloc[-1:]
    if not recent_anomalies.empty:
        anom_time = recent_anomalies.iloc[0]['timestamp']
        anom_severity = recent_anomalies.iloc[0]['severity']
        alert_text = {
            "en": f"🚨 **High Risk Alert:** An unusual water usage spike was detected at {anom_time.strftime('%I:%M %p, %b %d')}. Severity: {anom_severity}",
            "ar": f"🚨 **تنبيه عالي الخطورة:** تم رصد ارتفاع غير عادي في استهلاك المياه في {anom_time.strftime('%I:%M %p, %b %d')}. الشدة: {anom_severity}",
            "fr": f"🚨 **Alerte Risque Élevé:** Un pic de consommation d'eau inhabituel a été détecté à {anom_time.strftime('%I:%M %p, %b %d')}. Gravité : {anom_severity}"
        }
        st.markdown(f'<div class="anomaly-alert">{alert_text[lang]}</div>', unsafe_allow_html=True)
        
    st.markdown("---")
    
    # ----------------------------
    # New Predictive AI Visualization
    # ----------------------------
    predictive_header = {
        "en": "🧠 Predictive Leak Analysis",
        "ar": "🧠 تحليل التنبؤ بالتسرب",
        "fr": "🧠 Analyse prédictive des fuites"
    }
    predictive_caption = {
        "en": "The AI model has learned your normal water usage patterns from the past 30 days and predicts what your usage will be. Any sharp deviations indicate a potential leak.",
        "ar": "لقد تعلم نموذج الذكاء الاصطناعي أنماط استهلاكك الطبيعية للمياه من آخر 30 يومًا ويتنبأ بما سيكون عليه استهلاكك. أي انحرافات حادة تشير إلى تسرب محتمل.",
        "fr": "Le modèle d'IA a appris vos habitudes de consommation d'eau normales des 30 derniers jours et prédit quelle sera votre consommation. Toute déviation brusque indique une fuite potentielle."
    }
    
    st.subheader(predictive_header[lang])
    st.write(predictive_caption[lang])
    
    # Get the data for prediction
    training_data, forecast_df = run_prediction(df)
    
    # Combine training data and forecast for plotting
    plot_data = pd.concat([training_data.set_index('timestamp')['usage_liters'], forecast_df['predicted_usage']])
    plot_data = plot_data.reset_index()
    plot_data.columns = ['timestamp', 'usage_liters']
    
    # Create Plotly figure
    fig = go.Figure()

    # Add Actual Usage trace
    fig.add_trace(go.Scatter(
        x=training_data['timestamp'], 
        y=training_data['usage_liters'],
        mode='lines', 
        name='Actual Usage',
        line=dict(color='#0275d8')
    ))

    # Add Forecasted Usage trace
    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df['predicted_usage'],
        mode='lines',
        name='Predicted Usage',
        line=dict(color='#28a745', dash='dash')
    ))

    # Mark anomalies as red dots
    anomaly_df = df[df['anomaly'] == 'Anomaly'].iloc[-30*24:]
    if not anomaly_df.empty:
        fig.add_trace(go.Scatter(
            x=anomaly_df['timestamp'],
            y=anomaly_df['usage_liters'],
            mode='markers',
            name='Detected Leak',
            marker=dict(color='red', size=8)
        ))

    # Update layout for a professional look
    fig.update_layout(
        title=dict(text='Water Usage: Past 30 Days & Next 7-Day Forecast', x=0.5, font=dict(color='white')),
        xaxis_title='Date and Time',
        yaxis_title='Water Usage (Liters)',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")

    # Existing charts for daily and weekly usage
    with st.expander("Show Detailed Usage Graphs"):
        # Daily Usage Plot
        daily_df = df.groupby('date')['usage_liters'].sum().reset_index()
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        fig_daily = px.bar(daily_df, x='date', y='usage_liters',
                             title="Daily Water Consumption",
                             labels={'date': 'Date', 'usage_liters': 'Usage (Liters)'})
        st.plotly_chart(fig_daily, use_container_width=True)

        # Hourly Usage Plot with Anomalies
        fig_hourly = px.line(df, x='timestamp', y='usage_liters',
                              title="Hourly Water Usage with Anomaly Detection",
                              labels={'timestamp': 'Time', 'usage_liters': 'Usage (Liters)'},
                              color='anomaly',
                              color_discrete_map={'Normal': '#0275d8', 'Anomaly': 'red'})
        fig_hourly.update_traces(marker=dict(size=4))
        st.plotly_chart(fig_hourly, use_container_width=True)


    # Testimonials
    st.markdown("---")
    testimonials_header = {
        "en": "What Our Users Say",
        "ar": "ماذا يقول مستخدمونا",
        "fr": "Ce que disent nos utilisateurs"
    }
    st.subheader(testimonials_header[lang])
    
    testimonial_cols = st.columns(3)
    displayed_testimonials = random.sample(testimonial_data[lang], 3)
    displayed_profiles = random.sample(profiles, 3)
    
    for i, col in enumerate(testimonial_cols):
        with col:
            profile = displayed_profiles[i]
            emoji, name, email = profile
            
            st.markdown(f"""
            <div class="testimonial-card">
                <p>{displayed_testimonials[i]}</p>
                <div class="testimonial-profile">
                    <span class="emoji">{emoji}</span>
                    <div>
                        <strong>{name}</strong>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
