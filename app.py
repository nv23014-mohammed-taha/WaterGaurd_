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

sns.set_style("whitegrid")
st.set_page_config(page_title="WaterGuard", layout="wide")

# ---------- LANGUAGE TOGGLE ---------- #
language = st.sidebar.radio("🌐 Language / اللغة", ["English", "العربية"], key="language_radio")
lang = "ar" if language == "العربية" else "en"

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

# Wrap main content for screen reader navigation
st.markdown('<main role="main" tabindex="-1">', unsafe_allow_html=True)

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

# ---------- ALERTS WITH ACCESSIBLE LIVE REGION ---------- #
high_usage_threshold = daily_quota * 0.9

if day_usage > high_usage_threshold:
    alert_text_en = "🚨 High water consumption detected today!"
    alert_text_ar = "🚨 تم الكشف عن استهلاك مياه مرتفع اليوم!"
    alert_html_en = f'<div role="alert" aria-live="assertive" style="color: #9f3a38; font-weight: 700;">{alert_text_en}</div>'
    alert_html_ar = f'<div role="alert" aria-live="assertive" style="color: #9f3a38; font-weight: 700; direction: rtl; text-align: right;">{alert_text_ar}</div>'
    if lang == 'en':
        st.sidebar.markdown(alert_html_en, unsafe_allow_html=True)
    else:
        st.sidebar.markdown(alert_html_ar, unsafe_allow_html=True)

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

# --------- Real-Time Notifications & Alerts with aria-live --------- #
if "Anomaly" in df_day["anomaly"].values:
    alert_main_en = '🚨 High water consumption anomaly detected today!'
    alert_main_ar = '🚨 تم الكشف عن خلل استهلاك المياه اليوم!'
    alert_main_html_en = f'<div role="alert" aria-live="assertive" style="color: #9f3a38; font-weight: 700;">{alert_main_en}</div>'
    alert_main_html_ar = f'<div role="alert" aria-live="assertive" style="color: #9f3a38; font-weight: 700; direction: rtl; text-align: right;">{alert_main_ar}</div>'
    if lang == 'en':
        st.markdown(alert_main_html_en, unsafe_allow_html=True)
    else:
        st.markdown(alert_main_html_ar, unsafe_allow_html=True)

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
    - راقب استهلاكك للكشف عن أي تغييرات.
    """)

# --------- FAQ Section with aria roles --------- #
faq_id = "faq_section"
if lang == 'en':
    st.markdown(f'<section role="region" aria-labelledby="{faq_id}"><h2 id="{faq_id}">❓ Frequently Asked Questions</h2></section>', unsafe_allow_html=True)
    st.markdown("""
    <details>
      <summary><strong>How does WaterGuard detect leaks?</strong></summary>
      <p>WaterGuard uses AI to analyze water usage patterns and identify anomalies that suggest leaks.</p>
    </details>
    <details>
      <summary><strong>Can I use WaterGuard for commercial properties?</strong></summary>
      <p>Currently, WaterGuard is designed for residential homes, but we plan to support commercial use soon.</p>
    </details>
    <details>
      <summary><strong>Is my data secure?</strong></summary>
      <p>Yes, we prioritize your privacy and secure all usage data with encryption.</p>
    </details>
    """, unsafe_allow_html=True)
else:
    st.markdown(f'<section role="region" aria-labelledby="{faq_id}"><h2 id="{faq_id}">❓ الأسئلة الشائعة</h2></section>', unsafe_allow_html=True)
    st.markdown("""
    <details>
      <summary><strong>كيف يكتشف ووتر جارد التسريبات؟</strong></summary>
      <p>يستخدم ووتر جارد الذكاء الاصطناعي لتحليل أنماط استهلاك المياه واكتشاف أي خلل يشير إلى تسرب.</p>
    </details>
    <details>
      <summary><strong>هل يمكن استخدام ووتر جارد للمباني التجارية؟</strong></summary>
      <p>حالياً، يتم تصميم ووتر جارد للمنازل السكنية فقط، لكننا نخطط لدعم الاستخدام التجاري قريبًا.</p>
    </details>
    <details>
      <summary><strong>هل بياناتي آمنة؟</strong></summary>
      <p>نعم، نحن نولي خصوصيتك اهتمامًا كبيرًا ونؤمن جميع بيانات الاستخدام بالتشفير.</p>
    </details>
    """, unsafe_allow_html=True)

# --------- Testimonials Section --------- #
testimonials_id = "testimonials_section"
if lang == 'en':
    st.markdown(f'<section role="region" aria-labelledby="{testimonials_id}"><h2 id="{testimonials_id}">💬 Testimonials</h2></section>', unsafe_allow_html=True)
    st.markdown("""
    <blockquote>"WaterGuard helped me catch a hidden leak that saved me hundreds of dinars!" - Fatima A.</blockquote>
    <blockquote>"The real-time alerts are super helpful to monitor our daily usage." - Ali M.</blockquote>
    <blockquote>"I love how easy it is to understand my water consumption trends." - Sara K.</blockquote>
    """, unsafe_allow_html=True)
else:
    st.markdown(f'<section role="region" aria-labelledby="{testimonials_id}"><h2 id="{testimonials_id}">💬 آراء المستخدمين</h2></section>', unsafe_allow_html=True)
    st.markdown("""
    <blockquote>"ساعدني ووتر جارد على اكتشاف تسريب خفي وفر لي مئات الدنانير!" - فاطمة أ.</blockquote>
    <blockquote>"التنبيهات الفورية مفيدة جدًا لمراقبة استهلاكنا اليومي." - علي م.</blockquote>
    <blockquote>"أحب سهولة فهم اتجاهات استهلاكي للمياه." - سارة ك.</blockquote>
    """, unsafe_allow_html=True)

# Close main landmark
st.markdown('</main>', unsafe_allow_html=True)

