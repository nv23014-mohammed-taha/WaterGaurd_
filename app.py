import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter

# Load or create CSV
try:
    df = pd.read_csv("weather_data.csv")
except FileNotFoundError:
    df = pd.DataFrame(columns=["Date", "Temperature", "Condition", "Humidity", "Wind"])

# Page config
st.set_page_config(page_title="🌤️ Weather Tracker", page_icon="☀️", layout="wide")
st.title("🌤️ **Top-Tier Weather Tracker**")
st.markdown("Track local weather, analyze patterns, and predict the future!")

# Sidebar Menu
menu = ["🏠 Home", "📝 Record Observation", "📊 View Statistics", 
        "🔍 Search by Date", "📋 View All Observations", 
        "📅 Filter by Month/Season", "📈 Temperature Trends", 
        "🔮 Predict Tomorrow", "📉 Compare Years", "🏆 Record-breaking Weather"]
choice = st.sidebar.selectbox("Menu", menu)

# Helper function: weather emoji
def weather_icon(condition):
    icons = {"Sunny": "☀️", "Cloudy": "☁️", "Rainy": "🌧️", "Snowy": "❄️", 
             "Windy": "💨", "Stormy": "⛈️"}
    return icons.get(condition, "🌤️")

# --- 1. Home ---
if choice == "🏠 Home":
    st.image('https://www.bing.com/images/search?view=detailV2&ccid=NktPKR4F&id=46DC77E68F84C8DEDF94827E5FF75B150D3708CB&thid=OIP.NktPKR4FOCDaAxLwWcvpfAHaFj&mediaurl=https%3a%2f%2fbahrainfinder.bh%2fwp-content%2fuploads%2f2023%2f11%2fBahrainfinder_2023-11-13+7-54+AM_02.jpg&cdnurl=https%3a%2f%2fth.bing.com%2fth%2fid%2fR.364b4f291e053820da0312f059cbe97c%3frik%3dywg3DRVb919%252bgg%26pid%3dImgRaw%26r%3d0&exph=1000&expw=1333&q=HOUSE+MANAMA&FORM=IRPRST&ck=DB4C48AE4441B5C96B18B9CD09C2030A&selectedIndex=3&itb=0', use_column_width=True)
    st.markdown("Welcome to your **personal weather tracker dashboard**! Use the sidebar to navigate through features.")

# --- 2. Record Observation ---
elif choice == "📝 Record Observation":
    st.subheader("Record a New Weather Observation")
    col1, col2 = st.columns(2)
    with col1:
        date = st.date_input("Date")
        temp = st.number_input("Temperature (°C)", value=25.0)
        condition = st.selectbox("Condition", ["Sunny", "Cloudy", "Rainy", "Snowy", "Windy", "Stormy"])
    with col2:
        humidity = st.number_input("Humidity (%)", 0, 100, value=50)
        wind = st.number_input("Wind Speed (km/h)", value=10)

    if st.button("Add Observation"):
        new_entry = {"Date": date.strftime("%m-%d-%Y"),
                     "Temperature": temp,
                     "Condition": condition,
                     "Humidity": humidity,
                     "Wind": wind}
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        df.to_csv("weather_data.csv", index=False)
        st.success(f"✅ Observation recorded: {weather_icon(condition)} {temp}°C, {humidity}% humidity")

# --- 3. View Statistics ---
elif choice == "📊 View Statistics":
    st.subheader("Weather Statistics")
    if not df.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("🌡️ Avg Temp (°C)", round(df["Temperature"].mean(),2))
        col2.metric("❄️ Min Temp (°C)", df["Temperature"].min())
        col3.metric("🔥 Max Temp (°C)", df["Temperature"].max())
        st.metric("🌤️ Most Common Condition", f"{df['Condition'].mode()[0]} {weather_icon(df['Condition'].mode()[0])}")
    else:
        st.warning("No data available.")

# --- 4. Search by Date ---
elif choice == "🔍 Search by Date":
    st.subheader("Search Observations by Date")
    search_date = st.date_input("Select Date")
    results = df[df["Date"] == search_date.strftime("%m-%d-%Y")]
    if not results.empty:
        for index, row in results.iterrows():
            st.info(f"{row['Date']}: {row['Temperature']}°C, {row['Condition']} {weather_icon(row['Condition'])}, Humidity: {row['Humidity']}%, Wind: {row['Wind']} km/h")
    else:
        st.warning("No observations for this date.")

# --- 5. View All Observations ---
elif choice == "📋 View All Observations":
    st.subheader("All Observations")
    st.dataframe(df.style.format({"Temperature":"{:.1f}°C", "Humidity":"{}%", "Wind":"{} km/h"}))

# --- 6. Filter by Month/Season ---
elif choice == "📅 Filter by Month/Season":
    df['Month'] = pd.to_datetime(df['Date'], format="%m-%d-%Y").dt.month
    month = st.selectbox("Select Month", range(1,13))
    filtered = df[df['Month'] == month]
    st.subheader(f"Observations for Month {month}")
    st.dataframe(filtered if not filtered.empty else "No observations for this month.")

# --- 7. Temperature Trends ---
elif choice == "📈 Temperature Trends":
    if not df.empty:
        df['Date_dt'] = pd.to_datetime(df['Date'], format="%m-%d-%Y")
        df_sorted = df.sort_values('Date_dt')
        st.line_chart(df_sorted.set_index('Date_dt')['Temperature'])
        st.bar_chart(df_sorted.set_index('Date_dt')['Temperature'])
    else:
        st.warning("No data available for trends.")

# --- 8. Predict Tomorrow ---
elif choice == "🔮 Predict Tomorrow":
    if len(df) >= 3:
        predicted_temp = round(df['Temperature'].tail(3).mean(),2)
        st.success(f"📊 Predicted Temperature for Tomorrow: {predicted_temp}°C")
    else:
        st.warning("Not enough data to predict.")

# --- 9. Compare Years ---
elif choice == "📉 Compare Years":
    df['Year'] = pd.to_datetime(df['Date'], format="%m-%d-%Y").dt.year
    years = df['Year'].unique()
    if len(years) >= 2:
        year1, year2 = st.selectbox("Select first year", years), st.selectbox("Select second year", years)
        compare = df[df['Year'].isin([year1, year2])]
        st.bar_chart(compare.groupby('Year')['Temperature'].mean())
    else:
        st.warning("Not enough years of data to compare.")

# --- 10. Record-breaking Weather ---
elif choice == "🏆 Record-breaking Weather":
    if not df.empty:
        st.success(f"🔥 Hottest Temperature: {df['Temperature'].max()}°C")
        st.info(f"❄️ Coldest Temperature: {df['Temperature'].min()}°C")
        most_freq = df['Condition'].mode()[0]
        st.write(f"🌤️ Most Frequent Condition: {most_freq} {weather_icon(most_freq)}")
    else:
        st.warning("No data available.")
