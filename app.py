import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

# ----------------------------
# CONFIGURATION
# ----------------------------
st.set_page_config(page_title="🌤️ Weather Tracker", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #f0f8ff; }
    h1, h2, h3 { color: #004d99; text-align: center; }
    .stButton>button {
        background-color: #007acc;
        color: white;
        border-radius: 10px;
        height: 50px;
        width: 100%;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# FILE HANDLING
# ----------------------------
FILE_PATH = "weather_data.csv"

def load_data():
    if os.path.exists(FILE_PATH):
        return pd.read_csv(FILE_PATH)
    else:
        return pd.DataFrame(columns=["Date", "Temperature (°C)", "Condition", "Humidity (%)", "Wind Speed (km/h)"])

def save_data(df):
    df.to_csv(FILE_PATH, index=False)

data = load_data()

# ----------------------------
# PAGE TITLE
# ----------------------------
st.title("🌤️ Top-Tier Weather Tracker")
st.write("Track local weather, analyze patterns, and predict future conditions!")

# ----------------------------
# MENU
# ----------------------------
menu = st.sidebar.radio("📋 Menu", [
    "Record a new weather observation",
    "View weather statistics",
    "Search observations by date",
    "View all observations",
    "Filter by month or season",
    "Display temperature trends",
    "Predict tomorrow's weather",
    "Compare yearly data",
    "Record-breaking temperatures"
])

# ----------------------------
# RECORD OBSERVATION
# ----------------------------
if menu == "Record a new weather observation":
    st.header("🌦️ Record a New Observation")
    with st.form("observation_form"):
        date = st.text_input("Enter date (MM-DD-YYYY):")
        temp = st.number_input("Temperature (°C):", -50.0, 60.0, step=0.1)
        condition = st.selectbox("Weather Condition:", ["Sunny", "Cloudy", "Rainy", "Snowy", "Windy", "Stormy", "Foggy"])
        humidity = st.number_input("Humidity (%):", 0, 100)
        wind = st.number_input("Wind Speed (km/h):", 0.0, 300.0, step=0.1)
        submit = st.form_submit_button("Save Observation")

    if submit:
        try:
            datetime.strptime(date, "%m-%d-%Y")  # validate date format
            new_entry = pd.DataFrame({
                "Date": [date],
                "Temperature (°C)": [temp],
                "Condition": [condition],
                "Humidity (%)": [humidity],
                "Wind Speed (km/h)": [wind]
            })
            data = pd.concat([data, new_entry], ignore_index=True)
            save_data(data)
            st.success("✅ Observation recorded successfully!")
        except ValueError:
            st.error("❌ Invalid date format! Please use MM-DD-YYYY.")

# ----------------------------
# VIEW STATISTICS
# ----------------------------
elif menu == "View weather statistics":
    st.header("📊 Weather Statistics")
    if data.empty:
        st.warning("No data available yet.")
    else:
        avg_temp = data["Temperature (°C)"].mean()
        min_temp = data["Temperature (°C)"].min()
        max_temp = data["Temperature (°C)"].max()
        common_cond = data["Condition"].mode()[0]

        st.metric("🌡️ Average Temperature", f"{avg_temp:.2f} °C")
        st.metric("🥶 Minimum Temperature", f"{min_temp:.2f} °C")
        st.metric("🥵 Maximum Temperature", f"{max_temp:.2f} °C")
        st.metric("☁️ Most Common Condition", common_cond)

# ----------------------------
# SEARCH BY DATE
# ----------------------------
elif menu == "Search observations by date":
    st.header("🔍 Search by Date")
    search_date = st.text_input("Enter date (MM-DD-YYYY):")
    if st.button("Search"):
        result = data[data["Date"] == search_date]
        if result.empty:
            st.warning("No observations found for this date.")
        else:
            st.dataframe(result)

# ----------------------------
# VIEW ALL OBSERVATIONS
# ----------------------------
elif menu == "View all observations":
    st.header("📋 All Observations")
    if data.empty:
        st.warning("No data available.")
    else:
        st.dataframe(data)

# ----------------------------
# FILTER BY MONTH OR SEASON
# ----------------------------
elif menu == "Filter by month or season":
    st.header("📆 Filter Observations")
    if data.empty:
        st.warning("No data available.")
    else:
        data["Date_dt"] = pd.to_datetime(data["Date"], format="%m-%d-%Y", errors="coerce")
        filter_type = st.radio("Filter by:", ["Month", "Season"])

        if filter_type == "Month":
            month = st.selectbox("Select Month:", range(1, 13), format_func=lambda x: datetime(2025, x, 1).strftime("%B"))
            filtered = data[data["Date_dt"].dt.month == month]
        else:
            season = st.selectbox("Select Season:", ["Winter", "Spring", "Summer", "Autumn"])
            season_months = {"Winter": [12, 1, 2], "Spring": [3, 4, 5], "Summer": [6, 7, 8], "Autumn": [9, 10, 11]}
            filtered = data[data["Date_dt"].dt.month.isin(season_months[season])]

        if filtered.empty:
            st.warning("No data for selected period.")
        else:
            st.dataframe(filtered)

# ----------------------------
# TEMPERATURE TRENDS
# ----------------------------
elif menu == "Display temperature trends":
    st.header("📈 Temperature Trends")
    if data.empty:
        st.warning("No data available.")
    else:
        data["Date_dt"] = pd.to_datetime(data["Date"], format="%m-%d-%Y", errors="coerce")
        data = data.sort_values("Date_dt")
        plt.figure(figsize=(8,4))
        plt.plot(data["Date_dt"], data["Temperature (°C)"], marker='o')
        plt.title("Temperature Trend Over Time")
        plt.xlabel("Date")
        plt.ylabel("Temperature (°C)")
        st.pyplot(plt)

# ----------------------------
# PREDICT TOMORROW'S WEATHER
# ----------------------------
elif menu == "Predict tomorrow's weather":
    st.header("🤖 Predict Tomorrow's Weather")
    if data.empty:
        st.warning("No data to predict from.")
    else:
        last_temp = data["Temperature (°C)"].iloc[-1]
        avg_change = data["Temperature (°C)"].diff().mean()
        predicted_temp = last_temp + avg_change
        st.metric("🌡️ Predicted Temperature", f"{predicted_temp:.2f} °C")
        st.write("Condition likely:", data["Condition"].mode()[0])

# ----------------------------
# COMPARE YEARLY DATA
# ----------------------------
elif menu == "Compare yearly data":
    st.header("📆 Compare Yearly Data")
    if data.empty:
        st.warning("No data available.")
    else:
        data["Date_dt"] = pd.to_datetime(data["Date"], format="%m-%d-%Y", errors="coerce")
        data["Year"] = data["Date_dt"].dt.year
        yearly_stats = data.groupby("Year")["Temperature (°C)"].mean()
        st.bar_chart(yearly_stats)

# ----------------------------
# RECORD-BREAKING CONDITIONS
# ----------------------------
elif menu == "Record-breaking temperatures":
    st.header("🏆 Record-Breaking Conditions")
    if data.empty:
        st.warning("No data available.")
    else:
        st.write(f"🔥 Highest Temperature: {data['Temperature (°C)'].max()} °C")
        st.write(f"❄️ Lowest Temperature: {data['Temperature (°C)'].min()} °C")
        st.write(f"💨 Highest Wind Speed: {data['Wind Speed (km/h)'].max()} km/h")
