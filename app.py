# weather_tracker_streamlit.py

import streamlit as st
import pandas as pd
import os
from collections import Counter
import matplotlib.pyplot as plt

# -------------------------
# Create sample CSV if it doesn't exist
# -------------------------
if not os.path.exists("weather_data.csv"):
    sample_data = pd.DataFrame([
        {"Date":"11-02-2025","Temperature":28,"Condition":"Sunny","Humidity":45,"Wind":10},
        {"Date":"11-01-2025","Temperature":22,"Condition":"Rainy","Humidity":70,"Wind":15},
        {"Date":"10-31-2025","Temperature":25,"Condition":"Cloudy","Humidity":50,"Wind":12},
        {"Date":"10-30-2025","Temperature":30,"Condition":"Sunny","Humidity":40,"Wind":8},
        {"Date":"10-29-2025","Temperature":18,"Condition":"Snowy","Humidity":80,"Wind":20},
    ])
    sample_data.to_csv("weather_data.csv", index=False)

# -------------------------
# Load CSV
# -------------------------
def load_data():
    return pd.read_csv("weather_data.csv")

def save_data(df):
    df.to_csv("weather_data.csv", index=False)

# -------------------------
# Sidebar menu
# -------------------------
st.sidebar.title("🌤️ Top-Tier Weather Tracker")
menu = st.sidebar.radio("Menu", [
    "Record Observation",
    "View Statistics",
    "Search by Date",
    "View All Observations",
    "Filter by Month/Season",
    "Temperature Trends",
    "Predict Tomorrow",
    "Compare Yearly Data",
    "Record-Breaking Conditions"
])

data = load_data()

# -------------------------
# 1️⃣ Record Observation
# -------------------------
if menu == "Record Observation":
    st.header("📝 Record New Weather Observation")
    date = st.date_input("Date")
    temp = st.number_input("Temperature (°C)", value=25)
    condition = st.selectbox("Condition", ["Sunny","Cloudy","Rainy","Snowy","Windy","Foggy"])
    humidity = st.number_input("Humidity (%)", value=50)
    wind = st.number_input("Wind Speed (km/h)", value=10)

    if st.button("Add Observation"):
        new_entry = {
            "Date": date.strftime("%m-%d-%Y"),
            "Temperature": temp,
            "Condition": condition,
            "Humidity": humidity,
            "Wind": wind
        }
        data = data.append(new_entry, ignore_index=True)
        save_data(data)
        st.success("Observation recorded successfully!")

# -------------------------
# 2️⃣ View Statistics
# -------------------------
elif menu == "View Statistics":
    st.header("📊 Weather Statistics")
    if data.empty:
        st.warning("No data available.")
    else:
        avg_temp = data["Temperature"].mean()
        min_temp = data["Temperature"].min()
        max_temp = data["Temperature"].max()
        most_common = Counter(data["Condition"]).most_common(1)[0][0]

        st.write(f"**Average Temperature:** {avg_temp:.1f}°C")
        st.write(f"**Minimum Temperature:** {min_temp}°C")
        st.write(f"**Maximum Temperature:** {max_temp}°C")
        st.write(f"**Most Common Condition:** {most_common}")

# -------------------------
# 3️⃣ Search by Date
# -------------------------
elif menu == "Search by Date":
    st.header("🔍 Search Observations by Date")
    search_date = st.date_input("Select Date")
    search_str = search_date.strftime("%m-%d-%Y")
    results = data[data["Date"] == search_str]
    if results.empty:
        st.warning("No observations found for this date.")
    else:
        st.dataframe(results)

# -------------------------
# 4️⃣ View All Observations
# -------------------------
elif menu == "View All Observations":
    st.header("📋 All Weather Observations")
    if data.empty:
        st.warning("No data available.")
    else:
        st.dataframe(data)

# -------------------------
# 5️⃣ Filter by Month/Season
# -------------------------
elif menu == "Filter by Month/Season":
    st.header("🌦️ Filter Observations")
    month = st.selectbox("Select Month", range(1,13))
    filtered = data[pd.to_datetime(data["Date"]).dt.month == month]
    if filtered.empty:
        st.warning("No data for this month.")
    else:
        st.dataframe(filtered)

# -------------------------
# 6️⃣ Temperature Trends
# -------------------------
elif menu == "Temperature Trends":
    st.header("📈 Temperature Trends")
    if data.empty:
        st.warning("No data available.")
    else:
        data_sorted = data.sort_values(by="Date")
        plt.figure(figsize=(8,4))
        plt.plot(pd.to_datetime(data_sorted["Date"]), data_sorted["Temperature"], marker='o')
        plt.title("Temperature Over Time")
        plt.xlabel("Date")
        plt.ylabel("Temperature (°C)")
        plt.grid(True)
        st.pyplot(plt)

# -------------------------
# 7️⃣ Predict Tomorrow
# -------------------------
elif menu == "Predict Tomorrow":
    st.header("🔮 Predict Tomorrow's Temperature")
    if data.empty:
        st.warning("No data available.")
    else:
        avg_temp = data["Temperature"].mean()
        most_common = Counter(data["Condition"]).most_common(1)[0][0]
        st.write(f"Predicted Temperature: {avg_temp:.1f}°C")
        st.write(f"Predicted Condition: {most_common}")

# -------------------------
# 8️⃣ Compare Yearly Data
# -------------------------
elif menu == "Compare Yearly Data":
    st.header("📅 Compare Yearly Data")
    data["Year"] = pd.to_datetime(data["Date"]).dt.year
    year_stats = data.groupby("Year")["Temperature"].agg(["mean","min","max"])
    st.dataframe(year_stats)

# -------------------------
# 9️⃣ Record-Breaking Conditions
# -------------------------
elif menu == "Record-Breaking Conditions":
    st.header("🏆 Record-Breaking Temperatures/Conditions")
    if data.empty:
        st.warning("No data available.")
    else:
        hottest = data.loc[data["Temperature"].idxmax()]
        coldest = data.loc[data["Temperature"].idxmin()]
        st.write("**Hottest Day:**")
        st.write(hottest)
        st.write("**Coldest Day:**")
        st.write(coldest)
