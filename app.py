import streamlit as st
import pandas as pd
from collections import Counter

# --- Helper functions ---
DATA_FILE = "weather_data.csv"

def load_data():
    try:
        return pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        return pd.DataFrame(columns=["Date", "Temperature", "Condition", "Humidity", "Wind"])

def save_data(data):
    data.to_csv(DATA_FILE, index=False)

def add_observation(date, temp, condition, humidity, wind):
    new_entry = {
        "Date": date,
        "Temperature": temp,
        "Condition": condition,
        "Humidity": humidity,
        "Wind": wind
    }
    data = load_data()
    new_df = pd.DataFrame([new_entry])
    data = pd.concat([data, new_df], ignore_index=True)  # Fixed append error
    save_data(data)
    st.success("✅ Observation added successfully!")

def get_statistics(data):
    if data.empty:
        return None
    avg_temp = data["Temperature"].mean()
    min_temp = data["Temperature"].min()
    max_temp = data["Temperature"].max()
    most_common = Counter(data["Condition"]).most_common(1)[0][0]
    return avg_temp, min_temp, max_temp, most_common

def predict_tomorrow(data):
    if data.empty:
        return "No data available"
    latest_condition = data["Condition"].mode()[0]
    latest_temp = data["Temperature"].mean()
    return f"Predicted: {latest_condition}, Temp: {latest_temp:.1f}°C"

# --- Streamlit App ---
st.title("🌤️ Top-Tier Weather Tracker")
st.write("Track local weather, analyze patterns, and predict the future!")

menu = st.sidebar.selectbox("Menu", [
    "Record a new weather observation",
    "View weather statistics",
    "Search observations by date",
    "View all observations",
    "Filter observations by month/season",
    "Display temperature trends",
    "Predict tomorrow's weather",
    "Compare yearly data",
    "Record-breaking temperatures/conditions"
])

data = load_data()

# --- Record a new observation ---
if menu == "Record a new weather observation":
    st.header("📝 Add Observation")
    date = st.date_input("Date")
    temp = st.number_input("Temperature (°C)", value=25.0)
    condition = st.selectbox("Weather Condition", ["Sunny", "Cloudy", "Rainy", "Snowy", "Stormy", "Windy"])
    humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=50)
    wind = st.number_input("Wind Speed (km/h)", min_value=0, value=10)
    
    if st.button("Add Observation"):
        add_observation(date.strftime("%m-%d-%Y"), temp, condition, humidity, wind)

# --- View statistics ---
elif menu == "View weather statistics":
    st.header("📊 Weather Statistics")
    stats = get_statistics(data)
    if stats:
        avg_temp, min_temp, max_temp, most_common = stats
        st.write(f"Average Temperature: {avg_temp:.1f}°C")
        st.write(f"Minimum Temperature: {min_temp:.1f}°C")
        st.write(f"Maximum Temperature: {max_temp:.1f}°C")
        st.write(f"Most Common Condition: {most_common}")
    else:
        st.warning("No data available.")

# --- Search by date ---
elif menu == "Search observations by date":
    st.header("🔍 Search by Date")
    search_date = st.date_input("Select Date")
    filtered = data[data["Date"] == search_date.strftime("%m-%d-%Y")]
    if filtered.empty:
        st.warning("No observations found for this date.")
    else:
        st.dataframe(filtered)

# --- View all observations ---
elif menu == "View all observations":
    st.header("📋 All Observations")
    if data.empty:
        st.warning("No data available.")
    else:
        st.dataframe(data)

# --- Filter by month/season ---
elif menu == "Filter observations by month/season":
    st.header("🌦️ Filter Observations")
    filter_type = st.radio("Filter by", ["Month", "Season"])
    if data.empty:
        st.warning("No data available.")
    else:
        data["Date_dt"] = pd.to_datetime(data["Date"])
        if filter_type == "Month":
            month = st.selectbox("Select Month", range(1, 13), format_func=lambda x: pd.to_datetime(f"2025-{x}-01").strftime('%B'))
            filtered = data[data["Date_dt"].dt.month == month]
        else:
            season = st.selectbox("Select Season", ["Winter", "Spring", "Summer", "Autumn"])
            month_to_season = {
                "Winter": [12, 1, 2],
                "Spring": [3, 4, 5],
                "Summer": [6, 7, 8],
                "Autumn": [9, 10, 11]
            }
            filtered = data[data["Date_dt"].dt.month.isin(month_to_season[season])]
        if filtered.empty:
            st.warning("No data available for the selected filter.")
        else:
            st.dataframe(filtered)

# --- Display temperature trends ---
elif menu == "Display temperature trends":
    st.header("📈 Temperature Trends")
    if data.empty:
        st.warning("No data available.")
    else:
        st.line_chart(data.set_index(pd.to_datetime(data["Date"]))["Temperature"])

# --- Predict tomorrow's weather ---
elif menu == "Predict tomorrow's weather":
    st.header("🔮 Tomorrow's Weather Prediction")
    prediction = predict_tomorrow(data)
    st.write(prediction)

# --- Compare yearly data ---
elif menu == "Compare yearly data":
    st.header("📅 Yearly Data Comparison")
    if data.empty:
        st.warning("No data available.")
    else:
        data["Year"] = pd.to_datetime(data["Date"]).dt.year
        yearly_avg = data.groupby("Year")["Temperature"].mean()
        st.bar_chart(yearly_avg)

# --- Record-breaking temperatures/conditions ---
elif menu == "Record-breaking temperatures/conditions":
    st.header("🏆 Record-Breaking Data")
    if data.empty:
        st.warning("No data available.")
    else:
        max_temp = data["Temperature"].max()
        min_temp = data["Temperature"].min()
        st.write(f"Highest Temperature Recorded: {max_temp}°C")
        st.write(f"Lowest Temperature Recorded: {min_temp}°C")
        st.write("Extreme Weather Conditions:")
        extreme_conditions = data[data["Condition"].isin(["Stormy", "Snowy"])]
        st.dataframe(extreme_conditions)
