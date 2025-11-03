# -*- coding: utf-8 -*-
# Weather Tracker App (Your Dataset Version)
# Compatible with Streamlit 1.39+
# Developed for top-tier analysis

import streamlit as st
import pandas as pd
import altair as alt

# ----------------------------- #
# 🎯 APP CONFIGURATION
# ----------------------------- #
st.set_page_config(
    page_title="Weather Tracker",
    page_icon="🌤️",
    layout="wide",
)

st.title("🌦️ Top-Tier Weather Tracker")
st.write("Analyze and visualize your weather dataset interactively!")

# ----------------------------- #
# 📂 LOAD DATA
# ----------------------------- #
uploaded_file = st.file_uploader("Upload your Weather.csv file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Try to parse date columns if they exist
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    st.success("✅ Dataset loaded successfully!")

    # ----------------------------- #
    # 📊 SHOW RAW DATA
    # ----------------------------- #
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # ----------------------------- #
    # 📅 FILTER BY YEAR OR MONTH
    # ----------------------------- #
    if "date" in df.columns:
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month_name()

        years = sorted(df["year"].dropna().unique())
        selected_years = st.multiselect("Select years to analyze:", years, default=years)

        filtered_df = df[df["year"].isin(selected_years)]
    else:
        filtered_df = df

    # ----------------------------- #
    # 🔢 METRICS / STATISTICS
    # ----------------------------- #
    st.subheader("📈 Weather Summary")

    if all(col in df.columns for col in ["temp_max", "temp_min", "precipitation", "wind"]):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🌡️ Max Temp", f"{filtered_df['temp_max'].max():.1f}°C")
        with col2:
            st.metric("❄️ Min Temp", f"{filtered_df['temp_min'].min():.1f}°C")
        with col3:
            st.metric("💧 Max Precipitation", f"{filtered_df['precipitation'].max():.1f} mm")
        with col4:
            st.metric("🌬️ Max Wind Speed", f"{filtered_df['wind'].max():.1f} m/s")

    # ----------------------------- #
    # 📉 VISUALIZATIONS
    # ----------------------------- #
    st.subheader("🌤️ Visual Analysis")

    # Temperature Over Time
    if all(col in df.columns for col in ["date", "temp_max", "temp_min"]):
        st.markdown("### Temperature Range Over Time")
        temp_chart = (
            alt.Chart(filtered_df)
            .mark_area(opacity=0.4)
            .encode(
                x="date:T",
                y="temp_max:Q",
                y2="temp_min:Q",
                color="year:N"
            )
        )
        st.altair_chart(temp_chart, use_container_width=True)

    # Precipitation Distribution
    if "precipitation" in df.columns:
        st.markdown("### Precipitation Distribution by Month")
        precip_chart = (
            alt.Chart(filtered_df)
            .mark_bar()
            .encode(
                x="month:N",
                y="sum(precipitation):Q",
                color="year:N",
            )
        )
        st.altair_chart(precip_chart, use_container_width=True)

    # Weather Type Breakdown
    if "weather" in df.columns:
        st.markdown("### Weather Type Breakdown")
        weather_chart = (
            alt.Chart(filtered_df)
            .mark_arc()
            .encode(
                theta="count():Q",
                color="weather:N"
            )
        )
        st.altair_chart(weather_chart, use_container_width=True)

    # Wind Trend
    if all(col in df.columns for col in ["date", "wind"]):
        st.markdown("### Wind Speed Over Time")
        wind_chart = (
            alt.Chart(filtered_df)
            .mark_line()
            .encode(
                x="date:T",
                y="wind:Q",
                color="year:N"
            )
        )
        st.altair_chart(wind_chart, use_container_width=True)

else:
    st.info("👆 Please upload your Weather.csv file to begin.")
