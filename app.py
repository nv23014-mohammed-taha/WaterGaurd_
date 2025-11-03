import csv
import os
from datetime import datetime
from statistics import mean
from collections import Counter

FILENAME = "Weather.csv"

# --------------------- LOAD AND SAVE DATA ---------------------

def load_data(filename):
    data = []
    if os.path.exists(filename):
        with open(filename, mode="r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
    return data


def save_data(data, filename):
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        fieldnames = ["Date", "Temperature (°C)", "Condition", "Humidity (%)", "Wind Speed (km/h)"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

# --------------------- RECORD NEW OBSERVATION ---------------------

def record_observation(data):
    try:
        date = input("Enter date (MM-DD-YYYY): ")
        datetime.strptime(date, "%m-%d-%Y")  # Validate format

        temp = float(input("Enter temperature (°C): "))
        condition = input("Enter condition (Sunny, Cloudy, Rainy, etc.): ").capitalize()
        humidity = float(input("Enter humidity (%): "))
        wind_speed = float(input("Enter wind speed (km/h): "))

        data.append({
            "Date": date,
            "Temperature (°C)": temp,
            "Condition": condition,
            "Humidity (%)": humidity,
            "Wind Speed (km/h)": wind_speed
        })
        print("✅ Observation recorded successfully!")
    except ValueError:
        print("❌ Invalid input! Please try again.")

# --------------------- VIEW STATISTICS ---------------------

def view_statistics(data):
    if not data:
        print("No data available.")
        return

    temps = [float(row["Temperature (°C)"]) for row in data]
    avg_temp = mean(temps)
    min_temp = min(temps)
    max_temp = max(temps)

    conditions = [row["Condition"] for row in data]
    most_common = Counter(conditions).most_common(1)[0][0]

    print("\n🌡️ Weather Statistics:")
    print(f"Average Temperature: {avg_temp:.2f}°C")
    print(f"Minimum Temperature: {min_temp:.2f}°C")
    print(f"Maximum Temperature: {max_temp:.2f}°C")
    print(f"Most Common Condition: {most_common}")

# --------------------- SEARCH BY DATE ---------------------

def search_by_date(data):
    date = input("Enter date to search (MM-DD-YYYY): ")
    results = [row for row in data if row["Date"] == date]

    if results:
        print(f"\n📅 Observations for {date}:")
        for r in results:
            print(r)
    else:
        print("No records found for that date.")

# --------------------- VIEW ALL OBSERVATIONS ---------------------

def view_all(data):
    if not data:
        print("No observations recorded.")
        return

    print("\n📋 All Observations:")
    print(f"{'Date':<12} {'Temp(°C)':<10} {'Condition':<10} {'Humidity(%)':<12} {'Wind(km/h)':<10}")
    print("-" * 60)
    for row in data:
        print(f"{row['Date']:<12} {row['Temperature (°C)']:<10} {row['Condition']:<10} "
              f"{row['Humidity (%)']:<12} {row['Wind Speed (km/h)']:<10}")

# --------------------- FILTER BY MONTH / SEASON ---------------------

def filter_by_month_or_season(data):
    choice = input("Filter by (M)onth or (S)eason? ").strip().lower()
    if choice == 'm':
        month = input("Enter month number (1-12): ").zfill(2)
        filtered = [row for row in data if row["Date"].startswith(month)]
    elif choice == 's':
        season = input("Enter season (Winter, Spring, Summer, Autumn): ").capitalize()
        months_by_season = {
            "Winter": ["12", "01", "02"],
            "Spring": ["03", "04", "05"],
            "Summer": ["06", "07", "08"],
            "Autumn": ["09", "10", "11"]
        }
        filtered = [row for row in data if row["Date"][:2] in months_by_season.get(season, [])]
    else:
        print("Invalid choice.")
        return

    if filtered:
        print("\nFiltered Observations:")
        for row in filtered:
            print(row)
    else:
        print("No data found for selected period.")

# --------------------- TEMPERATURE TREND GRAPH ---------------------

def display_trend(data):
    if not data:
        print("No data available for trend.")
        return
    print("\n📈 Temperature Trend:")
    for row in data:
        bar = "*" * int(float(row["Temperature (°C)"]))
        print(f"{row['Date']}: {bar}")

# --------------------- MAIN PROGRAM ---------------------

def main():
    data = load_data(FILENAME)
    while True:
        print("\n🌤️ Weather Tracker Menu:")
        print("1. Record a new observation")
        print("2. View weather statistics")
        print("3. Search observations by date")
        print("4. View all observations")
        print("5. Filter by month or season")
        print("6. Display temperature trend")
        print("7. Exit")

        choice = input("Enter your choice: ").strip()
        if choice == "1":
            record_observation(data)
        elif choice == "2":
            view_statistics(data)
        elif choice == "3":
            search_by_date(data)
        elif choice == "4":
            view_all(data)
        elif choice == "5":
            filter_by_month_or_season(data)
        elif choice == "6":
            display_trend(data)
        elif choice == "7":
            save_data(data, FILENAME)
            print("Data saved successfully. Exiting program. 👋")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
