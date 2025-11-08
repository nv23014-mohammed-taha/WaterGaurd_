
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import base64

# ------------------------------
# 🎨 CUSTOM HTML + CSS TOP NAVBAR
# ------------------------------
st.markdown("""
    <style>
    /* Hide default Streamlit elements */
    #MainMenu, footer, header {visibility: hidden;}

    /* Top navigation bar styling */
    .topnav {
      background-color: #004aad;
      overflow: hidden;
      border-radius: 0px 0px 15px 15px;
      box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }

    .topnav a {
      float: left;
      color: white;
      text-align: center;
      padding: 16px 26px;
      text-decoration: none;
      font-size: 17px;
      transition: background 0.3s;
    }

    .topnav a:hover {
      background-color: #1b6ef3;
      color: white;
    }

    .topnav a.active {
      background-color: #012a73;
      color: white;
      font-weight: bold;
    }

    h1 {
      color: #004aad;
      text-align: center;
      padding-top: 10px;
    }

    .dataframe th, .dataframe td {
      text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------
# 🧭 TOP NAVIGATION
# ------------------------------
st.markdown("""
<div class="topnav">
  <a href="?page=Home" class="active">🏠 Home</a>
  <a href="?page=Statistics">📊 Statistics</a>
  <a href="?page=About">ℹ️ About</a>
</div>
""", unsafe_allow_html=True)

# Get page from query
query_params = st.query_params
page = query_params.get("page", ["Home"])[0]

# ------------------------------
# 📆 CREATE 1-YEAR WEATHER DATASET
# ------------------------------
def generate_weather_data():
    np.random.seed(42)
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2024, 12, 31)
    dates = pd.date_range(start_date, end_date)

    temps = np.random.normal(28, 5, len(dates))
    humidity = np.random.randint(40, 95, len(dates))
    wind_speed = np.random.uniform(0.5, 7.5, len(dates))
    conditions = np.random.choice(
        ['Sunny', 'Cloudy', 'Rainy', 'Stormy', 'Foggy'], len(dates)
    )

    df = pd.DataFrame({
        'Date': dates,
        'Temperature (°C)': temps.round(1),
        'Humidity (%)': humidity,
        'Wind Speed (m/s)': wind_speed.round(2),
        'Condition': conditions
    })
    return df

weather_df = generate_weather_data()

# ------------------------------
# 🏠 HOME PAGE
# ------------------------------
if page == "Home":
    st.title("🌦️ Weather Tracker Dashboard")
    st.markdown("### Welcome to your annual weather tracking system!")

    st.write("Use this tool to explore and analyze one year of synthetic weather data.")
    st.dataframe(weather_df.head(10))

# ------------------------------
# 📊 STATISTICS PAGE
# ------------------------------
elif page == "Statistics":
    st.title("📊 Weather Data Statistics")
    st.markdown("### Visualize and Explore Trends")

    st.write("#### Summary Statistics")
    st.write(weather_df.describe())

    st.line_chart(weather_df.set_index('Date')[['Temperature (°C)', 'Humidity (%)']])

    condition_counts = weather_df['Condition'].value_counts()
    st.bar_chart(condition_counts)

# ------------------------------
# ℹ️ ABOUT PAGE
# ------------------------------
elif page == "About":
    st.title("ℹ️ About This Application")
    st.markdown("""
    This web application was created with **Streamlit** and enhanced using **custom HTML/CSS**.
    
    - **Purpose:** Display and analyze yearly weather data  
    - **Features:**
        - Interactive charts  
        - Clean top navigation bar  
        - Built-in dataset for 2024  
        - No sidebar — full-screen layout for a modern feel
    
    💡 Built by Mohammed Taha.
    """)

# ------------------------------
# END OF APP
# ------------------------------

import kagglehub
vijaygiitk_multiclass_weather_dataset_path = kagglehub.dataset_download('vijaygiitk/multiclass-weather-dataset')

print('Data source import complete.')
import os

path = kagglehub.dataset_download("vijaygiitk/multiclass-weather-dataset")

print("Path to dataset files:", path)

import csv
import os
from datetime import datetime
from statistics import mean, mode
import pandas as pd

import pandas as pd
import random
from datetime import datetime, timedelta


start_date = datetime(2025, 1, 1)
dates = [(start_date + timedelta(days=i)).strftime("%m-%d-%Y") for i in range(365)]


conditions = ["Sunny", "Cloudy", "Rainy", "Stormy", "Windy", "Foggy"]


data = {
    "Date": dates,
    "Temperature": [random.randint(15, 40) for _ in range(365)],      # °C
    "Condition": [random.choice(conditions) for _ in range(365)],
    "Humidity": [random.randint(40, 90) for _ in range(365)],         # %
    "WindSpeed": [random.randint(5, 25) for _ in range(365)]          # km/h
}

df = pd.DataFrame(data)


df.to_csv("weather_data.csv", index=False)

print("✅ 1-Year (2025) Weather Dataset Created Successfully!")
print(f"Total Records: {len(df)}")
df.head(10)


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import random
import cv2
import tqdm as tqdm
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


root_dir = "/kaggle/input/multiclass-weather-dataset"
os.listdir(root_dir)
os.path.exists(root_dir)

# Cell A: Create a 1-year (2025) seasonally realistic weather dataset
import pandas as pd
import random
from datetime import datetime, timedelta

def generate_seasonal_temp(month):
    """Return a plausible temperature range for a month (Celsius)."""
    # Simple hemispheric model (adjust to your location if needed)
    # Jan (1) -> winter-like, Jul (7) -> summer-like (this is generic)
    month = int(month)
    if month in (12, 1, 2):   # winter
        return random.randint(8, 20)
    if month in (3, 4, 5):    # spring
        return random.randint(15, 26)
    if month in (6, 7, 8):    # summer
        return random.randint(25, 40)
    if month in (9, 10, 11):  # autumn
        return random.randint(18, 30)
    return random.randint(15, 30)

start_date = datetime(2025, 1, 1)
dates = [(start_date + timedelta(days=i)) for i in range(365)]
conditions = ["Sunny", "Cloudy", "Rainy", "Stormy", "Windy", "Foggy"]

rows = []
for dt in dates:
    month = dt.month
    temp = generate_seasonal_temp(month)
    # add some daily noise
    temp = int(temp + random.gauss(0, 2))
    # humidity tends to be higher when rainy/stormy/foggy
    cond = random.choices(
        conditions,
        weights=[40, 25, 20, 5, 6, 4],  # bias toward Sunny/Cloudy in generic locales
        k=1
    )[0]
    if cond == "Rainy":
        humidity = random.randint(70, 95)
    elif cond == "Stormy":
        humidity = random.randint(75, 98)
    elif cond == "Foggy":
        humidity = random.randint(80, 95)
    else:
        humidity = random.randint(35, 80)
    wind = random.randint(3, 30)
    rows.append({
        "Date": dt.strftime("%m-%d-%Y"),
        "Temperature": temp,
        "Condition": cond,
        "Humidity": humidity,
        "WindSpeed": wind
    })

df_year = pd.DataFrame(rows)
df_year.to_csv("weather_data.csv", index=False)
print("✅ Created 'weather_data.csv' with", len(df_year), "rows (2025).")
df_year.head()

foggy = "/kaggle/input/multiclass-weather-dataset/dataset/foggy"
sunrise = "/kaggle/input/multiclass-weather-dataset/dataset/sunrise"
shine = "/kaggle/input/multiclass-weather-dataset/dataset/shine"
rainy = "/kaggle/input/multiclass-weather-dataset/dataset/rainy"
cloudy = "/kaggle/input/multiclass-weather-dataset/dataset/cloudy"
test = "/kaggle/input/multiclass-weather-dataset/dataset/alien_test"

print("Number of Images in Each Directory:")
print(f"Foggy: {len(os.listdir(foggy))}")
print(f"Sunrise: {len(os.listdir(sunrise))}")
print(f"Shine: {len(os.listdir(shine))}")
print(f"Rainy: {len(os.listdir(rainy))}")
print(f"Cloudy: {len(os.listdir(cloudy))}")

x = []
y = []
dataset =[]
def create_dataset(directory,dir_name):
    for i in tqdm.tqdm(os.listdir(directory)):
        full_path = os.path.join(directory,i)
        try:
            img = cv2.imread(full_path)
            img = cv2.resize(img,(150,150))
        except:
            continue
        x.append(img)
        y.append(dir_name)
    return x,y

x,y= create_dataset(foggy,"foggy")
x,y= create_dataset(sunrise,"sunrise")
x,y= create_dataset(shine,"shine")
x,y= create_dataset(rainy,"rainy")
x,y= create_dataset(cloudy,"cloudy")

x =  np.array(x)
y = np.array(y)
x.shape,y.shape

import seaborn as sns
plt.figure(figsize=(9,7))
plt.style.use("fivethirtyeight")
sns.countplot(y)
plt.show()

fig = plt.figure(figsize=(12,7))
for i in range(15):
    sample =  random.choice(range(len(x)))
    image = x[sample]
    category = y[sample]
    plt.subplot(3,5,i+1)
    plt.subplots_adjust(hspace=0.3)
    plt.imshow(image)
    plt.xlabel(category)

plt.tight_layout()
plt.show()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

img_size =150

x_train = np.array(x_train)/255.0
x_test = np.array(x_test)/255.0


x_train = x_train.reshape(-1,img_size,img_size,3)
y_train = np.array(y_train)

x_test = x_test.reshape(-1,img_size,img_size,3)
y_test = np.array(y_test)

from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
y_train_lb = lb.fit_transform(y_train)
y_test_lb = lb.fit_transform(y_test)

y_train_lb.shape,y_test_lb.shape

from tensorflow.keras.applications.vgg19 import VGG19
vgg = VGG19(weights = "imagenet",include_top=False,input_shape=(img_size,img_size,3))

for layer in vgg.layers:
    layer.trainable = False

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense
model =Sequential()
model.add(vgg)
model.add(Flatten())
model.add(Dense(5,activation="softmax"))
model.summary()

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint("vgg19.h5",
                             monitor="val_accuracy",
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=False)

earlystop = EarlyStopping(monitor="val_accuracy",
                          patience=5,
                          verbose=1)

unique,counts = np.unique(y_train_lb,return_counts=True)
print(unique,counts)

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])
batch_size = 32
history = model.fit(x_train, y_train_lb,
                    epochs=15,
                    validation_data=(x_test, y_test_lb),
                    batch_size=batch_size,
                    verbose=1,
                    callbacks=[checkpoint, earlystop])

loss,accuracy = model.evaluate(x_test,y_test_lb)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")
y_pred = model.predict_classes(x_test)
y_pred[:15]
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
cm = confusion_matrix(y_test,y_pred)
plot_confusion_matrix(conf_mat = cm,figsize=(8,7),class_names = ["cloudy","foggy","rainy","shine","sunrise"],
                      show_normed = True);

plt.style.use("ggplot")
fig = plt.figure(figsize=(12,6))
epochs = range(1,16)
plt.subplot(1,2,1)
plt.plot(epochs,history.history["accuracy"],"go-")
plt.plot(epochs,history.history["val_accuracy"],"ro-")
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Train","val"],loc = "upper left")

plt.subplot(1,2,2)
plt.plot(epochs,history.history["loss"],"go-")
plt.plot(epochs,history.history["val_loss"],"ro-")
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Train","val"],loc = "upper left")
plt.show()
plt.figure(figsize=(12,9))
plt.style.use("ggplot")
for i in range(10):
    sample = random.choice(range(len(x_test)))
    plt.subplot(2,5,i+1)
    plt.subplots_adjust(hspace=0.3)
    plt.imshow(x_test[sample])
    plt.xlabel(f"Actual: {y_test[sample]}\n Predicted: {y_pred[sample]}")

plt.tight_layout()
plt.show()
