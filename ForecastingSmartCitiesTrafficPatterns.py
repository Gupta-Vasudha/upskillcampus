import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

# Load datasets
train_df = pd.read_csv("/content/drive/MyDrive/Tsp internship/train_aWnotuB.csv")
test_df = pd.read_csv("/content/drive/MyDrive/Tsp internship/datasets_8494_11879_test_BdBKkAj1.csv")

# Convert DateTime to datetime format
train_df["DateTime"] = pd.to_datetime(train_df["DateTime"], errors="coerce")
test_df["DateTime"] = pd.to_datetime(test_df["DateTime"], errors="coerce")

# automatically fetches Indian holidays
def is_holiday(date):
    india_holidays = holidays.India(years=date.year)
    return 1 if date in india_holidays else 0

# Extract features from DateTime
def extract_features(df):
    df["Hour"] = df["DateTime"].dt.hour
    df["Day"] = df["DateTime"].dt.day
    df["Month"] = df["DateTime"].dt.month
    df["Weekday"] = df["DateTime"].dt.weekday
    df["Holiday"] = df["DateTime"].apply(is_holiday)
    return df

train_df = extract_features(train_df)
test_df = extract_features(test_df)

# Drop DateTime column
train_df.drop(columns=["DateTime"], inplace=True)
test_df.drop(columns=["DateTime"], inplace=True)

# 1. Traffic Peak Analysis (Hourly Patterns)
plt.figure(figsize=(10, 5))
hourly_traffic = train_df.groupby("Hour")["Vehicles"].mean()
sns.lineplot(x=hourly_traffic.index, y=hourly_traffic.values)
plt.title("Average Traffic Volume by Hour")
plt.xlabel("Hour of the Day")
plt.ylabel("Average Vehicles")
plt.show()

# 2. Holiday vs. Non-Holiday Traffic Analysis
holiday_traffic = train_df.groupby("Holiday")["Vehicles"].mean()
print(f"Avg. Vehicles on Holidays: {holiday_traffic.get(1, 0):.2f}")
print(f"Avg. Vehicles on Regular Days: {holiday_traffic.get(0, 0):.2f}")

# 3. Junction-Specific Traffic Trends
plt.figure(figsize=(10, 5))
sns.boxplot(x=train_df["Junction"], y=train_df["Vehicles"])
plt.title("Traffic Volume Across Junctions")
plt.xlabel("Junction")
plt.ylabel("Vehicles")
plt.show()

# 4. Adding a Moving Average Feature for Trend Analysis
train_df["Moving_Avg"] = train_df.groupby("Junction")["Vehicles"].transform(lambda x: x.rolling(window=24, min_periods=1).mean())

test_df["Moving_Avg"] = test_df.groupby("Junction")["ID"].transform(lambda x: x.rolling(window=24, min_periods=1).mean())

# Prepare data for training
X = train_df.drop(columns=["Vehicles", "ID"])
y = train_df["Vehicles"]

# Split into training & validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost Model
model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# Validate the model
y_pred = model.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Predict for test dataset
X_test = test_df.drop(columns=["ID"])
test_df["Vehicles"] = model.predict(X_test).round().astype(int) 

# Save predictions
test_df.to_csv("test_predictions_xgboost.csv", index=False)
print("Predictions saved to test_predictions_xgboost.csv")
