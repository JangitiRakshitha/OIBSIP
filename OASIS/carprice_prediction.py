# Car Price Prediction Project

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# 2. Load Dataset
df = pd.read_csv("car data.csv")

# Display first rows
print(df.head())

# Dataset info
print(df.info())

# 3. Data Preprocessing
# Remove unnecessary column
df.drop(['Car_Name'], axis=1, inplace=True)

# Create Car Age feature
df['Current_Year'] = 2026
df['Car_Age'] = df['Current_Year'] - df['Year']

# Remove old columns
df.drop(['Year','Current_Year'], axis=1, inplace=True)

# 4. Convert Categorical Data
df = pd.get_dummies(df, drop_first=True)

print(df.head())

# 5. Define Features and Target
X = df.drop(['Selling_Price'], axis=1)
y = df['Selling_Price']

# 6. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 7. Train Model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# 8. Prediction
predictions = model.predict(X_test)

# 9. Model Evaluation
print("R2 Score:", r2_score(y_test, predictions))
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))

# 10. Visualization
plt.scatter(y_test, predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Price")
plt.show()

# 11. Predict Price for New Car
# Example values: Present_Price, Driven_kms, Owner, Car_Age, Fuel_Type_Diesel, Fuel_Type_Petrol, Seller_Type_Individual, Transmission_Manual
new_car = [[5.59, 27000, 0, 10, 0, 1, 1, 1]]

predicted_price = model.predict(new_car)

print("Predicted Car Price:", predicted_price)