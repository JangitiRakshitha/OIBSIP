# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("Unemployment in India.csv")

# Display first rows
print(data.head())

# Check column names
print(data.columns)

# Remove extra spaces from column names
data.columns = data.columns.str.strip()

# Check dataset info
print(data.info())

# Convert Date column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Display summary statistics
print(data.describe())

# -----------------------------
# 1. Average Unemployment Rate
# -----------------------------
avg_unemployment = data['Estimated Unemployment Rate (%)'].mean()
print("Average Unemployment Rate:", avg_unemployment)

# -----------------------------
# 2. Unemployment by Region
# -----------------------------
region_unemployment = data.groupby(
    'Region')['Estimated Unemployment Rate (%)'].mean()
print(region_unemployment)

# -----------------------------
# 3. Plot Unemployment Over Time
# -----------------------------
plt.figure(figsize=(10, 6))
sns.lineplot(x='Date', y='Estimated Unemployment Rate (%)', data=data)
plt.title("Unemployment Rate Over Time in India")
plt.xticks(rotation=45)
plt.show()

# -----------------------------
# 4. Unemployment by Region
# -----------------------------
plt.figure(figsize=(12, 6))
sns.barplot(x='Region', y='Estimated Unemployment Rate (%)', data=data)
plt.xticks(rotation=90)
plt.title("Unemployment Rate by Region")
plt.show()

# -----------------------------
# 5. Heatmap (Region vs Date)
# -----------------------------
pivot = data.pivot_table(values='Estimated Unemployment Rate (%)',
                         index='Region',
                         columns='Date')

plt.figure(figsize=(12, 8))
sns.heatmap(pivot, cmap="coolwarm")
plt.title("Unemployment Heatmap")
plt.show()
