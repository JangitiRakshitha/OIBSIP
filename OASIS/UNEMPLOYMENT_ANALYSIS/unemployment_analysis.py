
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Unemployment in India.csv")
print("\nFirst 5 rows of the dataset:")
print(df.head())
print("\nDataset Information:")
print(df.info())

df.columns = [
    "State",
    "Date",
    "Frequency",
    "Unemployment_Rate",
    "Employed",
    "Labour_Participation_Rate",
    "Region"
]
print("\nMissing values in dataset:")
print(df.isnull().sum())
df = df.dropna()
df["Date"] = pd.to_datetime(df["Date"])
print("\nDescriptive Statistics:")
print(df.describe())
plt.figure(figsize=(12, 6))
sns.lineplot(x="Date", y="Unemployment_Rate", data=df)
plt.title("Unemployment Rate Over Time in India")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.show()
covid_df = df[(df["Date"] >= "2020-03-01") & (df["Date"] <= "2020-12-31")]

plt.figure(figsize=(12, 6))
sns.lineplot(x="Date", y="Unemployment_Rate", data=covid_df)
plt.title("Unemployment Rate During Covid-19")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.show()
pre_covid = df[df["Date"] < "2020-03-01"]
covid = df[df["Date"] >= "2020-03-01"]

print("\nAverage Unemployment Rate (Pre-Covid):", pre_covid["Unemployment_Rate"].mean())
print("Average Unemployment Rate (Covid):", covid["Unemployment_Rate"].mean())

state_avg = df.groupby("State")["Unemployment_Rate"].mean().sort_values(ascending=False)

plt.figure(figsize=(14, 7))
state_avg.plot(kind="bar")
plt.title("Average Unemployment Rate by State")
plt.xlabel("State")
plt.ylabel("Unemployment Rate (%)")
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x="Region", y="Unemployment_Rate", data=df)
plt.title("Urban vs Rural Unemployment Rate Comparison")
plt.xlabel("Region")
plt.ylabel("Unemployment Rate (%)")
plt.show()

plt.figure(figsize=(12, 6))
sns.lineplot(x="Date", y="Labour_Participation_Rate", data=df)
plt.title("Labour Participation Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Labour Participation Rate (%)")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

print("\nPROJECT INSIGHTS:")
print("1. Unemployment rate increased sharply during early Covid-19 months.")
print("2. Urban areas experienced higher unemployment compared to rural areas.")
print("3. Some states consistently show higher unemployment rates.")
print("4. Labour participation rate dropped during lockdown, indicating discouraged workers.")

print("\n--- Unemployment Analysis Completed Successfully ---")

