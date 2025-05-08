import pandas as pd

# Load dataset (replace 'data.csv' with your actual dataset)
df = pd.read_csv("data.csv")

# Display basic information
print(df.info())  # Shows column types and missing values
print(df.head())  # View first few rows
# Check for missing values
print(df.isnull().sum())

# Fill or drop missing values (depending on data)
df.fillna(df.mean(), inplace=True)  # Example: Fill missing numeric values with column mean
# Summary statistics
print(df.describe())  # Includes mean, median, std, etc.
print(df.groupby('Category')['sales'].mean())  # Replace 'Category' and 'sales' with relevant columns
import matplotlib.pyplot as plt

df['date'] = pd.to_datetime(df['date'])  # Ensure date column is in datetime format
plt.plot(df['date'], df['sales'], marker='o', linestyle='-', color='green')
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Sales Trend Over Time")
plt.xticks(rotation=45)
plt.show()
df.groupby('Category')['sales'].mean().plot(kind='bar', color='blue')
plt.xlabel("Category")
plt.ylabel("Average Sales")
plt.title("Average Sales per Category")
plt.show()
plt.hist(df['sales'], bins=10, color='purple', edgecolor='black')
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.title("Sales Distribution")
plt.show()
plt.scatter(df['sepal_length'], df['petal_length'], color='red')
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.title("Sepal vs Petal Length Relationship")
plt.show()