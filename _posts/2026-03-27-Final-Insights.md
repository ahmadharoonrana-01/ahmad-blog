layout: post
Title: 5 Beyond the Code: What the Walmart Project Taught Me
date: 2026-03-27
categories: tutorial
The project was a success! My model could predict sales trends with high accuracy. But more than that, I learned the importance of Feature Engineering. I realized that holidays like Thanksgiving have a massive "spike" effect that simple models miss. This project gave me the confidence to move from data analysis to full-scale development.

'''Impact of Seasonal Trends and Holidays on Retail Performance'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Loading file / reading  file
sales_df = pd.read_csv("walmart sales prediction.csv")

# Changing date to computer format type:
sales_df['Date'] = pd.to_datetime(sales_df['Date'])

# Making colums of month and year from date:
sales_df['Month'] = sales_df['Date'].dt.month
sales_df['Year'] = sales_df['Date'].dt.year

# Making column of high/medium/low from weekly sales
def categorize_sales(sale):
    if sale < 2000:
        return 'Low'
    elif sale < 20000:
        return 'Medium'
    else:
        return 'High'

sales_df['label'] = sales_df['Weekly_Sales'].apply(categorize_sales)

# Printing first few rows
print(sales_df.head())

print(sales_df.info())

print(sales_df.describe(include="all"))

# Histogram plot from month column
plt.hist(sales_df["Month"])
plt.title("Data Distribution by Month")
plt.xlabel("Month")
plt.ylabel("Record Count")
plt.show()

# Count plot from label(medium/low/high)
sns.countplot(x = sales_df["label"])
plt.title("Class Distribution (Sales Levels)")
plt.show()

# Box plot between label and dept
sns.boxplot(data=sales_df, x="label", y="Dept")
plt.title("Department Distribution across Sales Levels")
plt.show()

# Label encoding for changing categorical data into numerical data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
sales_df["label"] = le.fit_transform(sales_df["label"])
sales_df["IsHoliday"] = le.fit_transform(sales_df["IsHoliday"])

# One hot encoding(used for same purpose)
sales_df = sales_df.drop("Date", axis=1)
sales_df = pd.get_dummies(sales_df, drop_first=True)
print(sales_df.columns)

# Train test split for target variable
X = sales_df.drop(["label","Weekly_Sales"], axis=1)
y = sales_df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision tree classifier
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)

# for saving data in model
joblib.dump(dt_clf, "sales_class_dt_model.pkl")
loaded_dt = joblib.load("sales_class_dt_model.pkl")

# For loading data
y_pred = loaded_dt.predict(X_test)
print(f"The prediction about dataset is  {y_pred[:15]}")

# For accuracy score of the model
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix used
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix for Sales Categories")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
