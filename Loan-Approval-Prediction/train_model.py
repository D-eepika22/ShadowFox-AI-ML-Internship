import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
df=pd.read_csv("C:/Users/Deepika/Desktop/Shadowfox project/Loan-Approval-Prediction/data/train_u6lujuX_CVtuZ9i.csv")
print(df.head())
print("\n data information")
print(df.info())
print("\n Statistical Summary")
print(df.describe())
print("\n missing values")
print(df.isnull().sum())
df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
df["Married"] = df["Married"].fillna(df["Married"].mode()[0])
df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0])
df["Self_Employed"] = df["Self_Employed"].fillna(df["Self_Employed"].mode()[0])

df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0])
df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])
print(df.head())
print(df[df["Gender"].isnull()])
print("\nMissing Values After Cleaning:")
print("Mode Gender:",df["Gender"].mode()[0])
df["Gender"]=df["Gender"].fillna(df["Gender"].mode()[0])
print("Gender Missing:",df["Gender"].isnull().sum())
print(df.head(10))
print(df["Gender"].isnull().sum())
print(df["LoanAmount"].isnull().sum())
print(df.isnull().sum())

plt.figure(figsize=(6,4))
sns.countplot(x="Loan_Status", data=df)
plt.title("Loan Approval Distribution")
plt.show()
plt.figure(figsize=(6,4))
sns.countplot(x="Gender", data=df)
plt.title("Gender Distribution")
plt.show()
plt.figure(figsize=(6,4))
sns.countplot(x="Education", data=df)
plt.title("Education Distribution")
plt.show()
plt.figure(figsize=(6,4))
sns.countplot(x="Property_Area", data=df)
plt.title("Property Area")
plt.show()
plt.figure(figsize=(8,5))
plt.hist(df["ApplicantIncome"], bins=30)
plt.title("Applicant Income Distribution")
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.show()
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

categorical_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

# Encode categorical columns
for column in categorical_columns:
    df[column] = encoder.fit_transform(df[column])

# Drop Loan_ID only ONCE
df.drop("Loan_ID", axis=1, inplace=True)

print(df.head())

from sklearn.model_selection import train_test_split

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)
lr=LogisticRegression(max_iter=1000)
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print("logistic regression Accuracy",accuracy)
cm=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
plt.xlabel("Confusion Matrix")
plt.ylabel("Actual")
plt.show()
print(classification_report(y_test,y_pred))

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("\nModel Comparison")
print("----------------------------")
print("Logistic Regression :", accuracy_score(y_test, y_pred))
print("Decision Tree       :", accuracy_score(y_test, dt_pred))
print("Random Forest       :", accuracy_score(y_test, rf_pred))
joblib.dump(rf,"model/loan_model.pkl")
joblib.dump(lr, "model/loan_model.pkl")
joblib.dump(dt, "model/loan_model.pkl")
print("model saved sucessfully")