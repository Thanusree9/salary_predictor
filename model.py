import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
data = pd.read_csv("employee_data.csv")
X = data.drop("Salary", axis=1)
y = data["Salary"]

# Define preprocessing
categorical = ["Education Level", "Role", "Department"]
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(), categorical)
], remainder="passthrough")

# Create model pipeline
model = Pipeline([
    ("preprocess", preprocessor),
    ("regressor", LinearRegression())
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "salary_model.pkl")
print("Model trained and saved successfully.")
