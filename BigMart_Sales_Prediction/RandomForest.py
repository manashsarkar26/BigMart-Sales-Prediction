# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# Load data
train = pd.read_csv("train_v9rqX0R.csv")
test = pd.read_csv("test_AbJTz2l.csv")

# Handle missing values
train["Item_Weight"].fillna(train["Item_Weight"].mean(), inplace=True)
test["Item_Weight"].fillna(train["Item_Weight"].mean(), inplace=True)

train["Outlet_Size"].fillna(train["Outlet_Size"].mode()[0], inplace=True)
test["Outlet_Size"].fillna(train["Outlet_Size"].mode()[0], inplace=True)

# Fix inconsistent categories
train["Item_Fat_Content"] = train["Item_Fat_Content"].replace({"LF":"Low Fat", "low fat":"Low Fat", "reg":"Regular"})
test["Item_Fat_Content"] = test["Item_Fat_Content"].replace({"LF":"Low Fat", "low fat":"Low Fat", "reg":"Regular"})

# Encode categorical variables
le = LabelEncoder()
for col in ["Item_Identifier", "Item_Fat_Content", "Item_Type", "Outlet_Identifier",
            "Outlet_Size", "Outlet_Location_Type", "Outlet_Type"]:
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

# Features and target
X = train.drop(["Item_Outlet_Sales"], axis=1)
y = train["Item_Outlet_Sales"]

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_val)

# Evaluation
r2 = r2_score(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mae = mean_absolute_error(y_val, y_pred)

print("R2 Score:", r2)
print("RMSE:", rmse)
print("MAE:", mae)

# Predict on test data
test_predictions = model.predict(test)

# Save submission
submission = pd.DataFrame({
    "Item_Identifier": pd.read_csv("test_AbJTz2l.csv")["Item_Identifier"],
    "Outlet_Identifier": pd.read_csv("test_AbJTz2l.csv")["Outlet_Identifier"],
    "Item_Outlet_Sales": test_predictions
})
submission.to_csv("Sales_predictions.csv", index=False)
print("Submission file saved as Sales_predictions.csv")



