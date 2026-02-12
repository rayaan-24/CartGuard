import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Step 1: Generate Fake Retail Data
np.random.seed(42)

data_size = 1000

data = pd.DataFrame({
    "time_spent": np.random.randint(1, 30, data_size),  # minutes
    "cart_value": np.random.randint(100, 5000, data_size),
    "num_items": np.random.randint(1, 10, data_size),
    "logged_in": np.random.randint(0, 2, data_size),
    "discount_applied": np.random.randint(0, 2, data_size),
    "previous_purchases": np.random.randint(0, 20, data_size),
    "device_mobile": np.random.randint(0, 2, data_size)
})

# Step 2: Create Target (Abandonment Logic)
data["abandoned"] = (
    (data["time_spent"] < 5) &
    (data["cart_value"] > 1000) &
    (data["logged_in"] == 0)
).astype(int)

# Step 3: Split Data
X = data.drop("abandoned", axis=1)
y = data["abandoned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 5: Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Step 6: Save Model
joblib.dump(model, "model.pkl")

print("Model trained and saved as model.pkl")
