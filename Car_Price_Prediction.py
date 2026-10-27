# ==========================================================
# CAR PRICE PREDICTION
# Comparing Linear Regression, Random Forest,
# and Gradient Boosting
# ==========================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)


# STEP 1: LOAD DATASET

print("\n========== CAR PRICE PREDICTION ==========")

df = pd.read_csv("quikr_car.csv")

print("Original dataset shape:", df.shape)


# STEP 2: CLEAN DATA

df = df.drop_duplicates()

# Remove extra spaces from text columns
for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].str.strip()

# Convert price to numeric
df["Price"] = pd.to_numeric(
    df["Price"].astype(str)
    .str.replace(",", "", regex=False),
    errors="coerce"
)

# Convert year to numeric
df["year"] = pd.to_numeric(
    df["year"],
    errors="coerce"
)

# Convert kilometers driven to numeric
df["kms_driven"] = pd.to_numeric(
    df["kms_driven"].astype(str)
    .str.replace(",", "", regex=False)
    .str.replace("kms", "", case=False, regex=False)
    .str.replace("km", "", case=False, regex=False)
    .str.strip(),
    errors="coerce"
)

# Remove rows missing essential information
df = df.dropna(
    subset=[
        "name",
        "company",
        "year",
        "Price",
        "kms_driven",
        "fuel_type"
    ]
)

# Remove invalid values
df = df[
    (df["Price"] > 0) &
    (df["kms_driven"] >= 0) &
    (df["year"] >= 1990) &
    (df["year"] <= 2026)
].copy()

print("Cleaned dataset shape:", df.shape)


# STEP 3: SELECT FEATURES AND TARGET

features = [
    "name",
    "company",
    "year",
    "kms_driven",
    "fuel_type"
]

X = df[features]
y = df["Price"]

categorical_features = [
    "name",
    "company",
    "fuel_type"
]

numerical_features = [
    "year",
    "kms_driven"
]

print("\nFeatures selected:", features)
print("Target variable: Price")


# STEP 4: SPLIT DATASET

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# STEP 5: PREPROCESSING

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            Pipeline(steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore")
                )
            ]),
            categorical_features
        ),
        (
            "numerical",
            Pipeline(steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                )
            ]),
            numerical_features
        )
    ]
)


# STEP 6: DEFINE MACHINE LEARNING MODELS

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# STEP 7: TRAIN AND EVALUATE EACH MODEL

results = []
trained_models = {}

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append({
        "Model": model_name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2 Score": round(r2, 4)
    })

    trained_models[model_name] = pipeline

    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))


# STEP 8: COMPARE MODELS

results_df = pd.DataFrame(results)

# Lower MAE means smaller average absolute prediction errors
results_df = results_df.sort_values(
    by="MAE",
    ascending=True
).reset_index(drop=True)

print("\n========== MODEL COMPARISON ==========")

print(results_df.to_string(index=False))

# Save comparison results
results_df.to_csv(
    "model_comparison.csv",
    index=False
)

print("\nComparison saved to model_comparison.csv")


# STEP 9: SELECT AND SAVE THE BEST MODEL

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

joblib.dump(
    best_model,
    "car_price_model.pkl"
)

print("\n========== BEST MODEL ==========")
print("Selected model:", best_model_name)
print("Saved model: car_price_model.pkl")


# STEP 10: PREDICT THE PRICE OF A SAMPLE CAR

sample_car = pd.DataFrame([{
    "name": "Hyundai Santro Xing XO eRLX Euro III",
    "company": "Hyundai",
    "year": 2012,
    "kms_driven": 45000,
    "fuel_type": "Petrol"
}])

predicted_price = best_model.predict(
    sample_car
)[0]

print("\n========== SAMPLE CAR PREDICTION ==========")
print("Car: Hyundai Santro Xing")
print("Year: 2012")
print("Kilometers driven: 45000")
print("Fuel type: Petrol")
print("Estimated price: Rs.", round(max(0, predicted_price), 2))


# STEP 11: COMPLETION MESSAGE

print("\n========== PROJECT COMPLETED SUCCESSFULLY ==========")
print("Models compared: 3")
print("Results file: model_comparison.csv")
print("Saved model: car_price_model.pkl")
