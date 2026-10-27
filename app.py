import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "car_price_model.pkl"
COMPARISON_PATH = BASE_DIR / "model_comparison.csv"
DATASET_PATH = BASE_DIR / "quikr_car.csv"


# ==========================================
# 2. CUSTOM CSS DESIGN
# ==========================================

st.markdown("""
<style>
    .stApp {
        background-color: #f4f7fb;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #17365d;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #526477;
        text-align: center;
        margin-bottom: 30px;
    }

    .section-title {
        color: #17365d;
        font-size: 25px;
        font-weight: 700;
    }

    .result-box {
        background-color: #e4f5e9;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #b7e4c7;
        text-align: center;
    }

    .result-label {
        color: #246b45;
        font-size: 18px;
        font-weight: 600;
    }

    .result-price {
        color: #14532d;
        font-size: 38px;
        font-weight: 800;
    }

    .footer {
        text-align: center;
        color: #64748b;
        padding: 25px;
        font-size: 14px;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        min-height: 45px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD DATASET AND TRAINED MODEL
# ==========================================

@st.cache_data
def load_car_data():
    df = pd.read_csv(DATASET_PATH)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Clean spaces from text values
    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].str.strip()

    # Remove missing car names and company names
    df = df.dropna(subset=["name", "company"])

    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    car_data = load_car_data()

except Exception as error:
    st.error(f"Could not load car dataset: {error}")
    st.stop()


try:
    model = load_model()

except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()


# ==========================================
# 4. WEBSITE HEADER
# ==========================================

st.markdown(
    '<div class="main-title">🚗 Car Price Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Estimate a used car price using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# 5. PROJECT INFORMATION
# ==========================================

info1, info2, info3 = st.columns(3)

with info1:
    st.info("🤖 **ML-Based Prediction**")

with info2:
    st.info("📊 **3 Models Compared**")

with info3:
    st.info("⚡ **Instant Estimation**")

st.write("")


# ==========================================
# 6. CAR DETAILS
# ==========================================

st.markdown(
    '<div class="section-title">📝 Enter Car Details</div>',
    unsafe_allow_html=True
)

st.write(
    "Provide the following details to estimate the car price."
)

st.write("")


# Get available companies from the dataset
companies = sorted(
    car_data["company"].dropna().unique().tolist()
)

if not companies:
    st.error("No car companies were found in the dataset.")
    st.stop()


# Two input columns
col1, col2 = st.columns(2)


# ------------------------------------------
# COLUMN 1: CAR INFORMATION
# ------------------------------------------

with col1:

    st.markdown("### 🚘 Car Information")

    selected_company = st.selectbox(
        "Select Car Company",
        companies,
        key="selected_company"
    )

    # Update car names according to selected company
    car_names = sorted(
        car_data.loc[
            car_data["company"] == selected_company,
            "name"
        ].dropna().unique().tolist()
    )

    if not car_names:
        st.warning("No car names are available for this company.")
        st.stop()

    selected_car = st.selectbox(
        "Select Car Name",
        car_names,
        key="selected_car"
    )

    year = st.number_input(
        "Manufacturing Year",
        min_value=1990,
        max_value=2026,
        value=2015,
        step=1
    )


# ------------------------------------------
# COLUMN 2: USAGE AND FUEL DETAILS
# ------------------------------------------

with col2:

    st.markdown("### ⚙️ Usage & Fuel Details")

    kms_driven = st.number_input(
        "Kilometers Driven",
        min_value=0,
        max_value=2000000,
        value=45000,
        step=1000
    )

    fuel_type = st.selectbox(
        "Fuel Type",
        ["Petrol", "Diesel", "LPG", "CNG", "Electric"]
    )

    st.caption(
        "Enter the car details as accurately as possible."
    )


# ==========================================
# 7. PREDICT CAR PRICE
# ==========================================

st.write("")

submitted = st.button(
    "🔍 Predict Car Price",
    type="primary",
    use_container_width=True
)


# ==========================================
# 8. PREDICTION RESULT
# ==========================================

if submitted:

    input_data = pd.DataFrame([{
        "name": selected_car,
        "company": selected_company,
        "year": int(year),
        "kms_driven": int(kms_driven),
        "fuel_type": fuel_type
    }])

    try:

        predicted_price = float(
            model.predict(input_data)[0]
        )

        if predicted_price < 0:

            st.warning(
                "The model returned a negative estimate. "
                "Please check the car details."
            )

        else:

            st.write("")
            st.divider()

            st.markdown(
                '<div class="section-title">'
                '💰 Prediction Result'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="result-box">
                    <div class="result-label">
                        Estimated Car Price
                    </div>
                    <div class="result-price">
                        ₹{predicted_price:,.0f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("")

            st.success(
                "Price estimation completed!"
            )

            st.caption(
                "This is an approximate model prediction, "
                "not a guaranteed market price."
            )

    except Exception as error:

        st.error(
            f"Prediction failed: {error}"
        )


# ==========================================
# 9. MODEL PERFORMANCE
# ==========================================

st.divider()

st.markdown(
    '<div class="section-title">📊 Model Performance</div>',
    unsafe_allow_html=True
)

st.write(
    "The project compares three regression algorithms "
    "to evaluate their performance."
)


if COMPARISON_PATH.exists():

    try:

        results = pd.read_csv(COMPARISON_PATH)

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )

        if {"Model", "MAE"}.issubset(results.columns):

            chart_data = results.set_index("Model")[["MAE"]]

            st.bar_chart(chart_data)

        st.caption(
            "Lower MAE means smaller average absolute prediction "
            "errors. Actual performance depends on the dataset "
            "and test examples."
        )

    except Exception as error:

        st.warning(
            f"Could not display model results: {error}"
        )

else:

    st.info(
        "Run Car_Price_Prediction.py first to generate "
        "model_comparison.csv."
    )


# ==========================================
# 10. ABOUT PROJECT
# ==========================================

with st.expander("ℹ️ About This Project"):

    st.write("""
    **Project:** Car Price Prediction

    **Domain:** Machine Learning

    **Programming Language:** Python

    **Frontend Framework:** Streamlit

    **Algorithms Compared:**
    - Linear Regression
    - Random Forest Regressor
    - Gradient Boosting Regressor

    **Input Features:**
    - Car name
    - Company
    - Manufacturing year
    - Kilometers driven
    - Fuel type

    The trained model uses these details to estimate
    the price of a used car.
    """)


# ==========================================
# 11. FOOTER
# ==========================================

st.markdown("""
<div class="footer">
    Car Price Prediction Project | Built with Python and Streamlit
</div>
""", unsafe_allow_html=True)