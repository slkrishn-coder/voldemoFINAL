import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

# Caching the data generation and model training
@st.cache_data
def generate_volunteer_data():
    """Generate volunteer data with consistent random seed."""
    np.random.seed(42)

    volunteer_data = pd.DataFrame({
        'volunteer_id': np.arange(1, 101),
        'past_tasks_completed': np.random.randint(1, 60, 100),  # Extended for new levels
        'activity_completion_rate': np.random.uniform(0.5, 1, 100),
        'sign_up_click_through': np.random.uniform(0.2, 0.8, 100),
        'urgency': np.random.choice(['High', 'Medium', 'Low'], 100, p=[0.3, 0.5, 0.2])
    })

    # Assign experience levels based on tasks completed
    volunteer_data['experience_level'] = volunteer_data['past_tasks_completed'].apply(assign_experience_level)

    # Add volunteer names
    volunteer_data['volunteer_name'] = [f"Volunteer {i}" for i in volunteer_data['volunteer_id']]
    
    return volunteer_data

def assign_experience_level(tasks_completed):
    """Assign experience level based on tasks completed."""
    if tasks_completed <= 10:
        return "Beginner"
    elif 11 <= tasks_completed <= 30:
        return "Intermediate"
    elif 31 <= tasks_completed <= 50:
        return "Mentor"
    else:
        return "Site Leader"

@st.cache_data
def preprocess_and_train_models(volunteer_data):
    """Preprocess data and train models with caching."""
    # Encode categorical variables
    le_experience = LabelEncoder()
    le_urgency = LabelEncoder()

    volunteer_data['experience_level_encoded'] = le_experience.fit_transform(volunteer_data['experience_level'])
    volunteer_data['urgency_encoded'] = le_urgency.fit_transform(volunteer_data['urgency'])

    # Features & Labels
    X = volunteer_data[['past_tasks_completed', 'activity_completion_rate', 'sign_up_click_through', 'urgency_encoded']]
    y = volunteer_data['experience_level_encoded']

    # Train Classification Model (Random Forest)
    rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42)  # Optimized estimators
    rf_classifier.fit(X, y)
    
    volunteer_data['predicted_category'] = le_experience.inverse_transform(rf_classifier.predict(X))

    # Train Fit Score Model (XGBoost)
    fit_model = XGBRegressor(n_estimators=50, learning_rate=0.1)  # Optimized estimators
    fit_model.fit(X, np.random.uniform(0, 1, len(volunteer_data)))

    volunteer_data['fit_score'] = fit_model.predict(X)

    return volunteer_data, rf_classifier, le_experience, le_urgency

# Streamlit UI
def main():
    st.markdown("<h1 style='text-align: center;'>🔹 AI-Powered Volunteer Allocation System</h1>", unsafe_allow_html=True)

    # Generate and preprocess data only once
    volunteer_data, rf_classifier, le_experience, le_urgency = preprocess_and_train_models(generate_volunteer_data())

    # Streamlit UI: Select Volunteer (Dropdown with Name + ID)
    selected_volunteer = st.selectbox(
        "Select a Volunteer",
        options=volunteer_data['volunteer_id'],
        format_func=lambda x: f"{volunteer_data.loc[volunteer_data['volunteer_id'] == x, 'volunteer_name'].values[0]} (ID: {x})"
    )

    # Efficient volunteer info retrieval
    volunteer_info = volunteer_data[volunteer_data['volunteer_id'] == selected_volunteer].iloc[0]

    # Display Predictions
    st.write(f"**Predicted Experience Level:** {volunteer_info['predicted_category']}")
    st.write(f"**Top Recommended Task (Based on Fit Score & Urgency):**")

    # Task Allocation Logic
    fit_score = volunteer_info['fit_score']
    if fit_score > 0.7:
        st.success("🚀 Disaster Relief Mapping (High Urgency)")
    elif fit_score > 0.4:
        st.info("🩺 Community Health Outreach (Medium Urgency)")
    else:
        st.warning("📜 Data Validation & Translation (Low Urgency)")

    # Progress bar with controlled float conversion
    progress_value = float(fit_score)
    progress_value = max(0.0, min(progress_value, 1.0))
    st.progress(progress_value)

if __name__ == "__main__":
    main()
