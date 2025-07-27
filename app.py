import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# --- Streamlit Application Configuration ---
# This must be the first Streamlit command in your script to avoid StreamlitSetPageConfigMustBeFirstCommandError
st.set_page_config(
    page_title="Olympic Gold Medal Prediction App", # Title for the browser tab
    page_icon="🏅", # Icon for the browser tab
    layout="wide" # Use wide layout for better visual space
)

# --- Data Loading and Model Training ---

@st.cache_data # Cache the data loading to prevent re-running on every interaction
def load_data():
    """
    Loads the Olympic medalists dataset from the provided CSV file.
    Includes basic error handling for file not found.
    """
    try:
        df = pd.read_csv('all_olympic_medalists.csv')
        # Rename 'athletes' column to 'athlete(s)' for clarity if it exists
        if 'athletes' in df.columns:
            df = df.rename(columns={'athletes': 'athlete(s)'})
        return df
    except FileNotFoundError:
        st.error("Error: 'all_olympic_medalists.csv' not found. "
                 "Please ensure the dataset file is in the same directory as the Streamlit script.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the dataset: {e}")
        return None

@st.cache_resource # Cache the model training (a heavy computation)
def train_model(df):
    """
    Trains a Logistic Regression model to predict Gold medals based on Olympic data.
    This function performs feature engineering and model training.
    """
    if df is None:
        return None, None, None, None

    # Create the target variable: 1 if medal is 'Gold', 0 otherwise
    df['is_gold'] = (df['medal'] == 'Gold').astype(int)

    # Define features (X) and target (y) for the model
    # We will use 'season', 'year', 'sport', 'event_gender', and 'country_code' as predictors.
    features = ['season', 'year', 'sport', 'event_gender', 'country_code']
    target = 'is_gold'

    # Drop rows with any missing values in the selected features or target
    # This ensures a clean dataset for model training
    df_cleaned = df.dropna(subset=features + [target]).copy() # Use .copy() to avoid SettingWithCopyWarning

    X = df_cleaned[features]
    y = df_cleaned[target]

    # Identify categorical and numerical columns for preprocessing
    categorical_features = ['season', 'sport', 'event_gender', 'country_code']
    numerical_features = ['year']

    # Create a preprocessing pipeline using ColumnTransformer:
    # - One-hot encode categorical features (handle_unknown='ignore' prevents errors for unseen categories)
    # - Pass through numerical features without modification
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numerical_features)
        ])

    # Create a full machine learning pipeline: preprocessing + classifier
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(solver='liblinear', random_state=42, max_iter=1000))
    ])

    # Split data into training and testing sets for model evaluation
    # stratify=y ensures that the proportion of Gold medals is roughly the same in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train the model on the training data
    model_pipeline.fit(X_train, y_train)

    # Evaluate model accuracy on the test set for demonstration purposes
    y_pred = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.sidebar.info(f"Model Training Accuracy (Gold Prediction): {accuracy:.2f}")

    return model_pipeline, df_cleaned, features, categorical_features

# Load the dataset and train the model when the app starts
olympic_df = load_data()
model, processed_df, features_for_model, categorical_features_for_model = train_model(olympic_df)

# If data loading or model training failed, stop the Streamlit app execution
if olympic_df is None or model is None:
    st.stop()

# --- Application Title and Introduction ---
st.title("🏅 Olympic Gold Medal Prediction App")
st.markdown("""
Welcome to the **Olympic Gold Medal Prediction App**!

This interactive tool allows you to:
* **Predict Gold Medals:** Use the controls in the **sidebar on the left** to select a **Season, Year, Sport, Event Gender, and Country**.
* **Get Instant Predictions:** Click the "Predict Gold Medal" button to see the model's likelihood of a Gold Medal.
* **Understand Model Outputs:** Review the **Prediction Results** section for details, including probabilities, and explore visualizations like the **Prediction Probability Visualization**.
* **Explore Historical Data:** Scroll down to the **Dataset Overview & Historical Visualizations** section to examine overall medal distributions, top gold-winning countries, and specific trends for your chosen sport and country.

Dive in and explore the patterns of Olympic success!
""")

# --- Sidebar for User Input ---
st.sidebar.header("Input for Prediction")
st.sidebar.markdown("Select the details for the Gold Medal prediction:")

# Get unique values from the processed (cleaned) DataFrame for dropdown options
seasons = sorted(processed_df['season'].unique().tolist())
# Filter years to a reasonable range if necessary, or use full range
years = sorted(processed_df['year'].unique().tolist())
sports = sorted(processed_df['sport'].unique().tolist())
event_genders = sorted(processed_df['event_gender'].unique().tolist())

# Create a mapping from full country name to country code for the model input
# Note: Some country names might map to multiple codes over different historical periods.
# This simple mapping takes the last encountered code for a given country name.
unique_countries_map = processed_df.set_index('country')['country_code'].to_dict()
display_countries = sorted(unique_countries_map.keys())

# Streamlit input widgets in the sidebar
selected_season = st.sidebar.selectbox("Select Season", seasons)
selected_year = st.sidebar.slider("Select Year", min_value=min(years), max_value=max(years), value=2016, step=1)
selected_sport = st.sidebar.selectbox("Select Sport", sports)
selected_event_gender = st.sidebar.selectbox("Select Event Gender", event_genders)
selected_country_name = st.sidebar.selectbox("Select Country", display_countries)

# Retrieve the country code corresponding to the selected country name
selected_country_code = unique_countries_map.get(selected_country_name)

if selected_country_code is None:
    st.sidebar.warning("Could not find country code for the selected country. Please select a different country.")
    st.stop()

# Create a DataFrame from the user's inputs, matching the feature structure used for training
input_data = pd.DataFrame([{
    'season': selected_season,
    'year': selected_year,
    'sport': selected_sport,
    'event_gender': selected_event_gender,
    'country_code': selected_country_code
}])

st.sidebar.markdown("---")
st.sidebar.subheader("Model & Data Insights")
st.sidebar.info("""
This application utilizes a Logistic Regression model to make predictions.
The model is trained on the full `all_olympic_medalists.csv` dataset,
predicting the probability of a Gold Medal based on historical patterns.
""")

# --- Prediction Section ---
st.header("Prediction Results")

# Button to trigger the prediction
if st.button("Predict Gold Medal", help="Click to get the model's prediction for the selected criteria."):
    st.subheader("Your Selected Criteria:")
    # Display the user's input in a clean table format
    st.dataframe(input_data, hide_index=True)

    try:
        # Get the prediction (0 or 1) and the prediction probabilities for both classes
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0] # [probability_of_0, probability_of_1]

        st.subheader("Model Prediction:")
        # Display the prediction with an appropriate visual cue (success/info)
        if prediction == 1:
            st.success(f"**Prediction: GOLD Medal Likely!** (Probability of Gold: {prediction_proba[1]:.2f})")
        else:
            st.info(f"**Prediction: Not a GOLD Medal Expected** (Probability of Gold: {prediction_proba[1]:.2f})")

        st.write(f"Probability of No Gold: {prediction_proba[0]:.2f}, Probability of Gold: {prediction_proba[1]:.2f}")

        # --- Visualization of Model Output (Prediction Probabilities) ---
        st.header("Prediction Probability Visualization")
        st.markdown("This chart shows the model's confidence for 'No Gold' vs. 'Gold' outcomes.")
        
        proba_df = pd.DataFrame({
            'Outcome': ['No Gold (0)', 'Gold (1)'],
            'Probability': prediction_proba
        })

        fig_proba, ax_proba = plt.subplots(figsize=(8, 5))
        sns.barplot(x='Outcome', y='Probability', data=proba_df, palette=['lightcoral', 'gold'], ax=ax_proba)
        ax_proba.set_ylim(0, 1) # Ensure y-axis for probability is from 0 to 1
        ax_proba.set_title("Predicted Probabilities for Gold Medal Outcome")
        ax_proba.set_ylabel("Probability")
        st.pyplot(fig_proba)

    except Exception as e:
        st.error(f"An error occurred during prediction. This might happen if the selected combination of features "
                 f"was completely unseen during model training or if there's an internal model error: {e}")


# --- Exploratory Data Analysis & Visualizations ---
st.header("Dataset Overview & Historical Visualizations")
st.markdown("Dive into the historical Olympic medal data to gain insights.")

# Option to show a sample of the raw dataset
if st.checkbox("Show Raw Data Sample", value=False):
    st.subheader("First 10 Rows of Olympic Medalists Data")
    st.dataframe(olympic_df.head(10))

# Visualization 1: Overall Medal Distribution by Season
st.subheader("Overall Medal Distribution by Season (Summer vs. Winter)")
fig_season, ax_season = plt.subplots(figsize=(8, 5))
# Use countplot to show the number of each medal type per season
sns.countplot(data=processed_df, x='season', hue='medal',
              order=processed_df['season'].value_counts().index, # Order by count
              palette={'Gold': 'gold', 'Silver': 'silver', 'Bronze': 'brown'}, ax=ax_season)
ax_season.set_title("Total Medals Awarded by Season")
ax_season.set_ylabel("Number of Medals")
ax_season.set_xlabel("Olympic Season")
st.pyplot(fig_season)


# Visualization 2: Top N Countries by Gold Medals (Overall History)
st.subheader("Top Countries by Gold Medals (All-Time)")
top_n = st.slider("Select Number of Top Countries to Display", min_value=5, max_value=25, value=10)
# Calculate gold medals per country and select the top N
gold_medals_by_country = processed_df[processed_df['medal'] == 'Gold']['country'].value_counts().head(top_n)

fig_top_countries, ax_top_countries = plt.subplots(figsize=(12, 7))
sns.barplot(x=gold_medals_by_country.index, y=gold_medals_by_country.values, palette='viridis', ax=ax_top_countries)
ax_top_countries.set_title(f"Top {top_n} Countries by Gold Medals (Overall Olympic History)")
ax_top_countries.set_xlabel("Country")
ax_top_countries.set_ylabel("Number of Gold Medals")
plt.xticks(rotation=45, ha='right') # Rotate labels for better readability
plt.tight_layout() # Adjust layout to prevent labels from overlapping
st.pyplot(fig_top_countries)

# Visualization 3: Medal Distribution for the Selected Sport
st.subheader(f"Medal Distribution for the Sport: '{selected_sport}'")
# Filter the dataset for the currently selected sport
sport_data = processed_df[processed_df['sport'] == selected_sport]
if not sport_data.empty:
    fig_sport, ax_sport = plt.subplots(figsize=(10, 6))
    sns.countplot(data=sport_data, x='medal', order=['Gold', 'Silver', 'Bronze'],
                  palette={'Gold': 'gold', 'Silver': 'silver', 'Bronze': 'brown'}, ax=ax_sport)
    ax_sport.set_title(f"Medals Awarded in {selected_sport}")
    ax_sport.set_ylabel("Number of Medals")
    ax_sport.set_xlabel("Medal Type")
    st.pyplot(fig_sport)
else:
    st.info(f"No medal data available in the dataset for the sport: **{selected_sport}**.")

# Visualization 4: Medal Distribution for the Selected Country
st.subheader(f"Medal Distribution for the Country: '{selected_country_name}'")
# Filter the dataset for the currently selected country
country_data = processed_df[processed_df['country'] == selected_country_name]
if not country_data.empty:
    fig_country, ax_country = plt.subplots(figsize=(10, 6))
    sns.countplot(data=country_data, x='medal', order=['Gold', 'Silver', 'Bronze'],
                  palette={'Gold': 'gold', 'Silver': 'silver', 'Bronze': 'brown'}, ax=ax_country)
    ax_country.set_title(f"Medals Won by {selected_country_name}")
    ax_country.set_ylabel("Number of Medals")
    ax_country.set_xlabel("Medal Type")
    st.pyplot(fig_country)
else:
    st.info(f"No medal data available in the dataset for the country: **{selected_country_name}**.")

st.markdown("---")
st.caption("Developed as an educational tool using Streamlit and historical Olympic data.")
