# 🥇 Olympic Gold Medal Prediction App

This is an interactive **Streamlit web application** that predicts the likelihood of a country winning a **Gold Medal** in a given Olympic sport, event gender, season, and year — all powered by historical Olympic data and a trained machine learning model.

---

## 🚀 Features

### 🎯 Gold Medal Prediction
- Input Olympic event parameters:
  - **Season** (Summer/Winter)
  - **Year**
  - **Sport**
  - **Event Gender**
  - **Country**
- Get real-time predictions:
  - Whether the country is likely to win a **Gold Medal**
  - Visual confidence levels (Gold vs. No Gold)

### 📊 Prediction Visualization
- View a **bar chart** of the prediction probabilities.

### 🔍 Historical Data Exploration
- **Preview raw data** samples.
- **Medal Distribution by Season** (Summer vs. Winter)
- **Top Gold Medal-Winning Countries** of all time.
- **Sport-specific medal trends** per country.

---

## 🛠️ How to Use

1. **Select Inputs**: Use the sidebar to set the values for Season, Year, Sport, Gender, and Country.
2. **Click Predict**: Get instant predictions with visual feedback.
3. **Explore Data**: Scroll down to interact with charts showing historical insights.

---

## 📚 Dataset

The app uses the `all_olympic_medalists.csv` dataset containing detailed historical Olympic medalist data.

Make sure this file is placed in the same directory as your main Python script (`olympic_app.py`).

---

## 🧠 Model Details

- Model Type: **Logistic Regression**
- Prediction Task: Binary classification — `Gold (1)` or `Not Gold (0)`
- Preprocessing: **One-Hot Encoding** of categorical features

---

## 💻 Setup (Local Run)

### 1. Clone the Repository
```bash
git clone <your-github-repo-link>
cd <your-repo-name>
