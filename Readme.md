# 📊 Startup Success Valuation Predictor

## 🧠 Project Overview

The **Startup Success Valuation Predictor** is a machine learning project that predicts the valuation of startups based on funding, investors, industry, country, and founding year. The goal is to demonstrate an end-to-end **regression-based ML pipeline**, including data preprocessing, feature engineering, model training, evaluation, and model persistence.

This project is designed as a **portfolio-ready ML project**, suitable for showcasing skills in **Python, data analysis, and machine learning**.

---

## 🎯 Objectives

* Predict startup valuation using numerical and categorical features
* Apply proper preprocessing techniques for real-world business data
* Improve model accuracy using feature engineering and scaling
* Save trained models for future inference

---

## 🗂️ Dataset

* **Source:** Startup Growth & Investment Dataset (CSV)
* **File:** `startup_growth_investment_data.csv`

### Key Features Used

* `Industry` (Categorical)
* `Country` (Categorical)
* `Year Founded`
* `Funding Rounds`
* `Investment Amount (USD)`
* `Number of Investors`
* `Growth Rate (%)`
* `Company Age`

### Target Variable

* **Valuation** (Regression Output)

---

## ⚙️ Tech Stack

* **Language:** Python 🐍
* **Libraries:**

  * Pandas
  * NumPy
  * Scikit-learn
  * Matplotlib / Seaborn
  * Joblib / Pickle
* **Environment:** Jupyter Notebook

---

## 🧪 Machine Learning Pipeline

### 1️⃣ Data Preprocessing

* Handling missing values
* Encoding categorical variables using **LabelEncoder**
* Feature scaling using **StandardScaler**

### 2️⃣ Feature Engineering

* Company age derived from founding year
* Log transformation on skewed financial features (if required)

### 3️⃣ Model Training

* Regression model trained on processed data
* Train-test split for validation

### 4️⃣ Model Evaluation

* Metrics used:

  * R² Score
  * Adjusted R²
  * Mean Squared Error (MSE)
  * Root Mean Squared Error (RMSE)

### 5️⃣ Model Persistence

Saved artifacts:

* `model.pkl` → Trained regression model
* `le.pkl` → Label encoder
* `ss.pkl` → Standard scaler

---

## 📁 Project Structure

```
├── Startup.ipynb                  # Main notebook
├── startup_growth_investment_data.csv
├── model.pkl                      # Trained model
├── le.pkl                         # Label encoder
├── ss.pkl                         # Standard scaler
├── README.md
```

---

## 🚀 How to Run the Project

1. Clone the repository

```bash
git clone <repo-url>
cd startup-valuation-predictor
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Open Jupyter Notebook

```bash
jupyter notebook Startup.ipynb
```

4. Run all cells to train and evaluate the model

---

## 📈 Results

* The model successfully captures the relationship between funding, investors, and valuation
* Achieved strong R² and Adjusted R² scores, validating model performance

---

## 🔮 Future Enhancements

* Use **One-Hot Encoding** instead of Label Encoding
* Try advanced models like **XGBoost / RandomForest**
* Add **Streamlit** web interface for predictions
* Perform hyperparameter tuning

---

## 👨‍💻 Author

**Kalim Mulani**
AI/ML Enthusiast | Software Developer (Fresher)

---

## ⭐ If you like this project

Give it a ⭐ and feel free to fork and improve it!
