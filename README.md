# car-evalution
# 🚗 Car Evaluation Classifier (UCI Dataset)

A machine learning app that classifies cars into categories — **unacceptable**, **acceptable**, **good**, and **very good** — based on six features, using the [Car Evaluation Dataset](https://archive.ics.uci.edu/dataset/19/car+evaluation) from the UCI Machine Learning Repository.

---

## 🧠 Overview

This project includes:
- Exploratory Data Analysis (EDA)
- Data Preprocessing & Cleaning
- Model Training and Comparison
- Deployment using Streamlit

---

## 📁 Files

| File | Description |
|------|-------------|
| `car_evaluation.ipynb` | EDA, preprocessing, model training and evaluation |
| `car_app.py` | Clean deployment-ready Streamlit app using best model |
| `car.csv` | UCI Car Evaluation Dataset |
| `README.md` | Project documentation |

---

## 📊 Dataset Info

- **Samples**: 1,728
- **Features**: 6 categorical
  - `buying`, `maint`, `doors`, `persons`, `lug_boot`, `safety`
- **Target**: `class` (Multi-class: `unacc`, `acc`, `good`, `vgood`)

---

## 🔍 Steps Followed

1. **Data Exploration** – checked shape, data types, and missing values.
2. **Data Cleaning** – handled categorical encoding using `LabelEncoder`.
3. **Visualization** – correlation heatmap and feature-vs-target plots.
4. **Model Training**
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Decision Tree
   - Random Forest
5. **Model Evaluation** – using Accuracy, Precision, Recall, F1 Score.
6. **Deployment** – best model deployed using Streamlit in `car_app.py`.

---

## 📈 Model Performance

| Model               | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression|    -     |     -     |   -    |     -    |
| KNN                |    -     |     -     |   -    |     -    |
| Decision Tree      |    -     |     -     |   -    |     -    |
| **Random Forest**  | ✅ Best  |  ✅ Best  | ✅ Best | ✅ Best  |


