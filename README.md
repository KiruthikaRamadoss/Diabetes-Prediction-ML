# Diabetes Prediction Using Machine Learning

An end-to-end machine learning project that predicts the likelihood of diabetes based on health indicators, featuring classification models, data preprocessing techniques, and performance visualizations.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Data Preprocessing & EDA](#data-preprocessing--eda)
- [Modeling](#modeling)
- [Results](#results)
- [Conclusion](#conclusion)
- [Supplementary Documents](#supplementary-documents)
- [How to Run](#how-to-run)
- [Contact](#contact)

---

## Overview

This project explores the use of machine learning models to predict diabetes using clinical health indicator data. The goal is to build models that can accurately classify patients as diabetic or non-diabetic based on features such as BMI, glucose level, age, and physical activity. 

The project includes:
- Feature engineering and preprocessing
- Handling class imbalance with SMOTE
- Training multiple classification models
- Evaluating models using accuracy, precision, recall, and F1 score
- Visualizing performance metrics

---

## Dataset

- **Source:** [Kaggle - Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)
- **Size:** ~100,000 records
- **Features Include:**
  - Age, BMI, Smoking History, HbA1c Level, Blood Glucose Level, Gender, Hypertension, Heart Disease, Physical Activity

📁 [Download Dataset (CSV)](diabetes_prediction_dataset.csv)

---

## Data Preprocessing & EDA

- Handled missing values and categorical variables
- Converted text to numeric using label encoding and one-hot encoding
- Applied SMOTE to handle class imbalance
- Performed exploratory data analysis:
  - Target variable distribution
  - Feature correlation heatmap
  - Histograms and boxplots for key indicators

📓 [View Jupyter Notebook](Diabetes_Prediction_Project.ipynb)

---

## Modeling

- **Train/Test Split:** 80/20
- **Vectorization:** Label encoding + One-hot for categorical features
- **Models Evaluated:**
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Neural Network (MLPClassifier)

- **Metrics Evaluated:** Accuracy, Precision, Recall, F1 Score, Confusion Matrix

---

## Results

- **Best Model:** Logistic Regression (Accuracy: 96%)
- **Insights:**
  - High glucose and HbA1c levels strongly correlate with diabetes
  - Lifestyle factors like physical inactivity and smoking also contribute significantly
  - SMOTE improved recall without overfitting

---

## Conclusion

- Logistic Regression and SVM performed best overall
- Data preprocessing and class balancing played a key role in boosting model performance
- The project demonstrates the utility of machine learning in early diabetes detection and preventive healthcare analytics

---

## Supplementary Documents

📝 [Project Proposal (PDF)](https://drive.google.com/file/d/1EEOXpQJsOrUAwJ8vKHx8h7iiIoSLjdWw/view?usp=drive_link)

📄 [Final Report (PDF)](https://drive.google.com/file/d/1TmLTZbrTbtQQHfebDo_tj_cfJPNraBu4/view?usp=drive_link)  


*Note: Documents are shared as view-only to preserve originality and prevent unauthorized edits.*


---

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/YourUsername/Diabetes-Prediction-ML.git
   cd Diabetes-Prediction-ML
   
2. Create and activate a virtual environment (Optional):
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`

3. Install dependencies:
   pip install -r requirements.txt

4. Launch the notebook:
   jupyter notebook Diabetes_Prediction_Project.ipynb

---

## Contact

For inquiries, collaboration, or feedback:

- [LinkedIn](https://www.linkedin.com/in/kiruthikaramadoss/)
- [GitHub](https://github.com/KiruthikaRamadoss)
- [Email](mailto:kiruthikaramadoss12@gmail.com)

