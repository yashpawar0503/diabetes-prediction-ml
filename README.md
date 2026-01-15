# diabetes-prediction-ml
Diabetes Prediction using Machine Learning

This project predicts whether a patient has diabetes using tree-based machine learning models trained on medical diagnostic data.

The models used include:

Decision Tree

Random Forest

XGBoost

Tuned XGBoost (Grid Search + Cross-Validation)

Objective

To build a reliable ML model that classifies patients as Diabetic or Non-Diabetic based on health indicators such as glucose, insulin, BMI, and blood pressure.

 Dataset

The dataset contains medical attributes like:
Glucose, BloodPressure, Insulin, BMI, SkinThickness, Age, Pregnancies and a target variable Outcome.

Invalid zero values in medical columns are treated as missing and replaced with the median.

 Data Preprocessing

Replace 0s with NaN in medical features

Fill missing values using median

Split data into 70% train and 30% test

 Models Used

Decision Tree (baseline)

Random Forest (bagging)

XGBoost (boosting)

Models are evaluated using Accuracy, Precision, Recall and F1-Score.

Hyperparameter Tuning

XGBoost is optimized using GridSearchCV with 5-fold Stratified Cross-Validation to maximize F1-score.

Final Model

The tuned XGBoost model achieves the best overall performance and is used for final predictions.

Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib

 How to Run

Open the .ipynb file in Google Colab, upload the dataset, and run all cells to train and evaluate the models.

 Output

The notebook prints:

Accuracy for each model

Classification reports

Best XGBoost hyperparameters

Final tuned model performance
