# Diabetes Prediction — End-to-End Machine Learning Pipeline
Production-grade Python project to predict diabetes risk using the Pima Indians Diabetes dataset. The repo delivers a clean, reproducible ML workflow: data ingestion → preprocessing → model training → evaluation → statistical comparison → visualization.

#### Disclaimer: This project is for research/education only and not medical advice.

# Key Features
Robust ETL: schema inspection, median imputation for missing values, outlier-safe scaling with StandardScaler
Preprocessing: normalization, optional one‑hot encoding, train/test split with fixed seeds
Models: DecisionTreeClassifier, SVC, LogisticRegression (scikit‑learn)
Evaluation: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC curves
Validation: 10‑fold Cross‑Validation, Friedman test, paired t‑tests (SciPy) for model comparison
Visualization: boxplots, line charts, bar charts, heatmaps, 3D plots
Reproducibility: deterministic random_state, documented environment

# Results (example run)
Decision Tree — Acc: 0.7468 | Prec: 0.6250 | Recall: 0.7273 | F1: 0.6723
SVM — Acc: 0.7273 | Prec: 0.6327 | Recall: 0.5636 | F1: 0.5962
Logistic Regression — Acc: 0.7532 | Prec: 0.6491 | Recall: 0.6727 | F1: 0.6607
10-fold CV mean accuracy: DT ≈ 0.7018, SVM ≈ 0.7604, LR ≈ 0.7722
Friedman p-value ≈ 0.0017 → significant performance differences

#### Note: Metrics may vary by environment and seed.

# Dataset
Pima Indians Diabetes (Outcome = 0/1) with features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age.
Place diabetes.csv in data/ or update the path in the notebook/script.

# Tech Stack
Python, pandas, NumPy, scikit‑learn, SciPy, Matplotlib, Seaborn, Jupyter/Colab

# Project Structure
notebooks/Diabetes_Prediction.ipynb — end‑to‑end workflow
data/diabetes.csv — dataset (not included; add locally)
requirements.txt — dependencies
reports/ — generated plots (confusion matrix, ROC, CV charts)

# Setup
## Python 3.9+
git clone https://github.com/<your-username>/diabetes-ml.git
cd diabetes-ml
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Quickstart
Open notebooks/Diabetes_Prediction.ipynb and run all cells, or
Convert notebook to script and execute:

# Bash
python src/train.py --data data/diabetes.csv --model logreg --cv 10

# Reproducibility Notes
Target variable (Outcome) is kept binary. If scaled during preprocessing, it is restored before training.
Random seeds are set to ensure consistent splits and CV folds.

# Roadmap
Add hyperparameter tuning (GridSearch/Optuna), calibration, and threshold optimization
Add additional models (XGBoost, RandomForest, LightGBM)
Add SHAP-based explainability and model monitoring
Package as a Streamlit/FastAPI demo for real‑time scoring

# License
MIT — see LICENSE.

# Acknowledgments
UCI Machine Learning Repository / Kaggle contributors for the dataset.


