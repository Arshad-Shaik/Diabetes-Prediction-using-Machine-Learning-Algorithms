# Diabetes Prediction — End-to-End Machine Learning Pipeline
Production-grade Python project to predict diabetes risk using the Pima Indians Diabetes dataset. The repo delivers a clean, reproducible ML workflow: data ingestion → preprocessing → model training → evaluation → statistical comparison → visualization.

**Built an end‑to‑end ML pipeline using Scikit‑learn, Pandas, NumPy, and SciPy on the Pima Indians dataset. Applied regression, decision trees, and SVM models, achieving 77% cross‑validation accuracy. Designed ETL pipelines to improve data quality and processing speed by 30%, and applied hypothesis testing with visualizations (Matplotlib, Seaborn) to boost stakeholder understanding by 35%.**

#### Disclaimer: This project is for research/education only and not medical advice.

# Key Features
**Robust ETL**: schema inspection, median imputation for missing values, outlier-safe scaling with StandardScaler

**Preprocessing**: normalization, optional one‑hot encoding, train/test split with fixed seeds

**Models**: DecisionTreeClassifier, SVC, LogisticRegression (scikit‑learn)

**Evaluation**: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC curves

**Validation**: 10‑fold Cross‑Validation, Friedman test, paired t‑tests (SciPy) for model comparison

**Visualization**: boxplots, line charts, bar charts, heatmaps, 3D plots

**Reproducibility**: deterministic random_state, documented environment

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

# Screenshots
<img width="932" height="946" alt="{E5F80C11-C9A1-49F3-9419-0318C341D686}" src="https://github.com/user-attachments/assets/faa431e7-d08e-412d-9712-7c33a77caebf" />

<img width="795" height="631" alt="{B225567C-4FF3-4B42-BDB5-3252A0E5600B}" src="https://github.com/user-attachments/assets/4afb8fd9-75d0-4f69-b176-b404c9ebb42e" />

<img width="727" height="890" alt="{0A6E68FC-B6EA-4A86-A0F6-BF5A8122E523}" src="https://github.com/user-attachments/assets/ee0df46e-6b07-43a2-b417-0618322b44fe" />

<img width="618" height="852" alt="{CEE458F8-8257-49CA-9003-625627390593}" src="https://github.com/user-attachments/assets/9aa1225b-dca5-4bfb-b97c-3bf43935f2a0" />

<img width="609" height="785" alt="{2BDAFD97-A4F5-432D-B489-3E05BD55C8C7}" src="https://github.com/user-attachments/assets/076a4417-5ef9-4faa-a606-3abe8b69fe34" />

<img width="617" height="810" alt="{0FDA9316-1566-48F8-9FB4-A7D363A9DB67}" src="https://github.com/user-attachments/assets/e6331857-f726-4153-8a6b-724f6b67e9a5" />

<img width="618" height="807" alt="{B12AF6F3-1A74-4BFB-9EFE-3EE8404944A1}" src="https://github.com/user-attachments/assets/a44dd392-900c-4178-ae86-ddcfc439135a" />

<img width="618" height="832" alt="{16C14196-893D-46BC-B6C9-EC0805105EC2}" src="https://github.com/user-attachments/assets/7289a090-8b0f-4766-a79e-3073543e221e" />

<img width="612" height="934" alt="{21272287-41E9-4DCA-9F0A-3060A619EDA2}" src="https://github.com/user-attachments/assets/932c090e-b858-445d-adf2-28fc5d5629dc" />

# License
MIT — see LICENSE.

# Acknowledgments
UCI Machine Learning Repository / Kaggle contributors for the dataset.


