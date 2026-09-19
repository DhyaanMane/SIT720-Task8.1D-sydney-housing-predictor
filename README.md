# SIT720 Task 8.1D — Sydney Housing Price Prediction and Decision Support System

Predicting residential sale prices across three Sydney suburbs — Chatswood (premium),
Parramatta (mid-market) and Blacktown (affordable) — from 102 manually collected
sold listings.

## Contents
| Path | Description |
|---|---|
| `SIT720_Task8.1D_226517659.ipynb` | Full analysis: EDA, feature engineering, three models, error analysis, ML/LLM/human comparison |
| `data/sydney_housing_dataset.xlsx` | 102 sold listings collected from domain.com.au (Oct 2025 – Sep 2026) |
| `data/part5_holdout.csv` | 12 held-out properties with ML, LLM and human predictions |
| `app.py` | Streamlit decision-support application |
| `model/` | Exported gradient boosting pipeline and metadata |
| `screenshots/` | Application screenshots |

## Results
Gradient boosting was selected after five-fold cross-validation:

| Model | MAE | MAPE | R² (log) |
|---|---|---|---|
| Ridge Regression | $329,147 | 18.5% | 0.887 |
| Random Forest | $299,438 | 18.5% | 0.867 |
| **Gradient Boosting** | **$231,328** | **14.4%** | **0.922** |

## Running the application
```bash
pip install -r requirements.txt
streamlit run app.py
```
Opens at `http://localhost:8501`.

## Reproducing the analysis
Open the notebook and run all cells. It reads `data/sydney_housing_dataset.xlsx`
and writes the model files to `model/`.

---
Dhyaan Mane · Deakin University · SIT720 Machine Learning
