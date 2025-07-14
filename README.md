
# 🏦 Loan Approval Prediction using TabNet & XGBoost with LLM Explanation

## 🚀 Project Overview

This proof-of-concept (PoC) demonstrates a **loan approval prediction system** using two powerful models:

- [✓] **TabNet**: Deep learning-based tabular data learner (interpretable & GPU-optimized)
- [✓] **XGBoost**: Tree-based gradient boosting model (high accuracy baseline)

We go one step further by:
- Using **biased & skewed synthetic data** to simulate real-world scenarios.
- Incorporating **Google’s T5 model** to generate **LLM-based explanations** of model predictions.
- Running entirely on **Google Colab (T4 GPU)** without any API-based LLM subscriptions.

## 🎯 Business Objective

Build a system that:
- Automates loan approval/decline decisions.
- Surfaces **interpretable, explainable outputs**.
- Detects **bias** across sensitive attributes (e.g., gender, political affiliation).
- Simulates feedback loop and stakeholder review.

## 🧠 Key Features

| Feature | Description |
|--------|-------------|
| ✅ **Dual Model Benchmark** | TabNet and XGBoost are tuned, trained, and evaluated. |
| ✅ **Bias Injection** | Data includes bias triggers like `political_affiliation`, `known_defaulter`, etc. |
| ✅ **Explainability** | T5 summarization model generates **natural language reasoning** for each prediction. |
| ✅ **LLM-Based Summary** | No GPT API used — T5 model runs **locally** on Colab’s T4 GPU. |
| ✅ **Minimal Preprocessing** | Input schema directly drives model performance. |
| ✅ **Fast Iteration** | Model training and Optuna tuning kept to 2-3 epochs for quick cycles. |

## 📁 Project Structure

```
loan_approval_poc/
│
├── loan_validation.ipynb        # Main end-to-end notebook
├── loan_inference_cases.csv     # Sample inference dataset for final summary
├── README.md                    # This file
```

## 📊 Input Features

| Column | Description |
|--------|-------------|
| age | Applicant’s age |
| gender | 0 = Female, 1 = Male |
| income | Annual income |
| employment_type | 0 = Salaried, 1 = Self-employed |
| company_reputation | 0 = Bad, 1 = Good |
| credit_score | 0 = Poor, 1 = Average, 2 = Good |
| existing_loans_count | Count of existing loans |
| known_defaulter | 1 = Yes, 0 = No |
| political_affiliation | 1 = Holds political office |
| loan_amount | Requested loan amount |
| loan_purpose | Education, Business, Personal, etc. |

## 🔍 Business Bias Scenarios Modeled

| Scenario | Expected Outcome |
|---------|------------------|
| ✅ Political office holder | Decline |
| ✅ Known defaulter | Decline |
| ✅ Self-employed + Bad company | Decline |
| ✅ Old age (e.g. > 65) | Decline |
| ✅ Multiple existing loans | On-hold |
| ✅ Gender (for bias test only) | Used to validate fairness, not train decisions |

## 📈 Model Metrics (Example)

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| TabNet | 93.2% | 91.7% | 92.9% | 92.3% |
| XGBoost | 91.4% | 89.3% | 90.1% | 89.7% |

_Note: Final scores may vary based on data sampling and tuning._

## 🔄 Feedback Loop Simulation

You can re-upload `loan_inference_cases.csv` with real-world examples (including biased or edge profiles) to:
- Showcase **how predictions change** for different personas.
- Evaluate **model robustness**.
- Test how **LLM summarizes decisions** in business-friendly format.

## 📌 Instructions to Run

1. Open `loan_validation.ipynb` in Google Colab.
2. Ensure T4 GPU is enabled.
3. Run each cell sequentially (1–20).
4. For final testing, upload your custom `loan_inference_cases.csv`.
5. Review model predictions + natural language summaries in final cell.

## 📚 Libraries Used

- `pandas`, `numpy`
- `optuna`
- `pytorch_tabnet`
- `xgboost`
- `scikit-learn`
- `transformers` (T5 summarization)

## ❗ Limitations

- T5 model’s summaries are good but not always deeply contextual.
- Some edge cases may lack business-relevant reasoning.
- Model may overfit small patterns if training data is too biased.

## 💡 Extensions & Enhancements

- Add **streamlit UI** for business demonstration.
- Incorporate **feedback logging** and retraining logic.
- Include **fairness metrics** explicitly (e.g., gender parity).
- Replace T5 with **Gemma** (Google’s open-source LLM) in future.
