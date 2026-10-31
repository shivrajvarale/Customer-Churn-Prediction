# Customer Churn Prediction
 
Predicting which telecom customers are likely to leave, so a retention team can reach out *before* they go.
 
Built on the IBM Telco Customer Churn dataset (7,043 customers, 21 columns). The notebook covers the full workflow: cleaning, EDA, feature encoding, model comparison, tuning, threshold selection, and saving a deployable model.
 
## Why this problem matters
 
About **26.5%** of customers in this dataset churned. Missing a churner costs more than contacting someone who would have stayed, so the project optimises for **recall** (catching churners) while keeping precision reasonable.
 
## Repository contents
 
| File | Description |
|---|---|
| `Customer_Churn_Prediction.ipynb` | End-to-end analysis and modelling notebook |
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Telco customer dataset |
| `app.py` | Streamlit app for scoring a single customer |
| `model/` | Saved model, scaler and decision threshold used by the app |
| `requirements.txt` | Python dependencies |
 
## Key findings from EDA
 
- **Tenure is the strongest signal.** Churn is 47.4% in the first 12 months and falls to 6.6% for customers with 61-72 months.
- **Contract type matters a lot.** Month-to-month customers churn at 42.7%, versus 11.3% on one-year and 2.8% on two-year contracts.
- **Fiber optic customers churn more** (41.9%) than DSL (19.0%).
- **Risks stack.** Fiber optic + month-to-month is the worst segment, at about 55% churn.
- **Support and security help.** Customers without tech support churn at 41.6% versus 15.2% with it. Online security shows a similar gap (41.8% vs 14.6%).
- **Electronic check users churn at 45.3%**, and they are over-represented among month-to-month customers, so these two risk factors overlap.
- **Gender has almost no effect** (26.9% vs 26.2%).
## Method
 
1. **Cleaning:** converted `TotalCharges` to numeric (11 blank values for tenure-0 customers set to 0), dropped `customerID`, merged "No internet service" / "No phone service" into "No".
2. **Feature handling:** Yes/No columns mapped to 1/0, multi-category columns one-hot encoded, `TotalCharges` dropped because it is nearly tenure x MonthlyCharges (multicollinearity), numeric features standardised.
3. **Split:** 80/20 train/test with `random_state=42`.
4. **Imbalance:** handled with `class_weight="balanced"` (and `scale_pos_weight` for XGBoost).
5. **Tuning:** 5-fold cross-validation with `GridSearchCV`, scored on ROC-AUC.
## Results
 
Test-set performance at the default 0.5 threshold:
 
| Model | Recall | Precision | ROC-AUC |
|---|---|---|---|
| **Logistic Regression** | **0.820** | 0.523 | **0.860** |
| XGBoost | 0.796 | 0.534 | 0.855 |
| Random Forest | 0.458 | 0.643 | 0.839 |
 
Logistic Regression was chosen as the final model: it performs best here and is the most interpretable.
 
### Threshold selection
 
Lowering the decision threshold trades precision for recall. The final model uses a threshold of **0.40**, which on the test set catches 327 of 373 churners (recall 0.877, precision 0.466).
 
| Threshold | Recall | Precision |
|---|---|---|
| 0.30 | 0.941 | 0.428 |
| 0.40 | 0.877 | 0.466 |
| 0.50 | 0.820 | 0.523 |
| 0.60 | 0.737 | 0.579 |
 
### What drives churn (odds ratios)
 
Odds ratio above 1 raises churn odds; below 1 lowers them. Numeric features are standardised, so their ratios are per one standard deviation.
 
- Fiber optic internet: **2.25**
- Streaming movies / TV: 1.38 / 1.31
- Electronic check payment: 1.18
- Two-year contract: **0.53**
- Higher tenure: **0.47**
## Saved artefacts
 
The `model/` folder holds three files created by the notebook: the trained model (`churn_model.pkl`), the fitted scaler (`scaler.pkl`) and the chosen threshold (`threshold.pkl`). They were saved with scikit-learn 1.6.1, which is pinned in `requirements.txt`. New data must go through the same scaler, use the same column order, and use the same threshold.
 
## How to run
 
```bash
git clone https://github.com/shivrajvarale/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
pip install -r requirements.txt
jupyter notebook Customer_Churn_Prediction.ipynb
```
 
The notebook was written in Google Colab and loads the CSV from `/content/`. If you run it elsewhere, change the path in the data-loading cell to `WA_Fn-UseC_-Telco-Customer-Churn.csv`.
 
## Run the Streamlit app
 
```bash
streamlit run app.py
```
 
Enter a customer's details (tenure, contract, services, billing) and the app returns the churn probability, a high/low risk flag at the 0.40 threshold, and the features pushing the prediction up or down.
 
## Limitations
 
- The decision threshold was chosen by looking at test-set results, so the reported 0.40 numbers are slightly optimistic. A separate validation split would be cleaner.
- EDA shows association, not causation.
- The dataset is a single snapshot from one telecom company; results may not transfer to other businesses.
## Tech stack
 
Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, XGBoost, joblib
 
## Author
 
**Shivraj Varale** - [GitHub](https://github.com/shivrajvarale)
 
