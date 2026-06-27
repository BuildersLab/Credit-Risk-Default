# Lending Club Credit Risk Prediction

This project builds a binary classification model to predict whether a borrower will repay their loan or default, using the Lending Club dataset sourced from Kaggle (wordsforthewise/lending-club), covering accepted loan applications from 2007 to 2018 Q4. The full pipeline covers data cleaning, leakage prevention, missing value imputation, outlier treatment, feature engineering, feature selection, and baseline modeling.


## Objective

Classify each borrower as a good or bad credit risk before the loan is issued. The prediction must rely exclusively on information that would have been available at application time, which drives many of the cleaning and leakage decisions described below.


## Dataset

The raw dataset contains over 1.3 million loan records. After filtering for resolved outcomes only, the working dataset consists of 1,076,751 good borrowers (Fully Paid) and 290,066 bad borrowers (Charged Off: 268,559, Default: 40, Late 31 to 120 days: 21,467), giving an approximate 79/21 class split.


## Step 1 — Target Variable Construction

The loan_status column contains eight distinct values. Only statuses with a clear, final resolution are kept. Fully Paid is labeled 0 (good borrower). Charged Off, Default, and Late (31 to 120 days) are combined into label 1 (bad borrower).

Five statuses are removed entirely. Current loans have no outcome yet and cannot be labeled. Late (16 to 30 days) borrowers represent 4,349 records that are in an ambiguous transition state — they may still recover or default — so assigning them to either class would introduce noise. In Grace Period loans are similarly unresolved. The two "Does not meet credit policy" categories carry a selection bias introduced by LendingClub's own underwriting rules and would pollute the training signal. Removing these records ensures the model only trains on loans with a definitive, observable outcome.


## Step 2 — Data Leakage Prevention

Before any modeling, 40 columns are dropped because they contain information that only becomes available after a loan decision has already been made. Using these at prediction time would be impossible in a real deployment and would produce misleadingly optimistic model performance during training.

The four categories of dropped columns are the following.

Post-approval funding columns (funded_amnt, funded_amnt_inv) are only known after investors commit capital, which happens after the credit decision.

LendingClub internal scoring columns (int_rate, installment, grade, sub_grade) are assigned by LendingClub's own risk model after application review. They are derived from the same borrower attributes used as features, making them a direct data leak — the model would essentially be learning to copy LendingClub's existing grade rather than building an independent assessment. The emp_title, title, zip_code, url, and desc columns are also dropped here, either because they are free text with over 500,000 unique variations that cannot be encoded meaningfully, or because they are administrative metadata with no predictive value.

Post-origination payment behavior columns (total_pymnt, recoveries, last_pymnt_d, last_pymnt_amnt, out_prncp, and related fields) only accumulate after the loan is active and the borrower begins repaying. A model trained on these would trivially predict default by observing that no payments were ever made.

Hardship and settlement program columns (hardship_flag, hardship_reason, settlement_status, settlement_amount, and 14 related fields) are only populated when a borrower enters distress programs during repayment. These are direct downstream consequences of default, not predictors of it.

After dropping, the number of remaining columns is reduced from the original count, leaving only features that a credit analyst would have access to at the moment of application.


## Step 3 — Data Type Corrections

Several columns were stored in formats that prevented numerical analysis.

The term column was stored as a string such as "36 months". The text " months" is stripped and the column is cast to integer.

The emp_length column contained strings like "< 1 year" and "10+ years". These are mapped to 0 and 10 respectively, and all other values have their digit extracted, giving a clean numeric employment duration.

The issue_d, earliest_cr_line, and sec_app_earliest_cr_line columns were stored as strings in the format "Mar-2016". All three are parsed into proper datetime objects using the format "%b-%Y".

The initial_list_status column used lowercase "w" and "f" while all other categorical columns used uppercase. This is standardized to uppercase to avoid creating duplicate categories during encoding.

The disbursement_method column mixed "DirectPay" (CamelCase) with "Cash" (plain text). Both are converted to uppercase for consistency.


## Step 4 — Impossible and Placeholder Value Correction

Five utilization rate columns (revol_util, il_util, all_util, bc_util, sec_app_revol_util) contained values far above 100 percent. The observed maximums were 892 percent for revol_util, 558 percent for il_util, 204 percent for all_util, 339 percent for bc_util, and 235 percent for sec_app_revol_util. A utilization rate above 100 percent is mathematically impossible and indicates a data recording error. Each column is flagged with a binary indicator before being hard-capped at 100, so the model retains the information that an anomaly was present while the corrupted value itself no longer distorts training.

The dti column contained a minimum of -1.0, which is impossible since debt-to-income cannot be negative, and a maximum of 999, which is a system sentinel value for unknown or error rather than a real observation. Both conditions are flagged together using a single dti_flag column, and the affected values are replaced with NaN so they can be handled in the imputation stage.

Three credit limit columns used 9,999,999 as a placeholder for "no limit" or "unknown". This affected total_rev_hi_lim and tot_hi_cred_lim, both flagged and replaced with NaN. The mo_sin_old_il_acct column used 999 as a placeholder, representing a claimed 83-year-old installment account, which is implausible. This is also flagged and replaced with NaN.


## Step 5 — Rare Category Consolidation

The home_ownership column contained three very rare categories: ANY with 304 rows, OTHER with 144 rows, and NONE with 48 rows. These are too sparse to produce stable model estimates and too vague to carry meaningful credit signal. ALL three are merged into a single OTHER category.

The addr_state column is dominated by a handful of large states. Iowa (IA) had only 7 rows in a dataset of over 1 million, making it essentially invisible to any model. All states with fewer than 100 records are relabeled as OTHER to prevent unstable one-hot encoding during modeling.


## Step 6 — Missing Value Handling

Missing values are handled in five structured groups, each based on the underlying reason the data is absent. Applying a single strategy across all columns would be incorrect because the appropriate treatment depends entirely on why the value is missing.

Ghost rows are dropped first. A small number of rows have no loan amount at all, meaning they contain no real borrower data. Since every genuine loan application must have a loan amount, these rows are removed before any imputation begins.

Group 1 covers joint loan columns, which are missing simply because the loan is individual and there is no second borrower. This is not unknown data — it is structurally absent. The 13 numeric joint columns (annual_inc_joint, dti_joint, revol_bal_joint, and the sec_app FICO and account fields) are filled with zero because zero is the correct representation of a non-existent co-borrower's contribution. The verification_status_joint column is filled with NOT_APPLICABLE for the same reason. A binary is_joint_application flag is created from the application_type column so the model can learn whether a loan is joint without having to infer it from the presence of zeros. The sec_app_earliest_cr_line date column is intentionally left as NaT and later dropped during feature selection.

Group 2 covers five columns that measure how many months ago a negative credit event occurred: mths_since_last_delinq, mths_since_last_record, mths_since_last_major_derog, mths_since_recent_bc_dlq, and mths_since_recent_revol_delinq. When these are missing, it does not mean the data is unknown — it means the event never happened, which is actually a positive credit signal. A borrower with no record of delinquency is meaningfully different from one whose delinquency record is simply unavailable. For this reason, a binary never-happened flag is created for each column before any imputation. Only after the flag is in place is the missing value filled with the median calculated exclusively from borrowers who did experience the event.

Group 3 covers 14 columns related to installment loans and revolving account activity that were simply not collected for older loans in the dataset. These include open_acc_6m, open_act_il, open_il_12m, open_il_24m, open_rv_12m, open_rv_24m, inq_last_12m, inq_fi, total_bal_il, total_cu_tl, max_bal_bc, all_util, il_util, and mths_since_rcnt_il. The missingness here is a function of loan vintage, not borrower behavior. A cross-check confirmed that mths_since_rcnt_il follows the same missing pattern as open_acc_6m, confirming it belongs to this group rather than Group 2. A single credit_bureau_data_available flag is created to capture the vintage signal, and each column is filled with its own median independently.

Group 4 covers 33 columns with moderate random gaps including employment length, months since recent inquiry, delinquency and account activity fields, and credit limit columns. These gaps follow no structural pattern and are treated as Missing Completely at Random. Employment length is filled with its mode rather than median because it represents ordered categories and the most common tenure is the most reasonable substitute for an unknown value. All other Group 4 columns are filled with their median because median is resistant to the outliers present in financial data.

Group 5 covers 7 columns with very small gaps: revol_util, dti, pub_rec_bankruptcies, collections_12_mths_ex_med, tax_liens, chargeoff_within_12_mths, and inq_last_6mths. The impact of imputation strategy is negligible at this scale and all are filled with median.

After all five groups are handled, the only intentionally remaining missing values are in sec_app_earliest_cr_line, which is later dropped entirely.


## Step 7 — Duplicate Check

The dataset is checked for exact duplicate rows. No duplicates are found.


## Step 8 — Suspicious Outlier Flagging

Before the train/test split, nine domain-knowledge flags are created for values that are technically possible but implausible for a personal loan platform. These flags are hard-coded thresholds derived from domain understanding, not from the data distribution, so they do not risk data leakage even when applied before the split.

annual_inc is flagged above 500,000 dollars. The observed maximum is 10,999,200 dollars, nearly 11 million, which is unrealistic for a borrower on a personal loan platform and likely a data entry error or outlier distorting income-based features.

revol_bal is flagged above 500,000 dollars. The observed maximum is 2,904,836 dollars, nearly 3 million in revolving debt, which is far outside the normal range for personal lending.

tot_coll_amt is flagged above 100,000 dollars. The observed maximum is 9,152,545 dollars, which is extraordinarily high for a collection amount and almost certainly a data error.

mths_since_rcnt_il, mths_since_recent_bc, mo_sin_rcnt_rev_tl_op, and mo_sin_rcnt_tl are all flagged above 300 months (25 years). Their observed maximums are 511 months (42 years), 639 months (53 years), 438 months (36 years), and 314 months (26 years) respectively. Values in this range would imply credit account histories that predate the existence of modern credit systems.

earliest_cr_line is flagged where the year falls before 1950. Some credit lines in the dataset are dated as far back as 1934, which would make the borrower over 80 years old and is almost certainly a data entry error.

delinq_amnt is flagged above 10,000 dollars. The observed maximum is 249,925 dollars against a mean of only 14.92 dollars, indicating a severe outlier distribution.


## Step 9 — Feature Engineering

Three categories of new features are created to give the model more predictive signal than the raw columns provide on their own.

FICO scores are consolidated from two columns (fico_range_low and fico_range_high) that always differ by exactly 4 points, making them redundant. A single fico_avg column takes the midpoint. A fico_risk_bucket categorical feature maps the average score into five standard credit risk tiers: poor (below 580), fair (580 to 669), good (670 to 739), very good (740 to 799), and exceptional (800 and above). This bucketing is important because the relationship between FICO score and default probability is non-linear. The difference between a score of 580 and 620 is far more consequential than the difference between 750 and 790, and a linear model cannot capture that without the bucket feature. The original two columns are dropped.

Five ratio features are constructed because raw amounts without context carry less signal than amounts relative to the borrower's capacity. loan_to_income divides the requested loan amount by annual income, capturing how large a commitment the borrower is taking on relative to their earnings. revol_bal_to_income divides existing revolving balance by income, measuring current debt burden. total_debt_to_income divides total current balance across all accounts by income plus one (to avoid division by zero for the rare cases of zero-income records). open_acc_ratio divides open accounts by total accounts plus one, indicating how active the credit profile is. delinq_to_acc_ratio divides the number of accounts ever 120 days past due by total accounts plus one, giving a normalized measure of historical repayment problems rather than a raw count that is not comparable across borrowers with different credit histories.

Credit age is computed as the number of years between the loan's issue date and the borrower's earliest credit line date. A borrower who has been managing credit for 20 years is meaningfully different from one who opened their first account 18 months ago.


## Step 10 — Feature Selection

Three filter-based criteria are applied before modeling to remove columns that cannot contribute useful information.

Constant or near-constant columns with variance below 0.01 are removed. Any column where nearly every row has the same value provides the model with no discriminating power and can cause numerical instability during training. The id column, which is a pure row identifier with no predictive meaning, is also dropped at this stage.

Highly correlated feature pairs with absolute correlation above 0.90 are identified. When two features carry essentially the same information, keeping both wastes model capacity and can destabilize coefficient estimation in linear models. For each pair above the threshold, the second column is dropped.

Features with near-zero correlation with the target variable (below 0.01 in absolute value) are removed. These columns add noise to the model without contributing to its ability to distinguish good from bad borrowers.

The remaining columns after all three filters constitute the final feature set.


## Step 11 — Train and Test Split

The dataset is split 80 percent training and 20 percent test using stratified sampling to preserve the original 79/21 class ratio in both sets. Stratification is important here because a random split on an imbalanced dataset could produce a test set that does not reflect the true class distribution, making evaluation metrics unreliable.

All subsequent outlier treatment, scaling, and any other data-derived transformations are fit exclusively on the training set and then applied identically to the test set. This is the core anti-leakage rule that runs through the entire pipeline. Even thresholds like quantile bounds and IQR fences are computed only on training data and stored before being applied to the test set.


## Step 12 — Outlier Treatment After Split

With the split in place, numeric columns are treated based on their skewness as measured on the training set.

Columns with absolute skewness above 2 receive a log transformation using log1p. These are columns with extreme tails such as annual_inc, which has a maximum of nearly 11 million against a median in the tens of thousands. Capping these columns alone would still leave an unnatural distribution shape. The log transformation compresses the entire scale so that the distance between a 50,000 dollar income and a 100,000 dollar income is treated comparably to the distance between 1 million and 2 million. A shift value is computed from the training minimum so that zero and negative values are handled safely.

Columns with absolute skewness between 0.5 and 2 are winsorized by clipping values at the 1st and 99th percentiles as computed on the training data. The distribution shape for these columns is already reasonable and only the extreme tails need to be controlled.

Columns with absolute skewness below 0.5 are treated with IQR-based clipping using a multiplier of 3. For each of these columns, an outlier flag is created first to mark rows where the original value fell outside the IQR fences before clipping removes that information. IQR is appropriate here precisely because the distribution is symmetric. On a skewed column, the IQR fence lands inside the bulk of legitimate data and flags normal observations as outliers, which is why it is reserved only for this low-skewness group.

All bounds, quantiles, shifts, and fences are computed on X_train only. The same stored values are applied to X_test without recalculation, ensuring no test-set information influences the treatment.

After treatment, skewness is recomputed on the training set and compared against the pre-treatment values to confirm improvement.


## Step 13 — Encoding and Scaling

Categorical features are converted to numeric using one-hot encoding with drop_first=True to avoid the dummy variable trap. Any infinite values introduced during ratio feature construction are replaced with NaN.

Feature scaling is applied using StandardScaler fit on the training set and applied to both sets. Scaling is required for Logistic Regression, which is sensitive to feature magnitude. The Random Forest model receives unscaled features since tree-based models are invariant to monotonic transformations.

