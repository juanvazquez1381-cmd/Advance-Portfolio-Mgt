import pandas as pd
import numpy as np
import statsmodels.api as sm
import re
import edhec_risk_kit as erk # Assumes your environment has the full erk library and its required data

# --- Setup from Original Script ---
Period = slice("1991", "2018")
data = erk.get_ind_returns(weighting="vw", n_inds=49)
Ind_49 = pd.DataFrame(data)
Ind_49 = Ind_49[Period]
Ind_49.columns = Ind_49.columns.str.strip().str.capitalize()
# ----------------------------------

# --------------------------------------------------------------------------------------
# 1. Load and Prepare the NEW Fama-French Data from F-F_Research_Data_Factors.CSV

Fama_French = pd.read_csv(
    "F-F_Research_Data_Factors.CSV",
    skiprows=3,  # Skip header rows
    header=0,
    index_col=0)

# Convert factor columns to numeric, coercing errors
for col in Fama_French.columns:
    Fama_French[col] = pd.to_numeric(Fama_French[col], errors='coerce')

# Drop rows where any factor is NaN (removes descriptive footer rows)
Fama_French = Fama_French.dropna()

# Fama-French data is published in percent, so divide by 100
Fama_French = Fama_French / 100

# CORRECTED CLEANING STEP: Filter the index to ensure only 6-digit YYYYMM strings are kept.
index_series = pd.Series(Fama_French.index.astype(str)).str.strip()
valid_index_mask = index_series.apply(lambda x: bool(re.match(r'^\d{6}$', x)))
Fama_French = Fama_French[valid_index_mask.values]

# Convert the valid index (YYYYMM format) to a PeriodIndex for monthly frequency
Fama_French.index = pd.to_datetime(Fama_French.index.astype(str), format="%Y%m").to_period("M")


# --------------------------------------------------------------------------------------
# 2. Align Data and Calculate Industry Excess Returns

# Only use the Fama-French data that overlaps with the Ind_49 returns (1991-2018)
common_index = Ind_49.index.intersection(Fama_French.index)

Ind_49 = Ind_49.loc[common_index]
Fama_French = Fama_French.loc[common_index]

# Calculate Ind_49_excess using the new RF values
RF_rate = Fama_French['RF']
Ind_49_excess = Ind_49.sub(RF_rate, axis=0)


# --------------------------------------------------------------------------------------
# 3. Fama-French Three-Factor Regression Analysis (Questions 7, 8, 9, 10)

FF_factors = Fama_French[['Mkt-RF', 'SMB', 'HML']].copy()
FF_factors["Constant"] = 1 # Add the intercept term

SMB_betas = {}
LargeCap_betas = {}
Value_betas = {}

for industry in Ind_49_excess.columns:
    Y = Ind_49_excess[industry] 
    X = FF_factors              
    
    lm = sm.OLS(Y, X).fit()
    
    SMB_betas[industry] = lm.params["SMB"]
    LargeCap_betas[industry] = lm.params["Mkt-RF"]
    Value_betas[industry] = lm.params["HML"]

# Question 7 & 10: Highest SMB beta (Small Cap tilt)
Highest_SMB_Industry = max(SMB_betas, key=SMB_betas.get)

# Question 8: Highest Mkt-RF beta (Large Cap tilt)
Highest_LargeCap_Industry = max(LargeCap_betas, key=LargeCap_betas.get)

# Question 9: Highest HML beta (Value tilt)
Highest_Value_Industry = max(Value_betas, key = Value_betas.get)

# --------------------------------------------------------------------------------------
# 4. Print Results

print("--- Results using F-F_Research_Data_Factors.CSV (1991-2018) ---")
print(f"Q7/Q10: The industry with the highest SMB exposure is: {Highest_SMB_Industry}")
print(f"Its SMB beta is: {SMB_betas[Highest_SMB_Industry]:.4f}")
print("-------------------------------------------------------------------------")

print(f"Q8: The industry with the highest Large Cap exposure is: {Highest_LargeCap_Industry}")
print(f"Its Large Cap beta is: {LargeCap_betas[Highest_LargeCap_Industry]:.4f}")
print("-------------------------------------------------------------------------")

print(f"Q9: The industry with the highest Value exposure is: {Highest_Value_Industry}")
print(f"Its Value beta is: {Value_betas[Highest_Value_Industry]:.4f}")
print("-------------------------------------------------------------------------")
