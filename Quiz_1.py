import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import edhec_risk_kit as erk

# Set up for questions 1 to 2

data = erk.get_ind_returns(weighting="vw", n_inds=49)

Ind_49 = pd.DataFrame(data)
Ind_49 = Ind_49["1991":"2018"]
Ind_49.columns = Ind_49.columns.str.strip().str.capitalize()
fff = erk.get_fff_returns()

Period = slice("1991","2018")
Ind_49_excess = Ind_49 - fff.loc[Period, ['RF']].values
Mar_excess = fff.loc[Period, ['Mkt-RF']]
Exp_var = Mar_excess.copy()
Exp_var["Constant"] = 1


# Question 1
lm = sm.OLS(Ind_49_excess["Beer"], Exp_var).fit()
Beta = lm.params["Mkt-RF"].round(2)
print(f"Beta for Beer is {Beta}")
print("-------------------------------------------------------------------------")
      
# Question 2
lm = sm.OLS(Ind_49_excess["Steel"], Exp_var).fit()
Beta = lm.params["Mkt-RF"].round(2)
print(f"Beta for Steel is {Beta}")
print("-------------------------------------------------------------------------")

# Set up for questions 3 to 4
Period = slice("2013","2018")
Ind_49_excess = Ind_49.loc[Period] - fff.loc[Period, ['RF']].values
Mar_excess = fff.loc[Period, ['Mkt-RF']]
Exp_var = Mar_excess.copy()
Exp_var["Constant"] = 1


# Question 3
lm = sm.OLS(Ind_49_excess["Beer"], Exp_var).fit()
Beta = lm.params["Mkt-RF"].round(2)
print(f"Beta for Beer for the 5 year period is {Beta}")
print("-------------------------------------------------------------------------")

# Question 4
lm = sm.OLS(Ind_49_excess["Steel"], Exp_var).fit()
Beta = lm.params["Mkt-RF"].round(2)
print(f"Beta for Steel for the 5 year period is {Beta}")
print("-------------------------------------------------------------------------")

# Set up for Questions 5 to 6
Period = slice("1991","1993")
Two_Yr_excess = Ind_49.loc[Period] - fff.loc[Period, ['RF']].values
Mar_excess = fff.loc[Period, ['Mkt-RF']]

Inds = {}

for industry in Two_Yr_excess:
    expf_var = Mar_excess.copy()
    expf_var["Constant"] = 1
    lm = sm.OLS(Two_Yr_excess[industry], expf_var).fit()
    Inds[industry] = lm.params['Mkt-RF']

High_Beta =  max(Inds, key = Inds.get)
Low_Beta  =  min(Inds, key = Inds.get)

# Question 5
print(f"The industry with highest beta is {High_Beta}")
print("-------------------------------------------------------------------------")

# Question 6
print(f"The industry with lowest beta is {Low_Beta}")
print("-------------------------------------------------------------------------")

# Set up for questions 7 to 8
Period = slice("1991", "2018")
data = erk.get_ind_returns(weighting="vw", n_inds=49)
Ind_49 = pd.DataFrame(data)
Ind_49 = Ind_49[Period]
Ind_49.columns = Ind_49.columns.str.strip().str.capitalize()

Fama_French = pd.read_csv(
    "F-F_Research_Data_Factors_m.csv",
    header=0,
    index_col=0)

Fama_French = Fama_French / 100

Fama_French.index = pd.to_datetime(Fama_French.index, format="%Y%m").to_period("M")
common_index = Ind_49.index.intersection(Fama_French.index)

Ind_49 = Ind_49.loc[common_index]
Fama_French = Fama_French.loc[common_index]
Fama_French = Fama_French[Period]


RF_rate = Fama_French['RF']
Ind_49_excess = Ind_49.sub(RF_rate, axis=0)

Mrk_excess = Fama_French['Mkt-RF']

# Question 7

FF_factors = Fama_French[['Mkt-RF', 'SMB', 'HML']].copy()
FF_factors["Constant"] = 1 # Add the intercept term


SMB_betas = {} # Use a better name for the results dictionary
for industry in Ind_49_excess.columns:
    # Dependent variable is the industry excess return
    Y = Ind_49_excess[industry] 
    
    # Independent variables are the FF factors
    X = FF_factors
    
    # Run OLS Regression
    lm = sm.OLS(Y, X).fit()
    
    # Store the SMB beta
    SMB_betas[industry] = lm.params["SMB"]

# 3. Find the industry with the Highest SMB beta
Highest_SMB_Industry = max(SMB_betas, key=SMB_betas.get)

print(f"The industry with the highest SMB exposure is: {Highest_SMB_Industry}")
print("-------------------------------------------------------------------------")
print(f"Its SMB beta is: {SMB_betas[Highest_SMB_Industry]:.4f}")
print("-------------------------------------------------------------------------")


# Questoin 8
F_factors = Fama_French[['Mkt-RF', 'SMB', 'HML']].copy()
FF_factors["Constant"] = 1 # Add the intercept term


LargeCap_betas = {} # Use a better name for the results dictionary
for industry in Ind_49_excess.columns:
    # Dependent variable is the industry excess return
    Y = Ind_49_excess[industry] 
    
    # Independent variables are the FF factors
    X = FF_factors
    
    # Run OLS Regression
    lm = sm.OLS(Y, X).fit()
    
    # Store the SMB beta
    LargeCap_betas[industry] = lm.params["HML"]

LargeCap_Industry = max(LargeCap_betas, key= LargeCap_betas.get)

print(f"The industry with the highest Large Cap exposure is: {LargeCap_Industry}")
print("-------------------------------------------------------------------------")
print(f"Its Large Cap beta is: {LargeCap_betas[LargeCap_Industry]:.4f}")
print("-------------------------------------------------------------------------")

# Set up question 9 to 10


FF_factors = Fama_French[['Mkt-RF', 'SMB', 'HML']].copy()
FF_factors["Constant"] = 1 # Add the intercept term

# Question 9
Value_betas = {} # Use a better name for the results dictionary
for industry in Ind_49_excess.columns:
    # Dependent variable is the industry excess return
    Y = Ind_49_excess[industry] 
    
    # Independent variables are the FF factors
    X = FF_factors
    
    # Run OLS Regression
    lm = sm.OLS(Y, X).fit()
    
    # Store the beta
    Value_betas[industry] = lm.params["HML"]

#  Find the industry with the Highest SMB beta
Value_Industry = max(Value_betas, key = Value_betas.get)

print(f"The industry with the highest Value exposure is: {Value_Industry}")
print("-------------------------------------------------------------------------")
print(f"Its Value beta is: {Value_betas[Value_Industry]:.4f}")
print("-------------------------------------------------------------------------")


# Question 10
FF_factors = Fama_French[['Mkt-RF', 'SMB', 'HML']].copy()
FF_factors["Constant"] = 1 # Add the intercept term


SMB_betas = {} # Use a better name for the results dictionary
for industry in Ind_49_excess.columns:
    # Dependent variable is the industry excess return
    Y = Ind_49_excess[industry] 
    
    # Independent variables are the FF factors
    X = FF_factors
    
    # Run OLS Regression
    lm = sm.OLS(Y, X).fit()
    
    # Store the SMB beta
    SMB_betas[industry] = lm.params['SMB']

# 3. Find the industry with the Highest SMB beta
Highest_SMB_Industry = max(SMB_betas, key=SMB_betas.get)

print(f"The industry with the highest SMB exposure is: {Highest_SMB_Industry}")
print("-------------------------------------------------------------------------")
print(f"Its SMB beta is: {SMB_betas[Highest_SMB_Industry]:.4f}")
print("-------------------------------------------------------------------------")


