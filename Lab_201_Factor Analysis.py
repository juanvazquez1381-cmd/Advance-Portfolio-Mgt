import pandas as pd
import numpy as np
import matplotlib as plt
import edhec_risk_kit as erk

# the returns of Berkshire Hathaway

# The main idea in Factor Analysis is to take a set of observed
# returns and decompose it into a set of explanatory returns.

brka_d = pd.read_csv("brka_d_ret.csv", parse_dates=True, index_col=0)
print(brka_d.head())
print("--------------------------------------")

print(brka_d.tail())
print("--------------------------------------")

brka_m = brka_d.resample('ME').apply(erk.compound).to_period('M') # converts daily returns to monthly returns
print(brka_m.head())
print("--------------------------------------")

brka_m.to_csv("brka_m.csv") # for possible future use


# Fama-French monthly returns data set
fff = erk.get_fff_returns()
Columns = {"Mkt-RF": "Market Risk Premium","SMB": 'Small Minus Big','HML': 'High Minus Low', "RF": "Risk Free Rate"}
print(Columns)
print("-------------------------------------------------------------------------------------------------------------")

print(fff.head())
print("--------------------------------------")

import statsmodels.api as sm
brka_excess = brka_m["1990":"2012-05"] - fff.loc["1990":"2012-05", ['RF']].values
mkt_excess = fff.loc["1990":"2012-05", ['Mkt-RF']]
exp_var = mkt_excess.copy()
exp_var["Constant"] = 1
lm = sm.OLS(brka_excess, exp_var).fit()

print(lm.summary())
print("------------------------------------------------------------------------------------------------")

exp_var["Value"] = fff.loc["1990":"2012-05",['HML']]
exp_var["Size"] = fff.loc["1990":"2012-05",['SMB']]
print(exp_var.head())
print("--------------------------------------------")

lm = sm.OLS(brka_excess, exp_var).fit()
print(lm.summary())
print("--------------------------------------------")

result = erk.regress(brka_excess, mkt_excess)
print(result.params)
print("--------------------------------------------")

print(result.tvalues)
print("--------------------------------------------")

print(result.pvalues)
print("--------------------------------------------")

print(result.rsquared_adj)
print("--------------------------------------------")

print(exp_var.head())
print("--------------------------------------------")

print(erk.regress(brka_excess, exp_var, alpha=False).summary())
      
