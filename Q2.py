import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import edhec_risk_kit as erk

# Set up data
Ind_Rrts = erk.get_ind_returns(weighting="vw", n_inds=30)["1997":"2018"]
Ind_MCPS = erk.get_ind_market_caps(30, weights=True)['1997':'2018']

# --- FIX: Pass Ind_Rrts as the first argument for both backtests ---
EWR = erk.backtest_ws(Ind_Rrts, estimation_window=36, weighting=erk.weight_ew)
# Pass Ind_Rrts for calculating returns, and Ind_MCPS for calculating weights
CWR = erk.backtest_ws(Ind_Rrts, estimation_window=36, weighting=erk.weight_cw, cap_weights=Ind_MCPS)

BTR = pd.DataFrame({"Equal Weights": EWR, "CapWeighted": CWR})
(1 + BTR).cumprod().plot(figsize=(12, 6), title="Industry Portfolios - CapWeighted vs Equal Weights")
plt.show()

# Get stats (Note the updated key "Annualized Volatility")
stats = erk.summary_stats(BTR.dropna())

# Answers for Q1 to Q4
Q1 = (stats.loc["CapWeighted","Annualized Return"]*100).round(2)
Q2 = (stats.loc["CapWeighted","Annualized Volatility"]*100).round(2) 

Q3 = (stats.loc["Equal Weights","Annualized Return"]*100).round(2)
Q4 = (stats.loc["Equal Weights","Annualized Volatility" ]*100).round(2)

print(f"CapWeighted Annualized Return: {Q1}%")
print(f"CapWeighted Annualized Volatility: {Q2}%")
print(f"Equal Weights Annualized Return: {Q3}%")
print(f"Equal Weights Annualized Volatility: {Q4}%")
