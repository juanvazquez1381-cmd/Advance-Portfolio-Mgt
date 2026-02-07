import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import edhec_risk_kit_206 as erk


# Set up for Problems 1 to 4
Ind_Rrts = erk.get_ind_returns(weighting = "vw", n_inds=30)["1997":"2018"]
Ind_MCPS = erk.get_ind_market_caps(30, weights = True)['1997':'2018']

EWR = erk.backtest_ws(Ind_Rrts, estimation_window = 36, weighting = erk.weight_ew)
CWR = erk.backtest_ws(Ind_Rrts, estimation_window = 36, weighting = erk.weight_cw,cap_weights = Ind_MCPS)
BTR = pd.DataFrame({"Equal Weights": EWR, "CapWeighted": CWR}) # BackTest Returns
(1 + BTR).cumprod().plot(figsize = (12, 6), title = "Industry Portfolios - CapWieghted vs Equal Weights")
plt.show()

stats = erk.summary_stats(BTR.dropna())

print(stats)

Q1 = (stats.loc["CapWeighted","Annualized Return",]*100).round(2)
print(f"CapWeighted Annualized Return: {Q1}%")
print("------------------------------------------")

Q2 = (stats.loc["CapWeighted","Annualized Vol",]*100).round(2)
print(f"CapWeighted Annualized Volume: {Q2}%")
print("-----------------------------------------")

Q3 = (stats.loc["Equal Weights","Annualized Return"]*100).round(2)
print(f"Equal Weights Annualized Return: {Q3}%")
print("-----------------------------------------")

Q4 = (stats.loc["Equal Weights","Annualized Vol"]*100).round(2)
print(f"Equal Weights Annualized Volatility: {Q4}%")
print("--------------------------------------------")

# Set up for Problems 5 and 6
Tethered_EW = erk.backtest_ws(Ind_Rrts, cap_weights = Ind_MCPS, max_cw_mult = 2, microcap_threhold = 0.01,estimation_window = 36)
                              


