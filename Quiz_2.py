import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import edhec_risk_kit as erk

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

Q2 = (stats.loc["CapWeighted","Annualized Volatility",]*100).round(2)
print(f"CapWeighted Annualized Volume: {Q2}%")
print("-----------------------------------------")

Q3 = (stats.loc["Equal Weights","Annualized Return"]*100).round(2)
print(f"Equal Weights Annualized Return: {Q3}%")
print("-----------------------------------------")

Q4 = (stats.loc["Equal Weights","Annualized Volatility"]*100).round(2)
print(f"Equal Weights Annualized Volatility: {Q4}%")
print("--------------------------------------------")

# Set up for Problems 5 and 6
Tethered_EW = erk.backtest_ws(Ind_Rrts,cap_weights = Ind_MCPS,  max_cw_mult = 2, microcap_threshold = 0.01,estimation_window = 36)
T_BTR = pd.DataFrame({"Equal Weights": EWR, "CapWeighted": CWR, "Tethered": Tethered_EW}) # BackTest Returns
(1 + T_BTR).cumprod().plot(figsize = (12, 6), title = "Industry Portfolios  Different Weigths")
plt.show()
T_stats = erk.summary_stats(T_BTR.dropna())
print(T_stats)

Q5 = (T_stats.loc["Tethered", "Annualized Return"]*100).round(2)
print(f"Tehered Portfolio Annualized Return: {Q5}%")
print("-----------------------------------------")

Q6 = (T_stats.loc["Tethered", "Annualized Volatility"]*100).round(2)
print(f"Tehered Portfolio Annualized Volatility: {Q6}%")
print("-----------------------------------------")

# Set up for Problmes 7
Q7 = (erk.tracking_error(EWR, CWR)*100).round(2)
print(f"Tracking Error fro EWR and CWR: {Q7}%")
print("-----------------------------------------")

# Set up for Problmes 8
Q8 = (erk.tracking_error(Tethered_EW, CWR)*100).round(2)
print(f"Tracking Error: for Tethered Portfolio: {Q8}%")
print("-----------------------------------------")

# Set up for Problems 9 to 10
GMV_rts = erk.backtest_ws(Ind_Rrts, estimation_window = 36, weighting = erk.weight_gmv,cov_estimator = erk.sample_cov)
GMV_BTR = pd.DataFrame({"Equal Weights": EWR, "CapWeighted": CWR, "Tethered": Tethered_EW, "GMV":GMV_rts}) # BackTest Returns
(1 + GMV_BTR).cumprod().plot(figsize = (12, 6), title = "Industry Portfolios  Different Weigths")
plt.show()
GMV_stats = erk.summary_stats(GMV_BTR.dropna())
print(GMV_stats)

Q9 = (GMV_stats.loc["GMV", "Annualized Return"]*100).round(2)
print(f'GMV Portfolio Annualized Return: {Q9}%')
print("-----------------------------------------")

Q10 = (GMV_stats.loc["GMV", "Annualized Volatility"]*100).round(2)
print(f'GMV Portfolio Annualized Volatility: {Q10}%')
print("-----------------------------------------")

# Set up for Problmes 11 to 12
GMV_dt = erk.backtest_ws(Ind_Rrts, estimation_window = 36,weighting = erk.weight_gmv,cov_estimator = erk.shrinkage_cov, delta = 0.25)
GMV_dt_BTR = pd.DataFrame({"Equal Weights": EWR, "CapWeighted": CWR, "Tethered": Tethered_EW, "GMV":GMV_rts, "GMV 0.25 Shrinkage": GMV_dt})
Shrink_stats = erk.summary_stats(GMV_dt_BTR.dropna())
(1 + GMV_dt_BTR).cumprod().plot(figsize = (12, 6), title = "Industry Portfolios  Different Weigths")
plt.show()

print(Shrink_stats)

Q11 = (Shrink_stats.loc["GMV 0.25 Shrinkage", "Annualized Return"]*100).round(2)
print(f"GMV 0.25 Shrinkage Annualize Return: {Q11}%")
print("-----------------------------------------")

Q12 = (Shrink_stats.loc["GMV 0.25 Shrinkage", "Annualized Volatility"]*100).round(2)
print(f"GMV 0.25 Shrinkage Annualize Volatility: {Q12}%")
print("-----------------------------------------")
