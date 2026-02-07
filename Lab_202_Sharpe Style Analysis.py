import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import edhec_risk_kit as erk

ind = erk.get_ind_returns()["2000":]

#  construct a manager that invests in 30% Beer, 50% in Smoke and
#  20% in other things that have  an average return of 0%
#  and an annualized vol of 15%

mgr_r = 0.30*ind["Beer"] + 0.50*ind["Smoke"] + 0.2*np.random.normal(scale =.15/(12**.5),size = ind.shape[0])

# assume we knew absolutely nothing about this manager and all we
# observed was the returns.
weights = erk.style_analysis(mgr_r, ind) * 100
weights.sort_values(ascending=False).head(6).plot.bar()


my_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

weights.sort_values(ascending=False).head(6).plot.bar(color=my_colors)


coeffs = erk.regress(mgr_r, ind).params*100
print(coeffs.sort_values().head())
print('-----------------------------------------------------')

coeffs.sort_values(ascending=False).head(6).plot.bar(color = my_colors)

brka_m = pd.read_csv("brka_m.csv", index_col=0, parse_dates=True).to_period('M')

mgr_r_b = brka_m["2000":]["BRKA"]
weights_b = erk.style_analysis(mgr_r_b, ind)
print(weights_b.sort_values(ascending=False).head().round(4)*100)
print('-----------------------------------------------------')

brk2009 = brka_m["2009":]["BRKA"]
ind2009 = ind["2009":]
print(erk.style_analysis(brk2009, ind2009).sort_values(ascending=False).head(6).round(4)*100)
plt.show()
