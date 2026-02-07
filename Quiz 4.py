import pandas as pd
import numpy as np
import edhec_risk_kit as erk


# Set up for 
ind49_rets = erk.get_ind_returns(weighting="vw", n_inds=49)["2014":"2018"]
ind49_mcap = erk.get_ind_market_caps(49, weights=True)["2014-01":]

Rets = ind49_rets
MCAP = ind49_mcap
Cov = Rets.cov()

results = erk.risk_contribution(erk.weight_cw(Rets, MCAP), Cov).sort_values(ascending = False)
print(results)
print('--------------------------------')
print(round(results.max()*100,2))
print('--------------------------------')

r2 = erk.risk_contribution(erk.weight_ew(Rets),Cov).sort_values(ascending = False)
print(r2)
print('--------------------------------')
print(round(r2.max()*100,2))

r3 = pd.DataFrame(np.transpose(erk.equal_risk_contributions(Cov)),index = Rets.columns,columns = ['Weight'])
ERC_w = r3 * 100
print(ERC_w.sort_values( by ='Weight', ascending = False))
print('--------------------------------')

print(round(ERC_w.max(),2))
print('--------------------------------')

print(round(ERC_w.min(),2))
print('--------------------------------')

print(round(ERC_w.max(),2) - round(ERC_w.min(),2))


# Cap Weighted difference
print(round(results.max()*100,2)- round(results.min()*100,2))
