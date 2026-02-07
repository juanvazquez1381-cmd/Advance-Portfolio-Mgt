import pandas as pd
import numpy as np
import edhec_risk_kit as erk

ind49_rets = erk.get_ind_returns(weighting="vw", n_inds=49)["2013":"2018"]
ind49_mcap = erk.get_ind_market_caps(49, weights=True)["2013":'2018']
inds = ['Hlth', 'Fin', 'Whlsl', 'Rtail', 'Food']

# Set up question 1
Mcaps = ind49_mcap[inds]
Rets = ind49_rets[inds]
Rho = Rets.corr()
Vols = Rets.std()*np.sqrt(12)
Sigma_Prior = Vols.dot(Vols.T) * Rho

Inds_W = Mcaps.values[0]/Mcaps.values[0].sum()
Cap_ws = pd.DataFrame(Inds_W, index = inds, columns = ["CapWeight"])

Anws = Cap_ws.sort_values('CapWeight')
print(Anws)
print('------------------------------------')

Q1 = 'Rtail'
print(f'{Q1} has the higest cap weight')
print('------------------------------------')

# Set up for questions 2 and 3

Pi = erk.implied_returns(delta = 2.5, sigma = Sigma_Prior, w = Inds_W)
Pi = Pi *100
 
Imp_Rrts = Pi.sort_values(ascending = True)
print(Imp_Rrts)
print('------------------------------------')

Q2 = 'Rtail'
print(f'{Q2} has the highest returns')
print('------------------------------------')

Q3 = 'Hlth'
print(f'{Q3} has the lowest returns')
print('------------------------------------')

# Set up for question 4 and 5
q = pd.Series([0.03])

p = pd.DataFrame([0]*len(inds), index=inds).T.astype(float)
w_Rtail = Cap_ws.loc['Rtail']/(Cap_ws.loc['Rtail'] + Cap_ws.loc['Whlsl'])
w_Whlsl = Cap_ws.loc['Whlsl']/(Cap_ws.loc['Rtail'] + Cap_ws.loc['Whlsl'])
p.iloc[0]['Hlth'] = 1
p.iloc[0]['Rtail'] = -w_Rtail
p.iloc[0]['Whlsl'] = -w_Whlsl
print(round(p,2))
print('---------------------------------------')

Q4 = {'Whlst': '-0.15'}
Q5 = {'Rtail': '-0.85' }

print(Q4)
print('---------------------------------------')

print(Q5)
print('---------------------------------------')

# Set up for questions 6 and 7
delta = 2.5
tau = 0.05
bl_mu , bl_sigma = erk.bl(Cap_ws, sigma_prior = Sigma_Prior, p = p, q = q,omega = None, delta = delta, tau = tau)
Q7 = (bl_mu * 100).round(2).sort_values(ascending = True)
Q7 = pd.DataFrame(Q7, index = inds, columns = ["Implied Returns"])
print(Q7)
print('---------------------------------------')

# Set up for question 8
Q8 = erk.w_msr(bl_sigma, bl_mu).sort_values()
Q8 = pd.DataFrame(Q8, index = inds, columns = ["Weights"])
print(Q8)
print('---------------------------------------')

# Set up for questions 10 and 11

q = pd.Series([0.05]) # change to 5 percent

p = pd.DataFrame([0]*len(inds), index=inds).T.astype(float)
w_Rtail = Cap_ws.loc['Rtail']/(Cap_ws.loc['Rtail'] + Cap_ws.loc['Whlsl'])
w_Whlsl = Cap_ws.loc['Whlsl']/(Cap_ws.loc['Rtail'] + Cap_ws.loc['Whlsl'])
p.iloc[0]['Hlth'] = 1
p.iloc[0]['Rtail'] = -w_Rtail
p.iloc[0]['Whlsl'] = -w_Whlsl
print(round(p,2))
print('---------------------------------------')
                  
delta = 2.5
tau = 0.05
bl_mu , bl_sigma = erk.bl(Cap_ws, sigma_prior = Sigma_Prior, p = p, q = q,omega = None, delta = delta, tau = tau)
Q10 = pd.DataFrame(bl_mu * 100, index = inds, columns = ["Implied Returns"])
Q10 = Q10.sort_values(by = 'Implied Returns', ascending = True)
print(Q10)
print('---------------------------------------')
                                                     

Q11 = erk.w_msr(bl_sigma, bl_mu).sort_values()
Q11 = pd.DataFrame(Q11, index = inds, columns = ["Weights"])
Q11 = Q11.sort_values(by = "Weights", ascending = True)
print(Q11)
print('---------------------------------------')

