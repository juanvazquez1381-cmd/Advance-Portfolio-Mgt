import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import edhec_risk_kit as erk

inds = ['Food', 'Beer', 'Smoke', 'Games', 'Books', 'Hshld', 'Clths', 'Hlth',
       'Chems', 'Txtls', 'Cnstr', 'Steel', 'FabPr', 'ElcEq', 'Autos', 'Carry',
       'Mines', 'Coal', 'Oil', 'Util', 'Telcm', 'Servs', 'BusEq', 'Paper',
       'Trans', 'Whlsl', 'Rtail', 'Meals', 'Fin', 'Other']


#inds=['Beer', 'Hlth', 'Fin','Rtail','Whlsl']

ind_rets = erk.get_ind_returns(weighting = "ew", n_inds = 49)["1974":]
ind_mcap = erk.get_ind_market_caps(49, weights = True)["1974":]

ewr = erk.backtest_ws(ind_rets, estimation_window = 36, weighting = erk.weight_ew)
cwr = erk.backtest_ws(ind_rets, estimation_window = 36, weighting = erk.weight_cw, cap_weights = ind_mcap)

btr = pd.DataFrame({"EW": ewr, "CW": cwr})
(1 + btr).cumprod().plot(figsize = (12, 6), title = "Industry Portfolios - CW vs EQ")
plt.show()

print(erk.summary_stats(btr.dropna()))
print("-----------------------------------------------------------------------------------")

def sample_cov(r, **kwargs):
    """
    Returns the sample covariance of the supplied returns
    """
    return r.cov()

def weight_gmv(r, cov_estimator = sample_cov, **kwargs):
    """
    Produces the weights of the GMV portfolio give a covariance matrix of returns
    """
    est_cov = cov_estimator(r, **kwargs)
    return erk.gmv(est_cov)

mv_s_r = erk.backtest_ws(ind_rets, estimation_window = 36, weighting = weight_gmv, cov_estimator = sample_cov)
btr = pd.DataFrame({"EW": ewr, "CW": cwr, "GMV-Sample": mv_s_r})
(1 + btr).cumprod().plot(figsize = (12, 6), title = "Industry Portfolios")
plt.show()
print(erk.summary_stats(btr.dropna()))
print("----------------------------------------------------------------------------------")


def cc_cov(r, **kwargs):
    """
    Estimates a covariance matrix by using the Elton/Gruber Constant Correlation model
    """
    rhos = r.corr()
    n = rhos.shape[0]
    # This is a symmetric matrix with diagonals all 1 - so the mean correlation is . . .
    rho_bar = (rhos.values.sum()-n)/(n*(n-1))
    ccor = np.full_like(rhos, rho_bar)
    np.fill_diagonal(ccor, 1.)
    sd = r.std()
    ccov = ccor * np.outer(sd, sd)
    # mh.corr2cov(ccor, sd)
    return pd.DataFrame(ccov,index = r.columns, columns = r.columns)


wts = pd.DataFrame({
    "EW": erk.weight_ew(ind_rets["2016":]),
    "CW": erk.weight_cw(ind_rets["2016":], cap_weights=ind_mcap),
    "GMV-Sample": weight_gmv(ind_rets["2016":], cov_estimator=sample_cov),
    "GMV-ConstCorr": weight_gmv(ind_rets["2016":], cov_estimator=cc_cov),
})
wts.T.plot.bar(stacked=True, figsize=(15,6), legend=False);
plt.show()


mv_cc_r = erk.backtest_ws(ind_rets, estimation_window = 36, weighting = weight_gmv, cov_estimator = cc_cov)
btr = pd.DataFrame({"EW": ewr, "CW": cwr, "GMV-Sample": mv_s_r, "GMV-CC": mv_cc_r})
(1 + btr).cumprod().plot(figsize = (12, 6), title = "Industry Portfolio")
plt.show()
print(erk.summary_stats(btr.dropna()))
print("----------------------------------------------------------------------------------")

def shrinkage_cov(r, delta = 0.5, **kwargs):
    """
    Covariance estimator that shrinks between the Sample Covariance and the Constant Correlation Estimators
    """
    prior = cc_cov(r, **kwargs)
    sample = sample_cov(r, **kwargs)
    return delta*prior + (1 -delta)*sample

wts = pd.DataFrame({
    "EW": erk.weight_ew(ind_rets["2013":]),
    "CW": erk.weight_cw(ind_rets["2013":], cap_weights=ind_mcap),
    "GMV-Sample": weight_gmv(ind_rets["2013":], cov_estimator=sample_cov),
    "GMV-ConstCorr": weight_gmv(ind_rets["2013":], cov_estimator=cc_cov),
    "GMV-Shrink 0.5": weight_gmv(ind_rets["2013":], cov_estimator=shrinkage_cov),
})
wts.T.plot.bar(stacked=True, figsize=(12,6), legend=False);
plt.show()

mv_sh_r = erk.backtest_ws(ind_rets, estimation_window=36, weighting=weight_gmv, cov_estimator=shrinkage_cov, delta=0.5)
btr = pd.DataFrame({"EW": ewr, "CW": cwr, "GMV-Sample": mv_s_r, "GMV-CC": mv_cc_r, 'GMV-Shrink 0.5': mv_sh_r})
(1+btr).cumprod().plot(figsize=(12,6), title="49 Industry Portfolios")
plt.show()
print(erk.summary_stats(btr.dropna()) )
print("----------------------------------------------------------------------------------")






    
    
    
