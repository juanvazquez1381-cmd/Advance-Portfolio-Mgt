import numpy as np
import pandas as pd

def as_colvec(x): # column vector
    if (x.ndim==2):
        return x
    else:
        return np.expand_dims(x, axis = 1)

x = np.arange(4)
print(as_colvec(x))
print('--------------------------------------')

def implied_returns(delta, sigma, w):
    """
    Obtain the implied expected returns by reverse engineering the weights

    Inputs:
            delta: Risk Aversion Coefficient (scalar)
            sigma: Variance-Covariance Matrix (N x N) as DataFrame
            w: Portfolio weights (N x 1) as Series

    Returns an N x 1 vector of Returns as Series
    """
    ir = delta * sigma.dot(w).squeeze() # to get a series from a 1-column dataframe
    ir. name = 'Implied Returns'
    return ir

# Assumes that Omega is proportional to the variance of the prior
def proportional_prior(sigma, tau, p):
    """
    Returns the He-Litterman simplified Omega
    Inputs:
    sigma: N x N Covariance Matrix as DataFrame
    tau: a scalar
    p: a K x N DataFrame linking Q and Assets
    returns a P x P DataFrame, a Matrix representing Prior Uncertainties
    """
    helit_omega = p.dot(tau * sigma).dot(p.T)
    # Make a diagonal matrix from the diagonal elements of Omega
    return pd.DataFrame(np.diag(np.diag(helit_omega.values)), index = p.index,columns = p.index)

from numpy.linalg import inv

def bl(w_prior, sigma_prior, p, q, omega = None, delta = 2.5, tau = .02):
       """
       Computes the posterior expected returns based on 
       the original black litterman reference model

       W.prior must be an N x 1 vector of weights, a Series
       Sigma.prior is an N x N covariance matrix, a DataFrame
       P must be a K x N matrix linking Q and the Assets, a DataFrame
       Q must be an K x 1 vector of views, a Series
       Omega must be a K x K matrix a DataFrame, or None

       if Omega is None, we assume it is
       proportional to variance of the prior
       delta and tau are scalars
       """
       if omega is None:
           omega = proportional_prior(sigma_prior, tau, p)

       # Force w.prior and Q to be column vectors
       # How many assets do we have?
       N = w_prior.shape[0]
       # And how many views?
       K = q.shape[0]
       # First, reverse-engineer the weights to get pi
       pi = implied_returns(delta, sigma_prior, w_prior)
       # Adjust (scale) Sigma by the uncertainty scaling factor
       sigma_prior_scaled = tau * sigma_prior
       # posterior estimate of the mean, use the "Master Formula"
       # we use the versions that do not require
       # Omega to be inverted (see previous section)
       # this is easier to read if we use '@' for matrixmult instead of .dot()
       # mu_bl = pi + sigma_prior_scaled @ p.T @ inv(p @ sigma_prior_scaled @ p.T + omega) @ (q - p @ pi)
       mu_bl = pi + sigma_prior_scaled.dot(p.T).dot(inv(p.dot(sigma_prior_scaled).dot(p.T) + omega).dot(q - p.dot(pi).values))
       # posterior estimate of uncertainty of mu.bl
       # sigma_bl = sigma_prior + sigma_prior_scaled - sigma_prior_scaled @ p.T @ inv(p @ sigma_prior_scaled @ p.T + omega) @ p @ sigma_prior_scaled
       sigma_bl = sigma_prior + sigma_prior_scaled - sigma_prior_scaled.dot(p.T).dot(inv(p.dot(sigma_prior_scaled).dot(p.T) + omega)).dot(p).dot(sigma_prior_scaled)
       return (mu_bl, sigma_bl)

tickers = ['INTC', 'PFE']
s = pd.DataFrame([[46.0, 1.06], [1.06, 5.33]],index = tickers, columns = tickers)* 10E-4
pi = implied_returns(delta = 2.5, sigma = s, w = pd.Series([.44,.56],index = tickers))
print(pi)
print('--------------------------------------')

# for convenience and readability, define the inverse of a dataframe
def inverse(d):
    """
    Invert the dataframe by inverting the underlying matrix
    """
    return pd.DataFrame(inv(d.values), index = d.columns, columns = d.index)

def w_msr(sigma, mu, scale = True):
    """
    Optimal (Tangent/Max Sharpe Ratio) Portfolio weights
    by using the Markowitz Optimization Procedure
    Mu is the vector of Excess expected Returns
    Sigma must be an N x N matrix as a DataFrame and Mu a column vector as a Series
    This implements page 188 Equation 5.2.28 of
    "The econometrics of financial markets" Campbell, Lo and Mackinlay.
    """
    w = inverse(sigma).dot(mu)
    if scale:
        w = w/sum(w) # fix: this assumes all w is +ve
    return w

mu_exp = pd.Series([.02, .04], index = tickers) # INTC and PFE
print(np.round(w_msr(s, mu_exp)*100, 2))
print('--------------------------------------')

# Absolute view 1: INTC will return 2%
# Absolute view 2: PFE will return 4%

q = pd.Series({"INTC":0.02, "PFE": 0.04})

# The Pick Matrix
# For View 2, it is for PFE
p = pd.DataFrame([
    # For View 1, this for INTC
    {'INTC':1, 'PFE': 0},
    # For View 2, it is for PFE
    {'INTC':0, 'PFE': 1}])

# Find the Black Litterman Expected Returns
bl_mu, bl_sigma = bl(w_prior = pd.Series({'INTC':.44, 'PFE':.56}),sigma_prior = s, p = p, q = q)
# Black Litterman Implied Mu
print(bl_mu)
print('--------------------------------------')

print(w_msr(bl_sigma, bl_mu))
print('--------------------------------------')

# Expected returns inferred from the cap-weights
print(pi)
print('--------------------------------------')

q = pd.Series([
    # Relative View 1: INTC will outperform PFE by 2%
  0.02
    ]
)
# The Pick Matrix
p = pd.DataFrame([
     # For View 1, this is for INTC outperforming PFE
  {'INTC': +1, 'PFE': -1}
])

# Find the Black Litterman Expected Returns
bl_mu, bl_sigma = bl(w_prior = pd.Series({'INTC': .44, 'PFE': .56}), sigma_prior = s, p = p, q = q)
# Black Litterman Implied Mu
print(bl_mu)
print('--------------------------------------')

# Use the Black Litterman expected returns and covariance matrix
print(w_msr(bl_sigma, bl_mu))
print('--------------------------------------')

print(w_msr(s, [.03,.01]))
print('--------------------------------------')

print(w_msr(s, [.02, .0]))
print('--------------------------------------')

# The 7 countries ...
countries  = ['AU', 'CA', 'FR', 'DE', 'JP', 'UK', 'US'] 
# Table 1 of the He-Litterman paper
# Correlation Matrix
rho = pd.DataFrame([
    [1.000,0.488,0.478,0.515,0.439,0.512,0.491],
    [0.488,1.000,0.664,0.655,0.310,0.608,0.779],
    [0.478,0.664,1.000,0.861,0.355,0.783,0.668],
    [0.515,0.655,0.861,1.000,0.354,0.777,0.653],
    [0.439,0.310,0.355,0.354,1.000,0.405,0.306],
    [0.512,0.608,0.783,0.777,0.405,1.000,0.652],
    [0.491,0.779,0.668,0.653,0.306,0.652,1.000]
], index=countries, columns=countries)

# Table 2 of the He-Litterman paper: volatilities
vols = pd.DataFrame([0.160,0.203,0.248,0.271,0.210,0.200,0.187],index=countries, columns=["vol"]) 
# Table 2 of the He-Litterman paper: cap-weights
w_eq = pd.DataFrame([0.016,0.022,0.052,0.055,0.116,0.124,0.615], index=countries, columns=["CapWeight"])
# Compute the Covariance Matrix
sigma_prior = vols.dot(vols.T) * rho
# Compute Pi and compare:
pi = implied_returns(delta=2.5, sigma=sigma_prior, w=w_eq)
print((pi*100).round(1))
print('--------------------------------------')

# Germany will outperform other European Equities (i.e. FR and UK) by 5%
q = pd.Series([.05]) # just one view
# start with a single view, all zeros and overwrite the specific view
p = pd.DataFrame([0.]*len(countries), index=countries).T
# find the relative market caps of FR and UK to split the
# relative outperformance of DE ...
w_fr =  w_eq.loc["FR"]/(w_eq.loc["FR"]+w_eq.loc["UK"])
w_uk =  w_eq.loc["UK"]/(w_eq.loc["FR"]+w_eq.loc["UK"])
p.iloc[0]['DE'] = 1.
p.iloc[0]['FR'] = -w_fr
p.iloc[0]['UK'] = -w_uk
print((p*100).round(1))
print('--------------------------------------')

delta = 2.5
tau = 0.05

bl_mu, bl_sigma = bl(w_eq, sigma_prior, p, q, tau = tau)
print((bl_mu * 100).round(1))
print('--------------------------------------')

def w_star(delta, sigma, mu):
    return(inverse(sigma).dot(mu))/delta

wstar = w_star(delta = 2.5, sigma = bl_sigma, mu = bl_mu)
# display w*
print((wstar*100).round(1))
print('--------------------------------------')

w_eq  = w_msr(delta*sigma_prior, pi, scale=False)
# Display the difference in Posterior and Prior weights
print(np.round(wstar - w_eq/(1+tau), 3)*100)
print('--------------------------------------')

view2 = pd.Series([.03], index=[1])
q = q._append(view2)
pick2 = pd.DataFrame([0.]*len(countries), index=countries, columns=[1]).T
p = p._append(pick2)
p.iloc[1]['CA']=+1
p.iloc[1]['US']=-1
print(np.round(p.T, 3)*100)
print('-------------------------------------')



