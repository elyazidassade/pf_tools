import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from pandas_datareader import data as pdr
import datetime


######################### Importing data #########################

def getData(ticker, start_date):
    today = datetime.date.today()
    data = pdr.get_data_yahoo(ticker, start=start_date, end=today)
    return np.round(data,2)


def getPortfolio(list_tickers, start_date, wanted_data='Adj Close'):
    """
    gets stock data for a list of tickers with specified data wanted i.e. High, Low, Open, Close, Volume and Adj Close 
    """
    portfolio = pd.DataFrame(columns=list_tickers)
    for t in list_tickers:
        portfolio[t] = getData(t, start_date)[wanted_data]    
    return portfolio

def daily_returns(df):
    return df.pct_change().dropna()

######################### Statistical tests on daily returns distributions #########################
import scipy.stats

def is_normal(r, level=0.05, test_name='sw'):
    """
    returns a boolean for whether a distribution is normal or isn't, the p-value and the statistic used.
    
    parameters :
    r : dataframe (or series) of returns
    level : float confidence level. if p-value > level we keep the H0 hypothesis, otherwise we reject it.
    test_name : for which test to apply, can only take 'sw' for Shapiro-Walk test or 'jb' for Jarque Bera test
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(is_normal)
    else:
        if test_name=='sw':
            statistic, p_value = scipy.stats.shapiro(r)
        elif test_name=='jb':
            statistic, p_value = scipy.stats.jarque_bera(r)
        return p_value > level, np.round(p_value,3), np.round(statistic,3)

def skewness(r):
    """
    calculates skewness of a given distribution
    """
    r_minus_mean = r - r.mean()
    r_sigma = r.std(ddof=0) #population standard deviation
    exp = (r_minus_mean**3).mean()
    return exp/r_sigma**3

######################### Volatility measures #########################

def historic_var(r, level, loss_only=True):
    """
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        if(loss_only):
            return max(0,-np.percentile(r, level))
        else:
            return -np.percentile(r, level)    

def conditional_var_historic(r, level):
    """
    returns the conditional value at risk of a set of returns at level%
    explanation : mean of all the returns under the historic value at risk at level%
    """
    if isinstance(r,pd.DataFrame):
        return r.aggregate(conditional_var_historic,level=level)
    else:
        rets_under_historic_var = r <= historic_var(r,level=level)
        return r[rets_under_historic_var].mean()
    
    
from scipy.stats import norm

def cornish_fisher_var(r, level=5):
    s = skewness(r)
    k = r.kurtosis(r)
    u = norm.ppf(level/100) 
    z = (u +(u**2 - 1)*s/6 + (u**3 -3*u)*(k-3)/24 -(2*u**3 - 5*u)*(s**2)/36)
    
    return -(r.mean() + z*r.std(ddof=0))    

######################### Annualizing, portfolio returns and volatility ############################

def annualized_returns(returns, periods_per_year):
    """
    gives annualized returns for given a set of a stock prices
    periods_per_year = 255 if daily prices, 12 if monthly ...    
    """
    N = returns.shape[0]
    return ((1+returns).prod())**(periods_per_year/N) -1

def annualized_vol(returns, periods_per_year):
    """
    gives annualized volatility for given a set of returns
    periods_per_year = 255 if daily returns, 12 if monthly returns, 2 if semestrial returns ...
    """
    return returns.std()*np.sqrt(periods_per_year)

def portfolio_return(weights, returns):
    return np.dot(returns,weights)


def portfolio_volatility(weights, cov):
    """
    formula : portfolio volatility = weights.T * covariance matrix(returns) * weights
    """
    w = np.array(weights)
    return np.dot(np.dot(w.transpose(), cov),w)**0.5


######################### Portfolio optimization ############################
from scipy.optimize import minimize

def minimize_volatility(target_return, er, covmat):
    N = er.shape[0]
    w0 = np.repeat(1/N,N)
    
    bounds = [[0,1]]*N
    
    sum_weights_is_1 = {'type' : 'eq', 'fun': lambda weights: weights.sum()-1}
    ret_equal_target = {'type' : 'eq', 'args':(er,), 'fun': lambda weights, er : target_return - portfolio_return(weights, er)}
    
    weights = minimize(portfolio_volatility, x0=w0, args=(covmat,),
                       constraints=(sum_weights_is_1, ret_equal_target),
                       bounds=bounds, method='SLSQP')
    
    return np.round(weights.x,5)
    
    
    
    
    
























