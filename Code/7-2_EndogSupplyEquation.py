# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 08:59:33 2024

@author: pbrock
"""

#%% Libraries
import pandas as pd
import numpy as np

import statsmodels.api as sm

from linearmodels.panel import PanelOLS
#%% Read in Data

#Greene Theorem THEOREM 13.2 and Newey & McFadden (1994) Theorem 3.4, Wikipedia: https://en.wikipedia.org/wiki/Generalized_method_of_moments

path = "C:/Users/pbrock/Desktop/KY19_Extension"

#Load Baseline Stock Characteristics
StocksQ = pd.read_stata(path + "/Data" + "/StocksQ.dta")
StocksQ["date"] = StocksQ["date"] - pd.offsets.MonthBegin()+pd.offsets.MonthEnd()

StocksQ = StocksQ.dropna(subset=['LNprc','LNshrout'])
StocksQ = StocksQ[StocksQ['date'].dt.year > 2001]
StocksQ = StocksQ[StocksQ['date'].dt.year < 2023]

#%% Simple Pooled Regression

df = StocksQ[['LNshrout', 'LNprc']]
df = df.assign(constant=1)

model = sm.OLS(df['LNshrout'],df.iloc[:,1:])
results = model.fit()
print(results.summary())

#%% Pooled Regression with Time Fixed Effects

#Get DataFrame
df = StocksQ[['LNshrout', 'LNprc', 'date']]
df = df.assign(constant=1)

#List of Dates 
list_DateDummys = ['date_' + item for item in df.date.unique().astype(str) + ' 00:00:00']
list_DateDummys.pop(0) #Drop first date as there is no dummy for this since it's included in the constant

#Generate Time Dummies (Leave out First Time Dummy as elsewise this causes rank issues with the constant)
df = pd.get_dummies(df, columns=['date'], drop_first=True)
df[list_DateDummys] = df[list_DateDummys].astype(int)

model = sm.OLS(df['LNshrout'],df.iloc[:,1:])
results = model.fit()
print(results.summary()) #Almost no difference to cross section.

#%% Fixed Effects Regression (Entire Sample)
df = StocksQ[['permno', 'date', 'LNshrout', 'LNprc']]

df = df.set_index(['permno','date'])

model = PanelOLS(df['LNshrout'], df['LNprc'], entity_effects=True, time_effects=True)
fe_results = model.fit()

print(fe_results)

#%% Time Series Regression

def ols_coeffs(df):
    # Define dependent and independent variables
    y = df['LNshrout']
    X = df[['LNprc', 'constant']]
    X = sm.add_constant(X)  # Add constant term for intercept

    # Fit the model
    model = sm.OLS(y, X).fit()

    # Return coefficients as a Series (for easy aggregation into DataFrame)
    return model.params


df = StocksQ[['permno', 'date', 'LNshrout', 'LNprc']]
df = df.assign(constant=1)

df_results = df.groupby('permno').apply(ols_coeffs).reset_index()
df_results = df_results.rename(columns = {'LNprc': 'elasticity'})

df_results.to_stata(path + "/Output" + "/Variance Decomposition Python" + "/Elasticities_TS.dta")
