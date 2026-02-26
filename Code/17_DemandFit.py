# -*- coding: utf-8 -*-
"""
WAS NOT USED IN THE PAPER
"""

#%% Libraries
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#%% Read in Data

path = "C:/Users/pbrock/Desktop/KY19_Extension"

#Load GMM Fits
Holdings = pd.read_stata(path + "/Output" + "/Fitted_Values_GMM_Baseline.dta")
Holdings['rdate'] =  pd.to_datetime(Holdings["rdate"]) #if reading in csv
# -----------------------------------------------------------------------------
#   !!! IMPORTANT !!!
#   Check Date format rdate. It must be end of Quarter Dates. If not, 
#   shift the dates to end of Quarter. Sometimes, reading in .dta files
#   can make End of Quarter Dates to Begin of Quarter Dates.
#   df['rdate'] = df['rdate'] + pd.offsets.MonthEnd(3)
# -----------------------------------------------------------------------------

#Load Baseline Stock Characteristics
StocksQ = pd.read_stata(path + "/Data" + "/StocksQ.dta")
StocksQ["date"] = StocksQ["date"] - pd.offsets.MonthBegin()+pd.offsets.MonthEnd()

Baseline_endog_Variables_Names = ['LNme','LNbe', 'profit', 'Gat', 'divA_be','beta']
Baseline_exog_Variables_Names  = ['IVme','LNbe', 'profit', 'Gat', 'divA_be','beta']

#%% Compute actual Portfolio Weights

#Compute Sum of rweights per manager within quarter
Rweight_sum = Holdings.groupby(['rdate','mgrno'])[['rweight','rweight_hat']].sum()

#Merge to Dataframe
Holdings = Holdings.merge(Rweight_sum, on = ['rdate','mgrno'], suffixes = ('', '_sum'), how = 'left')

#Compute actual Portfolio Holding
Holdings['P_weight'] = Holdings['rweight']/(1+Holdings['rweight_sum'])
Holdings['P_weight_hat'] = Holdings['rweight_hat']/(1+Holdings['rweight_hat_sum'])

#%% Compute Actual and Estimated Demand

Holdings['Demand'] = Holdings['P_weight']*Holdings['aum']
Holdings['Demand_hat'] = Holdings['P_weight_hat']*Holdings['aum']

#%% Compute Inside AUM (i.e. AUM invested in inside assets only)

# -----------------------------------------------------------------------------
#   !!! IMPORTANT !!!
#   AUM and Market Equity is in Millions!
# -----------------------------------------------------------------------------

Inside_AUM = Holdings.groupby(['rdate','mgrno'])['Demand'].sum()
Inside_AUM.rename('aum_inside', inplace=True)

#Merge to Dataframe
Holdings = Holdings.merge(Inside_AUM, on = ['rdate','mgrno'], suffixes = ('', ''), how = 'left')

#Sanity Check (the difference must never be negative)
np.min(Holdings['aum'] - Holdings['aum_inside'])

#%% Compute Demand per permno per Quarter

Q = Holdings.groupby(['rdate','permno'])[['Demand','Demand_hat']].sum()
Q = Q.reset_index()

Q = Q.merge(StocksQ[['date','permno','me']], left_on = ['rdate','permno'], right_on = ['date','permno'], how = 'left')
Q = Q.drop('date',axis=1)

#%% Compute Inside_AUM per quarter
df_aum = Holdings.drop_duplicates(subset = ['rdate','mgrno'])
df_aum = df_aum.groupby('rdate')['aum_inside'].sum()

#%% Compute Excess Demand of fitted weights per permno per Quarter

Q['Excess_Demand_hat'] = Q['Demand_hat'] - Q['me']

#%% Compute Excess Demand of fitted weights per permno

df_exD = Q.groupby(['permno'])['Excess_Demand_hat'].sum()
df_exD = df_exD.reset_index()

#This must be negative
df_exD['Excess_Demand_hat'].sum()

#Compute market equity per permno
df_me = Q.groupby(['permno'])['me'].sum()
df_me = df_me.reset_index()

#Place excess demand relative to the permnos market equity
df_exD['Relative_Excess_Demand_hat'] = df_exD['Excess_Demand_hat']/df_me['me']


#Remove Outliers
df_exD = df_exD[df_exD['Relative_Excess_Demand_hat'] < df_exD['Relative_Excess_Demand_hat'].quantile(0.995)]

#------ Plot Histogram of R^2 for Big Investors
counts, bin_edges = np.histogram(df_exD['Relative_Excess_Demand_hat'], bins=50)
# Calculate relative frequencies by dividing the counts by the total number of data points
relative_frequencies = counts / len(df_exD['Relative_Excess_Demand_hat'])

# Plot the histogram with relative frequencies
plt.bar(bin_edges[:-1], relative_frequencies, width=np.diff(bin_edges), edgecolor='black', align='edge',
        alpha=0.7)
ax = plt.gca()  # Get current axis
# Customize labels and title
ax.set_xlabel('Value', fontsize=13)
ax.set_ylabel('Relative Frequency', fontsize=13)
ax.set_title('Stylized Histogram with Background', fontsize=15, fontweight='bold')
plt.title('Histogram Relative Frequencies Relative Excess Demand (Relative to ME of Stock).')
plt.xlabel('Relative Excess Demand')
plt.ylabel('Relative Frequency')
plt.savefig(path + "/Output" + "/Plots" +"/Histogramm_Relative_Excess_Demand.png", dpi=700)#, bbox_inches='tight')  # Save with 300 DPI
plt.show()
#Cannot do a histogramm of actual weights since the scale is very off
#Since Excess Demand is in sum negative, the big stocks are left and small stocks are to the right of 0.


#Plot Histogram in Levels of Excess Demand
df_exD = df_exD[df_exD['Excess_Demand_hat'] > df_exD['Excess_Demand_hat'].quantile(0.01)]
df_exD = df_exD[df_exD['Excess_Demand_hat'] < df_exD['Excess_Demand_hat'].quantile(0.99)]

plt.hist(df_exD['Excess_Demand_hat'],bins=200, edgecolor='black',
        alpha=0.7) 

#%% Plot TS of Excess_Demand (Mispricing) as share of market equity of assets

"""
Fluctuates wildy between time points, don't plot this. 
"""

Q_new = copy.deepcopy(Q)
Q_new['Excess_Demand_hat'] = np.abs(Q_new['Excess_Demand_hat'] )

df_exD = Q_new.groupby(['rdate'])[['Excess_Demand_hat', 'me']].sum()