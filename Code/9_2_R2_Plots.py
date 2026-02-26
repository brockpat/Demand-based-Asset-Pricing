# -*- coding: utf-8 -*-
"""
Plots the R^2 for the Baseline Variables
"""

#%% Libraries
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

path = "C:/Users/pbrock/Desktop/KY19_Extension"

#%% Read in Data

Baseline_endog_Variables_Names = ['LNme','LNbe', 'profit', 'Gat', 'divA_be','beta']
Baseline_exog_Variables_Names  = ['IVme','LNbe', 'profit', 'Gat', 'divA_be','beta']

#---------------------------- OLS Estimates ----------------------------------

OLS_Estimates = pd.read_csv(path + "/Output" + "/Estimations" + "/OLS_unrestricted_R2.csv")
OLS_Estimates['rdate'] = pd.to_datetime(OLS_Estimates["rdate"]) #if reading in csv
OLS_Estimates = OLS_Estimates.drop_duplicates(subset = ['rdate','bin'])

#---------------------------- GMM Estimates -----------------------------------

GMM_Estimates = pd.read_csv(path + "/Output" + "/Estimations" + "/KY19_baseline_Restricted.csv")
GMM_Estimates['rdate'] = pd.to_datetime(GMM_Estimates["rdate"]) #if reading in csv
GMM_Estimates = GMM_Estimates.drop_duplicates(subset = ['rdate','bin'])
#Extract column names of Estimates
cols = list(GMM_Estimates.columns.drop(['rdate','bin','Error']))

#If Estimator did not converge, set all estimates to 0
GMM_Estimates.loc[np.isnan(GMM_Estimates['Error']) == True, cols] = 0

#If convergence is poor, set all estimates to 0
GMM_Estimates.loc[GMM_Estimates['Error']>0.9, cols] = 0

#If market equity coefficient too low, set all estimates to 0
GMM_Estimates.loc[GMM_Estimates['LNme'] < -20, cols] = 0

#If market equity coefficient greater than 1, set all estimates to 0
GMM_Estimates.loc[GMM_Estimates['LNme'] > 1, cols] = 0

#If any estimates take large positive or negative values, set all zero
GMM_Estimates[cols] = GMM_Estimates[cols].mask(GMM_Estimates[cols].abs() > 200, 0)

#---------------------------- NLLS Estimates ----------------------------------

NLLS_Estimates = pd.read_csv(path + "/Output" + "/Estimations" + "/NLLS_Estimates_CV_Restricted_SMAE.csv")
NLLS_Estimates['rdate'] = pd.to_datetime(NLLS_Estimates["rdate"]) #if reading in csv
NLLS_Estimates = NLLS_Estimates.drop_duplicates(subset = ['rdate','bin'])
#%% Plot R^2 for log(y) for OLS Estimates

df = copy.deepcopy(OLS_Estimates)
#Cut last two Quarters off so that the x-ticks nicely align in the plot
df = df[(df['rdate'] < '2022-09-30') & (df['rdate'] != '2013-03-31')]

df_HH = df[df['bin'] == 0]
df_small = df[df['bin'] < 191]
df_big = df[df['bin'] > 190]

df_small_grouped = df_small.groupby('rdate')['R2'].agg(
    R2_mean=('mean'),
    R2_quantile_25=lambda x: x.quantile(0.25),
    R2_quantile_75=lambda x: x.quantile(0.75)
).reset_index()

df_big_grouped = df_big.groupby('rdate')['R2'].agg(
    R2_mean=('mean'),
    R2_quantile_25=lambda x: x.quantile(0.25),
    R2_quantile_75=lambda x: x.quantile(0.75)
).reset_index()


# Create a figure with two subplots (2 rows, 1 column)
fig, axes = plt.subplots(2, 1, figsize=(10, 12))  # Adjust figsize as needed

# Plot 1: R^2 for log(y) for OLS Estimates
ax1 = axes[0]
ax1.plot(df_HH['rdate'], df_HH['R2'], label='Households', color='green')
ax1.plot(df_big_grouped['rdate'], df_big_grouped['R2_mean'], label='Individual Investors', color='blue')
ax1.fill_between(df_big_grouped['rdate'], df_big_grouped['R2_quantile_25'], df_big_grouped['R2_quantile_75'], color='blue', alpha=0.3)
ax1.plot(df_small_grouped['rdate'], df_small_grouped['R2_mean'], label='Grouped Investors', color='red')
ax1.fill_between(df_small_grouped['rdate'], df_small_grouped['R2_quantile_25'], df_small_grouped['R2_quantile_75'], color='red', alpha=0.3)

# Set formatting for Plot 1
ax1.xaxis.set_major_locator(mdates.MonthLocator(4))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.set_xlim(df['rdate'].min(), df['rdate'].max())
ax1.set_ylabel('$R^2$', fontsize=16)
ax1.legend(fontsize = 14)
ax1.tick_params(axis='x', labelsize=16) 
ax1.tick_params(axis='y', labelsize=16) 
#ax1.set_title('Time Series of $R^2$ for log(y) OLS Estimates')
ax1.tick_params(axis='x', rotation=90)

#------------------ Plot R^2 for y for OLS Estimates in comparison to R^2 for log(y)

df = pd.read_csv(path + "/Output/Fit/" + "OLS_R2LevelFits_vs_R2LogFits.csv")
df['rdate'] =  pd.to_datetime(df["rdate"]) #if reading in csv

#Cut last two Quarters off so that the x-ticks nicely align in the plot
df = df[(df['rdate'] < '2022-09-30') & (df['rdate'] != '2013-03-31')]

#Filter outliers
df = df[df['R_squared_OLS_Level'] > df['R_squared_OLS_Level'].quantile(0.02)]

df_HH = df[df['bin'] == 0]
df_small = df[df['bin'] < 191]
df_big = df[df['bin'] > 190]

df_grouped = df.groupby('rdate').agg(
    R2_mean_OLS_Level=('R_squared_OLS_Level', 'mean'),
    R2_quantile_25_OLS_Level=('R_squared_OLS_Level', lambda x: x.quantile(0.25)),
    R2_quantile_75_OLS_Level=('R_squared_OLS_Level', lambda x: x.quantile(0.75)),
    
    R2_mean_OLS_Logs=('R_squared_OLS_Logs', 'mean'),
    R2_quantile_25_OLS_Logs=('R_squared_OLS_Logs', lambda x: x.quantile(0.25)),
    R2_quantile_75_OLS_Logs=('R_squared_OLS_Logs', lambda x: x.quantile(0.75))
).reset_index()

df_small_grouped = df_small.groupby('rdate').agg(
    R2_mean_OLS_Level=('R_squared_OLS_Level', 'mean'),
    R2_quantile_25_OLS_Level=('R_squared_OLS_Level', lambda x: x.quantile(0.25)),
    R2_quantile_75_OLS_Level=('R_squared_OLS_Level', lambda x: x.quantile(0.75)),
    
    R2_mean_OLS_Logs=('R_squared_OLS_Logs', 'mean'),
    R2_quantile_25_OLS_Logs=('R_squared_OLS_Logs', lambda x: x.quantile(0.25)),
    R2_quantile_75_OLS_Logs=('R_squared_OLS_Logs', lambda x: x.quantile(0.75))
).reset_index()

df_big_grouped = df_big.groupby('rdate').agg(
    R2_mean_OLS_Level=('R_squared_OLS_Level', 'mean'),
    R2_quantile_25_OLS_Level=('R_squared_OLS_Level', lambda x: x.quantile(0.25)),
    R2_quantile_75_OLS_Level=('R_squared_OLS_Level', lambda x: x.quantile(0.75)),
    
    R2_mean_OLS_Logs=('R_squared_OLS_Logs', 'mean'),
    R2_quantile_25_OLS_Logs=('R_squared_OLS_Logs', lambda x: x.quantile(0.25)),
    R2_quantile_75_OLS_Logs=('R_squared_OLS_Logs', lambda x: x.quantile(0.75))
).reset_index()


# Plot 2: R^2 for y for OLS Estimates vs log(y)
ax2 = axes[1]
ax2.plot(df_grouped['rdate'], df_grouped['R2_mean_OLS_Logs'], label='R2 Log Prediction', color='black')
ax2.fill_between(df_grouped['rdate'], df_grouped['R2_quantile_25_OLS_Logs'], df_grouped['R2_quantile_75_OLS_Logs'], color='black', alpha=0.3)
ax2.plot(df_small_grouped['rdate'], df_small_grouped['R2_mean_OLS_Level'], label='R2 Level Prediction', color='saddlebrown', linewidth = 2.5)
ax2.fill_between(df_small_grouped['rdate'], df_small_grouped['R2_quantile_25_OLS_Level'], df_small_grouped['R2_quantile_75_OLS_Level'], color='saddlebrown', alpha=0.4)

# Set formatting for Plot 2
ax2.xaxis.set_major_locator(mdates.MonthLocator(4))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.set_xlim(df['rdate'].min(), df['rdate'].max())
ax2.set_ylabel('$R^2$', fontsize=16)
ax2.legend(fontsize = 14)
#ax2.set_title('Comparison of $R^2$: log(y) vs y OLS Estimates')
ax2.tick_params(axis='x', rotation=90, labelsize=16)
ax2.tick_params(axis='y', labelsize=16)  

# Adjust layout and save the multiplot
plt.tight_layout()
plt.savefig(path + "/Output/Plots/Fit/R2_OLS_TS_1.pdf", dpi=600, bbox_inches='tight')

# Display the multiplot
plt.show() #No need for t-tests between the different R^2 because this is the population!

####### Now compute the R2 for the actual Portfolio Weights to see if the rweight is the difference

#%% Plot R^2 for GMM Estimates

#Read in previously computed Data
df = pd.read_csv(path + "/Output/Fit" + "/GMM_R2_Fit.csv")
df['rdate'] =  pd.to_datetime(df["rdate"]) #if reading in csv

#Cut last two Quarters off so that the x-ticks nicely align in the plot
df = df[df['rdate'] < '2022-09-30']
df = df[df['rdate']!= '2013-03-31'] #Cut out this Quarter because Data is missing

#Filter outliers
df = df[df['R_squared_GMM'] > df['R_squared_GMM'].quantile(0.02)]

#Construct Household, Big & Grouped  Investors
df_HH = df[df['bin'] == 0]

#Compute Median and Quantiles of Grouped and Big Investors
df_small = df[df['bin'] < 191]
df_small_grouped = df_small.groupby('rdate').agg(
    R2_median=('R_squared_GMM', 'median'),
    R2_quantile_25_GMM=('R_squared_GMM', lambda x: x.quantile(0.25)),
    R2_quantile_75_GMM=('R_squared_GMM', lambda x: x.quantile(0.75)),
).reset_index()

df_big = df[df['bin'] > 190]
df_big_grouped = df_big.groupby('rdate').agg(
    R2_median=('R_squared_GMM', 'median'),
    R2_quantile_25_GMM=('R_squared_GMM', lambda x: x.quantile(0.25)),
    R2_quantile_75_GMM=('R_squared_GMM', lambda x: x.quantile(0.75)),
).reset_index()

#Group all R^2 < 0 to 0 because otherwise Histogramm cannot be reasonably visualised
df_small.loc[df_small['R_squared_GMM'] < 0, 'R_squared_GMM'] = -0.1
df_big.loc[df_big['R_squared_GMM'] < 0, 'R_squared_GMM'] = -0.1


#------ Plot Median Time Series R^2 of Grouped  Investors
plt.figure(figsize=(10,6))
plt.plot(df_small_grouped['rdate'], df_small_grouped['R2_median'], label='Grouped Investors', color='red')
plt.fill_between(df_small_grouped['rdate'], df_small_grouped['R2_quantile_25_GMM'], df_small_grouped['R2_quantile_75_GMM'], color='red', alpha=0.3)

# Set xticks for every year
ax = plt.gca()  # Get current axis
ax.xaxis.set_major_locator(mdates.MonthLocator(4))  # Set major ticks to April (Q2) of each year
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format the x-axis to only show the year
plt.xlim(df['rdate'].min(), df['rdate'].max())

# Rotate the x-axis labels for better readability
plt.xticks(rotation=90,fontsize = 14)
plt.yticks(fontsize = 14)
# Add labels and title
plt.ylabel('$R^2$',fontsize = 14)
#plt.title('Time Series Median $R^2$ GMM Estimates Grouped Investors (shade = IQR).')
plt.grid(True)
plt.savefig(path + "/Output" + "/Plots/Fit" +"/R2_Median_GMM_SmallInvestors.pdf", dpi=600, bbox_inches='tight')  # Save with 300 DPI
plt.show() 

#------ Plot Median Time Series R^2 of Grouped  Investors (Smaller Scale)
plt.figure(figsize=(10,6))
plt.plot(df_small_grouped['rdate'], df_small_grouped['R2_median'], label='Grouped Investors', color='red')
plt.fill_between(df_small_grouped['rdate'], df_small_grouped['R2_quantile_25_GMM'], df_small_grouped['R2_quantile_75_GMM'], color='red', alpha=0.3)

# Set xticks for every year
ax = plt.gca()  # Get current axis
ax.xaxis.set_major_locator(mdates.MonthLocator(4))  # Set major ticks to April (Q2) of each year
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format the x-axis to only show the year
plt.xlim(df['rdate'].min(), df['rdate'].max())
plt.ylim(-0.6,0.6)

# Rotate the x-axis labels for better readability
plt.xticks(rotation=90)
# Add labels and title
plt.ylabel('$R^2$',fontsize = 13)
#plt.title('Time Series Median $R^2$ GMM Estimates Grouped Investors (shade = IQR). Smaller Scale')
plt.grid(True)
plt.savefig(path + "/Output" + "/Plots/Fit" +"/R2_Median_GMM_SmallInvestors_SmallerScale.pdf", dpi=600, bbox_inches='tight')  # Save with 300 DPI
plt.show() 

#------ Plot Median Time Series R^2 of Big Investors
plt.figure(figsize=(10,6))
plt.plot(df_big_grouped['rdate'], df_big_grouped['R2_median'], label='Big Investors', color='blue')
plt.fill_between(df_big_grouped['rdate'], df_big_grouped['R2_quantile_25_GMM'], df_big_grouped['R2_quantile_75_GMM'], color='blue', alpha=0.2)

# Set xticks for every year
ax = plt.gca()  # Get current axis
ax.xaxis.set_major_locator(mdates.MonthLocator(4))  # Set major ticks to April (Q2) of each year
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format the x-axis to only show the year
plt.xlim(df['rdate'].min(), df['rdate'].max())

# Rotate the x-axis labels for better readability
plt.xticks(rotation=90,fontsize = 14)
plt.yticks(fontsize = 14)
# Add labels and title
plt.ylabel('$R^2$',fontsize = 14)
#plt.title('Time Series Median $R^2$ GMM Estimates Big Investors (shade = IQR).')
plt.grid(True)
plt.savefig(path + "/Output" + "/Plots/Fit" +"/R2_Median_GMM_BigInvestors.pdf", dpi=600, bbox_inches='tight')
plt.show() 

#------ Plot Median Time Series R^2 of Big Investors (Smaller Scale)
plt.figure(figsize=(10,6))
plt.plot(df_big_grouped['rdate'], df_big_grouped['R2_median'], label='Big Investors', color='blue')
plt.fill_between(df_big_grouped['rdate'], df_big_grouped['R2_quantile_25_GMM'], df_big_grouped['R2_quantile_75_GMM'], color='blue', alpha=0.2)

# Set xticks for every year
ax = plt.gca()  # Get current axis
ax.xaxis.set_major_locator(mdates.MonthLocator(4))  # Set major ticks to April (Q2) of each year
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format the x-axis to only show the year
plt.xlim(df['rdate'].min(), df['rdate'].max())
plt.ylim(0,1)

# Rotate the x-axis labels for better readability
plt.xticks(rotation=90)
# Add labels and title
plt.ylabel('$R^2$',fontsize = 13)
#plt.title('Time Series Median $R^2$ GMM Estimates Big Investors (shade = IQR). Smaller Scale')
plt.grid(True)
plt.savefig(path + "/Output" + "/Plots/Fit" +"/R2_Median_GMM_BigInvestors_SmallScale.pdf", dpi=600, bbox_inches='tight')
plt.show() 

#------ Plot TS Households R^2
plt.figure(figsize=(10,6))
plt.plot(df_HH['rdate'], df_HH['R_squared_GMM'], label='Households', color='green')

# Set xticks for every year
ax = plt.gca()  # Get current axis
ax.xaxis.set_major_locator(mdates.MonthLocator(4))  # Set major ticks to April (Q2) of each year
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format the x-axis to only show the year
plt.xlim(df['rdate'].min(), df['rdate'].max())

# Rotate the x-axis labels for better readability
plt.xticks(rotation=90,fontsize = 14)
plt.yticks(fontsize = 14)
# Add labels and title
plt.ylabel('$R^2$',fontsize = 14)
#plt.title('Time Series $R^2$ GMM Estimates Household.')
plt.grid(True)
plt.savefig(path + "/Output" + "/Plots/Fit" +"/R2_GMM_Households.pdf", dpi=600, bbox_inches='tight')
plt.show() 


# Create a 2x1 multiplot
fig, axes = plt.subplots(2, 1, figsize=(8, 10))  # 2 rows, 1 column

# Plot 1: Histogram for Big Investors
counts, bin_edges = np.histogram(df_big['R_squared_GMM'], bins=20)
relative_frequencies = counts / len(df_big['R_squared_GMM'])
axes[0].bar(bin_edges[:-1], relative_frequencies, width=np.diff(bin_edges), edgecolor='black', align='edge', alpha=0.7,
            label = 'Individual Investors')

# Customize Plot 1
axes[0].set_xlabel('$R^2$', fontsize=14)
axes[0].set_ylabel('Relative Frequency', fontsize=14)
axes[0].tick_params(axis='x', labelsize=14) 
axes[0].tick_params(axis='y', labelsize=14)
axes[0].grid(True)
axes[0].legend(fontsize = 14)
#axes[0].set_title('Histogram of $R^2$ GMM Estimates (Big Investors)', fontsize=15)

# Plot 2: Histogram for Grouped Investors
counts, bin_edges = np.histogram(df_small['R_squared_GMM'], bins=20)
relative_frequencies = counts / len(df_small['R_squared_GMM'])
axes[1].bar(bin_edges[:-1], relative_frequencies, width=np.diff(bin_edges), edgecolor='black', align='edge', alpha=0.7, 
            color='red', label = 'Grouped Investors')

# Customize Plot 2
axes[1].set_xlabel('$R^2$', fontsize=14)
axes[1].set_ylabel('Relative Frequency', fontsize=14)
axes[1].tick_params(axis='x', labelsize=14) 
axes[1].tick_params(axis='y', labelsize=14) 
axes[1].grid(True)
axes[1].legend(fontsize = 14)
#axes[1].set_title('Histogram of $R^2$ GMM Estimates (Grouped Investors)', fontsize=15)

# Adjust layout and save the multiplot
plt.tight_layout()
plt.savefig(path + "/Output/Plots/Fit/R2_Histograms_GMM.pdf", dpi=600, bbox_inches='tight')

# Display the multiplot
plt.show()

#%% Plot R^2 for NLLS Estimates

df = pd.read_csv(path + "/Output/Fit" + "/NLLS_CV_R2.csv")
df['rdate'] =  pd.to_datetime(df["rdate"]) #if reading in csv

    
#Cut last two Quarters off so that the x-ticks nicely align in the plot
df = df[df['rdate'] < '2022-09-30']
df = df[df['rdate']!= '2013-03-31'] #Cut out this Quarter because Data is missing
#df = df[df['rdate'].dt.year !=2013] #Cut out this year because Data is missing and scarce

#Filter outliers
df = df[df['R_squared_NLLS'] > df['R_squared_NLLS'].quantile(0.02)]

#Construct Household, Big & Grouped Investors
df_HH = df[df['bin'] == 0]

#Compute Median and Quantiles of Small and Big Investors
df_small = df[df['bin'] < 191]
df_small_grouped = df_small.groupby('rdate').agg(
    R2_median=('R_squared_NLLS', 'median'),
    R2_quantile_25_NLLS=('R_squared_NLLS', lambda x: x.quantile(0.25)),
    R2_quantile_75_NLLS=('R_squared_NLLS', lambda x: x.quantile(0.75)),
).reset_index()

df_big = df[df['bin'] > 190]
df_big_grouped = df_big.groupby('rdate').agg(
    R2_median=('R_squared_NLLS', 'median'),
    R2_quantile_25_NLLS=('R_squared_NLLS', lambda x: x.quantile(0.25)),
    R2_quantile_75_NLLS=('R_squared_NLLS', lambda x: x.quantile(0.75)),
).reset_index()

#Group all R^2 < 0 to 0 because otherwise Histogramm cannot be reasonably visualised
df_small.loc[df_small['R_squared_NLLS'] < 0, 'R_squared_NLLS'] = -0.1
df_big.loc[df_big['R_squared_NLLS'] < 0, 'R_squared_NLLS'] = -0.1


#------ Plot TS Households R^2
plt.figure(figsize=(10,6))
plt.plot(df_HH[df_HH['R_squared_NLLS']>0]['rdate'], df_HH[df_HH['R_squared_NLLS']>0]['R_squared_NLLS'], label='Households', color='green')

# Set xticks for every year
ax = plt.gca()  # Get current axis
ax.xaxis.set_major_locator(mdates.MonthLocator(4))  # Set major ticks to April (Q2) of each year
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format the x-axis to only show the year
plt.xlim(df['rdate'].min(), df['rdate'].max())

# Rotate the x-axis labels for better readability
plt.xticks(rotation=90)
# Add labels and title
plt.ylabel('$R^2$')
#plt.title('Time Series $R^2$ NLLS Estimates Household.')
plt.savefig(path + "/Output" + "/Plots/Fit" +"/R2_NLLS_Households.pdf", dpi=600)
plt.show() 


#------ Plot TS Individual & Grouped Investors R^2
# Create a 2x1 multiplot
fig, axes = plt.subplots(2, 1, figsize=(8, 10))  # 2 rows, 1 column

# Plot 1: Median Time Series for Big Investors
axes[0].plot(df_big_grouped['rdate'], df_big_grouped['R2_median'], label='Individual Investors', color='blue')
axes[0].fill_between(df_big_grouped['rdate'], df_big_grouped['R2_quantile_25_NLLS'], df_big_grouped['R2_quantile_75_NLLS'], color='blue', alpha=0.3)

# Customize Plot 1
axes[0].set_ylabel('$R^2$', fontsize=14)
#axes[0].set_title('Time Series Median $R^2$ NLLS Estimates (Big Investors)', fontsize=15)
axes[0].legend(fontsize=14)
axes[0].tick_params(axis='x', rotation=90, labelsize=14) 
axes[0].tick_params(axis='y', labelsize=14)
axes[0].grid(True)

# Set xticks for Plot 1
axes[0].xaxis.set_major_locator(mdates.MonthLocator(4))  # Set major ticks to April (Q2) of each year
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format the x-axis to only show the year
axes[0].set_xlim(df['rdate'].min(), df['rdate'].max())

# Plot 2: Median Time Series for Grouped Investors
axes[1].plot(df_small_grouped['rdate'], df_small_grouped['R2_median'], label='Grouped Investors', color='red')
axes[1].fill_between(df_small_grouped['rdate'], df_small_grouped['R2_quantile_25_NLLS'], df_small_grouped['R2_quantile_75_NLLS'], color='red', alpha=0.3)

# Customize Plot 2
axes[1].set_ylabel('$R^2$', fontsize=14)
#axes[1].set_title('Time Series Median $R^2$ NLLS Estimates (Grouped Investors)', fontsize=15)
axes[1].legend(fontsize=14)
axes[1].tick_params(axis='x', rotation = 90, labelsize=14) 
axes[1].tick_params(axis='y', labelsize=14)
axes[1].grid(True)

# Set xticks for Plot 2
axes[1].xaxis.set_major_locator(mdates.MonthLocator(4))  # Set major ticks to April (Q2) of each year
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format the x-axis to only show the year
axes[1].set_xlim(df['rdate'].min(), df['rdate'].max())

# Adjust layout and save the multiplot
plt.tight_layout()
plt.savefig(path + "/Output/Plots/Fit/R2_Median_NLLS.pdf", dpi=600, bbox_inches='tight')

# Display the multiplot
plt.show()


#------ Plot Histogram of R^2 for Big Investors
# Create a 2x1 multiplot
fig, axes = plt.subplots(2, 1, figsize=(8, 10))  # 2 rows, 1 column

# Plot 1: Histogram for Big Investors
counts, bin_edges = np.histogram(df_big['R_squared_NLLS'], bins=20)
relative_frequencies = counts / len(df_big['R_squared_NLLS'])
axes[0].bar(bin_edges[:-1], relative_frequencies, width=np.diff(bin_edges), edgecolor='black', align='edge', alpha=0.7, label='Big Investors')

# Customize Plot 1
axes[0].set_xlabel('$R^2$', fontsize=14)
axes[0].set_ylabel('Relative Frequency', fontsize=14)
axes[0].grid(True)
#axes[0].set_title('Histogram of $R^2$ NLLS Estimates (Big Investors)', fontsize=14)
axes[0].legend(fontsize=14)
axes[0].tick_params(axis='x', labelsize=14) 
axes[0].tick_params(axis='y', labelsize=14)

# Plot 2: Histogram for Grouped Investors
counts, bin_edges = np.histogram(df_small['R_squared_NLLS'], bins=20)
relative_frequencies = counts / len(df_small['R_squared_NLLS'])
axes[1].bar(bin_edges[:-1], relative_frequencies, width=np.diff(bin_edges), edgecolor='black', align='edge', alpha=0.7, color='red', label='Grouped Investors')

# Customize Plot 2
axes[1].set_xlabel('$R^2$', fontsize=14)
axes[1].set_ylabel('Relative Frequency', fontsize=14)
axes[1].grid(True)
axes[1].tick_params(axis='x', labelsize=14) 
axes[1].tick_params(axis='y', labelsize=14)
#axes[1].set_title('Histogram of $R^2$ NLLS Estimates (Grouped Investors)', fontsize=15)
axes[1].legend(fontsize=14)

# Adjust layout and save the multiplot
plt.tight_layout()
plt.savefig(path + "/Output/Plots/Fit/R2_Histograms_NLLS.pdf", dpi=600, bbox_inches='tight')

# Display the multiplot
plt.show()

#%% Plot lambda for NLLS to show that small Bins are heterogeneous

#Read in Data
df = pd.read_csv(path + "/Output" + "/Estimations" + "/NLLS_Estimates_CV_Restricted_SMAE.csv")
df['rdate'] =  pd.to_datetime(df["rdate"]) #if reading in csv
df.drop(['index', "Unnamed: 0"],axis = 1, inplace=True)

#Cut last two Quarters off so that the x-ticks nicely align in the plot
df = df[df['rdate'] < '2022-09-30']
df = df[df['rdate']!= '2013-03-31'] #Cut out this Quarter because Data is missing

#Make an Indicator for Small Bins
df['Small_Bin'] = (df['bin'] < 191).astype(int)

df_grouped = df.groupby(['rdate', 'Small_Bin']).agg(
    lambda_median=('lam', 'median'),
    lambda_quantile_25=('lam', lambda x: x.quantile(0.4)),
    lambda_quantile_75=('lam', lambda x: x.quantile(0.6)),
).reset_index()

df_small_grouped = df_grouped[df_grouped['Small_Bin'] == 1]
df_big_grouped = df_grouped[df_grouped['Small_Bin'] == 0]

#------ Plot Times Series of Lambda

# Create a 2x1 multiplot
fig, axes = plt.subplots(2, 1, figsize=(8, 10))  # 2 rows, 1 column

# Plot 1: Median Time Series Lambda for Big Investors
axes[0].plot(df_big_grouped['rdate'], df_big_grouped['lambda_median'], label='Big Investors', color='blue')
axes[0].fill_between(df_big_grouped['rdate'], df_big_grouped['lambda_quantile_25'], df_big_grouped['lambda_quantile_75'], color='blue', alpha=0.3)

# Customize Plot 1
axes[0].set_ylabel('$\lambda$', fontsize=14)
#axes[0].set_title('Time Series Median $\lambda$ (Penalty Parameter) - Big Investors', fontsize=15)
axes[0].legend(fontsize=14)
axes[0].grid(True)
axes[0].tick_params(axis='y', labelsize=14)

# Set xticks for Plot 1
axes[0].xaxis.set_major_locator(mdates.MonthLocator(4))  # Set major ticks to April (Q2) of each year
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format the x-axis to only show the year
axes[0].set_xlim(df['rdate'].min(), df['rdate'].max())
axes[0].tick_params(axis='x', rotation=90, labelsize=14)  # Rotate x-axis labels

# Plot 2: Median Time Series Lambda for Grouped Investors
axes[1].plot(df_small_grouped['rdate'], df_small_grouped['lambda_median'], label='Grouped Investors', color='red')
axes[1].fill_between(df_small_grouped['rdate'], df_small_grouped['lambda_quantile_25'], df_small_grouped['lambda_quantile_75'], color='red', alpha=0.3)

# Customize Plot 2
axes[1].set_ylabel('$\lambda$', fontsize=14)
#axes[1].set_title('Time Series Median $\lambda$ (Penalty Parameter) - Grouped Investors', fontsize=15)
axes[1].legend(fontsize=14)
axes[1].tick_params(axis='x', labelsize=14) 
axes[1].grid(True)

# Set xticks for Plot 2
axes[1].xaxis.set_major_locator(mdates.MonthLocator(4))  # Set major ticks to April (Q2) of each year
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format the x-axis to only show the year
axes[1].set_xlim(df['rdate'].min(), df['rdate'].max())
axes[1].tick_params(axis='x', rotation=90, labelsize=14)  # Rotate x-axis labels

# Adjust layout and save the multiplot
plt.tight_layout()
plt.savefig(path + "/Output/Plots/Fit/Lambda_Median_NLLS.pdf", dpi=600, bbox_inches='tight')

# Display the multiplot
plt.show()
