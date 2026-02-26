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

#Load Holdings Data
Holdings = pd.read_stata(path + "/Data" + "/Data1_clean_correct_bins.dta")
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


#%% Error Analysis for Baseline Unrestricted Estimates (else-wise with restricted Estimated you'd kick out all big investors where the fit is good)

#Read in appropriate Data
df_Estimates = pd.read_csv(path + "/Output/Estimations" + "/GMM_Estimates_Baseline_Restricted.csv")
df_Estimates['rdate'] =  pd.to_datetime(df_Estimates["rdate"]) #if reading in csv
df_Estimates = df_Estimates.drop_duplicates(subset = ['rdate','bin'])

df_UnrestrictedEstimates = pd.read_csv(path + "/Output/Estimations" + "/GMM_Estimates_Baseline_Unrestricted.csv")
df_UnrestrictedEstimates['rdate'] =  pd.to_datetime(df_UnrestrictedEstimates["rdate"]) #if reading in csv
df_UnrestrictedEstimates = df_UnrestrictedEstimates.drop_duplicates(subset = ['rdate','bin'])

#Computing Epsilon Function
def Epsilon(RegCoeffs,X,y):
    """
    Inputs
    ----------
    RegCoeffs : numpy vector of Regression Coefficients to be estimated (numerical solvers iterate over RegCoeffs)
    X : numpy matrix of Explanatory Variables which includes the column LNme (but not IVme!). Additionally,
        the last column must be 'cons'. 'cons' has no RegCoeff attached to it.
    y : numpy vector of relative weights (including 0 rweights)

    Returns
    -------
    numpy vector error term. Each element is the prediction error
    for an investor i holding a stock n.
    """
    return y* np.exp( -X[:,:-1] @ RegCoeffs - X[:,X.shape[1]-1] ) - 1

#Extract unique dates
Quarters = Holdings['rdate'].unique()
Quarters = Quarters[Quarters.year<2023]

#Prepare final results table
Results = pd.DataFrame(columns=['rdate', 'bin', 'Error_Mean_All', 'Error_Mean_NonZeros'])

#Loop over all Quarters
for quarter in Quarters:
    print(quarter)

    ### --- Slice Datasets
    #Holdings Sliced
    Holdings_Q = Holdings[Holdings['rdate'] == quarter]

    #Baseline Stock Characteristics Sliced
    StocksQ_Q = StocksQ[StocksQ['date'] == quarter]
    
    #Estimates Sliced
    df_UnrestrictedEstimates_Q = df_UnrestrictedEstimates[df_UnrestrictedEstimates['rdate']==quarter]

    ### --- Merge Stock Characteristics to Holdings Data to build X & Z Matrix for GMM
    #Merge Baseline Stock Characteristics
    df_Q = Holdings_Q.merge(StocksQ_Q[["permno", "date", "LNme", 'LNbe', 'profit', 'Gat', 'divA_be','beta']], left_on=["rdate", "permno"], right_on=["date", "permno"])

    #Drop any remaining Missing Values to avoid errors
    df_Q = df_Q.dropna(subset=Baseline_endog_Variables_Names + ['IVme','rweight'])

    #Assing the constant to the dataframe
    df_Q = df_Q.assign(constant=1)

    for i_bin in np.sort(df_Q['bin'].unique()):
        print("     Bin: " + str(i_bin))
        
        ### --- Slice DataSet on Bin
        df_Q_bin = df_Q[df_Q['bin'] == i_bin]
        
        #Slice Estimates on Bin
        df_UnrestrictedEstimates_bin = df_UnrestrictedEstimates_Q[df_UnrestrictedEstimates_Q['bin'] == i_bin]
        
        #Generate Results Dataframe
        Results_bin = pd.DataFrame(columns=['rdate', 'bin', 'Error_Mean_All', 'Error_Mean_NonZeros'])
        Results_bin.at[0,'rdate'] = quarter
        Results_bin.at[0,'bin'] = i_bin
        #Results_bin.set_index(['rdate', 'bin'],inplace=True)
        
        #Extract Data
        X = df_Q_bin[['rweight'] + Baseline_endog_Variables_Names + ['constant','cons']]
        
        X = X.dropna()
        
        #Extract variables from Data
        y_all = X['rweight'].values
        X_all = copy.deepcopy(X.drop('rweight',axis=1).values)
        
        #Extract exlusively non-zero variables from Data
        X_nz = X[X['rweight']>0]
        y_nz = X_nz['rweight'].values
        X_nz = X_nz.drop('rweight',axis=1).values
        
        #Extract regression coefficients
        beta = df_UnrestrictedEstimates_bin[Baseline_endog_Variables_Names + ['constant']].values.reshape(-1)
        
        #Compute vector of errors (Epsilon() = epsilon-1)
        epsilon_all = Epsilon(beta,X_all,y_all) + 1
        epsilon_nz = Epsilon(beta,X_nz,y_nz) + 1
        
        #Mean of non-zero holdings
        mean_epsilon_all = np.mean(epsilon_all)
        mean_epsilon_nz = np.mean(epsilon_nz)
                               
        Results_bin.loc[:,'Error_Mean_All'] = mean_epsilon_all
        Results_bin.loc[:,'Error_Mean_NonZeros'] = mean_epsilon_nz
        Results = pd.concat([Results,Results_bin])

        
#group_averages = Results.groupby('rdate')['Error_Mean_NonZeros'].mean()

#Results = Results[Results['Error_Mean']<10]
#Results['Error_Mean'].mean()

Results.to_csv(path + "/Output" + "/Error_Unrestricted_BaselineConditionalExpectation.csv")

#%% Error Analysis for AdjustedConditionalExpectation Unrestricted Estimates (else-wise with restricted Estimated you'd kick out all big investors where the fit is good)

df_Estimates = pd.read_csv(path + "/Output/Estimations" + "/GMM_Estimates_Baseline_Expectataion065_Restricted.csv")
df_Estimates['rdate'] =  pd.to_datetime(df_Estimates["rdate"]) #if reading in csv
df_Estimates = df_Estimates.drop_duplicates(subset = ['rdate','bin'])

df_UnrestrictedEstimates = pd.read_csv(path + "/Output/Estimations" + "/GMM_Estimates_Baseline_Expectation065_Unrestricted.csv")
df_UnrestrictedEstimates['rdate'] =  pd.to_datetime(df_UnrestrictedEstimates["rdate"]) #if reading in csv
df_UnrestrictedEstimates = df_UnrestrictedEstimates.drop_duplicates(subset = ['rdate','bin'])

#Computing Epsilon Function
def Epsilon(RegCoeffs,X,y):
    """
    Inputs
    ----------
    RegCoeffs : numpy vector of Regression Coefficients to be estimated (numerical solvers iterate over RegCoeffs)
    X : numpy matrix of Explanatory Variables which includes the column LNme (but not IVme!). Additionally,
        the last column must be 'cons'. 'cons' has no RegCoeff attached to it.
    y : numpy vector of relative weights (including 0 rweights)

    Returns
    -------
    numpy vector error term. Each element is the prediction error
    for an investor i holding a stock n.
    """
    return y* np.exp( -X[:,:-1] @ RegCoeffs - X[:,X.shape[1]-1] ) - 0.65

#Extract unique dates
Quarters = Holdings['rdate'].unique()
Quarters = Quarters[Quarters.year<2023]

#Prepare final results table
Results = pd.DataFrame(columns=['rdate', 'bin', 'Error_Mean_All', 'Error_Mean_NonZeros'])

#Loop over all Quarters
for quarter in Quarters:
    print(quarter)

    ### --- Slice Datasets
    #Holdings Sliced
    Holdings_Q = Holdings[Holdings['rdate'] == quarter]

    #Baseline Stock Characteristics Sliced
    StocksQ_Q = StocksQ[StocksQ['date'] == quarter]
    
    #Estimates Sliced
    df_UnrestrictedEstimates_Q = df_UnrestrictedEstimates[df_UnrestrictedEstimates['rdate']==quarter]

    ### --- Merge Stock Characteristics to Holdings Data to build X & Z Matrix for GMM
    #Merge Baseline Stock Characteristics
    df_Q = Holdings_Q.merge(StocksQ_Q[["permno", "date", "LNme", 'LNbe', 'profit', 'Gat', 'divA_be','beta']], left_on=["rdate", "permno"], right_on=["date", "permno"])

    #Drop any remaining Missing Values to avoid errors
    df_Q = df_Q.dropna(subset=Baseline_endog_Variables_Names + ['IVme','rweight'])

    #Assing the constant to the dataframe
    df_Q = df_Q.assign(constant=1)

    for i_bin in np.sort(df_Q['bin'].unique()):
        print("     Bin: " + str(i_bin))
        
        ### --- Slice DataSet on Bin
        df_Q_bin = df_Q[df_Q['bin'] == i_bin]
        
        #Slice Estimates on Bin
        df_UnrestrictedEstimates_bin = df_UnrestrictedEstimates_Q[df_UnrestrictedEstimates_Q['bin'] == i_bin]
        
        #Generate Results Dataframe
        Results_bin = pd.DataFrame(columns=['rdate', 'bin', 'Error_Mean'])
        Results_bin.at[0,'rdate'] = quarter
        Results_bin.at[0,'bin'] = i_bin
        #Results_bin.set_index(['rdate', 'bin'],inplace=True)
        
        #Extract Data
        X = df_Q_bin[['rweight'] + Baseline_endog_Variables_Names + ['constant','cons']]
        
        X = X.dropna()
        
        #Extract variables from Data
        y_all = X['rweight'].values
        X_all = copy.deepcopy(X.drop('rweight',axis=1).values)
        
        #Extract exlusively non-zero variables from Data
        X_nz = X[X['rweight']>0]
        y_nz = X_nz['rweight'].values
        X_nz = X_nz.drop('rweight',axis=1).values
        
        #Extract regression coefficients
        beta = df_UnrestrictedEstimates_bin[Baseline_endog_Variables_Names + ['constant']].values.reshape(-1)
        
        #Compute vector of errors (Epsilon() = epsilon-0.65)
        epsilon_all = Epsilon(beta,X_all,y_all) + 0.65
        epsilon_nz = Epsilon(beta,X_nz,y_nz) + 0.65
        
        #Mean of non-zero holdings
        mean_epsilon_all = np.mean(epsilon_all)
        mean_epsilon_nz = np.mean(epsilon_nz)
        
        Results_bin.loc[:,'Error_Mean_All'] = mean_epsilon_all
        Results_bin.loc[:,'Error_Mean_NonZeros'] = mean_epsilon_nz
        
        
        
        Results = pd.concat([Results,Results_bin])
        #Means fluctuate and are not 1 of the non-zeros (bias in the prediction)
        
#Results = Results[Results['Error_Mean']<10]
#Results['Error_Mean'].mean()

#group_averages = Results.groupby('rdate')['Error_Mean_NonZeros'].mean()
#However, now the average of the means is indeed 1, but each bin has a bias prediction.
#This can be visualised with a Histogram.

#So Error 0.65 is irrelevant for the slope coefficients, but it is very vital
#for the prediction as it affects the constant. Since we are interested in the
#prediction and not in the slope coefficients, this is really important!
#Notice However that the Mean Error 0.65 is ad-hoc and reverse engineered from
#the data. With a different share of zeros, we'd have to adjust this.

Results.to_csv(path + "/Output" + "/Error_Unrestricted_AdjustedConditionalExpectation.csv")

#%% Plot 1

#Time Series Means of Epsilons per Quarter (and their IQR) of Non-Zero Holdings
df_Baseline = pd.read_csv(path + "/Output" + "/Error_Unrestricted_BaselineConditionalExpectation.csv")
df_Baseline['rdate'] =  pd.to_datetime(df_Baseline["rdate"]) #if reading in csv

#Data cleaning
df_Baseline = df_Baseline[df_Baseline['Error_Mean_NonZeros'] < 10]


data =  df_Baseline.groupby('rdate')['Error_Mean_NonZeros'].agg(
    mean_of_means=('mean'),
    quantile_25=lambda x: x.quantile(0.25),
    quantile_75=lambda x: x.quantile(0.75)
).reset_index()

#Data cleaning
data = data[data['mean_of_means'] < 10]

#Cut last two Quarters off so that the x-ticks nicely align in the plot
data = data[data['rdate'] < '2022-09-30']

# Plotting
plt.figure(figsize=(10,6))

# Plot the mean (x)
plt.plot(data['rdate'], data['mean_of_means'], label='Mean of Means Epsilon', color='blue')

# Shade the area between the 10% quantile (y) and the 90% quantile (z)
plt.fill_between(data['rdate'], data['quantile_25'], data['quantile_75'], color='blue', alpha=0.3, label='IQR')

# Add a dotted horizontal line at the overall Time Series Mean
plt.axhline(y=data['mean_of_means'].mean(), color='blue', alpha=0.5, linestyle='--', linewidth=1.5, label='Overall TS Mean')

# Set xticks for every year
ax = plt.gca()  # Get current axis
ax.xaxis.set_major_locator(mdates.MonthLocator(4))  # Set major ticks to April (Q2) of each year
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format the x-axis to only show the year
plt.xlim(data['rdate'].min(), data['rdate'].max())

# Rotate the x-axis labels for better readability
plt.xticks(rotation=90)

# Add labels and title
#plt.xlabel('Date')
plt.ylabel('Mean of Means of Epsilon (Unbiased = 1)')
plt.title('Time Series of Mean of Means of Epsilon of all Bins grouped by Quarter, Non-Zeros only')
plt.legend()

#Save Plot
plt.savefig(path + "/Output" + "/Plots" +"/TS_Epsilon_Baseline_AllInvestors.png", dpi=700)#, bbox_inches='tight')  # Save with 300 DPI

# Display the plot
plt.show() #Plot looks virtually the Same with Medians of Means Epsilon

#%% Plot 2 

#Time Series Means of Epsilons per Quarter (and their IQR) FOR BIG (non-grouped) INVESTORS ONLY of Non-Zero Holdings
df_Baseline = pd.read_csv(path + "/Output" + "/Error_Unrestricted_BaselineConditionalExpectation.csv")
df_Baseline['rdate'] =  pd.to_datetime(df_Baseline["rdate"]) #if reading in csv

#Data cleaning
df_Baseline = df_Baseline[df_Baseline['Error_Mean_NonZeros'] < 10]

#Big investors only
df_Baseline = df_Baseline[df_Baseline['bin'] > 190]

data =  df_Baseline.groupby('rdate')['Error_Mean_NonZeros'].agg(
    mean_of_means=('mean'),
    quantile_25=lambda x: x.quantile(0.25),
    quantile_75=lambda x: x.quantile(0.75)
).reset_index()

#Data cleaning
data = data[data['mean_of_means'] < 10]

#Cut last two Quarters off so that the x-ticks nicely align in the plot
data = data[data['rdate'] < '2022-09-30']

# Plotting
plt.figure(figsize=(10,6))

# Plot the mean (x)
plt.plot(data['rdate'], data['mean_of_means'], label='Mean of Means Epsilon', color='blue')

# Shade the area between the 10% quantile (y) and the 90% quantile (z)
plt.fill_between(data['rdate'], data['quantile_25'], data['quantile_75'], color='blue', alpha=0.3, label='IQR')

# Add a dotted horizontal line at the overall Time Series Mean
plt.axhline(y=data['mean_of_means'].mean(), color='blue', alpha = 0.5, linestyle='--', linewidth=1.5, label='Overall TS Mean')

# Set xticks for every year
ax = plt.gca()  # Get current axis
ax.xaxis.set_major_locator(mdates.MonthLocator(4))  # Set major ticks to April (Q2) of each year
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format the x-axis to only show the year
plt.xlim(data['rdate'].min(), data['rdate'].max())

# Rotate the x-axis labels for better readability
plt.xticks(rotation=90)

# Add labels and title
#plt.xlabel('Date')
plt.ylabel('Mean of Means of Epsilon (Unbiased = 1)')
plt.title('Time Series of Mean of Means of Epsilon of big Investors grouped by Quarter, Non-Zeros only')
plt.legend()

#Save Plot
plt.savefig(path + "/Output" + "/Plots" +"/TS_Epsilon_Baseline_BigInvestors.png", dpi=700)#, bbox_inches='tight')  # Save with 300 DPI

# Display the plot
plt.show() 
#Bias not as bad, but given that these investors have a lot of AUM, even 'only' 20% (insteaf of 50%) of a shit ton of money is a shit ton of money

#%% Plot 2.5

"""
This plot is pointless because if we use ALL Estimates, then the GMM estimate (as it converged) sets the
mean of epsilon to the pre-defined level (e.g. 1 or 0.65)
"""

#Time Series Means of Epsilons per Quarter (and their IQR) of ALL Holdings
df_Baseline = pd.read_csv(path + "/Output" + "/Error_Unrestricted_BaselineConditionalExpectation.csv")
df_Baseline['rdate'] =  pd.to_datetime(df_Baseline["rdate"]) #if reading in csv

#Data cleaning
df_Baseline = df_Baseline[df_Baseline['Error_Mean_All'] < 10]


data =  df_Baseline.groupby('rdate')['Error_Mean_All'].agg(
    mean_of_means=('mean'),
    quantile_25=lambda x: x.quantile(0.25),
    quantile_75=lambda x: x.quantile(0.75)
).reset_index()

#Data cleaning
data = data[data['mean_of_means'] < 10]

#Cut last two Quarters off so that the x-ticks nicely align in the plot
data = data[data['rdate'] < '2022-09-30']

# Plotting
plt.figure(figsize=(10,6))

# Plot the mean (x)
plt.plot(data['rdate'], data['mean_of_means'], label='Mean of Means Epsilon', color='blue')

# Shade the area between the 10% quantile (y) and the 90% quantile (z)
plt.fill_between(data['rdate'], data['quantile_25'], data['quantile_75'], color='blue', alpha=0.3, label='IQR')

# Add a dotted horizontal line at the overall Time Series Mean
plt.axhline(y=data['mean_of_means'].mean(), color='blue', alpha=0.5, linestyle='--', linewidth=1.5, label='Overall TS Mean')

# Set xticks for every year
ax = plt.gca()  # Get current axis
ax.xaxis.set_major_locator(mdates.MonthLocator(4))  # Set major ticks to April (Q2) of each year
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format the x-axis to only show the year
plt.xlim(data['rdate'].min(), data['rdate'].max())

# Rotate the x-axis labels for better readability
plt.xticks(rotation=90)

# Add labels and title
#plt.xlabel('Date')
plt.ylabel('Mean of Means of Epsilon (Unbiased = 1)')
plt.title('Time Series of Mean of Means of Epsilon of all Bins grouped by Quarter, All Estimates')
plt.legend()

#Save Plot
plt.savefig(path + "/Output" + "/Plots" +"/TS_Epsilon_Baseline_AllInvestors_AllHoldings.png", dpi=700)#, bbox_inches='tight')  # Save with 300 DPI

# Display the plot
plt.show()

#%% Plot 3

#Time Series Means of Epsilons per Quarter (and their IQR) with adjusted conditional Expectation of Non-Zero Holdings
df_Baseline = pd.read_csv(path + "/Output" + "/Error_Unrestricted_AdjustedConditionalExpectation.csv")
df_Baseline['rdate'] =  pd.to_datetime(df_Baseline["rdate"]) #if reading in csv

#Data cleaning
df_Baseline = df_Baseline[df_Baseline['Error_Mean_NonZeros'] < 10]

data =  df_Baseline.groupby('rdate')['Error_Mean_NonZeros'].agg(
    mean_of_means=('mean'),
    quantile_25=lambda x: x.quantile(0.25),
    quantile_75=lambda x: x.quantile(0.75)
).reset_index()

#Data cleaning
data = data[data['mean_of_means'] < 10]

#Cut last two Quarters off so that the x-ticks nicely align in the plot
data = data[data['rdate'] < '2022-09-30']

# Plotting
plt.figure(figsize=(10,6))

# Plot the mean (x)
plt.plot(data['rdate'], data['mean_of_means'], label='Mean of Means Epsilon', color='blue')

# Shade the area between the 10% quantile (y) and the 90% quantile (z)
plt.fill_between(data['rdate'], data['quantile_25'], data['quantile_75'], color='blue', alpha=0.3, label='IQR')

# Add a dotted horizontal line at the overall Time Series Mean
plt.axhline(y=data['mean_of_means'].mean(), color='blue', alpha=0.5, linestyle='--', linewidth=1.5, label='Overall TS Mean')

# Set xticks for every year
ax = plt.gca()  # Get current axis
ax.xaxis.set_major_locator(mdates.MonthLocator(4))  # Set major ticks to April (Q2) of each year
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format the x-axis to only show the year
plt.xlim(data['rdate'].min(), data['rdate'].max())

# Rotate the x-axis labels for better readability
plt.xticks(rotation=90)

# Add labels and title
#plt.xlabel('Date')
plt.ylabel('Mean of Means of Epsilon (Unbiased = 1)')
plt.title('Time Series of Mean of Means of Epsilon of all Bins grouped by Quarter with adjusted conditional Expectation, Non-Zeros only')
plt.legend()

#Save Plot
plt.savefig(path + "/Output" + "/Plots" +"/TS_Epsilon_AdjustedCondExp_AllInvestors.png", dpi=700)#, bbox_inches='tight')  # Save with 300 DPI


# Display the plot
plt.show() 
#Dieser Plot ist genauso schlecht und im Verlauf exakt derselbe wie Plot 1, nur nach unten geschoben weil die Konstante
#adjustiert ist. Bei diesem Plot gilt 'ein Mal links und ein mal recht am Tor vorbeigeschossen ist im Schnitt ein Treffer',
#weil die over und under estimates von demand im Schnitt sich ausgleichen.

#%% Plot 4

#Time Series Means of Epsilons per Quarter (and their IQR) FOR BIG (non-grouped) INVESTORS ONLY with adjusted conditional Expectation of Non-Zero Holdings
df_Baseline = pd.read_csv(path + "/Output" + "/Error_Unrestricted_AdjustedConditionalExpectation.csv")
df_Baseline['rdate'] =  pd.to_datetime(df_Baseline["rdate"]) #if reading in csv

#Data cleaning
df_Baseline = df_Baseline[df_Baseline['Error_Mean_NonZeros'] < 10]

#Big investors only
df_Baseline = df_Baseline[df_Baseline['bin'] > 190]

data =  df_Baseline.groupby('rdate')['Error_Mean_NonZeros'].agg(
    mean_of_means=('mean'),
    quantile_25=lambda x: x.quantile(0.25),
    quantile_75=lambda x: x.quantile(0.75)
).reset_index()

#Data cleaning
data = data[data['mean_of_means'] < 10]

#Cut last two Quarters off so that the x-ticks nicely align in the plot
data = data[data['rdate'] < '2022-09-30']

# Plotting
plt.figure(figsize=(10,6))

# Plot the mean (x)
plt.plot(data['rdate'], data['mean_of_means'], label='Mean of Means Epsilon', color='blue')

# Shade the area between the 10% quantile (y) and the 90% quantile (z)
plt.fill_between(data['rdate'], data['quantile_25'], data['quantile_75'], color='blue', alpha=0.3, label='IQR')

# Add a dotted horizontal line at the overall Time Series Mean
plt.axhline(y=data['mean_of_means'].mean(), color='blue', alpha=0.5, linestyle='--', linewidth=1.5, label='Overall TS Mean')

# Set xticks for every year
ax = plt.gca()  # Get current axis
ax.xaxis.set_major_locator(mdates.MonthLocator(4))  # Set major ticks to April (Q2) of each year
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format the x-axis to only show the year
plt.xlim(data['rdate'].min(), data['rdate'].max())

# Rotate the x-axis labels for better readability
plt.xticks(rotation=90)

# Add labels and title
#plt.xlabel('Date')
plt.ylabel('Mean of Means of Epsilon (Unbiased = 1)')
plt.title('Time Series of Mean of Means of Epsilon of big Investors grouped by Quarter with adjusted conditional Expectation, Non-Zeros only')
plt.legend()

#Save Plot
plt.savefig(path + "/Output" + "/Plots" +"/TS_Epsilon_AdjustedCondExp_BigInvestors.png", dpi=700)#, bbox_inches='tight')  # Save with 300 DPI

# Display the plot
plt.show() 
#Dieser Plot ist genauso schlecht und im Verlauf exakt derselbe wie Plot 1, nur nach unten geschoben weil die Konstante
#adjustiert ist. Bei diesem Plot gilt 'ein Mal links und ein mal recht am Tor vorbeigeschossen ist im Schnitt ein Treffer',
#weil die over und under estimates von demand im Schnitt sich ausgleichen.
