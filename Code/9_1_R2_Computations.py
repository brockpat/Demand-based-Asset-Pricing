# -*- coding: utf-8 -*-
"""
Computes the R^2
"""

#%% Libraries
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

path = "C:/Users/pbrock/Desktop/KY19_Extension"

#%% Read in Data

#Greene Theorem THEOREM 13.2 and Newey & McFadden (1994) Theorem 3.4, Wikipedia: https://en.wikipedia.org/wiki/Generalized_method_of_moments


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

#---------------------------- OLS Estimates ----------------------------------

OLS_Estimates = pd.read_csv(path + "/Output" + "/Estimations" + "/OLS_Estimates_BaselineWithR2.csv")
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

#%% Compute R^2 for level y for OLS Estimates

#df = copy.deepcopy(OLS_Estimates)
#Cut last two Quarters off so that the x-ticks nicely align in the plot
#df = df[(df['rdate'] < '2022-09-30') & (df['rdate'] != '2013-03-31')]

#Extract unique dates
Quarters = Holdings['rdate'].unique()
Quarters = Quarters[Quarters.year<2023]

#Create DataFrame to store results
df = []

#------- Loop over all Quarters and compute the fitted level value rweight_hat
for quarter in Quarters:
    print(quarter)

    ### --- Slice Datasets
    #Holdings Sliced
    Holdings_Q = Holdings[Holdings['rdate'] == quarter][['rdate','bin','mgrno','permno','rweight','cons']]
    Holdings_Q = Holdings_Q[Holdings_Q['rweight']> 0] #Since OLS Results only make sense if they apply to Non-Zeros

    #Baseline Stock Characteristics Sliced
    StocksQ_Q = StocksQ[StocksQ['date'] == quarter]

    ### --- Merge Stock Characteristics to Holdings Data to build X & Z Matrix for GMM
    #Merge Baseline Stock Characteristics
    df_Q = Holdings_Q.merge(StocksQ_Q[["permno", "date", "LNme", 'LNbe', 'profit', 'Gat', 'divA_be','beta']], 
                            left_on=["rdate", "permno"], right_on=["date", "permno"])

    #Drop any remaining Missing Values to avoid errors
    df_Q = df_Q.dropna(subset=Baseline_endog_Variables_Names + ['rweight'])

    df_Q = df_Q.merge(OLS_Estimates.add_suffix("_beta"), 
                      left_on = ['rdate','bin'], 
                      right_on = ['rdate_beta','bin_beta'],
                      how = 'left', 
                      suffixes=('', ''))
    df_Q.drop(['rdate_beta','bin_beta'],axis = 1,inplace=True)

    #Assing the constant to the dataframe
    df_Q = df_Q.assign(constant=1)
    
    #Compute fitted Values
    Baseline_endog_Variables_Names = ['LNme','LNbe', 'profit', 'Gat', 'divA_be','beta']

    Baseline_Regressors = [item + "_beta" for item in Baseline_endog_Variables_Names + ['constant']]
    
    X = df_Q[Baseline_endog_Variables_Names + ['constant']].values
    beta = df_Q[Baseline_Regressors].values.T
    
    #Get predicted values. i-th predicted value is dot product of i-th row of X with i-th column of beta + cons[i]
    df_Q['rweight_hat'] = np.exp(np.array([np.dot(X[i], beta[:, i]) for i in range(X.shape[0])]) + df_Q['cons'].values)
    
    #Save Results
    df.append(df_Q)

df = pd.concat(df,  ignore_index=True)
df.rename(columns={'R_squared_beta': 'R_squared_OLS_log_rweight'},inplace=True)# Within each Bin compute the R^2

#------- Loop over all Bins and Compute the R^2 with the fitted values from the previous
df_level_error = []
for quarter in Quarters:
    print(quarter)
    
    df_Q = df[df['rdate'] == quarter]
    
    for i_bin in np.sort(df_Q['bin'].unique()):
        df_Q_bin = df_Q[df_Q['bin'] == i_bin]
        
        y = df_Q_bin['rweight']
        y_hat = df_Q_bin['rweight_hat']
        y_mean = np.mean(df_Q_bin['rweight'])
        
        R2 = 1- np.linalg.norm(y - y_hat)**2 /  np.linalg.norm(y - np.mean(y_mean))**2
        
        df_level_error_bin = pd.DataFrame(columns = ['rdate','bin','R_squared_OLS_Level'])
        df_level_error_bin.at[0,'rdate'] = quarter
        df_level_error_bin['rdate'] = pd.to_datetime(df_level_error_bin['rdate'])
        df_level_error_bin.at[0,'bin'] = i_bin
        df_level_error_bin.at[0,'R_squared_OLS_Level'] = R2
        
        df_level_error.append(df_level_error_bin)
        
df_level_error = pd.concat(df_level_error, ignore_index=True)


#Save Data
df_final = df_level_error.merge(OLS_Estimates[['rdate','bin', 'R_squared']], on = ['rdate','bin'], how = 'inner')

df_final.rename(columns={'R_squared': 'R_squared_OLS_Logs'},inplace=True)

df_final.to_csv(path + "/Output/Fit" + "/OLS_R2LevelFits_vs_R2LogFits.csv", index=False)

#%% Compute R^2 for GMM Estimates (this includes zero Holdings in the data)

#DataFrame to Store Results
df = []

#------- Loop over all Quarters and compute the fitted value rweight_hat in each bin
Quarters = Holdings['rdate'].unique()
Quarters = Quarters[Quarters.year<2023]
for quarter in Quarters:
    print(quarter)

    ### --- Slice Datasets
    #Holdings Sliced
    Holdings_Q = Holdings[Holdings['rdate'] == quarter][['rdate','bin','mgrno','permno','aum','rweight','cons']]

    #Baseline Stock Characteristics Sliced
    StocksQ_Q = StocksQ[StocksQ['date'] == quarter]

    ### --- Merge Stock Characteristics to Holdings Data to build X & Z Matrix for GMM
    #Merge Baseline Stock Characteristics
    df_Q = Holdings_Q.merge(StocksQ_Q[["permno", "date", "LNme", 'LNbe', 'profit', 'Gat', 'divA_be','beta']], 
                            left_on=["rdate", "permno"], right_on=["date", "permno"])

    #Drop any remaining Missing Values to avoid errors
    df_Q = df_Q.dropna(subset=Baseline_endog_Variables_Names + ['rweight'])

    df_Q = df_Q.merge(GMM_Estimates.add_suffix("_beta"), 
                      left_on = ['rdate','bin'], 
                      right_on = ['rdate_beta','bin_beta'],
                      how = 'left', 
                      suffixes=('', ''))
    df_Q.drop(['rdate_beta','bin_beta', 'Error_beta', 'date'],axis = 1,inplace=True)

    #Assing the constant to the dataframe
    df_Q = df_Q.assign(constant=1)
    
    #Compute fitted Values
    Baseline_endog_Variables_Names = ['LNme','LNbe', 'profit', 'Gat', 'divA_be','beta']

    Baseline_Regressors = [item + "_beta" for item in Baseline_endog_Variables_Names + ['constant']]
    
    X = df_Q[Baseline_endog_Variables_Names + ['constant']].values
    beta = df_Q[Baseline_Regressors].values.T
    
    #Get predicted values. i-th predicted value is dot product of i-th row of X with i-th column of beta + cons[i]
    df_Q['rweight_hat'] = np.exp(np.array([np.dot(X[i], beta[:, i]) for i in range(X.shape[0])]) + df_Q['cons'].values)
    
    #Save Results
    df.append(df_Q)
    
df = pd.concat(df, ignore_index=True)


#Remove overflow errors
print("There are " + str(np.sum((df['rweight_hat']==np.inf) & (df['rweight_hat']>1e307))) + " overflow Errors (" + str(np.sum(df['rweight_hat']==np.inf)/len(df)*100) + "%)")
df = df[(df['rweight_hat']<np.inf) & (df['rweight_hat']<1e307)]
df['rdate'] =  pd.to_datetime(df["rdate"]) 

#Data compression
for col in df[['rdate','bin','permno','mgrno','aum','rweight','rweight_hat']].select_dtypes(include=['int64', 'float64']).columns:
    if pd.api.types.is_float_dtype(df[col]):  # If float, check if it can be an integer
        if (df[col] == df[col].astype(int)).all():
            df[col] = df[col].astype(np.int32)  # Convert to integer if all values are whole numbers
        else:
            df[col] = pd.to_numeric(df[col], downcast='float')  # Downcast float64 to float32
    elif pd.api.types.is_integer_dtype(df[col]):  # Downcast integers
        df[col] = pd.to_numeric(df[col], downcast='integer')

#Save Results
df[['rdate','bin','permno','mgrno','aum','rweight','rweight_hat']].to_stata(path + "/Output/Fit" + "/Fitted_Values_GMM_Baseline.dta")


#------- Loop over all Bins and compute the R^2 with the fitted values from the previous
df_level_error = []
for quarter in Quarters:
    print(quarter)
    
    df_Q = df[df['rdate'] == quarter]
    
    for i_bin in np.sort(df_Q['bin'].unique()):
        df_Q_bin = df_Q[df_Q['bin'] == i_bin]
        
        y = df_Q_bin['rweight']
        y_hat = df_Q_bin['rweight_hat']
        y_mean = np.mean(df_Q_bin['rweight'])
        
        R2 = 1- np.linalg.norm(y - y_hat)**2 /  np.linalg.norm(y - np.mean(y_mean))**2
        
        df_level_error_bin = pd.DataFrame(columns = ['rdate','bin','R_squared_GMM'])
        df_level_error_bin.at[0,'rdate'] = quarter
        df_level_error_bin['rdate'] = pd.to_datetime(df_level_error_bin['rdate'])
        df_level_error_bin.at[0,'bin'] = i_bin
        df_level_error_bin.at[0,'R_squared_GMM'] = R2
        
        df_level_error.append(df_level_error_bin)

df_level_error = pd.concat(df_level_error, ignore_index=True)

#Save Data
df_level_error.to_csv(path + "/Output/Fit" + "/GMM_R2_Fit.csv", index=False)

#%% Compute R^2 for NLLS_CV

#DataFrame to Store Results
df = []

#------- Loop over all Quarters and compute the fitted value rweight_hat
Quarters = Holdings['rdate'].unique()
Quarters = Quarters[Quarters.year<2023]
for quarter in Quarters:
    print(quarter)

    ### --- Slice Datasets
    #Holdings Sliced
    Holdings_Q = Holdings[Holdings['rdate'] == quarter][['rdate','bin','mgrno','permno','aum','rweight','cons']]

    #Baseline Stock Characteristics Sliced
    StocksQ_Q = StocksQ[StocksQ['date'] == quarter]

    ### --- Merge Stock Characteristics to Holdings Data to build X & Z Matrix for GMM
    #Merge Baseline Stock Characteristics
    df_Q = Holdings_Q.merge(StocksQ_Q[["permno", "date", "LNme", 'LNbe', 'profit', 'Gat', 'divA_be','beta']], 
                            left_on=["rdate", "permno"], right_on=["date", "permno"])

    #Drop any remaining Missing Values to avoid errors
    df_Q = df_Q.dropna(subset=Baseline_endog_Variables_Names + ['rweight'])

    df_Q = df_Q.merge(NLLS_Estimates.add_suffix("_beta"), 
                      left_on = ['rdate','bin'], 
                      right_on = ['rdate_beta','bin_beta'],
                      how = 'left', 
                      suffixes=('', ''))
    df_Q.drop(['rdate_beta','bin_beta', 'date', 'best_index_beta', 'lam_beta', 'lam_range_beta'],axis = 1,inplace=True)

    #Assing the constant to the dataframe
    df_Q = df_Q.assign(constant=1)
    
    #Compute fitted Values
    Baseline_endog_Variables_Names = ['LNme','LNbe', 'profit', 'Gat', 'divA_be','beta']

    Baseline_Regressors = [item + "_beta" for item in Baseline_endog_Variables_Names + ['constant']]
    
    X = df_Q[Baseline_endog_Variables_Names + ['constant']].values
    beta = df_Q[Baseline_Regressors].values.T
    
    #Get predicted values. i-th predicted value is dot product of i-th row of X with i-th column of beta + cons[i]
    df_Q['rweight_hat'] = np.exp(np.array([np.dot(X[i], beta[:, i]) for i in range(X.shape[0])]) + df_Q['cons'].values)
    
    #Save Results
    df.append(df_Q)
    
df = pd.concat(df, ignore_index=True)

#Remove overflow errors
print("There are " + str(np.sum(df['rweight_hat']==np.inf)) + " overflow Errors (" + str(np.sum(df['rweight_hat']==np.inf)/len(df)*100) + "%)")
df = df[df['rweight_hat']<np.inf]
df['rdate'] =  pd.to_datetime(df["rdate"]) 

#Data compression
for col in df[['rdate','bin','permno','mgrno','aum','rweight','rweight_hat']].select_dtypes(include=['int64', 'float64']).columns:
    if pd.api.types.is_float_dtype(df[col]):  # If float, check if it can be an integer
        if (df[col] == df[col].astype(int)).all():
            df[col] = df[col].astype(np.int32)  # Convert to integer if all values are whole numbers
        else:
            df[col] = pd.to_numeric(df[col], downcast='float')  # Downcast float64 to float32
    elif pd.api.types.is_integer_dtype(df[col]):  # Downcast integers
        df[col] = pd.to_numeric(df[col], downcast='integer')

df[['rdate','bin','permno','mgrno','aum','rweight','rweight_hat']].to_stata(path + "/Output/Fit" + "/Fitted_Values_NLLS_CV.dta")


#------- Loop over all Bins and Compute the R^2 with the fitted values from the previous
df_level_error = []
for quarter in Quarters:
    print(quarter)
    
    df_Q = df[df['rdate'] == quarter]
    
    for i_bin in np.sort(df_Q['bin'].unique()):
        df_Q_bin = df_Q[df_Q['bin'] == i_bin]
        
        y = df_Q_bin['rweight']
        y_hat = df_Q_bin['rweight_hat']
        y_mean = np.mean(df_Q_bin['rweight'])
        
        R2 = 1- np.linalg.norm(y - y_hat)**2 /  np.linalg.norm(y - np.mean(y_mean))**2
        
        df_level_error_bin = pd.DataFrame(columns = ['rdate','bin','R_squared_NLLS'])
        df_level_error_bin.at[0,'rdate'] = quarter
        df_level_error_bin['rdate'] = pd.to_datetime(df_level_error_bin['rdate'])
        df_level_error_bin.at[0,'bin'] = i_bin
        df_level_error_bin.at[0,'R_squared_NLLS'] = R2
        
        df_level_error.append(df_level_error_bin)
        
df_level_error = pd.concat(df_level_error, ignore_index=True)

#Save Data
df_level_error.to_csv(path + "/Output/Fit" + "/NLLS_CV_R2.csv", index=False)