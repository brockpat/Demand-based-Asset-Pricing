# -*- coding: utf-8 -*-
"""
Compares the Fit for Households Out of Sample to assess whether Policy Experiments are good for the household

WAS NOT USED IN THE PAPER
"""

#%% Libraries
import pandas as pd
import scipy
import numpy as np
import matplotlib.pyplot  as plt
import statsmodels.api as sm
import scipy.optimize
from scipy.stats.mstats import winsorize

from sklearn.model_selection import train_test_split

from linearmodels.iv import IV2SLS

import time

import copy

#The following packages can be used to use Automatic Differentiation to compute the Jacobian of gmmObjective().
from autograd import jacobian
from functools import partial
import autograd.numpy as anp  # Note the alias to avoid conflict with standard numpy

#%% Functions
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
    return y* anp.exp( -X[:,:-1] @ RegCoeffs*step_size - X[:,X.shape[1]-1] ) - 1

def G_Matrix(Z,epsilon):
    """
    Inputs
    ----------
    Z : numpy matrix of instruments which is identical to X apart from two key aspects.
    Firstly, Z includes the column IVme (but not LNme), else-wise Z wouldn't be the matrix
    of instruments. Secondly, Z does NOT contain the column 'cons'. This is because 'cons'
    has no Regression Coefficient attached to it and therefore has no moment to satisfy.

    epsilon : prediction error (see function Epsilon() ).

    Returns
    -------
    The transpose of the numpy matrix where each (i,j)-element of Z is multiplied by epsilion(i) .
    In this final matrix called G, each i-th row and j-th column contains the product of the i-th
    variable of the j-th observation in the bin with the corresponding error. Summing up the columns
    will therefore be the sample moment approximating the Expectation.
    """
    return (Z * epsilon[:,np.newaxis]).T


def gmmObjective_Vector(RegCoeffs,X,Z,y):
    """
    Only returns g_avg from gmmObjective(). Doing root-finding on g_avg minimises
    the objective function as well. However, root-finding mostly does not work.
    """
    #Compute error vector
    epsilon = Epsilon(RegCoeffs,X,y)

    #Compute matrix of g_i. Each i-th column is g_i
    G = G_Matrix(Z,epsilon)

    #Estimate the expcation E[z_i *epsilon] where z_i is the i-th variable with the average
    g_avg = G.mean(axis=1)
    
    #Return g_avg and the Jacobian
    return g_avg , jacobian_gmmObjective_Vector(RegCoeffs,X,Z,y,epsilon),G

def jacobian_gmmObjective_Vector(RegCoeffs,X,Z,y,epsilon):
    #!!!! Make sure 'cons' is always the last element in selected characteristics
    jac = -1/len(Z)*Z.T @ ( X[:,:-1]* (epsilon[:, np.newaxis]+1) )
    return jac

def Newton(RegCoeffs,X,Z,y,damping = 0.0):
    try:
        # Function Value and Jacobian at Initial iteration
        g_avg , jac,_ = gmmObjective_Vector(RegCoeffs,X,Z,y)
        
        # Update Value
        beta_new = damping*RegCoeffs + (1-damping)* (-np.linalg.inv(jac)@g_avg + RegCoeffs)
        
        return beta_new
    except:
        return RegCoeffs.abs()*(-np.inf)

def get_GMM_Variables(df_Q_bin, selected_characteristics,
                            selected_instruments):
    #!!!! Make sure 'cons' is always the last element in selected characteristics
    """
    Inputs
    ----------
    df_Q_bin : Sliced Pandas Dataframe. Contains a quarter and bin slice of the overall data.
    selected_characteristics : Output of get_Bin_Characteristics_List()
    selected_instruments : Lists of Instruments based on selected_characteristics. Generated in main file.

    Returns
    -------
    X : numpy matrix of explanatory variables. Market Equity 1st column if selected by LASSO.
    Z : numpy matrix of explanatory variables. Instrument for Market Equity 1st column if selected by LASSO.
    y : numpy vector of the relative portfolio weights
    W : Weighting Matrix (redundant, see Weighting_Matrix() )
    """
    X,Z,y = df_Q_bin[selected_characteristics], df_Q_bin[selected_instruments], df_Q_bin["rweight"]

    #Convert matrices to numpy
    X = X.to_numpy()
    Z = Z.to_numpy()
    y = y.to_numpy()

    #Output Identity weighting matrix
    W = np.eye(Z.shape[1])

    return X,Z,y,W

def gmm_initial_guess(df_Q_bin,selected_characteristics,
                            selected_instruments):
    try:
        #Filter out the Zeros for linear Regression
        df_Q_bin = df_Q_bin[df_Q_bin.rweight>0]
    
        #Get X and Z matrix
        X_reg = df_Q_bin[[var for var in [selected_characteristics + ['IVme']][0] if var != 'cons']]
        
        #Get vector of independent variable (subtract cons so that constant has the right level)
        y_reg = np.log(df_Q_bin.rweight)
        y_reg = y_reg - df_Q_bin['cons']
            
        #Do 2SLS Regression
        beta_lin_IV = IV2SLS(dependent=y_reg, exog=
                             X_reg.drop(columns=['LNme', 'IVme'],inplace=False), 
                             endog=X_reg['LNme'], instruments=X_reg['IVme']).fit().params
        
        #Maintain order of selected_characteristics 
        beta_lin_IV = beta_lin_IV.reindex([var for var in selected_characteristics if var != 'cons'])
    
        return beta_lin_IV   
    
    except:
        return pd.Series(np.zeros(len(selected_instruments)), ['LNme' if var == 'IVme' else var for var in selected_instruments])

#%% Read in Data

#Greene Theorem THEOREM 13.2 and Newey & McFadden (1994) Theorem 3.4, Wikipedia: https://en.wikipedia.org/wiki/Generalized_method_of_moments

path = "C:/Users/pbrock/Desktop/KY19_Extension"

#Load Holdings Data
Holdings = pd.read_stata(path + "/Data" + "/Data1_clean_correct_bins.dta")
Holdings['rdate'] =  pd.to_datetime(Holdings["rdate"]) #if reading in csv
Holdings = Holdings[Holdings['bin'] == 90457]
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

df_HH_Estimates = pd.DataFrame(columns = ['rdate','bin'] + Baseline_endog_Variables_Names + ['constant'])


## Message: Only household can be fitted, the remaining investors have a terrible fit.
#This makes KY19 inadequate for policy relevant work.

#%% GMM Estimation 

df = Holdings.merge(StocksQ[["permno", "date", "LNme", 'LNbe', 'profit', 'Gat', 'divA_be','beta']], 
                        left_on=["rdate", "permno"], right_on=["date", "permno"])

df = df.dropna(subset= Baseline_endog_Variables_Names + ['IVme'] + ['rweight'])

df = df.assign(constant=1)

#Extract unique dates
Quarters = Holdings['rdate'].unique()

for quarter in Quarters:
    print(quarter)
    
    df_Q = df[df['rdate'] == quarter]
    
    df_HH_Estimates.at[0,'rdate'] = quarter
    df_HH_Estimates.at[0,'bin'] = 0
    df_HH_Estimates.set_index(['rdate', 'bin'],inplace=True)
    df_HH_Estimates[['Error'] + Baseline_endog_Variables_Names + ['constant']] = 0
    
    #Get Data
    df_X = df_Q[Baseline_endog_Variables_Names + ['IVme','constant','cons','rweight']]
    y = df_Q['rweight']
    
    df_X = df_X[df_X['rweight']>0]
    y = y[y > 0]
    
    #Get Training and Test Dataset
    df_X_train, df_X_test, y_train, y_test = train_test_split(
    df_X, y, test_size=0.3, random_state=42)
    
    #Convert Training Data to numpy
    X_train = df_X_train[Baseline_endog_Variables_Names + ['constant','cons']]#.values
    Z_train = df_X_train[Baseline_exog_Variables_Names + ['constant']]#.values
    y_train = y_train#.values
    
    #Convert Test Data to numpy
    X_test = df_X_test[Baseline_endog_Variables_Names + ['constant','cons']].values
    Z_test = df_X_test[Baseline_exog_Variables_Names + ['constant']].values
    y_test = y_test.values
    
    
    beta_initial = pd.Series(np.zeros(Z_train.shape[1]), index = Baseline_endog_Variables_Names + ['constant'])
    
    step_size = 1
    iteration = 0
    error = 1
    beta = copy.deepcopy(beta_initial)
    while iteration <100 and error > 1e-14:
        beta = Newton(beta,X_train,Z_train,y_train,damping = 0)
        g_avg, _ , _ = gmmObjective_Vector(beta,X_train,Z_train,y_train)
        iteration = iteration +1
        error = np.linalg.norm(g_avg)
        
    check = np.log(y_test) - (X_test[:,:-1] @ beta*step_size + X_test[:,X_test.shape[1]-1])
        
    fit = np.exp( X_test[:,:-1] @ beta*step_size + X_test[:,X_test.shape[1]-1])
    
    prediction_error = 1- np.linalg.norm(y_test - np.exp( X_test[:,:-1] @ beta*step_size + X_test[:,X_test.shape[1]-1]))**2 /  np.linalg.norm(y_test - np.mean(y_test))**2
    
    p_diff = np.mean(np.abs((y_test-fit)/fit))
    
    prediction_error_inSample =  1- np.linalg.norm(y_train - np.exp( X_train[:,:-1] @ beta*step_size + X_train[:,X_train.shape[1]-1]))**2 /  np.linalg.norm(y_train - np.mean(y_train))**2
    
    
    
    print("     Prediction Error = " + str(prediction_error))
    print("     Percentage Difference = " + str(p_diff))
    print("     Prediction Error InSample = " + str(prediction_error_inSample))
        
    
    
    