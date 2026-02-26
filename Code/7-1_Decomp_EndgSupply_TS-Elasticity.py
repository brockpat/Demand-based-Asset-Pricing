# -*- coding: utf-8 -*-
"""
Computes Variance Decomposition with Endogenous Supply
"""
#%% Libraries
import pandas as pd
import numpy as np
import scipy
import copy
#%% Functions

#Pull out year & Quarter of DataFrame
def extractTime(df,year,quarter):
    df_year = df[df['rdate'].dt.year == year] 
    df_yearQuarter = df_year[df_year['rdate'].dt.quarter == quarter]
    
    return df_yearQuarter


#Extract Portfolio Holdings, regression Coefficients & Characteristics from data
def getObjects(HoldingsData, year, quarter):
    """
    Parameters
    ----------
    HoldingsData :  Dataset containing the Portfolio Holdings & Regression Results of Equation (10). See
                    the DocString below this function to get a full description of what HoldingsData must look like.
    year :          Year to extract
    quarter :       Quarter to extract
    --> As the number of Investors and the number of Stocks held changes per quarter, 
    only one quarter at a time can be extracted. Else-wise, all objects have dimension errors
    if they also include a variable time dimension. <--

    Returns
    -------
    PFRweightMat :  IxN matrix containing log(w_i(n)/w_i(0))
    EpsilonMat :    IxN matrix containing log(epsilon_i(n))
    consMat :       IxN matrix containing the mean of log(w_i(n)/w_i(0)) per investor (does not vary over assets)
    betaMat :       Ix7 matrix containing the estimated regression coefficients of (10)
    aum :           Ix1 vector containing the assets under management of each investor.
    p :             Nx1 vector containing the log price of each asset
    s :             Nx1 vector containing the log supply (shares outstanding) of each asset
    x :             7xN matrix containing the characteristics of each asset (see HoldingsData docstring)
    
    This function essentially processes the input datasets to generate 
    matrices and vectors for further analysis or modeling
    
    """
    #----------------------------------------------------------------------#
    #0) Extract the relevant quarter of the data
    #----------------------------------------------------------------------#
    HoldingsData = extractTime(HoldingsData, year, quarter)
    HoldingsData = HoldingsData.drop_duplicates(subset=["rdate","mgrno", "permno"])
    
    #----------------------------------------------------------------------#
    #1.1) Create empty Holdings template to incorporate zero Portfolio holdings.
    #----------------------------------------------------------------------#
    #Dataset HoldingsData does not contain zero portfolio holdings. If this is not amended,
    #then portfolio weight vector w_i has a different dimension for each investor and looping
    #through w_i would correspond to different assets
    
    #Unique Stocks in Holdings (length = N)
    #If not sorted, then iterating over n might not refer to same manager or asset for different objects
    uniquePermno = pd.DataFrame(np.sort(HoldingsData['permno'].unique()),columns = ['permno'])
    
    #Unique Managers in Holdings (length = I)
    #If not sorted, then iterating over i ight not refer to same manager or asset for different objects
    uniqueMgrno = pd.DataFrame(np.sort(HoldingsData['mgrno'].unique()), columns = ['mgrno'])
    
    #Unique rdate in Holdings (in Case tensor with time is desired)
    #uniquerdate= pd.DataFrame(HoldingsData['rdate'].unique(), columns = ['rdate'])
    
    #Merge Cross Products to create empty Holdings template for incorporating zero Portfolio holdings
    Holdings  = pd.merge(uniqueMgrno, uniquePermno, how='cross')
    #Holdings  = pd.merge(Holdings,uniquerdate, how = 'cross') // If tensor with time desired
    
    #Create rdate
    Holdings['rdate'] = HoldingsData['rdate'][0]
    
    #Add Holdings variable (log of relative weights) and set it to -infinity (so level is zero)
    Holdings['LNrweight'] = -np.Inf
    
    #Add Error term and set it to -infinity (so level is zero)
    Holdings['unpref'] = -np.Inf
    
    #Add constant term and set it to 0
    Holdings['cons'] = 0
    
    #----------------------------------------------------------------------#
    # 1.2) Overwrite non-zero Portfolio Holdings to Holdings template
    #----------------------------------------------------------------------#
    #Merge non-zero Portfolio weights from data
    Holdings = Holdings.merge(HoldingsData[['mgrno', 'permno', 'rdate', 'LNrweight', 'unpref', 'cons']], how = 'outer', on = ['mgrno', 'permno', 'rdate'])
    Holdings['unpref_y'] = np.log(Holdings['unpref_y'])
    Holdings['LNrweight_y'] = Holdings['LNrweight_y'].fillna(Holdings['LNrweight_x'])
    Holdings = Holdings.drop(['LNrweight_x'], axis = 1)
    Holdings = Holdings.rename(columns={'LNrweight_y': 'LNrweight'})
    
    #Merge error from regression equation (10)
    Holdings['unpref_y'] = Holdings['unpref_y'].fillna(Holdings['unpref_x'])
    Holdings = Holdings.drop(['unpref_x'], axis = 1)
    Holdings = Holdings.rename(columns={'unpref_y': 'unpref'})
    
    #Merge cons from regression equation (10)
    Holdings = Holdings.drop(['cons_x'], axis = 1)
    Holdings = Holdings.rename(columns={'cons_y': 'cons'})
    Holdings['cons'] = Holdings['cons'].fillna(0)
    
    #Sanity Check: If portfolio weight > -inf, then error term should always also be >-inf
    if len(Holdings[(Holdings['LNrweight'] > -np.inf) &  (Holdings['unpref'] == -np.inf)]) > 0:
        print("     !!!! WARNING !!!!" + "\n" + "Portfolio Holdings positive, but error term is zero")
        print(" Problematic Managers are: ")
        print(Holdings[(Holdings['LNrweight'] > -np.inf) &  (Holdings['unpref'] == -np.inf)].mgrno.unique())
        #print("Action taken: Deleting the Managers")
        #mgrno_exclusion = Holdings[(Holdings['LNrweight'] > -np.inf) & (Holdings['unpref'] == -np.inf)].mgrno.unique()
        #Holdings = Holdings[~Holdings['mgrno'].isin(mgrno_exclusion)]
    
    if len(Holdings[(Holdings['LNrweight'] == -np.inf) & (Holdings['unpref'] > -np.inf)]) > 0:
        print("     !!!! WARNING !!!!" + "\n" + "Portfolio Holdings zero, but error term is positive")
        print(" Problematic Managers are: ")
        print(Holdings[(Holdings['LNrweight'] == -np.inf) & (Holdings['unpref'] > -np.inf)].mgrno.unique())
        #print("Action taken: Deleting the Managers")
        #mgrno_exclusion = Holdings[(Holdings['LNrweight'] == -np.inf) & (Holdings['unpref'] > -np.inf)].mgrno.unique()
        #Holdings = Holdings[~Holdings['mgrno'].isin(mgrno_exclusion)]

    #Sanity Check: No observations missing
    if len(Holdings) - len(uniqueMgrno) * len(uniquePermno) != 0:
        print("     !!!! WARNING !!!!" + "\n" + "Observations mismatched.")
        #print(" Action taken: None ")
        
    #----------------------------------------------------------------------#
    # 2) Create epsilon_i(n), w_i(n) & cons_i matrices
    #----------------------------------------------------------------------#
    PFRweight  = Holdings.pivot_table(index='mgrno', columns=['permno', 'rdate'], values='LNrweight')
    PFRweightMat = PFRweight.to_numpy()
    
    Epsilon  = Holdings.pivot_table(index='mgrno', columns=['permno', 'rdate'], values='unpref')
    EpsilonMat = Epsilon.to_numpy()
    #For tensor with time: TEpsilon = MEpsilon.reshape(Epsilon.index.nunique(), Epsilon.columns.levels[0].size, Epsilon.columns.levels[1].size)
        
    cons = Holdings.pivot_table(index='mgrno', columns=['permno', 'rdate'], values='cons')
    consMat = cons.to_numpy()
    
    #----------------------------------------------------------------------#
    # 3) Create Regression Coefficients matrix (Ix7)
    #----------------------------------------------------------------------#

    #Create Dataframe that only contains the mgrno and the reg coefficients
    RegCoeffs = HoldingsData[['mgrno'] + GMM_Estimates_cols].drop_duplicates(subset = ['mgrno'])
    RegCoeffs = RegCoeffs.sort_values('mgrno')
    RegCoeffs.set_index('mgrno',inplace=True)
    
    betaMat = RegCoeffs.values
    
    #If Tensor with Time desired:
    #betaPivot = RegCoeffs.pivot_table(index='mgrno', columns='rdate', values = ['b_LNme', 'b_LNbe', 'b_profit', 'b_beta', 'b_Gat', 'b_divA_be', 'b_cons'])
    #betaPivot.to_csv(path + "/" + "betaPivot.csv")
    #betaMat = betaPivot.to_numpy()
    #Tbeta = Mbeta.reshape(betaPivot.index.nunique(), betaPivot.columns.levels[0].size, betaPivot.columns.levels[1].size)

    #----------------------------------------------------------------------#
    # 4.1) Create Assets under Management Vector (Nx1)
    #----------------------------------------------------------------------#
    aum = HoldingsData[['mgrno', 'aum']].drop_duplicates(subset = ['mgrno']).sort_values('mgrno')
    #aum = aum['aum'].to_numpy()
    
    #----------------------------------------------------------------------#
    # 4.2) Create Log price and log supply Vector (Nx1)
    #----------------------------------------------------------------------#
    #psx = extractTime(StocksQ, year, quarter)
    #If not sorted, then iterating over assets n does not refer to same asset
    #psx = psx.sort_values('permno')
    #Only extract prices of stocks that are actually held. Elsewise dimension error imminent
    #psx = psx.loc[psx['permno'].isin(uniquePermno['permno'])][['LNprc', 'LNshrout', 'LNme', 'LNbe', 'profit', 'beta', 'Gat', 'divA_be']]
    
    p = HoldingsData[["permno", 'LNprc']].sort_values(by="permno").drop_duplicates()
    s = HoldingsData[['permno', 'LNshrout']].sort_values(by="permno").drop_duplicates()

    #----------------------------------------------------------------------#
    # 5) Create Matrix of Characteristics (Nx7)
    #----------------------------------------------------------------------#

    x = HoldingsData[["permno"] + Baseline_endog_Variables_Names + ['constant','cons']].drop_duplicates(subset="permno").sort_values(by="permno")
    #x = x.drop(["permno", "cons"], axis=1).T


    #----------------------------------------------------------------------#
    # 6) Create LNcfac  (Nx1)
    #----------------------------------------------------------------------#
    LNcfac = HoldingsData.sort_values(by="permno")[["permno", 'LNcfac']].drop_duplicates().set_index("permno")


    return PFRweight, PFRweightMat, Epsilon, EpsilonMat, cons, \
        consMat, RegCoeffs, betaMat, aum, p, s, x, uniquePermno, LNcfac
        
    """
    Holdings Dataset Description:
    ----------------------------
        Rows: Individual Investor (mgrno --> mgrno == 0 are households) 
                at a certain time period (rdate)
                holding a certain stock (permno)
                with portfolio weight LNrweight (in logs)
                with certain value of assets under management (aum) in million $
        Columns: Characteristics of the stock:
                    LNprc       (Log Stock Price)
                    LNshrout    (Log Shares Outstanding)
                    LNme        (Log Market Equity = LNprc + LNshrout)
                    LNbe        (Log Book Equity)
                    profit
                    LNcfac      (Log Cumulative factor to adjust price)
                    beta        (beta of the Stock --> Covariance with market)
                    Gat         (Log growth Assets)
                    divA_be     (Annual dividend to Book equity)
                    
                Portfolio Regression Coefficients (investor, but not stock specific):
                    b_LNme 
                    b_LNbe 
                    b_profit 
                    b_beta 
                    b_Gat 
                    b_divA_be 
                    b_cons      (constant of regression)
                    unpref      (latent demand --> residual)
                    
                More Investor Specifics:
                    cons (Mean LNrweight used in the regression for an individual specific intercept in the bins)
                    bin --> Some investors had too few stock holdings to estimate 
                            the coefficients of the Portfolio Regression. These
                            investors were binned together based on their AUM.
                            The only important thing is that all these investors
                            have the same regression coefficients    
    """
    
#Supply Function
def calibrateSupply(p, s):
    """
    Parameters
    ----------
    p : Baseline log market price from baseline
    s : Baseline log supply from baseline

    Returns
    -------
    Parameters s0 from function s = s0 + 1*p where s_baseline = s0+p_baseline
    """
    s0 = s-elasticity*p
    
    return s0

#Compute Supply
def LogAssetSupply(p,s0):
    """
    Parameters
    ----------
    p : Nx1 vector consisting of log price of asset n
    s : Nx1 vector consisting of log shares outstanding of asset n

    Returns
    -------
    S : Nx1 vector consisting of Supply of asset n in monetary value
    """
    s = s0+elasticity*p #Elasticity Estimated from Cross Section  
    return s
    
#Implement Characteristics-based demand equation (10) as a function of price
def predictLogWr(p,s0,x, betaMat, EpsilonMat, consMat):
    """
    Computes the estimated characteristics based demand equation (10) depending on the inputs
    for the stock characteristics. 
    
    Parameters
    ----------
    p :                                 1xN vector of log prices
    s :                                 1xN vector of log supply
    x :                                 Pandas Dataframe Stock Characteristics 
    betaMat:                            Ix7 matrix of manager regression coefficients
    EpsilonMat:                         IxN matrix of residuals of regression of (10)
    consMat:                            IxN matrix of constants used in regression of (10)

    Returns
    -------
    w : IxN matrix returning the predicted log relative portfolio weights from 
        characteristics demand based equation (10).
    """
    #Overwrite characteristics
    s = LogAssetSupply(p,s0)
    
    x['LNme'] = p + s
    x = x.drop(["permno", "cons"], axis = 1)
    
    x = x.to_numpy().T
    
    #Predict relative Portfolio weights through characteristics demand based equation (10)
    LNwR = betaMat @ x + EpsilonMat + consMat
    
    #Check If LNwR has extremely high values. 
    """
    If yes, then this indicates that something went wrong in the GMM estimates
    because demand is inexplicably large. The Values of LNwR must be capped
    in this case because otherwise the numerical routine will crash as it will
    predict infinite demand.
    
    Flag = False
    if np.max(LNwR) > 30:
        print("!!!! WARNING !!!! \n LNwR has too high values. Values are capped at 30 to ensure the numerical procedure remains in tact.")
        Flag = True #"LNwR capped"
        LNwR = np.clip(LNwR, None, 30)
    """

    return LNwR#, Flag
"""
                        SANITY CHECK
                        ------------
Below is a sanity check that was used to check that the function predictLogWr() is correct.


#LNwR,_ = predictLogWr(p["LNprc"].to_numpy(),s,x, betaMat, EpsilonMat, consMat)

#Sanity Check: Estimated weights in baseline must be equal to actual weights (since we include the error)
#However, numerical imprecisions possible. Check Supremum Distance and compare values
abs_diff = np.abs(np.exp(PFRweightMat) - np.exp(LNwR))
max_diff_index = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)

#Compare values.
print(np.exp(LNwR[max_diff_index]) - np.exp(PFRweightMat[max_diff_index]))
"""


#Extract individual Portfolio weight w_i(n) 
def portfolioWeights(LNwR):
    """
    Parameters
    ----------
    LNwR : IxN matrix of log relative Portfolio weights from predictLogWr()

    Returns
    -------
    PFweights : IxN matrix of w_i(n) --> Portfolio weight of Investor i (row) of asset n (column)

    """
    #Compute level relative Portfolio weights
    PFweights = np.exp(LNwR)
    
    #Sum the rows of PFweights and format accordingly
    rowSum = np.sum(PFweights, axis=1) #= \sum_n \delta_i(n) in (11) & (12)
    rowSum = rowSum.reshape(-1, 1)
    
    #Compute w_i(n) according to (11)
    PFweights = PFweights/(1+rowSum)
    
    #Compute w_i(0) according to (12)
    PFWeightOutside = 1/(1+rowSum)
    
    #Sanity Check
    SumInsidePFWeights = np.sum(PFweights,axis=1)
    
    Check = SumInsidePFWeights + PFWeightOutside.reshape(-1)
    
    if np.abs(np.mean(Check)-1)>0.01:
        print("       !!!!! WARNING !!!!!" + "\n" + "Portfolio Weights DO NOT sum to 1")
    
    #If LNwR too big, then PFweights will be nan. This causes the code to crash
    nan_indices = np.where(np.isnan(PFweights))
    PFweights = np.nan_to_num(PFweights, nan=0)
    #print("Portfolio weights: nans encountered. LNwR too Big for the following")
    #print("Row indices of NaN values:", nan_indices[0])
    #print("Column indices of NaN values:", nan_indices[1])
    #print("Action taken: Nans set to 0")
        
    return PFweights

#PFweights = portfolioWeights(LNwR)


#Compute Aggregate Demand
def AssetDemand(PFweights, aum):
    """
    Parameters
    ----------
    PFweights : IxN matrix consisting of w_i(n) Portfolio weights.
    aum : Ix1 vector consisting of Assets under management A_i

    Returns
    -------
    D : Nx1 vector consisting of aggregate demand for asset n in MONETARY value
    --> This is not the q demand of stocks which gives the amount of stocks demanded

    """
    aum = aum['aum'].to_numpy()
    D = PFweights.T @ aum
    #If Demand is degenerate due to some update, we need to fix this to avoid divide by 0 errors
    """
    while np.min(D) <1e-323:
        #Set all Values smaller than threshold to some level
          index = np.argmin(D)
          D[index] = 1e-300
          print("Degenerate Demand")
    """
    return D

#Q = AssetDemand(PFweights, aum)



#S = AssetSupply(p["LNprc"].to_numpy(),s)

#Sanity Check: Check before computing anything whether the default price is the actual market clearing price.
#dif = np.abs(Q-S)
#dif = dif/np.exp(x['LNme'].to_numpy())
#print(np.mean(np.abs(dif)))


#Define Market Clearing function g(p)
def MarketClearing(p, s0, x, aum, betaMat, EpsilonMat, consMat):
    """
    Compute the MarketClearing function g

    Parameters
    ----------
    p : log market clearing price (running variable)
    
    Fix arguments
    -------------
    s : log supply of the assets (Nx1)
    aum: Assets under Manamgement (Ix1)
    betaMat : Estimated Regression Coefficients (Ix7) 
    EpsilonMat : Error Terms (IxN)
    consMat : (IxN)
    LNbe, profit, beta, Gat, divA_be: Stock Characteristics

    Returns
    -------
    g(p) = f(p) - p (Nx1) which should be zero for the market to be cleared
    """
    
    #Predict Log Relative Portfolio Weights based on price update
    #!I can shorten this by extracting the other linear terms that don't change when the price changes
    LNwR = predictLogWr(p,s0, x , betaMat, EpsilonMat, consMat)
    
    #Obtain the levels of individual portfolio weights w_i(n). 
    #As MarketClearing() iterates over the price, this is w_i(n) as a function of price
    PFweights = portfolioWeights(LNwR)
    #If nan values encountered

    #Compute monetary demand per asset n
    Demand = AssetDemand(PFweights, aum)
    
    #Compute f(p) [log Demand - log Supply]
    fp = np.log(Demand) - LogAssetSupply(p, s0)
    
    #Compute g(p) which must be zero so that Demand = Supply, i.e. markets clear
    gp = fp - p
    
    #print(np.linalg.norm(gp))
    
    return gp  

#Sanity Check: Check if in the baseline the market is cleared
#dif = MarketClearing(p["LNprc"].to_numpy(), s, x, aum, betaMat, EpsilonMat, consMat)
#print(np.mean(np.abs(dif)))

#Compute the Market Clearing Price for the Counterfactuals

def solve_MarketClearing(p, s0, x, aum, betaMat, EpsilonMat, consMat, max_iterations=1, tolerance=1e-6): #max_iteration = 3, tolerance = 1e-8
    """
    Optimize the market clearing function using the Krylov method.

    Parameters:
    - p: Numpy array of prices that serve as the initial guess for the root finding
    - s, x, aum, betamat, EpsilonMat, consMat: Additional arguments passed to the MarketClearing function.
    - max_iterations: Maximum number of iterations to perform (default is 3).
    - tolerance: The stopping criterion for the optimization (default is 1e-8).

    Returns:
    - root: The result of the root-finding --> Intermediary price
    """
    iteration = 0
    stopping = 10
    x0 = p

    while iteration < max_iterations and stopping > tolerance:
        root = scipy.optimize.root(MarketClearing, x0,
                                    args=(s0, x, aum, betaMat, EpsilonMat, consMat), 
                                    method='Krylov', tol=tolerance,
                                    options={'maxiter':50}) #maxiter = 1_000
        stopping = np.linalg.norm(root.fun)
        iteration += 1
        x0 = root.x
        #Decrease tolerance to get more accuracy in subsequent iterations
        tolerance = tolerance/10
    
    return root

#%% Load Data

path = "C:/Users/pbrock/Desktop/KY19_Extension"

#Load LNcfac from StocksQ. LNcfac is used to correct misspeficiations in annual stock return.
StocksQ = pd.read_stata(path + "/Data/StocksQ.dta")
StocksQ["date"] = StocksQ["date"] - pd.offsets.MonthBegin()+pd.offsets.MonthEnd()
#StocksQ = StocksQ[["date", "permno", "LNcfac"]]

#Define Baseline Characteristic Names
Baseline_endog_Variables_Names = ['LNme','LNbe', 'profit', 'Gat', 'divA_be','beta']

#Get Regression Coefficient Names
GMM_Estimates_cols  = [s + "_beta" for s in Baseline_endog_Variables_Names + ['constant']]

#Define Dictionaries used in the root-finding loop to extract the correct quarter
day_dict = {1:"31", 2:"30", 3:"30", 4:"31"}
month_dict = {1:"03", 2:"06", 3:"09", 4:"12"}

#%% Root Finding

#Define Dataframes to store results
df_results = pd.DataFrame()
output_errors = pd.DataFrame(columns = ['rdate','root1_error','root2_error','root3_error','root4_error','root5_error', 'Flag_LNwR'])


#Read in Holdings Data and do root finding
for q in [2]:
    for year in range(2002, 2022):
        
        print("------------------------------------------- \n" + 
              "------------------------------------------- \n \n" +
              "              YEAR = " + str(year) + "\n \n" + 
              "------------------------------------------- \n" + 
              "------------------------------------------- \n")
        #---------------------------------------------------------------------#
        # 1) Read in Data
        #---------------------------------------------------------------------#
        
        #----Read in time t Holdings Data
        prevHoldingsData = pd.read_stata(path + "/Output/Variance Decomposition Python/" + f"Holdings_Decomp_GMMbaseline_{year}"f"-{month_dict[q]}"f"-{day_dict[q]}.dta")
        #Set rdate to datetime format
        prevHoldingsData['rdate'] =  pd.to_datetime(prevHoldingsData["rdate"])
        #Merge LNcfac to dataset
        prevHoldingsData = prevHoldingsData.merge(StocksQ[['date','permno','LNcfac']], left_on = ["rdate", "permno"],
                                                  right_on = ['date', 'permno'], how = 'left')
        #Drop Assets which no investor holds
        prevHoldingsData = prevHoldingsData[prevHoldingsData.groupby("permno").LNrweight.transform("count")>0]
        #Sort dataframe (extremely important!)
            #If not sorted, then in getObjects() while iterating over managers and stocks the indices
            #might not refer to same manager or stock for the outputs
        
        #Drop Zero Holdings to not distort the investment universe (Demand is zero anyway for these stocks)
        prevHoldingsData = prevHoldingsData.dropna(subset = 'LNrweight')
        
        #Sort dataframe (extremely important!)
            #If not sorted, then in getObjects() while iterating over managers and stocks the indices
            #might not refer to same manager or stock for the outputs
        prevHoldingsData = prevHoldingsData.sort_values(['mgrno','permno'])

        
        #----Read in time t+4 Holdings Data
        HoldingsData = pd.read_stata(path + "/Output/Variance Decomposition Python/" + f"Holdings_Decomp_GMMbaseline_{year + 1}"f"-{month_dict[q]}"f"-{day_dict[q]}.dta")
        #Set rdate to datetime format
        HoldingsData['rdate'] =  pd.to_datetime(HoldingsData["rdate"]) #if reading in csv
        #Merge LNcfac to dataset
        HoldingsData = HoldingsData.merge(StocksQ[['date','permno','LNcfac']], left_on = ["rdate", "permno"],
                                                  right_on = ['date', 'permno'], how = 'left')
        #Drop assets which no investors hold
        HoldingsData = HoldingsData[HoldingsData.groupby("permno").LNrweight.transform("count")>0]
        
        #Drop Zero Holdings to not distort the investment universe (Demand is zero anyway for these stocks)
        HoldingsData = HoldingsData.dropna(subset = 'LNrweight')
        
        #Sort dataframe (extremely important! --> see explanation for prevHoldingsData Holdings Data).
        HoldingsData = HoldingsData.sort_values(['mgrno','permno'])
                
            
        #---------------------------------------------------------------------#
        # 2) Get Objects
        #---------------------------------------------------------------------#
        
        #These Objects are used for the root-finding process
        prev_PFRweight, prev_PFRweightMat, prev_Epsilon, prev_EpsilonMat, prev_cons, prev_consMat, \
            prev_RegCoeffs, prev_betaMat, prev_aum, prev_p, prev_s, prev_x, prev_uniquePermno, prev_LNcfac = getObjects(prevHoldingsData, year, q)
          
        _, _, _, _, _, _, \
            _, _, _, Lead_p, Lead_s, _, \
                _, _ = getObjects(HoldingsData, year + 1, q)
        
        #---------------------------------------------------------------------#
        # 3) Lead supply, characteristics, aum, beta coefficients and compute
        #        the intermediary market clearing price
        #---------------------------------------------------------------------#
        
        #Compute the intermediary price resulting from a ceteris paribus supply change.
        print("------------------------------------------- \n" + 
              "              Leading Supply" + "\n"
              + "-------------------------------------------")
        
        df_elasticity = pd.read_stata(path + "/Output" + "/Variance Decomposition Python" + "/Elasticities_TS.dta").drop(['index','constant'],axis=1)
        
        #----- i) Lead constant of supply function
        #Create s0_prev
        df_s0 = prev_s.merge(prev_p, how = 'outer', on = 'permno')
        df_s0 = df_s0.merge(df_elasticity, how = 'inner', on = 'permno').sort_values(by = 'permno')
        df_s0['s0_prev'] = df_s0['LNshrout'] - df_s0['elasticity']*df_s0['LNprc']
        
        #Create s0_Lead
        df_s0 = df_s0.merge(Lead_s, how = 'outer', on = 'permno', suffixes = ("", "_Lead")).sort_values(by = 'permno')
        df_s0 = df_s0.merge(Lead_p, how = 'outer', on = 'permno', suffixes = ("", "_Lead")).sort_values(by = 'permno')
        df_s0['s0_Lead'] = df_s0['LNshrout_Lead'] - df_s0['elasticity']*df_s0['LNprc_Lead']
        
        #Overwrite s0_Lead with s0_prev if asset in t no longer exists        
        df_s0['s0_Lead'].fillna(df_s0['s0_prev'] ,inplace=True)
        
        #Restrict investment universe to permnos held in t
        s0_Lead = df_s0[df_s0['permno'].isin(prev_uniquePermno['permno'])][['permno', 's0_Lead', 'elasticity']].sort_values(by = 'permno')
       
        elasticity = s0_Lead['elasticity'].to_numpy()
        
        #Solve for the Market Clearing Price
        root1 = solve_MarketClearing(prev_p["LNprc"].to_numpy(), 
                                     s0_Lead['s0_Lead'].to_numpy(), prev_x, prev_aum, prev_betaMat, prev_EpsilonMat, prev_consMat)
        print("Root1 Approximation Error = " + str(np.linalg.norm(root1.fun)))
        
        #Save Results
        LNprc_1 = prev_p.copy()
        LNprc_1.loc[:, 'LNprc'] = root1.x 
        LNprc_1 = LNprc_1.rename(columns = {'LNprc':'LNprc_1'})
        

        #----- ii) Lead Stock Characteristics X
        
        #Compute the intermediary price resulting from a ceteris paribus Stock characteristics change.
        print("------------------------------------------- \n" + 
              "          Leading Characteristics" + "\n"
              + "-------------------------------------------")
        
        #-- Update Characteristics of Stocks that are still held in t+4
        #Create Dataframe with x_t and x_{t+4} for stocks at time t
        x_list = [item for item in prev_x.columns if item not in ['permno', 'constant','cons']]
        x_list_lead = [item + "_Lead" for item in prev_x.columns if item not in ['permno','constant','cons']]
        
        Lead_x = prev_x.merge(HoldingsData[['permno'] + x_list].drop_duplicates(subset = 'permno'), 
                              how='left', 
                               on='permno', suffixes=('', '_Lead'))
        
        #Update x_t with values from t+4 if they exist.
        Lead_x[x_list] = np.where(pd.notnull(Lead_x[x_list_lead]), Lead_x[x_list_lead], Lead_x[x_list])
        #Delete auxiliary column
        Lead_x.drop(x_list_lead,axis = 1, inplace = True)
        
        #Solve for the Market Clearing Price
        root2 = solve_MarketClearing(LNprc_1['LNprc_1'].to_numpy(), 
                                     s0_Lead['s0_Lead'].to_numpy(), Lead_x, prev_aum, prev_betaMat, prev_EpsilonMat, prev_consMat)
        print("Root2 Approximation Error = " + str(np.linalg.norm(root2.fun)))
        
        #Save Results
        LNprc_2 = prev_p.copy()
        LNprc_2.loc[:, 'LNprc'] = root2.x
        LNprc_2 = LNprc_2.rename(columns = {'LNprc':'LNprc_2'})
        
        
        #----- iii) Lead AUM 
       
        #Compute the intermediary price resulting from a ceteris paribus AUM Change.
        print("------------------------------------------- \n" + 
              "              Leading AUM" + "\n"
              + "-------------------------------------------")
        
        #-- Update AUM of Managers who still exist in t+4
        #Create Dataframe with aum_t and aum_{t+4} for stocks at time t
        Lead_aum = prev_aum.merge(HoldingsData[['mgrno','aum']].drop_duplicates(subset = 'mgrno'), how='left', 
                               on='mgrno', suffixes=('', '_Lead'))
        #Update aum_t with values from t+4 if they exist.
        Lead_aum['aum'] = np.where(pd.notnull(Lead_aum['aum_Lead']), Lead_aum['aum_Lead'], Lead_aum['aum'])
        #Delete auxiliary column
        Lead_aum.drop('aum_Lead',axis = 1, inplace = True)
        
        #Solve for the Market Clearing Price
        root3 = solve_MarketClearing(LNprc_2['LNprc_2'].to_numpy(), 
                                     s0_Lead['s0_Lead'].to_numpy(), Lead_x, Lead_aum, prev_betaMat, prev_EpsilonMat, prev_consMat)
        print("Root3 Approximation Error = " + str(np.linalg.norm(root3.fun)))
        
        #Save Results
        LNprc_3 = prev_p.copy()
        LNprc_3.loc[:, 'LNprc'] = root3.x
        LNprc_3 = LNprc_3.rename(columns = {'LNprc':'LNprc_3'})
        
        #----- iv) Lead  betas (coefficients) & 'cons' 
        
        #Compute the intermediary price resulting from a ceteris paribus Beta change
        print("------------------------------------------- \n" + 
              "          Leading Beta & 'cons' " + "\n"
              + "-------------------------------------------")
        
        #Extract the relevant columns
        beta_list = list(prev_RegCoeffs.columns) + ['cons']
        beta_list_Lead = [item + "_Lead" for item in beta_list]
        
        Lead_RegCoeffs = prevHoldingsData[['mgrno','permno'] + beta_list].merge(
            HoldingsData[beta_list + ['mgrno']].drop_duplicates(subset = 'mgrno'),
                                                                      on = 'mgrno', how = 'left',
                                                                      suffixes = ("", "_Lead"))
        
        #Update b_t with values from t+4 if they exist.
        Lead_RegCoeffs[beta_list] = np.where(pd.notnull(Lead_RegCoeffs[beta_list_Lead]), Lead_RegCoeffs[beta_list_Lead], Lead_RegCoeffs[beta_list])
        #Delete auxiliary column
        Lead_RegCoeffs.drop(beta_list_Lead,axis = 1, inplace = True)
        
        #Create betaMat
        Lead_betaMat = Lead_RegCoeffs.drop_duplicates(subset = 'mgrno')[beta_list].drop('cons',axis=1).to_numpy()
        #Create consMat
        Lead_consMat = np.nan_to_num(Lead_RegCoeffs.pivot_table(index='mgrno', columns=['permno'], values='cons').to_numpy(),0)
        
        #Solve for the Market Clearing Price
        root4 = solve_MarketClearing(LNprc_3['LNprc_3'].to_numpy(), 
                                     s0_Lead['s0_Lead'].to_numpy(), Lead_x, Lead_aum, Lead_betaMat, prev_EpsilonMat, Lead_consMat)
        print("Root4 Approximation Error = " + str(np.linalg.norm(root4.fun)))
        
        #Save Results
        LNprc_4 = prev_p.copy()
        LNprc_4.loc[:, 'LNprc'] = root4.x
        LNprc_4 = LNprc_4.rename(columns = {'LNprc':'LNprc_4'})
        
        #---------------------------------------------------------------------#
        # 4) Update Investment universe
        #---------------------------------------------------------------------#
    
        print("------------------------------------------- \n" + 
              "          Extensive Margin" + "\n"
              + "-------------------------------------------")

        #Merge unpref from t to unpref in t+4 if the link exists
        prev_unpref = HoldingsData[['mgrno','permno','unpref']].merge(
            prevHoldingsData[['mgrno','permno','unpref']], how = 'left',
            on = ['mgrno', 'permno'], suffixes=('', '_prev'))
        
        #Downgrade the values in t if downgrade exists
        prev_unpref['unpref'] = np.where(pd.notnull(prev_unpref['unpref_prev']), prev_unpref['unpref_prev'], prev_unpref['unpref'])
        prev_unpref.drop('unpref_prev',axis = 1, inplace = True)
        
        #Merge Downgraded Values to the DataFrame at time t+4
        HoldingsData = HoldingsData.merge(prev_unpref, on = ['mgrno','permno'], how = 'left', 
                                          suffixes = ("","_Lag"))
                
        HoldingsData.drop('unpref',axis = 1, inplace = True)
        HoldingsData = HoldingsData.rename(columns = {'unpref_Lag':'unpref'})

        #Get Objects for the Root Finding
        Lead_PFRweight, Lead_PFRweightMat, prev_Epsilon, prev_EpsilonMat, Lead_cons, Lead_consMat, \
            Lead_RegCoeffs, Lead_betaMat, Lead_aum, Lead_p, Lead_s, Lead_x, \
                Lead_uniquePermno, Lead_LNcfac = getObjects(HoldingsData, year + 1, q)
        
        #Compute s0 for the t+4 investment universe
        s0_Lead = Lead_p.merge(Lead_s, on = 'permno')
        s0_Lead = s0_Lead.merge(df_elasticity, on = 'permno', how = 'inner').sort_values(by='permno')

        s0_Lead['s0_Lead'] = s0_Lead['LNshrout'] - s0_Lead['elasticity']*s0_Lead['LNprc']
        
        elasticity = s0_Lead['elasticity'].to_numpy()
        
        
        #Compute the Market Clearing Price (notice: Price has different dimension as t+4 data has different pernos than t data)
        root5 = solve_MarketClearing(Lead_p["LNprc"].to_numpy(),
                                     s0_Lead['s0_Lead'].to_numpy(), Lead_x, Lead_aum, Lead_betaMat, prev_EpsilonMat, Lead_consMat)
        print("Root5 Approximation Error = " + str(np.linalg.norm(root5.fun)))
        
        #Save Results
        LNprc_5 = Lead_p.copy()
        LNprc_5.loc[:, 'LNprc'] = root5.x
        LNprc_5 = LNprc_5.rename(columns = {'LNprc':'LNprc_5'})
        
        #---------------------------------------------------------------------#
        # 5)  Compute & Save Final Returns
        #---------------------------------------------------------------------#
        
        #------------------ Price & Return Data -------------------------------
        
        #Save LNprc1 to LNprc6 --> LNprc6 is the actual observed price at t+4
        df_LNprc = prev_p.rename(columns = {'LNprc':'LNprc_prev'}).merge(
            LNprc_1, on = "permno", how = 'outer', suffixes = ("", ""))
        df_LNprc = df_LNprc.merge(LNprc_2, on = "permno", how = 'outer', suffixes = ("", ""))
        df_LNprc = df_LNprc.merge(LNprc_3, on = "permno", how = 'outer', suffixes = ("", ""))
        df_LNprc = df_LNprc.merge(LNprc_4, on = "permno", how = 'outer', suffixes = ("", ""))
        df_LNprc = df_LNprc.merge(LNprc_5, on = "permno", how = 'outer', suffixes = ("", ""))
        df_LNprc = df_LNprc.merge(
            HoldingsData.rename(columns = {'LNprc':'LNprc_6'})[['permno','LNprc_6']].drop_duplicates('permno'),
                                  how = 'outer', suffixes = ("","")).sort_values('permno')

        #Merge LNcfac that corrects returns
        df_LNprc = df_LNprc.merge(StocksQ[StocksQ['date'] == str(year) + "-" + month_dict[q] + "-" + day_dict[q]][['permno','LNcfac']],
                                  on = 'permno', how = 'inner')
        df_LNprc = df_LNprc.rename(columns = {'LNcfac': 'LNcfac_prev'})
        
        df_LNprc = df_LNprc.merge(StocksQ[StocksQ['date'] == str(year+1) + "-" + month_dict[q] + "-" + day_dict[q]][['permno','LNcfac']],
                                  on = 'permno', how = 'inner')
        df_LNprc = df_LNprc.rename(columns = {'LNcfac': 'LNcfac_Lead'})
        
        #Compute The intermediary returns central to the Variance Decomposition
        df_LNprc['LNret1'] 	= df_LNprc['LNprc_1']+df_LNprc['LNcfac_Lead']-(df_LNprc['LNprc_prev']+ df_LNprc['LNcfac_prev'])
        for i in range(2, 7):
            df_LNprc[f'LNret{i}'] = df_LNprc[f'LNprc_{i}'] - df_LNprc[f'LNprc_{i-1}']

        #Keep the intersection of permnos, i.e. permnos existing in both t and t+4
        df_LNprc = df_LNprc[df_LNprc['permno'].isin(Lead_uniquePermno['permno'])]
        df_LNprc = df_LNprc.dropna(subset=['LNprc_prev'])
        
        #Add date column
        df_LNprc['rdate'] = HoldingsData['rdate'][0]
        
        #Merge observed annual and dividend return
        df_LNprc = df_LNprc.merge(StocksQ[['date','permno','LNretA','LNretdA']], left_on = ['rdate','permno'],
                                  right_on = ['date','permno'], suffixes = ("","")).drop("date",axis=1)
        
        #Append results
        df_results = pd.concat([df_results,df_LNprc])
    
        #-------------------- Save the numerical errors -----------------------
        df_error= pd.DataFrame(columns = ['rdate', 'root1_error',
                                          'root2_error','root3_error',
                                          'root4_error','root5_error'])

        df_error.loc[0] = [
            HoldingsData["rdate"][0],
            np.linalg.norm(root1.fun),
            np.linalg.norm(root2.fun),
            np.linalg.norm(root3.fun),
            np.linalg.norm(root4.fun),
            np.linalg.norm(root5.fun)#, 
                ]
        output_errors = pd.concat([output_errors,df_error])
        
        #-------------------------   Export Results ---------------------------
        df_results.to_stata(path + "/Output/Variance Decomposition Python/GMMbaseline_EndgSupply_TSelasticity.dta")
        output_errors.to_csv(path + "/Output/Variance Decomposition Python/GMMbaseline_EndgSupply_TSelasticity.csv")