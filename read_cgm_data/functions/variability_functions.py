import numpy as np
import pandas as pd
import streamlit as st
from datetime import timedelta
from scipy.stats import entropy
from scipy.integrate import trapezoid
from ..utils.difference import difference, difference_m





def glucose_mean(x,**kwargs):
    return x['glucose'].mean()

def glucose_median(x,**kwargs):
    return x['glucose'].median()

def glucose_std(x,**kwargs):    
    return x['glucose'].std()

def glucose_cv(x,**kwargs):
    return x['glucose'].std()/x['glucose'].mean()

def glucose_N(x,**kwargs):
    return len(x)

def total_time(x,**kwargs):
    dt = timedelta(minutes = kwargs['deltat'])
    return dt*len(x)

def total_days(x,**kwargs):
    """
    total_days - total number of 24 hour periods in a set of data
    """
    ans = total_time(x,**kwargs)
    return ans.days+ans.seconds/(24*3600)

def total_hours(x,**kwargs):
    ans = total_time(x,**kwargs)
    return ans.days*24+ans.seconds/3600

def auc(x,**kwargs):
    """
    Area under the curve in unit*minute or unit * hour
    """
    #auc_units = 1 #puts the measure in mg/dL * minute
    auc_units = 60 #puts the measure in mg/dL *hour

    if 'deltat' in kwargs:
        deltat = kwargs['deltat']
    else:
        deltat = 5
    return trapezoid(x['glucose'])*deltat/auc_units

def auc_per_day(x,**kwargs):
    """
    auc_per_day - calculated per 24 hour period
        total auc / total number of 24 hour periods in data
    """
    return auc(x,**kwargs)/total_days(x,**kwargs)

def auc_thresh(x, **kwargs):
    """
    auc calculated if both x1,x2 are above threshold, otherwise
        sums 0 for that interval (although we could do better).
    """

    #auc_units = 1 #puts the measure in mg/dL * minute
    auc_units = 60 #puts the measure in mg/dL * hour

    if 'deltat' in kwargs:
        deltat = kwargs['deltat']
    else:
        deltat = 5
    if 'thresh' in kwargs:
        thresh = kwargs['thresh']
    else:
        thresh = 100
    arr = x['glucose'].values
    arr = arr-thresh
    ans=((arr[:-1]+arr[1:])[(arr[:-1]>=0)&(arr[1:]>=0)]/2).sum()*deltat/auc_units
    return ans

def percent_active(x,**kwargs):
    return kwargs['percent_active']

def time_in_range(x,**kwargs):
    """
    time in range - assumes equal intervals for all data and simply
        counts the number between lower and upper / total number of 
        observations.
    Input:  data - needs to be a series object with either datetime or 
        minutes after midnight as the index.
            lower - the lower bound to check
            upper - the upper bound to check
    Output: a tuple of floats that represent the 
        (%time below range, %time in range, %time above range)
        
    """
    units = kwargs['units']
    if units == 'mg':
        bins = np.array([0,54,70,180,250,350])
    elif units == 'mmol':
        bins = np.array([0,54,70,180,250,350])/18.018

    if 'bins' in kwargs.keys():
        bins = kwargs['bins']
        
    ans = np.histogram(x['glucose'],bins=bins)
    
    return np.round(ans[0]/len(x),5)

def conga(x,**kwargs):
    """
    conga - continuous overall net glycemic action (CONGA) McDonnell paper 
            and updated by Olawsky paper

    Input:  type = ['paper','easy']
            conga_h = integer, hours between observations in the calculation
            time_delta = amount of time between obs in the data

    """
    data = x['glucose']
    if 'type' in kwargs:
        type_ = kwargs['type']
    else:
        type_ = 'paper'
    if 'conga_h' in kwargs:
        h = kwargs['conga_h']
    else:
        h = 1
    if 'deltat' in kwargs:
        deltat = kwargs['deltat']
    else:
        deltat = 5
    
    sample_rate = deltat      
    samples_per_hour = 60//sample_rate
    line1 = data.shift(-samples_per_hour*h).dropna()
    delta = difference(data.dropna(),h)
    if type_ == 'paper':
        congah = delta.std()
        return congah
    if type_ == 'easy':
        k = len(delta)
        d_star = (abs(delta)).sum()/k
        ## make sure that this is what we want to do - talk to Olawsky if possible.
        congah = np.sqrt(((line1-d_star)**2).sum()/(k-1))
        return congah
    return congah

def lability_index(x,**kwargs):
    """
    lability_index - for glucose measurement at time Xt, Dt = difference
        of glucose measurement k minutes prior.
    Input:  data - pandas Series with index as a datetime, values are glucose 
                    readings associated with those times.
            k - length of time in minutes (5 min increments) used to find patterns
    Output: LI as a float.
    """
    if 'li_k' in kwargs:
        k = kwargs['li_k']
    else:
        k = 60
    data = x['glucose']
    Dt = difference_m(data,k)
    try: #if there are too few data values for the data given
        li = (Dt**2).sum()/(len(Dt))
    except:
        li = np.nan
    return li

def mean_absolute_glucose(x,**kwargs):
    """
    mean_absolute_glucose - Hermanides (2009) paper
    Input: data - pandas Series with index as a datetime, values are glucose 
                    readings associated with those times.
    Output: MAG as a float.
    """
    data = x['glucose']

    tt = total_time(x,**kwargs)
    total_hours = tt.days*24+tt.seconds/3600
    data = data[~data.isnull().values].values
    diff = np.abs(data[1:]-data[:-1])
    return diff.sum()/(total_hours)

def glycemic_variability_percentage(x,**kwargs):
    """
    glycemic_variability_percentage - Peyser paper length of curve / length
                    straight line with no movement (time[final]-time[initial])
    Input: data - pandas Series with index as a datetime, values are glucose 
                    readings associated with those times.
    Output: GVP as a float percentage.
    """
    data = x['glucose'].copy()
    time = data.index
    units = kwargs['units']
    if units == 'mmol':
        data = data*18.018
    deltat = kwargs['deltat']
    
    data = data.values
    t2 = np.array([((time[i+1]-time[i]).total_seconds()/60)**2 for i in range(len(time)-1)])
    y2 = (data[1:]-data[:-1])**2
    y2 = y2[t2<=(deltat**2+1)]
    t2 = t2[t2<=(deltat**2+1)]
    
    L0 = np.sqrt(t2).sum()
    L = (np.sqrt(np.array(t2+y2))).sum()
    return (L-L0)/L0 * 100

def j_index(x,**kwargs):
    """
    j_index - calculates J-index 

    Input: data - pandas Series with index as a datetime, values are glucose 
                    readings associated with those times.
    Output: J-index as a float.
    """
    units = kwargs['units']
    data = x['glucose'].copy()
    if units == 'mg':
        return (data.mean()+data.std())**2/1000
    if units =="mmol":
        return (18.018**2)*(data.mean()+data.std())**2/1000
    
def low_high_blood_glucose_index(x,**kwargs):
    """
    low_high_blood_glucose_index - calculates the blood glucose index 
                with three sets of indices.
    Input: data - pandas Series with index as a datetime, values are glucose 
                    readings associated with those times.
            type_- "paper", "easy", or "update" Default = "paper"
            unit - "mg" or "mmol" Default: "mg"
    """

    type_ = kwargs['type']
    n = len(x)
    c = 1
    if type_ == 'update':
        c = 22.77/10
    if type_ == 'paper':
        c = 10/10
    if type_ == 'easy':
        c = 10/10
    rl = c*x['rl']
    rh = c*x['rh']
    if type_ != 'easy':
        nl = n
        nh = n
    else:
        nl=(rl>0).sum()
        nh=(rh>0).sum()
    if nl>0 and nh>0:
        return np.round(np.array([rl.sum()/nl, rh.sum()/nh]),5)
    else:
        return np.array([np.nan,np.nan])
    

def glycemic_risk_assessment_diabetes_equation(x,**kwargs):
    """
    GRADE - or glycemic risk assessment diabetes equation

    Input: data - pandas Series with index as a datetime, values are glucose 
                    readings associated with those times.
            type_ - "paper" or "easy" Default: "paper"
            unit - "mg" or "mmol" Default: "mg"

    Output: GRADE as 4 numbers ================================= 
            (1) GRADE or mean/median of conversion, 
            (2) GRADE for values < 70.2(mg) or 3.9(mmol), 
            (3) GRADE for values between 70.2(mg) or 3.9(mmol) 
                                    and 140.4(mg) or 7.8(mmol),
            (4) GRADE for values above 140.4(mg) or 7.8(mmol)
    """
    type_ = kwargs['type']
    units = kwargs['units']
    g = x['glucose'].values
    c1,c2 = 3.9,7.8
    if units == 'mg':
        g = g/18.018
    if type_=='paper':
        c = 0.16
    if type_ == 'easy':
        c = 0.15554147
    h_log = lambda x,c: 425*((np.log10(np.log10(x))+c)**2)
    h_min = lambda x: x*(x<50)+50*(x>=50)
    h = lambda x,c: h_min(h_log(x,c))
    h_i = h(g,c)

    # separate glucose values into categories based on value
    gl = g[g<c1]
    gm = g[(c1<g)&(g<c2)]
    gh = g[g>c2]

    # run each group of glucose values through the functions
    hl = h(gl,c)
    hm = h(gm,c)
    hh = h(gh,c)
    h_sum = h_i.sum()
    if type_ == 'easy':
        grade = np.median(h_i)
    if type_ == 'paper':
        grade = h_i.mean()
    ans = np.array([grade,hl.sum()/h_sum,hm.sum()/h_sum,hh.sum()/h_sum])
    return np.round(ans,5)

def mean_amplitude_of_glycemic_excursions(x, **kwargs):
    """
    MAGE (Olawsky 2019)
    mean_amplitude_of_glycemic_excursions - MAGE mean of differences that are
        large compared to daily value.
    """

    if 'date' in x.columns:
        days = x['date'].unique()
    else:
        x['date'] = x.index[0].date()
        days = x['date'].unique()
    if 'deltat' in kwargs:
        deltat = kwargs['deltat']
    else:
        deltat = 5
    E = []
    
    for day in days:
        # g - glucose values for day=day
        g = x[x['date']==day]['glucose']
        # s - standard deviation for glucose on day=day
        s = g.std()
        # D - glucose values differenced (5 minutes)
        D = difference_m(g,deltat)
        # test if abs(d) > standard deviation for the day
        # it is a comparison  between the 
        # std of the daily glucose values and the differences
        #print(f"Std: {s}  == Max diff {D.max()}")
        new_E = list(D[abs(D)>s].values)
        E+=new_E
    ## Use numpy array to sort / find mean of data
    if len(E)>0:
        E = np.array(E)
        if len(E[E>0])>0:
            mage_plus = E[E>0].mean()
        else:
            mage_plus = np.nan
        if len(E[E<0])>0:
            mage_minus = E[E<0].mean()
        else:
            mage_minus = np.nan
        mage = np.array([abs(mage_minus),mage_plus])
        mage = mage[~np.isnan(mage)]
        mage = mage.sum()/len(mage)
    else:
        mage_plus = mage_minus = mage = np.nan
    return mage, mage_minus, mage_plus

def mean_of_daily_differences(x,**kwargs):
    """
    MODD - or mean of daily differences
    Input: data - pandas Series with index as a datetime, values are glucose 
                    readings associated with those times.
            type_ - algorithm to use - either "paper" or "easy" 
    Output: MODD as a float
    """
    type_ = kwargs['type']
    time_delta = kwargs['deltat']

    data = x['glucose']
    
    
    if len(data)>(1440/time_delta):
        delta = difference(data,24)
        if type_ == 'paper':
            return (abs(delta)).sum()/len(delta)
        if type_ == 'easy':
            delta = delta[delta != delta.max()]
            return (abs(delta)).sum()/(len(delta))
    else:
        return np.nan

def adrr(x,**kwargs):
    data = x.copy()
    if 'date' in x.columns:
        days = x['date'].unique()
    else:
        data['date'] = x.index[0].date()
        days = [x.index[0].date()]
    type_ = kwargs['type']
    if type_ == 'paper':
        LR = np.zeros(len(days))
        HR = np.zeros(len(days))
        for i,day in enumerate(days):
            day_data = data[data['date']==day]
            LR[i]=day_data['rl'].max()
            HR[i]=day_data['rh'].max()
    if type_ == 'easy':
        deltat = kwargs['deltat']
        daily = 1440//deltat
        num_days = max(len(data)//daily,1)
        LR = np.zeros(num_days)
        HR = np.zeros(num_days)
        data = data.iloc[:daily*num_days].copy()
        for i in range(num_days):
            day_data = data[i*daily:(i+1)*daily]
            LR[i]=day_data['rl'].max()
            HR[i]=day_data['rh'].max()
    return np.round(np.array([(LR+HR).mean(), LR.mean(), HR.mean()]),5)

def m_value(x,**kwargs):
    """
    m_value - calculates the M-value for a glucose 
                time series. 
    Input: data - pandas Series with index as a datetime,
                    values are glucose 
                    readings associated with those times.
            type_ - calculates either the algorithm 
                    from the "paper" or "easy"
            index - the value used as the index, 
                    default is 120
            unit - "mg" for milligrams per deciliter 
                    or "mmol" for millimoles per
                    liter. Default is "mg".
    Output:
        M-value as a float or None if type_ is not correct.

    """
    type_ = kwargs['type']
    unit = kwargs['units']
    index = kwargs['m_index']
    data = x['glucose'].copy()
    if unit == 'mmol':
        data = 18.018*data
    m_star_abs = np.abs((10*np.log10(data/index))**3)
    w = (data.max()-data.min())/20
    if type_=='paper':
        return m_star_abs.mean()+w
    if type_=='easy':
        return m_star_abs.mean()
    return np.nan

def glucose_management_indicator(x,**kwargs):
    """
    glucose_management_indicator - Bergenstal (2018), formerly 
        referred to as eA1C, or estimated A1C which is a measure 
        converting mean glucose from CGM data to an estimated 
        A1C using data from a population and regression.
        
    Input: data - pandas Series with index as a datetime, 
            values are glucose readings associated with those times.
            unit - "mg" for milligrams per deciliter or "mmol" 
            for milimoles per
                    liter. Default is "mg".
    """
    units = kwargs['units']
    data = x['glucose'].copy()
    if units == 'mmol':
        data = 18.018*data
    return 3.31+0.02392*data.mean()

def interquartile_range(x,**kwargs):
    """
    IQR - inter-quartile range 75th percentile - 25th percentile. 
        Danne (2017) had this calculation in one of the figures. 
    """
    data = x['glucose'].copy()
    units = kwargs["units"]
    if units == 'mmol':
        data = 18.018*data
    q75,q25 = np.percentile(data.values,[75,25])
    return q75-q25

def glycemic_risk_index(x,**kwargs):
    """
    Glycemic Risk Indicator - (Klonoff 2023)
        This risk index is a three number and letter result which represents a composite metric for
        the quality of the glycemia from a CGM. 
        
    Input - time in range vector representing [x1,x2,n,y2,y1] the percent time in each category
            [g<54,54<=g<70,70<=g<180,180<g<250,g>250]
    """
    tir = time_in_range(x,**kwargs)*100
    x1,x2,_,y2,y1 = tir
    f = lambda x1,x2:x1+0.8*x2
    g = lambda y1,y2:y1+0.5*y2
    h = lambda x1,x2,y1,y2: 3*f(x1,x2)+1.6*g(y1,y2)
    x = f(x1,x2)
    y = g(y1,y2)
    gri = h(x1,x2,y1,y2)
    return np.round(np.array([gri,x,y]),5)

def cogi(x,**kwargs):
    tir = time_in_range(x,**kwargs)
    sd = glucose_std(x,**kwargs)
    if kwargs['units']=='mmol':
        sd = sd*18.018
    f1 = lambda x1: 0.5*x1
    f2 = lambda x2:0.35*((-100/15*x2+100)*(0<=x2<15))
    f3 = lambda x3:0.15*(100*(x3<18)+(-10/9*(x3-18)+100)*(18<=x3<108))
    f = lambda x1,x2,x3: f1(x1)+f2(x2)+f3(x3)
    return f(tir[2]*100,(tir[0]+tir[1])*100,sd)

def eccentricity(x,**kwargs):
    X = x['glucose'].copy()
    X_shift=X.shift(-1)
    X_shift.rename('shift',inplace=True)
    X_new = pd.concat([X,X_shift],axis=1).dropna()
    X = X_new.values
    cov = np.cov(X.T)
    eigenvals, _ = np.linalg.eig(cov)
    #theta = np.linspace(0,2*np.pi, 1000)
    #ellipsis = (np.sqrt(eigenvals[None,:])*eigenvecs) @ [np.sin(theta),np.cos(theta)]
    long_axis,short_axis= 2*np.sqrt(eigenvals)
    a = max(long_axis,short_axis)
    b = min(long_axis,short_axis)
    return np.round(np.array([np.sqrt(1-b**2/a**2),a,b]),5)

def transition_matrix(data,intervals,shift_minutes,deltat):
    """
    computes probability transition matrix for a set of data where
        the data is a pandas series with timeseries as index and 
        glucose readings as the values in the series.
    """
    def apply_state(x):
        if x<=intervals[0]:
            return 0
        elif x>intervals[0] and x<=intervals[1]:
            return 1
        elif x>intervals[1] and x<=intervals[2]:
            return 2
        elif x>intervals[2] and x<=intervals[3]:
            return 3
        else:
            return 4
    X = data.copy()
    X.rename('t_(i)',inplace=True)
    X_shift = data.shift(-shift_minutes//deltat)
    X_shift.rename('t_(i+1)',inplace=True)
    X = pd.concat([X,X_shift],axis=1).dropna()
    for col in X.columns:
        X[col]=X[col].map(apply_state)
    P = np.zeros((5,5),dtype=float)
    A = np.zeros((5,5))
    for i in range(len(X)):
        A[X.iloc[i,0],X.iloc[i,1]]+=1
    for i in range(len(A)):
        if A[i,:].sum()==0:
            pass
        else:
            P[i,:]=A[i,:]/A[i,:].sum()
    P_ = P.T - np.eye(5)
    A = np.concatenate((P_,np.array([[1,1,1,1,1]])))
    b = np.array([[0],[0],[0],[0],[0],[1]])
    pi_star = np.linalg.inv(A.T@A)@(A.T@b)
    pi_star = pd.DataFrame(pi_star)
    er = [entropy(P[i,:]) for i in range(len(P))]
    pi_star=pi_star.values.reshape(-1) #flatten pi_star
    er = [float(er[i]*pi_star[i])
            if ~np.isnan(er[i]) else 0 for i in range(len(er))]

    return P,pi_star,er

def entropy_mc(x,**kwargs):
    data = x['glucose'].copy()
    intervals = [54,70,180,250]
    shift_minutes = kwargs['deltat']
    deltat = kwargs['deltat']
    _,_,er = transition_matrix(data,intervals,shift_minutes,deltat)
    return np.array(er).sum()

def tir_stats(x,**kwargs):
    """
    produces a time in range report based on a distribution of
        times in each bin. Specific function for one report. 
        Do not use on anything else.

    Inputs: x = the distribution of time in range based on the given bins
    """
    def convert_time(x):
        hours_ = int(x)
        min_ = (x-hours_)*60
        #sec_ = (min_-int(min_))*60
        return f"{hours_}hrs {round(min_)}mins"# {int(sec_)}secs" 
    units = kwargs['units']
    bins = kwargs['bins']
    #total_time = kwargs['total_time']
    if units=='mmol' and bins[-1]>100:
        #bins=bins/18
        c=2
    hours = {'all':24,'wake':18,'sleep':6}
    ans = pd.DataFrame()
    d = x.copy()
    idxs = x.index
    digit = 5
    c = 0*(units =='mg')+2*(units == 'mmol')
    div = 1*(units =='mg')+18*(units == 'mmol')
    normal = np.array([0.01,0.04,0.04,0.7,0.7,0.7,0.25,0.25,0.25,0.25,0.05])
    direction = ['<','<','<','>','>','>','<','<','<','<','<']
    #normal = normal.reshape(len(normal),1)
    
    for idx in idxs:
        x = d.loc[idx,:].values*100
        hrs = hours[idx] if idx in hours else 24
        #tt = total_time.loc[idx].iloc[0]
        tir = {}
        tir[f"<{bins[1]:0.{c}f}{units}"] = round(x[0],digit)
        tir[f"<{bins[2]:0.{c}f}{units}"] = round(x[:2].sum(),digit)
        tir[f"{bins[1]:0.{c}f}-{bins[2]-1/div:0.{c}f}{units}"] = round(x[1],digit)
        tir[f"{bins[2]:0.{c}f}-{bins[3]:0.{c}f}{units}"] = round(x[2],digit)
        tir[f"{bins[2]:0.{c}f}-{bins[4]:0.{c}f}{units}"] = round(x[2:4].sum(),digit)
        tir[f"{bins[2]:0.{c}f}-{bins[5]:0.{c}f}{units}"] = round(x[2:5].sum(),digit)
        tir[f">{bins[3]:0.{c}f}{units}"] = round(x[3:].sum(),digit)
        tir[f">{bins[4]:0.{c}f}{units}"] = round(x[4:].sum(),digit)
        tir[f">{bins[5]:0.{c}f}{units}"] = round(x[5:].sum(),digit)
        tir[f"{bins[5]+1/div:0.{c}f}-{bins[6]:0.{c}f}{units}"] = round(x[5],digit)
        tir[f">{bins[6]:0.{c}f}{units}"] = round(x[6:].sum(),digit)
        tir = pd.DataFrame(tir,index=[idx]).T
        #tir[idx+'_tot_mins']=tir[idx]/100*tt
        # norm = pd.DataFrame(normal,index=tir.index,columns=[idx])
        # print(norm)
        tir[idx+'_data']=(tir[idx]/100*hrs).map(convert_time)
        tir['recommend'] = (normal*hrs)
        tir['recommend'] = tir['recommend'].map(convert_time)
        tir['recommend'] = direction+tir['recommend']
        
        ans = pd.concat([ans,tir],axis=1)
    mli = pd.MultiIndex.from_product([idxs,['%time','data','recommend']])
    ans.columns = mli
    return ans

def mage(x,**kwargs):
    pass






    

