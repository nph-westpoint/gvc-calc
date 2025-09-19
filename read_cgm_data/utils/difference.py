import pandas as pd
from datetime import datetime,timedelta,time

def difference(data,h):
    """
    difference - given a pandas series data, return the 
                values shifted by h hours used by other 
                methods (conga-h)
    Input: data - pandas Series with index as a datetime, 
                values are glucose readings associated 
                with those times.
    Output: pandas Series differenced and shifted by h hours

    data.index.shift(freq=timedelta(minutes=1),periods = h*60) - shifting the
        index by 1 hour forward and one hour back, keeping only the ones in the
        original index. idx1 - start+h-hours finishes at the end; 
                        idx2 - start at beginning, finishes h hours prior to end
        This allows the difference to have the same number of elements.
    """
    index = data.index
    idx_shift = data.index.shift(freq=timedelta(minutes=1),periods=h*60)
    idx1 = idx_shift[idx_shift.isin(index)]
    idx_shift = data.index.shift(freq=timedelta(minutes=1),periods=-h*60)
    idx2 = idx_shift[idx_shift.isin(index)]
    diff = []
    for i in range(len(idx2)):
        diff.append(data[idx1[i]]-data[idx2[i]])
    return pd.Series(diff,index=idx1,dtype=float)

def difference_m(data,m):
    """
    difference_m - given a pandas series data, return the 
                difference shifted by m minutes used by 
                variability metrics.
    Input: data - pandas Series with index as a datetime, 
                values are glucose readings associated with 
                those times.
    Output: pandas Series diffenced and shifted by m minutes
    """
    index = data.index
    period = m
    idx_shift = data.index.shift(freq=timedelta(minutes=1),periods=period)
    idx1 = idx_shift[idx_shift.isin(index)]
    idx_shift = data.index.shift(freq=timedelta(minutes=1),periods=-period)
    idx2 = idx_shift[idx_shift.isin(index)]
    diff = []
    for i in range(len(idx2)):
        diff.append(data[idx1[i]]-data[idx2[i]])
    return pd.Series(diff,index=idx1,dtype=float)  