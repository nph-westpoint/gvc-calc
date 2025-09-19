from datetime import datetime,timedelta
import numpy as np

def tic(idx):
    """
    tic - (time_index_conversion) - converts a datetime to day and min past
        midnight in a day to access self.df
    """
    day = idx.strftime('%m/%d/%Y')
    time = idx.strftime('%H:%M')
    return day,time,int(time[:2])*60+int(time[3:])

def itc(day,mm):
    """
    itc - (index_time_conversion) - converts a day / time (min past midnight)  
        to a datetime needed to index the series
    """
    return datetime.strptime(day+f' {mm//60:0>2}:{mm%60:0>2}','%m/%d/%Y %H:%M')

def unique_days(data):
    """
    unique_days - given a pandas Series with datetime as it's index => return
                    the unique days in that set.
    Input: data - pandas Series with index as a datetime, values are glucose 
                    readings associated with those times.
                    
    Output: days - a list of unique days.
    """
    days = [tic(d)[0] for d in data.index]
    days = sorted(list(set(days)))
    return days

def convert_to_time(minutes):
    """
    given a list of minutes after midnight, convert to a string time.
    """
    time = []
    for m in minutes:
        hh = m//60
        mm = m%60
        time.append(f'{hh:0>2}:{mm:0>2}')
    return time

def linear_interpolate(glucose_values, indices):
    """
    linear_interpolate - interpolating values between data
    
    Input:
        glucose_values - all of the glucose values for a day (288)
        indices - the indicies of the glucose values that need imputing
    Output:
        interpolated_values - all of the glucose values linear interpolated
        for those that are missing.
    """
    x = np.array(np.arange(len(glucose_values)),dtype=int)
    mask = np.ones(len(glucose_values), dtype=bool)
    mask[indices] = False
    interpolated_values = np.interp(x, 
                                    x[mask], 
                                    glucose_values[mask].astype(float))
    return interpolated_values

