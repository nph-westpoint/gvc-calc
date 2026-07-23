import pandas as pd
import numpy as np

import streamlit as st
from datetime import datetime
import time

def initialize_session():
    st.session_state['cgm_data']=None
    st.session_state['current_file']=None
    st.session_state['skip_rows'] = 1
    st.session_state['date_col'] = None
    st.session_state['date_col_idx'] = None
    st.session_state['glucose_col'] = None
    st.session_state['glucose_col_idx'] = None
    st.session_state['date_format'] = '%Y-%m-%d %H:%M:%S'
    st.session_state['pages_dict'] = {":house: Home":"app_launch.py",
                                    ":information_source: Data Structure":"pages/1_read_data.py",
                                    #"Test":"pages/test_page.py",
                                    }
    st.session_state['time_delta'] = 5
    st.session_state['header_row'] = 0
    st.session_state['units']= "mg"
    st.session_state['cohort_stats'] = None
    #st.switch_page("app_launch.py")
    

def session():
    for key in st.session_state.keys():
        st.write(key, st.session_state[key])
    return None

def change_attribute(key1,key2):
    val = st.session_state[key2]
    st.session_state[key1]=val
    return None

def change_date_time(fmt_options,fmt_dates):
    val = st.session_state['dt_selected']
    idx = fmt_options.index(val)
    st.session_state['date_format'] = fmt_dates[idx]
    return None

def change_column(key1,key2,lst):
    val1 = st.session_state[key2]
    st.session_state[key1]=val1

    val2 = lst.index(val1)
    st.session_state[key1+'_idx']=val2


def display_page_links(pages):
    for key in pages.keys():
        st.sidebar.page_link(pages[key],label=key)
    return None

def add_page_link(page_key,page_link):
    pages = st.session_state['pages_dict']
    pages[page_key]=page_link
    st.session_state['pages_dict']=pages


def remove_page_link(page_key):
    pages = st.session_state['pages_dict']
    if page_key in pages.keys():
        del pages[page_key]
    st.session_state['pages_dict']=pages

def extract_time_period_data(df,period,name,hours,period_idx,deltat=5):
    total = (hours*60)//deltat+1
    minutes = np.arange(0,hours*60+deltat,deltat)
    try:
        ex = df.loc[period[0]:period[1],['imputed']].iloc[-total:]
        ex.index=minutes
        ex.columns = [f'{name}_{period_idx}']
    except:
        ex=df.loc[period[0]:period[1],'imputed']
    
    return ex

def timer(func):
    def wrapper(*args,**kwargs):
        start_time = time.time()  #Start time
        result = func(*args,**kwargs)
        end_time = time.time()
        print(f"Function {func.__name__!r} took: {end_time-start_time:0.4f} sec")
        return result
    return wrapper

def return_time_data(series,time0,time1):
    """
    return_time_data - returns glucose values for all days 
        between time0 and time1.
    """
    fmt = '%m/%d/%Y-'+'%H:%M'
    df = series.copy()
    day0 = series.index[0].date().strftime("%m/%d/%Y")
    if 'time' not in series.columns:
        time = [t.time() for t in df.index]
        df['time']=time
    time0 = datetime.strptime(day0+'-'+time0,fmt).time()
    time1 = datetime.strptime(day0+'-'+time1,fmt).time()
    df = df[(df['time']>=time0) & (df['time']<=time1)]

    return df

def return_day_data(data,day):
    """
    return_day_data - returns glucose values for a single day
        or a list of days.
        
    Input: data - dataframe with at least ['day', 'glucose'] as columns
           day - string or list of strings with format ('2025-03-28' or '03/28/2023')
    
    Output: pandas series object with datetime index and glucose readings as values.
    
    Example: return_day_data(df,'03/28/2023')
    
    """
    
    try:
        day = pd.to_datetime(day).date()
        glucose = data.loc[data['day']==day]['glucose']
    except:
        day=[pd.to_datetime(d).date() for d in day]
        glucose = data.loc[data['day'].isin(day)]['glucose']
    return glucose