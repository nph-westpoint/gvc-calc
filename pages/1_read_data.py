import streamlit as st
import pandas as pd
import numpy as np
import read_cgm_data as rd
from copy import deepcopy

pages_master = st.session_state['pages_master']
pages_dict = st.session_state['pages_dict']
rd.display_page_links(pages_dict)

options = []

st.subheader("Instructions")
body=''
body+=":red[This application takes `.csv` files with CGM data "
body+="as input and produces variability metrics for that data along "
body+="with graphs to visualize the data.\n This tool will assist the app user to "
body+="set up the file structure so that all of the files can be loaded. It is important that "
body+="the data used have the same structure in terms of the following:] \n"
body+="1) header row (the row that the header row appears on should be consistent) \n"
body+="2) rows to skip (the number of rows that needs to be skipped should be the same) \n"
body+="3) time delta (the difference between observations should have the same time difference) \n"
body+="4) unit of measure (chose between mg/dL or mmol/L) \n"
body+="5) date-time column in the same relative order and glucose column in the same order."

st.markdown(body)
body="#### Video: [Data/File Structure](https://youtu.be/Bsf5e3RWe8Q)"
st.sidebar.markdown(body)
restart=st.sidebar.button("Restart")
if restart:
    rd.initialize_session()
    st.switch_page("app_launch.py")
st.divider()


current_file = st.session_state['current_file']
skip_rows = st.session_state['skip_rows']
date_col = st.session_state['date_col']
date_col_idx = st.session_state['date_col_idx']
glucose_col = st.session_state['glucose_col']
glucose_col_idx = st.session_state['glucose_col_idx']
date_format = st.session_state['date_format']
header_row = st.session_state['header_row']
units = st.session_state['units']
time_delta_ = st.session_state['time_delta']

units_dict = {'mg':'mg/dL','mmol':'mmol/L'}
units_val = {'mg':0, 'mmol':1}

if current_file is None:
    current_file = st.file_uploader("Choose a file")
if current_file is not None:
    st.markdown('## Step 1: Choose the header / skip rows')
    ## Once a file has been selected, it will either be able to 
    ## be read as a dataframe or a list from `view_raw_data`
    st.session_state['current_file'] = deepcopy(current_file)
    #stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    data = rd.view_raw_data(current_file.read(),
                            skip_rows=skip_rows,
                            header_=header_row,
                            stop_=15+skip_rows)
    if isinstance(data,list):
        ## if df is a list it means that the user has not used a good header row
        ## or skip rows - view_list will display the list with the choosed header 
        ## row highlighted in salmon
        st.write(rd.view_list(data,header_row))
    else:
        ## if df is a data frame, then it can be used to pick the date col 
        ## and glucose col
        st.write(data)
    st.divider()
    header_row = st.sidebar.number_input("Choose the header row:",
                                         min_value = 0,
                                         max_value = 20,
                                         value = header_row,
                                         on_change = rd.change_attribute,
                                         key = 'header_key',
                                         args = ('header_row','header_key'))
    
    skip_rows = st.sidebar.number_input("Number of rows to skip:",
                                min_value = 1,
                                max_value = 20,
                                value = skip_rows,
                                on_change = rd.change_attribute,
                                key = 'skip_key',
                                args=('skip_rows','skip_key'))
    idx = [1,5,15].index(time_delta_)
    time_delta_ = st.sidebar.radio("Time Delta for this analysis:",
                                  key='time_delta_key',
                                  options=[1,5,15],
                                  on_change = rd.change_attribute,
                                  index = idx,
                                  args = ('time_delta','time_delta_key'))
    
    units_ = st.sidebar.radio("Units of Measurement in data:",
                             options = units_dict.values(),
                             index = units_val[units])
    
    try:
        ## Use the data from above to assign date/glucose columns
        st.markdown('## Step 2: Select the date and glucose columns.')
        view_cols = st.columns(2)
        with view_cols[0]:
            choices = list(data.columns)
            st.markdown("#### Select Date-Time Column:")
            if date_col is not None:
                dt_col = choices.index(date_col)
            else:
                dt_col = None
            date_col = st.radio("Select a date column:",
                                key = 'date_col_radio',
                                options = choices,
                                index = date_col_idx,
                                on_change = rd.change_column,
                                args = ('date_col','date_col_radio',choices))
        with view_cols[1]:
            st.markdown("#### Select Glucose Column:")
            if glucose_col is not None:
                gl_col = choices.index(glucose_col)
            else:
                gl_col = None
            glucose_col = st.radio("Select a glucose column:",
                                   key = 'glucose_col_radio',
                                   options = choices,
                                   index = glucose_col_idx,
                                   on_change = rd.change_column,
                                   args = ('glucose_col','glucose_col_radio',choices))
        if date_col is not None and glucose_col is not None:
            cols = [date_col,glucose_col]
            data = data[cols]
    except:
        body = ":red[Fix the header row or skip rows. Currently the values are: "
        body += "header="+str(header_row)+"; number of skip rows="+str(skip_rows)+".]"
        st.markdown(body)

        
    if date_col is not None and glucose_col is not None:
        try:## try to use the standard pandas.to_datetime without date_format
            data[date_col] = pd.to_datetime(data[date_col])
            ## show that the time_shift of the data yields a timedelta
            data['time_delta'] = data[date_col].shift(-1)-data[date_col]
            median_time_delta = data['time_delta'].map(lambda x:x.total_seconds()//60).median()
            mgv = data[glucose_col].mean()
            
            st.write(f"Median time_delta in your data is: {median_time_delta:.1f} minutes.")
            
            ## since date_format is only needed if the data cannot be transformed with
            ## the normal pandas.to_datetime method, make it None and use .to_datetime()
            st.session_state['date_format'] = '%Y-%m-%d %H:%M:%S'
            time_delta_ = int(median_time_delta)
            st.session_state['units'] = 'mg' if mgv > 30 else 'mmol'
            st.write(f"Glucose values in this data are in {st.session_state['units']}.")
            ok_btn = st.sidebar.button(label=":information_source:**OK** ->:file_cabinet: Import",
                                    on_click=rd.add_page_link,
                                    args=(":file_cabinet: Import_Data","pages/2_import_data.py")
            )
            if ok_btn:
                st.session_state['units']=units_.split('/')[0]
                #st.session_state['time_delta']=time_delta_
        except: ## give instructions to fix datetime and/or glucose values 
            st.write('Fix your datetime and glucose values before using this app.')
            st.markdown('[Link](https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html)')


