import streamlit as st
import pandas as pd
import read_cgm_data as rd

pages_dict = st.session_state['pages_dict']
pages_master = st.session_state['pages_master']
rd.display_page_links(pages_dict)
st.subheader("Import Data")

## Initialize session parameters
cgm_data = st.session_state['cgm_data']
time_delta = int(st.session_state['time_delta'])
date_col_idx = st.session_state['date_col_idx']
glucose_col_idx = st.session_state['glucose_col_idx']
date_format = st.session_state['date_format']
skip_rows = st.session_state['skip_rows']
header_row = st.session_state['header_row']
units = st.session_state['units']
ok_btn = False

if cgm_data is None:
    names = [];dataframes=[]
    uploaded_files = st.file_uploader("Select .csv files to upload.",
                                    type="csv",
                                    accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        names.append(uploaded_file.name)
        file = uploaded_file
        df = rd.read_data(
                          filename=file,
                          date_col=date_col_idx,
                          glucose_col=glucose_col_idx,
                          header_=header_row,
                          skip_rows=skip_rows,
                          )
        dataframes.append(df)


    if len(names)>0:
        cgm_data = rd.multiple_CGM(names,
                                dataframes,
                                dt_fmt=date_format,
                                units=units,
                                time_delta = time_delta 
                                )
        st.session_state['cgm_data'] = cgm_data
        rd.remove_page_link(pages_master[1][0])
        rd.remove_page_link(pages_master[2][0])
        ok_btn = st.sidebar.button("OK",key="sb_ok_pg2_",on_click=rd.add_page_link,
                        args=(pages_master[3][0],pages_master[3][1]))
    else:
        pass
        
else:
    st.switch_page("pages/3_explore_data.py")
    
        
        


