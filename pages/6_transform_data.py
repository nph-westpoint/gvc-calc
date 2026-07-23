import streamlit as st
import read_cgm_data as rd
import pandas as pd

cgm_data = st.session_state['cgm_data']
pages_dict = st.session_state['pages_dict']
pages_master = st.session_state['pages_master']
time_delta = int(st.session_state['time_delta'])
glucose_col = st.session_state['glucose_col']
rd.display_page_links(pages_dict)

st.subheader("Transform Data")
if cgm_data is not None:
    body = "Transform the data to work with Calculus parameter application. "
    body+="Designate the amount of time at the end of each time period (up to "
    body+="24 hours) to transform into row data for each file. Example "
    body+="(5 minute between observations): \n\n"
    body+="|file/period|min0|min5|min10|...|min287| \n"
    body+="|---|---|---|---|---|---| \n"
    body+="|filename1_period1|g0|g5|g10|...|g287| \n"
    body+="|filename1_period2|g0|g5|g10|...|g287| \n"
    body+="|filename2_period2|g0|g5|g10|...|g287| \n"
    st.markdown(body)
    hours = st.number_input(label="Hours of data (2-24)",
                  value=2,
                  min_value = 2,
                  max_value = 24)
    st.divider()
    st.markdown("#### First file example ")
    name = cgm_data.names[0]
    periods = cgm_data.data[name].periods
    ex = cgm_data.data[name].series
    tf_data = pd.DataFrame()
    for i,period in enumerate(periods):
        period_df=rd.extract_time_period_data(ex,period,name,hours,i,deltat=time_delta)
        tf_data=pd.concat([tf_data,period_df],axis=1)
    st.write(tf_data.T)

    #### OK button / execute
    body = "If this example looks good for the first file, then click OK to get the same data for all of the files"
    st.write(body)
    transform_ok_btn=st.button(label="OK")
    if transform_ok_btn:
        for name in cgm_data.names[1:]:
            ex = cgm_data.data[name].series
            periods = cgm_data.data[name].periods
            for i,period in enumerate(periods):
                period_df = rd.extract_time_period_data(ex,period,name,hours,i,deltat=time_delta)
                tf_data=pd.concat([tf_data,period_df],axis=1)
        st.write(tf_data.T)
        body="Download this data and drop into [Calculus Stats]"
        body+="(https://app-example-kodonculator.streamlit.app/). Make sure you use "
        body+="1 for the number of non-numerical rows in the Kondonculator (the default is 2)."
        st.markdown(body)


