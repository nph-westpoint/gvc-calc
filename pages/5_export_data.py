import streamlit as st
import read_cgm_data as rd

pages_dict = st.session_state['pages_dict']
pages_master = st.session_state['pages_master']
rd.display_page_links(pages_dict)
st.subheader("Export Data")


cgm_data = st.session_state['cgm_data']
if cgm_data is not None:
    units = st.radio("Select the units for export.",
                            options=['mg','mmol'],
                            index=0)
    
    st.write(units)
    filename=st.sidebar.text_input("Name of File","cgm_download_"+units+".csv")
    cgm_data.export_data(filename,units)