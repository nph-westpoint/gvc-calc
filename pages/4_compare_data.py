import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import read_cgm_data as rd

cgm_data = st.session_state['cgm_data']
pages_dict = st.session_state['pages_dict']


rd.display_page_links(pages_dict)


options = ["Select Statistics","Correlation","X-Y Plots"]
select = st.pills("Select a tool:",
                  options = options,
                  default = options[0])
if select == options[0]:
    st.subheader("Select Statistics")
    cohort_stats = st.session_state['cohort_stats']
    stats = list(cgm_data.stats_df.columns)
    st.multiselect("Choose the stats for the analysis.",
                            options = stats,
                            default = cohort_stats,
                            key = 'ms_stats')
if select == options[1]:
    try:
        st.session_state['cohort_stats'] = st.session_state['ms_stats']
    except:
        pass
    st.subheader("Correlation Matrix")
    cohort_stats = st.session_state['cohort_stats']
    data = cgm_data.df[cohort_stats].copy()
    data.index = np.arange(len(data))
    corr = data[cohort_stats].corr()
    st.write(corr)
    st.divider()
    st.subheader("Correlation Plot")
    fig = plt.figure()
    sns.heatmap(corr,cmap= "coolwarm",vmin=-1,vmax=1)
    st.pyplot(fig)
    

if select ==options[2]:
    st.subheader("X-Y Plot of Two Statistics to Show Relationship")
    cohort_stats = st.session_state['cohort_stats']
    if cohort_stats is not None:
        linear_compare = st.multiselect("Select two cohort stats to compare.",
                                        options = cohort_stats,
                                        key = 'compare',
                                        max_selections=2)
        if len(linear_compare)>1:
            fig = plt.figure()
            sns.scatterplot(x=linear_compare[0],y=linear_compare[1],
                            data = cgm_data.df[linear_compare])
            plt.title("Plot X="+linear_compare[0]+"; Y="+linear_compare[1])
            st.pyplot(fig)