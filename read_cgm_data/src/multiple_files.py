from .cgm_object import CGM
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

class multiple_CGM(object):
    """
    this class serves as the connection between the CGM class and the
        streamlit app. It holds the objects representing each data file
        in a list so that it can use them when selected in the app.
    """
    def __init__(self,names,
                 file_dfs,
                 dt_fmt='%Y-%m-%d %H:%M:%S',
                 units='mg',
                 time_delta = None,
                 ):
        self.names = names
        self.files = file_dfs
        self.units = units

        self.data={}
        df = pd.DataFrame()
        progress_text = ""
        my_progress_bar = st.progress(0,text=progress_text)
        with st.status("Calculating Statistics"):
            for i in range(len(names)):
                my_progress_bar.progress((i+1)/len(names),text=progress_text)
                name = names[i]
                file_df = file_dfs[i]
                
                st.write(name)
                self.data[name]=CGM(filename=name,
                                    file_df=file_df,
                                    max_break = 45,
                                    dt_fmt=dt_fmt,
                                    units=units,
                                    deltat=time_delta,
                                    )
                df = pd.concat([df,self.data[name].calculate_stats().T])
        my_progress_bar.empty()
        self.selected_file = self.names[0]
        self.df = df
        # cohort columns come from CGM.stats_functions dictionary
        # Lets the user choose functions that are not vectors to see correlations
        cols = self.data[name].cohort_cols
        self.time_delta = self.data[name].time_delta #assumes all time deltas are the same
        self.deltat = self.data[name].deltat
        self.stats_df=df[cols]

    def create_stats_dataframe(self,units='mg'):
        names = self.names
        stats_df = pd.DataFrame()
        for i in range(len(names)):
            name = names[i]
            stats_df = pd.concat([stats_df,self.data[name].calculate_stats(units=units).T])
        stats_df['total_time'] = pd.to_timedelta(stats_df['total_time'])
        return stats_df
            
    
    def ambulatory_glucose_profile(self,name):

        st.pyplot(self.data[name].agp_all(True))
        st.divider()
        st.write(self.data[name].stats.style.format(precision=3))
        st.divider()
        daily = st.checkbox(label="Display daily statistics",value=False)
        if daily:
            st.write(self.data[name].calculate_stats(group_by='date')
                                    .style.format(precision=3))
        else:
            st.write("Daily stats may take time to calculate depending on the number of days.")
        self.selected_file = name

    def agp_report(self,name):
        """
        agp_report - the report used when 'AGP Report' is selected on the Explore Data tab

        """
        options = ["Glucose Statistics and Targets",
                   "Time in Ranges",
                   "AGP",
                   "Daily Glucose Profile"]
        tab1,tab2,tab3,tab4 = st.tabs(options)
        with tab1:
            st.markdown("### :blue-background[GLUCOSE STATISTICS AND TARGETS]")
            st.write(self.data[name].tir_stats_table().style.format(precision=3))
           
        with tab2:    
            st.markdown("### :blue-background[TIME IN RANGES]")
            options = ['all','sleep/wake','wear_period','date']
            agg = st.selectbox("Period of time to consider:",
                               options = options)
            idx = options.index(agg)
            agg = ['all','day','period','date'][idx]
            st.write(self.data[name].tir_report(agg=agg).style.format(precision=3))

        with tab3:
            st.markdown("### :blue-background[AMUBULATORY GLUCOSE PROFILE (AGP)]]")
            body = "AGP is a summary of glucose values from the report period, "
            body+="with median (50%) and other percentiles (75%, 95%) shown as "
            body+="if occuring in a single day."

            st.markdown(body)
            st.pyplot(self.data[name].plot_agp())

        with tab4:
            st.markdown('### :blue-background[DAILY GLUCOSE PROFILES]')
            st.pyplot(self.data[name].plot_daily_traces())
        
        self.selected_file = name
        
    def view_df_series(self,name):
        st.write(self.data[name].df)
        st.divider()
        st.write(self.data[name].data)
        st.divider()
        st.write(self.data[name].periods)
        #st.divider()
        #st.write(self.data[name].stats)
        self.selected_file = name
        
    def view_gri(self,name):
        st.pyplot(self.data[name].plot_gri())
        self.selected_file = name

    def time_in_range_report(self,name):
        self.data[name].time_in_range_report()

    def visualize_data(self,name):
        options = ['Poincare Plot','Time Series']
        tab1,tab2 = st.tabs(options)
        with tab1:
            #options = [td for td in range(self.time_delta,12*self.time_delta+1,self.time_delta)]
            shift_minutes = st.number_input("Time between observations",min_value = self.deltat,
                            max_value = self.deltat*12,step = self.deltat)
            fig=self.data[name].poincare_plot(shift_minutes)
            st.pyplot(fig)
        with tab2:
            fig = self.data[name].time_series_plot()
            st.pyplot(fig)

    def episodes(self,name,):
        const = self.data[name].const
        # opts = ['\<','\>']
        # gorl = st.radio(label="Choose an inequality:",
        #                       options = ['\<','\>'],
        #                       index = 0,
        #                       )
        # idx = opts.index(gorl)
        # val = st.number_input(label = "Choose the value:",
        #                       min_value = 50/const,
        #                       max_value = 250/const,
        #                       step = 10/const,
        #                       )
        # st.write(idx,val)
        # ineq = ['less','greater'][idx]
        hypo1 = self.data[name].episodes(70/const,'less',cl=1)
        hypo2 = self.data[name].episodes(54/const,'less',cl=2)
        
        if len(hypo1)>0 or len(hypo2)>0:
            st.markdown('### Hypoglycemic Events')
        if len(hypo1)>0:
            body='Level 1 Hypoglycemia Alert - 54-70 mg/dL (3.0-3.9 mmol/L) '
            body+='for at least 15 consecutive minutes.'
            st.write(body)
            st.write(hypo1)
        if len(hypo2)>0:
            body='Level 2 Clinically Significant - <54 mg/dL (3.0 mmol/L) '
            body+='for at least 15 consecutive minutes.'
            st.write(body)
            st.write(hypo2)

        hyper1 = self.data[name].episodes(180/const,'greater',cl=1)
        hyper2 = self.data[name].episodes(250/const,'greater',cl=2)
        if len(hyper1)>0 or len(hyper2)>0:
            st.markdown('### Hyperglycemic Events')
        if len(hyper1)>0:
            body='Level 1 High Glucose Alert - 180-250 mg/dL (10-13.9 mmol/L) '
            body+='for at least 15 consecutive minutes.'
            st.write(body)
            st.write(hyper1)
        if len(hyper2)>0:
            body='Level 2 Very High Glucose Alert - >250 mg/dL (>13.9 mmol/L) '
            body+='for at least 15 consecutive minutes.'
            st.write(body)
            st.write(hyper2)



        

    def markov_analysis(self,name):
        res,fig = self.data[name].markov_chain_calculation([54,70,180,250])
        st.pyplot(fig)
        st.html(res)




        
    def export_data(self,filename,units):
        df = self.create_stats_dataframe(units=units)
        df['idx']=self.names
        df.set_index('idx',inplace=True)
        st.write(df)
        st.sidebar.download_button(label="Download csv",
                           data = df.to_csv().encode('utf-8'),
                           file_name=filename)
        
    def test_develop(self,name):
        """
        Using the test_develop method - allows for development of functions in streamlit
        """
        # fig = self.data[name].time_series_plot()
        # st.pyplot(fig)

        #self.data[name].markov_chain_calculation([54,70,180,240])
        #st.pyplot(self.data[name].time_series_plot(True))
        #self.episodes(name)
        #fig = self.data[name].poincare_plot()
        #st.pyplot(fig)
        res,fig = self.data[name].markov_chain_calculation([54,70,180,250])
        st.pyplot(fig)
        st.html(res)

        
        