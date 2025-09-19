import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t,entropy
import streamlit as st
import collections

from datetime import datetime,timedelta
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as patches
import matplotlib.dates as mdates

from ..utils import (return_time_data,)
from ..functions import (time_in_range,glucose_N,glucose_mean,
                         glucose_median,total_time,
                        glucose_std,glucose_cv,mean_absolute_glucose,
                        j_index,low_high_blood_glucose_index,
                        glycemic_risk_assessment_diabetes_equation,
                        glycemic_variability_percentage,
                        lability_index,mean_of_daily_differences,
                        conga,m_value,
                        glucose_management_indicator,interquartile_range,
                        auc_thresh,mean_amplitude_of_glycemic_excursions,
                        glycemic_risk_index,auc,
                        eccentricity,entropy_mc,
                        calculate_time_in_range,transition_matrix,
                        cogi,adrr,
                        )


class CGM(object):
    def __init__(self,
                 filename,
                 file_df,
                 max_break = 45, 
                 dt_fmt='%Y-%m-%dT%H:%M:%S',
                 units='mg',
                 first_full_day=False):
        """
        CGM data object for storing the original data, creating a time grid
            for all observations to be linearly interpolated to. Calculates
            statistics that are in the stats_functions dictionary by type and
            level. 
            - Converts all data to mg/dL for calculations.
            - 
        Inputs: 
        filename - name of file that will be displayed on export or
            selection while viewing data.
            
        file_df - dataframe from csv raw data.

        Attributes:
            df - raw data from csv read data dropping NA values.
            
            data - raw data converted to grid datetime values.
            
            days - the days in the format 'mm/dd/yyyy' from the time stamp
        
        times - the 288 times which represent the number of minutes 
            after midnight [0,5,10,...1440]
            
        shape - the number of days (rows) and times (columns) in the data
        
        rows - the days in the data that have missing data.
            Example: [0,3,8] day 0,3,8 are missing at least one instance
                of data
        cols - a list of columns from the data that is missing data. These
            will align with rows to give the columns in each row that are 
            missing.
            Example: [array([0,1,2,...238]), array(230,231), 
                      array(159,160,...287)] day 0 is missing the first part
                    of the day, day 3 is missing only the 230 and 231 columns.
        
        """
        ## First Full Day #######################################
        if first_full_day:
            first_time = file_df.index[0].time()
            if (first_time.hour != 0) and (first_time.minute != 0):
                file_df = file_df[file_df.index.date > file_df.index.date[0]]
                periods[0][0] = file_df.index[0]

        
        ## Assign object attributes #############################
        self.filename = filename
        self.df = file_df.dropna() 
        self.max_break = max_break
        
        ## Date - Time constructs self.dt_fmt represents the datetime string format
        # self.deltat is the median times between observations rounded to the nearest minute
        # self.time_delta is a timedelta version of deltat
        self.dt_fmt = dt_fmt
        self.util_time_delta() #=> time_delta and deltat


        ## convert 'mmol' to 'mg' for calculations (if needed)
        self.c=1
        self.units = units
        if units == 'mmol':
            self.df['glucose']=self.df['glucose']*18.018

        ## convert data frame to datetime grid
        data, periods, percent_active = self.create_time_grid()
        self.data = data.dropna()
        self.days = self.data['day'].unique()
        self.periods = periods
        self.percent_active = percent_active
        self.plot_data = self.init_plot_data()
        
        self.daily_stats = None
        self.params = {}
        self.set_params({'type':'paper','lower':70/self.c,'conga_h':1,
                         'highest':250/self.c,'upper':180/self.c,'lowest':54/self.c,
                         'thresh':100/self.c,'m_index':120,
                         'above':True,'li_k':60,'N':len(data),
                         'days':self.days,
                         'periods':periods,
                         'units':'mg', ## always converted to mg
                         'data':self.data,
                         'time_delta':self.deltat,
                         'deltat':self.deltat,
                         'day_data':return_time_data(self.data.copy(),'6:00','23:59'),
                         'night_data':return_time_data(self.data.copy(),'00:00','05:59'),
                         'calculation':'all',
                         'time_type':'all'})
        self.N = len(data)
        self.assign_functions()
        self.time_in_range = calculate_time_in_range(**self.params)

        self.calculate_stats()
        self.create_cohort_columns()
        
    ### Pulling in the data from the file ####################################
    ### Creating attributes that can explain features of the data ############
    ### data - dictionary, df - dataframe, series - series ###################
    
    def interpolate_period(self,data,period,time_delta):
        try:
            data=pd.Series(data['glucose'].values,index=data.index)
        except:
            pass
        start_min = (period[0].minute)
        end_min = (period[1].minute)
        start_min = start_min//time_delta*time_delta
        end_min = (end_min)//time_delta*time_delta+time_delta
        add_hour = 0
        if end_min == 60:
            end_min = 0
            add_hour+=1
        period[0] = period[0].replace(minute=start_min)
        period[1] = period[1].replace(minute=end_min)+timedelta(hours=add_hour)


        data = data.loc[period[0]:period[1]]
        datetime = pd.date_range(period[0].strftime("%Y-%m-%d %H:%M"),
                                 period[1].strftime("%Y-%m-%d %H:%M"),
                                freq = str(time_delta)+"min")

        ynew = np.interp(datetime,data.index,data.values,left=np.nan,right=np.nan)
        df = pd.DataFrame(ynew,index=datetime,columns=['glucose'])

        num = len(data.dropna()-1)
        denom = len(df.dropna())

        return(df,num,denom)    
    
    def util_time_delta(self):
        """
        Automatically figure out time delta - median over the entire spread of times.
            Creates two attributes of the object CGM - 1)time_delta - timedelta object 
            and 2)deltat - integer with same time delta 

            returns 
                `self.time_delta` which is a timedelta object
                `self.deltat` which is an int
        """
        df = self.df
        self.time_delta = (df.index[1:]-df.index[:-1]).median()
        self.deltat = int(round(self.time_delta.total_seconds()/60,0))
        return self.time_delta,self.deltat
    
    def build_periods(self,data,max_break=45):
        ts = data.index
        data = pd.DataFrame(data.values,index = data.index,columns=['glucose'])

        data['dates'] = pd.to_datetime(data.index)
        data['date_shift']=data['dates'].shift(-1)
        data['time_diff'] = (data['date_shift']-data['dates'])/timedelta(minutes=1)
        idxs = list(data[data['time_diff']>max_break][['dates','date_shift']].values)

        periods = [[ts[0],ts[-1]]]
        for idx in idxs:
            t0 = pd.Timestamp(idx[0])
            t1 = pd.Timestamp(idx[1])
            periods.append([t1,periods[-1][1]])
            periods[-2][1] = t0
        periods_ = []
        for per in periods:
            t0 = data.loc[per[0]:per[1]]['glucose'].first_valid_index()
            t1 = data.loc[per[0]:per[1]]['glucose'].last_valid_index()
            periods_.append([t0,t1])            
        return periods_
    
    def create_time_grid(self):
        """
        creates a time grid for values

        start_date - start date in string format (%m-%d-%Y)
        end_date - end date in string format (%m-%d-%Y)

        """
        data = self.df.copy()
        time_delta = self.deltat
        max_break = self.max_break
        
        data=pd.Series(data['glucose'].values,index=data.index)
        data = data.dropna()
        start_min = data.index[0].minute
        start_date = data.index[0].replace(minute=start_min//time_delta*time_delta)
        start_date = start_date.strftime("%Y-%m-%d %H:%M")
        
        #end_date needs an end time
        end_min = data.index[-1].minute//time_delta*time_delta + time_delta
        add_hour = 0
        if end_min == 60:
            end_min = 0
            add_hour += 1
        end_date = data.index[-1].replace(minute=end_min)
        end_date = (end_date+timedelta(hours=add_hour)).strftime("%Y-%m-%d %H:%M")
        dates = pd.date_range(start_date,end_date,freq=str(time_delta)+"min")
        periods = self.build_periods(data,max_break)

        #ynew = np.interp(dates,data.index,data.values,left=np.nan,right=np.nan)
        df=pd.DataFrame(np.empty(len(dates))*np.nan,index=dates,columns=['glucose'])
        top=0;bottom = 0
        for period in periods:
            df_period,num,denom = self.interpolate_period(data,period,time_delta)
            df.update(df_period,join='left',overwrite=True)
            top += num
            bottom += denom
        periods = self.build_periods(df.dropna(),max_break)
        df['day']=df.index.map(lambda t:t.date())
        df['time']=df.index.map(lambda t:t.time())
        df['min']=df['time'].astype(str).str.split(":").map(lambda t: int(t[0])*60+int(t[1]))
        self.table = df.pivot_table(values='glucose',index='day',columns='min')
        return(df,periods,top/bottom)
    
    
    def active_percent(self,**kwargs):
        return self.percent_active
    
    def __len__(self):
        return len(self.data)
    
    ##############################################################
    ####### Statistic Functions ##################################
    ##############################################################
    
    def assign_functions(self):
        
        self.stats_functions = {}
        
        self.stats_functions['num_obs'] = {}
        self.stats_functions['num_obs']['f'] = glucose_N
        self.stats_functions['num_obs']['description']="number of observations"
        self.stats_functions['num_obs']['normal']=[]
        self.stats_functions['num_obs']['mmol']=1
        self.stats_functions['num_obs']['daily'] = True
        self.stats_functions['num_obs']['cohort'] = False
        

        self.stats_functions['total_time'] = {}
        self.stats_functions['total_time']['f'] = total_time
        self.stats_functions['total_time']['description']="total time"
        self.stats_functions['total_time']['normal']=[]
        self.stats_functions['total_time']['daily'] = True
        self.stats_functions['total_time']['cohort'] = False
        self.stats_functions['total_time']['mmol']=1

        self.stats_functions['percent_active'] = {}
        self.stats_functions['percent_active']['f'] = self.active_percent
        self.stats_functions['percent_active']['description']="percent active"
        self.stats_functions['percent_active']['normal']=[]
        self.stats_functions['percent_active']['daily'] = False
        self.stats_functions['percent_active']['cohort'] = False
        self.stats_functions['percent_active']['mmol']=1
        
        self.stats_functions['mean']={}
        self.stats_functions['mean']['f'] = glucose_mean
        self.stats_functions['mean']['description']="mean"
        self.stats_functions['mean']['normal']=[] #fasting glucose <100
        self.stats_functions['mean']['daily'] = True
        self.stats_functions['mean']['cohort'] = True
        self.stats_functions['mean']['mmol']=1/18.018

        self.stats_functions['median']={}
        self.stats_functions['median']['f'] = glucose_median
        self.stats_functions['median']['description']="median"
        self.stats_functions['median']['normal']=[]
        self.stats_functions['median']['daily'] = True
        self.stats_functions['median']['cohort'] = True
        self.stats_functions['median']['mmol']=1/18.018
        
        self.stats_functions['std']={}
        self.stats_functions['std']['f'] = glucose_std
        self.stats_functions['std']['description']="std"
        #2017 Internation Consensus Statement - Hill 2011 => [0,3] 
        self.stats_functions['std']['normal']=[0,54] 
        self.stats_functions['std']['daily'] = True
        self.stats_functions['std']['cohort'] = True
        self.stats_functions['std']['mmol']=1/18.018
        
        self.stats_functions['cv'] = {}
        self.stats_functions['cv']['f'] = glucose_cv
        self.stats_functions['cv']['description']="CV"
        #2017 Internation Consensus Statement
        self.stats_functions['cv']['normal']=[0,0.36]  
        self.stats_functions['cv']['daily'] = True
        self.stats_functions['cv']['cohort'] = True
        self.stats_functions['cv']['mmol']=1
        
        self.stats_functions['mag']={}
        self.stats_functions['mag']['f']=mean_absolute_glucose
        self.stats_functions['mag']['description']="MAG"
        #[0.5,2.2] Hill 2011
        self.stats_functions['mag']['normal']=[9,39.64] 
        self.stats_functions['mag']['daily'] = True
        self.stats_functions['mag']['cohort'] = True
        self.stats_functions['mag']['mmol']=1/18.018
        
        self.stats_functions['tir']={}
        self.stats_functions['tir']['f']=time_in_range
        self.stats_functions['tir']['description']="TIR"
        ## Normal Time in Range 0% less than 54, 4% below 70, >70% of the time
        ## between 70-180, <25% of the time greater than 180, 0% time greater than 250
        self.stats_functions['tir']['normal']=[0,4,70,25,0] #Cleveland Clinic website
        self.stats_functions['tir']['daily'] = True
        self.stats_functions['tir']['cohort'] = False
        self.stats_functions['tir']['mmol']=1
        self.stats_functions['tir']['levels']=['<54','54-70','70-180','180-250','>250']
        
        self.stats_functions['j_index']={}
        self.stats_functions['j_index']['f']=j_index
        self.stats_functions['j_index']['description']='J_Index'
        self.stats_functions['j_index']['normal']=[4.7,23.6] #Hill 2011
        self.stats_functions['j_index']['daily'] = True
        self.stats_functions['j_index']['cohort'] = True
        self.stats_functions['j_index']['mmol']=1
        
        self.stats_functions['bgi']={}
        self.stats_functions['bgi']['f']=low_high_blood_glucose_index
        self.stats_functions['bgi']['description'] = 'LBGI_HGBI'
        self.stats_functions['bgi']['type'] = ['paper','easy']
        self.stats_functions['bgi']['normal'] = {'LBGI':[0,6.9],
                                                 'HBGI':[0,7.7]} #Hill 2011
        self.stats_functions['bgi']['daily'] = True
        self.stats_functions['bgi']['cohort'] = False
        self.stats_functions['bgi']['mmol']=1
        self.stats_functions['bgi']['levels'] = ['low','high']

        self.stats_functions['grade']={}
        self.stats_functions['grade']['f']=glycemic_risk_assessment_diabetes_equation
        self.stats_functions['grade']['description']='GRADE'
        self.stats_functions['grade']['type'] = ['paper','easy']
        self.stats_functions['grade']['normal']=[0,4.6] #Hill 2011
        self.stats_functions['grade']['daily'] = True
        self.stats_functions['grade']['cohort'] = False
        self.stats_functions['grade']['mmol']=1
        self.stats_functions['grade']['levels']=['overall','low','target','high']
        
        self.stats_functions['gvp']={}
        self.stats_functions['gvp']['f']=glycemic_variability_percentage
        self.stats_functions['gvp']['description']='GVP'
        ## 0-20 Minimal, 20-30 Low, 30-50 Moderate, >50 High
        self.stats_functions['gvp']['normal']=[0,20,30,50] #Peyser 2018
        self.stats_functions['gvp']['daily'] = True
        self.stats_functions['gvp']['cohort'] = True
        self.stats_functions['gvp']['mmol']=1
        
        self.stats_functions['li']={}
        self.stats_functions['li']['f']=lability_index
        self.stats_functions['li']['description']='Lability_Index'
        #Hill 2011 - [0,4.7]
        self.stats_functions['li']['normal']= [0,1525.8]
        self.stats_functions['li']['daily'] = True
        self.stats_functions['li']['cohort'] = True
        self.stats_functions['li']['mmol']=1/18.018**2
        
        self.stats_functions['modd']={}
        self.stats_functions['modd']['f'] = mean_of_daily_differences
        self.stats_functions['modd']['description']='MODD'
        self.stats_functions['modd']['type'] = ['paper','easy']
        self.stats_functions['modd']['normal']=[0,3.5] #Hill 2011
        self.stats_functions['modd']['daily'] = False
        self.stats_functions['modd']['cohort'] = True
        self.stats_functions['modd']['mmol']=1/18.018
        
        self.stats_functions['adrr']={}
        self.stats_functions['adrr']['f'] = adrr
        self.stats_functions['adrr']['description']="ADRR"
        self.stats_functions['adrr']['type'] = ['paper','easy']
        self.stats_functions['adrr']['normal']=[0,8.7] #Hill 2011
        self.stats_functions['adrr']['daily'] = [True,False]
        self.stats_functions['adrr']['cohort'] = False
        self.stats_functions['adrr']['mmol']=1
        self.stats_functions['adrr']['levels'] = ['overall','low','high']
        
        self.stats_functions['conga']={}
        self.stats_functions['conga']['f'] = conga
        self.stats_functions['conga']['description']='conga'
        self.stats_functions['conga']['type']=['paper','easy']
        self.stats_functions['conga']['normal']=[3.6,5.5] #Hill 2011
        self.stats_functions['conga']['daily'] = True
        self.stats_functions['conga']['cohort'] = True
        self.stats_functions['conga']['mmol']=1/18.018
        
        self.stats_functions['m_value']={}
        self.stats_functions['m_value']['f'] = m_value
        self.stats_functions['m_value']['description']='M_Value'
        self.stats_functions['m_value']['type'] = ['paper','easy']
        self.stats_functions['m_value']['normal']=[0,12.5] #Hill 2011
        self.stats_functions['m_value']['daily'] = True
        self.stats_functions['m_value']['cohort'] = True
        self.stats_functions['m_value']['mmol']=1
        
        
        self.stats_functions['gmi']={}
        self.stats_functions['gmi']['f'] = glucose_management_indicator
        self.stats_functions['gmi']['description']='gmi'
        self.stats_functions['gmi']['normal']=[0,6] #Danne 2017
        self.stats_functions['gmi']['daily'] = True
        self.stats_functions['gmi']['cohort'] = True
        self.stats_functions['gmi']['mmol']=1
        
        self.stats_functions['iqr']={}
        self.stats_functions['iqr']['f'] = interquartile_range
        self.stats_functions['iqr']['description']='Inter-quartile range'
        self.stats_functions['iqr']['normal']=[13,29] #Danne 2017      
        self.stats_functions['iqr']['daily'] = True
        self.stats_functions['iqr']['cohort'] = True
        self.stats_functions['iqr']['mmol']=1/18.018
        
        self.stats_functions['auc']={}
        self.stats_functions['auc']['f']=auc
        self.stats_functions['auc']['description']='AUC'
        self.stats_functions['auc']['type']=['all','wake','sleep']
        self.stats_functions['auc']['normal'] = []
        self.stats_functions['auc']['daily'] = True
        self.stats_functions['auc']['cohort'] = True
        self.stats_functions['auc']['mmol']=1/18.018

        self.stats_functions['auc_100']={}
        self.stats_functions['auc_100']['f']=auc_thresh
        self.stats_functions['auc_100']['description']='AUC_100'
        self.stats_functions['auc_100']['type']=['all+','wake+','sleep+']
        self.stats_functions['auc_100']['normal'] = []
        self.stats_functions['auc_100']['daily'] = True
        self.stats_functions['auc_100']['cohort'] = True
        self.stats_functions['auc_100']['mmol']=1/18.018
        
        self.stats_functions['mage']={}
        self.stats_functions['mage']['f']=mean_amplitude_of_glycemic_excursions
        self.stats_functions['mage']['description']='MAGE'
        self.stats_functions['mage']['normal'] = []
        self.stats_functions['mage']['daily'] = False
        self.stats_functions['mage']['cohort'] = False
        self.stats_functions['mage']['mmol']=1/18.018
        self.stats_functions['mage']['levels'] = ['decrease','increase']
        
        self.stats_functions['gri'] = {}
        self.stats_functions['gri']['f']=glycemic_risk_index
        self.stats_functions['gri']['description']='glycemic risk index'
        ##ZoneA - 0-20;ZoneB - 20-40;ZoneC - 40-60;ZoneD - 60-80;ZoneE - 80-100## 
        self.stats_functions['gri']['normal'] = [0,20,40,60,80,100]
        self.stats_functions['gri']['daily'] = True
        self.stats_functions['gri']['cohort'] = False
        self.stats_functions['gri']['mmol']=1
        self.stats_functions['gri']['levels'] = ['gri','hypo','hyper']

        self.stats_functions['cogi'] = {}
        self.stats_functions['cogi']['f']=cogi
        self.stats_functions['cogi']['description']='COGI'
        self.stats_functions['cogi']['normal'] = [90,100]
        self.stats_functions['cogi']['daily'] = True
        self.stats_functions['cogi']['cohort'] = True
        self.stats_functions['cogi']['mmol']=1

        self.stats_functions['eccentricity']={}
        self.stats_functions['eccentricity']['f']=eccentricity
        self.stats_functions['eccentricity']['description']='Poincare Eccentricity'
        self.stats_functions['eccentricity']['normal'] = [0,1] #not sure
        self.stats_functions['eccentricity']['daily'] = True
        self.stats_functions['eccentricity']['cohort']= False
        self.stats_functions['eccentricity']['mmol']=1
        self.stats_functions['eccentricity']['levels']=['ecc','a','b']

        self.stats_functions['entropy']={}
        self.stats_functions['entropy']['f']=entropy_mc
        self.stats_functions['entropy']['description']='Markov Entropy'
        self.stats_functions['entropy']['normal'] = [0,1] #not sure
        self.stats_functions['entropy']['daily'] = True
        self.stats_functions['entropy']['cohort']= False
        self.stats_functions['entropy']['mmol']=1

        return None
    
    def create_cohort_columns(self):
        cohort_cols = []
        for key in self.stats['mg'].keys():
            cohort_cols.append(key)
        self.cohort_cols = cohort_cols
    
    
    def set_params(self,params):
        for key,value in params.items():
            self.params[key]=value
        return self.params
    
    def stats_search(self,keyword,units = 'mg'):
        def find_level(dict1,key):
            for k in dict1.keys():
                if key in k:
                    return k
            return None
        stats = self.stats[units]
        stats_keys = stats.keys()
        try:
            level_keys = self.stats_functions[keyword]['levels']
        except:
            return None
        if 'type' in self.stats_functions[keyword].keys():
            type_ = self.params['type']
            level_keys = [type_+"_"+lk for lk in level_keys]
        dict1 = {stat:stats[stat] for stat in stats_keys if keyword in stat}
        keys = [find_level(dict1,k1) for k1 in level_keys]
        vals = [stats[k] for k in keys]
        return vals
    
    def calculate_stats(self,data = None, name = "",units = 'mg'):
        """
        calculate_stats - this function is run once when during the constructor method
            to get all of the 'mg' and 'mmol' statistics for the entire trace.
        """
        if isinstance(data,type(None)):
            data = self.data['glucose']
            if hasattr(self,'stats'):
                return self.stats
            
        self.params['data']=data
        _type_ = self.params['type']
        stats = {}
        stats['mg']={}
        stats['mmol']={}
        ## calculate mg first for each key in stats functions
        for key in self.stats_functions:
            ## for each type in stats functions, calculate
            if 'type' in self.stats_functions[key].keys():
                types = self.stats_functions[key]['type']
                for type_ in types:
                    self.params['type']=type_
                    key_str = key+'_'+type_
                    value = self.stats_functions[key]['f'](**self.params)
                    if isinstance(value,collections.abc.Iterable):
                        for j in range(len(value)):
                            description = self.stats_functions[key]['levels'][j]
                            stats['mg'][key_str+'_'+description]=value[j]
                    else:
                        stats['mg'][key_str]=value
            ## if there is no type in stats functions
            else:
                key_str = key
                value = self.stats_functions[key]['f'](**self.params)
                if isinstance(value,collections.abc.Iterable):
                    for j in range(len(value)):
                        description = self.stats_functions[key]['levels'][j]
                        stats['mg'][key_str+'_'+description]=value[j]
                else:
                    stats['mg'][key_str]=value

        
        ## calculate mmol for each key in mg stats
        for key in stats['mg'].keys():
            if key in self.stats_functions.keys():
                mmol = self.stats_functions[key]['mmol']
                stats['mmol'][key]=mmol*stats['mg'][key]
            else:
                try:
                    mkey = key.split("_")[0]
                    mmol = self.stats_functions[mkey]['mmol']
                    stats['mmol'][key]=mmol*stats['mg'][key]
                except:
                    mkey = key.split("_")[0] + "_" + key.split("_")[1]
                    mmol = self.stats_functions[mkey]['mmol']
                    stats['mmol'][key]=mmol*stats['mg'][key]

        self.params['type']=_type_
        if name == "":
            self.stats = stats
            return stats
        else:
            return pd.DataFrame(stats[units].values(),
                                index=stats[units].keys(),
                                columns=[name])
    
    def stats_by_day(self,units='mg'):
        if self.daily_stats is None:
            res = pd.DataFrame()
            data = self.data
            self.set_params({'time_type':'daily'})
            for day in self.days:
                glucose = data[data['day']==day]['glucose']
                res = pd.concat([res,self.calculate_stats(glucose,name=day,units=units)],axis=1)
            self.daily_stats = res
        else:
            res = self.daily_stats
        return res
    
    def overall_stats_dataframe(self,units='mg'):
        index = []
        vals = []
        for k,v in self.stats[units].items():
            index.append(k)
            try:
                vals.append(round(v,3))
            except:
                try:
                    vals.append([round(e,3) for e in v])
                except:
                    vals.append(v)
                    
        display = pd.DataFrame(vals,index=index,columns=[self.filename]).T
        display['total_time']=pd.to_timedelta(display['total_time'])
        return display
            
    
    ##############################################################################
    ############### Graphics Functions ###########################################
    ##############################################################################
    
    
    def plot_all(self):
        """
        plot_all is a summary of all of the days in the file.
        """
        stats = self.stats['mg']
        plt.style.use('ggplot')
        df = self.df.copy()
        def convert_to_time(minutes):
            time = []
            for m in minutes:
                hh = m//60
                mm = m%60
                time.append(f'{hh:0>2}:{mm:0>2}')
            return time
        alpha1 = 0.95
        alpha2 = 0.75

        means = df.mean(axis=0)
        stds = df.std(axis=0)

        x = np.array(df.columns)
        x_labels = convert_to_time(x)
        
        plotdata = pd.DataFrame()
        plotdata['mean']=means
        plotdata['std']=stds
        plotdata['dof']=(~df.isna()).sum(axis=0)
        plotdata['t1'] = t.ppf(alpha1,plotdata['dof'],0,1)
        plotdata['t2'] = t.ppf(alpha2,plotdata['dof'],0,1)
        plotdata['low1']=plotdata['mean']-plotdata['t1']*plotdata['std']
        plotdata['low2']=plotdata['mean']-plotdata['t2']*plotdata['std']
        plotdata['high2']=plotdata['mean']+plotdata['t2']*plotdata['std']
        plotdata['high1']=plotdata['mean']+plotdata['t1']*plotdata['std']

        cols_to_plot = ['mean','low1','low2','high1','high2']
        data = plotdata[cols_to_plot].copy()
        datalow1=np.array(data['low1'].values,dtype=float)
        datahigh1=np.array(data['high1'].values,dtype=float)
        datalow2 = np.array(data['low2'].values,dtype=float)
        datahigh2=np.array(data['high2'].values,dtype=float)
        
        fig = plt.figure(figsize=(15,8))
        fig.subplots_adjust(wspace=0.1,hspace=0.4)
        ax = plt.subplot2grid((1,9),(0,1),colspan=8,rowspan=1)
        ax1 = plt.subplot2grid((1,9),(0,0),colspan=1,rowspan=1)
        ax.plot(x,data['mean'],color='black',lw=3,zorder=10)
        ax.plot(x,data['low1'],color='goldenrod',lw=1,ls='--',zorder=5)
        ax.plot(x,data['high1'],color='goldenrod',lw=1,ls='--',zorder=5)
        ax.fill_between(x,datalow1,datahigh1,color='goldenrod',alpha=0.1,zorder=5)
        ax.plot(x,data['low2'],color='cadetblue',lw=2,zorder=7)
        ax.plot(x,data['high2'],color='cadetblue',lw=2,zorder=7)
        ax.fill_between(x,datalow2,datahigh2,color='lightblue',zorder=7)
        low_bar = 70;high_bar=180
        
        ax.hlines([low_bar,high_bar],x[0],x[-1],color='red',lw=0.5)
        ax.set_xticks(ticks = x)
        ax.set_xticklabels(x_labels)
        ax.xaxis.set_major_locator(MultipleLocator(36))
        ax.axvspan(x[0],x[72],facecolor='0.3',alpha=0.5,zorder=-100)
        ax.axvspan(x[72],x[-1],facecolor='0.7',alpha=0.5,zorder=-100)
        ax.set_xlim(x[0],x[-1]+1)
        ax.yaxis.set_visible(False)
        ax.set_ylim()
        
        data = stats['tir'][1:-1]
        
        ax1.sharey(ax)
        
        ax1.bar(['TimeInRange'],[low_bar],color='firebrick',alpha=0.5)
        ax1.bar(['TimeInRange'],[high_bar-low_bar],bottom=[low_bar],color='green',alpha=0.5)
        ax1.bar(['TimeInRange'],[250-high_bar],bottom=[high_bar],color='firebrick',alpha=0.5)
        
        # xlim = ax1.get_xlim()
        ylim = ax1.get_ylim()
        
        heights = [(70+ylim[0])/2,(180+70)//2,(ylim[-1]+180)//2]
        
        ax1.annotate('Time in Range',xy=(-.37,ylim[-1]-15))
        for i,d in enumerate(data):
            
            ax1.annotate(f'{d*100:0.1f}%',xy=(-.2,heights[i]))
        ax.set_title(f"Ambulatory Glucose Profile: {self.days[0]} through {self.days[-1]}",
                    fontsize=16)
        
        return fig
    
    def ax_non1(self,ax):
        """
        Glucose Exposure Closeup
        """
        stats = self.stats['mg']
        xlim = ax.get_xlim()
        x_range = xlim[1]-xlim[0]
        ylim = ax.get_ylim()
        y_range = ylim[1]-ylim[0]
        y_mid = (ylim[0]+ylim[1])/2
        x_starts = np.linspace(xlim[0],xlim[1],5)
        ## 1st column #############################
        x_ = xlim[0]+0.02*x_range
        y_10 = y_mid+0.15*y_range
        y_11 = y_mid+0.05*y_range
        y_12 = y_mid-0.05*y_range
        y_13 = y_mid-0.15*y_range
        ax.annotate("Average",xy=(x_,y_10))
        ax.annotate("Daily",xy=(x_,y_11))
        ax.annotate("AUC",xy=(x_,y_12))
        ax.annotate("(mg/dL)*h",xy=(x_,y_13))
        
        ## 2nd Column ############################
        x_=x_starts[1]+0.02*x_range
        y_21 = ylim[1]-0.1*y_range
        y_22 = ylim[1]-0.2*y_range
        ax.annotate("Wake",xy=(x_+0.05*x_range,y_21))
        ax.annotate("6am-12am",xy=(x_,y_22))
        ax.annotate(f"{stats['auc_wake']}",xy=(x_,y_11),weight='bold',fontsize=15)
        ax.annotate("89-121 *",xy=(x_+0.03*x_range,y_13),fontsize=8)

        ## 3rd Column #############################
        x_=x_starts[2]+0.02*x_range
        ax.annotate("Sleep",xy=(x_+0.05*x_range,y_21))
        ax.annotate("12am-6am", xy=(x_,y_22))
        ax.annotate(f"{stats['auc_sleep']}",xy=(x_,y_11),weight='bold',fontsize=15)
        ax.annotate("85-109 *",xy=(x_+0.03*x_range,y_13),fontsize=8)
        
        ## 4th Column #############################
        x_=x_starts[3]+0.02*x_range
        ax.annotate("24 Hours",xy=(x_,y_21))
        ax.annotate(f"{stats['auc_all']}",xy=(x_,y_11),weight='bold',fontsize=15)
        ax.annotate("89-113 *",xy=(x_+0.03*x_range,y_13),fontsize=8)
        
        
        ### Bottom ################################
        x_ = xlim[0]+0.2*x_range
        y_ = ylim[0]+0.1*y_range
        ax.annotate("GLUCOSE EXPOSURE CLOSE-UP",xy=(x_,y_),fontsize=12)
        
        rect = patches.Rectangle((0,0),1,1,transform=ax.transAxes,alpha=0.5,
                                linewidth=5,edgecolor='black',facecolor='white')
        ax.add_patch(rect)
        return ax
    
    def ax_non2(self,ax):
        """
        IQR, GVP, MODD, HGBI, LGBI
        """
        stats = self.stats['mg']
        xlim = ax.get_xlim()
        x_range = xlim[1]-xlim[0]
        ylim = ax.get_ylim()
        y_range = ylim[1]-ylim[0]
        y_mid = (ylim[0]+ylim[1])/2
        x_starts = np.linspace(xlim[0],xlim[1],7)
        ## 1st column #############################
        x_ = xlim[0]+0.01*x_range
        y_11 = ylim[1]-0.1*y_range
        y_12 = ylim[1]-0.2*y_range
        y_13 = y_mid+0.05*y_range
        y_14 = y_mid-0.15*y_range
        ax.annotate("IQR",xy = (x_+0.01*x_range,y_11))
        ax.annotate("mg/dL",xy=(x_,y_12))
        ax.annotate(f"{stats['iqr']:0.1f}",xy=(x_,y_13),weight='bold',fontsize=15)
        ax.annotate("13-29 *",xy=(x_,y_14),fontsize=8)
        
        ## 2nd column ############################
        x_=x_starts[1]-0.07*x_range
        y_21 = y_mid+0.2*y_range
        y_22 = y_mid+0.1*y_range
        y_23 = y_mid-0.05*y_range
        ymin = ylim[0]+0.3*y_range
        ymax = ylim[1]-0.1*y_range
        ax.axvline(x=x_, ymin=ymin, ymax=ymax,color='black')
        x_+=0.07*x_range
        x_11 = x_-0.03*x_range
        ax.annotate("GVP",xy=(x_11,y_11))
        gvp = stats['gvp']
        ax.annotate(f'{gvp:0.2f}',xy=(x_11,y_12-0.02*y_range),
                    weight='bold',fontsize=15)
        ax.axhline(y=y_21,xmin=x_11,xmax=x_starts[2]-0.1*x_range, color='black')
        ax.annotate("MODD",xy=(x_11,y_22))
        modd = stats['modd_paper']
        ax.annotate(f'{modd:0.2f}',xy=(x_11,y_23),
                    weight='bold',fontsize=15)
        ## 3rd Column #############################
        x_=x_starts[2]-0.08*x_range 
        ax.axvline(x=x_, ymin=ymin, ymax=ymax,color='black')
        x_+=0.07*x_range
        x_11 = x_-0.03*x_range
        ax.annotate("HGBI",xy=(x_11,y_11))
        gbi = [stats['bgi_paper_low'],stats['bgi_paper_high']]
        ax.annotate(f'{gbi[1]:0.2f}',xy=(x_11,y_12-0.02*y_range),
                    weight='bold',fontsize=15)
        ax.axhline(y=y_21,xmin=x_11,xmax=x_starts[3]*x_range-0.11*x_range, color='black')
        ax.annotate("LGBI",xy=(x_11,y_22))
        
        ax.annotate(f'{gbi[0]:0.2f}',xy=(x_11,y_23),
                    weight='bold',fontsize=15)
        
        ## 4th Column #############################
        ## Grade ##
        y4 = np.linspace(ylim[1],ylim[0],11)
        ymin = ylim[0]+0.3*y_range
        ymax = ylim[1]-0.1*y_range
        x_=x_starts[3]-0.08*x_range 
        ax.axvline(x=x_, ymin=ymin, ymax=ymax,color='black')
        x_+=0.07*x_range
        x_11 = x_-0.03*x_range
        ax.annotate("GRADE",xy=(x_11,y4[1]))
        stat=[stats['grade_paper_overall'],stats['grade_paper_low'],
              stats['grade_paper_target'],stats['grade_paper_high']]
        ax.annotate(f'{stat[0]:0.2f}',xy=(x_11,y4[2]),
                    weight='bold',fontsize=15)
        ax.annotate("HYPER",xy=(x_11,y4[3]))
        ax.annotate(f'{stat[3]:0.2f}',xy=(x_11,y4[4]),
                    weight='bold',fontsize=15)
        ax.annotate("EU",xy=(x_11,y4[5]))
        ax.annotate(f'{stat[2]:0.2f}',xy=(x_11,y4[6]),
                    weight='bold',fontsize=15)
        ax.annotate("HYPO",xy=(x_11,y4[7]))
        ax.annotate(f'{stat[1]:0.2f}',xy=(x_11,y4[8]),
                    weight='bold',fontsize=15)
        
        ## 5th Column ###############################
        y5 = np.linspace(ylim[1],ylim[0],10)
        ymin = ylim[0]+0.3*y_range
        ymax = ylim[1]-0.1*y_range
        x_=x_starts[4]-0.08*x_range
        ax.axvline(x=x_, ymin=ymin, ymax=ymax,color='black')
        x_+=0.05*x_range
        x_11 = x_-0.03*x_range 
        stat = [stats['adrr_paper_overall'],stats['adrr_paper_low'],stats['adrr_paper_high']]
        ax.annotate("ADRR_TOTAL",xy=(x_11,y5[1]))
        ax.annotate(f'{stat[0]:0.2f}',xy=(x_11,y5[2]),
                    weight='bold',fontsize=15)
        ax.annotate("ADRR_HIGH",xy=(x_11,y5[3]))
        ax.annotate(f'{stat[2]:0.2f}',xy=(x_11,y5[4]),
                    weight='bold',fontsize=15)
        ax.annotate("ADRR_LOW",xy=(x_11,y5[5]))
        ax.annotate(f'{stat[1]:0.2f}',xy=(x_11,y5[6]),
                    weight='bold',fontsize=15)
        
        ## 6th Column ###############################
        y6 = np.linspace(ylim[0],ylim[1],100)
        stat = [stats['gri_gri'],stats['gri_hypo'],stats['gri_hyper']]
        ymin = ylim[0]+0.1*y_range
        ymax = ylim[1]-0.1*y_range
        x_=x_starts[5]-0.09*x_range
        ax.axvline(x=x_, ymin=ymin, ymax=ymax,color='black')
        x_+=0.05*x_range
        x_11 = x_-0.02*x_range  
        x_12 = x_-0.04*x_range
        
        ax.annotate("GLYCEMIA RISK INDEX",
                    xy=(x_11-0.01*x_range,y6[90]),
                    weight="bold",color="blue")
        ax.annotate("(GRI)",xy=(x_11,y6[84]),weight="bold",color="blue")
        colors=["green","darkgoldenrod","darkorange","tomato","firebrick"]
        idx = int(stat[0]//20)
        ax.annotate(f'{stat[0]:0.2f}',xy=(x_11,y6[73]),
                    weight='bold',fontsize=15,color=colors[idx])
        ax.annotate("[0-20]",xy=(x_12,y6[66]),fontsize=8,color="green")
        ax.annotate("[21-40]",xy=(x_12+0.04*x_range,y6[66]),
                    fontsize=8,color="darkgoldenrod")
        ax.annotate("[41-60]",xy=(x_12+0.09*x_range,y6[66]),
                    fontsize=8,color="darkorange")
        ax.annotate("[61-80]",xy=(x_12+0.14*x_range,y6[66]),
                    fontsize=8,color="tomato")
        ax.annotate("[>80]",xy=(x_12+0.19*x_range,y6[66]),
                    fontsize=8,color="firebrick")
        ax.annotate("HYPER COMP",xy=(x_11,y6[55]))
        ax.annotate(f'{stat[2]:0.2f}',xy=(x_11,y6[45]),
                    weight='bold',fontsize=15)
        ax.annotate("HYPO COMP",xy=(x_11,y6[30]))
        ax.annotate(f'{stat[1]:0.2f}',xy=(x_11,y6[20]),
                    weight='bold',fontsize=15)
        
        ##### Bottom ########################
        x_ = xlim[0]+0.2*x_range
        y_ = ylim[0]+0.1*y_range
        ax.annotate("VARIABILITY CLOSE-UP",xy=(x_,y_),fontsize=12)
        
        rect = patches.Rectangle((0,0),1,1,transform=ax.transAxes,alpha=0.5,
                                linewidth=5,edgecolor='black',facecolor='white')
        ax.add_patch(rect)
        return ax
    
    def ax_text(self,params):
        ax = params['ax']
        txt = params['txt']
        line_offset = params['line_offset']
        val_text_offset = params['val_text_offset']
        vals = params['vals']
        norms = params['norms']
        bottom = params['bottom']
        n = len(txt)
        m = len(bottom)
        
        xlim = ax.get_xlim()
        x_range = xlim[1]-xlim[0]
        x_starts = np.linspace(xlim[0],xlim[1],n+1)
        
        ylim = ax.get_ylim()
        y_range = ylim[1]-ylim[0]
        y_mid = (ylim[0]+ylim[1])/2
            
        for i in range(n):
            if i != n-1:
                x_ = (x_starts[i]+x_starts[i+1])/2+line_offset[i]*x_range
                if m==n:
                    ymin = 0.05*(y_range)+ylim[0]
                else:
                    ymin = 0.3*(y_range)+ylim[0]
                ax.axvline(x=x_,
                           ymin=ymin,
                           ymax=ylim[1]-0.1*(y_range),
                           color='black')
            x_ = x_starts[i]+0.05*x_range
            x_ += val_text_offset[i]*x_range
            # Values in middle of figure
            try:
                ax.annotate(f'{vals[i]:0.1f}',(x_,y_mid),weight='bold',fontsize=15)
            except:
                ax.annotate(str(vals[i]),(x_,y_mid),weight = 'bold',fontsize=15)
            ax.annotate(norms[i],xy=(x_,y_mid-0.2*y_range),fontsize=8)
            for j in range(len(txt[i])):
                ## text at the top of the figure
                x_ = x_starts[i]+0.05*x_range
                y_ = ylim[1]-0.1*y_range*j
                ax.annotate(txt[i][j],xy=(x_,y_),fontsize=10)
        
        ## Bottom 
        x_starts = np.linspace(xlim[0],xlim[1],m+1)
        for i in range(len(bottom)):
            x_ = x_starts[i]+0.05*x_range
            for j in range(len(bottom[i])):
                y_ = y_mid-(0.32+0.1*j)*y_range #ylim[0]+0.1*y_range
                if len(bottom[i])==1:
                    y_-=0.1
                ax.annotate(bottom[i][j],xy=(x_,y_))
            
                
        rect = patches.Rectangle((0,0),1,1,transform=ax.transAxes,alpha=0.5,
                                linewidth=5,edgecolor='black',facecolor='white')
        ax.add_patch(rect)
        return ax
    
    def init_plot_data(self):

        df = self.table
        means = df.mean(axis=0)
        medians = df.median(axis=0)
        stds = df.std(axis=0)

        ## 75th and 95th percentile lines
        alpha1 = 0.95
        alpha2 = 0.75

        plotdata = pd.DataFrame()
        plotdata['mean']=means
        plotdata['median']=medians
        plotdata['std']=stds
        plotdata['dof']=(~df.isna()).sum(axis=0)
        plotdata['t1'] = t.ppf(alpha1,plotdata['dof'],0,1)
        plotdata['t2'] = t.ppf(alpha2,plotdata['dof'],0,1)
        plotdata['low1']=plotdata['mean']-plotdata['t1']*plotdata['std']
        plotdata['low2']=plotdata['mean']-plotdata['t2']*plotdata['std']
        plotdata['high2']=plotdata['mean']+plotdata['t2']*plotdata['std']
        plotdata['high1']=plotdata['mean']+plotdata['t1']*plotdata['std']

        return plotdata

    def agp_plot_only(self):
        def convert_to_time(minutes):
            time = []
            for m in minutes:
                hh = m//60
                mm = m%60
                time.append(f'{hh:0>2}:{mm:0>2}')
            return time
        x = np.array(self.table.columns)
        x_labels = convert_to_time(x)
        cols_to_plot = ['median','low1','low2','high1','high2']
        data = self.plot_data[cols_to_plot].copy()
        datalow1=np.array(data['low1'].values,dtype=float)
        datahigh1=np.array(data['high1'].values,dtype=float)
        datalow2 = np.array(data['low2'].values,dtype=float)
        datahigh2=np.array(data['high2'].values,dtype=float)

        fig,ax = plt.subplots(figsize=(15,8))
        colors = ['black','firebrick','cadetblue','lightblue']
        ax.plot(x,data['median'],color=colors[0],lw=3,zorder=10)
        ax.plot(x,data['low1'],color=colors[1],lw=1,ls='--',zorder=5)
        ax.plot(x,data['high1'],color=colors[1],lw=1,ls='--',zorder=5)
        ax.fill_between(x,datalow1,datahigh1,color=colors[1],alpha=0.1,zorder=5)
        ax.plot(x,data['low2'],color=colors[2],lw=2,zorder=7)
        ax.plot(x,data['high2'],color=colors[2],lw=2,zorder=7)
        ax.fill_between(x,datalow2,datahigh2,color=colors[3],zorder=7)
        low_bar = 70;high_bar=180;
        other = [54,250,350]
        
        ax.hlines([low_bar,high_bar],x[0],x[-1],color='green',lw=2)
        ax.hlines(other,x[0],x[-1],color='black',lw=1)
        ax.set_xticks(ticks = x)
        ax.set_xticklabels(x_labels)
        ax.xaxis.set_major_locator(MultipleLocator(36))

        ## Dark/Light shading for night/day
        idx = int(360/self.deltat)
        ax.axvspan(x[0],x[idx],facecolor='0.3',alpha=0.5,zorder=-100)
        ax.axvspan(x[idx],x[-1],facecolor='0.7',alpha=0.5,zorder=-100)
        ax.set_xlim(x[0],x[-1]+1)

        txt = "____  Target Range _____"
        ax.text(-0.041,0.2,txt,transform=ax.transAxes,
                rotation=90,fontsize=12,color='green')
        ax.text(-0.02,0.19,"70",transform=ax.transAxes,
                fontsize=12,color='darkgreen')
        ax.text(-0.027,0.5,"180",transform=ax.transAxes,
                fontsize=12,color='darkgreen')
        ax.text(-0.02,0.14,"54",transform=ax.transAxes,
                fontsize=12,color='black')
        ax.text(-0.027,0.7,"250",transform=ax.transAxes,
                fontsize=12,color='black')
        ax.text(-0.027,0.98,"350",transform=ax.transAxes,
                fontsize=12,color='black')
        ax.yaxis.set_visible(False)
        ax.set_ylim(0,351)
        st.pyplot(fig)

    def plot_agp(self):
        """
        plot_agp - from 2017 International Consensus on use of CGM. This plot is trying to emulate
            Figure 1 in the paper.
            
        Input: Dexcom object
        
        Output: Figure 1
        """
        stats = self.stats['mg']
        self.set_params({'data':self.data['glucose']})
        
        plt.style.use('ggplot')
        
        def convert_to_time(minutes):
            time = []
            for m in minutes:
                hh = m//60
                mm = m%60
                time.append(f'{hh:0>2}:{mm:0>2}')
            return time
        
        x = np.array(self.table.columns)
        x_labels = convert_to_time(x)
        
        
        cols_to_plot = ['mean','median','low1','low2','high1','high2']
        data = self.plot_data[cols_to_plot].copy()
        datalow1=np.array(data['low1'].values,dtype=float)
        datahigh1=np.array(data['high1'].values,dtype=float)
        datalow2 = np.array(data['low2'].values,dtype=float)
        datahigh2=np.array(data['high2'].values,dtype=float)
        
        fig = plt.figure(figsize=(15,15))
        fig.subplots_adjust(wspace=0.01,hspace=0.01)
        
        ax = plt.subplot2grid((15,15),(7,2),colspan=13,rowspan=8)
        ax1 = plt.subplot2grid((15,15),(7,0),colspan=2,rowspan=8)
        
        ## Average Glucose / Glycemic Estimate
        vals = [stats['mean'],stats['gmi']]
        norms = ['88-116 *', "<6 *"]
        bottom = [['GLUCOSE EXPOSURE']]
        ax2 = plt.subplot2grid((15,15),(0,0),colspan=3,rowspan=3)
        ax2.set_axis_off()
        ax2=self.ax_text({'ax':ax2,
                          'line_offset':[0.2], #offset or None
                          'txt':[["","Avg","Glucose"],["","Glycemic","Estimate"]],
                          'val_text_offset':[0.08,0.1],
                          'vals':vals,
                          'norms':norms,
                          'bottom':bottom})
        
        ## Time In Range Charts
        tir = np.array(self.stats_search('tir'))
        vals = tir*100
        
        norms = ['0 *','<4 *','>90 *','<6 *','0 *']
        bottom = [["Level 2"],["Level 1"], ["GLUCOSE", "RANGES"], ["Level 1"],["Level 2"]]
        ax3 = plt.subplot2grid((15,15),(0,3),colspan=7,rowspan=3)
        ax3.set_axis_off()
        ax3=self.ax_text({'ax':ax3,
                      'line_offset':[0.1,0.1,0.14,0.13], #offset or None
                      'txt':[["","Very Low","Below 54","mg/dL"],
                             ["","Low Alert","Below 70","mg/dL"],
                             ["","In Target","Range 70-180","mg/dL"],
                             ["","High Alert","Above 180","mg/dL"],
                             ["","Very High","Above 250","mg/dL"]],
                          'val_text_offset':[0.05,0.05,0.05,0.05,0.05],
                          'norms':norms,
                          'vals':vals,
                          'bottom':bottom})
        
        ## Coefficient of Variation / Std
        vals = [stats['cv']*100,stats['std']]
        norms = ['19.25 *', '10-26 *']
        bottom = [["GLUCOSE VARIABILITY"]]
        ax4 = plt.subplot2grid((15,15),(0,10),colspan=3,rowspan=3)
        ax4.set_axis_off()
        ax4=self.ax_text({'ax':ax4,
                          'line_offset':[0.2], #offset or None
                          'txt':[["","Coefficient","of Variation"],
                                 ["","SD","mg/dL"]],
                          'val_text_offset':[0.1,0.1],
                          'vals':vals,
                          'norms':norms,
                          'bottom':bottom})
        
        ## % Time CGM Active
        vals = [stats['percent_active']*100]
        norms = [""]
        bottom = [['DATA','SUFFICIENCY']]
        ax5 = plt.subplot2grid((15,15),(0,13),colspan=2,rowspan=3)
        ax5.set_axis_off()
        ax5=self.ax_text({'ax':ax5,
                          'line_offset':[0.2], #offset or None
                          'txt':[["","% Time CGM","Active"]],
                          'val_text_offset':[0.2],
                          'vals':vals,
                          'norms':norms,
                          'bottom':bottom})
        
        ## Glucose Exposure Closeup - ax_non1
        ax6 = plt.subplot2grid((15,15),(3,0),colspan=5,rowspan=3)
        ax6.set_axis_off()
        ax6 = self.ax_non1(ax6)
        
        ## Variability Closeup - ax_non2
        ax7 = plt.subplot2grid((15,15),(3,5),colspan=10,rowspan=3)
        ax7.set_axis_off()
        ax7 = self.ax_non2(ax7)
        
        
        #colors - [50% color, 90% color, 75%color outer, 75% color inner]
        colors = ['darkgoldenrod','firebrick','cadetblue','lightblue']
        ax.plot(x,data['median'],color=colors[0],lw=3,zorder=10)
        ax.plot(x,data['low1'],color=colors[1],lw=1,ls='--',zorder=5)
        ax.plot(x,data['high1'],color=colors[1],lw=1,ls='--',zorder=5)
        ax.fill_between(x,datalow1,datahigh1,color=colors[1],alpha=0.1,zorder=5)
        ax.plot(x,data['low2'],color=colors[2],lw=2,zorder=7)
        ax.plot(x,data['high2'],color=colors[2],lw=2,zorder=7)
        ax.fill_between(x,datalow2,datahigh2,color=colors[3],zorder=7)
        low_bar = 70;high_bar=180
        
        ax.hlines([low_bar,high_bar],x[0],x[-1],color='red',lw=0.5)
        ax.set_xticks(ticks = x)
        ax.set_xticklabels(x_labels)
        ax.xaxis.set_major_locator(MultipleLocator(36))
        ax.axvspan(x[0],x[72],facecolor='0.3',alpha=0.5,zorder=-100)
        ax.axvspan(x[72],x[-1],facecolor='0.7',alpha=0.5,zorder=-100)
        ax.set_xlim(x[0],x[-1]+1)
        ax.yaxis.set_visible(False)
        ax.set_ylim()
        
        data = [tir[0]+tir[1],tir[2],tir[3]+tir[4]]
        
        ax1.sharey(ax)
        
        ax1.bar(['TimeInRange'],[low_bar],color='firebrick',alpha=0.5)
        ax1.bar(['TimeInRange'],[high_bar-low_bar],bottom=[low_bar],color='green',alpha=0.5)
        ax1.bar(['TimeInRange'],[250-high_bar],bottom=[high_bar],color='firebrick',alpha=0.5)
        
        # xlim = ax1.get_xlim()

        ax1.set_ylim(0,250)
        ylim = ax1.get_ylim()
        
        heights = [(70+ylim[0])/2,(180+70)//2,(ylim[-1]+180)//2]
        
        ax1.annotate('Time in Range',xy=(-.38,ylim[-1]-10),fontsize=12,weight='bold')
        for i,d in enumerate(data):
            
            ax1.annotate(f'{d*100:0.1f}%',xy=(-.2,heights[i]),fontsize=15)
        title_ = f"Ambulatory Glucose Profile:  {self.df.index[0]} through {self.df.index[-1]} "
        title_ += f"(Total Days = {len(self.days)})"
        ax.set_title(title_,
                    fontsize=16)
        return fig
    
    def plot_words(self,ax,text,**kwargs):
        ax.text("Here")
    
    def plot_agp_report(self):
        """
        Bar Chart with labels.
        This is the graph displayed by multiple_CGM.agp_report, `Time in Ranges`.
        It is a bar chart with colors representing the different time
        in ranges for the file being observed.
        """
        def convert_time(time):
            res = "("
            if time > 60:
                hrs = time//60
                mins = time-hrs*60
                res += f'{hrs:1.0f} hours {mins:2.0f} minutes'
            else:
                res += f'{time:1.0f} minutes'
            res+=")"
            return res
        tir = self.stats_search('tir',units = 'mg')
        fig=plt.figure(frameon=False,figsize=(7,5))

        ax = fig.add_subplot(111)
        ax.set_axis_off()
    
        ax.bar(0,height=54-40,width=10,color = 'firebrick',bottom = 40)
        ax.bar(0,height=70-56,width=10,color = 'red',bottom = 56)
        ax.bar(0,height=180-72,width=10,color='green',bottom=72)
        ax.bar(0,height=250-182,width=10,color='yellow',bottom = 182)
        ax.bar(0,height=300-252,width = 10,color='orange',bottom = 252)


        vals = [40,54,70,180,250,300]
        
        text_vals = [54,70,]
        lt = 5; btm = 40
        bg_font=10;sm_font=8
        mid=[];minutes = []

        ## Step 1 - boundaries, %, and times, create formatting variables mid
        for i in range(5):
            mid.append((vals[i+1]+vals[i])/2-5)
            minutes.append(convert_time(tir[i]*1440))
            ax.annotate(f'{tir[i]*100:3.0f}%',fontsize=bg_font,xy=(lt+20,mid[i]))
            ax.annotate(minutes[i],xy=(28,mid[i]+1),fontsize=sm_font)
            if i != 0:
                ax.annotate(f'{vals[i]:3.0f}',xy=(-6.5,vals[i]),fontsize=7)
        ## Step 2 - bottom row
        i=0
        ax.annotate("Very Low",xy=(7,mid[i]),fontsize=bg_font)
        ax.annotate("(<54 mg/dL)",xy=(12,mid[i]+1),fontsize=sm_font)
        ax.annotate("."*28,xy=(17.5,mid[i]+1),fontsize=sm_font)
        
        ## Second row
        i=1
        ax.annotate("Low",xy=(7,mid[i]),fontsize=bg_font)
        ax.annotate("(54-69 mg/dL)",xy=(9.25,mid[i]+1),fontsize=sm_font)
        ax.annotate("."*36,xy=(15.5,mid[i]+1),fontsize=sm_font)       

        i=2
        ax.annotate("Target Range",xy=(7,mid[i]),fontsize=bg_font)
        ax.annotate("(70-180 mg/dL)",xy=(14.25,mid[i]+1),fontsize=sm_font)
        ax.annotate("."*14,xy=(21,mid[i]+1),fontsize=sm_font)

        i=3
        ax.annotate("High",xy=(7,mid[i]),fontsize=bg_font)
        ax.annotate("(181-250 mg/dL)",xy=(9.5,mid[i]+1),fontsize=sm_font)
        ax.annotate("."*28,xy=(17.5,mid[i]+1),fontsize=sm_font)

        i=4
        ax.annotate("Very High",xy=(7,mid[i]),fontsize=bg_font)
        ax.annotate("(>250 mg/dL)",xy=(12.25,mid[i]+1),fontsize=sm_font)
        ax.annotate("."*24,xy=(18.5,mid[i]+1),fontsize=sm_font)

        plt.xlim(-8,35)
        plt.ylim(20,300)
        st.pyplot (fig)
    

    def plot_daily_trace(self,data,date_num,ax):
        """
        data - a days worth of data, index is a datatime object
        """
        data = data[~data.isnull().values]
        
        #fig=plt.figure(frameon=False,figsize=(7,5))
        #ax = fig.add_subplot(111)
        font_tiny=6
        ax.set_axis_off()
        x1 = np.arange(0,1440,self.time_delta)#data.index
        x2 = data.index
        ax.plot(data.index,data.values)
        ax.set_xlim(0,1435)
        ax.set_ylim(0,300)
        ax.hlines(y=[70,180],xmin=x1[0],xmax=x1[-1],color='black')

        ax.annotate("12pm",xy=(1440/2-200,30),fontsize=font_tiny)
        ax.annotate(date_num,xy=(100,250),fontsize=font_tiny)
        
        ax.vlines(x=1440/2,ymin=10,ymax=20,color='black')
        ax.vlines(x=[0,1435],ymin=10,ymax=300,color='black')
        ax.hlines(y=[10,300],xmin=0,xmax=1435,color='black')
        ax.vlines(x=1440/2,ymin=290,ymax=300,color='black')
        ax.fill_between(x=x1,y1=[70]*len(x1),y2=[180]*len(x1),color='gray',alpha=0.4)
        y1 = [180]*len(x2); y2=data.values
        ax.fill_between(x=x2,y1=y1,y2=y2,where = y2>y1,color='yellow')
        y1 = [70]*len(x2); y2=data.values
        ax.fill_between(x=x2,y1=y1,y2=y2,where=y2<y1, color='red')

        return ax

    def plot_daily_traces(self):
        def num_rows(N,idx):
            calendar = []
            first_row = 1
            tot_rem = N-(7-idx)
            complete_rows = tot_rem//7
            tot_rem -= 7*complete_rows
            final_row = 1 * (tot_rem>0)
            nrows = first_row+complete_rows+final_row
            calendar = []
            total=N
            for i in range(idx,7):
                calendar.append([0,i])
                total-=1
            j=1
            while total > 0:
                for i in range(7):
                    calendar.append([j,i])
                    total-=1
                    if total == 0:
                        break
                j+=1
            return nrows,calendar
        days_df=self.table.copy()
        days = days_df.index
        N = len(days)
        idx_day = days[0].weekday()
        rows,calendar = num_rows(N,idx_day)
        rows = max(2,rows)       
        fig, axs = plt.subplots(nrows=rows,ncols=7,figsize=(7,rows),frameon=False)
        fig.subplots_adjust(wspace=0.02,hspace=0.05)
        for i in range(rows):
            for j in range(7):
                axs[i,j].set_axis_off()

        cols = ['Monday','Tuesday','Wednesday','Thursday',
                'Friday','Saturday','Sunday']
        for ax,col in zip(axs[0],cols):
            ax.set_title(col,size='small')
        for i,day in enumerate(days):
            df_ = days_df.loc[day]
            dt = day.strftime("%d")
            x = calendar[i][0]
            y = calendar[i][1]

            axs[x,y]=self.plot_daily_trace(df_,dt, axs[x,y])

        st.pyplot(fig)

        

    def plot_agp_report_stats(self):
        stats=self.stats['mg']
        stat1=stats['total_time']
        stat2=stats['percent_active']*100
        stat3=stats['mean']
        stat4=stats['median']
        stat5=stats['gmi']
        stat6=stats['cv']*100
        start_time = self.periods[0][0]
        start_time = start_time.strftime("%d %b %Y")
        end_time = self.periods[-1][1]
        end_time = end_time.strftime("%d %b %Y")
        start_end = start_time+" - "+end_time
        body = f"#### Dates: {start_end} ({stat1})"
        st.markdown(body)
        body="|:blue[Statistic]|:blue[Value] |:blue[Target Value]| \n"
        body+="|---|---|---| \n"
        body+=f"|*% Time CGM is Active*|{stat2:3.1f}%|>70%| \n"
        body+=f"|*Average Glucose*|{stat3:3.1f}|88-116| \n"
        body+=f"|*Median Glucose*|{stat4:3.1f}|<None>| \n"
        body+=f"|*Glucose Management Indicator (GMI)*|{stat5:3.1f}|<6| \n"
        body+=f"|*Glucose Variability (%CV)*|{stat6:3.1f}%*|<36%|"
        st.markdown(body)
        st.markdown("*Defined as percent coefficient of variation (%CV).")
        st.divider()
        body=f"|:blue[Glucose Ranges]|:blue[Targets in Percentage]|:blue[Targets in Time per Day]|\n"
        body+="|---|---|---| \n"
        body+="|Target Range 70-180 mg/dL|Greater than 70%| Greater than 16h 28min| \n"
        body+="|Below 70 mg/dL|Less than 4%|Less than 58min| \n"
        body+="|Below 54 mg/dL|Less than 1%|Less than 14min| \n"
        body+="|Above 180 mg/dL|Less than 25%|Less than 6h| \n"
        body+="|Above 250 mg/dL|Less than 5%|Less than 1h 12min| \n"
        st.markdown(body)
        body=""
        body+="Each 5% increase in time in range (70-180 mg/dL) is "
        body+="clinically beneficial."
        st.markdown(body)
        st.divider()
        body=""
    
    def time_in_range_report(self):
        """ 
        Based on Andrew Koutnik recommendations
        """
        stats = self.stats['mg']
        def header_row(curr,items,col_span = None):
            if col_span is None:
                col_span = [1]*len(items)
            curr+='<tr>'
            for i in range(len(items)):
                cs = col_span[i]
                if cs>1:
                    curr+=f'<th colspan="{cs}">'+str(items[i])+'</th>'
                else:
                    curr+='<th>'+str(items[i])+'</th>'
            curr+='</tr>'
            return curr
        
        def normal_row(curr,items,col_span=None):
            if col_span is None:
                col_span = [1]*len(items)
            curr+='<tr>'
            for i in range(len(items)):
                cs = col_span[i]
                if cs>1:
                    curr+=f'<td colspan="{cs}">'+str(items[i])+'</td>'
                else:
                    curr+='<td>'+str(items[i])+'</td>'
            curr+='</tr>'
            return curr
        
        def convert_time(hours):
            hours_ = int(hours)
            min_ = int((hours-hours_)*60)
            return f"{hours_} hrs {min_} mins"

        st.markdown(":blue-background[Stats]")
        body= '<style> table,th,td {border:1px solid black;border-collapse:collapse;text-align:center;padding:3px}</style>'
        body+='<table>'

        items = ["Statistic","Value","Normal Range"]
        body = header_row(body,items)
        ## stats is the 
        stats_ = ['mean','median','iqr','cv','auc_all','auc_wake',
                 'auc_sleep','auc_100_all+','auc_100_wake+','auc_100_sleep+']
        for stat in stats_:
            items = [stat,round(stats[stat],2),None]
            body = normal_row(body,items)
        
        body += "</table>"
        st.html(body)

        st.markdown(":blue-background[Time In Range]")
        tir = self.time_in_range
        body= '<style> table,th,td {border:1px solid black;border-collapse:collapse;text-align:center;padding:3px}</style>'
        body+='<table>'
        
        items = ['Range','% Time', 'Total Minutes','Average Hrs-Mins per Day']
        col_span = [1,3,3,3]
        body = header_row(body,items,col_span)
        
        items = ['','All','Awake','Sleep',"All",'Awake','Sleep','All','Awake','Sleep']
        body = header_row(body,items)

        rows = []
        for rng in tir.keys():
            row=[]
            row.append(rng)
            for elem in tir[rng].keys():
                row.append(tir[rng][elem])
            rows.append(row)
        total_days = len(self.days)            
        for row in rows:
            items[0]=row[0]
            items[1]=round(row[2]*100,1)
            items[2]=round(row[4]*100,1)
            items[3]=round(row[6]*100,1)
            items[4]=row[1]
            items[5]=row[3]
            items[6]=row[5]
            items[7]=convert_time(row[1]/60/total_days)
            items[8]=convert_time(row[3]/60/total_days)
            items[9]=convert_time(row[5]/60/total_days)

            body = normal_row(body,items)
        
        body += "</table>"
        st.html(body)

    
    def plot_gri(self):
        """
        plot the glycemic risk index on a chart from the paper.
        """
        points = glycemic_risk_index(**self.params)
        zones = np.array([0,20,40,60,80,100])
        zones_=['A','B','C','D','E']
        
        pt0 = points[0]
        idx = np.where(pt0>=zones)[0][-1]
        zone = zones_[idx]
        
        x = np.linspace(0,30,1000)
        fa = lambda x:(20-3*x)/1.6
        fb = lambda x:(40-3*x)/1.6
        fc = lambda x:(60-3*x)/1.6
        fd = lambda x:(80-3*x)/1.6
        fe = lambda x:(100-3*x)/1.6
        fig,ax = plt.subplots(figsize=(12,12))
        ya = fa(x)
        yb = fb(x)
        yc = fc(x)
        yd = fd(x)
        ye = fe(x)
        ax.plot(x,ya,color="green")
        ax.fill_between(x,0,ya,color='green',alpha=0.3)
        ax.plot(x,yb,color='yellow')
        ax.fill_between(x,ya,yb,color='yellow',alpha=0.3)
        ax.plot(x,yc,color='orange')
        ax.fill_between(x,yb,yc,color='orange',alpha=0.3)
        ax.plot(x,yd,color='orangered')
        ax.fill_between(x,yc,yd,color='orangered',alpha=0.4)
        ax.plot(x,ye,color='darkred')
        ax.fill_between(x,yd,ye,color='darkred',alpha=0.3)
        ax.set_xlim(0,30)
        ax.set_ylim(0,60)
        ax.set_xlabel("Hypoglycemia Component (%)")
        ax.set_ylabel("Hyperglycemia Component (%)")
        
        ax.scatter(points[1],points[2],s=50,color = 'black',marker = 'o')
        ax.annotate(zone,xy=(points[1]+0.5,points[2]+0.5))
        return (fig)
    
    def poincare_plot(self,shift_minutes = 5):
        plt.style.use('ggplot')
        fig,ax = plt.subplots(figsize=(6,6))
        data = self.data['glucose'].copy()
        X_shift=data.shift(-shift_minutes//self.deltat)
        X_shift.rename('shift',inplace=True)
        X_new = pd.concat([data,X_shift],axis=1).dropna()
        X = X_new.values
        cov = np.cov(X.T)
        eigenvals, eigenvecs = np.linalg.eig(cov)
        theta = np.linspace(0,2*np.pi, 1000)
        ellipsis = (np.sqrt(eigenvals[None,:])*eigenvecs) @ [np.sin(theta),np.cos(theta)]
        long_axis,short_axis= 2*np.sqrt(eigenvals)
        wider_spread = max(long_axis,short_axis)
        smaller_spread = min(long_axis,short_axis)


        ellipsis += X.mean(axis=0).reshape(ellipsis.shape[0],1)
        ax.set_title("Poincare Plot",fontsize=30)
        ax.set_xlim(0,400);ax.set_ylim(0,400)
        ax.set_xlabel(r"blood glucose $BG(t_{i-1})$ mg/dL")
        ax.set_ylabel(r"blood glucose $BG(t_{i})$ mg/dL")
        ax.scatter(X[:,0],X[:,1],s=5)
        ax.plot(ellipsis[0,:],ellipsis[1,:],color='blue')
        body = f"Major axis = {wider_spread:0.1f}"
        ax.annotate(body,xy=(5,25),fontsize=15,color='blue',weight='bold')
        body = f"Minor axis = {smaller_spread:0.1f}"
        ax.annotate(body,xy=(5,5),fontsize=15,color='blue',weight='bold')
        return (fig)
    
    def time_series_plot(self):
        plt.style.use('ggplot')
        n = len(self.periods)
        data1 = self.data['glucose'].copy()
        if n>1:
            fig,ax = plt.subplots(nrows=n,ncols=1,figsize=(8,int(n*4)))
            fig.subplots_adjust(hspace=0.45)
            for i, period in enumerate(self.periods):
                idx = np.arange(period[0],period[-1],timedelta(minutes=self.deltat))
                ax[i].plot(idx,data1.loc[idx],zorder=1)
                formatter = mdates.DateFormatter('%d|%H:%M')
                ax[i].xaxis.set_major_formatter(formatter)
                ax[i].set_title("Time Series: "+str(period[0].date())+
                                " through " +str(period[1].date()) )
                ax[i].tick_params(axis='x',which='major',labelsize=8,
                                  labelrotation=90)

        else:
            period = self.periods[0]
            fig,ax = plt.subplots(nrows=n,ncols=1,figsize=(8,4))
            idx = data1.index
            ax.plot(idx,data1,zorder=1)
            formatter = mdates.DateFormatter('%m-%d')
            ax.xaxis.set_major_formatter(formatter)
            ax.set_title("Time Series: "+str(period[0].date())+
                                " through " +str(period[1].date()))
        return(fig)
    
    def markov_chain_calculation(self,intervals):
        data = self.data['glucose'].copy()
        fig,ax = plt.subplots(figsize=(5,1))
        name = 'intervals'
        val = intervals[0]
        ax.barh(name,val,color="firebrick")
        val = intervals[1] - intervals[0]
        ax.annotate("State1",xy=((intervals[1])/2-5,-0.2),
                    fontsize=10,rotation = 90)
        ax.barh(name,val,left = intervals[0],color='gold')
        val = intervals[2] - intervals[1]
        ax.annotate("State2",xy=((intervals[0]+intervals[1])/2-5,-0.2),
                    fontsize=10,rotation = 90)
        ax.barh(name,val,left = intervals[1],color='green')
        val = intervals[3]-intervals[2]
        ax.annotate("State3",xy=((intervals[1]+intervals[2])/2-5,-0.2),
                    fontsize=10,rotation = 90)
        ax.barh(name,val,left=intervals[2],color='gold')
        val = 350-intervals[3]
        ax.annotate("State4",xy=((intervals[2]+intervals[3])/2-5,-0.2),
                    fontsize=10,rotation = 90)
        ax.barh(name,val,left=intervals[3],color='firebrick')
        ax.annotate("State5",xy=((intervals[3]+350)/2-5,-0.2),
                    fontsize=10,rotation = 90)
        ax.set_xlim(0,350)
        ax.set_xticks(intervals,[str(i) for i in intervals])
        st.pyplot(fig)
        tm,pi_star,er = transition_matrix(data,intervals,5,5)
        tm = pd.DataFrame(tm)
        cols = [f"x<{intervals[0]}",f"{intervals[0]}<x<{intervals[1]}",
                      f"{intervals[1]}<x<{intervals[2]}",f"{intervals[2]}<x<{intervals[3]}",
                      f"x>{intervals[3]}"]
        tm.columns=cols
        tm.index = cols
        st.write(tm)
        st.write("Time spent in states.")
        pi_star = pd.DataFrame(pi_star)
        pi_star.index = cols
        pi_star.columns=["long-term fraction of time"]
        st.write(pi_star)
        st.write("Entropy Rate of Markov Chain:")
        st.write(np.array(er).sum())


        