import collections.abc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t,entropy
#import streamlit as st
import collections

from datetime import datetime,timedelta
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as patches
import matplotlib.dates as mdates


from ..functions import (time_in_range,glucose_N,glucose_mean,
                         glucose_median,total_time,
                        glucose_std,glucose_cv,mean_absolute_glucose,
                        j_index,low_high_blood_glucose_index,
                        glycemic_risk_assessment_diabetes_equation,
                        glycemic_variability_percentage,
                        lability_index,mean_of_daily_differences,
                        conga,m_value,total_hours,
                        glucose_management_indicator,interquartile_range,
                        percent_active,mean_amplitude_of_glycemic_excursions,
                        glycemic_risk_index,auc,auc_thresh,
                        eccentricity,entropy_mc,auc_per_day,total_days,
                        tir_stats,transition_matrix,
                        cogi,adrr,
                        )


class CGM(object):
    def __init__(self,
                 filename,
                 file_df,
                 deltat = None,
                 max_break = 45,
                 nostats = False,
                 dt_fmt='%Y-%m-%d %H:%M:%S',
                 units='mg',
                 ):
        """
        CGM data object for storing the original data, creating a time grid
            for all observations to be linearly interpolated to. Calculates
            statistics that are in the stats_functions dictionary by type and
            level. 
            - Converts all data to mg/dL for calculations.
        Inputs: 
        filename - name of file that will be displayed on export or
            selection while viewing data.
            
        file_df - dataframe from csv raw data.

        Attributes:
            df - raw data from csv read data dropping NA values.
            data - raw data converted to grid datetime values.
            
        """
        self.params = {}
        self.filename = filename
        self.max_break = max_break
        self.date_format = dt_fmt
        self.df = file_df.dropna()
        ## get rid of the local timezone
        self.df.index = self.df.index.tz_localize(None)
        self.missing_values = None
        self.observations = len(self.df)


        ## Handle interval ###
        if deltat == None:
            self.util_time_delta()
        else:
            self.deltat = deltat
            self.time_delta = timedelta(minutes = deltat)
            
        self.raw_periods = self.build_periods(data=self.df.copy())
        ## Initial values
        
        self._units = units
        self.const = 1 if units == 'mg' else 18.018
        
        self.params['units']=units
        self.data = self.project_time_grid()
        self.plot_data = self.init_plot_data()

        self.assign_functions()
        self.stats_dict = {}
        if nostats:
            pass
        else:
            self.stats = self.calculate_stats()
            self.cohort_cols = list(self.stats.index)
        
    @property
    def units(self):
        return self._units
        
    @units.setter
    def units(self,val):
        keys = ['bins','thresh']
        if self.units == 'mmol' and val == 'mg':
            self.df = self.df*18.018
            self.table = self.table*18.018
            self._units = 'mg'
            self.params['units']=val
            self.const = 1
            for key in keys:
                self.params[key] = 18.018*self.params[key]
            if hasattr(self,'stats'):
                del self.stats
        elif self.units == 'mg' and val == 'mmol':
            self.df = self.df/18.018
            self.table = self.table/18.018
            self._units = 'mmol'
            self.params['units']=val
            self.const = 18.018
            for key in keys:
                self.params[key] = self.params[key]/18.018
            if hasattr(self,'stats'):
                del self.stats
        elif self.units == val:
            pass
        else:
            raise ValueError(f'Does not support {val} as units => mg or mmol.')
        self.plot_data = self.init_plot_data()
        self.project_time_grid()
            
    def build_periods(self,**kwargs):
        """
        create periods based on max_break, max number of minutes 
            indicating different wear periods
            
        data - the data that we want to break into periods based
            based on max_break period
        """
        deltat = self.deltat
        if 'data' in kwargs:
            data = kwargs['data']
        else:
            data = self.df.copy()
        if 'max_break' in kwargs:
            max_break = kwargs['max_break']
        else:
            max_break = self.max_break
        
        ts = data.index
        data['dates'] = pd.to_datetime(data.index)
        data['date_shift']=data['dates'].shift(-1)
        data['time_diff'] = (data['date_shift']-data['dates'])/timedelta(minutes=1)

        ## missing_values counted using self.df (original data)
        if self.missing_values is None:
            td = data[data.time_diff>deltat+1]
            self.missing_values = round((td[td['time_diff']<=max_break]['time_diff']/deltat)-1,0).sum()
            self.missing_values = int(self.missing_values)
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

    def util_time_delta(self,**kwargs):
        """
        Automatically figures out time delta - median over the entire spread of times.
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
    
    def time_grid(self,start,end,**kwargs):
        """
        creates the on-the-hour times between start and end
            This method handles all the rounding involved in creating the
            on-the-hour times for both start and end.
        """
        if 'deltat' in kwargs:
            deltat = kwargs['deltat']
            time_delta = timedelta(minutes=deltat)
        else:
            deltat = self.deltat
            time_delta = self.time_delta
        
        start_min = start.minute//deltat*deltat+deltat
        add_hour = 0
        if start_min == 60:
            start_min = 0
            add_hour += 1    
        start_date = start.replace(minute=start_min, second = 0)
        start_date = (start_date+timedelta(hours=add_hour)).strftime(self.date_format)
        
        
        end_min = end.minute//deltat*deltat 
        add_hour = 0
        if end_min == 60:
            end_min = 0
            add_hour += 1
        end_date = end.replace(minute = end_min,second=0)
        end_date = (end_date+timedelta(hours=add_hour)).strftime(self.date_format)
        
        dates = pd.date_range(start_date,end_date,freq=time_delta)
        
        return dates
        
    
    def project_time_grid(self,**kwargs):
        """
        based on the raw_periods, creates the data object which 
            contains a time grid for all of the data, periods
            Also creates the columns that will be used by the cgm
            metrics in order to speed up that process.
        """
        if 'deltat' in kwargs:
            deltat = kwargs['deltat']
        else:
            deltat = self.deltat
        data = pd.DataFrame()
        datetimes = []
        for period in self.raw_periods:
            pdates = self.time_grid(period[0],period[1],deltat = deltat)
            pdata = self.df.loc[period[0]:period[1]]
            datetimes += list(pdates)
            projection = np.interp(pdates,pdata.index,pdata.values.reshape(-1))
            pdata = pd.DataFrame(projection,index=pdates,columns=['glucose'])
            data = pd.concat([data,pdata])

        ## add columns to data that will be used in computations
        data['date']=data.index.map(lambda x: x.date())
        data['time']=data.index.map(lambda x: x.time())
        data['all']=1
        #data['stats_use']=1
        ## normalize data ##
        if self.units == 'mg':
            data['normalize']=data['glucose'].map(lambda x: 1.509*(np.log(x)**1.084-5.381))
            data['rl']=data['normalize'].map(lambda x: 10*x**2 if x<0 else 0)
            data['rh']=data['normalize'].map(lambda x: 10*x**2 if x>0 else 0)
        else:
            data['normalize']=data['glucose'].map(lambda x: 1.509*(np.log(x*18.018)**1.084-5.381))
            data['rl']=data['normalize'].map(lambda x: 10*x**2 if x<0 else 0)
            data['rh']=data['normalize'].map(lambda x: 10*x**2 if x>0 else 0)
        time06 = datetime(month=1,day=1,year=2025,hour=6,minute=0).time()
        time24 = datetime(month=1,day=1,year=2025,hour=23,minute=59).time()
        data['day']=data['time'].map(lambda x: 1 if time06<=x<=time24 else 0)
        self.data = data
        self.date_times = datetimes
        self.periods = self.build_periods(data=data.copy())
        self.table = data.pivot_table(values = 'glucose',index=['date'],columns=['time'])
        self.days = list(self.table.index)
        self.day_data, self.night_data = self.day_data_()
        periods = np.array([p[0] for p in self.periods])
        self.data['period']=self.data.index.map(lambda x: np.where(x>=periods)[0].max())

        return data
    
    def day_data_(self):
        """
        return 2 dataframes - first has day data, the second has night data
        """
        if not hasattr(self,'day_data'):
            day = pd.DataFrame()
            night = pd.DataFrame()
            for i in range(len(self.periods)):
                dd = self[i].day==1
                day = pd.concat([day,self[i][dd]])
                night = pd.concat([night,self[i][~dd]])
            self.day_data = day
            self.night_data = night
        else:
            return self.day_data,self.night_data
        return day,night
    
    def __len__(self):
        return len(self.periods)
    
    def __getitem__(self,key):
        if isinstance(key,int):
            if key<0 or key>=len(self):
                raise IndexError(f"Valid integers between 0 and {len(self)-1}")
            else:
                period = self.periods[key]
                return self.data.loc[period[0]:period[1]]
        elif isinstance(key,slice):
            d = pd.DataFrame()
            key0 = key.start; key1=key.stop
            if key0 is None:
                key0=0
            if key1 is None or key1 > len(self)-1:
                key1=len(self)
            for k in range(key0,key1):
                d = pd.concat([d,self[k]])
            return d
        
    def percent_active(self):
        return 1-self.missing_values/self.observations
    
    def episodes(self,glevel,direction = 'less',cl=1,tot_min=15):
        periods = self.periods
        event = {}

        for period in range(len(periods)):
            df = self[period].copy()
            if direction =='less':
                df['ineq'] = df['glucose']>=glevel
            else:
                df['ineq']= df['glucose']<=glevel
            df['last'] = df.index
            df['last'] = df.groupby('ineq')['last'].diff()-timedelta(minutes=5)
            df.loc[df['ineq']==False,'last'] = None
            temp = df[df['last']>=timedelta(minutes=tot_min)]
            for i in range(len(temp)):
                start = temp.index[i]-temp.iloc[i]['last']
                stop = temp.index[i]-timedelta(minutes=5)
                arr=df.loc[start:stop,'glucose'].values
                event[i]={}
                event[i]['datetime']=start.strftime('%Y-%m-%d %H:%M')
                event[i]['minutes']=len(arr)*5
                event[i]['mean'] = arr.mean()
                event[i]['level'] = cl
                event[i]['values']=np.round(arr,2)
        try:
            ans = pd.DataFrame(event).T.sort_values('datetime').set_index('datetime')
        except:
            ans = pd.DataFrame(event).T

        return ans
            
    def assign_functions(self):
        
        self.stats_functions = {}
        
        self.stats_functions['num_obs'] = {}
        self.stats_functions['num_obs']['f'] = glucose_N
        self.stats_functions['num_obs']['description']="number of observations"
        self.stats_functions['num_obs']['normal']=[]
        self.stats_functions['num_obs']['time_agg'] = ['all','date','period','day']
        self.stats_functions['num_obs']['levels'] = ['num_obs']
        

        self.stats_functions['total_time'] = {}
        self.stats_functions['total_time']['f'] = total_time
        self.stats_functions['total_time']['description']="total time"
        self.stats_functions['total_time']['normal']=[]
        self.stats_functions['total_time']['time_agg'] = ['all','date','period','day']
        self.stats_functions['total_time']['levels'] = ['total_time']

        self.stats_functions['total_hours'] = {}
        self.stats_functions['total_hours']['f'] = total_hours
        self.stats_functions['total_hours']['description']="total time"
        self.stats_functions['total_hours']['normal']=[]
        self.stats_functions['total_hours']['time_agg'] = []
        self.stats_functions['total_hours']['levels'] = ['total_time']

        self.stats_functions['total_days'] = {}
        self.stats_functions['total_days']['f'] = total_days
        self.stats_functions['total_days']['description']="total time"
        self.stats_functions['total_days']['normal']=[]
        self.stats_functions['total_days']['time_agg'] = []
        self.stats_functions['total_days']['levels'] = ['total_time']


        self.stats_functions['percent_active'] = {}
        self.stats_functions['percent_active']['f'] = percent_active
        self.stats_functions['percent_active']['description']="percent active"
        self.stats_functions['percent_active']['normal']=[]
        self.stats_functions['percent_active']['time_agg'] = ['all']
        self.stats_functions['percent_active']['levels']=['percent_active']
        
        self.stats_functions['mean']={}
        self.stats_functions['mean']['f'] = glucose_mean
        self.stats_functions['mean']['description']="mean"
        self.stats_functions['mean']['normal']=['88-116']
        self.stats_functions['mean']['time_agg'] = ['all','date','period','day']
        self.stats_functions['mean']['levels'] = ['mean']


        self.stats_functions['median']={}
        self.stats_functions['median']['f'] = glucose_median
        self.stats_functions['median']['description']="median"
        self.stats_functions['median']['normal']=[]
        self.stats_functions['median']['time_agg'] = ['all','date','period','day']
        self.stats_functions['median']['levels'] = ['median']
        
        self.stats_functions['std']={}
        self.stats_functions['std']['f'] = glucose_std
        self.stats_functions['std']['description']="std"
        self.stats_functions['std']['normal']=['<6'] 
        self.stats_functions['std']['time_agg'] = ['all','date','period','day']
        self.stats_functions['std']['levels'] = ['std']
        
        self.stats_functions['cv'] = {}
        self.stats_functions['cv']['f'] = glucose_cv
        self.stats_functions['cv']['description']="CV"
        self.stats_functions['cv']['normal']=['<0.36']  
        self.stats_functions['cv']['time_agg'] = ['all','date','period','day']
        self.stats_functions['cv']['levels'] = ['cv']
        
        self.stats_functions['mag']={}
        self.stats_functions['mag']['f']=mean_absolute_glucose
        self.stats_functions['mag']['description']="MAG"
        self.stats_functions['mag']['normal']=[] 
        self.stats_functions['mag']['time_agg'] = ['all','date','period','day']
        self.stats_functions['mag']['levels'] = ['mag']
        
        self.stats_functions['tir']={}
        self.stats_functions['tir']['f']=time_in_range
        self.stats_functions['tir']['description']="TIR"
        self.stats_functions['tir']['normal']=['<1%','<4%','>70%','<25%','<5%'] 
        self.stats_functions['tir']['time_agg'] = ['all','date','period','day']
        self.stats_functions['tir']['levels']=['<54','54-70','70-180','180-250','>250']
        
        self.stats_functions['j_index']={}
        self.stats_functions['j_index']['f']=j_index
        self.stats_functions['j_index']['description']='J_Index'
        self.stats_functions['j_index']['normal']=[] 
        self.stats_functions['j_index']['time_agg'] = ['all','date','period','day']
        self.stats_functions['j_index']['levels'] = ['j_index']
        
        self.stats_functions['bgi']={}
        self.stats_functions['bgi']['f']=low_high_blood_glucose_index
        self.stats_functions['bgi']['description'] = 'LBGI_HGBI'
        self.stats_functions['bgi']['type'] = ['paper','easy']
        self.stats_functions['bgi']['time_agg'] = ['all','date','period','day']
        self.stats_functions['bgi']['normal'] = {'LBGI':[],
                                                 'HBGI':[]} 
        self.stats_functions['bgi']['levels'] = ['(lbgi)','(hbgi)']

        self.stats_functions['grade']={}
        self.stats_functions['grade']['f']=glycemic_risk_assessment_diabetes_equation
        self.stats_functions['grade']['description']='GRADE'
        self.stats_functions['grade']['type'] = ['paper','easy']
        self.stats_functions['grade']['normal']=[]
        self.stats_functions['grade']['time_agg'] = ['all','date','period','day']
        self.stats_functions['grade']['levels']=['overall','low','target','high']
        
        self.stats_functions['gvp']={}
        self.stats_functions['gvp']['f']=glycemic_variability_percentage
        self.stats_functions['gvp']['description']='GVP'
        self.stats_functions['gvp']['normal'] = []
        self.stats_functions['gvp']['time_agg'] = ['all']
        self.stats_functions['gvp']['levels'] = ['gvp']
        
        self.stats_functions['li']={}
        self.stats_functions['li']['f']=lability_index
        self.stats_functions['li']['description']='Lability_Index'
        self.stats_functions['li']['normal']= []
        self.stats_functions['li']['time_agg'] = ['all']
        self.stats_functions['li']['levels'] = ['li']
        
        self.stats_functions['modd']={}
        self.stats_functions['modd']['f'] = mean_of_daily_differences
        self.stats_functions['modd']['description']='MODD'
        self.stats_functions['modd']['type'] = ['paper','easy']
        self.stats_functions['modd']['normal']=[]
        self.stats_functions['modd']['time_agg'] = ['all']
        self.stats_functions['modd']['levels'] = ['modd']

        
        self.stats_functions['adrr']={}
        self.stats_functions['adrr']['f'] = adrr
        self.stats_functions['adrr']['description']="ADRR"
        self.stats_functions['adrr']['type'] = ['paper','easy']
        self.stats_functions['adrr']['normal']=[] 
        self.stats_functions['adrr']['time_agg'] = ['all','date','period','day']
        self.stats_functions['adrr']['levels'] = ['overall','low','high']
        
        self.stats_functions['conga']={}
        self.stats_functions['conga']['f'] = conga
        self.stats_functions['conga']['description']='conga'
        self.stats_functions['conga']['type']=['paper','easy']
        self.stats_functions['conga']['normal']=[]
        self.stats_functions['conga']['time_agg'] = ['all'] 
        self.stats_functions['conga']['levels'] = ['conga']
        
        self.stats_functions['m_value']={}
        self.stats_functions['m_value']['f'] = m_value
        self.stats_functions['m_value']['description']='M_Value'
        self.stats_functions['m_value']['type'] = ['paper','easy']
        self.stats_functions['m_value']['normal']=[]
        self.stats_functions['m_value']['time_agg'] = ['all','date','period','day']  
        self.stats_functions['m_value']['levels'] = ['m_value']
        
        self.stats_functions['gmi']={}
        self.stats_functions['gmi']['f'] = glucose_management_indicator
        self.stats_functions['gmi']['description']='gmi'
        self.stats_functions['gmi']['normal']=[] 
        self.stats_functions['gmi']['time_agg'] = ['all','date','period','day']  
        self.stats_functions['gmi']['levels'] = ['gmi']
        
        self.stats_functions['iqr']={}
        self.stats_functions['iqr']['f'] = interquartile_range
        self.stats_functions['iqr']['description']='Inter-quartile range'
        self.stats_functions['iqr']['normal']=['13-29']   
        self.stats_functions['iqr']['time_agg'] = ['all','date','period','day']     
        self.stats_functions['iqr']['levels'] = ['iqr']
        
        self.stats_functions['auc']={}
        self.stats_functions['auc']['f']=auc
        self.stats_functions['auc']['description']='AUC'
        self.stats_functions['auc']['normal'] = []
        self.stats_functions['auc']['time_agg'] = ['all','date','period','day']  
        self.stats_functions['auc']['levels'] = ['auc']

        self.stats_functions['auc_rate'] = {}
        self.stats_functions['auc_rate']['f'] = auc_per_day
        self.stats_functions['auc_rate']['description']="AUC rate"
        self.stats_functions['auc_rate']['normal']=[]
        self.stats_functions['auc_rate']['time_agg'] = ['all','date','period','day']
        self.stats_functions['auc_rate']['levels'] = ['auc_rate']

        self.stats_functions['auc_100']={}
        self.stats_functions['auc_100']['f']=auc_thresh
        self.stats_functions['auc_100']['description']='AUC_100'
        self.stats_functions['auc_100']['normal'] = []
        self.stats_functions['auc_100']['time_agg'] = ['all','date','period','day']
        self.stats_functions['auc_100']['levels'] = ['auc_rate']
        
        self.stats_functions['mage']={}
        self.stats_functions['mage']['f']=mean_amplitude_of_glycemic_excursions
        self.stats_functions['mage']['description']='MAGE'
        self.stats_functions['mage']['normal'] = []
        self.stats_functions['mage']['time_agg'] = ['all']  
        self.stats_functions['mage']['levels'] = ['overall','decrease','increase']
        
        self.stats_functions['gri'] = {}
        self.stats_functions['gri']['f']=glycemic_risk_index
        self.stats_functions['gri']['description']='glycemic risk index'
        self.stats_functions['gri']['normal'] = [0,20,40,60,80,100]
        self.stats_functions['gri']['time_agg'] = ['all','date','period','day']  
        self.stats_functions['gri']['levels'] = ['overall','hypo','hyper']

        self.stats_functions['cogi'] = {}
        self.stats_functions['cogi']['f']=cogi
        self.stats_functions['cogi']['description']='COGI'
        self.stats_functions['cogi']['normal'] = [90,100]
        self.stats_functions['cogi']['time_agg'] = ['all','date','period','day']  
        self.stats_functions['cogi']['levels'] = ['cogi']

        self.stats_functions['eccentricity']={}
        self.stats_functions['eccentricity']['f']=eccentricity
        self.stats_functions['eccentricity']['description']='Poincare Eccentricity'
        self.stats_functions['eccentricity']['normal'] = []
        self.stats_functions['eccentricity']['time_agg'] = ['all','date','period','day'] 
        self.stats_functions['eccentricity']['levels']=['ecc','a','b']

        self.stats_functions['entropy']={}
        self.stats_functions['entropy']['f']=entropy_mc
        self.stats_functions['entropy']['description']='Markov Entropy'
        self.stats_functions['entropy']['normal'] = [] 
        self.stats_functions['entropy']['time_agg'] = ['all','date','period','day'] 
        self.stats_functions['entropy']['levels'] = ['entropy']
        
        ## global parameters
        self.params['bins']= np.array([0,54,70,180,250,350])/self.const
        self.params['thresh'] = 100/self.const
        self.params['deltat'] = self.deltat
        self.params['li_k'] = 60
        self.params['conga_h'] = 1
        self.params['m_index'] = 120
        self.params['days']=self.days
        self.params['periods']=self.periods
        self.params['type']='paper'
        self.params['percent_active']=self.percent_active()

        return None
    
    def calc_stat(self,stat_names,**kwargs):
        """
        method used to calculate specific stat given keyword arguments
            that may be different than the standard statistics, such as 
            time in range for different bins that the standard.
        Input: `stat_name` - a key from the stats_functions above.
               `kwargs` that need to be changed to run this version of 
                    of the stats function, see examples below.
        Output: stats functions results for specific kwargs 
            """
        if isinstance(stat_names,str):
            stat_names = [stat_names]
        else:
            stat_names = list(stat_names)
        d = pd.DataFrame()
        while len(stat_names)>0:
            stat_name = stat_names.pop(0)
            params = self.params.copy()
            levels = self.stats_functions[stat_name]['levels']
            func = self.stats_functions[stat_name]['f']
            if 'group_by' in kwargs:
                group_by = kwargs['group_by']
                if group_by == 'all':
                    idx_name = ['all']
                elif group_by == 'day':
                    idx_name = ['sleep','wake']
                elif group_by == 'period':
                    idx_name = [period[0].strftime("%Y%m%d")+'-'+period[1].strftime("%Y%m%d")+f'_{i}' 
                            for i,period in enumerate(self.periods)]
                elif group_by == 'date':
                    idx_name = [d.strftime("%Y%m%d") for d in self.days]
            else:
                group_by = 'all'
                idx_name = ['all']
            if 'bins' in kwargs and stat_name == 'tir': #specifically for time in range
                bins = kwargs['bins']
                levels = ['<'+str(bins[1])]
                levels += [str(bins[i])+'-'+str(bins[i+1]) for i in range(1,len(bins)-2)]
                levels += ['>'+str(bins[-2])]
            for k in kwargs:
                params[k]=kwargs[k]
            if 'type' in self.stats_functions[stat_name]:
                if len(levels)>1:
                    cols = [stat_name+'_'+params['type']+'_'+lvl for lvl in levels]
                else:
                    cols = [stat_name+'_'+params['type']]
            else:
                if len(levels)>1:
                    cols = [stat_name+'_'+lvl for lvl in levels]
                else:
                    cols = [stat_name]
            
            ans = self.data.groupby(list([group_by])).apply(func,**params,include_groups=False)
            idx = list(ans.index)
            data = np.stack(ans.values)
            if idx_name is not None:
                idx = idx_name
            d = pd.concat([d,pd.DataFrame(data,index=idx,columns = cols)],axis=1)
        return d

    def return_group_stats(self,key,group_by):
        """ 
        return_group_stats - given a key and group_by level ('all','day',
        'period','date') returns a single stat given by key with all of the
        types and levels associated with that stat.
        Input:
            key - the dictionary key from stats_functions that triggers the 
                function needed for the statistics.
            group_by - the time_agg level that the statistics will return.

        Output 
            single stat in dataframe with time_agg groups as rows and 
                stat levels and types as columns.

        Example: cgm.return_group_stats('adrr','day')  adrr has 3 levels and 2 types ... (cols)
                                                        day = 0 or 1 depending on day or night (rows)
    adrr_paper_overall 	adrr_paper_low 	adrr_paper_high 	adrr_easy_overall 	adrr_easy_low 	adrr_easy_high
0 	    12.07234 	    3.31325 	    8.75908 	        28.13814 	        9.68648 	    18.45166
1 	    25.82490 	    12.47077 	    13.35413 	        29.83252 	        14.25410 	    15.57842

        """
        columns = list(self.data.columns)
        levels = self.stats_functions[key]['levels']
        if 'type' in self.stats_functions[key].keys():
            df = pd.DataFrame()
            types = self.stats_functions[key]['type']
            cols = []
            for type_ in types:
                if len(levels)>1:
                    cols = [key+'_'+type_+'_'+level for level in levels]
                else:
                    cols = [key+'_'+type_]
                
                self.params['type']=type_
                ans = self.data.groupby(group_by)[columns].apply(self.stats_functions[key]['f'],
                                                **self.params,
                                                include_groups=True,
                                                )
                idx = list(ans.index)
                data = np.stack(ans.values)
                df = pd.concat([df,pd.DataFrame(data,index=idx,columns=cols)],axis=1)
            return df

        else:
            if len(levels)>1:
                cols = [key+'_'+level for level in levels]
            else:
                cols = [key]
            ans = self.data.groupby(group_by)[columns].apply(self.stats_functions[key]['f'],
                                                **self.params,
                                                include_groups=True,
                                                )
            idx = list(ans.index)
            data = np.stack(ans.values)
            return pd.DataFrame(data,index=idx,
                                columns = cols)

    
    def group_stats(self,stats,agg='all'):
        """
        group_stats - returns a dataframe for the group stats and aggregation requested.
        Input:
            stats - a list of statistics the we want to put into the table
            agg - the time frame that we are interested in: 'all','day','period',or 'date'
        Output: returns a dataframe with stats and time periods as rows and columns.
        """
        df = pd.DataFrame()
        for stat in stats:
            df=pd.concat([df,self.return_group_stats(stat,agg)],axis=1)
        
        if agg == 'all':
            df.index = ['all']
        elif agg == 'day':
            df.index = ['sleep','wake']
        return df.T
    
    
    def calculate_stats(self,**kwargs):
        """
        This functions is designed to calculate statistics based on the structure
            of the stats_function dictionary.
        """

        ## Saving time when stats have already been calculated for a unit of measure
        pd.options.display.float_format = '{:,.5f}'.format
        if 'units' in kwargs:
            units = kwargs['units']
            self.units = units
        else:
            units = self.units
        if 'group_by' in kwargs:
            group_by = kwargs['group_by']
        else:
            group_by = 'all'
        if hasattr(self,'stats_dict'):
            if (units,group_by) in self.stats_dict.keys():
                self.stats = self.stats_dict[(units,group_by)]
                return self.stats

        ## If stats have not already been calculated
        keys = []
        for key in self.stats_functions.keys():
            if group_by in self.stats_functions[key]['time_agg']:
                keys.append(key)
        self.stats = self.group_stats(keys,agg=group_by)
        self.stats_dict[(units,group_by)] = self.stats

        return self.stats
    
    
    ##############################################################################
    ############### Graphics Functions ###########################################
    ##############################################################################
    
    def init_plot_data(self):
        """
        creates plotdata for the object 
        """
        def convert_to_times(times):
            times_ = []
            for time in times:
                h = time.hour
                m = time.minute
                times_.append(f'{h:0>2}:{m:0>2}')
            return times_

        df = self.table.copy()
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
        plotdata['time_labels'] = convert_to_times(plotdata.index)
        plotdata['times'] = [(x.hour*60+x.minute) for x in plotdata.index]
        return plotdata
    
    def plot_agp(self,ax=None):
        """
        plot the agp time series only - input is an axis, output is to the
            input axis.
        """

        plt.style.use('ggplot')
        if ax==None:
            fig,ax = plt.subplots(figsize=(15,5))
            ret_fig = True
        else:
            ret_fig = False

        x = self.plot_data['times']
        x_labels = self.plot_data['time_labels']
        
        cols_to_plot = ['median','low1','low2','high1','high2']
        data = self.plot_data[cols_to_plot].copy()
        
        datalow1=np.array(data['low1'].values,dtype=float)
        datahigh1=np.array(data['high1'].values,dtype=float)
        datalow2 = np.array(data['low2'].values,dtype=float)
        datahigh2=np.array(data['high2'].values,dtype=float)

        colors = ['black','firebrick','cadetblue','lightblue']
        ax.plot(x,data['median'],color=colors[0],lw=3,zorder=10)
        ax.plot(x,data['low1'],color=colors[1],lw=1,ls='--',zorder=5)
        ax.plot(x,data['high1'],color=colors[1],lw=1,ls='--',zorder=5)
        ax.fill_between(x,datalow1,datahigh1,color=colors[1],alpha=0.1,zorder=5)
        ax.plot(x,data['low2'],color=colors[2],lw=2,zorder=7)
        ax.plot(x,data['high2'],color=colors[2],lw=2,zorder=7)
        ax.fill_between(x,datalow2,datahigh2,color=colors[3],zorder=7)
        bins = self.params['bins']

        colors = ['black','green','green','black']
        lws = [1,2,2,1]
        ax.hlines(bins[1:5],x.iloc[0],x.iloc[-1],color=colors,lw=lws)

        ax.set_xticks(ticks = x)
        ax.set_xticklabels(x_labels)
        ax.xaxis.set_major_locator(MultipleLocator(36))
        
        ## Dark / Light shading for day / night
        idx = int(360/self.deltat)
        ax.axvspan(x.iloc[0],x.iloc[idx],facecolor='0.3',alpha=0.5,zorder=-100)
        ax.axvspan(x.iloc[idx],x.iloc[-1],facecolor='0.7',alpha=0.5,zorder=-100)
        ax.set_xlim(x.iloc[0],x.iloc[-1]+1)
        
        ## Explanation - words on graphic
        txt = "Target Range"
        ax.text(-0.05,0.22,txt,transform=ax.transAxes,
                rotation=90,fontsize=12,color='green')
        
        ax.text(-0.034,0.14,f"{bins[1]:0.1f}",transform=ax.transAxes,
                fontsize=12,color='black')
        ax.text(-0.034,0.19,f"{bins[2]:0.1f}",transform=ax.transAxes,
                fontsize=12,color='darkgreen')
        ax.text(-0.038,0.5,f"{bins[3]:0.1f}",transform=ax.transAxes,
                fontsize=12,color='darkgreen')
        ax.text(-0.038,0.7,f"{bins[4]:0.1f}",transform=ax.transAxes,
                fontsize=12,color='black')
        ax.text(-0.038,0.98,f"{bins[5]:0.1f}",transform=ax.transAxes,
                fontsize=12,color='black')
        ylim = [0,351/self.const]
        ax.set_ylim(ylim[0],ylim[1])
        ax.yaxis.set_visible(False)
        
        # title_ = f"Glucose Profile: {self.data.index[0].date()} => {self.data.index[-1].date()}"
        # ax.set_title(title_,fontsize=16)
        if ret_fig:
            return fig
        else:
            return ax
    
    def bar_chart(self,ax):
        """
        displays the bar chart for the agp window.
        """
        stats = self.calculate_stats()
        scols = ['tir_<54', 'tir_54-70','tir_70-180', 
                 'tir_180-250', 'tir_>250']
        
        tir = stats.T[scols].values[0]
        c = self.const
        bars = np.array([0,54/c,70/c,180/c,250/c,350/c])
            
        #ax1.sharey(ax)
        ax.yaxis.set_visible(False)
        ax.bar(['TimeInRange'],[bars[1]],color='firebrick',alpha=0.3)
        ax.bar(['TimeInRange'],[bars[2]-bars[1]],bottom=bars[1],color='khaki',alpha=0.3)
        ax.bar(['TimeInRange'],[bars[3]-bars[2]],bottom=bars[2],color='green',alpha=0.3)
        ax.bar(['TimeInRange'],[bars[4]-bars[3]],bottom=bars[3],color='khaki',alpha=0.3)
        ax.bar(['TimeInRange'],[bars[5]-bars[4]],bottom=bars[4],color='firebrick',alpha=0.3)
    
    
        colors = ['firebrick','darkgoldenrod','green','darkgoldenrod','firebrick']
        heights = [(bars[i]+bars[i+1])/2 for i in range(len(bars[:-1]))]
        for i,h in enumerate(heights):
            ax.text(0,h,f"{tir[i]*100:0.1f}%",color=colors[i],fontweight=1000,
                        fontsize=20,ha='center',va='center')
        return ax
        
    
    def variability_closeup(self,ax,**kwargs):
        """
        provides the bottom left corner of the stats on the agp
            designed to change between mg/dL and mmol/L
            IQR is only reported in mg/dL
            many other metrics are unit agnostic
        """
        bottom = 0.3;top = 0.90;padding=0.01;
        line_color = 'lightgray'
        if 'cols' in kwargs:
            cols = kwargs['cols']
        else:
            cols = [0.05,0.2,0.35,0.5,0.65,0.8]
        if 'vlines' in kwargs:
            vlines = kwargs['vlines']
        else:
            vlines = [(cols[i]+cols[i+1])/2 for i in range(len(cols)-1)]
            
        if 'font_size' in kwargs:
            font_size=kwargs['font_size']
        else:
            font_size = [20,10,10,5]
        weight = ['normal','bold','normal']
        
        ymin = [bottom]*4
        ymin += [0.1]
        ymax = [top]
        
        scols = ['iqr','gvp','modd_paper','bgi_paper_(hbgi)','bgi_paper_(lbgi)',
                 'grade_paper_overall','grade_paper_high','grade_paper_target',
                 'grade_paper_low','adrr_paper_overall', 
                 'adrr_paper_high','adrr_paper_low', 'gri_overall',
                 'gri_hyper','gri_hypo',
                ]
        vals = self.stats.T[scols].values[0]
        i = 0 ####### First column #########################################
        rows = [0.85,0.65,0.45]
        ax.text(
            cols[i],rows[0],"IQR\nmg/dL",fontsize=font_size[0],
                weight=weight[0],ha = 'center', va = 'center'
               )
        ax.text(
            cols[i],rows[1],f"{vals[0]:0.1f}",fontsize=font_size[1],weight=weight[1],
                ha='center',va = 'center'
               )
        ax.text(
            cols[i],rows[2],"13-29",fontsize=font_size[2],weight=weight[2],
                ha='center',va = 'center',
        )
        ax.axvline(vlines[i],ymin=ymin[0],ymax=ymax[0], color=line_color,)
        
        i = 1 ####### Second column #########################################
        rows = [0.9, 0.8, 0.7, 0.6, 0.5]
        ax.text(cols[i],rows[0],"GVP",fontsize=font_size[0],
                weight=weight[0],ha = 'center', va = 'center',
               )
        ax.text(cols[i],rows[1],f"{vals[1]:0.2f}",fontsize=font_size[1],weight=weight[1],
                ha='center',va = 'center'
               )
        ax.axhline(y=rows[2],xmin=cols[i]-4*padding,xmax=cols[i+1]-11*padding,color='gray')
        
        ax.text(cols[i],rows[3],"MODD",fontsize=font_size[0],
                weight=weight[0],ha = 'center', va = 'center',
               )
        ax.text(cols[i],rows[4],f"{vals[2]:0.2f}",fontsize=font_size[1],weight=weight[1],
                ha='center',va = 'center'
               )
        ax.axvline(vlines[i],ymin=ymin[0],ymax=ymax[0], color=line_color,)
        
        i = 2 ####### Third column #########################################
        rows = [0.9, 0.8, 0.7, 0.6, 0.5]
        ax.text(cols[i],rows[0],"HBGI",fontsize=font_size[0],
                weight=weight[0],ha = 'center', va = 'center',
               )
        ax.text(cols[i],rows[1],f"{vals[3]:0.2f}",fontsize=font_size[1],weight=weight[1],
                ha='center',va = 'center'
               )
        ax.axhline(y=rows[2],xmin=cols[i]-4*padding,xmax=cols[i+1]-11*padding,color='gray')
        
        ax.text(cols[i],rows[3],"LBGI",fontsize=font_size[0],
                weight=weight[0],ha = 'center', va = 'center',
               )
        ax.text(cols[i],rows[4],f"{vals[4]:0.2f}",fontsize=font_size[1],weight=weight[1],
                ha='center',va = 'center'
               )
        
        ax.axvline(vlines[i],ymin=ymin[0],ymax=ymax[0], color=line_color,)
        
        i = 3 ####### Fourth column #########################################
        rows = [0.9, 0.8, 0.66, 0.56, 0.42, 0.32, 0.18, 0.08]
        ax.text(cols[i],rows[0],"GRADE",fontsize=font_size[0],
                weight=weight[0],ha = 'center', va = 'center',
               )
        ax.text(cols[i],rows[1],f"{vals[5]:0.2f}",fontsize=font_size[1],weight=weight[1],
                ha='center',va = 'center'
               )
        
        ax.text(cols[i],rows[2],"HYPER",fontsize=font_size[0],
                weight=weight[0],ha = 'center', va = 'center',
               )
        ax.text(cols[i],rows[3],f"{vals[6]:0.2f}",fontsize=font_size[1],weight=weight[1],
                ha='center',va = 'center'
               )
        
        ax.text(cols[i],rows[4],"EU",fontsize=font_size[0],
                weight=weight[0],ha = 'center', va = 'center',
               )
        ax.text(cols[i],rows[5],f"{vals[7]:0.2f}",fontsize=font_size[1],weight=weight[1],
                ha='center',va = 'center'
               )
        ax.text(cols[i],rows[6],"HYPO",fontsize=font_size[0],
                weight=weight[0],ha = 'center', va = 'center',
               )
        ax.text(cols[i],rows[7],f"{vals[8]:0.2f}",fontsize=font_size[1],weight=weight[1],
                ha='center',va = 'center'
               )
        ax.axvline(vlines[i],ymin=ymin[0],ymax=ymax[0], color=line_color,)

        i = 4 ####### Fifth column #########################################
        rows = [0.9, 0.8, 0.65, 0.55, 0.4, 0.3]
        ax.text(cols[i],rows[0],"ADRR Total",fontsize=font_size[0],
                weight=weight[0],ha = 'center', va = 'center',
               )
        ax.text(cols[i],rows[1],f"{vals[9]:0.2f}",fontsize=font_size[1],weight=weight[1],
                ha='center',va = 'center'
               )
        
        ax.text(cols[i],rows[2],"ADRR High",fontsize=font_size[0],
                weight=weight[0],ha = 'center', va = 'center',
               )
        ax.text(cols[i],rows[3],f"{vals[10]:0.2f}",fontsize=font_size[1],weight=weight[1],
                ha='center',va = 'center'
               )
        
        ax.text(cols[i],rows[4],"ADRR Low",fontsize=font_size[0],
                weight=weight[0],ha = 'center', va = 'center',
               )
        ax.text(cols[i],rows[5],f"{vals[11]:0.2f}",fontsize=font_size[1],weight=weight[1],
                ha='center',va = 'center'
               )
        ax.axvline(vlines[i],ymin=ymin[0],ymax=ymax[0], color=line_color,)
        
        
        i = 5 ####### Sixth column #########################################
        rows = [0.85, 0.68, 0.55, 0.45, 0.35, 0.25]
        ax.text(cols[i],rows[0],"Glycemia Risk Index\n(GRI)",fontsize=font_size[0],
                weight=weight[0],ha = 'center', va = 'center',
               )
        ax.text(cols[i],rows[1],f"{vals[12]:0.2f}",fontsize=font_size[1],weight=weight[1],
                ha='center',va = 'center'
               )
        
        ax.text(cols[i],rows[2],"HYPER COMP",fontsize=font_size[0],
                weight=weight[0],ha = 'center', va = 'center',
               )
        ax.text(cols[i],rows[3],f"{vals[13]:0.2f}",fontsize=font_size[1],weight=weight[1],
                ha='center',va = 'center'
               )
        
        ax.text(cols[i],rows[4],"HYPO COMP",fontsize=font_size[0],
                weight=weight[0],ha = 'center', va = 'center',
               )
        ax.text(cols[i],rows[5],f"{vals[14]:0.2f}",fontsize=font_size[1],weight=weight[1],
                ha='center',va = 'center'
               )
        
        
        #### Finish box, turn off grid lines #################
        ax.text(0.1,0.1,"VARIABILITY CLOSE-UP", fontsize=font_size[1],weight=1000)
        ax.set_axis_off()
        rect = patches.Rectangle((0,0),1,1,transform=ax.transAxes,alpha = 0.5,
                                linewidth=5,edgecolor='black',facecolor='white',
                                )
        ax.add_patch(rect)
        
        return ax
    
    def ax_stats(self,ax,**kwargs):
        top = 1
        bottom = 0
        stats = kwargs['stats']
        vals = kwargs['vals']
        cols = kwargs['cols']
        vlines = kwargs['vlines']
        normal = kwargs['normal']
        font_size = kwargs['font_size']
        decimals = 1
        
        if 'decimals' in kwargs:
            decimals = kwargs['decimals']
        
        if 'header' in kwargs:
            header = kwargs['header']
            top = 0.8
            
        if 'footer' in kwargs:
            footer = kwargs['footer']
            bottom = 0.2
        
        if 'footer_x' in kwargs:
            footer_x = kwargs['footer_x']
        else:
            footer_x =cols[0]
        
        if 'rows' in kwargs:
            rows = kwargs['rows']
        else:
            rows = [0.8,0.5,0.3]

        n = len(stats)
        
        for i,x in enumerate(cols):
            ax.text(x,rows[0],stats[i],fontsize=font_size[0],ha='center')
            ax.text(x,rows[1],f"{vals[i]:0.{decimals}f}",fontsize=font_size[1],
                    weight='bold',ha='center')
            ax.text(x,rows[2],normal[i],fontsize=font_size[2],ha='center')
        for i,x in enumerate(vlines[:-1]):
            mid = (vlines[i]+vlines[i+1])/2
            ax.axvline(mid,ymin=bottom+0.1,ymax=top-0.2,color='grey')
        
        ax.set_axis_off()
        rect = patches.Rectangle((0,0),1,1,transform=ax.transAxes,alpha = 0.5,
                                linewidth=5,edgecolor='black',facecolor='white',
                                )
        ax.add_patch(rect)
        if 'footer' in kwargs:
            ax.text(footer_x,0.1,footer,weight="bold",fontsize=font_size[2],ha='left')
        return ax    
    
    def agp_all(self,bar=True):
        """
        plots the text to go along with the ambulatory glucose
            profile. subplot2grid does not break up an axis so this
            cannot be preformed individually.
        """
        fs = (17,9)
        fig,ax = plt.subplots(figsize=fs,frameon=False)
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
        fig.subplots_adjust(wspace=0,hspace=0.1);
        self.stats = self.calculate_stats()
        
        ## Glucose Exposure ##
        scols = ['mean','gmi']
        ax1 = plt.subplot2grid(fs,(0,0),rowspan=4,colspan=2)
        kwargs = {'stats':['Avg\nGlucose',"Glycemic\nEstimate"],
                  'vals':self.stats.T[scols].values[0],
                  'decimals':1,
                  'normal':['88-116','<6'],
                  'cols':[0.2,0.7],
                  'rows':[0.75,0.55,0.3],
                  'vlines':[0.3,0.6],
                  'font_size':[12,12,10],
                  'footer':"Glucose Exposure",
                 }
        if self.units == 'mmol':
            kwargs['normal']=['4.9-6.4','<6']
        self.ax_stats(ax1,**kwargs)
        
        ## Time in Range ##
        scols = ['tir_<54','tir_54-70','tir_70-180','tir_180-250','tir_>250']
        ax2 = plt.subplot2grid(fs,(0,2),rowspan=4,colspan=4)
        kwargs = {'stats':['Very Low\nBelow 54\nmg/dL',
                           "Low Alert\nBelow 70\nmg/dL",
                           "In Target\n70-180\nmg/dL",
                           "High Alert\nAbove 180\nmg/dL",
                           "Very High\nAbove 250\nmg/dL",
                          ],
                  'vals':self.stats.T[scols].values[0]*100,
                  'normal':['0','<4','>90',"<6","0"],
                  'cols':[0.1,0.3,0.5,0.7,0.9],
                  'rows':[0.65,0.5,0.3],
                  'vlines':[0.1,0.3,0.5,0.7,0.9],
                  'font_size':[12,12,10],
                  'footer':"Time In Range (% of Overall Time)",
                  'footer_x':0.05
                 }
        if self.units == 'mmol':
            kwargs['stats']=['Very Low\nBelow 3\nmmol/L',
                           "Low Alert\nBelow 3.9\nmmol/L",
                           "In Target\n3.9-10\nmmol/L",
                           "High Alert\nAbove 10\nmmol/L",
                           "Very High\nAbove 13.9\nmmol/L",
                          ]
        ax2=self.ax_stats(ax2,**kwargs)
        
        
        ## Glucose Variability ##
        scols = ['cv','std']
        ax3 = plt.subplot2grid(fs,(0,6),rowspan=4,colspan=2)
        kwargs = {'stats':['Coefficient\nof Variation',"Std Dev\nmg/dL"],
                  'vals':self.stats.T[scols].values[0]*[100,1],
                  'normal':['<36','10-26'],
                  'cols':[0.25,0.7],
                  'rows':[0.75,0.55,0.3],
                  'vlines':[0.4,0.6],
                  'font_size':[12,12,10],
                  'footer':"Glucose Variability",
                 }
        ax3=self.ax_stats(ax3,**kwargs)
        
        ## Data Sufficiency ##
        ax4 = plt.subplot2grid(fs,(0,8),rowspan=4,colspan=1)
        scols = ['percent_active']
        kwargs = {'stats':['Percent Time\nCGM Active'],
                  'vals':self.stats.T[scols].values[0]*100,
                  'normal':[''],
                  'cols':[0.5],
                  'rows':[0.75,0.55,0.3],
                  'vlines':[],
                  'font_size':[12,12,10],
                  'footer':"Data\nSufficiency",
                  'footer_x':0.1
                 }
        ax4= self.ax_stats(ax4,**kwargs
                    )  
        
        stats1 = self.calc_stat('auc',group_by='day').stack().values
        stats1[1],stats1[0] = stats1[0],stats1[1]
        stats2 = self.calc_stat('auc').stack().values
        stats = np.hstack([stats1,stats2])

        ax5 = plt.subplot2grid(fs,(4,0),rowspan=4,colspan=3)
        scols = ['auc_wake','auc_sleep','auc_all']
        kwargs = {'stats':['Wake\n6am-12am','Sleep\n12am-6am','24 Hours'],
                  'vals':stats,
                  'normal':['','',''],
                  'decimals':0,
                  'cols':[0.2,0.5,0.8],
                  'rows':[0.75,0.55,0.4],
                  'vlines':[0.2,0.5,0.8],
                  'font_size':[12,12,10],
                  'footer':"GLUCOSE EXPOSURE CLOSEUP\nAverage Daily AUC (mg/dL)",
                  'footer_x':0.1,
                  'footer_y':0.1
                 }
        if self.units == 'mmol':
            kwargs['footer'] = "GLUCOSE EXPOSURE CLOSEUP\nAverage Daily AUC (mmol/L)"
            kwargs['normal'] = ['','','']
        ax5 = self.ax_stats(ax5,**kwargs)
        
        ax6 = plt.subplot2grid(fs,(4,3),rowspan=4,colspan=6)
        kwargs = {
            'cols':[0.05,0.2,0.35,0.5,0.65,0.85],
            'vlines':[0.12,0.28,0.42,0.57,0.72],
            'font_size':[12,12,10,5]
        }

        ax6 = self.variability_closeup(ax6,**kwargs
                    )
        
        if bar:
            ax7 = plt.subplot2grid(fs,(8,0),rowspan=9,colspan=8)
            ax7 = self.plot_agp(ax=ax7)
            ax8 = plt.subplot2grid(fs,(8,8),rowspan=9,colspan=1)
            ax8.sharey(ax7)
            ax8 = self.bar_chart(ax=ax8)
        else:
            ax7 = plt.subplot2grid(fs,(8,0),rowspan=9,colspan=9)
            ax7 = self.plot_agp(ax=ax7)            
        return fig
    

    
    def plot_daily_trace(self,data,date_num,ax):
        """
        plot_daily_traces - plots a small, single day of data 
            passed as a series object into this method.
            
        Input:
            data - (pandas Series) - time (datetime) as index
            date_num - day date printed in upper left corner
            ax - matplotlib axis 
        """
        
        c = self.const
        ylim = [0,300/c]
        hlines = [70/c,180/c]
        twelve_y = 30/c
        date_y = 250/c
        tick = [10/c,20/c]
        border = [10/c,300/c]
        top_tick = [290/c,300/c]

        font_tiny=10
        ax.set_axis_off()
        start = 60*data.index[0].hour+data.index[0].minute
        x1 = np.arange(start,1440,self.deltat)#data.index
        x2 = data.index.map(lambda x: x.hour*60+x.minute)
        ax.plot(x2,data.values)
        #x2 = data.index
        #ax.plot(data.index,data.values)
        ax.set_xlim(0,1435)
        ax.set_ylim(ylim[0],ylim[1])
        ax.hlines(y=hlines,xmin=x1[0],xmax=x1[-1],color='black')
        ax.annotate("12pm",xy=(1440/2-200,twelve_y),fontsize=font_tiny)
        ax.annotate(date_num,xy=(100,date_y),fontsize=font_tiny+3)
        ax.vlines(x=1440/2,ymin=tick[0],ymax=tick[1],color='black')
        ax.vlines(x=[0,1435],ymin=border[0],ymax=border[1],color='black')
        ax.hlines(y=border,xmin=0,xmax=1435,color='black')
        ax.vlines(x=1440/2,ymin=top_tick[0],ymax=top_tick[1],color='black')
        ax.fill_between(x=x1,y1=[hlines[0]]*len(x1),y2=[hlines[1]]*len(x1),color='gray',alpha=0.1)
        y1 = [hlines[1]]*len(x2); y2=data.values
        ax.fill_between(x=x2,y1=y1,y2=y2,where = y2>y1,color='yellow')
        y1 = [hlines[0]]*len(x2); y2=data.values
        ax.fill_between(x=x2,y1=y1,y2=y2,where=y2<y1, color='red')

        return ax
    
    def plot_daily_traces(self):
        """
        works with plot_daily_trace to present all of the traces in 
            a calendar format. One issue is we assume that each day 
            follows the previous which is not correct. Will have to
            come back to fix this later. This is just a day of the week
            issue.
        """
        
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
        fig, axs = plt.subplots(nrows=rows,ncols=7,figsize=(14,2*rows),frameon=False)
        fig.subplots_adjust(wspace=0,hspace=0)
        for i in range(rows):
            for j in range(7):
                axs[i,j].set_axis_off()

        cols = ['Monday','Tuesday','Wednesday','Thursday',
                'Friday','Saturday','Sunday']
        for ax,col in zip(axs[0],cols):
            ax.set_title(col,fontsize=10)
        for i,day in enumerate(days):
            df_ = days_df.loc[day]
            dt = day.strftime("%d")
            x = calendar[i][0]
            y = calendar[i][1]

            axs[x,y]=self.plot_daily_trace(df_,dt, axs[x,y])

        return (fig)    
    
    def plot_gri(self):
        """
        plot the glycemic risk index on a chart from the paper.
        """
        points = glycemic_risk_index(self.data,**self.params)
        zones = np.array([0,20,40,60,80,100])
        zones_=['A','B','C','D','E','E']
        
        pt0 = points[0]
        idx = np.where(pt0>=zones)[0][-1]
        zone = zones_[idx]
        
        x = np.linspace(0,30,1000)
        fa = lambda x:(20-3*x)/1.6
        fb = lambda x:(40-3*x)/1.6
        fc = lambda x:(60-3*x)/1.6
        fd = lambda x:(80-3*x)/1.6
        fe = lambda x:(100-3*x)/1.6
        fig,ax = plt.subplots(figsize=(10,5))
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
        """
        plots blood glucose vs blood glucose (shifted by shift_minutes).
        
        
        """
        plt.style.use('ggplot')
        fig,ax = plt.subplots(figsize=(6,3))
        
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

        ## ellipsis is centered at (0,0) - move it to the middle of the data
        ellipsis += X.mean(axis=0).reshape(ellipsis.shape[0],1)
        ax.set_title("Poincare Plot",fontsize=30)
        plim = [0,350]
        major = (5,25)
        minor = (5,5)
        xlabel = r"blood glucose $BG(t_{i-1})$ mg/dL"
        ylabel = r"blood glucose $BG(t_{i})$ mg/dL"
        if self.units == 'mmol':
            plim = [p/18.018 for p in plim]
            major = [m/18.018 for m in major]
            minor = [m/18.018 for m in minor]
            xlabel = r"blood glucose $BG(t_{i-1})$ mmol/L"
            ylabel = r"blood glucose $BG(t_{i})$ mmol/L"
        ax.set_xlim(plim[0],plim[1]);ax.set_ylim(plim[0],plim[1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.scatter(X[:,0],X[:,1],s=5)
        ax.plot(ellipsis[0,:],ellipsis[1,:],color='blue')
        body = f"Major axis = {wider_spread:0.1f}"
        ax.annotate(body,xy=major,fontsize=10,color='blue',weight='bold')
        body = f"Minor axis = {smaller_spread:0.1f}"
        ax.annotate(body,xy=minor,fontsize=10,color='blue',weight='bold')
        return (fig)
    
    def time_series_plot(self):
        """
        plots the time series for each wear period in the data, each in its own graph.
        """
        plt.style.use('ggplot')
        n = len(self.periods)
        
        ## Multiple time periods for the data
        if n>1:
            fig,ax = plt.subplots(nrows=n,ncols=1,figsize=(8,int(n*4)))
            fig.subplots_adjust(hspace=0.45)
            for i, period in enumerate(self.periods):
                pdata = self[i]['glucose']
                idx = np.arange(pdata.index[0],pdata.index[-1],timedelta(minutes=self.deltat))
                ax[i].plot(idx,pdata.loc[idx])
                formatter = mdates.DateFormatter('%d|%H:%M')
                ax[i].xaxis.set_major_formatter(formatter)
                ax[i].set_title("Time Series: "+str(period[0].date())+
                                " through " +str(period[1].date()) )
                ax[i].tick_params(axis='x',which='major',labelsize=8,
                                  labelrotation=90)
        
        ## One time period for all of the data
        else:    
            pdata = self.data['glucose'].copy()
            fig,ax = plt.subplots(nrows=n,ncols=1,figsize=(8,4))
            td = timedelta(minutes=self.deltat)
            idx = np.arange(pdata.index[0],pdata.index[-1],td)
            ax.plot(idx,pdata.loc[idx])
            self.pdata = (idx,pdata)
            formatter = mdates.DateFormatter('%m-%d')
            ax.xaxis.set_major_formatter(formatter)
            ax.set_title("Time Series: "+str(pdata.index[0].date())+
                                " through " +str(pdata.index[-1].date()))
        return(fig)
    
    def markov_chain_calculation(self,intervals=[54,70,180,250]):
        """
        computes and displays the 5 states based on the intervals for the markov chain
        """
        res = ""
        data = self.data['glucose'].copy()
        fig,ax = plt.subplots(figsize=(5,1))
        name = 'intervals'
        
        ## self.const = 1 if units == 'mg'; 18.018 if units == 'mmol'
        c = self.const
        intervals = np.array(intervals)/c #convert based on units
        val = intervals[0]
        ax.barh(name,val,color="firebrick")
        val = intervals[1] - intervals[0]
        ax.annotate("State1",xy=((intervals[0])/2-5/c,-0.2),
                    fontsize=10,rotation = 90)
        ax.barh(name,val,left = intervals[0],color='gold')
        val = intervals[2] - intervals[1]
        ax.annotate("State2",xy=((intervals[0]+intervals[1])/2-5/c,-0.2),
                    fontsize=10,rotation = 90)
        ax.barh(name,val,left = intervals[1],color='green')
        val = intervals[3]-intervals[2]
        ax.annotate("State3",xy=((intervals[1]+intervals[2])/2-5/c,-0.2),
                    fontsize=10,rotation = 90)
        ax.barh(name,val,left=intervals[2],color='gold')
        val = 350-intervals[3]
        ax.annotate("State4",xy=((intervals[2]+intervals[3])/2-5/c,-0.2),
                    fontsize=10,rotation = 90)
        ax.barh(name,val,left=intervals[3],color='firebrick')
        ax.annotate("State5",xy=((intervals[3]+350/c)/2-5/c,-0.2),
                    fontsize=10,rotation = 90)
        ax.set_xlim(0,350/c)
        ax.set_xticks(intervals,labels=[f'{i:.1f}' for i in intervals])
        plt.tick_params(axis='x',labelsize=5)
        
        tm,pi_star,er = transition_matrix(data,intervals,5,5)
        tm = pd.DataFrame(tm)
        cols = [f"x<{intervals[0]:.1f}",
                f"{intervals[0]:.1f}&ltx&lt{intervals[1]:.1f}",
                f"{intervals[1]:.1f}&ltx&lt{intervals[2]:.1f}",
                f"{intervals[2]:.1f}&ltx&lt{intervals[3]:.1f}",
                f"x>{intervals[3]:.1f}"]
        tm.columns=cols
        tm.index = cols
        res += '<p style="color:red;font-size:25px"> Transition Matrix </p>'
        res += (tm.style.set_properties(**{'text-align': 'center'}).format(precision=3).to_html())
        res += "</br> </br>"
        res += '<p style="color:red;font-size:25px"> Time spent in states. </p>'
        pi_star = pd.DataFrame(np.abs(np.round(pi_star,3)),index = cols)
        pi_star.columns=["long-term fraction of time"]
        res+= pi_star.style.set_properties(**{'text-align': 'center'}).format(precision=3).to_html()
        res+= f"Entropy Rate of Markov Chain: {np.array(er).sum():0.4f}"
        return res,fig

    # def plot_agp_report(self):
    #     """
    #     Bar Chart with labels.
    #     This is the graph displayed by multiple_CGM.agp_report, `Time in Ranges`.
    #     It is a bar chart with colors representing the different time
    #     in ranges for the file being observed.
    #     """
    #     def convert_time(time):
    #         res = "("
    #         if time > 60:
    #             hrs = time//60
    #             mins = time-hrs*60
    #             res += f'{hrs:1.0f} hours {mins:2.0f} minutes'
    #         else:
    #             res += f'{time:1.0f} minutes'
    #         res+=")"
    #         return res
    #     scols = ['tir_<54', 'tir_54-70',
    #              'tir_70-180', 'tir_180-250', 
    #              'tir_>250']
    #     tir = self.stats[scols].values.reshape(len(scols),)
    #     fig=plt.figure(frameon=False,figsize=(7,5))

    #     ax = fig.add_subplot(111)
    #     ax.set_axis_off()
    
    #     ax.bar(0,height=54-40,width=10,color = 'firebrick',bottom = 40)
    #     ax.bar(0,height=70-56,width=10,color = 'red',bottom = 56)
    #     ax.bar(0,height=180-72,width=10,color='green',bottom=72)
    #     ax.bar(0,height=250-182,width=10,color='yellow',bottom = 182)
    #     ax.bar(0,height=300-252,width = 10,color='orange',bottom = 252)


    #     vals = [40,54,70,180,250,300]
        
    #     lt = 5
    #     bg_font=10;sm_font=8
    #     mid=[];minutes = []

    #     ## Step 1 - boundaries, %, and times, create formatting variables mid
    #     for i in range(5):
    #         mid.append((vals[i+1]+vals[i])/2-5)
    #         minutes.append(convert_time(tir[i]*1440))
    #         ax.annotate(f'{tir[i]*100:3.0f}%',fontsize=bg_font,xy=(lt+20,mid[i]))
    #         ax.annotate(minutes[i],xy=(28,mid[i]+1),fontsize=sm_font)
    #         if i != 0:
    #             ax.annotate(f'{vals[i]:3.0f}',xy=(-6.5,vals[i]),fontsize=7)
    #     ## Step 2 - bottom row
    #     i=0
    #     ax.annotate("Very Low",xy=(7,mid[i]),fontsize=bg_font)
    #     ax.annotate("(<54 mg/dL)",xy=(12,mid[i]+1),fontsize=sm_font)
    #     ax.annotate("."*28,xy=(17.5,mid[i]+1),fontsize=sm_font)
        
    #     ## Second row
    #     i=1
    #     ax.annotate("Low",xy=(7,mid[i]),fontsize=bg_font)
    #     ax.annotate("(54-69 mg/dL)",xy=(9.25,mid[i]+1),fontsize=sm_font)
    #     ax.annotate("."*36,xy=(15.5,mid[i]+1),fontsize=sm_font)       

    #     i=2
    #     ax.annotate("Target Range",xy=(7,mid[i]),fontsize=bg_font)
    #     ax.annotate("(70-180 mg/dL)",xy=(14.25,mid[i]+1),fontsize=sm_font)
    #     ax.annotate("."*14,xy=(21,mid[i]+1),fontsize=sm_font)

    #     i=3
    #     ax.annotate("High",xy=(7,mid[i]),fontsize=bg_font)
    #     ax.annotate("(181-250 mg/dL)",xy=(9.5,mid[i]+1),fontsize=sm_font)
    #     ax.annotate("."*28,xy=(17.5,mid[i]+1),fontsize=sm_font)

    #     i=4
    #     ax.annotate("Very High",xy=(7,mid[i]),fontsize=bg_font)
    #     ax.annotate("(>250 mg/dL)",xy=(12.25,mid[i]+1),fontsize=sm_font)
    #     ax.annotate("."*24,xy=(18.5,mid[i]+1),fontsize=sm_font)

    #     plt.xlim(-8,35)
    #     plt.ylim(20,300)
    #     return fig
    
    ##################################################################################
    ##      AGP Report
    ##
    ##################################################################################    
    
    def tir_stats_table(self):
        stats = self.calculate_stats()
        stats_=['mean','median','iqr','cv','auc','auc_100']
        stats_tbl = stats.loc[stats_].T
        stats_tbl = pd.concat([stats_tbl,self.calc_stat(stats_,group_by='day')])

        return stats_tbl.T

    def tir_report(self,agg='all'):
        bins = np.array([0,54,70,120,140,180,250,500])/self.const
        tir = self.calc_stat('tir',bins = bins,group_by = agg)
        df = tir_stats(tir,units=self.units,bins=bins)
        
        return df
    

