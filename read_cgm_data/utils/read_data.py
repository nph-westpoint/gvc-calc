import pandas as pd
import numpy as np
import streamlit as st
from io import StringIO
from datetime import datetime

def panda_row_conversion(header,skip):
    """
    converts the header / skip_rows into a form used by pandas
    """
    skip_rows = []
    for i in range(len(header)):
        skip_rows.append(i)
    for i in range(header+1,header+1+skip):
        skip_rows.append(i)
    return skip_rows

def datetime_format_generator():
    """
    generate all types of datetime formats tested in our app's date conversion
        this is only useful when the datetime is different in each row.
    """
    date_fmts = [
        '%m/%d/%Y',
        '%Y/%m/%d',
        '%d/%m/%Y',

        '%m-%d-%Y',
        '%d-%m-%Y',
        '%Y-%m-%d',
    ]
    middle = [
        ' ',
        'T',
    ]
    time_fmts = [
        '%H:%M:%S',
        '%H:%M',
    ]
    datetime_fmts = []
    for date in date_fmts:
        for mid in middle:
            for time in time_fmts:
                datetime_fmts.append(date+mid+time)
    return datetime_fmts

def datetime_format(x,datetime_fmts):
        """
        the function passed to map
        """
        for dt_fmt in datetime_fmts:
            try:
                return datetime.strptime(x,dt_fmt)
            except:
                pass
        return 'Not a known format: ' + x



def read_data(filename,date_col=1,glucose_col=2,
              header_=0,skip_rows=1
              ):
    """
    output is a dataframe with datetime index and 
        glucose (mg/dL) as values in a column
    """
    dtf = datetime_format_generator()

    try:
        if header_ != 0:
            skip_rows = panda_row_conversion(header_,skip_rows)
            header_=0
        data = pd.read_csv(filename,header=header_).iloc[skip_rows-1:]
    except:
        data = read_io_data(filename.read(),skip_rows=skip_rows,header_= header_)
    columns = list(data.columns)
    cols = [columns[date_col],columns[glucose_col]]
    data = data.loc[:,cols]
    data.columns = ['datetime','glucose']
    
    try:
        data['datetime']=pd.to_datetime(data['datetime'])
    except:
        try:
            body = "This process is taking longer than usual. Most likely because "
            body += "the datetime values in this file are not in accordance with ISO 8601."
            st.markdown(body)
            data['datetime']=data['datetime'].map(lambda x: datetime_format(x,datetime_fmts = dtf))
        except:
            data['datetime']=pd.to_datetime(data['datetime'],format = 'mixed',dayfirst=True)
    data = data.set_index('datetime')
    data = data.dropna()
    data['glucose'] = data['glucose']
    data = data[~data.index.duplicated()]
    data = data[~data.index.isnull()]
    data = data.sort_index()
    

    return data


def read_io_data(filename,header_=0,skip_rows=1):
    """
    filename is a string IO from streamlit, import the data for the 
        glucose readings and the datetime for those readings into a dataframe
        that can be used by read_data.

    This function is used when pd.read_csv is not working for some reason in
        the read_data function above.
        
    Input:  filename: passed as an IO from streamlit
            header_: the index of the row that is the header row
            skip_rows: the number of rows that need to be skipped before 
                reading the data

    Output: raw dataframe
    """

    try:
        infile = convert_raw_data(filename)
    except:
        with open(filename,'r') as f:
            infile = StringIO(f.read()).read().split('\n')
    ## collect all of the data into lst_data one row at a time
    ## while also converting strings into integers, floats, and strings
    ## let all blank values '' = np.nan so that a column can be a number
    lst_data = []
    for line in infile:
        row = line.split(",")
        try:
            row[i]=row[i].replace('"','')
            if row[i] == '':
                row[i]=np.nan
        except:
            pass
        for i in range(len(row)):
            try:
                ## Integer
                row[i]=int(row[i])
            except:
                try:
                    ## Float
                    row[i]=float(row[i])
                except:
                    ## String, not a blank one though
                    pass
        lst_data.append(row)
    data = {}
    header = lst_data[header_]
    for i in range(len(header)):
        data[header[i]]=[]
    for line in lst_data[header_+skip_rows:]:
        if len(line)==len(header):
            for i, col in enumerate(line[:len(header)]):
                data[header[i]].append(col)
    try:
        data = pd.DataFrame.from_dict(data,orient='index')
        data = data.T
    except:
        pass
    return data

def raw_data_datetime(iofile, date_col = 1, skip_rows = 1,header=0):
    """
    attempt at figuring out datetime column
    """
    datetime_fmts = datetime_format_generator()
    
    try:
        infile = convert_raw_data(iofile)
    except:
        with open(iofile,'r') as f:
            infile = StringIO(f.read()).read().split('\n')
    lst_data = []
    for i,line in enumerate(infile):
        if i>header+skip_rows:
            row = line.split(",")
            ans = datetime_format(row[date_col],datetime_fmts)
            lst_data.append([row[date_col],ans[0],ans[1]])
    return lst_data

def convert_raw_data(iofile):
    """
    convert_raw_data - an attempt to get rid of any different encodings other
        than utf-8 so that we can always read the data.

    Input:
        iofile - the raw data from the file read in by streamlit's 
            file_uploader function which are `bytes`.
    Output:

    """
    infile = StringIO(iofile.decode('utf-8-sig'))
    infile = infile.getvalue()
    infile = infile.encode('utf-8',errors = 'ignore').decode('utf-8')
    results = StringIO(infile)
    return results
        

def view_raw_data(iofile,skip_rows = 1,header_ = 0,stop_=None):
    """
    view_raw_data - reads a file from Jupyter Notebook or streamlit
        displays the results. 
    either an iofile from streamlit or a filename from Jupyter Notebook
    this is just to read the data and display it on the screen
     
    """
    try:
        infile = convert_raw_data(iofile)
    except:
        with open(iofile,'r',encoding='utf-8',errors='ignore') as f:
            infile = StringIO(f.read()).read().split('\n')
    #str_data = ""
    lst_data = []
    for j,line in enumerate(infile):
        #str_data += 'row '+str(j)+' '+line +"\n"
        row = line.split(",")

        for i in range(len(row)):
            try:
                row[i]=row[i].replace('"','')
            except:
                pass
            try:
                row[i]=int(row[i])
            except:
                try:
                    row[i]=float(row[i])
                except:
                    pass

        lst_data.append(row)
        if j == stop_:
            break
    

    data = {}
    header = lst_data[header_]


    for i in range(len(header)):
        data[header[i]]=[]
    for line in lst_data[header_+skip_rows:]:
        #if len(line)==len(header):
        for i, col in enumerate(line[:len(header)]):
            data[header[i]].append(col)
    
    try:
        data = pd.DataFrame.from_dict(data,orient='index')
        data = data.T
        if len(data)<2:
            raise ValueError("Run Exception and return data as list")
    except:
         data=[]
         data.append(header)
         for l in lst_data[skip_rows:]:
            data.append(l)
    return data


def view_list(data,idx=0,color='salmon'):
    num_columns = 0
    for d in data:
        num_columns = max(num_columns,len(d))
    for d in data:
        while len(d)<num_columns:
            d.append(np.nan)
            
    header = ['col'+str(i) for i in range(num_columns)]
    data=pd.DataFrame(data,columns=header)
    data=data.style.applymap(lambda _:f"background-color: {color}",subset=(data.index[idx],))
    return data