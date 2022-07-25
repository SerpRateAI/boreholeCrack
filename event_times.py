"""
This module identifies the initial arrival times picked by
amplitude peak picking
"""

# TODO : remove the day141_raw from here, it doesnt need to be here <- this is false it does need to be here according to the functions
# TODO : function out the hanning window AIC picker
# TODO : write documentation for how all this tsuff is working

import numpy as np
import obspy
from hydrophone_data_processing import load, useful_variables, plotting, signal_processing
import scipy.signal as signal
import pandas as pd
import matplotlib.dates as dates
import obspy.signal.trigger as trigger

day141_paths = useful_variables.make_hydrophone_data_paths(borehole='a', year=2019, julian_day=141)
day141 = load.import_corrected_data_for_single_day(day141_paths)

day141_snapshot = day141.slice(starttime=obspy.UTCDateTime('2019-05-21T07:30:00'), endtime=obspy.UTCDateTime('2019-05-21T08:38:30'))

# day141_snapshot.filter(type='highpass', corners=4, zerophase=True, freq=20)
day141_raw = day141_snapshot.copy()
# day141_snapshot.filter(type='highpass', corners=1, zerophase=False, freq=40)
day141_snapshot.filter(type='highpass', corners=1, zerophase=False, freq=50)
# day141_snapshot.filter(type='highpass', corners=1, zerophase=False, freq=10)


#######
#######
# here is where you are cheating because you know that there is a 
# downward migration of the cracking events so you simply used your
# eyes to look at when is the maximum magnitudes on each hydrophone
# so you detect across which ones are arriving first
detector_data = {
    'h3':{
        'start':obspy.UTCDateTime('2019-05-21T07:35:00Z')
        ,'end':obspy.UTCDateTime('2019-5-21T07:48:00Z')
        ,'height':0.25
        ,'distance':250
        ,'obspy_idx':2
    }
    ,'h4':{
        'start':obspy.UTCDateTime('2019-05-21T07:48:00Z')
        ,'end':obspy.UTCDateTime('2019-5-21T08:07:00Z')
        ,'height':0.5
        ,'distance':250
        ,'obspy_idx':3
    }
    ,'h5':{
        'start':obspy.UTCDateTime('2019-05-21T08:07:00Z')
        ,'end':obspy.UTCDateTime('2019-5-21T08:34:00Z')
        ,'height':0.5
        ,'distance':250
        ,'obspy_idx':4
    }
    ,'h6':{
        'start':obspy.UTCDateTime('2019-05-21T08:34:00Z')
        ,'end':obspy.UTCDateTime('2019-5-21T08:38:00Z')
        ,'height':0.5
        ,'distance':250
        ,'obspy_idx':5
    }
}

def get_event_window(event_time, hydrophone):
    """
    gets data around event with lead time and follow times 
    defined by the start_window_edge and end_window_edge
    
    The edges are in units of seconds
    """
    start_window_edge = -0.1
    end_window_edge = 0.25
    starttime = event_time + start_window_edge
    endtime = event_time + end_window_edge
    return day141_snapshot[detector_data[hydrophone]['obspy_idx']].slice(starttime=starttime, endtime=endtime).copy()

def get_event_window_raw(event_time, hydrophone):
    """
    gets data around event with lead time and follow times 
    defined by the start_window_edge and end_window_edge
    
    The edges are in units of seconds
    """
    start_window_edge = -0.1
    end_window_edge = 0.25
    starttime = event_time + start_window_edge
    endtime = event_time + end_window_edge
    return day141_raw[detector_data[hydrophone]['obspy_idx']].slice(starttime=starttime, endtime=endtime)

def peak_finder(obspydata, height, distance):
    """
    finds peaks in square of the amplitude
    """
    amp_sq = obspydata.data**2
    data_idx, props = signal.find_peaks(amp_sq, height=height, distance=distance)
    return data_idx, props

def get_event_times(hydrophone):
    """
    returns event times
    
    this function generates the catalog of initial event times
    """
    data = day141_snapshot.slice(starttime=detector_data[hydrophone]['start']
                                ,endtime=detector_data[hydrophone]['end'])
    
    data = data[detector_data[hydrophone]['obspy_idx']]
    idx, props = peak_finder(obspydata=data
                             , height=detector_data[hydrophone]['height']
                             , distance=detector_data[hydrophone]['distance'])
    
    detector_data[hydrophone]['event_times'] = data.times('matplotlib')[idx]
    detector_data[hydrophone]['event_times (abs)'] = data.times('utcdatetime')[idx]
    return detector_data[hydrophone]['event_times']

# this puts the initial picsk in the detector_data dictionary
[get_event_times(k) for k in list(detector_data.keys())]

# this turns everything into a dataframe
df = pd.DataFrame()
for h in detector_data.keys():
    df_temp = pd.DataFrame()
    df_temp['event_times (mpl)'] = detector_data[h]['event_times']
    df_temp['event_times (abs)'] = detector_data[h]['event_times (abs)']
    df_temp['hphone_idx'] = h
    df = pd.concat([df, df_temp])
    
df['ones'] = 1
df['event_times'] = df['event_times (mpl)'].apply(dates.num2date)
df.reset_index(inplace=True)

p_arrival_times = []

event_ids = df.index
for event_id in event_ids:
    event_meta = df.iloc[event_id]
    event = get_event_window(event_meta['event_times (abs)'], hydrophone=event_meta['hphone_idx'])
    event.taper(type='hann', max_percentage=0.5)
    aic = trigger.aic_simple(event.data)
    aic_t = pd.to_datetime(dates.num2date(event.times('matplotlib')[aic.argmin()]))
    p_arrival_times.append(aic_t)

# TODO : pick a better name than this
df['p_arrival_hphone_idx'] = p_arrival_times

print('number of events detected:', df.shape)
