"""
This script presents the entire pipeline for locating events and calculating their depths in the raw data

# HOW TO USE

This script generates a single catalog for a single day's worth of events.

To run you should open the terminal and type:

python event-pipeline.py <day number>

The <day number> argument should be known to you as a day there are many events based on visual inspection of the raw waveforms.

This will call a bunch of meta data from the config.py file. This metadata tells what raw data to select for the particular day number. If you pick a day number not found in config.py the script will exit as it it will have nothing to do.

This script will then run and produce two CSV files called precision.csv and hmmm.csv. Ignore hmm.csv. All data is stored in precision.csv.

You can then go and look at the notebook day-141-188-197-211-viz which visualizes all the data from the catalog generated in this script.
"""

import numpy as np
import obspy
from hydrophone_data_processing import load, useful_variables, plotting, signal_processing
import scipy.signal as signal
import pandas as pd
import matplotlib.dates as dates
import obspy.signal.trigger as trigger
import sys
import config
from Event import Event

hydrophone_metadata = config.hydrophone_metadata['141']



        

if __name__ == '__main__':
    import sys
    
    args = sys.argv
    day_number = args[1]

    swarm_starttime = config.swarm_starttime[day_number]
    swarm_endtime = config.swarm_endtime[day_number]

    hydrophone_metadata = config.hydrophone_metadata[day_number]

    paths = useful_variables.make_hydrophone_data_paths(borehole='a', year=2019, julian_day=day_number)

    # loads data for all hydrophones
    # converts to pascals
    # flips the sign on hydrophone 3 if there it is borehole B due to wiring problem
    waveforms = load.import_corrected_data_for_single_day(paths=paths)
    print('loading data from:', paths)
    # filter and transform data

    ## trim data to be only for swarm times
    waveforms.trim(starttime=swarm_starttime, endtime=swarm_endtime)
    print('trimming data to be between:\n', swarm_starttime, '\n', swarm_endtime)
    
    ## 50hz high pass filter
    waveforms.filter(type='highpass', corners=1, zerophase=False, freq=50)
    print('applying 50Hz, corners=1, no zerophase, high pass filter')
    
    ## make copy for precision detection later
    waveforms_copy = waveforms.copy()
    print('making copy of data')

    ## square amplitude
    for n, tr in enumerate(waveforms):
        waveforms[n].data = tr.data**2

    print('squaring amplitude of data')
    
    ## peak finder
    ## finds peaks in datat for each hydrophone
    peak_times = {}
    for n, tr in enumerate(waveforms):
        hydrophone_id = 'h' + str(n+1)

        # trim data for only selected data from hydrophone data
        tr_trim = tr.slice(starttime=hydrophone_metadata[hydrophone_id]['start']
                           ,endtime=hydrophone_metadata[hydrophone_id]['end'])
        t = tr_trim.times('matplotlib')
        print('trimming data for detection on', hydrophone_id, 'with starttime \n'
              ,hydrophone_metadata[hydrophone_id]['start']
              ,'\n and end time \n'
              ,hydrophone_metadata[hydrophone_id]['end']
             )

        # apply peak finding algorithm
        idx, props = signal.find_peaks(tr_trim.data, height=0.25, distance=250)
        print('finding peaks in squared data')

        # record initial event times for each peak detected for each hydrophone
        peak_times[hydrophone_id] = np.array(t[idx])

    ### 
    print(peak_times.keys())
    for k in peak_times.keys():
        print('hydrophone', k, 'number of events:', len(peak_times[k]))
    ## store initial picks in dataframe
    
    df_picks = pd.DataFrame()
    index_start = 0
    for k in peak_times.keys():
        init_arrivals = peak_times[k]
        n_events = init_arrivals.shape[0]
        index = np.arange(index_start, index_start+n_events, 1)
        rows = pd.DataFrame({'arrival_hydrophone':(k,)*n_events
                      ,'init_arrival_time':init_arrivals
                     }, index=index
                    )
        index_start = n_events
        print('hydrophone', k, 'number of events:', rows.shape)
        df_picks = pd.concat([df_picks, rows])
        
    print('storing initial peaks in dataframe')
    print('number of initial events detected:', df_picks.shape)
    print('first event:', dates.num2date(df_picks.init_arrival_time.min()))
    print('last event:', dates.num2date(df_picks.init_arrival_time.max()))
    # return to raw data to make closer picks
    
    ## create function for multiprocessing
    def do(id=id):
        df_picks_row = df_picks.iloc[id]
        e = Event(id=id
                  , starttime=df_picks_row.init_arrival_time
                  , init_first_hphone=df_picks_row.arrival_hydrophone
                  , waveforms=waveforms.copy()
                  , velocity_model=1750
                 )
        
        event = {
            'id':id
            ,'depth':e.depth
            ,'relative_depth':e.relative_depth
            ,'radius':e.radius
            ,'aic_t':e.aic_t
            # ,'aics':e.aics
            ,'aics':list(e.aics[0])
            ,'aic_maxes':e.maxes
            ,'first_hydrophone':e.first_hydrophone_id
            ,'second_hydrophone':e.second_hydrophone_id
            ,'arrival_time':e.aic_t[e.first_hydrophone_id]
            ,'first_arrival':dates.num2date(e.hphone1_time)
            ,'second_arrival':dates.num2date(e.hphone2_time)
            ,'dt':(dates.num2date(e.hphone1_time) - dates.num2date(e.hphone2_time)).total_seconds()
            ,'parrival':e.parrival
            ,'max_amp':e.stream[e.first_hydrophone_id].data.max()
            ,'cum_amp':abs(e.stream[e.first_hydrophone_id].data).cumsum()[-1]
            # this calculates the true origin time and not the arrival time on the hydrophoneg
            ,'origin_time':obspy.UTCDateTime(dates.num2date(e.hphone1_time)) - (e.relative_depth / e.velocity_model)
            ,'init_arrival_time':df_picks_row.init_arrival_time
        }
        return event

    print('calculating precision peaks')
    rows = []
    idx = np.arange(0, df_picks.shape[0], 1)

    for id in idx:
        print('calculating event', id)
        rows.append(do(id=id))

    df_precision = pd.DataFrame(rows)
    df_precision.to_csv(str(day_number)+'precision.csv')
    
    ## make dataframe of event times, depths
    
    # df = df_picks.join(df_precision)
    print('final number of events:', df_precision.shape)
    
    print('first event time:', df_precision.first_arrival.min())
    print('last event time:', df_precision.first_arrival.max())

