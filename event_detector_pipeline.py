"""
This script presents the entire pipeline for locating events and calculating their depths in the raw data
"""

import numpy as np
import obspy
from hydrophone_data_processing import load, useful_variables, plotting, signal_processing
import scipy.signal as signal
import pandas as pd
import matplotlib.dates as dates
import obspy.signal.trigger as trigger

# important variables
## data for day 141
swarm_starttime = obspy.UTCDateTime('2019-05-21T07:30:00')
swarm_endtime = obspy.UTCDateTime('2019-05-21T08:38:30')

hydrophone_metadata = {
    'h1':{
        # start and end identifies the start time of the swarm where the amplitude magnitude is the highest
        'start':obspy.UTCDateTime('2019-05-21T07:35:00Z')
        ,'end':obspy.UTCDateTime('2019-5-21T07:48:00Z')
        # depth of the hydrophone
        ,'depth':30
        ,'velocity_model':1750
    }
    ,    'h2':{
        'start':obspy.UTCDateTime('2019-05-21T07:35:00Z')
        ,'end':obspy.UTCDateTime('2019-5-21T07:48:00Z')
        ,'depth':100        
        ,'velocity_model':1750

    }
    ,    'h3':{
        'start':obspy.UTCDateTime('2019-05-21T07:35:00Z')
        ,'end':obspy.UTCDateTime('2019-5-21T07:48:00Z')
        ,'depth':170        
        ,'velocity_model':1750

    }
    ,'h4':{
        'start':obspy.UTCDateTime('2019-05-21T07:48:00Z')
        ,'end':obspy.UTCDateTime('2019-5-21T08:07:00Z')
        ,'depth':240
        ,'velocity_model':1750
    }
    ,'h5':{
        'start':obspy.UTCDateTime('2019-05-21T08:07:00Z')
        ,'end':obspy.UTCDateTime('2019-5-21T08:34:00Z')
        # ,'end':obspy.UTCDateTime('2019-5-21T08:38:00Z')
        ,'depth':310
        ,'velocity_model':1750
    }
    ,'h6':{
        'start':obspy.UTCDateTime('2019-05-21T08:34:00Z')
        ,'end':obspy.UTCDateTime('2019-5-21T08:38:00Z')
        ,'depth':380
        ,'velocity_model':1750
    }
}

class Event:
    """
    Data holder class for cracking event from hydrophone
    """
    def __init__(self, id, starttime, init_first_hphone, waveforms, velocity_model=1750):        
        self.id = id
        self.data = waveforms
        self.velocity_model = velocity_model
        self._max_dx = 70 # meters spacing between hydrophones
        self._max_dt = self._max_dx / self.velocity_model
        starttime = dates.num2date(starttime)
        self.starttime = obspy.UTCDateTime(starttime)
        self.first_hydrophone_id = init_first_hphone
        self.stream = self.get_waveforms(starttime=self.starttime)
        
        # self.aic_t, self.aics = self.aic_pick()
        self.maxes, self.aic_t, self.aics = self.aic_pick()
        
        self._get_first_second_hydrophones()
        
        self.hphone1_time = self.aic_t[self.first_hydrophone_id]
        self.hphone2_time = self.aic_t[self.second_hydrophone_id]
        print(dates.num2date(self.hphone2_time))
        # the problem happens after here...
        
        self.get_depth()

    def get_waveforms(self, starttime):
        """
        Returns a 1 second long trimmed event for the starttime and endtime
        
        Applies hanning window to edges to smooth to zero.

        Parameters
        -----------------
        starttime : obspy.UTCDatetime
            starttime for the event

        Returns
        -----------------
        data : obspy.Stream
            trimmed waveforms for 1 second event window
        """
        starttime = starttime - 0.2
        endtime = starttime + 0.5
        trimmed = self.data.copy().trim(starttime=starttime, endtime=endtime)
        trimmed.taper(type='hann', max_percentage=0.5)
        return trimmed

    def aic_pick(self):
        """
        Uses obspy aic_simple to pick the start time of an event

        Parameters
        --------------------
        event : obspy.Stream
            an obspy stream with traces inside. the expected data
            will be only for a single event, not the whole data set

        Returns
        --------------------
        aic_t : list
            the times per hydrophone for each aic picked event
        aics : list
            the raw aics calculated for each event
        """

        # calculates aic score
        aics = [trigger.aic_simple(tr.data) for tr in self.stream]

        # finds minimum and returns index for aic scores
        diffs = [np.diff(aic, n=1) for aic in aics]
        maxes = [np.argmax(diff) for diff in diffs]
        

        # uses minimum index to retrieve the timestamp
        aic_t = [self.stream[n].times('matplotlib')[i] for n, i in enumerate(maxes)]

        # return aic_t, aics
        return maxes, aic_t, aics

    def _get_first_second_hydrophones(self):
        # we skip the first two hydrophones because they are always useless and often can have AICs that come in the very beginning
        sorted_aic = np.argsort(self.aic_t[2:])
        # we add 2 because np.argsort returns the index of the sorted array and because we skip the first two hydrophones we chnage the indices
        self.first_hydrophone_id = sorted_aic[0] + 2
        self.second_hydrophone_id = sorted_aic[1] + 2

            
    # def get_depth(self, hA, hB):
    def get_depth(self):
        t_A = dates.num2date(self.hphone1_time)
        t_B = dates.num2date(self.hphone2_time)
        
        dt = (t_A - t_B).total_seconds()
        
        sign = self.first_hydrophone_id - self.second_hydrophone_id
        # sign = -(self.first_hydrophone_id - self.second_hydrophone_id)
        
        dz_phone = np.min([self.first_hydrophone_id, self.second_hydrophone_id])
        # dz_phone_label = 'h' + str(dz_phone)
        dz_phone_label = 'h' + str(dz_phone+1)
        
        hydrophone_depth = hydrophone_metadata[dz_phone_label]['depth']
        
        dz = 35 - 0.5 * dt * self.velocity_model * sign
        # dz = 35 - 0.5 * dt * self.velocity_model
        
        self.relative_depth = dz
        
        z = dz + hydrophone_depth
        # z = dz - hydrophone_depth
        
        self.depth = z

if __name__ == '__main__':
    import sys
    args = sys.argv
    # print(args)
    # import raw data
    paths = useful_variables.make_hydrophone_data_paths(borehole='a', year=2019, julian_day=141)

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
    # print(peak_times['h6'])
    for k in peak_times.keys():
        print('hydrophone', k, 'number of events:', len(peak_times[k]))
    ## store initial picks in dataframe
    
    df_picks = pd.DataFrame()
    index_start = 0
    for k in peak_times.keys():
        init_arrivals = peak_times[k]
        n_events = init_arrivals.shape[0]
        index = np.arange(index_start, index_start+n_events, 1)
        # print('index',index.shape)
        # print('init_arrivals', init_arrivals.shape)
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
        # print(e.depth)
        # print(e.aics)
        event = {
            'id':id
            ,'depth':e.depth
            ,'relative_depth':e.relative_depth
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
        }
        return event

    ## multiprocess do over events to get all event data
    print('calculating precision peaks')
    rows = []
    idx = np.arange(0, df_picks.shape[0], 1)
    # for id in df_picks.index.values:
    for id in idx:
        print('calculating event', id)
        rows.append(do(id=id))
    # rows = [do(id=id) for id in df_picks.index.values]
#     from multiprocess import Pool
    
#     pool = Pool(10)
    
#     rows = pool.map(do, df_picks.index.values)
    
#     pool.close()
    
    # df_precision = pd.concat(rows)
    df_precision = pd.DataFrame(rows)
    df_precision.to_csv('precision.csv')
    
    ## make dataframe of event times, depths
    
    df = df_picks.join(df_precision)
    print('final number of events:', df.shape)
    
    print('first event time:', df.first_arrival.min())
    print('last event time:', df.first_arrival.max())

    ## write dataframe to file
    print('writing picks to file')
    df.to_csv('hmmm.csv')