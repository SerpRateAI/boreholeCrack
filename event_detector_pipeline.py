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
       # obspy_idx is the index within the stream for this data (all data is sorted from top to bottom of the borehole this way in lists)
        ,'obspy_idx':0
        # depth of the hydrophone
        ,'depth':30
        ,'velocity_model':1750
    }
    ,    'h2':{
        'start':obspy.UTCDateTime('2019-05-21T07:35:00Z')
        ,'end':obspy.UTCDateTime('2019-5-21T07:48:00Z')
        ,'obspy_idx':1
        ,'depth':100        
        ,'velocity_model':1750

    }
    ,    'h3':{
        'start':obspy.UTCDateTime('2019-05-21T07:35:00Z')
        ,'end':obspy.UTCDateTime('2019-5-21T07:48:00Z')
        ,'obspy_idx':2
        ,'depth':170        
        ,'velocity_model':1750

    }
    ,'h4':{
        'start':obspy.UTCDateTime('2019-05-21T07:48:00Z')
        ,'end':obspy.UTCDateTime('2019-5-21T08:07:00Z')
        ,'obspy_idx':3
        ,'depth':240
        ,'velocity_model':1750
    }
    ,'h5':{
        'start':obspy.UTCDateTime('2019-05-21T08:07:00Z')
        ,'end':obspy.UTCDateTime('2019-5-21T08:34:00Z')
        ,'obspy_idx':4
        ,'depth':310
        ,'velocity_model':1750
    }
    ,'h6':{
        'start':obspy.UTCDateTime('2019-05-21T08:34:00Z')
        ,'end':obspy.UTCDateTime('2019-5-21T08:38:00Z')
       ,'obspy_idx':5
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
        # _event = df_picks.iloc[id]
        # self.starttime = _event['event_times (abs)']
        # self.starttime = starttime
        # print(starttime)
        # print(dates.num2date(starttime))
        starttime = dates.num2date(starttime)
        self.starttime = obspy.UTCDateTime(starttime)
        # print(self.starttime)
        # self.first_hydrophone_id = _event['hphone_idx']
        self.first_hydrophone_id = init_first_hphone
        self.stream = self.get_waveforms(starttime=self.starttime)
        # print(self.stream)
        # self.mpl_times = [tr.times('matplotlib') for tr in self.stream]
        
        self.aic_t, self.aics = self.aic_pick()
        # print(self.aic_t)
        self._get_first_second_hydrophones()
        # self._get_second_arrival_hydrophone()
        # self.arrival_time = self.aic_t
        self.first_hphone_label = 'h'+str(self.first_hydrophone_id)
        self.second_hphone_label = 'h'+str(self.second_hydrophone_id)
        
        self.hphone1_time = self.aic_t[self.first_hydrophone_id]
        self.hphone2_time = self.aic_t[self.second_hydrophone_id]
        # self.depth = self.get_depth(hA=self.first_hydrophone_id, hB=self.second_hydrophone_id)
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

        return aic_t, aics
        
    def plot(self, kind):
        if kind == 'waveforms':
            return self._plot_waveforms_with_aic()
        if kind == 'event depth':
            return self._plot_event_depth()
    
    def _plot_waveforms_with_aic(self):
        """
        plots the waveforms with the AIC scores and AIC picks
        for the each hydrophone
        
        Parameters
        ----------------
        None
        
        Return
        ----------------
        fig : matplotlib.pyplot.Figure
            matplotlib Figure
        axes : numpy.array
            array of matplotlib.pyplot.Axes axes
        """
        fig, axes = plotting.plot_waveforms(self.stream, color='black')
        for n, ax in enumerate(axes):
            ax2 = ax.twinx()
            t = self.stream[n].times('matplotlib')
            aic = self.aics[n]
            ax2.plot(t, aic, color='red')
            ax.plot((self.aic_t[n], self.aic_t[n]), (-2000, 2000), color='dodgerblue')
        return fig, axes
    
    def _plot_event_depth(self):
        """
        Plots the depth profile for the event
        """
        x = np.zeros(6)
        h_depths = -1 * np.array([hydrophones[h]['depth'] for h in hydrophones])

        hA_depth = h_depths[hydrophones[self.first_hydrophone_id]['idx']]
        hB_depth = h_depths[hydrophones[self.second_hydrophone_id]['idx']]
        
        fig, ax = plt.subplots(figsize=(5, 15))
        
        # plot hydrophone cable axis
        ax.plot((0, 0), (0, -400), color='black')
        ax.plot(x, h_depths, marker='s', color='black')
        ax.plot((0, 0), (hA_depth, hB_depth), marker='s', color='limegreen', markersize=10, linewidth=5)

        ax.set_yticks(h_depths)
        
        # make a label for each hydrophone
        for n, h in enumerate(h_depths):
            ax.text(s='h{n}'.format(n=n+1), x=0.005, y=h)
        
        ax.plot((0,), -self.depth, marker='*', color='red', markersize=15)
        return fig, ax

    def _get_first_second_hydrophones(self):
        # we skip the first two hydrophones because they are always useless and often can have AICs that come in the very beginning
        sorted_aic = np.argsort(self.aic_t[2:])
        # we add 2 because np.argsort returns the index of the sorted array and because we skip the first two hydrophones we chnage the indices
        # print(self.aic_t)
        # print(sorted_aic)
        self.first_hydrophone_id = sorted_aic[0] + 2
        # self.first_hydrophone_id = sorted_aic[0]
        self.second_hydrophone_id = sorted_aic[1] + 2
        # self.second_hydrophone_id = sorted_aic[1]

            
    # def get_depth(self, hA, hB):
    def get_depth(self):
        t_A = dates.num2date(self.hphone1_time)
        t_B = dates.num2date(self.hphone2_time)
        # print(t_A, t_B, (t_A - t_B).total_seconds())
        
        dt = (t_A - t_B).total_seconds()
        # print(dt)
        
        sign = self.first_hydrophone_id - self.second_hydrophone_id
        # print(sign)
        
        dz_phone = np.min([self.first_hydrophone_id, self.second_hydrophone_id])
        dz_phone_label = 'h' + str(dz_phone)
        
        hydrophone_depth = hydrophone_metadata[dz_phone_label]['depth']
        # print(hydrophone_depth)
        
        dz = 35 - 0.5 * dt * self.velocity_model * sign
        # print(dz)
        
        z = dz + hydrophone_depth
        # print(z)
        
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

    # filter and transform data

    ## trim data to be only for swarm times
    waveforms.trim(starttime=swarm_starttime, endtime=swarm_endtime)

    ## 50hz high pass filter
    waveforms.filter(type='highpass', corners=1, zerophase=False, freq=50)
    
    ## make copy for precision detection later
    waveforms_copy = waveforms.copy()

    ## square amplitude
    for n, tr in enumerate(waveforms):
        waveforms[n].data = tr.data**2

    ## peak finder
    peak_times = {}
    for n, tr in enumerate(waveforms):
        hydrophone_id = 'h' + str(n+1)

        # trim data for only selected data from hydrophone data
        tr_trim = tr.slice(starttime=hydrophone_metadata[hydrophone_id]['start']
                           ,endtime=hydrophone_metadata[hydrophone_id]['end'])
        t = tr_trim.times('matplotlib')

        # apply peak finding algorithm
        idx, props = signal.find_peaks(tr_trim.data, height=0.25, distance=250)

        # record initial event times for each peak detected for each hydrophone
        peak_times[hydrophone_id] = np.array(t[idx])

    ### 

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
        df_picks = pd.concat([df_picks, rows])
        
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
        event = {
            'id':id
            ,'depth':e.depth
            ,'aic_t':e.aic_t
            ,'first_hydrophone':e.first_hydrophone_id
            ,'second_hydrophone':e.second_hydrophone_id
            ,'arrival_time':e.aic_t[e.first_hydrophone_id]
        }
        return event

    ## multiprocess do over events to get all event data
    from multiprocess import Pool
    
    pool = Pool(10)
    
    rows = pool.map(do, df_picks.index.values)
    
    pool.close()
    
    # df_precision = pd.concat(rows)
    df_precision = pd.DataFrame(rows)
    
    
    ## make dataframe of event times, depths
    
    df = df_picks.join(df_precision)

    ## write dataframe to file
    
    df.to_csv('hmmm.csv')