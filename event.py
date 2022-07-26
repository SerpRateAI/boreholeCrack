import obspy
import obspy.signal.trigger as trigger
import matplotlib.pyplot as plt
import numpy as np
from hydrophone_data_processing import load, preprocessing, tempmatch, useful_variables, plotting
from matplotlib.dates import num2date
import event_times as iet
import itertools

hydrophones = {'h1':{'depth':30, 'idx':0}
              ,'h2':{'depth':100, 'idx':1}
              ,'h3':{'depth':170, 'idx':2}
              ,'h4':{'depth':240, 'idx':3}
              ,'h5':{'depth':310, 'idx':4}
              ,'h6':{'depth':380, 'idx':5}}

paths = useful_variables.make_hydrophone_data_paths(borehole='a', year=2019, julian_day=141)
data = load.get_raw_stream(paths=paths)
data.filter(type='highpass', freq=50, corners=1, zerophase=False)

class Event:
    """
    Data holder class for cracking event from hydrophone
    """
    def __init__(self, id, velocity_model=1600):
        import event_times as iet
        
        self.id = id
        self.velocity_model = velocity_model
        _event = iet.df.iloc[self.id]
        self.starttime = _event['event_times (abs)']
        self.first_hydrophone_id = _event['hphone_idx']
        self.stream = self.get_waveforms(starttime=self.starttime)
        self.mpl_times = [tr.times('matplotlib') for tr in self.stream]
        
        self.aic_t, self.aics = self.aic_pick()
        
        self._get_second_arrival_hydrophone()
        self.depth = self.get_depth(hA=self.first_hydrophone_id, hB=self.second_hydrophone_id)

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
        # starttime = starttime - 0.25
        starttime = starttime - 0.2
        # endtime = starttime + 0.75
        endtime = starttime + 0.5
        trimmed = data.copy().trim(starttime=starttime, endtime=endtime)
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
        aic_t_idx = [aic.argmin() for aic in aics]

        # uses minimum index to retrieve the timestamp
        aic_t = [self.stream[n].times('matplotlib')[i] for n, i in enumerate(aic_t_idx)]

        return aic_t, aics
    
    def lta_sta_pick(self):
        """
        uses obspy classic lta_sta to find events
        """
        # cft = trigger.classic_sta_lta(a=self.stream, nsta=20, nlta=80)
        cfts = [trigger.classic_sta_lta(a=tr, nsta=20, nlta=80) for tr in self.stream]
        return cfts
    
    def maxamp_pick(self):
        """
        labels the maximum amplitude
        """
        traces_squared = [tr.data**2 for tr in self.stream.copy()]
        tr_argmax = [np.argmax(tr) for tr in traces_squared]
        
        max_times = []
        for n, tr_max in enumerate(tr_argmax):
            max_times.append(self.mpl_times[n][tr_max])
        return max_times
        
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
        fig, axes = plotting.plot_waveforms(self.stream)
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
    
    def _get_second_arrival_hydrophone(self):
        """
        Uses the time difference between the arrival hydrophone
        and the hydrophone above and below to determine what the 
        next hydrophone arrival time is so it can be used  to 
        estimate the depth.
        
        Parameters
        -----------
        None
        
        Return
        -----------
        None
        """
        first_idx = hydrophones[self.first_hydrophone_id]['idx']
    
        # print('the first index is:', first_idx, '; this corresponts to hydrophone id:')
        second_idx_above = first_idx - 1
        second_idx_below = first_idx + 1
        
        if second_idx_above  < 0:
            second_idx_above = 0
        
        if second_idx_below > 5:
            second_idx_below = 5
            
        above_tdelta = (num2date(self.aic_t[first_idx]) - num2date(self.aic_t[second_idx_above])).total_seconds()
        below_tdelta = (num2date(self.aic_t[first_idx]) - num2date(self.aic_t[second_idx_below])).total_seconds()
        
        # the minimum time distance is the closer one and therefore the next arrival
        argmin = np.argmin([above_tdelta, below_tdelta])

        if argmin == 0:
            # print(second_idx_above, 'is next')
            self.second_hydrophone_id = 'h'+str(second_idx_above+1)
            # print('my dumb ass self made "above" ID is:', self.second_hydrophone_id)
        elif argmin == 1:
            # print(second_idx_below, 'is next')
            self.second_hydrophone_id = 'h'+str(second_idx_below+1)
            # print('my dumb ass self made "below" ID is:', self.second_hydrophone_id)

        else:
            raise ValueError(argmin, 'should be 0 or 1')
            
    def get_depth(self, hA, hB):
        A_idx = hydrophones[hA]['idx']
        B_idx = hydrophones[hB]['idx']
        # print(hA, hydrophones[hA])
        # print(hB, hydrophones[hB])
        
        # picking_method = {'aic':self.aic_t
        #                  ,'ltasta':self.ltasta_t}
        
        t_A = num2date(self.aic_t[A_idx])
        t_B = num2date(self.aic_t[B_idx])
        
#         print('t_A', t_A)
#         print('t_B', t_B)
        
        dt = (t_A - t_B).total_seconds()
        # print('dt:',dt)
        
        dz_A = 35 + 0.5 * self.velocity_model * dt
        # print('dz_A:', dz_A)
        # print(hA, hydrophones[hA]['depth'])
        
        return hydrophones[hA]['depth'] + dz_A