import obspy
import obspy.signal.trigger as trigger
import matplotlib.pyplot as plt
import numpy as np
from hydrophone_data_processing import load, preprocessing, tempmatch, useful_variables, plotting
# from matplotlib.dates import num2date
import matplotlib.dates as dates
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
        self._max_dx = 70 # meters spacing between hydrophones
        self._max_dt = self._max_dx / self.velocity_model
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
        starttime = starttime - 0.2
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
        # aic_t_idx = [aic.argmin() for aic in aics]
        # diffs = [np.diff(aic) for aic in aics]
        # diffs = [np.diff(aic, n=20) for aic in aics]
        diffs = [np.diff(aic, n=1) for aic in aics]
        maxes = [np.argmax(diff) for diff in diffs]
        
        # print(maxes)

        # uses minimum index to retrieve the timestamp
        # aic_t = [self.stream[n].times('matplotlib')[i] for n, i in enumerate(aic_t_idx)]
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
    
#     def _get_second_arrival_hydrophone(self):
        
#         if self.first_hydrophone_id in ['h1', 'h2', 'h3']:
#             self.first_hydrophone_id = 'h3'
#             self.second_hydrophone_id = 'h4'
#             return None
        
#         elif self.first_hydrophone_id == 'h6':
#             self.first_hydrophone_id = 'h5'
#             self.second_hydrophone_id = 'h6'
#             return None
        
#         elif self.first_hydrophone_id in ['h4', 'h5']:
#             # get list index for first hydrophone
#             h_Aidx = hydrophones[self.first_hydrophone_id]['idx']
#             # get the arrival times in datetime format from the aic
#             # picker
#             times = np.array(dates.num2date(np.array(self.aic_t)))
            
#             # get the arrival time for thei first hydrophone
#             # t_A = times[hydrophones[self.first_hydrophone_id]['idx']]
#             t_A = times[h_Aidx]
            
#             # take the difference between all the arrival times
#             # and the arrival time for the first hydrophone
#             # tdiff = np.array([t.total_seconds() for t in (ta - times)])
#             tdiff = np.array([t.total_seconds() for t in (t_A - times)])
            
#             # if the time diffrence is greater then the maximum difference
#             # then the other hydrophone is in the pair
            
#             if np.abs(tdiff[h_Aidx-1]) > self._max_dt:
                
#                 self.second_hydrophone_id = 'h'+str(h_Aidx+1)
                
#             else:
#                 self.second_hydrophone_id = 'h'+str(h_Aidx-1)
                
            
#         else:
#             raise ValueError('nothing works.')
            
            

    def _get_second_arrival_hydrophone(self):
        
        if self.first_hydrophone_id in ['h1', 'h2', 'h3']:
            self.first_hydrophone_id = 'h3'
            self.second_hydrophone_id = 'h4'
            return None
        
        elif self.first_hydrophone_id == 'h4':
            # self._hydrophone_above = 'h3'
            # self._hydrophone_below = 'h5'
            # self.second_hydrophone_id = 'h3'
            self.second_hydrophone_id = 'h5'
            return None
            
        elif self.first_hydrophone_id == 'h5':
            # self._hydrophone_above = 'h4'
            # self._hydrophone_below = 'h6'
            # self.second_hydrophone_id = 'h6'
            self.first_hydrophone_id = 'h4'
            self.second_hydrophone_id = 'h5'
            return None
            
        elif self.first_hydrophone_id == 'h6':
            self.second_hydrophone_id = 'h5'
            return None
        
        else:
            raise ValueError('this is not a hydrophone ID:{}'.format(self.first_hydrophone_id))
            
#         first_idx = hydrophones[self.first_hydrophone_id]['idx']
#         second_idx_above = hydrophones[self._hydrophone_above]['idx']
#         second_idx_below = hydrophones[self._hydrophone_below]['idx']
        
#         above_tdelta = (num2date(self.aic_t[first_idx]) - num2date(self.aic_t[second_idx_above])).total_seconds()
#         below_tdelta = (num2date(self.aic_t[first_idx]) - num2date(self.aic_t[second_idx_below])).total_seconds()
        
#         # the minimum time distance is the closer one and therefore the next arrival
#         argmin = np.argmin([above_tdelta, below_tdelta])

#         if argmin == 0:
#             self.second_hydrophone_id = 'h'+str(second_idx_above+1)
#         elif argmin == 1:
#             self.second_hydrophone_id = 'h'+str(second_idx_below+1)

#         else:
#             raise ValueError(argmin, 'should be 0 or 1')
            
    def get_depth(self, hA, hB):
        A_idx = hydrophones[hA]['idx']
        B_idx = hydrophones[hB]['idx']

        # t_A = num2date(self.aic_t[A_idx])
        t_A = dates.num2date(self.aic_t[A_idx])
        # t_B = num2date(self.aic_t[B_idx])
        t_B = dates.num2date(self.aic_t[B_idx])
        
        dt = (t_A - t_B).total_seconds()
        # print('the dt is:', dt)
        
        dz_A = 35 + 0.5 * self.velocity_model * dt

        return hydrophones[hA]['depth'] + dz_A