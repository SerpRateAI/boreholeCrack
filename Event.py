import numpy as np
import obspy
from hydrophone_data_processing import load, useful_variables, plotting, signal_processing
import scipy.signal as signal
import pandas as pd
import matplotlib.dates as dates
import obspy.signal.trigger as trigger
import sys
import config


hydrophone_metadata = config.hydrophone_metadata['141']

class Event:
    """
    Data holder class for cracking event from hydrophone
    """
    def __init__(self, id, starttime, init_first_hphone, waveforms, velocity_model=1750, hanning=True):
        # INITIALIZE DATA
        self.id = id
        self.data = waveforms
        self.velocity_model = velocity_model
        self._max_dx = 70 # meters spacing between hydrophones
        self._max_dt = self._max_dx / self.velocity_model
        starttime = dates.num2date(starttime)
        self.starttime = obspy.UTCDateTime(starttime)
        self.first_hydrophone_id = init_first_hphone
        self.hanning = hanning
        self.stream = self.get_waveforms(starttime=self.starttime, hanning=self.hanning)
        
        # DEPTH CALCULATION
        self.maxes, self.aic_t, self.aics = self.aic_pick()
        
        self._get_first_second_hydrophones()
        
        self.hphone1_time = self.aic_t[self.first_hydrophone_id]
        self.hphone2_time = self.aic_t[self.second_hydrophone_id]
        
        self.get_depth()
        
        # RADIUS CALCULATION
        self.get_pwaveforms()
        self.get_aicp()
        self.calc_radius()
        
        

    # def get_waveforms(self, starttime):
    def get_waveforms(self, starttime, hanning):
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
        if hanning == True:
            trimmed.taper(type='hann', max_percentage=0.5)
        return trimmed
    
    def get_pwaveforms(self):
        """
        Creates class variables for p arrrival estimation
        """
        window_start = self.starttime - 0.2
        window_end = self.starttime + 0.3
        self.p_waveforms = self.data.copy().trim(starttime=window_start, endtime=window_end)
        self.p_waveforms.filter(type='highpass', freq=200, zerophase=False, corners=1)
        

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
    
    def get_aicp(self):
        """
        Creates variables to calculate aic for parrival time. Also calculates parrival time
        """
        self.aic_p = trigger.aic_simple(self.p_waveforms[self.first_hydrophone_id])
        t = self.p_waveforms[self.first_hydrophone_id].times('matplotlib')
        self.parrival = t[np.argmin(self.aic_p)]
        self.parrival = dates.num2date(self.parrival)

    def _get_first_second_hydrophones(self):
        # NOTE: THIS ONLY WORKS IF YOU ARE GENERATING A CATALOG, FAILS OTHERWISE
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
        
        dz_phone = np.min([self.first_hydrophone_id, self.second_hydrophone_id])
        dz_phone_label = 'h' + str(dz_phone+1)
        
        hydrophone_depth = hydrophone_metadata[dz_phone_label]['depth']
        
        dz = 35 - 0.5 * dt * self.velocity_model * sign
        
        self.relative_depth = dz
        
        z = dz + hydrophone_depth
        
        self.depth = z
        

    def calc_radius(self):
        """
        calculates radial distance event is from borehole in meters
        """
        vrock = 4500 # m/s 5500 default
        vtm = self.velocity_model
        dz = self.depth - hydrophone_metadata['h'+str(self.first_hydrophone_id+1)]['depth']
        
        mode_t = dates.num2date(self.hphone1_time)
        dt = (mode_t - self.parrival).total_seconds()
        
        self.radius =  np.sqrt(vrock**2 * dt**2 - dz**2)