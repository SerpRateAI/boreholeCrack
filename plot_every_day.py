
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as dates
import obspy
from hydrophone_data_processing import load, useful_variables, plotting, signal_processing
# import event_detector_pipeline as edp
import event_pipeline
import sys

args = sys.argv
_, borehole = args

def get_data(year, day):
    # paths = useful_variables.make_hydrophone_data_paths(borehole='a', year=year, julian_day=day)
    # paths = useful_variables.make_hydrophone_data_paths(borehole='b', year=year, julian_day=day)
    paths = useful_variables.make_hydrophone_data_paths(borehole=borehole, year=year, julian_day=day)
    waveforms = load.import_corrected_data_for_single_day(paths=paths)
    return waveforms

def plot_day(waveforms, year, day):
    fig, ax = plt.subplots(6, 1, figsize=(15, 15), sharex=True, sharey=True)
    for n, tr in enumerate(waveforms):
        ax[n].plot(tr.times('matplotlib'), tr.data, color='black', linewidth=0.5)
    ax[n].set_ylim(-10, 10)
    ax[n].xaxis.set_major_formatter(plotting.PrecisionDateFormatter("%H:%M:%S.{ms}"))
    ax[0].set_title('year:{y} day:{d}'.format(y=year, d=day), fontsize=15)
    fig.tight_layout()
    # fig.savefig('everyday/{y}.{d}.pdf'.format(y=year, d=day), bbox_inches='tight')
    fig.savefig('everyday/{b}/{y}.{d}.pdf'.format(b=borehole, y=year, d=day), bbox_inches='tight')
    plt.close()
    
if __name__=='__main__':
    # for year in (2019, 2020):
    for year in (2020,):
        for day in np.arange(1, 366, 1):
            try:
                waveforms = get_data(year=year, day=day)
                plot_day(waveforms=waveforms, year=year, day=day)
                del waveforms
                print('plotted {y}.{d} raw data'.format(y=year, d=day))
            except:
                pass
