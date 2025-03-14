# Program to visualize each pick from a desired picklists. 
# Picks that exceed the pick uncertainty (0.5 s) should be manually modified or flagged in picklist csv.
#
# Example:
# python vis_picklist.py --exp_name test --csv picklist_2022-08-11T03-30-00.csv


import argparse
from constants import PICKLISTS_PATH, DATA_MSEED, STALOCS_PATH
import pandas as pd
from obspy import UTCDateTime, read
from utils_process import get_rawdata
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 

    # csv parameters
    parser.add_argument("--exp_name", default='test', type=str,
                        help='folder (under data/picks-list) that contains the desired pick list csv')
    parser.add_argument("--csv", required=True, type=str,
                        help='name of csv (e.g. picklist_2024-07-15T08-00-00) that contains the desired picks')

    # data retrieval parameters
    parser.add_argument("--datadirect", default=True, action=argparse.BooleanOptionalAction, 
                        help="Pass --no-datadirect to retrieve seismic data through utils_process/Obspy")
    args = parser.parse_args()

    # read in picks csv
    EXP_PATH = PICKLISTS_PATH / args.exp_name
    df_picks = pd.read_csv(PICKLISTS_PATH / args.exp_name / f'{args.csv}.csv')
    df_picks = df_picks[df_picks['flag']==False]
    df_picks.reset_index(inplace=True)
    
    # store dictionary of station locations and channel lists
    sta_dict = dict()
    stalocs_df = pd.read_csv(STALOCS_PATH / 'all_stations_locations.csv')
    netsta_list = np.unique(df_picks['station'].to_numpy())
    for net_sta in netsta_list:
        net_sta_split = net_sta.split('_')
        net = net_sta_split[0]
        sta = net_sta_split[1]

        row = stalocs_df[(stalocs_df['network'] == net) & (stalocs_df['station'] == sta)]
        loc = row['location'].tolist()[0]
        chan_list = row['chan_list'].tolist()[0]
        chan_list = chan_list.split('.')
        sta_dict[net_sta] = [loc, chan_list]

    # iterate through each pick and plot 5 seconds before event start time and 5 seconds after end time
    for i, pick_time in enumerate(df_picks['arrivaltime_utc']):
        # store station parameters
        net_sta = df_picks.loc[i, 'station']
        netsta_split = net_sta.split('_')
        net = netsta_split[0]
        sta = netsta_split[1]
        loc = sta_dict[net_sta][0]
        if np.isnan(loc):
            loc = ''
        chan_list = sta_dict[net_sta][1]

        # store pick/event parameters
        event_start_utc = UTCDateTime( df_picks.loc[i, 'event_start_utc'])
        event_end_utc = UTCDateTime( df_picks.loc[i, 'event_end_utc'])
        pick_confidence = df_picks.loc[i, 'pick_confidence']

        # set plot to start 5 seconds before and after the event
        plot_start = event_start_utc - 5
        plot_end = event_end_utc + 5
        duration = plot_end - plot_start
        pick_time_rel = UTCDateTime(pick_time.split('+')[0]) - plot_start

        # plot unfiltered waveforms
        fig, ax = plt.subplots(3,1)
        for j, chan in enumerate(chan_list):
            if args.datadirect:
                st_1c = read(DATA_MSEED/ sta / f'{sta}.{net}.{loc}.{chan}.{plot_start.year}.{plot_start.julday}', format='MSEED')
                st_1c.trim(starttime=plot_start, endtime=plot_end)
            else:
                st_1c = get_rawdata(net, sta, '00', chan, str(plot_start), duration, samp_rate=100)
            times = st_1c[0].times()
            ax[j].plot(times, st_1c[0].data, color='black', label=chan)
            ax[j].axvline(pick_time_rel)
            ax[j].set_xlim([times[0], times[-1]])
            ax[j].legend(loc='upper left')
            if j == 0:
                st_3c = st_1c
            else:
                st_3c = st_3c + st_1c
        ax[1].set_ylabel('Counts')
        ax[2].set_xlabel('Time [s]')
        phase = df_picks.loc[i, 'phase']
        ax[0].set_title(f'{sta} {phase} pick, plot start: {str(plot_start)[:23]},\n duration: {duration} s, confidence: {pick_confidence:.3f}', fontsize=12)
        plt.tight_layout()
        plt.show()

        # plot filtered waveforms
        st_3c.filter('bandpass', freqmin=0.5, freqmax=5, zerophase=True)
        fig, ax = plt.subplots(3,1)
        for j, chan in enumerate(chan_list):
            ax[j].plot(times, st_3c.select(channel=chan)[0].data, color='black', label=chan)
            ax[j].axvline(pick_time_rel)
            ax[j].set_xlim([times[0], times[-1]])
            ax[j].legend(loc='upper left')
        ax[1].set_ylabel('Counts')
        ax[2].set_xlabel('Time [s]')
        ax[0].set_title(f'0.5 - 5 Hz Filtered\n{sta} {phase} pick, plot start: {str(plot_start)[:23]},\n duration: {duration} s, confidence: {pick_confidence:.3f}', fontsize=12)
        plt.tight_layout()
        plt.show()



    
