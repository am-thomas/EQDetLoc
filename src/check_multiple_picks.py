import argparse
import pandas as pd
from constants import DETECTIONS_PATH, STALOCS_PATH, DATA_MSEED
from obspy import UTCDateTime
import numpy as np
import matplotlib.pyplot as plt
from obspy import read
from utils_process import get_rawdata

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 

    # parameters to retrieve the desired detection csvs
    parser.add_argument("--net_stalist", required=True, nargs='+', 
                        help ='list of stations in the format NET_STATION (e.g. Z5_NAKA, N4_L42A) to iterate through')
    parser.add_argument("--start_day", default='2024-07-14', type=str,
                        help='start day of combined csv. must be consistent across all stations')
    parser.add_argument("--days", default=7, type=int, help='number of days in experiment. must be consistent across all stations')
    parser.add_argument("--model_name", default='EQTransformer', type=str,
                        help='name of ML model used for detection events and phases')
    parser.add_argument("--exp_name", default='test', type=str,
                        help='folder name to save under data/picks-list to store all pick list csvs')
    
    # data retrieval parameters
    parser.add_argument("--datadirect", default=True, action=argparse.BooleanOptionalAction, 
                        help="Pass --no-datadirect to retrieve seismic data through utils_process/Obspy")
    args = parser.parse_args()

    # get and store combined csvs for each station
    df_list = []
    new_netsta_list = []
    for net_sta in args.net_stalist:
        try:
            df = pd.read_csv(DETECTIONS_PATH / net_sta / 'combined' / f'combined_{args.model_name}_{args.start_day}_days{args.days}.csv')
            df_list.append(df)
            new_netsta_list.append(net_sta)
        except FileNotFoundError:
            response = input(f"No combined csv file found for {net_sta}. Skipping to next station...")
            continue

    # store dictionary of station locations and channel lists
    sta_dict = dict()
    stalocs_df = pd.read_csv(STALOCS_PATH / 'all_stations_locations.csv')
    for net_sta in new_netsta_list:
        net_sta_split = net_sta.split('_')
        net = net_sta_split[0]
        sta = net_sta_split[1]

        row = stalocs_df[(stalocs_df['network'] == net) & (stalocs_df['station'] == sta)]
        loc = row['location'].tolist()[0]
        chan_list = row['chan_list'].tolist()[0]
        chan_list = chan_list.split('.')
        sta_dict[net_sta] = [loc, chan_list]

    # check for flags in each df
    for i, df in enumerate(df_list):
        net_sta = new_netsta_list[i]
        df = df[df['flag'].notna()]
        df_cut = df[df['flag']!='True, missing P and S arrival']
        df_cut.reset_index(inplace=True)
        
        for event_idx in range(len(df_cut)):
            extra_picks = []
            loc = sta_dict[net_sta][0]
            chan_list = sta_dict[net_sta][1]

            event_start_utc = UTCDateTime( df.loc[i, 'start_time'])
            event_end_utc = UTCDateTime( df.loc[i, 'end_time'])

            plot_start = event_start_utc - 5
            plot_end = event_end_utc + 5
            duration = plot_end - plot_start

            # store relative P and S times if they exist
            P_time = df.loc[i, 'P_time']
            Ptime_exists = not (isinstance(P_time, (float, int)) and np.isnan(P_time))
            S_time = df.loc[i, 'S_time']
            Stime_exists = not (isinstance(S_time, (float, int)) and np.isnan(S_time))
            if Ptime_exists:
                Ptime_rel = UTCDateTime(P_time[:26]) - plot_start
            if Stime_exists:
                Stime_rel = UTCDateTime(S_time[:26]) - plot_start

            # get list of additional picks
            extra_Ps = df.loc[i, 'additional_Ps'][2:-2].split(',')
            extra_Ss = df.loc[i, 'additional_Ss'][2:-2].split(',')
            extra_picks = []
            for pick in [extra_Ps, extra_Ss]:
                extra_picks.extend(pick)
            
            # compute relative times (w.r.t plot starts) of extra picks
            extrapicks_rel = []
            for pick in extra_picks:
                if pick == '':
                    continue
                extrapicks_rel.append(UTCDateTime(pick[:26]) - plot_start)

            # plot each component waveforms and picks
            fig, ax = plt.subplots(3,1, figsize=(10,7))
            for j, chan in enumerate(chan_list):
                if args.datadirect:
                    st_1c = read(DATA_MSEED/ sta / f'{sta}.{net}.{loc}.{chan}.{plot_start.year}.{plot_start.julday}', format='MSEED')
                    st_1c.trim(starttime=plot_start, endtime=plot_end)
                else:
                    st_1c = get_rawdata(net, sta, '00', chan, str(plot_start), duration, samp_rate=100)
                times = st_1c[0].times()
                ax[j].plot(times, st_1c[0].data, color='black', label=chan)
                ax[j].set_xlim([times[0], times[-1]])

                if Ptime_exists:
                    ax[j].axvline(Ptime_rel, label='current P', color='red')
                if Stime_exists:
                    ax[j].axvline(Stime_rel, label='current S', color='blue')
                for extrapick in extrapicks_rel:
                    ax[j].axvline(extrapicks_rel, label='extra picks', color='green')

                ax[j].legend(loc='upper left', fontsize='small')
            ax[0].set_title(f'{sta}, plot start time: {str(plot_start)[:23]}, duration: {duration} s')
            ax[1].set_ylabel('Counts')
            plt.tight_layout()
            plt.show()

