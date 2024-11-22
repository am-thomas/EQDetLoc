import argparse
from constants import PICKLISTS_PATH, DATA_MSEED
import pandas as pd
from obspy import UTCDateTime, read
from utils_process import get_rawdata
import matplotlib.pyplot as plt

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

    EXP_PATH = PICKLISTS_PATH / args.exp_name
    df_picks = pd.read_csv(PICKLISTS_PATH / args.exp_name / f'{args.csv}.csv')
    
    chan_list = ['BHZ', 'BH1', 'BH2']
    loc = ''
    for i, pick_time in enumerate(df_picks['arrivaltime_utc']):
        netsta_split = df_picks.loc[i, 'station'].split('_')
        net = netsta_split[0]
        sta = netsta_split[1]
        event_start_utc = UTCDateTime( df_picks.loc[i, 'event_start_utc'])
        event_end_utc = UTCDateTime( df_picks.loc[i, 'event_end_utc'])

        plot_start = event_start_utc - 5
        plot_end = event_end_utc + 5
        duration = plot_end - plot_start

        pick_time_rel = UTCDateTime(pick_time[:19]) - plot_start

        fig, ax = plt.subplots(3,1)
        for j, chan in enumerate(chan_list):
            if args.datadirect:
                st_1c = read(DATA_MSEED/ sta / f'{sta}.{net}.{loc}.{chan}.{plot_start.year}.{plot_start.julday}', format='MSEED')
                st_1c.trim(starttime=plot_start, endtime=plot_end)
            else:
                st_1c = get_rawdata(net, sta, '00', chan, str(plot_start), duration, samp_rate=100)
            times = st_1c[0].times()
            ax[j].plot(times, st_1c[0].data, color='black')
            ax[j].axvline(pick_time_rel)
            ax[j].set_xlim([times[0], times[-1]])

        ax[1].set_ylabel('Counts')
        ax[2].set_xlabel('Time [s]')

        phase = df_picks.loc[i, 'phase']
        ax[0].set_title(f'{phase} pick: {pick_time[:19]}, plot duration: {duration}')
        plt.tight_layout()
        plt.show()


    
