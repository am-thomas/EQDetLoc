# Program to apply a machine learning model (via Seisbench) on single-station data
# detections and picks are saved in constants.DETECTIONS_PATH 
#
# Example to apply EQTransformer on 3 days of station N4.M44A data:
# python MLdetector.py --net N4 --sta M44A --loc 00 --chan_list HHZ HH1 HH2 --start_time 2024-07-14T00:00:00 --days 3

import numpy as np
import utils_process
from obspy import UTCDateTime, read
from obspy.clients.fdsn import Client
client = Client('IRIS')
from constants import DETECTIONS_PATH, DATA_MSEED
import os 
import argparse
import seisbench.models as sbm
import torch 
import warnings
import pandas as pd 
import warnings

# suppress specific warnings
warnings.filterwarnings("ignore", message=".*Selected high corner frequency.*")
# suppress FutureWarning globally
warnings.simplefilter(action='ignore', category=FutureWarning)


def detect_signals(args): 
    # Load the model from SeisBench
    if args.model_name == 'EQTransformer':
        model = sbm.EQTransformer().from_pretrained(args.weights)
    elif args.model_name == 'PhaseNet': 
        if args.weights == "original":
            warnings.warn("'original' weights param does not exist for PhaseNet, using stead instead")
            args.weights = 'stead'
        model = sbm.PhaseNet().from_pretrained(args.weights)  
    else: 
        warnings.warn("Model not implemented")
        sys.exit()
    if torch.cuda.is_available():
        model.cuda()

    # make sure the detections directories for the staion exists, else create it
    STATION_PATH =  DETECTIONS_PATH / f'{args.net}_{args.sta}'
    EVENTS_PATH = STATION_PATH / 'events'
    PICKS_PATH = STATION_PATH / 'picks'
    COMB_PATH = STATION_PATH / 'combined'
    for path in [STATION_PATH, EVENTS_PATH, PICKS_PATH, COMB_PATH]:
        os.makedirs(path, exist_ok=True)

    # apply Seisbench model to 3-component data
    start = UTCDateTime(args.start_time)
    duration_ext = args.duration + 120   #retrieve 2 extra minutes of data
    event_traceids_all = []
    event_starttimes_all = []
    event_endtimes_all = []
    event_maxconfs_all = []
    pick_traceids_all = []
    pick_starttimes_all = []
    pick_endtimes_all = []
    pick_peaktimes_all = []
    pick_maxconfs_all = []
    pick_phase_all = []
    for day in range(args.days):
        print(f"Computing for day {day}, Start Time: {start}")
        skip = False
        for i in range(3):
            chan =  args.chan_list[i]
            start_ext = start - 60 
            start_ext_str = str(start_ext)
            # get 1-component data
            try:
                # retrieve directly from mseed files in folder constants.DATA_MSEED (specific to local computer)  
                if args.datadirect:
                    julday = start.julday
                    julday_str = f'{julday}' if julday >= 100 else f'0{julday}'
                    st_1c = read(DATA_MSEED/ args.sta / f'{args.sta}.{args.net}.{args.loc}.{chan}.{start.year}.{julday_str}', format='MSEED')
                else: # function to check for locally saved data and retrieve from EarthScope Web Services if it does not exist 
                    st_1c = utils_process.get_rawdata(args.net, args.sta, args.loc, chan, start_ext_str, duration_ext, args.samp_rate, plot_wave=False, save=False)
            except:
                print("Could not retrieve data from ", start, ". Skipping to next day...")
                skip = True
                break

            # combine components to one stream
            if i == 0:
                st_3c = st_1c
            else:
                st_3c = st_3c + st_1c
        
        # skip to next day if one component's data could not be retrieved
        if skip:
            start = start + (60*60*24)
            continue

        # apply EQT with default parameters
        output = model.classify(st_3c, detection_threshold=0.3, P_threshold=0.1, S_threshold=0.1)

        # store all event/detection parameters
        event_ids = [detection.trace_id for detection in output.detections]
        event_starttimes = [ detection.start_time for detection in output.detections]
        event_endtimes = [ detection.end_time for detection in output.detections]
        event_maxconf = [ detection.peak_value for detection in output.detections]    
        event_traceids_all.extend(event_ids)
        event_starttimes_all.extend(event_starttimes)
        event_endtimes_all.extend(event_endtimes)
        event_maxconfs_all.extend(event_maxconf)

        # store all phase pick parameters
        pick_ids = [pick.trace_id for pick in output.picks]
        pick_starttimes = [ pick.start_time for pick in output.picks]
        pick_endtimes = [ pick.end_time for pick in output.picks]
        pick_peaktime = [ pick.peak_time for pick in output.picks]
        pick_maxconf = [ pick.peak_value for pick in output.picks]  
        pick_phase = [ pick.phase for pick in output.picks]
        pick_traceids_all.extend(pick_ids)
        pick_starttimes_all.extend(pick_starttimes)
        pick_endtimes_all.extend(pick_endtimes)
        pick_peaktimes_all.extend(pick_peaktime)
        pick_maxconfs_all.extend(pick_maxconf)
        pick_phase_all.extend(pick_phase)

        # update to the next day
        start = start + (60*60*24)
    print('')

    # save events/detections to csv
    event_durations = np.array(event_endtimes_all) - np.array(event_starttimes_all)
    det_dict = {'trace_id': event_traceids_all, 'start_time': event_starttimes_all, 
                'end_time': event_endtimes_all, 'duration': event_durations, 'max_confidence': event_maxconfs_all}
    det_df = pd.DataFrame.from_dict(det_dict)
    det_df.to_csv(EVENTS_PATH / f'detections_{args.model_name}_{args.start_time[:10]}_days{args.days}.csv',index=False)

    # save picks to csv
    # note: picks are not specifically associated with a detection, this association is done in the next code block
    pick_dict = {'trace_id': pick_traceids_all, 'start_time': pick_starttimes_all, 
                'end_time': pick_endtimes_all, 'peak_time': pick_peaktimes_all, 
                'max_confidence': pick_maxconfs_all, 'phase': pick_phase_all}
    pick_df = pd.DataFrame.from_dict(pick_dict)
    pick_df.to_csv(PICKS_PATH / f'picks_{args.model_name}_{args.start_time[:10]}_days{args.days}.csv',index=False)

    # iterate through each event and find picks within the event duration (associate phases with an event)
    num_events = len(event_traceids_all)
    flags = ['']*num_events
    event_Ptimes = [np.nan] * num_events
    event_Stimes = [np.nan] * num_events
    event_Pmaxconf = [np.nan] * num_events
    event_Smaxconf = [np.nan] * num_events
    additional_Ptimes = [[] for _ in range(num_events)]   # list to store additional P picks if many are detected in the same event
    additional_Stimes = [[] for _ in range(num_events)]   # same as above but for S picks
    for event_idx in range(num_events):
        found_P = False
        found_S = False
        for pick_idx in range(len(pick_traceids_all)):
            pick_peaktime = pick_peaktimes_all[pick_idx]
            pick_phase = pick_phase_all[pick_idx]
            # check if a pick falls within a detected event
            if pick_peaktime >= event_starttimes_all[event_idx] and pick_peaktime <= event_endtimes_all[event_idx]:
                if pick_phase == 'P' and found_P == False:
                    event_Ptimes[event_idx] = pick_peaktime
                    event_Pmaxconf[event_idx] = pick_maxconfs_all[pick_idx]
                    found_P = True
                elif pick_phase == 'S' and found_S == False:
                    event_Stimes[event_idx] = pick_peaktime
                    event_Smaxconf[event_idx] = pick_maxconfs_all[pick_idx]
                    found_S = True
                elif found_P or found_S:
                    print('[FLAG] Multiple phases of the same type found in event', event_idx)
                    if found_P and found_S:
                        flags[event_idx] = 'True, more than one P and S arrival detected in the event'
                    elif found_P:
                        flags[event_idx] = 'True, more than one P arrival detected in the event'
                    else:
                        flags[event_idx] = 'True, more than one S arrival detected in the event'
                    
                    # add additional arrival times to list
                    if pick_phase == 'P':
                        additional_Ptimes[event_idx].append(str(pick_peaktime)[:19].replace(' ','T'))
                    else:
                        additional_Stimes[event_idx].append(str(pick_peaktime)[:19].replace(' ','T'))

        if found_P == False and found_S == False:
            flags[event_idx] = 'True, missing P and S arrival'

    # create combined dataframe of events, with associated pick times
    comb_dict = {'trace_id': event_traceids_all, 'start_time': event_starttimes_all, 
                    'end_time': event_endtimes_all, 'duration': event_durations, 
                    'event_maxconfidence': event_maxconfs_all, 'P_time': event_Ptimes,
                    'P_maxconfidence': event_Pmaxconf, 'S_time': event_Stimes,
                    'S_maxconfidence': event_Smaxconf, 'flag': flags, 
                    'additional_Ps': additional_Ptimes, 'additional_Ss': additional_Stimes}
    combined_df = pd.DataFrame(comb_dict)
    combined_df.to_csv(COMB_PATH / f'combined_{args.model_name}_{args.start_time[:10]}_days{args.days}.csv',index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 

    # Station Parameters 
    parser.add_argument("--net", default='NW', type=str, help='network code')
    parser.add_argument("--sta", default='HQIL', type=str, help='station code') 
    parser.add_argument("--loc", default='', type=str, help='location code') 
    parser.add_argument('-l','--chan_list', nargs='+', default=['BHZ','BH1','BH2'], help='list of channel codes')
    parser.add_argument("--samp_rate", default=40, type=int, help='station sampling rate (Hz)') 

    # Time and algorithm parameters
    parser.add_argument("--start_time", default='2015-04-01T00:00:00', type=str,
                        help='start time of desired time series (midnight of UTC time)')
    parser.add_argument("--duration", default=60 * 60 * 24, type=float, help='duration (usually 24 h) of data to save to file')
    parser.add_argument("--days", default=7, type=int)

    # Data retrieval parameters
    parser.add_argument("--datadirect", default=True, action=argparse.BooleanOptionalAction, 
                        help="Directly searches for locally saved mseed files in the constants.DATA_MSEED folder. Pass --no-datadirect to retrieve seismic data through utils_process/Web Services")
                        
    # Machine learning model parameters 
    # See list of available models here: https://seisbench.readthedocs.io/en/stable/pages/models.html#models-integrated-into-seisbench")
    parser.add_argument("--model_name", default='EQTransformer', 
                         help="""Options: 
                         # EQTransformer
                         # PhaseNet
                         # GPD
                         # CRED
                         # BasicPhaseAE
                         """)
                         
    parser.add_argument("--weights", default='original', 
                        help="""Pretrained weights to load. Options: 
                                    # original
                                    # original_nonconservative
                                    # ethz   
                                    # scedc
                                    # stead
                                    # geofon
                                    # neic 
                                    """)
    args = parser.parse_args()
    
    # apply ML detector and save results to files
    detect_signals(args)

