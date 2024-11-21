from constants import DETECTIONS_PATH, NUM_SECS_DAY, PICKLISTS_PATH, STALOCS_PATH
import argparse
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

def get_picklist(net_stalist, start_day, days, model_name):

    # get and store combined csvs for each station
    df_list = []
    for net_sta in args.net_stalist:
        df = pd.read_csv(DETECTIONS_PATH / net_sta / 'combined' / f'combined_{args.model_name}_{args.start_day}_days{args.days}.csv')
        lendf_orig = len(df)
        df = df[df['event_maxconfidence'] >= args.detection_threshold]
        lenddf_cut = len(df)
        if lendf_orig != lenddf_cut:
            print(f'Ignoring events detected in {net_sta} which have a max confidence < {args.detection_threshold}')
        df_list.append(df)

    # store a dictionary of [latitude[deg], longitude[deg], elevation[m]] values for each station
    stacoords = dict()
    stalocs_df = pd.read_csv(STALOCS_PATH / 'all_stations_locations.csv')
    for net_sta in args.net_stalist:
        net_sta_split = net_sta.split('_')
        net = net_sta_split[0]
        sta = net_sta_split[1]

        row = stalocs_df[(stalocs_df['network'] == net) & (stalocs_df['station'] == sta)]
        if len(row) == 0:
            print('')
            print(f'Station {net_sta} not included in all_stations_locations.csv. Please include this station or check for typos in csv. Exiting program...')
            exit()

        lat = row['latitude_deg'].tolist()[0]
        lon = row['longitude_deg'].tolist()[0]
        elv = row['elevation_m'].tolist()[0]
        stacoords[net_sta] = [lat, lon, elv]


    # create a dataframe of reference times (10-minute time steps starting from YYYY-MM-DDT00:00:00)
    tolerance_s = 60*10  
    iterations = int(NUM_SECS_DAY/tolerance_s) * days
    start_time = pd.to_datetime(start_day + 'T00:00:00.000000Z')
    all_times = []
    time = start_time
    for i in range(iterations):
        all_times.append(time)
        time = time + pd.Timedelta(seconds=tolerance_s)
    df_reftimes = pd.DataFrame()
    df_reftimes['ref_time'] = all_times

    # create a merged dataframe of phase arrivals that are within 10 minutes of the reference times
    merged_df = df_reftimes
    for idx, net_sta in enumerate(net_stalist):
        for phase in ['P', 'S']:
            df = df_list[idx]
            df = df[[f'{phase}_time', f'{phase}_maxconfidence']]
            df = df.sort_values(by=f'{phase}_time')
            df[f'{phase}_time'] = pd.to_datetime(df[f'{phase}_time'])
            
            # drop rows with nan values in phase time column
            df = df.dropna(subset=[f'{phase}_time'])
            if len(df) == 0:
                continue 

            # rename columns to specify the staton
            df.rename(columns={f'{phase}_time': f'{net_sta}_{phase}_time', 
                      f'{phase}_maxconfidence': f'{net_sta}_{phase}_maxconfidence'}, inplace=True)
            
            merged_df = pd.merge_asof(merged_df, df, 
                        left_on=f'ref_time', 
                        right_on=f'{net_sta}_{phase}_time', 
                        tolerance=pd.Timedelta(seconds=tolerance_s))
    
    # drop indices and rows with all NaN values 
    merged_df_cols = merged_df.columns
    columns_to_check = [col for col in merged_df_cols if col != 'ref_time']
    merged_df = merged_df.dropna(subset=columns_to_check, axis=0, how='all')
    merged_df = merged_df.reset_index(drop=True)

    # create individual csv files for picks within 10 min of ref times that contain more than 2 phase arrivals
    saved_refs = []
    for idx, ref_time in enumerate(merged_df['ref_time']):
        station_list = []
        station_latitudes = []
        station_longitudes = []
        station_elevations = []
        pick_times = []
        pick_relativetimes = []
        pick_maxconfidences = []
        pick_phases = []
        for col in merged_df_cols[1:]:
            if col.endswith('time') and pd.notna( merged_df.loc[idx, col] ):
                colsplit = col.split('_')
                net_sta = f'{colsplit[0]}_{colsplit[1]}'
                station_list.append(net_sta)
                station_latitudes.append( stacoords[net_sta][0] )
                station_longitudes.append( stacoords[net_sta][1] )
                station_elevations.append( stacoords[net_sta][2] )

                phase = colsplit[2]
                pick_time = merged_df.loc[idx, f'{net_sta}_{phase}_time'] 
                pick_times.append( pick_time)
                pick_relativetimes.append( (pick_time - ref_time).total_seconds() )
                pick_maxconfidences.append( merged_df.loc[idx, f'{net_sta}_{phase}_maxconfidence'] )
                pick_phases.append(phase)

        len_picks_1event = len(pick_times)
        unique_stas = np.unique(station_list)
        if len_picks_1event > 2 and len(unique_stas) >= args.minstations:
            reftime_col = [ref_time] * len_picks_1event
            event_df = pd.DataFrame({'reftime_utc': reftime_col, 'station':station_list, 'station_latitude_deg': station_latitudes, 
                                     'station_longitude_deg': station_longitudes, 'station_elevation_m': station_elevations, 
                                     'arrivaltime_utc': pick_times, 'relative_arrivaltime_s':pick_relativetimes, 
                          'phase': pick_phases, 'pick_confidence': pick_maxconfidences})
            reftime_title = str(ref_time)[:19].replace(':','-').replace(' ','T')
            event_df.to_csv(EXP_PATH / f'picklist_{reftime_title}.csv', index=False)
            saved_refs.append(ref_time)

    # raise a flag if two adjacent ref times were saved, contain more than 2 phase arrivals. They may need to be combined.
    print('')
    for j, savedref in enumerate(saved_refs):
        if j != 0 and (savedref - saved_refs[j-1]).total_seconds() == 600:
            print('[FLAG] CSVs with the following two reference times may need to be merged/reevaluated',
                   str(savedref)[:19].replace(' ', 'T'), str(saved_refs[j-1])[:19].replace(' ', 'T'))
        
    return merged_df


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
    
    # threshold parameters
    parser.add_argument("--detection_threshold", default=0.5, type=float, 
                        help='If the max confidence of an event is below the detection threshold, events and their associated picks will be ignored')
    parser.add_argument("--minstations", default=5, type=int,
                         help='If the number of stations with phase arrivals for a reference time is below this value, a picklist csv will not be created.')
    
    # saving parameter
    parser.add_argument("--exp_name", default='test', type=str,
                        help='folder name to save under data/picks-list to store all pick list csvs')
    args = parser.parse_args()

    EXP_PATH = PICKLISTS_PATH / args.exp_name
    os.makedirs(EXP_PATH, exist_ok=True) 

    time = datetime.now()
    command = "python " + " ".join(sys.argv) + "\n"
    logfile = open(EXP_PATH / "get_picklist_log", "a")
    logfile.write(str(time) + "\n" + command + "\n")
    logfile.close()

    get_picklist(args.net_stalist, args.start_day, args.days, args.model_name)
