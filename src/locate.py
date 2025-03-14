import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from constants import PICKLISTS_PATH, EQLOCS_PATH
import locate_utils
import warnings
import os
from datetime import datetime, timedelta

# suppress specific warnings
warnings.filterwarnings("ignore", message=".*The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*")

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 

    # csv parameters
    parser.add_argument("--exp_name", default='test', type=str,
                        help='folder (under data/picks-list) that contains the desired pick list csv')
    parser.add_argument("--csv", required=True, type=str,
                        help='name of csv (e.g. picklist_2024-07-15T08-00-00) that contains the desired picks')
    parser.add_argument("--invert4depth", default=False, action=argparse.BooleanOptionalAction, 
                        help="True if inverting for depth. False if keeping depth fixed at 10 km")
    parser.add_argument("--ploterrormatrix", default=False, action=argparse.BooleanOptionalAction, 
                        help="True to plot error ellipses. False otherwise")
    parser.add_argument("--damp", default=0.2, type=float,
                        help='constant factor to multiply the damping diagonal matrix by. should be similar in order of magnitude to the inverse of the standard deviation of (obs arrival time - predicted arrival time based on initial guess)')
    parser.add_argument("--uncertainty_atm", default=0.5, type=float,
                        help='uncertainty of arrival time pick in seconds')
    args = parser.parse_args()

    df_picks = pd.read_csv(PICKLISTS_PATH / args.exp_name / f'{args.csv}.csv')
    df_picks = df_picks[df_picks['flag']==False]       #ignore flagged events

    # read data from pick list csv
    n_arrivals,reftime,sta,ylat,xlon,elv,atm,phases = locate_utils.readpickscsv(df_picks)
    print('reference time:', reftime)

    # set first initial guess of location, based on the station with the first arrival time
    imin = np.argmin(atm)
    minatm_phase = phases[imin]
    vel_dict = {'P':5.8, 'S':3.36}             # velocities at 10 km and above in the IASP91 model
    lonq = xlon[imin]                          # initial guess of source longitude
    latq = ylat[imin]                          # initial guess of source latitude
    hq = 10.                                   # initial guess of source depth
    tq = atm[imin]-hq/vel_dict[minatm_phase]   # initial guess of origin time (s relative to reference time)
    startloc = [tq,lonq,latq,hq]
    print('initial guess')
    print('[origin time, lon, lat, depth]:', startloc)
    print('')

    # perform least squares inversion
    loc, covm, otime_utc_final, k, converg_crit, percchange_converg, loc_nodamp = locate_utils.locatequake(n_arrivals,reftime,ylat,xlon,elv,atm,phases,'IASP91.csv',startloc,
                                         args.invert4depth,damp_factor=args.damp,sigmaT=args.uncertainty_atm,annotate=False)
    
    # compute and plot error ellipse matrix for each pair of source parameters
    if args.ploterrormatrix:
        fig, ax = plt.subplots(3,2, figsize=(6,6))
        ax = ax.flatten()
    majors = dict()
    minors = dict()
    rotas = dict()
    pairs = ['otime_lon', 'otime_lat', 'lon_lat', 'otime_depth', 'lon_depth', 'lat_depth']
    paramidx_dict = {'otime':0, 'lon':1, 'lat':2, 'depth':3}
    labels_dict = {'otime': 'origin time [s]', 'lat': 'latitude [deg]', 'lon': 'longitude [deg]', 'depth': 'depth [km]'}
    # iterate through all parameter pairs
    for i, pair in enumerate(pairs):
        params = pair.split('_')
        param1 = params[0]
        param2= params[1]

        if args.ploterrormatrix:
            ax[i].set_xlabel( labels_dict[param1] )
            ax[i].set_ylabel( labels_dict[param2] )
    
        # skip pairs with "depth" parameter if not inverting for depth
        if not args.invert4depth and pair in ['otime_depth','lon_depth', 'lat_depth']:
            majors[pair] = np.nan
            minors[pair] = np.nan
            rotas[pair] = np.nan
            continue

        # get and store major length, minor length, and angle (rota) of error ellipse
        major,minor,rota = locate_utils.merrors(covm,param1,param2)
        majors[pair] = major
        minors[pair] = minor
        rotas[pair] = rota

        # plot ellipse
        if args.ploterrormatrix:
            x = loc[-1][paramidx_dict[param1]]
            y = loc[-1][paramidx_dict[param2]]
            elip = Ellipse((x,y),major,minor,angle=rota)
            elip.set_alpha(0.1)
            ax[i].add_artist(elip)
            ax[i].plot(x,y,'o',alpha=0.7)
            #ax[i].plot(loc[0][paramidx_dict[param1]], loc[0][paramidx_dict[param2]], 'x', color='red')
            xmin = x - (major)
            xmax = x + (major)
            ymin = y - (major)
            ymax = y + (major)
            ax[i].set_xlim([xmin,xmax])
            ax[i].set_ylim([ymin,ymax])

    if args.ploterrormatrix:
        plt.tight_layout()
        plt.show()


    # read in existing EQ locations csv and check if a solution already exists for the exp_name and pick_file
    df_allloc = pd.read_csv(EQLOCS_PATH / 'all_eqlocations.csv')
    match = (df_allloc['exp_name'] == args.exp_name) & (df_allloc['pick_csv'] == args.csv)
    check_to_save = True
    if match.any():   # if solution exists, ask user if they want to overwrite
        print('')
        print('A solution is already recorded in all_eqlocations.csv for this experiment name and pick file. If you continue, you will overwrite the existing solution. Otherwise, the existing solution will remain and the new solution will not be recorded in the csv')
        response = input("Do you want to continue and save the new location? (y/n): ").strip().lower()

        if response == 'n':
            print('Exiting program without saving solution to csv...')
            exit()
        elif response == 'y':
            df_allloc = df_allloc[~match]
            check_to_save = False
            print('Adding new solution to all_eqlocations.csv...')
        else:
            print('Response other than y/n was recorded. Exiting program without saving solution to csv...')
            exit()

    # ask user if they want to save location to file
    if check_to_save == True:
        response = input("Do you want to save the location to csv (y/n)? ").strip().lower()
        if response == 'n':
            print('Exiting program without saving solution to csv...')
            exit()
        elif response == 'y':
            print('Adding new solution to all_eqlocations.csv...')
        else:
            print('Response other than y/n was recorded. Exiting program without saving solution to csv...')
            exit()

    # save solution to csv
    new_row = pd.DataFrame({'reference_time_utc': reftime, 'exp_name': args.exp_name, 'pick_csv': args.csv,
            'source_origintime': otime_utc_final, 'source_latitude_deg': loc[-1][2], 'source_longitude_deg': loc[-1][1],
            'source_depth': loc[-1][3], 'nodamp_origintime': datetime.fromisoformat(reftime) + timedelta(seconds=loc_nodamp[0]), 
            'nodamp_latitude_deg': loc_nodamp[2], 'nodamp_longitude_deg': loc_nodamp[1], 
            'nodamp_source_depth': loc_nodamp[3], 'num_arrivals':n_arrivals, 'damp_factor': args.damp,
            'uncertainty_atm_s': args.uncertainty_atm, 'initial_guess': str(startloc), 'invert4depth': args.invert4depth,
            'num_iterations': k, 'convergence_criterion': converg_crit, 'criterionchange_percent': percchange_converg,
            'errmajor_otime_lon': majors['otime_lon'], 'errminor_otime_lon': minors['otime_lon'], 
            'errangle_otime_lon': rotas['otime_lon'], 'errmajor_otime_lat': majors['otime_lat'],
            'errminor_otime_lat': minors['otime_lat'], 'errangle_otime_lat': rotas['otime_lat'],
            'errmajor_lon_lat': majors['lon_lat'], 'errminor_lon_lat': minors['lon_lat'], 
            'errangle_lon_lat': rotas['lon_lat'], 'errmajor_otime_depth': majors['otime_depth'],
            'errminor_otime_depth': minors['otime_depth'], 'errangle_otime_depth': rotas['otime_depth'],
            'errmajor_lon_depth': majors['lon_depth'], 'errminor_lon_depth': minors['lon_depth'],
            'errangle_lon_depth': rotas['lon_depth'], 'errmajor_lat_depth': majors['lat_depth'],
            'errminor_lat_depth': minors['lat_depth'], 'errangle_lat_depth': rotas['lat_depth']}, index=[0])
    df_allloc_ext = pd.concat( [df_allloc, new_row], ignore_index=True)
    os.rename(EQLOCS_PATH / 'all_eqlocations.csv', EQLOCS_PATH / 'all_eqlocations_beforeupdate.csv')
    df_allloc_ext.to_csv(EQLOCS_PATH / 'all_eqlocations.csv', index=False)

    


