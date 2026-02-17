# Program to visualize maximum amplitudes and and compute local magnitudes
# Averaged local magnitudes are saved in the data/eq-locations/all_eqlocations.csv. MLs for each station are saved in the input pick list
# Earthquake locations must be computed and saved before running this file. 

import argparse
from constants import PICKLISTS_PATH, DATA_MSEED, STALOCS_PATH, METADATA_PATH, EQLOCS_PATH
import pandas as pd
from obspy import UTCDateTime, read, read_inventory
from obspy.signal.invsim import paz_to_freq_resp, cosine_sac_taper
from utils_process import get_rawdata
import matplotlib.pyplot as plt
import numpy as np
from locate_utils import get_dists_azimuths

def get_mL_centralUS(r,max_amp,S):
    # Function to compute local magnitude for the Central United States (Miao and Langston, 2007)
    log_A = np.log10(max_amp)
    m_L = log_A + 0.939*np.log10(r/100) - 0.000276*(r-100) + 3.0 + S
    return m_L

def get_mL_Norway(r,max_amp,S):
    # Function to compute local magnitude for the Norway (Alsaker et al, 1991)
    a = 0.91
    b = 0.00087
    log_A = np.log10(max_amp)
    m_L = log_A + a*np.log10(r/60) - b*(r-60) + 2.68 + S
    return m_L

def get_mL_Norway_noWA(D,max_amp):
    # Function to compute local magnitude for Norway using a Wood-Anderson approximation model (Alsaker et al, 1991)
    m_L = 0.925*np.log10(max_amp) + 0.91*np.log10(D) +0.00087*D -1.31
    return m_L

def get_mL_Ethirift(r,max_amp,S):
    # Function to compute local magnitude for Ethiopia within the rift area (Keir et al, 2006)
    log_A = np.log10(max_amp)
    m_L = log_A + 1.196997*np.log10(r/17) + 0.001066*(r-17) + 2 + S
    return m_L

def get_mL_Tanzania(r,max_amp,S):
    # Function to compute local magnitude for Tanzania (Langston et al, 1998)
    log_A = np.log10(max_amp)
    m_L = log_A + 0.776*np.log10(r/17) + 0.000902*(r-17) + 2.0 + S
    return m_L

# Transfer properties for Wood-Anderson instrument response
# Sensitivity is 2080 according to:
# Bormann, P. (ed.) (2002). IASPEI New Manual of Seismological Observatory
# Practice (NMSOP), GeoForschungsZentrum Potsdam, ISBN: 3-9808780-0-7,
# IASPEI Chapter 3, page 24
# (PITSA has 2800)
# input data must be in units of velocity to convert to convert to WA seismogram using this reponse
WOODANDERSON = {'poles': [-6.283 + 4.7124j, -6.283 - 4.7124j],
                'zeros': [0 + 0j], 'gain': 1.0, 'sensitivity': 2080}

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 

    # command-line parameters
    parser.add_argument("--exp_name", default='test', type=str,
                        help='folder (under data/picks-list) that contains the desired pick list csv')
    parser.add_argument("--csv", required=True, type=str,
                        help='name of csv (e.g. picklist_2024-07-15T08-00-00) that contains the desired picks')
    parser.add_argument("--datadirect", default=True, action=argparse.BooleanOptionalAction, 
                        help="Pass --no-datadirect to retrieve seismic data through utils_process/Obspy")
    parser.add_argument("--plot", default=True, action=argparse.BooleanOptionalAction, 
                        help="Pass --no-plot to compute magnitudes without plotting ")
    args = parser.parse_args()

    # read in locations csv and save source hypocenter parameters
    df_eqlocations = pd.read_csv(EQLOCS_PATH / 'all_eqlocations.csv')
    eqmatch_idx = df_eqlocations.loc[(df_eqlocations["exp_name"] == args.exp_name) & (df_eqlocations["pick_csv"] == args.csv)].index[0]
    eqloc_row = df_eqlocations.loc[eqmatch_idx]
    eq_lat = eqloc_row['source_latitude_deg']
    eq_lon = eqloc_row['source_longitude_deg']
    eq_depth = eqloc_row['source_depth']
    print('EQ latitude, longitude, depth:', eq_lat, eq_lon, eq_depth)

    # read in picks csv
    picklist_path = PICKLISTS_PATH / args.exp_name 
    df_picks = pd.read_csv(picklist_path / f'{args.csv}.csv')
    len_picks = len(df_picks)
    # df_picks = df_picks[df_picks['flag']==False]
    # df_picks.reset_index(inplace=True)
    
    # store dictionary of station locations and channel lists
    sta_dict = dict()
    stalocs_df = pd.read_csv(STALOCS_PATH / 'all_stations_locations.csv', dtype={'location': str})
    netsta_list = np.unique(df_picks['station'].to_numpy())
    for net_sta in netsta_list:
        net_sta_split = net_sta.split('_')
        net = net_sta_split[0]
        sta = net_sta_split[1]

        row = stalocs_df.loc[(stalocs_df['network'] == net) & (stalocs_df['station'] == sta)].squeeze()
        loc = row['location']
        samp_rate = row['samp_rate']
        chan_list = row['chan_list']
        lat_deg = row['latitude_deg']
        lon_deg = row['longitude_deg']
        elev_m = row['elevation_m']
        sta_dict[net_sta] = [loc, chan_list, samp_rate, lat_deg, lon_deg, elev_m]

    # iterate through each pick and plot 5 seconds before event start time and 5 seconds after end time
    prev_netsta = None
    for i, pick_time in enumerate(df_picks['arrivaltime_utc']):

        # skip if magnitude already measured for this station
        net_sta = df_picks.loc[i, 'station']
        if df_picks.loc[i, 'flag'] == True or net_sta == prev_netsta:
            continue

        # store station parameters
        netsta_split = net_sta.split('_')
        net = netsta_split[0]
        sta = netsta_split[1]
        sta_params = sta_dict[net_sta]
        loc = sta_params[0]
        if pd.isna(loc):
            loc = ''
        chan_list = sta_params[1]
        chan_list = chan_list.split('.')
        samp_rate = sta_params[2]
        sta_lat = sta_params[3]
        sta_lon = sta_params[4]
        sta_depth_km = sta_params[5]/1000

        # get distance and azimuth
        delta, az = get_dists_azimuths(eq_lat, eq_lon, [sta_lat], [sta_lon])
        grcdist_km = delta * (2*np.pi*6371/360)
        hdist_km = np.sqrt(grcdist_km**2 + (eq_depth - sta_depth_km)**2)
        hdist_km = hdist_km[0]
        az = az[0]
        print('Azimuth [deg]', az)
        if az < 0:
            az = 360+az
        print('Hypocentral distance [km]', hdist_km)

        # store pick/event parameters and plot duration
        plot_start = UTCDateTime( df_picks.loc[i, 'event_start_utc'])-1
        event_end = UTCDateTime( df_picks.loc[i, 'event_end_utc'])
        pick_confidence = df_picks.loc[i, 'pick_confidence']
        event_duration = event_end - plot_start
        plot_duration = 3*event_duration
        plot_end = plot_start + plot_duration

        # store pick amplitude for each component
        argmax_horizontal_list = []
        max_horizontal_list = []
        for j, chan in enumerate(chan_list):

            # get raw data
            if args.datadirect:
                # Julian day from plot_start, zero-padded to 3 digits (e.g., 1 -> "001")
                julday_str = f"{plot_start.julday:03d}"
                st_1c = read(DATA_MSEED/ sta / f'{sta}.{net}.{loc}.{chan}.{plot_start.year}.{julday_str}', format='MSEED')
                st_1c.trim(starttime=plot_start, endtime=plot_end)
                print("raw data", sta)
            else:
                st_1c = get_rawdata(net, sta, loc, chan, str(plot_start), plot_duration, samp_rate=samp_rate)
                print("iris webservices", sta)
            times = st_1c[0].times()

            if j == 0:
                st_raw = st_1c
            else:
                st_raw = st_raw + st_1c

        # store data parameters, filters, and Wood-Anderson response
        t_samp = 1/samp_rate
        npts_st =  st_raw[0].stats.npts
        WAresponse, freqs = paz_to_freq_resp(WOODANDERSON['poles'], WOODANDERSON['zeros'],
                                            WOODANDERSON['gain'], t_samp, npts_st, freq=True)
        inv = read_inventory(METADATA_PATH / f'{net}.{sta}.xml')
        pre_filt = [0.018, 0.03, 16, 19.99]
        freq_domain_taper = cosine_sac_taper(freqs, flimit=pre_filt)
        
        # filter, convert to velocity, and simulate Wood Anderson response for ML Method 1
        st_fullWA = st_raw.copy()
        st_fullWA = st_fullWA.detrend('linear')
        gain_dict = dict()     #dictionary to store stage zero sensitivities for each channel for ML Method 2
        for chan in chan_list:
            tr = st_fullWA.select(channel=chan)[0]
            response = inv.get_response(f"{net}.{sta}.{loc}.{chan}", st_raw[0].stats.starttime)
            gain_dict[chan] = response.instrument_sensitivity.value
            response, freqs = response.get_evalresp_response(t_samp, npts_st, output="VEL")
            response[0] = 1
            tr_fft = np.fft.rfft(tr.data, npts_st)
            tr_fft = tr_fft * freq_domain_taper
            tr_fft_proc = tr_fft/response * WAresponse * WOODANDERSON['sensitivity']
            tr_proc_data = np.fft.irfft(tr_fft_proc)
            st_fullWA.select(channel=chan)[0].data = tr_proc_data

        # store a stream object that is only gain corrected for ML Method 2
        st_gaincorrec = st_raw.copy()
        for st_i in range(3):
            chan = st_gaincorrec[st_i].stats.channel
            st_gaincorrec[st_i].data = st_gaincorrec[st_i].data/gain_dict[chan]

        # rotate both stream objects (ML Methods 1 and 2) to components ZRT
        for stream in [st_fullWA, st_gaincorrec]:
            stream.rotate(method='->ZNE', inventory=inv)
            stream.rotate(method='NE->RT', back_azimuth=az)

        # store max horizontal amplitudes for both methods
        Hmax_list = {'WAsimulated':[], 'gaincorrected':[] }
        Hmax_arg_list = {'WAsimulated':[], 'gaincorrected':[] }
        for i_st, st in enumerate([st_fullWA, st_gaincorrec]):
            if i_st ==0:
                method = 'WAsimulated'
            else:
                method = 'gaincorrected'

            for tr in st:
                if tr.stats.channel.endswith('Z'):
                    continue
                abs_data = abs(tr.data)
                argmax = np.argmax(abs_data)
                Hmax_arg_list[method].append(argmax)
                Hmax_list[method].append(abs_data[argmax])

        # print('Method 1: Full Wood Anderson simulation')
        A_m_WAsimulated = np.max(Hmax_list['WAsimulated'])
        A_mm_WAsimulated = A_m_WAsimulated * 1e3
        ml_Norway_WAsim = get_mL_Norway(hdist_km, A_mm_WAsimulated,0)
        ml_Ethiopianrift = get_mL_Ethirift(hdist_km, A_mm_WAsimulated,0)
        ml_Tanzania = get_mL_Tanzania(hdist_km, A_mm_WAsimulated,0)
        print('Local magnitude with zero station correction (Norway):', ml_Norway_WAsim)
        print('Local magnitude with zero station correction (Tanzania):', ml_Tanzania)
        print('Local magnitude with zero station correction (Ethiopian rift):', ml_Ethiopianrift)

        # print('Method 2: Approximated Wood Anderson')
        A_m_gaincorrec = np.max(Hmax_list['gaincorrected'])
        A_nm_gaincorrec = A_m_gaincorrec * 1e9
        ml_Norway_WAapprox = get_mL_Norway_noWA(hdist_km, A_nm_gaincorrec)
        print('Local magnitude with zero station correction (Norway, WA approx):', ml_Norway_WAapprox)
        print('')

        # plot Z and T components with maximum amplitudes (ML Method 1 only)
        if args.plot:
            fig, ax = plt.subplots(2,1)
            ax[0].set_title(net_sta)
            ax[1].set_xlabel('Time [s]')
            for comp_i, comp in enumerate(['R', 'T']):
                ax[comp_i].set_ylabel(f'{comp} [m]')
                ax[comp_i].plot(st_fullWA.select(component=comp)[0].data)
                ax[comp_i].axhline(y=A_m_WAsimulated, color='red', linestyle='dashed')
                ax[comp_i].axhline(y=-A_m_WAsimulated, color='red', linestyle='dashed')
            plt.tight_layout()
            plt.show()

        # save magnitude parameters to pick_list_csv
        df_picks.loc[i, 'azimuth_deg'] = az
        df_picks.loc[i, 'hdistance_km'] = hdist_km
        df_picks.loc[i, 'max_hamp_WAsim_m'] = A_m_WAsimulated
        df_picks.loc[i, 'max_hamp_WAapprox_m'] = A_m_gaincorrec
        df_picks.loc[i, 'ML_Norway_WAsim'] = ml_Norway_WAsim
        df_picks.loc[i, 'ML_EthiopianRift'] = ml_Ethiopianrift
        df_picks.loc[i, 'ML_Tanzania'] = ml_Tanzania
        df_picks.loc[i, 'ML_Norway_WAapprox'] = ml_Norway_WAapprox

        prev_netsta = net_sta

    # save new picklist to same directory
    df_picks.to_csv(picklist_path / f'{args.csv}.csv', index=False)

    # store average ML mags and standard deviations in eq locations csv
    ml_Norway_WAsim_avg = df_picks["ML_Norway_WAsim"].mean()  
    ml_Norway_WAsim_std = df_picks["ML_Norway_WAsim"].std()
    ml_Ethiopianrift_avg = df_picks["ML_EthiopianRift"].mean()  
    ml_Ethiopianrift_std = df_picks["ML_EthiopianRift"].std()
    ml_Tanzania_avg = df_picks["ML_Tanzania"].mean()  
    ml_Tanzania_std = df_picks["ML_Tanzania"].std()
    ml_Norway_WAapprox_avg = df_picks["ML_Norway_WAapprox"].mean()  
    ml_Norway_WAapprox_std = df_picks["ML_Norway_WAapprox"].std()
    
    df_eqlocations.loc[eqmatch_idx, 'ML_Norway_WAsim'] = ml_Norway_WAsim_avg
    df_eqlocations.loc[eqmatch_idx, 'ML_Ethiopianrift'] = ml_Ethiopianrift_avg
    df_eqlocations.loc[eqmatch_idx, 'ML_Tanzania'] = ml_Tanzania_avg
    df_eqlocations.loc[eqmatch_idx, 'ML_Norway_WAapprox'] = ml_Norway_WAapprox_avg
    df_eqlocations.loc[eqmatch_idx, 'ML_Norway_WAsim_std'] = ml_Norway_WAsim_std
    df_eqlocations.loc[eqmatch_idx, 'ML_Ethiopianrift_std'] = ml_Ethiopianrift_std
    df_eqlocations.loc[eqmatch_idx, 'ML_Tanzania_std'] = ml_Tanzania_std
    df_eqlocations.loc[eqmatch_idx, 'ML_Norway_WAapprox_std'] = ml_Norway_WAapprox_std
    df_eqlocations.to_csv(EQLOCS_PATH / 'all_eqlocations.csv', index=False)
    print('Averaged Local Magnitude (Norway):', ml_Norway_WAsim_avg )
    print('Averaged Local Magnitude (Ethiopia):', ml_Ethiopianrift_avg)
    print('Averaged Local Magnitude (Tanzania):', ml_Tanzania_avg)
    print('Averaged Local Magnitude (Norway, WA approx):', ml_Norway_WAapprox_avg)




    
