import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from constants import VELMODEL_PATH
from datetime import datetime, timedelta
from obspy.taup import TauPyModel
model = TauPyModel(model="iasp91")

def readpickscsv(df):
    '''
    Function to read in csv of station parameters and picks
    param:
        df: pandas DataFrame
            Dataframe with the following columns: reftime_utc, station, station_latitude_deg, station_longitude_deg,
            station_elevation_m, arrivaltime_utc, relative_arrivaltime_s, phase, pick_confidence. There may be also a "Notes" column. 
    return:
        n: integer, number of arrival times
        ref_time: str, reference time in UTC
        sta: numpy array, list of station names
        lat: numpy array, list of latitudes (deg)
        lon: numpy array, list of longitudes (deg)
        elv: numpy array, list of elevations (km above sea level)
        atm: numpy array, list of arrival times (s, with respect to a reference time) for the desired wave
        phase: numpy array, list of phases ('P' or 'S') corresponding to each arrival time
    '''
    n = len(df)
    reftime = df.loc[0, 'reftime_utc']
    sta = df['station'].to_numpy()
    lat = df['station_latitude_deg'].to_numpy()
    lon = df['station_longitude_deg'].to_numpy()
    elv = df['station_elevation_m'].to_numpy() / 1000  # convert m to km
    atm = df['relative_arrivaltime_s'].to_numpy()
    phases = df['phase'].to_numpy()
    
    return n,reftime,sta,lat,lon,elv,atm,phases


def merrors(covm, param1='lon', param2='lat',annotate=False):
    '''Function to return error ellipse parameters (for patches.Ellipse) given a covariance matrix (covm) and the two desired parameters
    param:
        covm: array, covariance matrix with order origin time, longitude, latitude, and depth
        param1: string, name of 1st parameter ('otime' = origin time, 'lon' = longitude, 'lat' = latitude, or 'depth' = depth)
        param2: string, name of 2nd parameter ('otime' = origin time, 'lon' = longitude, 'lat' = latitude, or 'depth' = depth)
        annotate: Boolean, True if covariance matrix, eigenvectors, and eigenvalues should be printed. False if otherwise    
    return:
        major: float, major axis length
        minor: float, minor axis length
        rota: float, ellipse rotation angle (degrees anti-clockwise)
    '''

    # set the appropriate parameter index
    paramidx_dict = {'otime':0, 'lon':1, 'lat':2, 'depth':3}
    param1_idx = paramidx_dict[param1]
    param2_idx = paramidx_dict[param2]

    if annotate:
        print( '\n covariance matrix:')
        print( covm)
        print( 'std in tq,lonq,latq,hq:')
        print( np.sqrt(covm[0,0]),np.sqrt(covm[1,1]),np.sqrt(covm[2,2]))

    # compute eigenvectors and eigenvalues
    if annotate:
        w,v = np.linalg.eig(covm)
        w1 = np.sqrt(w)
        print( 'sqrt of eigenvalues, eigenvalues (covariances), and eigenvectors:')
        for i in np.arange(len(w)):
            print(  i, w1[i],w[i],v[:,i])
    # get the marginal distribution (submatrix) of the multivariate covariate matrix, using the desired parameters
    covlatlon = covm[[param1_idx, param2_idx], :][:, [param1_idx, param2_idx]]          
    w,v = np.linalg.eig(covlatlon)

    # order eigen values and eigen vectors by decreasing eigevalues
    idx = w.argsort()[::-1]   
    w = w[idx]
    v = v[:,idx]  

    # total length of axes (diameters): 
    # radius is thrice the standard deviation for a ~99% confidence interval
    major = 6*np.sqrt(w[0])
    minor = 6*np.sqrt(w[1])
    rota = np.degrees(np.arctan2(v[1,0],v[0,0]))

    if annotate:
        print('eigenvalues & -vectors:')
        print( 0, w[0], v[:,0])
        print( 1, w[1], v[:,1])
        print( "ellipse rotation angle = " ,rota)
    return major,minor,rota


def get_dists_azimuths(source_lat, source_lon, station_lats, station_lons):
    '''
    Function that computes great circle distances/deltas and azimuths. 
    param:
        source_lat: float, source latitude in degrees
        source_lon: float, source longitude in degrees
        station_lats: array-like, list of station latitudes in degrees
        station_lons: array-like, list of station longitudes in degrees
    return:
        deltas: numpy array, list of great circle distances in degrees
        azimuths: numpy array, list of azimuths in degrees
    '''
    deltas = []
    azimuths = []
    teta1 = np.radians(source_lat)
    fi1 = np.radians(source_lon)
    for sta_i, sta_lat in enumerate(station_lats):
        teta2 = np.radians(sta_lat)
        fi2 = np.radians(station_lons[sta_i])

        # compute great circle distance in degrees
        ct1 = np.cos(teta1)
        ct2 = np.cos(teta2)
        st1 = np.sin(teta1)
        st2 = np.sin(teta2)
        term1  = st1*st2
        factor2  = ct1*ct2 
        factor1 = np.cos(fi1-fi2)
        term2 = factor1*factor2
        som = term1 + term2
        delta = np.degrees(np.arccos(som))

        # # convert great circle distance to km
        # dist_km = np.radians(delta) * 6371

        # compute azimuths in degrees
        term1 = st2*ct1 
        term2 = ct2* st1*factor1
        teller = ct2*np.sin(fi2-fi1)
        rnoemer = term1 - term2
        azimuth = np.degrees(np.arctan2(teller,rnoemer))

        deltas.append(delta)
        azimuths.append(azimuth)

    return np.array(deltas), np.array(azimuths)


def get_velocity(depth, phase, model_csv='IASP91.csv'):
    # Function to retrieve velocity for a given depth, phase, and velocity model. csv for velocity model should contain the following columns:
    # r: distance from the center of Earth (km)
    # v_P: P wave velocity (km/s)
    # v_S: S wave velocity (km/s)
    # csv should be within data/velocity_models
    df = pd.read_csv(VELMODEL_PATH / model_csv)
    r = 6371 - depth
    closest_index = (df.r - r).abs().idxmin()
    if phase == 'P':
        vel = df.loc[closest_index, 'v_P']
    elif phase == 'S':
        vel = df.loc[closest_index, 'v_S']
    return vel


def get_partial_phi(azimuth_deg, rho):
    # compute partial derivative of travel time with respect to longitude/phi. 
    # inputs: azimuths [deg], ray parameter [s/deg]
    partial_phi = - np.sin( np.radians(azimuth_deg) ) * rho
    return partial_phi


def get_partial_theta(azimuth_deg, rho):
    # compute partial derivative of travels time with respect to latitude/theta. 
    # inputs: azimuths [deg], ray parameter [s/deg]
    partial_theta = - np.cos( np.radians(azimuth_deg) ) * rho
    return partial_theta


def get_partial_h(vels_atdepth, ta_deg):
    # compute partial derivative of travels time with respect to depth/h
    # inputs: velocities at depth [km/s], take-off angles [deg]
    partial_h = - np.cos(np.radians(ta_deg)) / vels_atdepth
    return partial_h


def get_damp(lbda, tas_deg,  Zbool, alpha=6):
    '''
    Returns a diagonal matrix with damping values for each model parameter (order: origin time, longitude, latitude, depth [optional])
    param: 
        lbda: float, constant factor to multiply each damping value
        tas_deg: array-like, list of take-off angles (degrees) 
        Zbool: Boolean, True if inverting for depth, False otherwise
        alpha: float, P wave velocity [km/s] to use as a rough value
    '''
    ta_med = np.median(tas_deg)
    d_otime = lbda                                                 
    d_lon = lbda*np.sin(np.radians(ta_med)) / (alpha*0.01)
    d_lat = d_lon
    if Zbool:
        d_h = lbda*np.cos(np.radians(ta_med))/alpha
        damping = np.diag( [d_otime, d_lon, d_lat, d_h])
    else:
        damping = np.diag( [d_otime, d_lon, d_lat])
    return damping


def locatequake(n,reftime, lat_sta,lon_sta,elv,atm,phases,velmodel_csv,startloc,Zbool,damp_factor,sigmaT=0.5,annotate=True):
    '''
    Function that performs inversion on a given list of arrival P/S times
    param:
        n: int, number of arrival times/stations
        ref_time: str, reference time in UTC
        lat_sta: array, station latitudes [deg]
        lon_sta: array, station latitudes [deg]]
        elv: array, station elevations [km]
        atm: array, P or S arrival times [s after reference time]
        phases: list, list of phases ('P' or 'S') corresponding to each arrival
        velmodel_csv: string, name of csv that contains P and S velocities of a desired model. Should be within data/velocity-models folder
        startloc: list/array, 4-element list-like object containing initial guesses for (1) origin time in s after ref time, 
                (2) longitude in km,  (3) latitude in km, and (4) depth in km
        Zbool: Bool, True to perform inversion for depth or False to keep depth fixed at initial guess
        damp_factor: integer, constant factor to multiply to damping matrix
        sigmaT: float, uncertainty (s) of arrival times 
        annotate: Bool, True to print/plot guesses at each iterations, False if otherwise
    returns:
        loc: list of source parameters at each inversion iteration, list of 4-element list-like object containing (1) origin time in s after ref time, 
                (2) longitude in km,  (3) latitude in km, and (4) depth in km
        covm: covariance matrix of final solution
        otime_utc_final: datetime object, final solution for origin time in UTC
    '''

    if annotate:
        # print intial guess, m = (dt (s), dlon (deg), dlat (deg)) and optionally depth (in km)
        print("m, initial guess = ", startloc)
        print('')

    # store zero array for model parameters. account for inverting for depth or not
    if Zbool: 
        m = np.zeros(4)
    else:
        m = np.zeros(3)

    # create an array of surface velocities that match each arrival type
    surface_vel_dict = {'P': get_velocity(depth=0, phase='P', model_csv=velmodel_csv), 
                    'S': get_velocity(depth=0, phase='S', model_csv=velmodel_csv)}
    surface_vels = []
    for type in phases:
        surface_vels.append( surface_vel_dict[type])
    surface_vels = np.array(surface_vels)

    # set initial values
    loc = [startloc]                # initial guess of source parameters   
    converg_threshold = 3*sigmaT    # threshold for solution convergence 
    converg_crit = 1e9              # initial arbitrary values for convergence criterion
    percchange_converg = 1e9        # ^ for percent change in convergence criterion
    k = 0                           # iteration number

    # update locations until solution converges OR number of iterations exceed 100
    while (converg_crit > converg_threshold or percchange_converg > 0.10) and k < 100:
        # get initial guess or guess from previous iteration
        tq,lonq,latq,hq = loc[k]
        # compute great circle distances and azimuths
        grc_deltas, azs = get_dists_azimuths(latq, lonq, lat_sta, lon_sta)

        # compute predicted travel times (t_ts, s), ray parameters (rhos, s/deg), incident angles (ias, deg),
        # and  takeoff angles (tas, deg) using taup, with respect to initial/previous guess
        t_ts = []
        rhos = []
        ias_deg = []
        tas_deg = []
        for pick_i, delta in enumerate(grc_deltas):
            if phases[pick_i]=='P': phase_list=['p','P']
            elif phases[pick_i]=='S': phase_list=['s','S']
            arrivals = model.get_travel_times(source_depth_in_km=hq,distance_in_degree=delta, 
                                            receiver_depth_in_km=0, phase_list=phase_list)
            t_ts.append(arrivals[0].time)
            rhos.append(arrivals[0].ray_param_sec_degree)
            ias_deg.append(arrivals[0].incident_angle)
            tas_deg.append(arrivals[0].takeoff_angle)

        # store parameters as numpy arrays
        t_ts = np.array(t_ts)
        rhos = np.array(rhos) 
        ias_deg = np.array(ias_deg)
        tas_deg = np.array(tas_deg)

        # compute data vectors, correcting for variable elevations
        elv_correc = elv/(surface_vels * np.cos(np.radians(ias_deg)))       # time to travel the elevation of the station
        d = atm - elv_correc - tq - t_ts
        
        # plot the difference between the observed travel times and the predicted travel times based on the initial guess
        if k ==0:
            std_initial = np.std(d)
            plt.hist(d)
            plt.xlabel('travel time misfit between initial guess and observation [s]')
            plt.ylabel('number of observations')
            plt.title(f'stdev: {std_initial}')
            plt.show()

            # print arrivals that exceed a residual of 10 s
            print('Checking arrivals with travel time misfit between initial guess and observation [s] which exceed 10 s...')
            for d_i, residual in enumerate(d):
                if abs(residual) > 10:
                    print(f'Condition met for {phases[d_i]} arrival at station with (lon, lat): ({lon_sta[d_i]},{lat_sta[d_i]})' )
            print('')
        
        # compute each component of the G matrix 
        rowt = np.ones(n)
        rowo = get_partial_phi(azs, rhos)
        rowa = get_partial_theta(azs, rhos)
        if Zbool:
            # create an array of velocities at source depth that match each arrival type
            vel_atdepth_dict = {'P': get_velocity(depth=hq, phase='P', model_csv=velmodel_csv), 
                            'S': get_velocity(depth=hq, phase='S', model_csv=velmodel_csv)}
            vels_atdepth = []
            for type in phases:
                vels_atdepth.append( vel_atdepth_dict[type])
            vels_atdepth = np.array(vels_atdepth)

            rowz = get_partial_h(vels_atdepth, tas_deg)
            GT = np.array([rowt,rowo,rowa,rowz])
        else:
            GT = np.array([rowt,rowo,rowa])
        
        # apply damping and solve for the model parameters
        damp = get_damp(damp_factor,tas_deg, Zbool)
        G = GT.T
        GTG = np.dot(GT,G)
        GTGm1 = np.linalg.inv(GTG+(damp**2))
        Gmg = np.dot(GTGm1,GT)
        m = np.dot(Gmg,d)

        # compute the convergence criterion and store percent change from previous criterion
        diff_Gm_d = np.dot(G, m) - d
        new_converg_crit = np.sqrt(np.sum(diff_Gm_d**2) / len(d))
        percchange_converg = (np.abs(new_converg_crit - converg_crit) / converg_crit) * 100
        converg_crit = new_converg_crit
        
        # print updated m vector
        if Zbool:
            newloc = [tq+m[0], lonq+m[1], latq+m[2], hq+m[3]]
        else:
            newloc = [tq+m[0], lonq+m[1], latq+m[2], hq]
        
        if annotate:
            print('iteration:', k)
            print( '|d| = ', np.linalg.norm(d),' |d|^2 = ', np.linalg.norm(d)**2)
            print( 'm = ', m)
            print( 'newloc = ', newloc)
            
            # plot new guess with older guesses
            # fig, ax = plt.subplots(1,2)
            # ax[0].plot(lon_sta, lat_sta, 'v',label='stations')
            # ax[0].plot(startloc[1],startloc[2], 'x',markersize=10,label='simple start')
            # for idx, lctn in enumerate(loc):
            #     ax[0].plot(lctn[1],lctn[2],'.',label= f'guess {idx}')
            # ax[0].plot(newloc[1],newloc[2],'o',label='latest')
            # ax[0].legend()
            # ax[0].set_xlabel('longitude [deg]')
            # ax[0].set_ylabel('latitude [deg]')
            # ax[1].hist(d, alpha=0.4,histtype='stepfilled',label=str(k))
            # ax[1].set_xlabel('Residuals [s]')
            # plt.show()

        # add new location to list of guesses
        loc = loc + [newloc]
        k = k + 1
        
    if k == 100:
        print('Convergence was not reached after 100 iterations')
        response = input("Do you want to continue (y/n)?: ").strip().lower()

        if response == 'n':
            print('Exiting program...')
            exit()
        elif response == 'y':
            print('')
        else:
            print('Response other than yes was recorded. Exiting program...')
            exit()

    # plot the difference between the observed travel times and the predicted travel times based on the final guess
    std_final = np.std(d)
    plt.hist(d)
    plt.xlabel('travel time misfit between final guess and observation [s]')
    plt.ylabel('number of observations')
    plt.title(f'stdev: {std_final}')
    plt.show()
    
    print( f'final solution after {k} iterations')
    print( 'convergence criterion:', converg_crit)
    print( f'criterion percent diff from the {k-1}-th iteration: {percchange_converg}')
    print( 'origin time after reference time (s):', loc[-1][0])
    otime_utc_final = datetime.fromisoformat(reftime) + timedelta(seconds=loc[-1][0])
    print('origin time (utc):', otime_utc_final)
    print( 'lat and lon (degrees): ',loc[-1][2],loc[-1][1])
    print( 'depth (km): ',loc[-1][3])

    # compute covariance matrix
    varT = sigmaT**2
    covm = varT*GTGm1 
   
    return loc, covm, otime_utc_final, k, converg_crit, percchange_converg