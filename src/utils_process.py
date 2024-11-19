# TODO: delete if code does not use get_rawdata()
# Program contains additional functions to extract seismic data from IRIS and create PSDs. 

from obspy import UTCDateTime, read
from obspy.clients.fdsn import Client
client = Client('IRIS')
import scipy.signal as scisig
from obspy.signal import invsim
import numpy as np
import constants
import os
from matplotlib import dates 


def get_rawdata(net, sta, loc, chan, starttime_str, duration, samp_rate, plot_wave = False, save = False, format = 'MSEED'):
    '''
    Function that extracts a piece of requested data, either by using local files or through IRIS Web Services. Segments
    with gaps are 'filled' with interpolated values and gap times are saved to a text file.
    :param net: string, network name (e.g. 'NW')
    :param sta: string, station name (e.g. 'HQIL')
    :param loc: string, location number (e.g. '00')
    :param chan: string, channel name (e.g. 'HHZ')
    :param starttime_str: string, start-time of data request in UTC format (e.g. '2018-12-29T04:01:54.8')
    :param duration: string, total duration of data request
    :param samp_rate: float, desired sampling rate of station data
    :param plot_wave: Boolean, True if requested data is to be plotted and False if otherwise
    :param save: Boolean, True if requested data (if 24 hours) is to be saved as a local file and False if otherwise
    :param format: string, format (i.e. 'MSEED' or 'SAC') of the data files to be used/saved.
    :return: Obspy stream object, raw waveform data
    '''

    # store parameters of data request
    year = starttime_str[0:4]
    startday = starttime_str[0:10]
    starttime = UTCDateTime(starttime_str)
    endtime = starttime + duration
    endtime_str = str(endtime)
    endday = endtime_str[0:10]
    prnt_folder = f'{net}_{sta}'

    # save file extension with the desired format
    if format == 'MSEED':
        file_ext = '.mseed'
    elif format == 'SAC':
        file_ext = '.sac'

    # go to appropriate local directory to look for and save files
    file_path = constants.RAW24H_PATH / prnt_folder / chan / year
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # if data request only spans 1 UTC day, store the name of the desired local file
    filenames = []
    if startday == endday or (starttime_str[11:] == '00:00:00' and duration == (24*60*60)):
        filename = net + '_' + sta + '_' + chan + '_' + startday + file_ext
        filenames.append(filename)
    #
    # elif starttime_str[11:] == '00:00:00' and duration == (24*60*60):
    #     filename = net + '_' + sta + '_' + chan + '_' + startday + '.mseed'
    #     filenames.append(filename)

    # for data requests that spans more than 1 UTC day, store a list of all desired file names
    else:
        day_i = startday
        strt = UTCDateTime(starttime_str)
        days = [day_i]
        filename = net + '_' + sta + '_' + chan + '_' + day_i + file_ext
        filenames.append(filename)
        while day_i != endday:
            next_day = strt + (60 * 60 * 24)
            next_day_str = str(next_day)
            day_i = next_day_str[0:10]
            days.append(day_i)
            filename = net + '_' + sta + '_' + chan + '_' + day_i + file_ext
            filenames.append(filename)
            strt = next_day

    # for data requests spanning only 1 UTC day, check if requested day is locally saved.
    # If file is available, extract appropriate stream object
    files_available = False
    if startday == endday or (starttime_str[11:] == '00:00:00' and duration == (24*60*60)):
        filename = filenames[0]
        for file in os.listdir(file_path):
            if file == filename:
                st = read(constants.RAW24H_PATH / prnt_folder / chan / year / filename)
                if duration != (60*60*24):
                    st[0].trim(starttime = starttime, endtime = endtime)
                    if len(st[0].data) < 2:
                        raise ValueError('Trimming 24h stream caused issues. Resulting stream only has a length of', len(st[0].data))
                files_available = True

    # for data requests spanning more than one UTC day check if all the appropriate files are available
    # If the files are available, extract all the appropriate stream object and merge them together
    else:
        files_avail = []
        for file in os.listdir(file_path):
            for desired_file in filenames:
                if file == desired_file:
                    files_avail.append(file)

        if len(files_avail) == len(filenames):
            files_available = True
            streams = []
            # print('days', days)
            # print(startday)
            # print(endday)
            for i in range(len(filenames)):
                filename = filenames[i]
                if days[i] == startday:
                    start = starttime
                if days[i] == endday:
                    end = endtime
                if days[i] != endday:
                    end_str = days[i + 1] + 'T:00:00:00'
                    end = UTCDateTime(end_str)
                st = read(constants.RAW24H_PATH / prnt_folder/ chan / year / filename)
                st.trim(starttime=start, endtime=end)
                streams.append(st)
                start = end
            st = streams[0]
            for i in range(len(streams)):
                if i != 0:
                    st = st + streams[i]
            st.merge()

    # If no data files are locally available, get waveform using IRIS Web Services
    if files_available == False:
        #print('No saved data file available. Accessing waveform data using Obspy...')
        st = client.get_waveforms(net, sta, loc, chan, starttime, endtime, attach_response=True)
        # print(st.get_gaps())
        # print(len(st.get_gaps()))

        # fill gaps with interpolated values if needed and save segments of gaps to a text file
        get_gaps = st.get_gaps()
        if len(get_gaps) != 0:
            print('Gaps present. Merging traces using interpolation...')
            st.merge(fill_value='interpolate')
            if starttime_str[11:] == '00:00:00' and duration == (24 * 60 * 60) and save==True:
                print('Writing gaps to a file...')
                filename_gaps = net + '_' + sta + '_' + chan + '_' + startday + '_gaps.txt'
                rfile_gaps = constants.RAW24H_PATH / prnt_folder/ chan / year / filename_gaps
                file_gaps = open(rfile_gaps, "w+")
                for list in get_gaps:
                    file_gaps.write(str(list[4]) + '\t' + str(list[5]) + '\n')
                file_gaps.close()

        # Decimate to the desired sampling rate if needed
        if round(st[0].stats.sampling_rate,2) != samp_rate:
            dec_factor = st[0].stats.sampling_rate / samp_rate
            if dec_factor.is_integer():
                dec_factor = int(dec_factor)
            else:
                raise ValueError('Factor to downsample stream to desired rate must be an integer. Original sampling rate is ', st[0].stats.sampling_rate )
            st.decimate(dec_factor)
            print('Stream was decimated by a factor of ' + str(dec_factor))

        # save to file if requested and if it includes 24 hours of data
        if save == True or (starttime_str[11:] == '00:00:00' and duration == (24*60*60)):
            print('Trying to save file...')
            if duration != (60*60*24):
                raise ValueError('Data file directory only includes raw 24 h streams ')
            st.write(constants.RAW24H_PATH / prnt_folder / chan / year / filename, format=format)

    # Plot stream data if requested
    if plot_wave == True:
        st.plot()

    # return Obspy stream object of requested data
    return st


def classicsmooth(x, octaveFraction=1.0/8.0):
    # Perform octave smoothing on an x array
    smooth = []
    rightFactor = 2.0 ** (octaveFraction / 2.0)
    leftFactor = 1.0 / rightFactor

    for n in range(len(x)):
        left = int(n * leftFactor)
        right = int(n * rightFactor)
        smooth_val = sum(x[left: right + 1]) / (right - left + 1)
        smooth.append(smooth_val)

    return smooth


def my_PSD(sta_data, chan, duration, samp_rate=100, dt_sgmt=60 * 60, dt_subgmt =60 * 15, ovlap_sgmt=0.50,
           n_subsgmts=13, ovlap_subsgmt=0.75, tpr_alpha=0.20, tpr_factor = 1.142857,
           octavefrac=1.0 / 8.0, resp_correct = False, smooth = True, median = False, sta='HQIL'):
    '''
    Computes the power spectral density as described by McNamera and Buland (2004)

    :param sta_data (array-like):
    :param chan (string): channel code
    :param starttime (string): startime of desired stream, in format YYYY-MM-DDTHH:MM:SS
    :param duration (int): duration of entire stream, in seconds
    :param samp_rate (int): sampling rate of desired stream, in Hz
    :param dt_sgmt (int): time window of stream segments, in s (default = 60*60 for 1 hour)
    :param ovlap_sgmt (float): percentage of overlap between subsequent segments
    :param n_subsgmts (integer): number of subsegments within each segment (default = 13)
    :param dt_subgmt (int): time window of subsegmemt, in s (default = 60*15 for 15 min)
    :param ovlap_subsgmt (float): percentage of overlap between subsequent subsegments (default = 0.75)
    :param tpr_alpha (float): fraction of subsegment window within cosine taper (default = 0.20)
    :param tpr_factor (float): correcting factor for smoothing due to cosine tapering (default = 1.142857)
    :param octavefrac (float): 1/N fraction for classic octave smoothing (default = 1.0/8.0)
    :param resp_correct (Bool): True if input data (must be in counts) is to be converted to units of ACC.
                                Only can be done if station (sta) is 'HQIL'
    :param smooth (Bool): True if Octave smoothing is to be performed an PSD.
    :param median (Bool): True if median, rather than the average of the PSD subsegments, must be taken and returned
    :param sta (string): station code

    :return freq (array): sample frequencies corresponding to the PSD estimates
    :return psd (array): power spectral density estimates (before smoothing)
    :return smooth_psd (arrray): psd after 1/N octave smoothing. Only applicable if smooth == True
    '''


    #Raise error if stream data does not have the expected number of points
    if len(sta_data) < int(duration * samp_rate):
        raise ValueError('Input data has less samples than expected from the input duration and sampling rate')

    #Convert to list
    sta_data = sta_data.tolist()

    npts_data = len(sta_data)                                         # number of samples in raw data stream
    npts_sgmt = int(dt_sgmt * samp_rate)                              # number of samples in a segment of duration dt_sgmt

    # Store integer number of complete segments (of duration dt_sgmt) in raw data
    if duration >= (2 * dt_sgmt):
        n_segments = int(npts_data / npts_sgmt) * (1 / (1 - ovlap_sgmt))
        # Subtract final segment with missing data
        n_segments = int(n_segments) - 1
    else:
        n_segments = 1

    # Initialize values
    npts_subsgmt_0 =  int(dt_subgmt * samp_rate)                      # number of samples in a subsegment (before truncation)
    nfft_subsgmt = 2 ** int(np.log2(npts_subsgmt_0))                  # number of samples in a subsegment (after truncation to a power of 2)
    dt = 1 / samp_rate                                                # time interval between subsequent samples

    # Create a cosine taper with the given alpha (tpr_alpha)
    costaper = scisig.tukey(nfft_subsgmt, alpha=tpr_alpha)

    start_i = int(0)
    # Loop through each segment and compute the PSD estimate via scipy.signal.welch
    for sgmt_i in range(n_segments):
        end_i = int(start_i + npts_sgmt)
        data_sgmt_0 = sta_data[start_i:end_i]
        data_sgmt_1 = []                                              # new list of truncated subsegments
        Pxx_list = []                                                 # list of PSD estimates to be averaged
        startsub_i = 0
        # Truncate each subsegment to a power of 2 and append to data_sgmt_1
        for subsgmt_i in range(n_subsgmts):
            endsub_i = startsub_i + nfft_subsgmt
            subsgmt = data_sgmt_0[startsub_i:endsub_i]
            data_sgmt_1.extend(subsgmt)
            startsub_i = startsub_i + (npts_subsgmt_0 * (1 - ovlap_subsgmt))
            startsub_i = int(startsub_i)
        # Compute power values for each segment using scipy.signal.welch. 
        # Note 1: noverlap = 0 because data_sgmt_1 is a concatanated list of overlapping, truncated segments. We have performed the overlapping manually to be consistent with McNamara and Buland (2004)
        # Note 2: Scaling for the reduction in power values due to tapering (factor of 1.142857) is automatically done in welch()
        f, Pxx_sgmt = scisig.welch(data_sgmt_1, samp_rate, window=costaper,
                                   nperseg=nfft_subsgmt, noverlap=0,
                                   detrend=False)
        Pxx_list.append(Pxx_sgmt)
        start_i = start_i + (npts_sgmt * (1 - ovlap_sgmt))
        start_i = int(start_i)

    # If there are more than 1 segment in the time series, compute the average of their PSD estimates
    if len(Pxx_list) > 1:
        Pxx = []
        for i in range(len(Pxx_list[0])):
            Pxx_i = []
            for j in range(len(Pxx_list)):
                val = Pxx_list[j][i]
                Pxx_i.append(val)
            if median:
                val = np.median(Pxx_i)
            else:
                val = np.mean(Pxx_i)
            Pxx.append(val)
    else:
        Pxx = Pxx_sgmt

    # Remove the first point (for freq=0) from the frequency and PSD arrays
    freq = f[1:]
    power = Pxx[1:]

    # take the absolute value of the PSD array
    power = np.abs(power)
    # power = np.abs(power) * tpr_factor              

    if resp_correct == True:
        # Remove instrument response by dividing power by the instrument transfer function to acceleration
        if sta != 'HQIL':
            raise ValueError('Response correction can only be done within the function for NW.HQIL. Input sta_data in units of acceleration as an alternative')
        resp_file = '../metadata/NW_HQIL_' + chan + '_RESP.txt'
        inst_resp, freqs = invsim.evalresp(dt, nfft_subsgmt, resp_file, UTCDateTime('2013-12-29T00:00:00'), units = 'ACC', freq = True)
        inst_resp = np.abs(inst_resp[1:])
        power = power / (inst_resp ** 2)

    # Convert to decibel scale
    psd = 10 * np.log10(power)

    if smooth:
        # Compute the power after 1/8 octave smoothing
        smooth_pwr = classicsmooth(power, octaveFraction=octavefrac)
        smooth_psd = 10 * np.log10(smooth_pwr)
        return freq, psd, smooth_psd
    else:
        return freq, psd
    
def cut_gaps(times, s_lt, s_st, chan, prnt_folder, gapfilename='NQ_HQIL_gaps_savedfiles.txt'):
    gapfile = np.loadtxt(constants.RAW24H_PATH / prnt_folder / chan / gapfilename, dtype='str')
    data_start = dates.date2num(times - s_lt)
    data_end = dates.date2num(times + s_st)
    indices_mask = np.ones((len(times)), dtype=bool)  # Initially, all elements are considered as selected (True)

    print(f'Checking for gaps in {chan} data... This could take a few minutes')
    for gap in gapfile:
        #print("Running for gap: ", gap)
        gapstart_mpl = dates.date2num(UTCDateTime(gap[0]))
        gapend_mpl = dates.date2num(UTCDateTime(gap[1]))
        gap_indices = np.where(~((data_start > gapend_mpl) | (data_end < gapstart_mpl)))[0]

        if len(gap_indices) > 0:
            indices_mask[gap_indices] = False  # Set the indices which are part of the gap to false 
            print(f"Found {len(gap_indices)} indices within:", gap)

    new_times = times[indices_mask]
    cut_times = list(map(str, times[~indices_mask]))

    return new_times, cut_times, indices_mask 
