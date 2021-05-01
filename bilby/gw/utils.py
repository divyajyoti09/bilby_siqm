import os
import json
from math import fmod

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from ..core.utils import (ra_dec_to_theta_phi,
                          speed_of_light, logger, run_commandline,
                          check_directory_exists_and_if_not_mkdir,
                          SamplesSummary, theta_phi_to_ra_dec)

try:
    from gwpy.timeseries import TimeSeries
except ImportError:
    logger.debug("You do not have gwpy installed currently. You will "
                 " not be able to use some of the prebuilt functions.")

try:
    import lal
    import lalsimulation as lalsim
except ImportError:
    logger.debug("You do not have lalsuite installed currently. You will"
                 " not be able to use some of the prebuilt functions.")


def asd_from_freq_series(freq_data, df):
    """
    Calculate the ASD from the frequency domain output of gaussian_noise()

    Parameters
    ==========
    freq_data: array_like
        Array of complex frequency domain data
    df: float
        Spacing of freq_data, 1/(segment length) used to generate the gaussian noise

    Returns
    =======
    array_like: array of real-valued normalized frequency domain ASD data

    """
    return np.absolute(freq_data) * 2 * df**0.5


def psd_from_freq_series(freq_data, df):
    """
    Calculate the PSD from the frequency domain output of gaussian_noise()
    Calls asd_from_freq_series() and squares the output

    Parameters
    ==========
    freq_data: array_like
        Array of complex frequency domain data
    df: float
        Spacing of freq_data, 1/(segment length) used to generate the gaussian noise

    Returns
    =======
    array_like: Real-valued normalized frequency domain PSD data

    """
    return np.power(asd_from_freq_series(freq_data, df), 2)


def time_delay_geocentric(detector1, detector2, ra, dec, time):
    """
    Calculate time delay between two detectors in geocentric coordinates based on XLALArrivaTimeDiff in TimeDelay.c

    Parameters
    ==========
    detector1: array_like
        Cartesian coordinate vector for the first detector in the geocentric frame
        generated by the Interferometer class as self.vertex.
    detector2: array_like
        Cartesian coordinate vector for the second detector in the geocentric frame.
        To get time delay from Earth center, use detector2 = np.array([0,0,0])
    ra: float
        Right ascension of the source in radians
    dec: float
        Declination of the source in radians
    time: float
        GPS time in the geocentric frame

    Returns
    =======
    float: Time delay between the two detectors in the geocentric frame

    """
    gmst = fmod(lal.GreenwichMeanSiderealTime(time), 2 * np.pi)
    theta, phi = ra_dec_to_theta_phi(ra, dec, gmst)
    omega = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    delta_d = detector2 - detector1
    return np.dot(omega, delta_d) / speed_of_light


def get_polarization_tensor(ra, dec, time, psi, mode):
    """
    Calculate the polarization tensor for a given sky location and time

    See Nishizawa et al. (2009) arXiv:0903.0528 for definitions of the polarisation tensors.
    [u, v, w] represent the Earth-frame
    [m, n, omega] represent the wave-frame
    Note: there is a typo in the definition of the wave-frame in Nishizawa et al.

    Parameters
    ==========
    ra: float
        right ascension in radians
    dec: float
        declination in radians
    time: float
        geocentric GPS time
    psi: float
        binary polarisation angle counter-clockwise about the direction of propagation
    mode: str
        polarisation mode

    Returns
    =======
    array_like: A 3x3 representation of the polarization_tensor for the specified mode.

    """
    gmst = fmod(lal.GreenwichMeanSiderealTime(time), 2 * np.pi)
    theta, phi = ra_dec_to_theta_phi(ra, dec, gmst)
    u = np.array([np.cos(phi) * np.cos(theta), np.cos(theta) * np.sin(phi), -np.sin(theta)])
    v = np.array([-np.sin(phi), np.cos(phi), 0])
    m = -u * np.sin(psi) - v * np.cos(psi)
    n = -u * np.cos(psi) + v * np.sin(psi)

    if mode.lower() == 'plus':
        return np.einsum('i,j->ij', m, m) - np.einsum('i,j->ij', n, n)
    elif mode.lower() == 'cross':
        return np.einsum('i,j->ij', m, n) + np.einsum('i,j->ij', n, m)
    elif mode.lower() == 'breathing':
        return np.einsum('i,j->ij', m, m) + np.einsum('i,j->ij', n, n)

    # Calculating omega here to avoid calculation when model in [plus, cross, breathing]
    omega = np.cross(m, n)
    if mode.lower() == 'longitudinal':
        return np.einsum('i,j->ij', omega, omega)
    elif mode.lower() == 'x':
        return np.einsum('i,j->ij', m, omega) + np.einsum('i,j->ij', omega, m)
    elif mode.lower() == 'y':
        return np.einsum('i,j->ij', n, omega) + np.einsum('i,j->ij', omega, n)
    else:
        raise ValueError("{} not a polarization mode!".format(mode))


def get_vertex_position_geocentric(latitude, longitude, elevation):
    """
    Calculate the position of the IFO vertex in geocentric coordinates in meters.

    Based on arXiv:gr-qc/0008066 Eqs. B11-B13 except for the typo in the definition of the local radius.
    See Section 2.1 of LIGO-T980044-10 for the correct expression

    Parameters
    ==========
    latitude: float
        Latitude in radians
    longitude:
        Longitude in radians
    elevation:
        Elevation in meters

    Returns
    =======
    array_like: A 3D representation of the geocentric vertex position

    """
    semi_major_axis = 6378137  # for ellipsoid model of Earth, in m
    semi_minor_axis = 6356752.314  # in m
    radius = semi_major_axis**2 * (semi_major_axis**2 * np.cos(latitude)**2 +
                                   semi_minor_axis**2 * np.sin(latitude)**2)**(-0.5)
    x_comp = (radius + elevation) * np.cos(latitude) * np.cos(longitude)
    y_comp = (radius + elevation) * np.cos(latitude) * np.sin(longitude)
    z_comp = ((semi_minor_axis / semi_major_axis)**2 * radius + elevation) * np.sin(latitude)
    return np.array([x_comp, y_comp, z_comp])


def inner_product(aa, bb, frequency, PSD):
    """
    Calculate the inner product defined in the matched filter statistic

    Parameters
    ==========
    aa, bb: array_like
        Single-sided Fourier transform, created, e.g., by the nfft function above
    frequency: array_like
        An array of frequencies associated with aa, bb, also returned by nfft
    PSD: bilby.gw.detector.PowerSpectralDensity

    Returns
    =======
    The matched filter inner product for aa and bb

    """
    psd_interp = PSD.power_spectral_density_interpolated(frequency)

    # calculate the inner product
    integrand = np.conj(aa) * bb / psd_interp

    df = frequency[1] - frequency[0]
    integral = np.sum(integrand) * df
    return 4. * np.real(integral)


def noise_weighted_inner_product(aa, bb, power_spectral_density, duration):
    """
    Calculate the noise weighted inner product between two arrays.

    Parameters
    ==========
    aa: array_like
        Array to be complex conjugated
    bb: array_like
        Array not to be complex conjugated
    power_spectral_density: array_like
        Power spectral density of the noise
    duration: float
        duration of the data

    Returns
    ======
    Noise-weighted inner product.
    """

    integrand = np.conj(aa) * bb / power_spectral_density
    return 4 / duration * np.sum(integrand)


def matched_filter_snr(signal, frequency_domain_strain, power_spectral_density, duration):
    """
    Calculate the _complex_ matched filter snr of a signal.
    This is <signal|frequency_domain_strain> / optimal_snr

    Parameters
    ==========
    signal: array_like
        Array containing the signal
    frequency_domain_strain: array_like

    power_spectral_density: array_like

    duration: float
        Time duration of the signal

    Returns
    =======
    float: The matched filter signal to noise ratio squared

    """
    rho_mf = noise_weighted_inner_product(
        aa=signal, bb=frequency_domain_strain,
        power_spectral_density=power_spectral_density, duration=duration)
    rho_mf /= optimal_snr_squared(
        signal=signal, power_spectral_density=power_spectral_density,
        duration=duration)**0.5
    return rho_mf


def optimal_snr_squared(signal, power_spectral_density, duration):
    """

    Parameters
    ==========
    signal: array_like
        Array containing the signal
    power_spectral_density: array_like

    duration: float
        Time duration of the signal

    Returns
    =======
    float: The matched filter signal to noise ratio squared

    """
    return noise_weighted_inner_product(signal, signal, power_spectral_density, duration)


def overlap(signal_a, signal_b, power_spectral_density=None, delta_frequency=None,
            lower_cut_off=None, upper_cut_off=None, norm_a=None, norm_b=None):
    low_index = int(lower_cut_off / delta_frequency)
    up_index = int(upper_cut_off / delta_frequency)
    integrand = np.conj(signal_a) * signal_b
    integrand = integrand[low_index:up_index] / power_spectral_density[low_index:up_index]
    integral = (4 * delta_frequency * integrand) / norm_a / norm_b
    return sum(integral).real


__cached_euler_matrix = None
__cached_delta_x = None


def euler_rotation(delta_x):
    """
    Calculate the rotation matrix mapping the vector (0, 0, 1) to delta_x
    while preserving the origin of the azimuthal angle.

    This is decomposed into three Euler angle, alpha, beta, gamma, which rotate
    about the z-, y-, and z- axes respectively.

    Parameters
    ==========
    delta_x: array-like (3,)
        Vector onto which (0, 0, 1) should be mapped.

    Returns
    =======
    total_rotation: array-like (3,3)
        Rotation matrix which maps vectors from the frame in which delta_x is
        aligned with the z-axis to the target frame.
    """
    global __cached_delta_x
    global __cached_euler_matrix

    delta_x = delta_x / np.sum(delta_x**2)**0.5
    if np.array_equal(delta_x, __cached_delta_x):
        return __cached_euler_matrix
    else:
        __cached_delta_x = delta_x
    alpha = np.arctan(- delta_x[1] * delta_x[2] / delta_x[0])
    beta = np.arccos(delta_x[2])
    gamma = np.arctan(delta_x[1] / delta_x[0])
    rotation_1 = np.array([
        [np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0],
        [0, 0, 1]])
    rotation_2 = np.array([
        [np.cos(beta), 0, - np.sin(beta)], [0, 1, 0],
        [np.sin(beta), 0, np.cos(beta)]])
    rotation_3 = np.array([
        [np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1]])
    total_rotation = np.einsum(
        'ij,jk,kl->il', rotation_3, rotation_2, rotation_1)
    __cached_delta_x = delta_x
    __cached_euler_matrix = total_rotation
    return total_rotation


def zenith_azimuth_to_theta_phi(zenith, azimuth, ifos):
    """
    Convert from the 'detector frame' to the Earth frame.

    Parameters
    ==========
    kappa: float
        The zenith angle in the detector frame
    eta: float
        The azimuthal angle in the detector frame
    ifos: list
        List of Interferometer objects defining the detector frame

    Returns
    =======
    theta, phi: float
        The zenith and azimuthal angles in the earth frame.
    """
    delta_x = ifos[0].geometry.vertex - ifos[1].geometry.vertex
    omega_prime = np.array([
        np.sin(zenith) * np.cos(azimuth),
        np.sin(zenith) * np.sin(azimuth),
        np.cos(zenith)])
    rotation_matrix = euler_rotation(delta_x)
    omega = np.dot(rotation_matrix, omega_prime)
    theta = np.arccos(omega[2])
    phi = np.arctan2(omega[1], omega[0]) % (2 * np.pi)
    return theta, phi


def zenith_azimuth_to_ra_dec(zenith, azimuth, geocent_time, ifos):
    """
    Convert from the 'detector frame' to the Earth frame.

    Parameters
    ==========
    kappa: float
        The zenith angle in the detector frame
    eta: float
        The azimuthal angle in the detector frame
    geocent_time: float
        GPS time at geocenter
    ifos: list
        List of Interferometer objects defining the detector frame

    Returns
    =======
    ra, dec: float
        The zenith and azimuthal angles in the sky frame.
    """
    theta, phi = zenith_azimuth_to_theta_phi(zenith, azimuth, ifos)
    gmst = lal.GreenwichMeanSiderealTime(geocent_time)
    ra, dec = theta_phi_to_ra_dec(theta, phi, gmst)
    ra = ra % (2 * np.pi)
    return ra, dec


def get_event_time(event):
    """
    Get the merger time for known GW events.

    See https://www.gw-openscience.org/catalog/GWTC-1-confident/html/
    Last update https://arxiv.org/abs/1811.12907:

    - GW150914
    - GW151012
    - GW151226
    - GW170104
    - GW170608
    - GW170729
    - GW170809
    - GW170814
    - GW170817
    - GW170818
    - GW170823

    Parameters
    ==========
    event: str
        Event descriptor, this can deal with some prefixes, e.g.,
        '151012', 'GW151012', 'LVT151012'

    Returns
    =======
    event_time: float
        Merger time
    """
    event_times = {'150914': 1126259462.4,
                   '151012': 1128678900.4,
                   '151226': 1135136350.6,
                   '170104': 1167559936.6,
                   '170608': 1180922494.5,
                   '170729': 1185389807.3,
                   '170809': 1186302519.8,
                   '170814': 1186741861.5,
                   '170817': 1187008882.4,
                   '170818': 1187058327.1,
                   '170823': 1187529256.5}
    if 'GW' or 'LVT' in event:
        event = event[-6:]

    try:
        event_time = event_times[event[-6:]]
        return event_time
    except KeyError:
        print('Unknown event {}.'.format(event))
        return None


def get_open_strain_data(
        name, start_time, end_time, outdir, cache=False, buffer_time=0, **kwargs):
    """ A function which accesses the open strain data

    This uses `gwpy` to download the open data and then saves a cached copy for
    later use

    Parameters
    ==========
    name: str
        The name of the detector to get data for
    start_time, end_time: float
        The GPS time of the start and end of the data
    outdir: str
        The output directory to place data in
    cache: bool
        If true, cache the data
    buffer_time: float
        Time to add to the begining and end of the segment.
    **kwargs:
        Passed to `gwpy.timeseries.TimeSeries.fetch_open_data`

    Returns
    =======
    strain: gwpy.timeseries.TimeSeries
        The object containing the strain data. If the connection to the open-data server
        fails, this function retruns `None`.

    """
    filename = '{}/{}_{}_{}.txt'.format(outdir, name, start_time, end_time)

    if buffer_time < 0:
        raise ValueError("buffer_time < 0")
    start_time = start_time - buffer_time
    end_time = end_time + buffer_time

    if os.path.isfile(filename) and cache:
        logger.info('Using cached data from {}'.format(filename))
        strain = TimeSeries.read(filename)
    else:
        logger.info('Fetching open data from {} to {} with buffer time {}'
                    .format(start_time, end_time, buffer_time))
        try:
            strain = TimeSeries.fetch_open_data(name, start_time, end_time, **kwargs)
            logger.info('Saving cache of data to {}'.format(filename))
            strain.write(filename)
        except Exception as e:
            logger.info("Unable to fetch open data, see debug for detailed info")
            logger.info("Call to gwpy.timeseries.TimeSeries.fetch_open_data returned {}"
                        .format(e))
            strain = None

    return strain


def read_frame_file(file_name, start_time, end_time, channel=None, buffer_time=0, **kwargs):
    """ A function which accesses the open strain data

    This uses `gwpy` to download the open data and then saves a cached copy for
    later use

    Parameters
    ==========
    file_name: str
        The name of the frame to read
    start_time, end_time: float
        The GPS time of the start and end of the data
    buffer_time: float
        Read in data with `t1-buffer_time` and `end_time+buffer_time`
    channel: str
        The name of the channel being searched for, some standard channel names are attempted
        if channel is not specified or if specified channel is not found.
    **kwargs:
        Passed to `gwpy.timeseries.TimeSeries.fetch_open_data`

    Returns
    =======
    strain: gwpy.timeseries.TimeSeries

    """
    loaded = False
    strain = None

    if channel is not None:
        try:
            strain = TimeSeries.read(source=file_name, channel=channel, start=start_time, end=end_time, **kwargs)
            loaded = True
            logger.info('Successfully loaded {}.'.format(channel))
        except RuntimeError:
            logger.warning('Channel {} not found. Trying preset channel names'.format(channel))

    if loaded is False:
        ligo_channel_types = ['GDS-CALIB_STRAIN', 'DCS-CALIB_STRAIN_C01', 'DCS-CALIB_STRAIN_C02',
                              'DCH-CLEAN_STRAIN_C02', 'GWOSC-16KHZ_R1_STRAIN',
                              'GWOSC-4KHZ_R1_STRAIN']
        virgo_channel_types = ['Hrec_hoft_V1O2Repro2A_16384Hz', 'FAKE_h_16384Hz_4R',
                               'GWOSC-16KHZ_R1_STRAIN', 'GWOSC-4KHZ_R1_STRAIN']
        channel_types = dict(H1=ligo_channel_types, L1=ligo_channel_types, V1=virgo_channel_types)
        for detector in channel_types.keys():
            for channel_type in channel_types[detector]:
                if loaded:
                    break
                channel = '{}:{}'.format(detector, channel_type)
                try:
                    strain = TimeSeries.read(source=file_name, channel=channel, start=start_time, end=end_time,
                                             **kwargs)
                    loaded = True
                    logger.info('Successfully read strain data for channel {}.'.format(channel))
                except RuntimeError:
                    pass

    if loaded:
        return strain
    else:
        logger.warning('No data loaded.')
        return None


def get_gracedb(gracedb, outdir, duration, calibration, detectors, query_types=None, server=None):
    candidate = gracedb_to_json(gracedb, outdir=outdir)
    trigger_time = candidate['gpstime']
    gps_start_time = trigger_time - duration
    cache_files = []
    if query_types is None:
        query_types = [None] * len(detectors)
    for i, det in enumerate(detectors):
        output_cache_file = gw_data_find(
            det, gps_start_time, duration, calibration,
            outdir=outdir, query_type=query_types[i], server=server)
        cache_files.append(output_cache_file)
    return candidate, cache_files


def gracedb_to_json(gracedb, cred=None, service_url='https://gracedb.ligo.org/api/', outdir=None):
    """ Script to download a GraceDB candidate

    Parameters
    ==========
    gracedb: str
        The UID of the GraceDB candidate
    cred:
        Credentials for authentications, see ligo.gracedb.rest.GraceDb
    service_url:
        The url of the GraceDB candidate
        GraceDB 'https://gracedb.ligo.org/api/' (default)
        GraceDB-playground 'https://gracedb-playground.ligo.org/api/'
    outdir: str, optional
        If given, a string identfying the location in which to store the json
    """
    logger.info(
        'Starting routine to download GraceDb candidate {}'.format(gracedb))
    from ligo.gracedb.rest import GraceDb

    logger.info('Initialise client and attempt to download')
    logger.info('Fetching from {}'.format(service_url))
    try:
        client = GraceDb(cred=cred, service_url=service_url)
    except IOError:
        raise ValueError(
            'Failed to authenticate with gracedb: check your X509 '
            'certificate is accessible and valid')
    try:
        candidate = client.event(gracedb)
        logger.info('Successfully downloaded candidate')
    except Exception as e:
        raise ValueError("Unable to obtain GraceDB candidate, exception: {}".format(e))

    json_output = candidate.json()

    if outdir is not None:
        check_directory_exists_and_if_not_mkdir(outdir)
        outfilepath = os.path.join(outdir, '{}.json'.format(gracedb))
        logger.info('Writing candidate to {}'.format(outfilepath))
        with open(outfilepath, 'w') as outfile:
            json.dump(json_output, outfile, indent=2)

    return json_output


def gw_data_find(observatory, gps_start_time, duration, calibration,
                 outdir='.', query_type=None, server=None):
    """ Builds a gw_data_find call and process output

    Parameters
    ==========
    observatory: str, {H1, L1, V1}
        Observatory description
    gps_start_time: float
        The start time in gps to look for data
    duration: int
        The duration (integer) in s
    calibrartion: int {1, 2}
        Use C01 or C02 calibration
    outdir: string
        A path to the directory where output is stored
    query_type: string
        The LDRDataFind query type

    Returns
    =======
    output_cache_file: str
        Path to the output cache file

    """
    logger.info('Building gw_data_find command line')

    observatory_lookup = dict(H1='H', L1='L', V1='V')
    observatory_code = observatory_lookup[observatory]

    if query_type is None:
        logger.warning('No query type provided. This may prevent data from being read.')
        if observatory_code == 'V':
            query_type = 'V1Online'
        else:
            query_type = '{}_HOFT_C0{}'.format(observatory, calibration)

    logger.info('Using LDRDataFind query type {}'.format(query_type))

    cache_file = '{}-{}_CACHE-{}-{}.lcf'.format(
        observatory, query_type, gps_start_time, duration)
    output_cache_file = os.path.join(outdir, cache_file)

    gps_end_time = gps_start_time + duration
    if server is None:
        server = os.environ.get("LIGO_DATAFIND_SERVER")
        if server is None:
            logger.warning('No data_find server found, defaulting to CIT server. '
                           'To run on other clusters, pass the output of '
                           '`echo $LIGO_DATAFIND_SERVER`')
            server = 'ldr.ldas.cit:80'

    cl_list = ['gw_data_find']
    cl_list.append('--observatory {}'.format(observatory_code))
    cl_list.append('--gps-start-time {}'.format(int(np.floor(gps_start_time))))
    cl_list.append('--gps-end-time {}'.format(int(np.ceil(gps_end_time))))
    cl_list.append('--type {}'.format(query_type))
    cl_list.append('--output {}'.format(output_cache_file))
    cl_list.append('--server {}'.format(server))
    cl_list.append('--url-type file')
    cl_list.append('--lal-cache')
    cl = ' '.join(cl_list)
    run_commandline(cl)
    return output_cache_file


def build_roq_weights(data, basis, deltaF):
    """
    For a data array and reduced basis compute roq weights

    Parameters
    ==========
    data: array-like
        data set
    basis: array-like
        (reduced basis element)*invV (the inverse Vandermonde matrix)
    deltaF: float
        integration element df

    """
    weights = np.dot(data, np.conjugate(basis)) * deltaF * 4.
    return weights


def blockwise_dot_product(matrix_a, matrix_b, max_elements=int(2 ** 27),
                          out=None):
    """
    Memory efficient
    Computes the dot product of two matrices in a block-wise fashion.
    Only blocks of `matrix_a` with a maximum size of `max_elements` will be
    processed simultaneously.

    Parameters
    ==========
    matrix_a, matrix_b: array-like
        Matrices to be dot producted, matrix_b is complex conjugated.
    max_elements: int
        Maximum number of elements to consider simultaneously, should be memory
        limited.
    out: array-like
        Output array

    Returns
    =======
    out: array-like
        Dot producted array
    """
    def block_slices(dim_size, block_size):
        """Generator that yields slice objects for indexing into
        sequential blocks of an array along a particular axis
        Useful for blockwise dot
        """
        count = 0
        while True:
            yield slice(count, count + block_size, 1)
            count += block_size
            if count > dim_size:
                return

    matrix_b = np.conjugate(matrix_b)
    m, n = matrix_a.shape
    n1, o = matrix_b.shape
    if n1 != n:
        raise ValueError(
            'Matrices are not aligned, matrix a has shape ' +
            '{}, matrix b has shape {}.'.format(matrix_a.shape, matrix_b.shape))

    if matrix_a.flags.f_contiguous:
        # prioritize processing as many columns of matrix_a as possible
        max_cols = max(1, max_elements // m)
        max_rows = max_elements // max_cols

    else:
        # prioritize processing as many rows of matrix_a as possible
        max_rows = max(1, max_elements // n)
        max_cols = max_elements // max_rows

    if out is None:
        out = np.empty((m, o), dtype=np.result_type(matrix_a, matrix_b))
    elif out.shape != (m, o):
        raise ValueError('Output array has incorrect dimensions.')

    for mm in block_slices(m, max_rows):
        out[mm, :] = 0
        for nn in block_slices(n, max_cols):
            a_block = matrix_a[mm, nn].copy()  # copy to force a read
            out[mm, :] += np.dot(a_block, matrix_b[nn, :])
            del a_block

    return out


def convert_args_list_to_float(*args_list):
    """ Converts inputs to floats, returns a list in the same order as the input"""
    try:
        args_list = [float(arg) for arg in args_list]
    except ValueError:
        raise ValueError("Unable to convert inputs to floats")
    return args_list


def lalsim_SimInspiralTransformPrecessingNewInitialConditions(
        theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1, mass_2,
        reference_frequency, phase):

    args_list = convert_args_list_to_float(
        theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1, mass_2,
        reference_frequency, phase)

    return lalsim.SimInspiralTransformPrecessingNewInitialConditions(*args_list)


def lalsim_GetApproximantFromString(waveform_approximant):
    if isinstance(waveform_approximant, str):
        return lalsim.GetApproximantFromString(waveform_approximant)
    else:
        raise ValueError("waveform_approximant must be of type str")


def lalsim_SimInspiralFD(
        mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y,
        spin_2z, luminosity_distance, iota, phase,
        longitude_ascending_nodes, eccentricity, mean_per_ano, delta_frequency,
        minimum_frequency, maximum_frequency, reference_frequency,
        waveform_dictionary, approximant):
    """
    Safely call lalsimulation.SimInspiralFD

    Parameters
    ==========
    phase: float, int
    mass_1: float, int
    mass_2: float, int
    spin_1x: float, int
    spin_1y: float, int
    spin_1z: float, int
    spin_2x: float, int
    spin_2y: float, int
    spin_2z: float, int
    reference_frequency: float, int
    luminosity_distance: float, int
    iota: float, int
    longitude_ascending_nodes: float, int
    eccentricity: float, int
    mean_per_ano: float, int
    delta_frequency: float, int
    minimum_frequency: float, int
    maximum_frequency: float, int
    waveform_dictionary: None, lal.Dict
    approximant: int, str
    """

    args = convert_args_list_to_float(
        mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z,
        luminosity_distance, iota, phase, longitude_ascending_nodes,
        eccentricity, mean_per_ano, delta_frequency, minimum_frequency,
        maximum_frequency, reference_frequency)

    approximant = _get_lalsim_approximant(approximant)

    return lalsim.SimInspiralFD(*args, waveform_dictionary, approximant)


def lalsim_SimInspiralChooseFDWaveform(
        mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y,
        spin_2z, luminosity_distance, iota, phase,
        longitude_ascending_nodes, eccentricity, mean_per_ano, delta_frequency,
        minimum_frequency, maximum_frequency, reference_frequency,
        waveform_dictionary, approximant):
    """
    Safely call lalsimulation.SimInspiralChooseFDWaveform

    Parameters
    ==========
    phase: float, int
    mass_1: float, int
    mass_2: float, int
    spin_1x: float, int
    spin_1y: float, int
    spin_1z: float, int
    spin_2x: float, int
    spin_2y: float, int
    spin_2z: float, int
    reference_frequency: float, int
    luminosity_distance: float, int
    iota: float, int
    longitude_ascending_nodes: float, int
    eccentricity: float, int
    mean_per_ano: float, int
    delta_frequency: float, int
    minimum_frequency: float, int
    maximum_frequency: float, int
    waveform_dictionary: None, lal.Dict
    approximant: int, str
    """

    args = convert_args_list_to_float(
        mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z,
        luminosity_distance, iota, phase, longitude_ascending_nodes,
        eccentricity, mean_per_ano, delta_frequency, minimum_frequency,
        maximum_frequency, reference_frequency)

    approximant = _get_lalsim_approximant(approximant)

    return lalsim.SimInspiralChooseFDWaveform(*args, waveform_dictionary, approximant)


def _get_lalsim_approximant(approximant):
    if isinstance(approximant, int):
        pass
    elif isinstance(approximant, str):
        approximant = lalsim_GetApproximantFromString(approximant)
    else:
        raise ValueError("approximant not an int")
    return approximant


def lalsim_SimInspiralChooseFDWaveformSequence(
        phase, mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y,
        spin_2z, reference_frequency, luminosity_distance, iota,
        waveform_dictionary, approximant, frequency_array):
    """
    Safely call lalsimulation.SimInspiralChooseFDWaveformSequence

    Parameters
    ==========
    phase: float, int
    mass_1: float, int
    mass_2: float, int
    spin_1x: float, int
    spin_1y: float, int
    spin_1z: float, int
    spin_2x: float, int
    spin_2y: float, int
    spin_2z: float, int
    reference_frequency: float, int
    luminosity_distance: float, int
    iota: float, int
    waveform_dictionary: None, lal.Dict
    approximant: int, str
    frequency_array: np.ndarray, lal.REAL8Vector
    """

    [mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z,
     luminosity_distance, iota, phase, reference_frequency] = convert_args_list_to_float(
        mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z,
        luminosity_distance, iota, phase, reference_frequency)

    if isinstance(approximant, int):
        pass
    elif isinstance(approximant, str):
        approximant = lalsim_GetApproximantFromString(approximant)
    else:
        raise ValueError("approximant not an int")

    if not isinstance(frequency_array, lal.REAL8Vector):
        old_frequency_array = frequency_array.copy()
        frequency_array = lal.CreateREAL8Vector(len(old_frequency_array))
        frequency_array.data = old_frequency_array

    return lalsim.SimInspiralChooseFDWaveformSequence(
        phase, mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y,
        spin_2z, reference_frequency, luminosity_distance, iota,
        waveform_dictionary, approximant, frequency_array)


def lalsim_SimInspiralWaveformParamsInsertTidalLambda1(
        waveform_dictionary, lambda_1):
    try:
        lambda_1 = float(lambda_1)
    except ValueError:
        raise ValueError("Unable to convert lambda_1 to float")

    return lalsim.SimInspiralWaveformParamsInsertTidalLambda1(
        waveform_dictionary, lambda_1)


def lalsim_SimInspiralWaveformParamsInsertdQuadMon1(
        waveform_dictionary, dQuadMon1):
    try:
        dQuadMon1 = float(dQuadMon1)
    except ValueError:
        raise ValueError("Unable to convert dQuadMon1  to float")

    return lalsim.SimInspiralWaveformParamsInsertdQuadMon1(
        waveform_dictionary, dQuadMon1)


def lalsim_SimInspiralWaveformParamsInsertdQuadMon2(
        waveform_dictionary, dQuadMon2):
    try:
        dQuadMon2 = float(dQuadMon2)
    except ValueError:
        raise ValueError("Unable to convert dQuadMon1  to float")

    return lalsim.SimInspiralWaveformParamsInsertdQuadMon2(
        waveform_dictionary, dQuadMon2)


def lalsim_SimInspiralWaveformParamsInsertTidalLambda2(
        waveform_dictionary, lambda_2):
    try:
        lambda_2 = float(lambda_2)
    except ValueError:
        raise ValueError("Unable to convert lambda_2 to float")

    return lalsim.SimInspiralWaveformParamsInsertTidalLambda2(
        waveform_dictionary, lambda_2)


def spline_angle_xform(delta_psi):
    """
    Returns the angle in degrees corresponding to the spline
    calibration parameters delta_psi.
    Based on the same function in lalinference.bayespputils

    Parameters
    ==========
    delta_psi: calibration phase uncertainity

    Returns
    =======
    float: delta_psi in degrees

    """
    rotation = (2.0 + 1.0j * delta_psi) / (2.0 - 1.0j * delta_psi)

    return 180.0 / np.pi * np.arctan2(np.imag(rotation), np.real(rotation))


def plot_spline_pos(log_freqs, samples, nfreqs=100, level=0.9, color='k', label=None, xform=None):
    """
    Plot calibration posterior estimates for a spline model in log space.
    Adapted from the same function in lalinference.bayespputils

    Parameters
    ==========
    log_freqs: array-like
        The (log) location of spline control points.
    samples: array-like
        List of posterior draws of function at control points ``log_freqs``
    nfreqs: int
        Number of points to evaluate spline at for plotting.
    level: float
        Credible level to fill in.
    color: str
        Color to plot with.
    label: str
        Label for plot.
    xform: callable
        Function to transform the spline into plotted values.

    """
    freq_points = np.exp(log_freqs)
    freqs = np.logspace(min(log_freqs), max(log_freqs), nfreqs, base=np.exp(1))

    data = np.zeros((samples.shape[0], nfreqs))

    if xform is None:
        scaled_samples = samples
    else:
        scaled_samples = xform(samples)

    scaled_samples_summary = SamplesSummary(scaled_samples, average='mean')
    data_summary = SamplesSummary(data, average='mean')

    plt.errorbar(freq_points, scaled_samples_summary.average,
                 yerr=[-scaled_samples_summary.lower_relative_credible_interval,
                       scaled_samples_summary.upper_relative_credible_interval],
                 fmt='.', color=color, lw=4, alpha=0.5, capsize=0)

    for i, sample in enumerate(samples):
        temp = interp1d(
            log_freqs, sample, kind="cubic", fill_value=0,
            bounds_error=False)(np.log(freqs))
        if xform is None:
            data[i] = temp
        else:
            data[i] = xform(temp)

    plt.plot(freqs, np.mean(data, axis=0), color=color, label=label)
    plt.fill_between(freqs, data_summary.lower_absolute_credible_interval,
                     data_summary.upper_absolute_credible_interval,
                     color=color, alpha=.1, linewidth=0.1)
    plt.xlim(freq_points.min() - .5, freq_points.max() + 50)


class PropertyAccessor(object):
    """
    Generic descriptor class that allows handy access of properties without long
    boilerplate code. The properties of Interferometer are defined as instances
    of this class.

    This avoids lengthy code like

    .. code-block:: python

        @property
        def length(self):
            return self.geometry.length

        @length_setter
        def length(self, length)
            self.geometry.length = length

    in the Interferometer class
    """

    def __init__(self, container_instance_name, property_name):
        self.property_name = property_name
        self.container_instance_name = container_instance_name

    def __get__(self, instance, owner):
        return getattr(getattr(instance, self.container_instance_name), self.property_name)

    def __set__(self, instance, value):
        setattr(getattr(instance, self.container_instance_name), self.property_name, value)
