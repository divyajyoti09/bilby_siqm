import numpy as np

from ..core import utils
from ..core.utils import logger
from .conversion import bilby_to_lalsimulation_spins
from .utils import (lalsim_GetApproximantFromString,
                    lalsim_SimInspiralFD,
                    lalsim_SimInspiralChooseFDWaveform,
                    lalsim_SimInspiralWaveformParamsInsertTidalLambda1,
                    lalsim_SimInspiralWaveformParamsInsertTidalLambda2,
                    lalsim_SimInspiralChooseFDWaveformSequence)


def lal_binary_black_hole(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs):
    """ A Binary Black Hole waveform model using lalsimulation

    Parameters
    ==========
    frequency_array: array_like
        The frequencies at which we want to calculate the strain
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    luminosity_distance: float
        The luminosity distance in megaparsec
    a_1: float
        Dimensionless primary spin magnitude
    tilt_1: float
        Primary tilt angle
    phi_12: float
        Azimuthal angle between the two component spins
    a_2: float
        Dimensionless secondary spin magnitude
    tilt_2: float
        Secondary tilt angle
    phi_jl: float
        Azimuthal angle between the total binary angular momentum and the
        orbital angular momentum
    theta_jn: float
        Angle between the total binary angular momentum and the line of sight
    phase: float
        The phase at coalescence
    kwargs: dict
        Optional keyword arguments
        Supported arguments:

        - waveform_approximant
        - reference_frequency
        - minimum_frequency
        - maximum_frequency
        - catch_waveform_errors
        - pn_spin_order
        - pn_tidal_order
        - pn_phase_order
        - pn_amplitude_order
        - mode_array:
          Activate a specific mode array and evaluate the model using those
          modes only.  e.g. waveform_arguments =
          dict(waveform_approximant='IMRPhenomHM', mode_array=[[2,2],[2,-2])
          returns the 22 and 2-2 modes only of IMRPhenomHM.  You can only
          specify modes that are included in that particular model.  e.g.
          waveform_arguments = dict(waveform_approximant='IMRPhenomHM',
          mode_array=[[2,2],[2,-2],[5,5],[5,-5]]) is not allowed because the
          55 modes are not included in this model.  Be aware that some models
          only take positive modes and return the positive and the negative
          mode together, while others need to call both.  e.g.
          waveform_arguments = dict(waveform_approximant='IMRPhenomHM',
          mode_array=[[2,2],[4,-4]]) returns the 22 and 2-2 of IMRPhenomHM.
          However, waveform_arguments =
          dict(waveform_approximant='IMRPhenomXHM', mode_array=[[2,2],[4,-4]])
          returns the 22 and 4-4 of IMRPhenomXHM.
        - lal_waveform_dictionary:
          A dictionary (lal.Dict) of arguments passed to the lalsimulation
          waveform generator. The arguments are specific to the waveform used.

    Returns
    =======
    dict: A dictionary with the plus and cross polarisation strain modes
    """
    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomPv2', reference_frequency=50.0,
        minimum_frequency=20.0, maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)
    waveform_kwargs.update(kwargs)
    return _base_lal_cbc_fd_waveform(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12,
        phi_jl=phi_jl, **waveform_kwargs)


def lal_binary_neutron_star(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, lambda_1, lambda_2,
        **kwargs):
    """ A Binary Neutron Star waveform model using lalsimulation

    Parameters
    ==========
    frequency_array: array_like
        The frequencies at which we want to calculate the strain
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    luminosity_distance: float
        The luminosity distance in megaparsec
    a_1: float
        Dimensionless primary spin magnitude
    tilt_1: float
        Primary tilt angle
    phi_12: float
        Azimuthal angle between the two component spins
    a_2: float
        Dimensionless secondary spin magnitude
    tilt_2: float
        Secondary tilt angle
    phi_jl: float
        Azimuthal angle between the total binary angular momentum and the
        orbital angular momentum
    theta_jn: float
        Orbital inclination
    phase: float
        The phase at coalescence
    lambda_1: float
        Dimensionless tidal deformability of mass_1
    lambda_2: float
        Dimensionless tidal deformability of mass_2
    kwargs: dict
        Optional keyword arguments
        Supported arguments:

        - waveform_approximant
        - reference_frequency
        - minimum_frequency
        - maximum_frequency
        - catch_waveform_errors
        - pn_spin_order
        - pn_tidal_order
        - pn_phase_order
        - pn_amplitude_order
        - mode_array:
          Activate a specific mode array and evaluate the model using those
          modes only.  e.g. waveform_arguments =
          dict(waveform_approximant='IMRPhenomHM', mode_array=[[2,2],[2,-2])
          returns the 22 and 2-2 modes only of IMRPhenomHM.  You can only
          specify modes that are included in that particular model.  e.g.
          waveform_arguments = dict(waveform_approximant='IMRPhenomHM',
          mode_array=[[2,2],[2,-2],[5,5],[5,-5]]) is not allowed because the
          55 modes are not included in this model.  Be aware that some models
          only take positive modes and return the positive and the negative
          mode together, while others need to call both.  e.g.
          waveform_arguments = dict(waveform_approximant='IMRPhenomHM',
          mode_array=[[2,2],[4,-4]]) returns the 22 and 2-2 of IMRPhenomHM.
          However, waveform_arguments =
          dict(waveform_approximant='IMRPhenomXHM', mode_array=[[2,2],[4,-4]])
          returns the 22 and 4-4 of IMRPhenomXHM.

    Returns
    =======
    dict: A dictionary with the plus and cross polarisation strain modes
    """
    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomPv2_NRTidal', reference_frequency=50.0,
        minimum_frequency=20.0, maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)
    waveform_kwargs.update(kwargs)
    return _base_lal_cbc_fd_waveform(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12,
        phi_jl=phi_jl, lambda_1=lambda_1, lambda_2=lambda_2, **waveform_kwargs)


def lal_eccentric_binary_black_hole_no_spins(
        frequency_array, mass_1, mass_2, eccentricity, luminosity_distance,
        theta_jn, phase, **kwargs):
    """ Eccentric binary black hole waveform model using lalsimulation (EccentricFD)

    Parameters
    ==========
    frequency_array: array_like
        The frequencies at which we want to calculate the strain
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    eccentricity: float
        The orbital eccentricity of the system
    luminosity_distance: float
        The luminosity distance in megaparsec
    theta_jn: float
        Orbital inclination
    phase: float
        The phase at coalescence
    kwargs: dict
        Optional keyword arguments
        Supported arguments:

        - waveform_approximant
        - reference_frequency
        - minimum_frequency
        - maximum_frequency
        - catch_waveform_errors
        - pn_spin_order
        - pn_tidal_order
        - pn_phase_order
        - pn_amplitude_order
        - mode_array:
          Activate a specific mode array and evaluate the model using those
          modes only.  e.g. waveform_arguments =
          dict(waveform_approximant='IMRPhenomHM', mode_array=[[2,2],[2,-2])
          returns the 22 and 2-2 modes only of IMRPhenomHM.  You can only
          specify modes that are included in that particular model.  e.g.
          waveform_arguments = dict(waveform_approximant='IMRPhenomHM',
          mode_array=[[2,2],[2,-2],[5,5],[5,-5]]) is not allowed because the
          55 modes are not included in this model.  Be aware that some models
          only take positive modes and return the positive and the negative
          mode together, while others need to call both.  e.g.
          waveform_arguments = dict(waveform_approximant='IMRPhenomHM',
          mode_array=[[2,2],[4,-4]]) returns the 22 and 2-2 of IMRPhenomHM.
          However, waveform_arguments =
          dict(waveform_approximant='IMRPhenomXHM', mode_array=[[2,2],[4,-4]])
          returns the 22 and 4-4 of IMRPhenomXHM.

    Returns
    =======
    dict: A dictionary with the plus and cross polarisation strain modes
    """
    waveform_kwargs = dict(
        waveform_approximant='EccentricFD', reference_frequency=10.0,
        minimum_frequency=10.0, maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)
    waveform_kwargs.update(kwargs)
    return _base_lal_cbc_fd_waveform(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        eccentricity=eccentricity, **waveform_kwargs)


def _base_lal_cbc_fd_waveform(
        frequency_array, mass_1, mass_2, luminosity_distance, theta_jn, phase,
        a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0,
        lambda_1=0.0, lambda_2=0.0, eccentricity=0.0, **waveform_kwargs):
    """ Generate a cbc waveform model using lalsimulation

    Parameters
    ==========
    frequency_array: array_like
        The frequencies at which we want to calculate the strain
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    luminosity_distance: float
        The luminosity distance in megaparsec
    a_1: float
        Dimensionless primary spin magnitude
    tilt_1: float
        Primary tilt angle
    phi_12: float
        Azimuthal angle between the component spins
    a_2: float
        Dimensionless secondary spin magnitude
    tilt_2: float
        Secondary tilt angle
    phi_jl: float
        Azimuthal angle between the total and orbital angular momenta
    theta_jn: float
        Orbital inclination
    phase: float
        The phase at coalescence
    eccentricity: float
        Binary eccentricity
    lambda_1: float
        Tidal deformability of the more massive object
    lambda_2: float
        Tidal deformability of the less massive object
    kwargs: dict
        Optional keyword arguments

    Returns
    =======
    dict: A dictionary with the plus and cross polarisation strain modes
    """
    import lal
    import lalsimulation as lalsim

    waveform_approximant = waveform_kwargs['waveform_approximant']
    reference_frequency = waveform_kwargs['reference_frequency']
    minimum_frequency = waveform_kwargs['minimum_frequency']
    maximum_frequency = waveform_kwargs['maximum_frequency']
    catch_waveform_errors = waveform_kwargs['catch_waveform_errors']
    pn_spin_order = waveform_kwargs['pn_spin_order']
    pn_tidal_order = waveform_kwargs['pn_tidal_order']
    pn_phase_order = waveform_kwargs['pn_phase_order']
    pn_amplitude_order = waveform_kwargs['pn_amplitude_order']
    waveform_dictionary = waveform_kwargs.get(
        'lal_waveform_dictionary', lal.CreateDict()
    )

    approximant = lalsim_GetApproximantFromString(waveform_approximant)

    if pn_amplitude_order != 0:
        start_frequency = lalsim.SimInspiralfLow2fStart(
            minimum_frequency, int(pn_amplitude_order), approximant)
    else:
        start_frequency = minimum_frequency

    delta_frequency = frequency_array[1] - frequency_array[0]

    frequency_bounds = ((frequency_array >= minimum_frequency) *
                        (frequency_array <= maximum_frequency))

    luminosity_distance = luminosity_distance * 1e6 * utils.parsec
    mass_1 = mass_1 * utils.solar_mass
    mass_2 = mass_2 * utils.solar_mass

    iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = bilby_to_lalsimulation_spins(
        theta_jn=theta_jn, phi_jl=phi_jl, tilt_1=tilt_1, tilt_2=tilt_2,
        phi_12=phi_12, a_1=a_1, a_2=a_2, mass_1=mass_1, mass_2=mass_2,
        reference_frequency=reference_frequency, phase=phase)

    longitude_ascending_nodes = 0.0
    mean_per_ano = 0.0

    lalsim.SimInspiralWaveformParamsInsertPNSpinOrder(
        waveform_dictionary, int(pn_spin_order))
    lalsim.SimInspiralWaveformParamsInsertPNTidalOrder(
        waveform_dictionary, int(pn_tidal_order))
    lalsim.SimInspiralWaveformParamsInsertPNPhaseOrder(
        waveform_dictionary, int(pn_phase_order))
    lalsim.SimInspiralWaveformParamsInsertPNAmplitudeOrder(
        waveform_dictionary, int(pn_amplitude_order))
    lalsim_SimInspiralWaveformParamsInsertTidalLambda1(
        waveform_dictionary, lambda_1)
    lalsim_SimInspiralWaveformParamsInsertTidalLambda2(
        waveform_dictionary, lambda_2)

    for key, value in waveform_kwargs.items():
        func = getattr(lalsim, "SimInspiralWaveformParamsInsert" + key, None)
        if func is not None:
            func(waveform_dictionary, value)

    if waveform_kwargs.get('numerical_relativity_file', None) is not None:
        lalsim.SimInspiralWaveformParamsInsertNumRelData(
            waveform_dictionary, waveform_kwargs['numerical_relativity_file'])

    if ('mode_array' in waveform_kwargs) and waveform_kwargs['mode_array'] is not None:
        mode_array = waveform_kwargs['mode_array']
        mode_array_lal = lalsim.SimInspiralCreateModeArray()
        for mode in mode_array:
            lalsim.SimInspiralModeArrayActivateMode(mode_array_lal, mode[0], mode[1])
        lalsim.SimInspiralWaveformParamsInsertModeArray(waveform_dictionary, mode_array_lal)

    if lalsim.SimInspiralImplementedFDApproximants(approximant):
        wf_func = lalsim_SimInspiralChooseFDWaveform
    else:
        wf_func = lalsim_SimInspiralFD
    try:
        hplus, hcross = wf_func(
            mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y,
            spin_2z, luminosity_distance, iota, phase,
            longitude_ascending_nodes, eccentricity, mean_per_ano, delta_frequency,
            start_frequency, maximum_frequency, reference_frequency,
            waveform_dictionary, approximant)
    except Exception as e:
        if not catch_waveform_errors:
            raise
        else:
            EDOM = (e.args[0] == 'Internal function call failed: Input domain error')
            if EDOM:
                failed_parameters = dict(mass_1=mass_1, mass_2=mass_2,
                                         spin_1=(spin_1x, spin_2y, spin_1z),
                                         spin_2=(spin_2x, spin_2y, spin_2z),
                                         luminosity_distance=luminosity_distance,
                                         iota=iota, phase=phase,
                                         eccentricity=eccentricity,
                                         start_frequency=start_frequency)
                logger.warning("Evaluating the waveform failed with error: {}\n".format(e) +
                               "The parameters were {}\n".format(failed_parameters) +
                               "Likelihood will be set to -inf.")
                return None
            else:
                raise

    h_plus = np.zeros_like(frequency_array, dtype=complex)
    h_cross = np.zeros_like(frequency_array, dtype=complex)

    if len(hplus.data.data) > len(frequency_array):
        logger.debug("LALsim waveform longer than bilby's `frequency_array`" +
                     "({} vs {}), ".format(len(hplus.data.data), len(frequency_array)) +
                     "probably because padded with zeros up to the next power of two length." +
                     " Truncating lalsim array.")
        h_plus = hplus.data.data[:len(h_plus)]
        h_cross = hcross.data.data[:len(h_cross)]
    else:
        h_plus[:len(hplus.data.data)] = hplus.data.data
        h_cross[:len(hcross.data.data)] = hcross.data.data

    h_plus *= frequency_bounds
    h_cross *= frequency_bounds

    if wf_func == lalsim_SimInspiralFD:
        dt = 1 / hplus.deltaF + (hplus.epoch.gpsSeconds + hplus.epoch.gpsNanoSeconds * 1e-9)
        time_shift = np.exp(-1j * 2 * np.pi * dt * frequency_array[frequency_bounds])
        h_plus[frequency_bounds] *= time_shift
        h_cross[frequency_bounds] *= time_shift

    return dict(plus=h_plus, cross=h_cross)


def binary_black_hole_roq(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **waveform_arguments):
    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomPv2', reference_frequency=20.0)
    waveform_kwargs.update(waveform_arguments)
    return _base_roq_waveform(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_jl=phi_jl,
        phi_12=phi_12, lambda_1=0.0, lambda_2=0.0, **waveform_kwargs)


def binary_neutron_star_roq(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, phi_jl, lambda_1, lambda_2, theta_jn, phase,
        **waveform_arguments):
    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomD_NRTidal', reference_frequency=20.0)
    waveform_kwargs.update(waveform_arguments)
    return _base_roq_waveform(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_jl=phi_jl,
        phi_12=phi_12, lambda_1=lambda_1, lambda_2=lambda_2, **waveform_kwargs)


def _base_roq_waveform(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, lambda_1, lambda_2, phi_jl, theta_jn, phase,
        **waveform_arguments):
    """
    See https://git.ligo.org/lscsoft/lalsuite/blob/master/lalsimulation/src/LALSimInspiral.c#L1460

    Parameters
    ==========
    frequency_array: np.array
        This input is ignored for the roq source model
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    luminosity_distance: float
        The luminosity distance in megaparsec
    a_1: float
        Dimensionless primary spin magnitude
    tilt_1: float
        Primary tilt angle
    phi_12: float

    a_2: float
        Dimensionless secondary spin magnitude
    tilt_2: float
        Secondary tilt angle
    phi_jl: float

    theta_jn: float
        Orbital inclination
    phase: float
        The phase at coalescence

    Waveform arguments
    ===================
    Non-sampled extra data used in the source model calculation
    frequency_nodes_linear: np.array
    frequency_nodes_quadratic: np.array
    reference_frequency: float
    approximant: str

    Note: for the frequency_nodes_linear and frequency_nodes_quadratic arguments,
    if using data from https://git.ligo.org/lscsoft/ROQ_data, this should be
    loaded as `np.load(filename).T`.

    Returns
    =======
    waveform_polarizations: dict
        Dict containing plus and cross modes evaluated at the linear and
        quadratic frequency nodes.
    """
    from lal import CreateDict
    frequency_nodes_linear = waveform_arguments['frequency_nodes_linear']
    frequency_nodes_quadratic = waveform_arguments['frequency_nodes_quadratic']
    reference_frequency = waveform_arguments['reference_frequency']
    approximant = lalsim_GetApproximantFromString(
        waveform_arguments['waveform_approximant'])

    luminosity_distance = luminosity_distance * 1e6 * utils.parsec
    mass_1 = mass_1 * utils.solar_mass
    mass_2 = mass_2 * utils.solar_mass

    waveform_dictionary = CreateDict()
    lalsim_SimInspiralWaveformParamsInsertTidalLambda1(
        waveform_dictionary, lambda_1)
    lalsim_SimInspiralWaveformParamsInsertTidalLambda2(
        waveform_dictionary, lambda_2)

    iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = bilby_to_lalsimulation_spins(
        theta_jn=theta_jn, phi_jl=phi_jl, tilt_1=tilt_1, tilt_2=tilt_2,
        phi_12=phi_12, a_1=a_1, a_2=a_2, mass_1=mass_1, mass_2=mass_2,
        reference_frequency=reference_frequency, phase=phase)

    h_linear_plus, h_linear_cross = lalsim_SimInspiralChooseFDWaveformSequence(
        phase, mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y,
        spin_2z, reference_frequency, luminosity_distance, iota,
        waveform_dictionary, approximant, frequency_nodes_linear)

    waveform_dictionary = CreateDict()
    lalsim_SimInspiralWaveformParamsInsertTidalLambda1(
        waveform_dictionary, lambda_1)
    lalsim_SimInspiralWaveformParamsInsertTidalLambda2(
        waveform_dictionary, lambda_2)

    h_quadratic_plus, h_quadratic_cross = lalsim_SimInspiralChooseFDWaveformSequence(
        phase, mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y,
        spin_2z, reference_frequency, luminosity_distance, iota,
        waveform_dictionary, approximant, frequency_nodes_quadratic)

    waveform_polarizations = dict()
    waveform_polarizations['linear'] = dict(
        plus=h_linear_plus.data.data, cross=h_linear_cross.data.data)
    waveform_polarizations['quadratic'] = dict(
        plus=h_quadratic_plus.data.data, cross=h_quadratic_cross.data.data)

    return waveform_polarizations


def binary_black_hole_frequency_sequence(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs):
    """ A Binary Black Hole waveform model using lalsimulation. This generates
    a waveform only on specified frequency points. This is useful for
    likelihood requiring waveform values at a subset of all the frequency
    samples. For example, this is used for MBGravitationalWaveTransient.

    Parameters
    ==========
    frequency_array: array_like
        The input is ignored.
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    luminosity_distance: float
        The luminosity distance in megaparsec
    a_1: float
        Dimensionless primary spin magnitude
    tilt_1: float
        Primary tilt angle
    phi_12: float
        Azimuthal angle between the two component spins
    a_2: float
        Dimensionless secondary spin magnitude
    tilt_2: float
        Secondary tilt angle
    phi_jl: float
        Azimuthal angle between the total binary angular momentum and the
        orbital angular momentum
    theta_jn: float
        Angle between the total binary angular momentum and the line of sight
    phase: float
        The phase at coalescence
    kwargs: dict
        Required keyword arguments
        - frequencies:
            ndarray of frequencies at which waveforms are evaluated

        Optional keyword arguments
        - waveform_approximant
        - reference_frequency
        - catch_waveform_errors
        - pn_spin_order
        - pn_tidal_order
        - pn_phase_order
        - pn_amplitude_order
        - mode_array:
          Activate a specific mode array and evaluate the model using those
          modes only.  e.g. waveform_arguments =
          dict(waveform_approximant='IMRPhenomHM', mode_array=[[2,2],[2,-2])
          returns the 22 and 2-2 modes only of IMRPhenomHM.  You can only
          specify modes that are included in that particular model.  e.g.
          waveform_arguments = dict(waveform_approximant='IMRPhenomHM',
          mode_array=[[2,2],[2,-2],[5,5],[5,-5]]) is not allowed because the
          55 modes are not included in this model.  Be aware that some models
          only take positive modes and return the positive and the negative
          mode together, while others need to call both.  e.g.
          waveform_arguments = dict(waveform_approximant='IMRPhenomHM',
          mode_array=[[2,2],[4,-4]]) returns the 22 and 2-2 of IMRPhenomHM.
          However, waveform_arguments =
          dict(waveform_approximant='IMRPhenomXHM', mode_array=[[2,2],[4,-4]])
          returns the 22 and 4-4 of IMRPhenomXHM.

    Returns
    =======
    dict: A dictionary with the plus and cross polarisation strain modes
    """
    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomPv2', reference_frequency=50.0,
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)
    waveform_kwargs.update(kwargs)
    return _base_waveform_frequency_sequence(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_jl=phi_jl,
        phi_12=phi_12, lambda_1=0.0, lambda_2=0.0, **waveform_kwargs)


def binary_neutron_star_frequency_sequence(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, phi_jl, lambda_1, lambda_2, theta_jn, phase,
        **kwargs):
    """ A Binary Neutron Star waveform model using lalsimulation. This generates
    a waveform only on specified frequency points. This is useful for
    likelihood requiring waveform values at a subset of all the frequency
    samples. For example, this is used for MBGravitationalWaveTransient.

    Parameters
    ==========
    frequency_array: array_like
        The input is ignored.
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    luminosity_distance: float
        The luminosity distance in megaparsec
    a_1: float
        Dimensionless primary spin magnitude
    tilt_1: float
        Primary tilt angle
    phi_12: float
        Azimuthal angle between the two component spins
    a_2: float
        Dimensionless secondary spin magnitude
    tilt_2: float
        Secondary tilt angle
    phi_jl: float
        Azimuthal angle between the total binary angular momentum and the
        orbital angular momentum
    lambda_1: float
        Dimensionless tidal deformability of mass_1
    lambda_2: float
        Dimensionless tidal deformability of mass_2
    theta_jn: float
        Angle between the total binary angular momentum and the line of sight
    phase: float
        The phase at coalescence
    kwargs: dict
        Required keyword arguments
        - frequencies:
            ndarray of frequencies at which waveforms are evaluated

        Optional keyword arguments
        - waveform_approximant
        - reference_frequency
        - catch_waveform_errors
        - pn_spin_order
        - pn_tidal_order
        - pn_phase_order
        - pn_amplitude_order
        - mode_array:
          Activate a specific mode array and evaluate the model using those
          modes only.  e.g. waveform_arguments =
          dict(waveform_approximant='IMRPhenomHM', mode_array=[[2,2],[2,-2])
          returns the 22 and 2-2 modes only of IMRPhenomHM.  You can only
          specify modes that are included in that particular model.  e.g.
          waveform_arguments = dict(waveform_approximant='IMRPhenomHM',
          mode_array=[[2,2],[2,-2],[5,5],[5,-5]]) is not allowed because the
          55 modes are not included in this model.  Be aware that some models
          only take positive modes and return the positive and the negative
          mode together, while others need to call both.  e.g.
          waveform_arguments = dict(waveform_approximant='IMRPhenomHM',
          mode_array=[[2,2],[4,-4]]) returns the 22 and 2-2 of IMRPhenomHM.
          However, waveform_arguments =
          dict(waveform_approximant='IMRPhenomXHM', mode_array=[[2,2],[4,-4]])
          returns the 22 and 4-4 of IMRPhenomXHM.

    Returns
    =======
    dict: A dictionary with the plus and cross polarisation strain modes
    """
    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomPv2_NRTidal', reference_frequency=50.0,
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)
    waveform_kwargs.update(kwargs)
    return _base_waveform_frequency_sequence(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_jl=phi_jl,
        phi_12=phi_12, lambda_1=lambda_1, lambda_2=lambda_2, **waveform_kwargs)


def _base_waveform_frequency_sequence(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, lambda_1, lambda_2, phi_jl, theta_jn, phase,
        **waveform_kwargs):
    """ Generate a cbc waveform model on specified frequency samples

    Parameters
    ----------
    frequency_array: np.array
        This input is ignored
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    luminosity_distance: float
        The luminosity distance in megaparsec
    a_1: float
        Dimensionless primary spin magnitude
    tilt_1: float
        Primary tilt angle
    phi_12: float
    a_2: float
        Dimensionless secondary spin magnitude
    tilt_2: float
        Secondary tilt angle
    phi_jl: float
    theta_jn: float
        Orbital inclination
    phase: float
        The phase at coalescence
    waveform_kwargs: dict
        Optional keyword arguments

    Returns
    -------
    waveform_polarizations: dict
        Dict containing plus and cross modes evaluated at the linear and
        quadratic frequency nodes.
    """
    from lal import CreateDict
    import lalsimulation as lalsim

    frequencies = waveform_kwargs['frequencies']
    reference_frequency = waveform_kwargs['reference_frequency']
    approximant = lalsim_GetApproximantFromString(waveform_kwargs['waveform_approximant'])
    catch_waveform_errors = waveform_kwargs['catch_waveform_errors']
    pn_spin_order = waveform_kwargs['pn_spin_order']
    pn_tidal_order = waveform_kwargs['pn_tidal_order']
    pn_phase_order = waveform_kwargs['pn_phase_order']
    pn_amplitude_order = waveform_kwargs['pn_amplitude_order']
    waveform_dictionary = waveform_kwargs.get(
        'lal_waveform_dictionary', CreateDict()
    )

    lalsim.SimInspiralWaveformParamsInsertPNSpinOrder(
        waveform_dictionary, int(pn_spin_order))
    lalsim.SimInspiralWaveformParamsInsertPNTidalOrder(
        waveform_dictionary, int(pn_tidal_order))
    lalsim.SimInspiralWaveformParamsInsertPNPhaseOrder(
        waveform_dictionary, int(pn_phase_order))
    lalsim.SimInspiralWaveformParamsInsertPNAmplitudeOrder(
        waveform_dictionary, int(pn_amplitude_order))
    lalsim_SimInspiralWaveformParamsInsertTidalLambda1(
        waveform_dictionary, lambda_1)
    lalsim_SimInspiralWaveformParamsInsertTidalLambda2(
        waveform_dictionary, lambda_2)

    for key, value in waveform_kwargs.items():
        func = getattr(lalsim, "SimInspiralWaveformParamsInsert" + key, None)
        if func is not None:
            func(waveform_dictionary, value)

    if waveform_kwargs.get('numerical_relativity_file', None) is not None:
        lalsim.SimInspiralWaveformParamsInsertNumRelData(
            waveform_dictionary, waveform_kwargs['numerical_relativity_file'])

    if ('mode_array' in waveform_kwargs) and waveform_kwargs['mode_array'] is not None:
        mode_array = waveform_kwargs['mode_array']
        mode_array_lal = lalsim.SimInspiralCreateModeArray()
        for mode in mode_array:
            lalsim.SimInspiralModeArrayActivateMode(mode_array_lal, mode[0], mode[1])
        lalsim.SimInspiralWaveformParamsInsertModeArray(waveform_dictionary, mode_array_lal)

    luminosity_distance = luminosity_distance * 1e6 * utils.parsec
    mass_1 = mass_1 * utils.solar_mass
    mass_2 = mass_2 * utils.solar_mass

    iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = bilby_to_lalsimulation_spins(
        theta_jn=theta_jn, phi_jl=phi_jl, tilt_1=tilt_1, tilt_2=tilt_2,
        phi_12=phi_12, a_1=a_1, a_2=a_2, mass_1=mass_1, mass_2=mass_2,
        reference_frequency=reference_frequency, phase=phase)

    try:
        h_plus, h_cross = lalsim_SimInspiralChooseFDWaveformSequence(
            phase, mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y,
            spin_2z, reference_frequency, luminosity_distance, iota,
            waveform_dictionary, approximant, frequencies)
    except Exception as e:
        if not catch_waveform_errors:
            raise
        else:
            EDOM = (e.args[0] == 'Internal function call failed: Input domain error')
            if EDOM:
                failed_parameters = dict(mass_1=mass_1, mass_2=mass_2,
                                         spin_1=(spin_1x, spin_2y, spin_1z),
                                         spin_2=(spin_2x, spin_2y, spin_2z),
                                         luminosity_distance=luminosity_distance,
                                         iota=iota, phase=phase)
                logger.warning("Evaluating the waveform failed with error: {}\n".format(e) +
                               "The parameters were {}\n".format(failed_parameters) +
                               "Likelihood will be set to -inf.")
                return None
            else:
                raise

    return dict(plus=h_plus.data.data, cross=h_cross.data.data)


def sinegaussian(frequency_array, hrss, Q, frequency, **kwargs):
    tau = Q / (np.sqrt(2.0) * np.pi * frequency)
    temp = Q / (4.0 * np.sqrt(np.pi) * frequency)
    fm = frequency_array - frequency
    fp = frequency_array + frequency

    h_plus = ((hrss / np.sqrt(temp * (1 + np.exp(-Q**2)))) *
              ((np.sqrt(np.pi) * tau) / 2.0) *
              (np.exp(-fm**2 * np.pi**2 * tau**2) +
              np.exp(-fp**2 * np.pi**2 * tau**2)))

    h_cross = (-1j * (hrss / np.sqrt(temp * (1 - np.exp(-Q**2)))) *
               ((np.sqrt(np.pi) * tau) / 2.0) *
               (np.exp(-fm**2 * np.pi**2 * tau**2) -
               np.exp(-fp**2 * np.pi**2 * tau**2)))

    return{'plus': h_plus, 'cross': h_cross}


def supernova(
        frequency_array, realPCs, imagPCs, file_path, luminosity_distance, **kwargs):
    """ A supernova NR simulation for injections """

    realhplus, imaghplus, realhcross, imaghcross = np.loadtxt(
        file_path, usecols=(0, 1, 2, 3), unpack=True)

    # waveform in file at 10kpc
    scaling = 1e-3 * (10.0 / luminosity_distance)

    h_plus = scaling * (realhplus + 1.0j * imaghplus)
    h_cross = scaling * (realhcross + 1.0j * imaghcross)
    return {'plus': h_plus, 'cross': h_cross}


def supernova_pca_model(
        frequency_array, pc_coeff1, pc_coeff2, pc_coeff3, pc_coeff4, pc_coeff5,
        luminosity_distance, **kwargs):
    """ Supernova signal model """

    realPCs = kwargs['realPCs']
    imagPCs = kwargs['imagPCs']

    pc1 = realPCs[:, 0] + 1.0j * imagPCs[:, 0]
    pc2 = realPCs[:, 1] + 1.0j * imagPCs[:, 1]
    pc3 = realPCs[:, 2] + 1.0j * imagPCs[:, 2]
    pc4 = realPCs[:, 3] + 1.0j * imagPCs[:, 3]
    pc5 = realPCs[:, 4] + 1.0j * imagPCs[:, 5]

    # file at 10kpc
    scaling = 1e-23 * (10.0 / luminosity_distance)

    h_plus = scaling * (pc_coeff1 * pc1 + pc_coeff2 * pc2 + pc_coeff3 * pc3 +
                        pc_coeff4 * pc4 + pc_coeff5 * pc5)
    h_cross = scaling * (pc_coeff1 * pc1 + pc_coeff2 * pc2 + pc_coeff3 * pc3 +
                         pc_coeff4 * pc4 + pc_coeff5 * pc5)

    return {'plus': h_plus, 'cross': h_cross}

precession_only = {
    "tilt_1", "tilt_2", "phi_12", "phi_jl", "chi_1_in_plane", "chi_2_in_plane",
}

spin = {
    "a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl", "chi_1", "chi_2",
    "chi_1_in_plane", "chi_2_in_plane",
}
mass = {
    "chirp_mass", "mass_ratio", "total_mass", "mass_1", "mass_2",
    "symmetric_mass_ratio",
}
primary_spin_and_q = {
    "a_1", "chi_1", "mass_ratio"
}
tidal = {
    "lambda_1", "lambda_2", "lambda_tilde", "delta_lambda_tilde"
}
phase = {
    "phase", "delta_phase",
}
extrinsic = {
    "azimuth", "zenith", "luminosity_distance", "psi", "theta_jn",
    "cos_theta_jn", "geocent_time", "time_jitter", "ra", "dec",
    "H1_time", "L1_time", "V1_time",
}

PARAMETER_SETS = dict(
    spin=spin, mass=mass, phase=phase, extrinsic=extrinsic,
    tidal=tidal, primary_spin_and_q=primary_spin_and_q,
    intrinsic=spin.union(mass).union(phase).union(tidal),
    precession_only=precession_only,
)
