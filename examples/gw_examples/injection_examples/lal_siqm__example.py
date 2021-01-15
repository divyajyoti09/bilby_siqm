#!/usr/bin/env python
"""
Tutorial to demonstrate running parameter estimation on a binary neutron star
system taking into account tidal deformabilities.

This example estimates the masses using a uniform prior in both component masses
and also estimates the tidal deformabilities using a uniform prior in both
tidal deformabilities
"""

from __future__ import division, print_function

import numpy as np

import bilby

# Specify the output directory and the name of the simulation.
outdir = 'outdir'
label = 'siqm_example'
bilby.core.utils.setup_logger(outdir=outdir, label=label)

# Set up a random seed for result reproducibility.  This is optional!
np.random.seed(88170235)

# We are going to inject a non-BBH  waveform.  We first establish a
# dictionary of parameters that includes all of the different waveform
# parameters, including masses of the two black holes (mass_1, mass_2),
# aligned spins of both black holes (chi_1, chi_2), etc.
# dQuadMon1 =0  and dQuadMon2=0 will give BH signal

injection_parameters = dict(mass_1=20., mass_2=10., a_1=0.7, a_2=0.3, tilt_1=0.5, tilt_2=1.0,phi_12=1.7, phi_jl=0.3, luminosity_distance=200., theta_jn=0.4, psi=2.659,phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108, dQuadMon1=20,,dQuadMon2=0)

# Set the duration and sampling frequency of the data segment that we're going
# to inject the signal into. For the
# TaylorF2 waveform, we cut the signal close to the isco frequency
duration = 4
sampling_frequency = 2 * 1024
start_time = injection_parameters['geocent_time'] + 2 - duration

# Fixed arguments passed into the source model. The analysis starts at 40 Hz.
waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',
                                  reference_frequency=50., minimum_frequency=40.)


# Create the waveform_generator using a LAL SIQM  source function
waveform_generator = bilby.gw.WaveformGenerator(duration=duration, sampling_frequency=sampling_frequency,frequency_domain_source_model=bilby.gw.source.lal_siqm,parameter_conversion=bilby.gw.conversion.convert_to_lal_siqm_parameters,waveform_arguments=waveform_arguments)
# (LIGO-Hanford (H1), LIGO-Livingston (L1), and Virgo (V1)).
# These default to their design sensitivity and start at 40 Hz.

ifos = bilby.gw.detector.InterferometerList(['H1', 'L1'])
ifos.set_strain_data_from_power_spectral_densities(sampling_frequency=sampling_frequency, duration=duration,start_time=injection_parameters['geocent_time'] - 3)
ifos.inject_signal(waveform_generator=waveform_generator,parameters=injection_parameters)

priors = bilby.gw.prior.BBHPriorDict()
priors['geocent_time'] = bilby.core.prior.Uniform(
            minimum=injection_parameters['geocent_time'] - 1,
                maximum=injection_parameters['geocent_time'] + 1,
                    name='geocent_time', latex_label='$t_c$', unit='$s$')
for key in ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'psi', 'ra','dec', 'geocent_time', 'phase']:
    priors[key] = injection_parameters[key]
priors['dQuadMonS'] = bilby.core.prior.Uniform(0, 500, name='dQuadMonS')
priors['dQuadMonA'] = 0
likelihood = bilby.gw.GravitationalWaveTransient(interferometers=ifos, waveform_generator=waveform_generator)

result = bilby.run_sampler(likelihood=likelihood, priors=priors, sampler='dynesty', npoints=1000,injection_parameters=injection_parameters, outdir=outdir, label=label)

# Make a corner plot.
result.plot_corner()
