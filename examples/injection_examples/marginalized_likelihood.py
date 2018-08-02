#!/usr/bin/env python
"""
Tutorial to demonstrate how to improve the speed and efficiency of parameter estimation on an injected signal using
phase and distance marginalisation.
"""
from __future__ import division, print_function
import tupak
import numpy as np


duration = 4.
sampling_frequency = 2048.
outdir = 'outdir'

np.random.seed(170608)

injection_parameters = dict(
    mass_1=36., mass_2=29., a_1=0.4, a_2=0.3, tilt_1=0.5, tilt_2=1.0, phi_12=1.7, phi_jl=0.3, iota=0.4,
    luminosity_distance=4000., psi=2.659, phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108)

waveform_arguments = dict(waveform_approximant='IMRPhenomPv2', reference_frequency=50.)

# Create the waveform_generator using a LAL BinaryBlackHole source function
waveform_generator = tupak.gw.waveform_generator.BinaryBlackHole(
    duration=duration, sampling_frequency=sampling_frequency, waveform_arguments=waveform_arguments)

# Set up interferometers.
interferometers = tupak.gw.detector.InterferometerSet(['H1', 'L1', 'V1'])
interferometers.set_strain_data_from_power_spectral_densities(sampling_frequency=sampling_frequency, duration=duration)
interferometers.inject_signal(parameters=injection_parameters, waveform_generator=waveform_generator)

# Set up prior
priors = tupak.gw.prior.BBHPriorSet()
# These parameters will not be sampled
for key in ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'iota', 'ra', 'dec', 'geocent_time']:
    priors[key] = injection_parameters[key]

# Initialise GravitationalWaveTransient
# Note that we now need to pass the: priors and flags for each thing that's being marginalised.
# This is still under development so care should be taken with the marginalised likelihood.
likelihood = tupak.gw.GravitationalWaveTransient(
    interferometers=IFOs, waveform_generator=waveform_generator, prior=priors,
    distance_marginalization=False, phase_marginalization=True,
    time_marginalization=False)

# Run sampler
result = tupak.run_sampler(likelihood=likelihood, priors=priors, sampler='dynesty',
                           injection_parameters=injection_parameters, outdir=outdir, label='MarginalisedLikelihood')
result.plot_corner()

