#!/usr/bin/env python
"""
A script to show how to create your own time domain source model.
A simple damped Gaussian signal is defined in the time domain, injected into noise in
two interferometers (LIGO Livingston and Hanford at design sensitivity),
and then recovered.
"""

import tupak
import numpy as np


# define the time-domain model
def time_domain_damped_sinusoid(time, amplitude, damping_time, frequency, phase):
    """
    This example only creates a linearly polarised signal with only plus polarisation.
    """

    plus = amplitude * np.exp(-time / damping_time) * np.sin(2.*np.pi*frequency*time + phase)
    cross = np.zeros(len(time))

    return {'plus': plus, 'cross': cross}


# define parameters to inject.
injection_parameters = dict(
    amplitude=5e-22, damping_time=0.1, frequency=50, phase=0, ra=0, dec=0, psi=0, geocent_time=0.)

duration = 0.5
sampling_frequency = 2048
outdir = 'outdir'
label = 'time_domain_source_model'

# call the waveform_generator to create our waveform model.
waveform = tupak.gw.waveform_generator.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    time_domain_source_model=time_domain_damped_sinusoid)


# inject the signal into three interferometers
interferometers = tupak.gw.detector.InterferometerSet(['H1', 'L1', 'V1'])
interferometers.set_strain_data_from_power_spectral_densities(sampling_frequency=sampling_frequency, duration=duration)
interferometers.inject_signal(parameters=injection_parameters, waveform_generator=waveform)

#  create the priors
prior = injection_parameters.copy()
prior['amplitude'] = tupak.core.prior.Uniform(1e-23, 1e-21, '$h_0$')
prior['damping_time'] = tupak.core.prior.Uniform(0, 1, 'damping time')
prior['frequency'] = tupak.core.prior.Uniform(0, 200, 'frequency')
prior['phase'] = tupak.core.prior.Uniform(-np.pi / 2, np.pi / 2, '$\phi$')


# define likelihood
likelihood = tupak.gw.likelihood.GravitationalWaveTransient(interferometers, waveform)

# launch sampler
result = tupak.core.sampler.run_sampler(likelihood, prior, sampler='dynesty', npoints=1000,
                                        injection_parameters=injection_parameters, outdir=outdir, label=label)
result.plot_corner()
