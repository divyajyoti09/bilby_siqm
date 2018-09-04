from __future__ import absolute_import

import tupak
import unittest
import mock
from mock import MagicMock
from mock import patch
import numpy as np
import scipy.signal.windows
import gwpy


class TestDetector(unittest.TestCase):

    def setUp(self):
        self.name = 'name'
        self.power_spectral_density = MagicMock()
        self.minimum_frequency = 10
        self.maximum_frequency = 20
        self.length = 30
        self.latitude = 1
        self.longitude = 2
        self.elevation = 3
        self.xarm_azimuth = 4
        self.yarm_azimuth = 5
        self.xarm_tilt = 0.
        self.yarm_tilt = 0.
        # noinspection PyTypeChecker
        self.ifo = tupak.gw.detector.Interferometer(name=self.name, power_spectral_density=self.power_spectral_density,
                                                    minimum_frequency=self.minimum_frequency,
                                                    maximum_frequency=self.maximum_frequency, length=self.length,
                                                    latitude=self.latitude, longitude=self.longitude,
                                                    elevation=self.elevation,
                                                    xarm_azimuth=self.xarm_azimuth, yarm_azimuth=self.yarm_azimuth,
                                                    xarm_tilt=self.xarm_tilt, yarm_tilt=self.yarm_tilt)
        self.ifo.strain_data.set_from_frequency_domain_strain(
            np.linspace(0, 4096, 4097), sampling_frequency=4096, duration=2)

    def tearDown(self):
        del self.name
        del self.power_spectral_density
        del self.minimum_frequency
        del self.maximum_frequency
        del self.length
        del self.latitude
        del self.longitude
        del self.elevation
        del self.xarm_azimuth
        del self.yarm_azimuth
        del self.xarm_tilt
        del self.yarm_tilt
        del self.ifo

    def test_name_setting(self):
        self.assertEqual(self.ifo.name, self.name)

    def test_psd_setting(self):
        self.assertEqual(self.ifo.power_spectral_density, self.power_spectral_density)

    def test_min_freq_setting(self):
        self.assertEqual(self.ifo.strain_data.minimum_frequency, self.minimum_frequency)

    def test_max_freq_setting(self):
        self.assertEqual(self.ifo.strain_data.maximum_frequency, self.maximum_frequency)

    def test_length_setting(self):
        self.assertEqual(self.ifo.length, self.length)

    def test_latitude_setting(self):
        self.assertEqual(self.ifo.latitude, self.latitude)

    def test_longitude_setting(self):
        self.assertEqual(self.ifo.longitude, self.longitude)

    def test_elevation_setting(self):
        self.assertEqual(self.ifo.elevation, self.elevation)

    def test_xarm_azi_setting(self):
        self.assertEqual(self.ifo.xarm_azimuth, self.xarm_azimuth)

    def test_yarm_azi_setting(self):
        self.assertEqual(self.ifo.yarm_azimuth, self.yarm_azimuth)

    def test_xarm_tilt_setting(self):
        self.assertEqual(self.ifo.xarm_tilt, self.xarm_tilt)

    def test_yarm_tilt_setting(self):
        self.assertEqual(self.ifo.yarm_tilt, self.yarm_tilt)

    def test_vertex_without_update(self):
        _ = self.ifo.vertex
        with mock.patch('tupak.gw.utils.get_vertex_position_geocentric') as m:
            m.return_value = np.array([1])
            self.assertFalse(np.array_equal(self.ifo.vertex, np.array([1])))

    def test_vertex_with_latitude_update(self):
        with mock.patch('tupak.gw.utils.get_vertex_position_geocentric') as m:
            m.return_value = np.array([1])
            self.ifo.latitude = 5
            self.assertEqual(self.ifo.vertex, np.array([1]))

    def test_vertex_with_longitude_update(self):
        with mock.patch('tupak.gw.utils.get_vertex_position_geocentric') as m:
            m.return_value = np.array([1])
            self.ifo.longitude = 5
            self.assertEqual(self.ifo.vertex, np.array([1]))

    def test_vertex_with_elevation_update(self):
        with mock.patch('tupak.gw.utils.get_vertex_position_geocentric') as m:
            m.return_value = np.array([1])
            self.ifo.elevation = 5
            self.assertEqual(self.ifo.vertex, np.array([1]))

    def test_x_without_update(self):
        _ = self.ifo.x
        self.ifo.unit_vector_along_arm = MagicMock(return_value=np.array([1]))

        self.assertFalse(np.array_equal(self.ifo.x,
                                        np.array([1])))

    def test_x_with_xarm_tilt_update(self):
        self.ifo.unit_vector_along_arm = MagicMock(return_value=np.array([1]))
        self.ifo.xarm_tilt = 0
        self.assertTrue(np.array_equal(self.ifo.x,
                                       np.array([1])))

    def test_x_with_xarm_azimuth_update(self):
        self.ifo.unit_vector_along_arm = MagicMock(return_value=np.array([1]))
        self.ifo.xarm_azimuth = 0
        self.assertTrue(np.array_equal(self.ifo.x,
                                       np.array([1])))

    def test_x_with_longitude_update(self):
        self.ifo.unit_vector_along_arm = MagicMock(return_value=np.array([1]))
        self.ifo.longitude = 0
        self.assertTrue(np.array_equal(self.ifo.x,
                                       np.array([1])))

    def test_x_with_latitude_update(self):
        self.ifo.unit_vector_along_arm = MagicMock(return_value=np.array([1]))
        self.ifo.latitude = 0
        self.assertTrue(np.array_equal(self.ifo.x,
                                       np.array([1])))

    def test_y_without_update(self):
        _ = self.ifo.y
        self.ifo.unit_vector_along_arm = MagicMock(return_value=np.array([1]))

        self.assertFalse(np.array_equal(self.ifo.y,
                                        np.array([1])))

    def test_y_with_yarm_tilt_update(self):
        self.ifo.unit_vector_along_arm = MagicMock(return_value=np.array([1]))
        self.ifo.yarm_tilt = 0
        self.assertTrue(np.array_equal(self.ifo.y,
                                       np.array([1])))

    def test_y_with_yarm_azimuth_update(self):
        self.ifo.unit_vector_along_arm = MagicMock(return_value=np.array([1]))
        self.ifo.yarm_azimuth = 0
        self.assertTrue(np.array_equal(self.ifo.y,
                                       np.array([1])))

    def test_y_with_longitude_update(self):
        self.ifo.unit_vector_along_arm = MagicMock(return_value=np.array([1]))
        self.ifo.longitude = 0
        self.assertTrue(np.array_equal(self.ifo.y,
                                       np.array([1])))

    def test_y_with_latitude_update(self):
        self.ifo.unit_vector_along_arm = MagicMock(return_value=np.array([1]))
        self.ifo.latitude = 0
        self.assertTrue(np.array_equal(self.ifo.y,
                                       np.array([1])))

    def test_detector_tensor_without_update(self):
        _ = self.ifo.detector_tensor
        with mock.patch('numpy.einsum') as m:
            m.return_value = 1
            self.assertIsInstance(self.ifo.detector_tensor, np.ndarray)

    def test_detector_tensor_with_x_update(self):
        with mock.patch('numpy.einsum') as m:
            m.return_value = 1
            self.ifo.xarm_azimuth = 12
            self.assertEqual(self.ifo.detector_tensor, 0)

    def test_detector_tensor_with_y_update(self):
        with mock.patch('numpy.einsum') as m:
            m.return_value = 1
            self.ifo.yarm_azimuth = 12
            self.assertEqual(self.ifo.detector_tensor, 0)

    def test_amplitude_spectral_density_array(self):
        self.ifo.power_spectral_density.power_spectral_density_interpolated = MagicMock(return_value=np.array([1, 4]))
        self.assertTrue(np.array_equal(self.ifo.amplitude_spectral_density_array, np.array([1, 2])))

    def test_power_spectral_density_array(self):
        self.ifo.power_spectral_density.power_spectral_density_interpolated = MagicMock(return_value=np.array([1, 4]))
        self.assertTrue(np.array_equal(self.ifo.power_spectral_density_array, np.array([1, 4])))

    def test_antenna_response_default(self):
        with mock.patch('tupak.gw.utils.get_polarization_tensor') as m:
            with mock.patch('numpy.einsum') as n:
                m.return_value = 0
                n.return_value = 1
                self.assertEqual(self.ifo.antenna_response(234, 52, 54, 76, 'plus'), 1)

    def test_antenna_response_einsum(self):
        with mock.patch('tupak.gw.utils.get_polarization_tensor') as m:
            m.return_value = np.ones((3, 3))
            self.assertAlmostEqual(self.ifo.antenna_response(234, 52, 54, 76, 'plus'), self.ifo.detector_tensor.sum())

    def test_get_detector_response_default_behaviour(self):
        self.ifo.antenna_response = MagicMock(return_value=1)
        self.ifo.time_delay_from_geocenter = MagicMock(return_value=0)
        self.ifo.epoch = 0
        self.minimum_frequency = 10
        self.maximum_frequency = 20
        # self.ifo.frequency_array = np.array([8, 12, 16, 20, 24])
        plus = np.linspace(0, 4096, 4097)
        response = self.ifo.get_detector_response(
            waveform_polarizations=dict(plus=plus),
            parameters=dict(ra=0, dec=0, geocent_time=0, psi=0))
        self.assertTrue(np.array_equal(response, plus * self.ifo.frequency_mask * np.exp(-0j)))

    def test_get_detector_response_with_dt(self):
        self.ifo.antenna_response = MagicMock(return_value=1)
        self.ifo.time_delay_from_geocenter = MagicMock(return_value=0)
        self.ifo.epoch = 1
        self.minimum_frequency = 10
        self.maximum_frequency = 20
        # self.ifo.frequency_array = np.array([8, 12, 16, 20, 24])
        plus = np.linspace(0, 4096, 4097)
        response = self.ifo.get_detector_response(
            waveform_polarizations=dict(plus=plus),
            parameters=dict(ra=0, dec=0, geocent_time=0, psi=0))
        expected_response = plus * self.ifo.frequency_mask * np.exp(-1j * 2 * np.pi * self.ifo.frequency_array)
        self.assertTrue(np.allclose(abs(response),
                                    abs(plus * self.ifo.frequency_mask * np.exp(
                                        -1j * 2 * np.pi * self.ifo.frequency_array))))

    def test_get_detector_response_multiple_modes(self):
        self.ifo.antenna_response = MagicMock(return_value=1)
        self.ifo.time_delay_from_geocenter = MagicMock(return_value=0)
        self.ifo.epoch = 0
        self.minimum_frequency = 10
        self.maximum_frequency = 20
        # self.ifo.frequency_array = np.array([8, 12, 16, 20, 24])
        plus = np.linspace(0, 4096, 4097)
        cross = np.linspace(0, 4096, 4097)
        response = self.ifo.get_detector_response(
            waveform_polarizations=dict(plus=plus, cross=cross),
            parameters=dict(ra=0, dec=0, geocent_time=0, psi=0))
        self.assertTrue(np.array_equal(response, (plus + cross) * self.ifo.frequency_mask * np.exp(-0j)))

    def test_inject_signal_no_waveform_polarizations(self):
        with self.assertRaises(ValueError):
            self.ifo.inject_signal(injection_polarizations=None, parameters=None)

    def test_unit_vector_along_arm_default(self):
        with self.assertRaises(ValueError):
            self.ifo.unit_vector_along_arm('z')

    def test_unit_vector_along_arm_x(self):
        with mock.patch('numpy.array') as m:
            m.return_value = 1
            self.ifo.xarm_tilt = 0
            self.ifo.xarm_azimuth = 0
            self.ifo.yarm_tilt = 0
            self.ifo.yarm_azimuth = 90
            self.assertAlmostEqual(self.ifo.unit_vector_along_arm('x'), 1)

    def test_unit_vector_along_arm_y(self):
        with mock.patch('numpy.array') as m:
            m.return_value = 1
            self.ifo.xarm_tilt = 0
            self.ifo.xarm_azimuth = 90
            self.ifo.yarm_tilt = 0
            self.ifo.yarm_azimuth = 180
            self.assertAlmostEqual(self.ifo.unit_vector_along_arm('y'), -1)

    def test_time_delay_from_geocenter(self):
        with mock.patch('tupak.gw.utils.time_delay_geocentric') as m:
            m.return_value = 1
            self.assertEqual(self.ifo.time_delay_from_geocenter(1, 2, 3), 1)

    def test_vertex_position_geocentric(self):
        with mock.patch('tupak.gw.utils.get_vertex_position_geocentric') as m:
            m.return_value = 1
            self.assertEqual(self.ifo.vertex_position_geocentric(), 1)

    def test_optimal_snr_squared(self):
        """ Merely checks parameters are given in the right order """
        with mock.patch('tupak.gw.utils.noise_weighted_inner_product') as m:
            m.side_effect = lambda a, b, c, d: [a, b, c, d]
            signal = 1
            expected = [signal, signal, self.ifo.power_spectral_density_array, self.ifo.strain_data.duration]
            actual = self.ifo.optimal_snr_squared(signal=signal)
            self.assertListEqual(expected, actual)

    def test_matched_filter_snr_squared(self):
        """ Merely checks parameters are given in the right order """
        with mock.patch('tupak.gw.utils.noise_weighted_inner_product') as m:
            m.side_effect = lambda a, b, c, d: [b, [a, c, d]]
            signal = 1
            expected = [self.ifo.frequency_domain_strain, [signal, self.ifo.power_spectral_density_array,
                                                           self.ifo.strain_data.duration]]
            actual = self.ifo.matched_filter_snr_squared(signal=signal)
            self.assertTrue(np.array_equal(expected[0], actual[0]))  # array-like element has to be evaluated separately
            self.assertListEqual(expected[1], actual[1])


class TestInterferometerStrainData(unittest.TestCase):

    def setUp(self):
        self.minimum_frequency = 10
        self.maximum_frequency = 20
        self.ifosd = tupak.gw.detector.InterferometerStrainData(
            minimum_frequency=self.minimum_frequency,
            maximum_frequency=self.maximum_frequency)

    def tearDown(self):
        del self.minimum_frequency
        del self.maximum_frequency
        del self.ifosd

    def test_frequency_mask(self):
        self.minimum_frequency = 10
        self.maximum_frequency = 20
        with mock.patch('tupak.core.utils.create_frequency_series') as m:
            m.return_value = np.array([5, 15, 25])
            self.ifosd.set_from_frequency_domain_strain(
                frequency_domain_strain=np.array([0, 1, 2]), frequency_array=np.array([5, 15, 25]))
            self.assertTrue(np.array_equal(self.ifosd.frequency_mask, [False, True, False]))

    def test_frequency_array_setting_direct(self):
        with mock.patch('tupak.core.utils.create_frequency_series') as m:
            m.return_value = np.array([5, 15, 25])
            self.ifosd.set_from_frequency_domain_strain(
                frequency_domain_strain=np.array([0, 1, 2]), frequency_array=np.array([5, 15, 25]))
            self.assertTrue(np.array_equal(self.ifosd.frequency_array, np.array(np.array([5, 15, 25]))))

    def test_duration_setting(self):
        with mock.patch('tupak.core.utils.create_frequency_series') as m:
            m.return_value = np.array([0, 1, 2])
            self.ifosd.set_from_frequency_domain_strain(
                frequency_domain_strain=np.array([0, 1, 2]), frequency_array=np.array([0, 1, 2]))
            self.assertAlmostEqual(self.ifosd.duration, 1)

    def test_sampling_frequency_setting(self):
        with mock.patch('tupak.core.utils.create_frequency_series') as n:
            with mock.patch('tupak.core.utils.get_sampling_frequency_and_duration_from_frequency_array') as m:
                m.return_value = 8, 456
                n.return_value = np.array([1, 2, 3])
                self.ifosd.set_from_frequency_domain_strain(
                    frequency_domain_strain=np.array([0, 1, 2]), frequency_array=np.array([0, 1, 2]))
                self.assertEqual(8, self.ifosd.sampling_frequency)

    def test_frequency_array_setting(self):
        duration = 3
        sampling_frequency = 1
        with mock.patch('tupak.core.utils.create_frequency_series') as m:
            m.return_value = [1, 2, 3]
            self.ifosd.set_from_frequency_domain_strain(
                frequency_domain_strain=np.array([0, 1, 2]), duration=duration,
                sampling_frequency=sampling_frequency)
            self.assertTrue(np.array_equal(
                self.ifosd.frequency_array,
                tupak.core.utils.create_frequency_series(duration=duration,
                                                         sampling_frequency=sampling_frequency)))

    def test_set_data_fails(self):
        with mock.patch('tupak.core.utils.create_frequency_series') as m:
            m.return_value = [1, 2, 3]
            with self.assertRaises(ValueError):
                self.ifosd.set_from_frequency_domain_strain(
                    frequency_domain_strain=np.array([0, 1, 2]))

    def test_set_data_fails_too_much(self):
        with mock.patch('tupak.core.utils.create_frequency_series') as m:
            m.return_value = [1, 2, 3]
            with self.assertRaises(ValueError):
                self.ifosd.set_from_frequency_domain_strain(
                    frequency_domain_strain=np.array([0, 1, 2]), frequency_array=np.array([1, 2, 3]),
                    duration=3, sampling_frequency=1)

    def test_start_time_init(self):
        with mock.patch('tupak.core.utils.create_frequency_series') as m:
            m.return_value = [1, 2, 3]
            duration = 3
            sampling_frequency = 1
            self.ifosd.set_from_frequency_domain_strain(
                frequency_domain_strain=np.array([0, 1, 2]), duration=duration,
                sampling_frequency=sampling_frequency)
            self.assertTrue(self.ifosd.start_time == 0)

    def test_start_time_set(self):
        with mock.patch('tupak.core.utils.create_frequency_series') as m:
            m.return_value = [1, 2, 3]
            duration = 3
            sampling_frequency = 1
            self.ifosd.set_from_frequency_domain_strain(
                frequency_domain_strain=np.array([0, 1, 2]), duration=duration,
                sampling_frequency=sampling_frequency, start_time=10)
            self.assertTrue(self.ifosd.start_time == 10)

    def test_time_array_frequency_array_consistency(self):
        duration = 1
        sampling_frequency = 10
        time_array = tupak.core.utils.create_time_series(
            sampling_frequency=sampling_frequency, duration=duration)
        time_domain_strain = np.random.normal(0, 1, len(time_array))
        self.ifosd.roll_off = 0
        self.ifosd.set_from_time_domain_strain(
            time_domain_strain=time_domain_strain, duration=duration,
            sampling_frequency=sampling_frequency)

        frequency_domain_strain, freqs = tupak.core.utils.nfft(
            time_domain_strain, sampling_frequency)

        self.assertTrue(np.all(
            self.ifosd.frequency_domain_strain == frequency_domain_strain * self.ifosd.frequency_mask))

    def test_time_array_when_set(self):
        test_array = np.array([1])
        self.ifosd.time_array = test_array
        self.assertTrue(test_array, self.ifosd.time_array)

    @patch.object(tupak.core.utils, 'create_time_series')
    def test_time_array_when_not_set(self, m):
        self.ifosd.start_time = 3
        self.ifosd.sampling_frequency = 1000
        self.ifosd.duration = 5
        m.return_value = 4
        self.assertEqual(m.return_value, self.ifosd.time_array)
        m.assert_called_with(sampling_frequency=self.ifosd.sampling_frequency,
                             duration=self.ifosd.duration,
                             starting_time=self.ifosd.start_time)

    def test_time_array_without_sampling_frequency(self):
        self.ifosd.sampling_frequency = None
        self.ifosd.duration = 4
        with self.assertRaises(ValueError):
            test = self.ifosd.time_array

    def test_time_array_without_duration(self):
        self.ifosd.sampling_frequency = 4096
        self.ifosd.duration = None
        with self.assertRaises(ValueError):
            test = self.ifosd.time_array

    def test_frequency_array_when_set(self):
        test_array = np.array([1])
        self.ifosd.frequency_array = test_array
        self.assertTrue(test_array, self.ifosd.frequency_array)

    @patch.object(tupak.core.utils, 'create_frequency_series')
    def test_time_array_when_not_set(self, m):
        self.ifosd.sampling_frequency = 1000
        self.ifosd.duration = 5
        m.return_value = 4
        self.assertEqual(m.return_value, self.ifosd.frequency_array)
        m.assert_called_with(sampling_frequency=self.ifosd.sampling_frequency,
                             duration=self.ifosd.duration)

    def test_frequency_array_without_sampling_frequency(self):
        self.ifosd.sampling_frequency = None
        self.ifosd.duration = 4
        with self.assertRaises(ValueError):
            test = self.ifosd.frequency_array

    def test_frequency_array_without_duration(self):
        self.ifosd.sampling_frequency = 4096
        self.ifosd.duration = None
        with self.assertRaises(ValueError):
            test = self.ifosd.frequency_array

    def test_time_within_data_before(self):
        self.ifosd.start_time = 3
        self.ifosd.duration = 2
        self.assertFalse(self.ifosd.time_within_data(2))

    def test_time_within_data_during(self):
        self.ifosd.start_time = 3
        self.ifosd.duration = 2
        self.assertTrue(self.ifosd.time_within_data(3))
        self.assertTrue(self.ifosd.time_within_data(4))
        self.assertTrue(self.ifosd.time_within_data(5))

    def test_time_within_data_after(self):
        self.ifosd.start_time = 3
        self.ifosd.duration = 2
        self.assertFalse(self.ifosd.time_within_data(6))

    def test_time_domain_window_no_roll_off_no_alpha(self):
        self.ifosd._time_domain_strain = np.array([3])
        self.ifosd.duration = 5
        self.ifosd.roll_off = 2
        expected_window = scipy.signal.windows.tukey(len(self.ifosd._time_domain_strain), alpha=self.ifosd.alpha)
        self.assertEqual(expected_window,
                         self.ifosd.time_domain_window())
        self.assertEqual(np.mean(expected_window ** 2), self.ifosd.window_factor)

    def test_time_domain_window_sets_roll_off_directly(self):
        self.ifosd._time_domain_strain = np.array([3])
        self.ifosd.duration = 5
        self.ifosd.roll_off = 2
        expected_roll_off = 6
        self.ifosd.time_domain_window(roll_off=expected_roll_off)
        self.assertEqual(expected_roll_off, self.ifosd.roll_off)

    def test_time_domain_window_sets_roll_off_indirectly(self):
        self.ifosd._time_domain_strain = np.array([3])
        self.ifosd.duration = 5
        self.ifosd.roll_off = 2
        alpha = 4
        expected_roll_off = alpha * self.ifosd.duration / 2
        self.ifosd.time_domain_window(alpha=alpha)
        self.assertEqual(expected_roll_off, self.ifosd.roll_off)

    def test_time_domain_strain_when_set(self):
        expected_strain = 5
        self.ifosd._time_domain_strain = expected_strain
        self.assertEqual(expected_strain, self.ifosd.time_domain_strain)

    @patch('tupak.core.utils.infft')
    def test_time_domain_strain_from_frequency_domain_strain(self, m):
        m.return_value = 5
        self.ifosd.sampling_frequency = 200
        self.ifosd.duration = 4
        self.ifosd._frequency_domain_strain = self.ifosd.frequency_array
        self.ifosd.sampling_frequency = 123
        self.assertEqual(m.return_value, self.ifosd.time_domain_strain)

    def test_time_domain_strain_not_set(self):
        self.ifosd._time_domain_strain = None
        self.ifosd._frequency_domain_strain = None
        with self.assertRaises(ValueError):
            test = self.ifosd.time_domain_strain

    def test_frequency_domain_strain_when_set(self):
        self.ifosd.sampling_frequency = 200
        self.ifosd.duration = 4
        expected_strain = self.ifosd.frequency_array*self.ifosd.frequency_mask
        self.ifosd._frequency_domain_strain = expected_strain
        self.assertTrue(np.array_equal(expected_strain,
                                        self.ifosd.frequency_domain_strain))

    @patch('tupak.core.utils.nfft')
    def test_frequency_domain_strain_from_frequency_domain_strain(self, m):
        self.ifosd.start_time = 0
        self.ifosd.duration = 4
        self.ifosd.sampling_frequency = 200
        m.return_value = self.ifosd.frequency_array, self.ifosd.frequency_array
        self.ifosd._time_domain_strain = self.ifosd.time_array
        self.assertTrue(np.array_equal(self.ifosd.frequency_array * self.ifosd.frequency_mask,
                                       self.ifosd.frequency_domain_strain))

    def test_frequency_domain_strain_not_set(self):
        self.ifosd._time_domain_strain = None
        self.ifosd._frequency_domain_strain = None
        with self.assertRaises(ValueError):
            test = self.ifosd.frequency_domain_strain

    def test_set_frequency_domain_strain(self):
        self.ifosd.duration = 4
        self.ifosd.sampling_frequency = 200
        self.ifosd.frequency_domain_strain = np.ones(len(self.ifosd.frequency_array))
        self.assertTrue(np.array_equal(np.ones(len(self.ifosd.frequency_array)),
                                       self.ifosd._frequency_domain_strain))

    def test_set_frequency_domain_strain_wrong_length(self):
        self.ifosd.duration = 4
        self.ifosd.sampling_frequency = 200
        with self.assertRaises(ValueError):
            self.ifosd.frequency_domain_strain = np.array([1])


class TestInterferometerList(unittest.TestCase):

    def setUp(self):
        self.frequency_arrays = np.linspace(0, 4096, 4097)
        self.name1 = 'name1'
        self.name2 = 'name2'
        self.power_spectral_density1 = MagicMock()
        self.power_spectral_density1.get_noise_realisation = MagicMock(return_value=(self.frequency_arrays,
                                                                                     self.frequency_arrays))
        self.power_spectral_density2 = MagicMock()
        self.power_spectral_density2.get_noise_realisation = MagicMock(return_value=(self.frequency_arrays,
                                                                                     self.frequency_arrays))
        self.minimum_frequency1 = 10
        self.minimum_frequency2 = 10
        self.maximum_frequency1 = 20
        self.maximum_frequency2 = 20
        self.length1 = 30
        self.length2 = 30
        self.latitude1 = 1
        self.latitude2 = 1
        self.longitude1 = 2
        self.longitude2 = 2
        self.elevation1 = 3
        self.elevation2 = 3
        self.xarm_azimuth1 = 4
        self.xarm_azimuth2 = 4
        self.yarm_azimuth1 = 5
        self.yarm_azimuth2 = 5
        self.xarm_tilt1 = 0.
        self.xarm_tilt2 = 0.
        self.yarm_tilt1 = 0.
        self.yarm_tilt2 = 0.
        # noinspection PyTypeChecker
        self.ifo1 = tupak.gw.detector.Interferometer(name=self.name1,
                                                     power_spectral_density=self.power_spectral_density1,
                                                     minimum_frequency=self.minimum_frequency1,
                                                     maximum_frequency=self.maximum_frequency1, length=self.length1,
                                                     latitude=self.latitude1, longitude=self.longitude1,
                                                     elevation=self.elevation1,
                                                     xarm_azimuth=self.xarm_azimuth1, yarm_azimuth=self.yarm_azimuth1,
                                                     xarm_tilt=self.xarm_tilt1, yarm_tilt=self.yarm_tilt1)
        self.ifo2 = tupak.gw.detector.Interferometer(name=self.name2,
                                                     power_spectral_density=self.power_spectral_density2,
                                                     minimum_frequency=self.minimum_frequency2,
                                                     maximum_frequency=self.maximum_frequency2, length=self.length2,
                                                     latitude=self.latitude2, longitude=self.longitude2,
                                                     elevation=self.elevation2,
                                                     xarm_azimuth=self.xarm_azimuth2, yarm_azimuth=self.yarm_azimuth2,
                                                     xarm_tilt=self.xarm_tilt2, yarm_tilt=self.yarm_tilt2)
        self.ifo1.strain_data.set_from_frequency_domain_strain(
            self.frequency_arrays, sampling_frequency=4096, duration=2)
        self.ifo2.strain_data.set_from_frequency_domain_strain(
            self.frequency_arrays, sampling_frequency=4096, duration=2)
        self.ifo_list = tupak.gw.detector.InterferometerList([self.ifo1, self.ifo2])

    def tearDown(self):
        del self.frequency_arrays
        del self.name1
        del self.name2
        del self.power_spectral_density1
        del self.power_spectral_density2
        del self.minimum_frequency1
        del self.minimum_frequency2
        del self.maximum_frequency1
        del self.maximum_frequency2
        del self.length1
        del self.length2
        del self.latitude1
        del self.latitude2
        del self.longitude1
        del self.longitude2
        del self.elevation1
        del self.elevation2
        del self.xarm_azimuth1
        del self.xarm_azimuth2
        del self.yarm_azimuth1
        del self.yarm_azimuth2
        del self.xarm_tilt1
        del self.xarm_tilt2
        del self.yarm_tilt1
        del self.yarm_tilt2
        del self.ifo1
        del self.ifo2
        del self.ifo_list

    def test_init_with_string(self):
        with self.assertRaises(ValueError):
            tupak.gw.detector.InterferometerList("string")

    def test_init_with_string_list(self):
        """ Merely checks if this ends up in the right bracket """
        with mock.patch('tupak.gw.detector.get_empty_interferometer') as m:
            m.side_effect = ValueError
            with self.assertRaises(ValueError):
                tupak.gw.detector.InterferometerList(['string'])

    def test_init_with_other_object(self):
        with self.assertRaises(ValueError):
            tupak.gw.detector.InterferometerList([object()])

    def test_init_with_actual_ifos(self):
        ifo_list = tupak.gw.detector.InterferometerList([self.ifo1, self.ifo2])
        self.assertEqual(self.ifo1, ifo_list[0])
        self.assertEqual(self.ifo2, ifo_list[1])

    def test_init_inconsistent_duration(self):
        self.ifo2.strain_data.set_from_frequency_domain_strain(
            np.linspace(0, 4096, 4097), sampling_frequency=4096, duration=3)
        with self.assertRaises(ValueError):
            tupak.gw.detector.InterferometerList([self.ifo1, self.ifo2])

    def test_init_inconsistent_sampling_frequency(self):
        self.ifo2.strain_data.set_from_frequency_domain_strain(
            np.linspace(0, 4096, 4097), sampling_frequency=234, duration=2)
        with self.assertRaises(ValueError):
            tupak.gw.detector.InterferometerList([self.ifo1, self.ifo2])

    def test_init_inconsistent_start_time(self):
        self.ifo2.strain_data.start_time = 1
        with self.assertRaises(ValueError):
            tupak.gw.detector.InterferometerList([self.ifo1, self.ifo2])

    @patch.object(tupak.gw.detector.Interferometer, 'set_strain_data_from_power_spectral_density')
    def test_set_strain_data_from_power_spectral_density(self, m):
        self.ifo_list.set_strain_data_from_power_spectral_densities(sampling_frequency=123, duration=6.2, start_time=3)
        m.assert_called_with(sampling_frequency=123, duration=6.2, start_time=3)
        self.assertEqual(len(self.ifo_list), m.call_count)

    def test_inject_signal_pol_and_wg_none(self):
        with self.assertRaises(ValueError):
            self.ifo_list.inject_signal(injection_polarizations=None, waveform_generator=None)

    @patch.object(tupak.gw.waveform_generator.WaveformGenerator, 'frequency_domain_strain')
    def test_inject_signal_pol_none_calls_frequency_domain_strain(self, m):
        waveform_generator = tupak.gw.waveform_generator.WaveformGenerator(
            frequency_domain_source_model=lambda x, y, z: x)
        self.ifo1.inject_signal = MagicMock(return_value=None)
        self.ifo2.inject_signal = MagicMock(return_value=None)
        self.ifo_list.inject_signal(parameters=None, waveform_generator=waveform_generator)
        self.assertTrue(m.called)

    @patch.object(tupak.gw.waveform_generator.WaveformGenerator, 'frequency_domain_strain')
    def test_inject_signal_pol_none_sets_wg_parameters(self, m):
        waveform_generator = tupak.gw.waveform_generator.WaveformGenerator(
            frequency_domain_source_model=lambda x, y, z: x)
        parameters = dict(y=1, z=2)
        self.ifo1.inject_signal = MagicMock(return_value=None)
        self.ifo2.inject_signal = MagicMock(return_value=None)
        self.ifo_list.inject_signal(parameters=parameters, waveform_generator=waveform_generator)
        self.assertDictEqual(parameters, waveform_generator.parameters)

    @patch.object(tupak.gw.detector.Interferometer, 'inject_signal')
    def test_inject_signal_with_inj_pol(self, m):
        self.ifo_list.inject_signal(injection_polarizations=dict(plus=1))
        m.assert_called_with(parameters=None, injection_polarizations=dict(plus=1))
        self.assertEqual(len(self.ifo_list), m.call_count)

    @patch.object(tupak.gw.detector.Interferometer, 'inject_signal')
    def test_inject_signal_returns_expected_polarisations(self, m):
        m.return_value = dict(plus=1, cross=2)
        injection_polarizations = dict(plus=1, cross=2)
        ifos_pol = self.ifo_list.inject_signal(injection_polarizations=injection_polarizations)
        self.assertDictEqual(self.ifo1.inject_signal(injection_polarizations=injection_polarizations), ifos_pol[0])
        self.assertDictEqual(self.ifo2.inject_signal(injection_polarizations=injection_polarizations), ifos_pol[1])

    @patch.object(tupak.gw.detector.Interferometer, 'save_data')
    def test_save_data(self, m):
        self.ifo_list.save_data(outdir='test_outdir', label='test_outdir')
        m.assert_called_with(outdir='test_outdir', label='test_outdir')
        self.assertEqual(len(self.ifo_list), m.call_count)

    def test_number_of_interferometers(self):
        self.assertEqual(len(self.ifo_list), self.ifo_list.number_of_interferometers)

    def test_duration(self):
        self.assertEqual(self.ifo1.strain_data.duration, self.ifo_list.duration)
        self.assertEqual(self.ifo2.strain_data.duration, self.ifo_list.duration)

    def test_sampling_frequency(self):
        self.assertEqual(self.ifo1.strain_data.sampling_frequency, self.ifo_list.sampling_frequency)
        self.assertEqual(self.ifo2.strain_data.sampling_frequency, self.ifo_list.sampling_frequency)

    def test_start_time(self):
        self.assertEqual(self.ifo1.strain_data.start_time, self.ifo_list.start_time)
        self.assertEqual(self.ifo2.strain_data.start_time, self.ifo_list.start_time)

    def test_frequency_array(self):
        self.assertTrue(np.array_equal(self.ifo1.strain_data.frequency_array, self.ifo_list.frequency_array))
        self.assertTrue(np.array_equal(self.ifo2.strain_data.frequency_array, self.ifo_list.frequency_array))

    def test_append_with_ifo(self):
        self.ifo_list.append(self.ifo2)
        names = [ifo.name for ifo in self.ifo_list]
        self.assertListEqual([self.ifo1.name, self.ifo2.name, self.ifo2.name], names)

    def test_append_with_ifo_list(self):
        self.ifo_list.append(self.ifo_list)
        names = [ifo.name for ifo in self.ifo_list]
        self.assertListEqual([self.ifo1.name, self.ifo2.name, self.ifo1.name, self.ifo2.name], names)

    def test_extend(self):
        self.ifo_list.extend(self.ifo_list)
        names = [ifo.name for ifo in self.ifo_list]
        self.assertListEqual([self.ifo1.name, self.ifo2.name, self.ifo1.name, self.ifo2.name], names)

    def test_insert(self):
        new_ifo = self.ifo1
        new_ifo.name = 'name3'
        self.ifo_list.insert(1, new_ifo)
        names = [ifo.name for ifo in self.ifo_list]
        self.assertListEqual([self.ifo1.name, new_ifo.name, self.ifo2.name], names)


if __name__ == '__main__':
    unittest.main()
