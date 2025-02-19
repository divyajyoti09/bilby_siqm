import unittest
import os
from shutil import rmtree

import numpy as np
import lal
import lalsimulation as lalsim
from gwpy.timeseries import TimeSeries
from gwpy.detector import Channel
from scipy.stats import ks_2samp

import bilby
from bilby.gw import utils as gwutils


class TestGWUtils(unittest.TestCase):
    def setUp(self):
        self.outdir = "outdir"
        bilby.core.utils.check_directory_exists_and_if_not_mkdir(self.outdir)

    def tearDown(self):
        try:
            rmtree(self.outdir)
        except FileNotFoundError:
            pass

    def test_asd_from_freq_series(self):
        freq_data = np.array([1, 2, 3])
        df = 0.1
        asd = gwutils.asd_from_freq_series(freq_data, df)
        self.assertTrue(np.all(asd == freq_data * 2 * df ** 0.5))

    def test_psd_from_freq_series(self):
        freq_data = np.array([1, 2, 3])
        df = 0.1
        psd = gwutils.psd_from_freq_series(freq_data, df)
        self.assertTrue(np.all(psd == (freq_data * 2 * df ** 0.5) ** 2))

    def test_time_delay_from_geocenter(self):
        """
        The difference in the two detector case is due to rounding error.
        Different hardware gives different numbers in the last decimal place.
        """
        det1 = np.array([0.1, 0.2, 0.3])
        det2 = np.array([0.1, 0.2, 0.5])
        ra = 0.5
        dec = 0.2
        time = 10
        self.assertEqual(gwutils.time_delay_geocentric(det1, det1, ra, dec, time), 0)
        self.assertAlmostEqual(
            gwutils.time_delay_geocentric(det1, det2, ra, dec, time),
            1.3253791114055397e-10,
            14,
        )

    def test_get_polarization_tensor(self):
        ra = 1
        dec = 2.0
        time = 10
        psi = 0.1
        for mode in ["plus", "cross", "breathing", "longitudinal", "x", "y"]:
            p = gwutils.get_polarization_tensor(ra, dec, time, psi, mode)
            self.assertEqual(p.shape, (3, 3))
        with self.assertRaises(ValueError):
            gwutils.get_polarization_tensor(ra, dec, time, psi, "not-a-mode")

    def test_inner_product(self):
        aa = np.array([1, 2, 3])
        bb = np.array([5, 6, 7])
        frequency = np.array([0.2, 0.4, 0.6])
        PSD = bilby.gw.detector.PowerSpectralDensity.from_aligo()
        ip = gwutils.inner_product(aa, bb, frequency, PSD)
        self.assertEqual(ip, 0)

    def test_noise_weighted_inner_product(self):
        aa = np.array([1e-23, 2e-23, 3e-23])
        bb = np.array([5e-23, 6e-23, 7e-23])
        frequency = np.array([100, 101, 102])
        PSD = bilby.gw.detector.PowerSpectralDensity.from_aligo()
        psd = PSD.power_spectral_density_interpolated(frequency)
        duration = 4
        nwip = gwutils.noise_weighted_inner_product(aa, bb, psd, duration)
        self.assertEqual(nwip, 239.87768033598326)

        self.assertEqual(
            gwutils.optimal_snr_squared(aa, psd, duration),
            gwutils.noise_weighted_inner_product(aa, aa, psd, duration),
        )

    def test_matched_filter_snr(self):
        signal = np.array([1e-23, 2e-23, 3e-23])
        frequency_domain_strain = np.array([5e-23, 6e-23, 7e-23])
        frequency = np.array([100, 101, 102])
        PSD = bilby.gw.detector.PowerSpectralDensity.from_aligo()
        psd = PSD.power_spectral_density_interpolated(frequency)
        duration = 4

        mfsnr = gwutils.matched_filter_snr(
            signal, frequency_domain_strain, psd, duration
        )
        self.assertEqual(mfsnr, 25.510869054168282)

    def test_get_event_time(self):
        events = [
            "GW150914",
            "GW170104",
        ]
        for event in events:
            self.assertTrue(isinstance(gwutils.get_event_time(event), float))

        with self.assertRaises(ValueError):
            gwutils.get_event_time("GW010290")

    def test_read_frame_file(self):
        start_time = 0
        end_time = 10
        channel = "H1:GDS-CALIB_STRAIN"
        N = 100
        times = np.linspace(start_time, end_time, N)
        data = np.random.normal(0, 1, N)
        ts = TimeSeries(data=data, times=times, t0=0)
        ts.channel = Channel(channel)
        ts.name = channel
        filename = os.path.join(self.outdir, "test.gwf")
        ts.write(filename, format="gwf")

        # Check reading without time limits
        strain = gwutils.read_frame_file(
            filename, start_time=None, end_time=None, channel=channel
        )
        self.assertEqual(strain.channel.name, channel)
        self.assertTrue(np.all(strain.value == data[:-1]))

        # Check reading with time limits
        start_cut = 2
        end_cut = 8
        strain = gwutils.read_frame_file(
            filename, start_time=start_cut, end_time=end_cut, channel=channel
        )
        idxs = (times > start_cut) & (times < end_cut)
        # Dropping the last element - for some reason gwpy drops the last element when reading in data
        self.assertTrue(np.all(strain.value == data[idxs][:-1]))

        # Check reading with unknown channels
        strain = gwutils.read_frame_file(filename, start_time=None, end_time=None)
        self.assertTrue(np.all(strain.value == data[:-1]))

        # Check reading with incorrect channel
        strain = gwutils.read_frame_file(
            filename, start_time=None, end_time=None, channel="WRONG"
        )
        self.assertTrue(np.all(strain.value == data[:-1]))

        ts = TimeSeries(data=data, times=times, t0=0)
        ts.name = "NOT-A-KNOWN-CHANNEL"
        ts.write(filename, format="gwf")
        strain = gwutils.read_frame_file(filename, start_time=None, end_time=None)
        self.assertEqual(strain, None)

    def test_convert_args_list_to_float(self):
        self.assertEqual(
            gwutils.convert_args_list_to_float(1, "2", 3.0), [1.0, 2.0, 3.0]
        )
        with self.assertRaises(ValueError):
            gwutils.convert_args_list_to_float(1, "2", "ten")

    def test_lalsim_SimInspiralTransformPrecessingNewInitialConditions(self):
        a = gwutils.lalsim_SimInspiralTransformPrecessingNewInitialConditions(
            0.1, 0, 0.6, 0.5, 0.6, 0.1, 0.8, 30.6, 23.2, 50, 0
        )
        self.assertTrue(len(a) == 7)

    def test_get_approximant(self):
        with self.assertRaises(ValueError):
            gwutils.lalsim_GetApproximantFromString(10)

    def test_lalsim_SimInspiralChooseFDWaveform(self):
        a = gwutils.lalsim_SimInspiralChooseFDWaveform(
            35.2,
            20.4,
            0.1,
            0.2,
            0.2,
            0.2,
            0.2,
            0.1,
            1000,
            2,
            2.3,
            45.0,
            0.1,
            10,
            0.01,
            10,
            1000,
            20,
            None,
            lalsim.IMRPhenomPv2,
        )
        self.assertEqual(len(a), 2)
        self.assertEqual(type(a[0]), lal.COMPLEX16FrequencySeries)
        self.assertEqual(type(a[1]), lal.COMPLEX16FrequencySeries)

        with self.assertRaises(RuntimeError):
            _ = gwutils.lalsim_SimInspiralChooseFDWaveform(
                35.2,
                20.4,
                0.1,
                0.2,
                0.2,
                0.2,
                0.2,
                0.1,
                1000,
                2,
                2.3,
                45.0,
                0.1,
                10,
                0.01,
                10,
                1000,
                20,
                None,
                "Fail",
            )

        with self.assertRaises(ValueError):
            _ = gwutils.lalsim_SimInspiralChooseFDWaveform(
                35.2,
                20.4,
                0.1,
                0.2,
                0.2,
                0.2,
                0.2,
                0.1,
                1000,
                2,
                2.3,
                45.0,
                0.1,
                10,
                0.01,
                10,
                1000,
                20,
                None,
                1.5,
            )


class TestSkyFrameConversion(unittest.TestCase):

    def setUp(self) -> None:
        self.priors = bilby.core.prior.PriorDict()
        self.priors["ra"] = bilby.core.prior.Uniform(0, 2 * np.pi)
        self.priors["dec"] = bilby.core.prior.Cosine()
        self.priors["azimuth"] = bilby.core.prior.Uniform(0, 2 * np.pi)
        self.priors["zenith"] = bilby.core.prior.Sine()
        self.priors["time"] = bilby.core.prior.Uniform(-0.1, 0.1)
        self.ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
        self.samples = self.priors.sample(10000)

    def tearDown(self) -> None:
        del self.priors
        del self.ifos
        del self.samples

    def test_conversion_gives_correct_prior(self) -> None:
        zeniths = self.samples["zenith"]
        azimuths = self.samples["azimuth"]
        times = self.samples["time"]
        args = zip(*[
            (zenith, azimuth, time, self.ifos)
            for zenith, azimuth, time in zip(zeniths, azimuths, times)
        ])
        ras, decs = zip(*map(bilby.gw.utils.zenith_azimuth_to_ra_dec, *args))
        self.assertGreaterEqual(ks_2samp(self.samples["ra"], ras).pvalue, 0.01)
        self.assertGreaterEqual(ks_2samp(self.samples["dec"], decs).pvalue, 0.01)


if __name__ == "__main__":
    unittest.main()
