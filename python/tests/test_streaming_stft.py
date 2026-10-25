"""Framing parity, exact sample ownership and boundary-event controls."""

import unittest

import numpy as np
from scipy.signal import stft

from chuff_mpu.streaming_stft import ContinuousSTFT


class ContinuousSTFTTests(unittest.TestCase):
    def test_arbitrary_chunks_match_one_shot_complex_spectra(self):
        signal = np.random.default_rng(43).normal(size=32123)
        for n_fft, hop in ((512, 160), (512, 128), (511, 127), (32, 32)):
            _, times, expected = stft(signal, fs=16000, nperseg=n_fft,
                noverlap=n_fft-hop, boundary=None, padded=False)
            for sizes in ([len(signal)], [1]*len(signal), [0, 127, 16000, 3, 4, 15989]):
                with self.subTest(n_fft=n_fft, hop=hop, chunks=len(sizes)):
                    model, offset, outputs = ContinuousSTFT(n_fft, hop), 0, []
                    for size in sizes:
                        output = model.push(signal[offset:offset+size])
                        self.assertTrue(np.all(output.stop_samples > offset))
                        offset += min(size, len(signal)-offset)
                        self.assertTrue(np.all(output.stop_samples <= offset))
                        outputs.append(output)
                    if offset < len(signal):
                        outputs.append(model.push(signal[offset:]))
                    spectrum = np.concatenate([o.spectrum for o in outputs], axis=1)
                    starts = np.concatenate([o.start_samples for o in outputs])
                    np.testing.assert_allclose(spectrum, expected, rtol=2e-13, atol=2e-15)
                    np.testing.assert_array_equal(starts, np.arange(expected.shape[1])*hop)
                    np.testing.assert_allclose((starts+n_fft/2)/16000, times, atol=1e-15)
                    self.assertLess(len(model._pending), n_fft)
                    self.assertEqual(model.finish()["uncovered_tail_samples"], len(signal)-(starts[-1]+n_fft))

    def test_boundary_impulse_is_in_a_continuous_frame(self):
        signal = np.zeros(32000)
        signal[15999] = 1.
        model = ContinuousSTFT()
        first, second = model.push(signal[:16000]), model.push(signal[16000:])
        crossing = (second.start_samples < 16000) & (second.stop_samples > 16000)
        self.assertTrue(np.any(np.abs(second.spectrum[:, crossing]) > 0))
        # Separate unpadded clips omit the end of the first second entirely.
        _, _, clipped = stft(signal[:16000], nperseg=512, noverlap=352, boundary=None, padded=False)
        self.assertFalse(np.any(clipped))
        self.assertFalse(np.any(first.spectrum))

    def test_empty_short_and_overlap_tail_accounting(self):
        model = ContinuousSTFT(8, 3)
        self.assertEqual(model.push([]).spectrum.shape, (5, 0))
        self.assertEqual(model.push(np.ones(7)).spectrum.shape[1], 0)
        self.assertEqual(model.finish()["uncovered_tail_samples"], 7)
        model.reset()
        self.assertEqual(model.push(np.ones(8)).spectrum.shape[1], 1)
        self.assertEqual(model.finish()["uncovered_tail_samples"], 0)
        self.assertEqual(model.finish()["frames"], 1)
        with self.assertRaises(ValueError):
            model.push([])

    def test_invalid_input_preserves_state_and_input_is_not_aliased(self):
        model, reference = ContinuousSTFT(8, 3), ContinuousSTFT(8, 3)
        signal = np.arange(5.)
        model.push(signal)
        reference.push(signal.copy())
        signal[:] = 999.
        for bad in ([np.nan], [[1.]], [1j], [True], ["1"]):
            with self.assertRaises(ValueError):
                model.push(bad)
        with np.errstate(over="ignore", invalid="ignore"), self.assertRaises(ValueError):
            model.push(np.full(8, np.finfo(np.float64).max))
        actual, expected = model.push(np.arange(10.)), reference.push(np.arange(10.))
        np.testing.assert_array_equal(actual.spectrum, expected.spectrum)
        np.testing.assert_array_equal(actual.start_samples, expected.start_samples)
        self.assertEqual(model.finish(), reference.finish())

    def test_invalid_geometry(self):
        for n_fft, hop in ((True, 1), (1, 1), (8, 0), (8, 9), (8, True), (8., 2)):
            with self.assertRaises(ValueError):
                ContinuousSTFT(n_fft, hop)


if __name__ == "__main__":
    unittest.main()
