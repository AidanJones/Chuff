"""Capture/replay behaviour and restart/schema safeguards, without real devices."""

from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

from chuff_mpu import config
from chuff_mpu.attention import AttentionConfig, ForegroundLearner
from chuff_mpu.memory import EmbeddingStore
from chuff_mpu.mood import Mood
from chuff_mpu.selective import SelectiveConfig, SelectiveController
from chuff_mpu.selective_demo import main, read_wav
from chuff_mpu.sim.scene import AcousticScene, SceneConfig


def example_clips():
    quiet = np.random.default_rng(21).normal(0, 0.01, 16000).astype(np.float32)
    event = quiet.copy()
    t = np.arange(4000) / 16000
    event[4000:8000] += 0.1 * np.sin(2 * np.pi * 1600 * t) * np.hanning(4000)
    return quiet, event


def trained_listener():
    quiet, event = example_clips()
    listener = ForegroundLearner()
    for clip in [quiet, quiet, quiet, event, quiet, event]:
        result = listener.observe(clip)
    assert result.status == "learned"
    return listener, result.prototype_id


class SelectiveControllerTests(unittest.TestCase):
    def test_curiosity_admission_habituation_and_background_have_distinct_responses(self):
        quiet, event = example_clips()
        controller = SelectiveController(learning_cue=True)
        results = [controller.process(clip) for clip in [quiet] * 3 + [event, quiet, event, quiet, event]]
        self.assertEqual([r.observation.status for r in results],
                         ["calibrating"] * 3 + ["candidate", "background", "learned", "background", "recognised"])
        self.assertEqual(results[3].mood.mood, Mood.MILD_NOVELTY)
        self.assertEqual(results[5].mood.mood, Mood.NEW_SOUND_LEARNED)
        self.assertEqual(results[6].mood.mood, Mood.CONTENTMENT)
        self.assertEqual(results[7].mood.mood, Mood.SONG_RECOGNISED)
        self.assertLess(results[7].mood.energy, results[5].mood.energy)
        self.assertEqual([r.cue is not None for r in results], [False] * 5 + [True, False, False])
        self.assertLessEqual(float(np.abs(results[5].cue).max()), config.SAFE_PEAK_AMPLITUDE)
        self.assertLessEqual(len(results[5].cue), int(config.SAFE_MAX_DURATION_S * 16000))

    def test_changing_background_and_silence_never_request_a_cue(self):
        controller = SelectiveController(learning_cue=True)
        for frame in AcousticScene(17, SceneConfig(rounds=60, event_probability=0, nuisance_probability=0.5)):
            self.assertIsNone(controller.process(frame.audio).cue)
        for _ in range(4):
            self.assertIsNone(controller.process(np.zeros(16000)).cue)
        self.assertEqual(controller.listener.memory_size, 0)

    def test_idle_state_is_boredom_not_an_unimplemented_dream(self):
        controller = SelectiveController(settings=SelectiveConfig(bored_after_s=4))
        for _ in range(4):
            result = controller.process(np.zeros(16000))
        self.assertEqual(result.mood.mood, Mood.BORED)
        self.assertIsNone(result.cue)

    def test_legacy_memory_cannot_be_attached_to_selective_controller(self):
        with self.assertRaises(TypeError):
            SelectiveController(EmbeddingStore(32))


class ForegroundPersistenceTests(unittest.TestCase):
    def test_restart_retains_identity_but_recalibrates_background_and_pending_events(self):
        listener, identifier = trained_listener()
        quiet, event = example_clips()
        other = quiet.copy()
        t = np.arange(4000) / 16000
        other[4000:8000] += 0.1 * np.sin(2 * np.pi * 650 * t) * np.hanning(4000)
        self.assertEqual(listener.observe(other).status, "candidate")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.npz"
            listener.save(path)
            restored = ForegroundLearner.load(path)
        self.assertEqual(restored.memory_size, 1)
        self.assertEqual(restored.candidate_count, 0)
        self.assertEqual(restored.settings, listener.settings)
        for _ in range(3):
            self.assertEqual(restored.observe(quiet).status, "calibrating")
        known = restored.observe(event)
        self.assertEqual(known.status, "recognised")
        self.assertEqual(known.prototype_id, identifier)
        self.assertEqual(restored.observe(other).status, "candidate")

    def test_legacy_and_malformed_feature_files_are_rejected(self):
        listener, _ = trained_listener()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.npz"
            np.savez(path, np.ones(13))
            with self.assertRaises(ValueError):
                ForegroundLearner.load(path)
            listener.save(path)
            with np.load(path, allow_pickle=False) as data:
                original = {name: data[name].copy() for name in data.files}
            bad_features = [np.ones((1, 13)), np.full((1, 96), np.nan), np.zeros((1, 96))]
            for features in bad_features:
                np.savez(path, **{**original, "features": features})
                with self.assertRaises(ValueError):
                    ForegroundLearner.load(path)
            metadata = json.loads(original["metadata"].item())
            metadata["settings"]["sample_rate"] = float("nan")
            np.savez(path, **{**original, "metadata": np.array(json.dumps(metadata))})
            with self.assertRaises(ValueError):
                ForegroundLearner.load(path)

    def test_failed_atomic_save_preserves_the_previous_file(self):
        listener, _ = trained_listener()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.npz"
            listener.save(path)
            original = path.read_bytes()
            with patch("chuff_mpu.attention.np.savez_compressed", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    listener.save(path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_empty_memory_and_nondefault_settings_round_trip(self):
        listener = ForegroundLearner(replace(AttentionConfig(), capacity=4))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.npz"
            listener.save(path)
            restored = ForegroundLearner.load(path)
        self.assertEqual(restored.memory_size, 0)
        self.assertEqual(restored.settings.capacity, 4)


class CaptureReplayTests(unittest.TestCase):
    def test_wav_replay_is_silent_without_mic_or_serial_and_restores_state(self):
        quiet, event = example_clips()
        samples = np.concatenate([quiet] * 3 + [event, quiet, event, quiet, event])
        with tempfile.TemporaryDirectory() as directory:
            wav = Path(directory) / "input.wav"
            state = Path(directory) / "state.npz"
            wavfile.write(wav, 16000, (samples * 32767).astype(np.int16))
            for run in range(2):
                output = io.StringIO()
                with (patch("sys.argv", ["chuff-selective", "--input-wav", str(wav), "--state", str(state)]),
                      patch("chuff_mpu.audio.capture_clip", side_effect=AssertionError("unexpected mic access")),
                      patch("chuff_mpu.audio_safety.safe_play", side_effect=AssertionError("unexpected speaker access")),
                      patch("chuff_mpu.ipc.McuLink", side_effect=AssertionError("unexpected serial access")),
                      redirect_stdout(output)):
                    main()
                records = [json.loads(line) for line in output.getvalue().splitlines()]
                self.assertEqual(records[-1]["windows"], 8)
                self.assertEqual(records[-1]["memory_size"], 1)
                self.assertTrue(all(r.get("cue_samples", 0) == 0 for r in records))
                if run == 1:
                    self.assertEqual(records[0]["restored_memories"], 1)
                    self.assertNotIn("learned", records[-1]["statuses"])

    def test_mocked_capture_sends_real_serial_protocol_and_only_one_learning_cue(self):
        quiet, event = example_clips()
        captures = iter([quiet] * 3 + [event, quiet, event, quiet])
        durations = []

        def capture(duration, rate):
            durations.append(duration)
            return next(captures) if duration == 1 else np.zeros(int(duration * rate))

        output = io.StringIO()
        with (patch("sys.argv", ["chuff-selective", "--mic", "--port", "TEST", "--max-windows", "7",
                                 "--play-learning-cue"]),
              patch("chuff_mpu.audio.capture_clip", side_effect=capture),
              patch("chuff_mpu.audio_safety.safe_play") as player,
              patch("chuff_mpu.ipc.serial.Serial") as serial,
              redirect_stdout(output)):
            main()
        player.assert_called_once()
        self.assertEqual(durations.count(1), 7)
        self.assertEqual(durations.count(0.25), 1)
        messages = [json.loads(call.args[0]) for call in serial.return_value.write.call_args_list]
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        observations = [r for r in records if r["type"] == "observation"]
        self.assertEqual(len(messages), 7)
        self.assertEqual([m["mood"] for m in messages], [r["mood"] for r in observations])
        self.assertEqual(messages[5]["mood"], "new_sound_learned")
        self.assertEqual(messages[6]["mood"], "contentment")
        serial.return_value.close.assert_called_once()

    def test_wav_formats_rates_and_partial_tail_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            wav = Path(directory) / "input.wav"
            wavfile.write(wav, 16000, np.full((16005, 2), 128, dtype=np.uint8))
            self.assertTrue(np.all(read_wav(wav, 16000) == 0))
            output = io.StringIO()
            with (patch("sys.argv", ["chuff-selective", "--input-wav", str(wav)]), redirect_stdout(output)):
                main()
            records = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual(records[0]["discarded_tail_samples"], 5)
            self.assertEqual(records[-1]["windows"], 1)
            with self.assertRaises(ValueError):
                read_wav(wav, 8000)


if __name__ == "__main__":
    unittest.main()
