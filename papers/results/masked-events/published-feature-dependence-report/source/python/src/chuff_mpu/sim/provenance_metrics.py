"""Explicit-label diagnostic scorer, kept separate from frozen historical scores.

This copies the original episode scorer with only prototype assignment changed.
Labels are evaluator mappings keyed by (mode, prototype ID). Actual episode
truth, admission timing, eligibility and probe rules remain unchanged.
Admission-label parity must be checked against the original scorer on each run.
"""

from collections import Counter
from chuff_mpu.sim.localization import DETECTED

REFERENCE_SCORER_SHA256 = 'a1f67d0bb47b0463bcf314351fd64cb71f7b3ac8d2036098340273cfa252691b'


def provenance_episode_metrics(rows, *, labels, details=False, exclusion=False):
    result = {}
    for mode in sorted({row["mode"] for row in rows}):
        selected = [r for r in rows if r["mode"] == mode]
        assigned, admitted_episode = {}, {}
        episodes, counts = {}, Counter()
        for row in selected:
            episode, label, status = row["episode_id"], row["recording_id"], row["status"]
            counts["windows"] += 1
            if episode is None:
                counts["background_windows"] += 1
                counts["background_attention"] += status in DETECTED
                counts["background_admissions"] += status == "learned"
                if "previous_source_tail_s" in row:
                    tail = row["previous_source_tail_s"] > 0
                    counts["background_with_room_tail"] += tail
                    counts["room_tail_attention"] += tail and status in DETECTED
                    counts["background_without_room_tail"] += not tail
                    counts["no_room_tail_attention"] += not tail and status in DETECTED
            else:
                item = episodes.setdefault(episode, dict(recording_id=label, occurrence=row["occurrence"],
                    detected=False, novel=False, correct=False, source_dependent=False,
                    deletion_correct=False, deletion_attention=False, wrong=False))
                item["detected"] |= status in DETECTED
                item["novel"] |= status in ("candidate", "learned")
                # Recognition learned earlier in THIS episode is not repeat recall.
                prior = (row["prototype_id"] in admitted_episode
                         and admitted_episode[row["prototype_id"]] != episode)
                correct = status == "recognised" and prior and assigned.get(row["prototype_id"]) == label
                wrong = status == "recognised" and prior and assigned.get(row["prototype_id"]) != label
                probe = row["without_episode_probe"]
                probe_prior = (probe["prototype_id"] in admitted_episode
                               and admitted_episode[probe["prototype_id"]] != episode)
                probe_correct = (probe["status"] == "recognised" and probe_prior
                                 and assigned.get(probe["prototype_id"]) == label)
                item["correct"] |= correct
                item["wrong"] |= wrong
                item["source_dependent"] |= correct and not probe_correct
                item["deletion_correct"] |= probe_correct
                item["deletion_attention"] |= probe["status"] in DETECTED
            if status == "learned":
                assigned[row["prototype_id"]] = labels[mode, row["prototype_id"]]
                admitted_episode[row["prototype_id"]] = episode
        repeats = [item for item in episodes.values() if item["occurrence"] >= 3]
        all_items = list(episodes.values())
        counts.update(episodes=len(all_items), eligible_repeat_episodes=len(repeats),
            detected_episodes=sum(item["detected"] for item in all_items),
            correct_repeat_episodes=sum(item["correct"] for item in repeats),
            source_dependent_repeat_episodes=sum(item["source_dependent"] for item in repeats),
            deletion_correct_repeat_episodes=sum(item["deletion_correct"] for item in repeats),
            deletion_attention_episodes=sum(item["deletion_attention"] for item in all_items),
            wrong_repeat_episodes=sum(item["wrong"] for item in repeats),
            wrong_only_repeat_episodes=sum(item["wrong"] and not item["correct"] for item in repeats),
            first_episode_novel_attention=sum(item["novel"] for item in all_items if item["occurrence"] == 1),
            distinct_recordings=len({item["recording_id"] for item in all_items}),
            admitted_recordings=len({label for label in assigned.values() if label is not None}))
        divide = lambda a, b: counts[a]/counts[b] if counts[b] else None
        result[mode] = dict(counts=dict(counts), rates=dict(
            episode_detection=divide("detected_episodes", "episodes"),
            repeat_identity=divide("correct_repeat_episodes", "eligible_repeat_episodes"),
            source_dependent_repeat=divide("source_dependent_repeat_episodes", "eligible_repeat_episodes"),
            deletion_repeat_identity=divide("deletion_correct_repeat_episodes", "eligible_repeat_episodes"),
            deletion_attention=divide("deletion_attention_episodes", "episodes"),
            wrong_repeat_identity=divide("wrong_repeat_episodes", "eligible_repeat_episodes"),
            background_attention=divide("background_attention", "background_windows"),
            first_episode_novel_attention=divide("first_episode_novel_attention", "distinct_recordings")))
        if "background_with_room_tail" in counts:
            result[mode]["rates"].update(
                room_tail_background_attention=divide("room_tail_attention", "background_with_room_tail"),
                no_room_tail_background_attention=divide("no_room_tail_attention", "background_without_room_tail"))
        if details:
            by_recording = {}
            for label in sorted({item["recording_id"] for item in all_items}):
                items = [item for item in all_items if item["recording_id"] == label]
                eligible = [item for item in items if item["occurrence"] >= 3]
                by_recording[label] = dict(episodes=len(items), eligible_repeat_episodes=len(eligible),
                    detected_episodes=sum(item["detected"] for item in items),
                    correct_repeat_episodes=sum(item["correct"] for item in eligible),
                    source_dependent_repeat_episodes=sum(item["source_dependent"] for item in eligible),
                    deletion_correct_repeat_episodes=sum(item["deletion_correct"] for item in eligible),
                    wrong_repeat_episodes=sum(item["wrong"] for item in eligible))
            result[mode]["by_recording"] = by_recording
        if exclusion:
            strict = sum(item["correct"] and not item["deletion_correct"] for item in repeats)
            result[mode]["counts"]["strict_episode_exclusion_hits"] = strict
            result[mode]["rates"]["strict_episode_exclusion"] = strict/len(repeats) if repeats else None
            if details:
                for label, item in result[mode]["by_recording"].items():
                    item["strict_episode_exclusion_hits"] = sum(
                        episode["correct"] and not episode["deletion_correct"]
                        for episode in repeats if episode["recording_id"] == label)
    return result
