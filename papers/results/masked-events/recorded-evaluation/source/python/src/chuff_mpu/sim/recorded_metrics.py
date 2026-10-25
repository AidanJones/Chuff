"""Recording-episode metrics. Consecutive windows are not independent repeats."""

from collections import Counter

import numpy as np

from chuff_mpu.sim.localization import DETECTED


def episode_metrics(rows, *, details=False):
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
                assigned[row["prototype_id"]] = label
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
    return result


def aggregate(summaries):
    result = []
    for suite in sorted({r["suite"] for r in summaries}):
        runs = sorted((r for r in summaries if r["suite"] == suite), key=lambda r: r["seed"])
        draws = np.random.default_rng(731).integers(0, len(runs), (10000, len(runs)))
        for mode in runs[0]["episodes"]:
            metrics = {}
            for name in runs[0]["episodes"][mode]["rates"]:
                values = [r["episodes"][mode]["rates"][name] for r in runs]
                if any(value is None for value in values):
                    metrics[name] = None
                    continue
                values = np.array(values)
                metrics[name] = dict(mean=float(values.mean()),
                    bootstrap_95=np.quantile(values[draws].mean(axis=1), [.025, .975]).tolist())
            baseline = [r["episodes"]["adaptive_gain"]["rates"]["source_dependent_repeat"] for r in runs]
            current = [r["episodes"][mode]["rates"]["source_dependent_repeat"] for r in runs]
            if not any(value is None for value in baseline+current):
                diff = np.array(current)-np.array(baseline)
                metrics["source_dependent_repeat_minus_adaptive"] = dict(mean=float(diff.mean()),
                    bootstrap_95=np.quantile(diff[draws].mean(axis=1), [.025, .975]).tolist())
            result.append(dict(suite=suite, mode=mode, seeds=len(runs), metrics=metrics))
    return result
