"""Evaluator-only region overlap. Window coincidence is not event localization."""

from collections import Counter


THRESHOLDS = (.1, .25, .5)
DETECTED = ("candidate", "learned", "recognised")


def overlap_iou(region, event):
    a, b = region
    c, d = event
    if not 0 <= a < b or not 0 <= c < d:
        raise ValueError("expected positive intervals with nonnegative starts")
    intersection = max(0., min(b, d)-max(a, c))
    return intersection/(b-a+d-c-intersection)


def score_regions(rows):
    """Score pre-channel source intervals, with explicit missing timing counts.

    Regions span the full support of selected FFT frames. Source intervals
    exclude room and IIR microphone tails; previous-source room-tail windows
    are flagged separately. This is a reproducible localization proxy, not
    clean-source attribution or calibrated perceptual onset matching.
    """
    result = {}
    for mode in sorted({r["mode"] for r in rows}):
        counts = Counter()
        hits = {str(t): 0 for t in THRESHOLDS}
        correct_repeats = {str(t): 0 for t in THRESHOLDS}
        occurrences, assigned = Counter(), {}
        for row in (r for r in rows if r["mode"] == mode):
            label, status = row["truth"], row["status"]
            detected = status in DETECTED
            region = row.get("region_start_s"), row.get("region_stop_s")
            event = row.get("event_start_s"), row.get("event_stop_s")
            if label is not None:
                counts["event_windows"] += 1
                counts["window_detections"] += detected
                repeat = occurrences[label] >= 2
                counts["repeat_windows"] += repeat
                occurrences[label] += 1
                if None in event:
                    counts["missing_event_timing"] += 1
                elif detected:
                    if None in region:
                        counts["missing_detected_region"] += 1
                    else:
                        iou = overlap_iou(region, event)
                        for threshold in THRESHOLDS:
                            hits[str(threshold)] += iou >= threshold
                            correct_repeats[str(threshold)] += (iou >= threshold and repeat
                                and status == "recognised" and assigned.get(row["prototype_id"]) == label)
            else:
                counts["background_windows"] += 1
                counts["false_attention_windows"] += detected
                if row.get("previous_source_tail_s", 0) > 0:
                    counts["background_with_room_tail"] += 1
                    counts["tail_attention_windows"] += detected
                else:
                    counts["background_without_room_tail"] += 1
                    counts["no_room_tail_attention_windows"] += detected
            if status == "learned":
                assigned[row["prototype_id"]] = label
        complete = not (counts["missing_event_timing"] or counts["missing_detected_region"])
        result[mode] = dict(counts=dict(counts), iou_hits=hits, localized_correct_repeats=correct_repeats,
            localized_detection_rate={key: value/counts["event_windows"] if complete and counts["event_windows"] else None
                                      for key, value in hits.items()},
            localized_repeat_recognition_rate={key: value/counts["repeat_windows"] if complete and counts["repeat_windows"] else None
                                               for key, value in correct_repeats.items()})
    return result
