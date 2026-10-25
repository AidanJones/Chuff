"""Evaluator-only memory error attribution; labels never reach a learner."""

from collections import Counter


REPEAT_OUTCOMES = ("missed_detection", "pending_candidate", "first_admission",
                   "duplicate_admission", "wrong_identity", "correct_identity")


def diagnose(rows):
    """Use admission-time class labels, not retrospective majority relabelling.

    Repeat eligibility starts after two true occurrences for EVERY policy,
    including those requiring more confirmations. Duplicate admission means
    another prototype for an already admitted evaluator class, not proof
    that every acoustic instance of that class should have one identity.
    """
    result = {}
    for mode in sorted({r["mode"] for r in rows}):
        occurrences, admitted = Counter(), Counter()
        assigned, first_delay = {}, {}
        outcomes = Counter({key: 0 for key in REPEAT_OUTCOMES})
        false = Counter({"current_nuisance": 0, "no_current_source": 0})
        for row in (r for r in rows if r["mode"] == mode):
            label, status, identifier = row["truth"], row["status"], row["prototype_id"]
            if label is not None:
                if occurrences[label] >= 2:
                    if status == "candidate":
                        outcome = "pending_candidate"
                    elif status == "learned":
                        outcome = "duplicate_admission" if admitted[label] else "first_admission"
                    elif status == "recognised":
                        outcome = "correct_identity" if assigned.get(identifier) == label else "wrong_identity"
                    else:
                        outcome = "missed_detection"
                    outcomes[outcome] += 1
                occurrences[label] += 1
            if status == "learned":
                assigned[identifier] = label
                if label is None:
                    false["current_nuisance" if row["nuisance"] else "no_current_source"] += 1
                else:
                    admitted[label] += 1
                    first_delay.setdefault(label, occurrences[label])
        repeats = sum(max(0, count-2) for count in occurrences.values())
        assert sum(outcomes.values()) == repeats
        result[mode] = {"repeat_events": repeats, "repeat_outcomes": dict(outcomes),
                        "admissions_per_class": dict(admitted),
                        "first_admission_occurrence": first_delay,
                        "observed_classes": len(occurrences), "admitted_classes": len(admitted),
                        "false_admissions_by_current_source": dict(false)}
    return result
