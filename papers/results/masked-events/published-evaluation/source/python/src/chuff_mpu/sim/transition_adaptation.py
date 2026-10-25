"""Evaluator-only mode stability and latency; conditions never enter the actor."""


def adaptation_metrics(rows, case, change_step):
    target = case not in ("stationary", "fast_stationary", "slow_stationary", "shallow")
    result = {}
    for mode in ("calibrated_gain", "adaptive_gain", "guarded_gain"):
        selected = [r for r in rows if r["mode"] == mode]
        if not selected:
            continue
        after = [r for r in selected if r["step"] >= change_step]
        stable = [r["step"]-change_step for index, r in enumerate(after[:-4])
                  if all(x["gain_compensation_enabled"] == target for x in after[index:index+5])]
        prior = selected[change_step-1]
        final = selected[-1]
        result[mode] = dict(target_compensation=target,
            initial_mode_before_change=prior["gain_compensation_enabled"],
            final_mode=final["gain_compensation_enabled"],
            switches_before=prior["mode_switches"],
            switches_after=final["mode_switches"]-prior["mode_switches"],
            first_five_window_correct_mode_delay=stable[0] if stable else None,
            wrong_mode_windows_after=sum(r["gain_compensation_enabled"] != target for r in after))
    return result
