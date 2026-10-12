#pragma once

/*
 * Tunable constants for Chuff's mood/animation behaviour.
 *
 * Mirrors the "all thresholds live in config.py, never hardcoded
 * elsewhere" convention from the Chuff spec, applied to this C codebase.
 * As more mood behaviour lands (novelty scoring, dream loop, comms),
 * its thresholds belong here too.
 */

/* How long with no touch before Chuff drops into "dreaming" (idle). */
#define CHUFF_IDLE_TIMEOUT_MS       (5UL * 60UL * 1000UL)   /* 5 min */

/* Idle-animation cadence while dreaming vs. while awake. Dreaming is a
 * slow "breathing" cadence; awake is the normal blink/look-around pace. */
#define CHUFF_DREAM_IDLE_PERIOD_MS  12000
#define CHUFF_NORMAL_IDLE_PERIOD_MS 4000
