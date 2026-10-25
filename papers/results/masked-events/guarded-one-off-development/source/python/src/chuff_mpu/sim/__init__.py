"""In-silico test of the Chuff listen -> remember -> repeat loop.

No real microphone or speaker is required: `world.py` synthesises a
small fixed set of "real-world sounds" plus background noise standing in
for a room and a mic; `agent.py` wraps the real (not stub) `audio.py` /
`memory.py` / `comms.py` modules to actually hear, remember, and repeat
those sounds in its own voice, including hearing its own played-back
call mixed back in with fresh noise. Every sample that would go to a
speaker still passes through `audio_safety.py`'s limiter regardless of
whether real playback is enabled (see `run_demo.py --play`).

Run it with:

    uv run chuff-sim
"""
