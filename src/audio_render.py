"""
MIDI -> WAV rendering so the browser can play generated music directly
instead of forcing every user to download a .mid file and open it in a
separate app (a common friction point in similar projects).

Tries, in order:
  1. FluidSynth + a General MIDI soundfont, if installed on the host.
  2. A lightweight pure-Python sine-wave synthesizer fallback so the
     feature still works out-of-the-box on platforms (like Streamlit
     Community Cloud) where installing FluidSynth isn't convenient.
"""

import os
import shutil
import subprocess
import wave
import struct
import math

import pretty_midi

from . import config


def _find_soundfont():
    for path in config.SOUNDFONT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def render_with_fluidsynth(midi_path: str, wav_path: str) -> bool:
    soundfont = _find_soundfont()
    if not soundfont or not shutil.which("fluidsynth"):
        return False

    cmd = [
        "fluidsynth", "-ni", soundfont, midi_path,
        "-F", wav_path, "-r", str(config.DEFAULT_SAMPLE_RATE),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return os.path.exists(wav_path)
    except Exception:
        return False


def render_with_sine_fallback(midi_path: str, wav_path: str, sample_rate=config.DEFAULT_SAMPLE_RATE):
    """Very simple additive sine-wave synth so playback works even
    without FluidSynth/a soundfont installed."""
    pm = pretty_midi.PrettyMIDI(midi_path)
    total_duration = max(pm.get_end_time(), 0.5)
    n_frames = int(total_duration * sample_rate) + sample_rate  # 1s tail

    samples = [0.0] * n_frames

    for instrument_track in pm.instruments:
        for note_obj in instrument_track.notes:
            freq = 440.0 * (2.0 ** ((note_obj.pitch - 69) / 12.0))
            start_frame = int(note_obj.start * sample_rate)
            end_frame = int(note_obj.end * sample_rate)
            amplitude = min(max(note_obj.velocity / 127.0, 0.05), 0.9) * 0.2

            for i in range(start_frame, min(end_frame, n_frames)):
                t = (i - start_frame) / sample_rate
                # simple decay envelope so notes don't click/pop
                envelope = math.exp(-2.0 * t)
                samples[i] += amplitude * envelope * math.sin(2 * math.pi * freq * t)

    # normalize to avoid clipping
    peak = max(1e-6, max(abs(s) for s in samples))
    scale = 0.9 / peak

    with wave.open(wav_path, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        frames = b"".join(
            struct.pack("<h", int(max(-1.0, min(1.0, s * scale)) * 32767))
            for s in samples
        )
        wav_file.writeframes(frames)

    return wav_path


def render_midi_to_wav(midi_path: str, wav_path: str) -> str:
    if render_with_fluidsynth(midi_path, wav_path):
        return wav_path
    return render_with_sine_fallback(midi_path, wav_path)
