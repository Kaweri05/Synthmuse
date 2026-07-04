"""
Emotion Engine — the flagship feature that sets SynthMuse apart from a
plain "seed -> LSTM -> MIDI" generator.

A user can either:
  1. Pick a mood directly ("happy", "sad", "energetic", ...), or
  2. Type a free-text sentence / journal entry / caption, which is run
     through a lightweight sentiment + keyword analysis to infer a mood.

The inferred mood is mapped (via config.EMOTION_PROFILES) to concrete
musical control knobs: scale/mode, tempo multiplier, octave shift, note
density and sampling temperature. Those knobs are consumed by
generator.py to steer the LSTM's output rather than just sampling it
blindly, and are also used to re-key generated notes into the right
scale so the mood is audibly present, not just theoretical.
"""

import re
from typing import Dict

from . import config

# A tiny, dependency-free sentiment lexicon so the feature works even
# without downloading NLTK/vader corpora (handy for offline / low-resource
# deployments such as Streamlit Community Cloud).
_POSITIVE_WORDS = {
    "happy", "joy", "joyful", "excited", "love", "great", "amazing",
    "wonderful", "bright", "fun", "cheerful", "sunny", "delighted",
    "victory", "win", "hopeful", "peaceful", "calm", "relaxed", "serene",
}
_NEGATIVE_WORDS = {
    "sad", "cry", "crying", "lonely", "lost", "grief", "depressed",
    "dark", "pain", "hurt", "broken", "tired", "angry", "fear",
    "afraid", "anxious", "melancholy", "sorrow", "heartbroken",
}
_ENERGETIC_WORDS = {
    "run", "fast", "power", "energy", "party", "dance", "fight",
    "adrenaline", "hype", "intense", "storm", "explosive", "wild",
}
_MYSTERIOUS_WORDS = {
    "mystery", "secret", "shadow", "night", "unknown", "strange",
    "eerie", "haunted", "fog", "whisper", "hidden",
}
_ROMANTIC_WORDS = {
    "love", "romance", "heart", "kiss", "together", "forever",
    "soulmate", "valentine", "embrace",
}

# Musical scale intervals (semitones from root), used to re-key notes.
SCALE_INTERVALS: Dict[str, list] = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
}


def infer_mood_from_text(text: str) -> str:
    """Very small rule-based sentiment classifier returning a mood label
    that matches a key in config.EMOTION_PROFILES.
    """
    words = re.findall(r"[a-zA-Z']+", text.lower())
    word_set = set(words)

    scores = {
        "happy": len(word_set & _POSITIVE_WORDS),
        "sad": len(word_set & _NEGATIVE_WORDS),
        "energetic": len(word_set & _ENERGETIC_WORDS),
        "mysterious": len(word_set & _MYSTERIOUS_WORDS),
        "romantic": len(word_set & _ROMANTIC_WORDS),
    }

    best_mood = max(scores, key=scores.get)
    if scores[best_mood] == 0:
        return config.DEFAULT_EMOTION
    return best_mood


def get_profile(mood: str) -> dict:
    return config.EMOTION_PROFILES.get(mood, config.EMOTION_PROFILES[config.DEFAULT_EMOTION])


def nearest_scale_pitch(pitch_class: int, scale: str) -> int:
    """Snap a pitch class (0-11) to the nearest note within `scale`."""
    intervals = SCALE_INTERVALS.get(scale, SCALE_INTERVALS["major"])
    return min(intervals, key=lambda i: min(abs(i - pitch_class), 12 - abs(i - pitch_class)))


def available_moods():
    return list(config.EMOTION_PROFILES.keys())
