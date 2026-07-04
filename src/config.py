"""
Central configuration for SynthMuse.
Keeping all tunable constants in one place makes the rest of the
codebase easier to read and to extend with new features.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
MIDI_SAMPLES_DIR = os.path.join(DATA_DIR, "midi_samples")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

MODEL_PATH = os.path.join(MODELS_DIR, "music_model.h5")
NOTES_PATH = os.path.join(MODELS_DIR, "notes.pkl")
MAPPINGS_PATH = os.path.join(MODELS_DIR, "note_mappings.pkl")

# "My Library" — persistent history of generated tracks
HISTORY_DIR = os.path.join(OUTPUTS_DIR, "history")
HISTORY_FILES_DIR = os.path.join(HISTORY_DIR, "files")
HISTORY_INDEX_PATH = os.path.join(HISTORY_DIR, "index.json")

# ---------------------------------------------------------------------------
# Model / training hyperparameters
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = 100          # number of previous notes fed to the LSTM
LSTM_UNITS = 512
DROPOUT_RATE = 0.3
BATCH_SIZE = 64
EPOCHS = 100
VALIDATION_SPLIT = 0.1

# ---------------------------------------------------------------------------
# Generation defaults
# ---------------------------------------------------------------------------
DEFAULT_GENERATION_LENGTH = 200      # number of notes to generate
DEFAULT_TEMPERATURE = 1.0            # sampling temperature (creativity)
MIN_TEMPERATURE = 0.2
MAX_TEMPERATURE = 2.0

# ---------------------------------------------------------------------------
# Emotion engine: maps a mood label to musical parameters
# (scale, tempo multiplier, octave shift, note density, temperature bias)
# This is the core of the "Mood-to-Music" feature that differentiates
# SynthMuse from plain seed -> LSTM -> MIDI generators.
# ---------------------------------------------------------------------------
EMOTION_PROFILES = {
    "happy": {
        "scale": "major",
        "tempo_multiplier": 1.25,
        "octave_shift": 1,
        "note_density": 1.2,
        "temperature": 1.1,
        "color": "#FFC93C",
    },
    "sad": {
        "scale": "minor",
        "tempo_multiplier": 0.75,
        "octave_shift": -1,
        "note_density": 0.7,
        "temperature": 0.8,
        "color": "#4C6EF5",
    },
    "energetic": {
        "scale": "mixolydian",
        "tempo_multiplier": 1.5,
        "octave_shift": 1,
        "note_density": 1.5,
        "temperature": 1.3,
        "color": "#FF4D4D",
    },
    "calm": {
        "scale": "dorian",
        "tempo_multiplier": 0.65,
        "octave_shift": 0,
        "note_density": 0.6,
        "temperature": 0.6,
        "color": "#38D9A9",
    },
    "mysterious": {
        "scale": "phrygian",
        "tempo_multiplier": 0.9,
        "octave_shift": -1,
        "note_density": 0.9,
        "temperature": 1.4,
        "color": "#9775FA",
    },
    "romantic": {
        "scale": "lydian",
        "tempo_multiplier": 0.85,
        "octave_shift": 0,
        "note_density": 0.8,
        "temperature": 0.9,
        "color": "#F783AC",
    },
}

DEFAULT_EMOTION = "calm"

# ---------------------------------------------------------------------------
# Audio rendering
# ---------------------------------------------------------------------------
SOUNDFONT_CANDIDATES = [
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
]
DEFAULT_SAMPLE_RATE = 44100
