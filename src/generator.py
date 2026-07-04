"""
Core generation engine.

Wraps the trained LSTM and turns raw next-note predictions into a MIDI
stream, applying the Emotion Engine's musical parameters (scale,
tempo, octave shift, density, temperature) so the same model can output
audibly different music depending on requested mood — instead of every
generation sounding like a random walk through the training set.
"""

import pickle
import random

import numpy as np
from music21 import stream, note, chord, tempo as m21tempo, instrument as m21instrument

from . import config
from .emotion_engine import get_profile, nearest_scale_pitch


class MusicGenerator:
    def __init__(self, model=None, note_to_int=None, int_to_note=None):
        self.model = model
        self.note_to_int = note_to_int
        self.int_to_note = int_to_note
        self.n_vocab = len(int_to_note) if int_to_note else 0

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, model_path=config.MODEL_PATH, mappings_path=config.MAPPINGS_PATH):
        from tensorflow.keras.models import load_model

        model = load_model(model_path)
        with open(mappings_path, "rb") as f:
            mappings = pickle.load(f)
        return cls(model=model, note_to_int=mappings["note_to_int"], int_to_note=mappings["int_to_note"])

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    @staticmethod
    def _sample_with_temperature(probabilities, temperature):
        temperature = max(config.MIN_TEMPERATURE, min(config.MAX_TEMPERATURE, temperature))
        probabilities = np.asarray(probabilities).astype("float64")
        probabilities = np.log(probabilities + 1e-9) / temperature
        exp_probs = np.exp(probabilities)
        probabilities = exp_probs / np.sum(exp_probs)
        return np.random.choice(len(probabilities), p=probabilities)

    def _predict_next(self, pattern, temperature):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(self.n_vocab)
        probabilities = self.model.predict(prediction_input, verbose=0)[0]
        index = self._sample_with_temperature(probabilities, temperature)
        return index

    # ------------------------------------------------------------------
    # Seed handling
    # ------------------------------------------------------------------
    def random_seed(self, sequence_length=config.SEQUENCE_LENGTH):
        return [random.randrange(self.n_vocab) for _ in range(sequence_length)]

    def seed_from_notes(self, note_strings, sequence_length=config.SEQUENCE_LENGTH):
        """Turn a list of note strings (e.g. extracted from an uploaded MIDI
        for the 'continue my melody' feature) into an integer seed pattern,
        padding/truncating and mapping unseen notes to the closest known
        vocabulary entry.
        """
        ints = []
        for n in note_strings[-sequence_length:]:
            if n in self.note_to_int:
                ints.append(self.note_to_int[n])
            else:
                ints.append(random.randrange(self.n_vocab))

        if len(ints) < sequence_length:
            pad = [random.randrange(self.n_vocab) for _ in range(sequence_length - len(ints))]
            ints = pad + ints

        return ints

    def blend_seeds(self, seed_a, seed_b, ratio=0.5):
        """Style-Fusion / 'Music DNA Mixer' feature: interleave two seed
        patterns according to `ratio` (0 = all A, 1 = all B) to produce a
        hybrid starting pattern that carries traits of both source styles.
        """
        length = min(len(seed_a), len(seed_b))
        blended = []
        for i in range(length):
            blended.append(seed_b[i] if random.random() < ratio else seed_a[i])
        return blended

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate_notes(self, seed_pattern, length=config.DEFAULT_GENERATION_LENGTH, mood=config.DEFAULT_EMOTION):
        profile = get_profile(mood)
        temperature = profile["temperature"]

        pattern = list(seed_pattern)
        generated_ints = []

        # note density controls how many of the raw predictions we keep
        # vs. skip, which changes how "busy" the resulting piece feels.
        density = profile["note_density"]

        for _ in range(length):
            index = self._predict_next(pattern, temperature)
            if random.random() <= min(density, 1.0) or density > 1.0:
                generated_ints.append(index)
                extra = 1 if density <= 1.0 else int(density)
                for _ in range(max(extra - 1, 0)):
                    generated_ints.append(self._predict_next(pattern, temperature))

            pattern.append(index)
            pattern = pattern[1:]

        return [self.int_to_note[i] for i in generated_ints if i in self.int_to_note]

    # ------------------------------------------------------------------
    # MIDI rendering
    # ------------------------------------------------------------------
    def notes_to_midi_stream(self, note_strings, mood=config.DEFAULT_EMOTION, base_tempo=120):
        profile = get_profile(mood)
        octave_shift = profile["octave_shift"]
        scale = profile["scale"]

        output_stream = stream.Stream()
        output_stream.append(m21instrument.Piano())
        output_stream.append(m21tempo.MetronomeMark(number=base_tempo * profile["tempo_multiplier"]))

        offset = 0
        for symbol in note_strings:
            try:
                if "." in symbol or symbol.isdigit():
                    chord_notes = []
                    for pc in symbol.split("."):
                        pitch_class = nearest_scale_pitch(int(pc) % 12, scale)
                        n = note.Note(pitch_class)
                        n.octave = 4 + octave_shift
                        chord_notes.append(n)
                    new_chord = chord.Chord(chord_notes)
                    new_chord.offset = offset
                    output_stream.append(new_chord)
                else:
                    n = note.Note(symbol)
                    n.pitch.octave = (n.pitch.octave or 4) + octave_shift
                    n.offset = offset
                    output_stream.append(n)
            except Exception:
                continue
            offset += 0.5

        return output_stream

    def generate_midi_file(self, output_path, seed_pattern=None, length=config.DEFAULT_GENERATION_LENGTH,
                            mood=config.DEFAULT_EMOTION, base_tempo=120):
        if seed_pattern is None:
            seed_pattern = self.random_seed()

        note_strings = self.generate_notes(seed_pattern, length=length, mood=mood)
        midi_stream = self.notes_to_midi_stream(note_strings, mood=mood, base_tempo=base_tempo)
        midi_stream.write("midi", fp=output_path)
        return output_path, note_strings
