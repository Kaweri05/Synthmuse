"""
Music DNA Mixer — blend two source styles (e.g. a classical piece and a
jazz piece, or two of the user's own MIDI uploads) into one hybrid seed,
then let the LSTM continue from that hybrid pattern.

This is a second differentiator vs. the average "single seed -> LSTM"
generator: instead of one input shaping the whole piece, two inputs are
interleaved according to a user-controlled ratio, so the same model
checkpoint can produce a spectrum of hybrids between any two references.
"""

from .data_preprocessing import extract_notes_from_midi
from .generator import MusicGenerator
from . import config


def fuse_and_generate(generator: MusicGenerator, midi_path_a: str, midi_path_b: str,
                       ratio: float = 0.5, length: int = config.DEFAULT_GENERATION_LENGTH,
                       mood: str = config.DEFAULT_EMOTION,
                       sequence_length: int = config.SEQUENCE_LENGTH):
    """Blend two MIDI files' note patterns and generate a hybrid piece.

    ratio: 0.0 -> output leans fully towards style A, 1.0 -> fully towards B.
    Returns (generated_notes, seed_a_notes, seed_b_notes).
    """
    notes_a = extract_notes_from_midi(midi_path_a)
    notes_b = extract_notes_from_midi(midi_path_b)

    if not notes_a or not notes_b:
        raise ValueError("Could not extract notes from one of the two MIDI files.")

    seed_a = generator.seed_from_notes(notes_a, sequence_length=sequence_length)
    seed_b = generator.seed_from_notes(notes_b, sequence_length=sequence_length)

    blended_seed = generator.blend_seeds(seed_a, seed_b, ratio=ratio)
    generated_notes = generator.generate_notes(blended_seed, length=length, mood=mood)

    return generated_notes, notes_a, notes_b
