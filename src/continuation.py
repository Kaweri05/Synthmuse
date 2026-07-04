"""
'Finish My Melody' feature.

Lets a user upload their own short MIDI idea (hummed into a MIDI
keyboard app, exported from a DAW, etc.) and have SynthMuse analyze
its note sequence and continue composing in a matching style, instead of
only ever generating from a random or fully pre-baked seed. This is the
main "different from other related projects" feature alongside the
Emotion Engine and Style Fusion.
"""

from .data_preprocessing import extract_notes_from_midi
from .generator import MusicGenerator
from . import config


def continue_midi(generator: MusicGenerator, uploaded_midi_path: str,
                   extra_length: int = config.DEFAULT_GENERATION_LENGTH,
                   mood: str = config.DEFAULT_EMOTION,
                   sequence_length: int = config.SEQUENCE_LENGTH):
    """Analyze an uploaded MIDI file and generate a continuation.

    Returns (full_note_sequence, original_notes, generated_notes).
    """
    original_notes = extract_notes_from_midi(uploaded_midi_path)

    if not original_notes:
        raise ValueError("No notes could be extracted from the uploaded MIDI file.")

    seed_pattern = generator.seed_from_notes(original_notes, sequence_length=sequence_length)
    generated_notes = generator.generate_notes(seed_pattern, length=extra_length, mood=mood)

    full_sequence = original_notes + generated_notes
    return full_sequence, original_notes, generated_notes
