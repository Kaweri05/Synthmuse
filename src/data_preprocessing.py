"""
Data preprocessing utilities.

Responsible for turning a folder of raw MIDI files into the note/chord
sequences the LSTM is trained on, and for building the integer <-> note
vocabulary mappings used everywhere else in the project.
"""

import glob
import pickle
from collections import Counter

import numpy as np
from music21 import converter, instrument, note, chord

from . import config


def extract_notes_from_midi(midi_path: str):
    """Parse a single MIDI file and return a list of note/chord strings.

    Reads across *all* instrument parts (not just the first), since many
    MIDI files spread the actual melodic content across several tracks
    and leave earlier tracks (e.g. tempo/control tracks) empty.
    """
    notes = []
    midi = converter.parse(midi_path)

    try:
        parts = instrument.partitionByInstrument(midi)
    except Exception:
        parts = None

    if parts and len(parts.parts) > 0:
        elements = []
        for part in parts.parts:
            elements.extend(part.recurse().notes)
    else:
        elements = midi.flatten().notes

    for element in elements:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append(".".join(str(n) for n in element.normalOrder))

    return notes


def build_corpus(midi_folder: str = config.MIDI_SAMPLES_DIR):
    """Walk every MIDI file in `midi_folder` and build the full note corpus."""
    all_notes = []
    midi_files = glob.glob(f"{midi_folder}/*.mid") + glob.glob(f"{midi_folder}/*.midi")

    if not midi_files:
        raise FileNotFoundError(
            f"No MIDI files found in {midi_folder}. Add training data there "
            "before running preprocessing."
        )

    for path in midi_files:
        try:
            all_notes.extend(extract_notes_from_midi(path))
        except Exception as exc:
            print(f"[preprocessing] Skipped {path}: {exc}")

    return all_notes


def build_vocabulary(all_notes):
    """Create sorted vocabulary + note<->int mapping dictionaries."""
    vocabulary = sorted(set(all_notes))
    note_to_int = {n: i for i, n in enumerate(vocabulary)}
    int_to_note = {i: n for i, n in enumerate(vocabulary)}
    return vocabulary, note_to_int, int_to_note


def prepare_sequences(all_notes, note_to_int, sequence_length=config.SEQUENCE_LENGTH):
    """Turn the flat note corpus into (X, y) training sequences."""
    network_input = []
    network_output = []

    for i in range(len(all_notes) - sequence_length):
        seq_in = all_notes[i:i + sequence_length]
        seq_out = all_notes[i + sequence_length]
        network_input.append([note_to_int[n] for n in seq_in])
        network_output.append(note_to_int[seq_out])

    n_vocab = len(note_to_int)
    X = np.reshape(network_input, (len(network_input), sequence_length, 1))
    X = X / float(n_vocab)

    from tensorflow.keras.utils import to_categorical
    y = to_categorical(network_output, num_classes=n_vocab)

    return X, y, network_input


def save_artifacts(all_notes, note_to_int, int_to_note):
    with open(config.NOTES_PATH, "wb") as f:
        pickle.dump(all_notes, f)
    with open(config.MAPPINGS_PATH, "wb") as f:
        pickle.dump({"note_to_int": note_to_int, "int_to_note": int_to_note}, f)


def note_frequency_report(all_notes, top_n=20):
    """Small helper used by the Streamlit dashboard to show corpus stats."""
    return Counter(all_notes).most_common(top_n)


if __name__ == "__main__":
    notes = build_corpus()
    vocab, n2i, i2n = build_vocabulary(notes)
    save_artifacts(notes, n2i, i2n)
    print(f"Processed {len(notes)} notes across vocabulary of {len(vocab)} symbols.")
