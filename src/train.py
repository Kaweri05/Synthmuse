"""
Training entry point.

Usage:
    python -m src.train

Reads MIDI files from data/midi_samples, builds the vocabulary and
training sequences, trains the LSTM defined in model.py, and saves the
resulting weights + note mappings into the models/ folder so app.py can
load them.
"""

import argparse

from . import config
from .data_preprocessing import (
    build_corpus,
    build_vocabulary,
    prepare_sequences,
    save_artifacts,
)
from .model import build_model, default_callbacks


def main():
    parser = argparse.ArgumentParser(description="Train the SynthMuse LSTM model.")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--sequence-length", type=int, default=config.SEQUENCE_LENGTH)
    args = parser.parse_args()

    print("[train] Building corpus from MIDI files...")
    all_notes = build_corpus()

    print("[train] Building vocabulary...")
    vocab, note_to_int, int_to_note = build_vocabulary(all_notes)
    save_artifacts(all_notes, note_to_int, int_to_note)
    print(f"[train] Vocabulary size: {len(vocab)}")

    print("[train] Preparing training sequences...")
    X, y, _ = prepare_sequences(all_notes, note_to_int, args.sequence_length)
    print(f"[train] Training samples: {X.shape[0]}")

    print("[train] Building model...")
    model = build_model(n_vocab=len(vocab), sequence_length=args.sequence_length)
    model.summary()

    print("[train] Training...")
    model.fit(
        X,
        y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=default_callbacks(),
    )

    model.save(config.MODEL_PATH)
    print(f"[train] Model saved to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
