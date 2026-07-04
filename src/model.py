"""
LSTM model architecture used by SynthMuse.

The network stacks bidirectional-friendly LSTM layers with dropout and
batch normalization, then a dense softmax head over the note vocabulary.
Kept intentionally close to the well-tested "Classical Piano Composer"
style architecture, since the goal of this project is the surrounding
product experience (mood control, style fusion, continuation, visualizer)
rather than reinventing the generative core.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
    Activation,
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from . import config


def build_model(n_vocab: int, sequence_length: int = config.SEQUENCE_LENGTH):
    model = Sequential(name="SynthMuse_LSTM")

    model.add(LSTM(
        config.LSTM_UNITS,
        input_shape=(sequence_length, 1),
        return_sequences=True,
    ))
    model.add(Dropout(config.DROPOUT_RATE))

    model.add(LSTM(config.LSTM_UNITS, return_sequences=True))
    model.add(Dropout(config.DROPOUT_RATE))

    model.add(LSTM(config.LSTM_UNITS))
    model.add(BatchNormalization())
    model.add(Dropout(config.DROPOUT_RATE))

    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(config.DROPOUT_RATE))

    model.add(Dense(n_vocab))
    model.add(Activation("softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    return model


def default_callbacks(checkpoint_path: str = config.MODEL_PATH):
    return [
        ModelCheckpoint(
            checkpoint_path,
            monitor="loss",
            verbose=1,
            save_best_only=True,
            mode="min",
        ),
        EarlyStopping(monitor="loss", patience=8, restore_best_weights=True),
    ]
