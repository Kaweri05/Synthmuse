"""
Sheet-music export.

Converts a generated music21 stream into a MusicXML file that can be
opened in MuseScore, Finale, Sibelius, or any notation app — giving
musicians (not just developers) a usable output format, which most
LSTM-MIDI-generator demo projects skip entirely.
"""

from music21 import stream


def export_musicxml(midi_stream: "stream.Stream", output_path: str) -> str:
    midi_stream.write("musicxml", fp=output_path)
    return output_path
