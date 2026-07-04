"""
Piano-roll visualizer.

Turns a generated note sequence into an interactive Plotly piano-roll
chart so users can *see* the shape of the melody (pitch over time,
colored by mood) before or while listening to it — most LSTM music
generator demos only offer an audio player and a raw MIDI download.
"""

import plotly.graph_objects as go
from music21 import note, chord

from .emotion_engine import get_profile


def _symbol_to_pitches(symbol: str):
    if "." in symbol or symbol.isdigit():
        return [int(pc) % 12 + 60 for pc in symbol.split(".")]
    try:
        return [note.Note(symbol).pitch.midi]
    except Exception:
        return []


def build_piano_roll(note_strings, mood: str = "calm", step_duration: float = 0.5):
    profile = get_profile(mood)
    color = profile.get("color", "#4C6EF5")

    fig = go.Figure()
    x = []
    y = []

    for i, symbol in enumerate(note_strings):
        pitches = _symbol_to_pitches(symbol)
        for p in pitches:
            x.extend([i * step_duration, (i + 1) * step_duration, None])
            y.extend([p, p, None])

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines",
        line=dict(color=color, width=6),
        hoverinfo="skip",
        showlegend=False,
    ))

    fig.update_layout(
        title=f"Piano Roll — {mood.capitalize()} mood",
        xaxis_title="Time (beats)",
        yaxis_title="MIDI pitch",
        template="plotly_dark",
        height=420,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig
