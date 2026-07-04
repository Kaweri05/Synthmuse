"""
'My Library' — persistent history of every track SynthMuse generates.

Each entry's audio/MIDI files are copied into outputs/history/files/ and
described in a single JSON index (outputs/history/index.json), so the
library survives across app restarts, similar to a podcast app's
Downloads/History screens.
"""

import json
import os
import shutil
import uuid
from datetime import datetime

from . import config


def _ensure_dirs():
    os.makedirs(config.HISTORY_FILES_DIR, exist_ok=True)


def _load_index():
    _ensure_dirs()
    if not os.path.exists(config.HISTORY_INDEX_PATH):
        return []
    try:
        with open(config.HISTORY_INDEX_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_index(entries):
    _ensure_dirs()
    with open(config.HISTORY_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def add_entry(feature: str, mood: str, midi_path: str, wav_path: str = None,
              note_count: int = 0, extra: dict = None):
    """Save a generated track into the library.

    feature: one of 'mood', 'continuation', 'fusion'
    Returns the new entry dict.
    """
    _ensure_dirs()
    entry_id = uuid.uuid4().hex[:12]
    entry_dir = os.path.join(config.HISTORY_FILES_DIR, entry_id)
    os.makedirs(entry_dir, exist_ok=True)

    stored_midi = os.path.join(entry_dir, "track.mid")
    shutil.copy(midi_path, stored_midi)

    stored_wav = None
    if wav_path and os.path.exists(wav_path):
        stored_wav = os.path.join(entry_dir, "track.wav")
        shutil.copy(wav_path, stored_wav)

    entry = {
        "id": entry_id,
        "feature": feature,
        "mood": mood,
        "note_count": note_count,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "midi_path": stored_midi,
        "wav_path": stored_wav,
        "extra": extra or {},
    }

    entries = _load_index()
    entries.insert(0, entry)  # newest first
    _save_index(entries)
    return entry


def list_entries(feature_filter: str = "all"):
    entries = _load_index()
    if feature_filter and feature_filter != "all":
        entries = [e for e in entries if e.get("feature") == feature_filter]
    return entries


def delete_entry(entry_id: str):
    entries = _load_index()
    remaining = [e for e in entries if e["id"] != entry_id]
    entry_dir = os.path.join(config.HISTORY_FILES_DIR, entry_id)
    if os.path.isdir(entry_dir):
        shutil.rmtree(entry_dir, ignore_errors=True)
    _save_index(remaining)


def clear_all():
    _save_index([])
    if os.path.isdir(config.HISTORY_FILES_DIR):
        shutil.rmtree(config.HISTORY_FILES_DIR, ignore_errors=True)
    os.makedirs(config.HISTORY_FILES_DIR, exist_ok=True)
