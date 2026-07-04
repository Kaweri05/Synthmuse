# Migrating from Music_Gener_AI → SynthMuse

This restructure keeps your original trained model reusable — nothing
needs to be retrained unless you want to.

## 1. Rename the repository on GitHub
`Settings → General → Repository name` → change `Music_Gener_AI` to
`SynthMuse` (GitHub automatically redirects the old URL).

## 2. Copy your existing files into the new structure

| Old file | New location |
|---|---|
| `app.py` | replaced by the new `app.py` (Streamlit multi-tab studio) |
| `music_model.h5` | `models/music_model.h5` |
| `notes.pkl` | `models/notes.pkl` — then run `python -m src.data_preprocessing` once to also produce `models/note_mappings.pkl` (the new code expects note<->int mappings saved separately) |
| `Rothchild Symphony Rmw12 2mov.mid` | `data/midi_samples/Rothchild Symphony Rmw12 2mov.mid` |
| `Music_Gener_AI.ipynb` | `notebooks/synthmuse_training_exploration.ipynb` |
| `requirements.txt`, `runtime.txt` | replaced by the updated versions in this project |

## 3. Regenerate note_mappings.pkl from your existing notes.pkl

If you don't want to re-run preprocessing from raw MIDI, you can build
just the mapping file from your existing `notes.pkl`:

```python
import pickle
from src import config

with open(config.NOTES_PATH, "rb") as f:
    all_notes = pickle.load(f)

vocabulary = sorted(set(all_notes))
note_to_int = {n: i for i, n in enumerate(vocabulary)}
int_to_note = {i: n for i, n in enumerate(vocabulary)}

with open(config.MAPPINGS_PATH, "wb") as f:
    pickle.dump({"note_to_int": note_to_int, "int_to_note": int_to_note}, f)
```

> ⚠️ Important: the vocabulary order must match what your `music_model.h5`
> was trained with (the output layer size = vocabulary size). If your old
> code built `note_to_int` the same way (`sorted(set(notes))`), this will
> match exactly. If unsure, it's safest to retrain with `src/train.py`.

## 4. Install & run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 5. Push to GitHub

```bash
git add .
git commit -m "Restructure into SynthMuse with mood, continuation, and style-fusion features"
git push
```
