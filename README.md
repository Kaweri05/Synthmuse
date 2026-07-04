# 🎧 SynthMuse

**An emotion-aware AI music composer** — turn a mood, a hummed melody
fragment, or two different musical styles into an original, playable,
downloadable composition. Built on an LSTM generative core (TensorFlow /
Keras + music21) and wrapped in a Streamlit studio interface.

> Formerly `Music_Gener_AI`. Renamed and rebuilt into **SynthMuse** with
> a proper project structure and three new features not found in most
> LSTM-MIDI-generator tutorials.

---

## ✨ What makes it different

Most open-source LSTM music generators do exactly one thing: sample a
trained model from a random seed and export a `.mid` file. SynthMuse
keeps that solid generative core but adds:

| Feature | What it does |
|---|---|
| 🧠 **Emotion Engine** | Pick a mood (or type a sentence) and the app maps it to real musical controls — scale/mode, tempo, note density, and sampling temperature — instead of leaving the model to guess. |
| ✍️ **Finish My Melody** | Upload your own short MIDI idea and SynthMuse analyzes it and composes a continuation in a matching style. |
| 🧬 **Music DNA Mixer** | Upload two reference MIDI files and blend them (with an adjustable ratio) into one hybrid seed, producing a genuinely new hybrid piece — not just two songs concatenated. |
| 🎹 **Live Piano-Roll Visualizer** | See the generated melody as an interactive, mood-colored piano roll, not just an audio player. |
| 🔊 **In-Browser Playback** | MIDI is rendered to WAV automatically (FluidSynth if available, a pure-Python synth fallback otherwise) — no external MIDI player needed. |
| 🎼 **Sheet Music Export** | One-click MusicXML export so musicians can open the piece in MuseScore, Finale, or Sibelius. |

---

## 📂 Project Structure

```
SynthMuse/
├── app.py                     # Streamlit studio UI (entry point)
├── requirements.txt
├── runtime.txt
├── .streamlit/config.toml     # dark, futuristic theme
├── src/
│   ├── config.py              # paths, hyperparameters, emotion profiles
│   ├── data_preprocessing.py  # MIDI -> note corpus -> training sequences
│   ├── model.py                # LSTM architecture
│   ├── train.py                # training entry point (python -m src.train)
│   ├── generator.py            # sampling, seeding, blending, MIDI rendering
│   ├── emotion_engine.py       # mood <-> musical parameter mapping + text sentiment
│   ├── continuation.py         # "Finish My Melody" feature
│   ├── style_fusion.py         # "Music DNA Mixer" feature
│   ├── visualizer.py           # Plotly piano-roll rendering
│   ├── audio_render.py         # MIDI -> WAV (FluidSynth or sine fallback)
│   └── sheet_music.py          # MusicXML export
├── models/                    # music_model.h5 + note_mappings.pkl (gitignored)
├── data/midi_samples/         # training MIDI files go here
├── notebooks/                 # original training notebook
├── outputs/                   # generated files land here at runtime
└── assets/screenshots/
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Kaweri05/SynthMuse.git
cd SynthMuse
pip install -r requirements.txt
```

> Optional, for higher-fidelity audio playback: install
> [FluidSynth](https://www.fluidsynth.org/) and a General MIDI soundfont
> (e.g. `FluidR3_GM.sf2`). Without it, SynthMuse automatically falls
> back to a built-in sine-wave synthesizer so playback still works.

### Bring your trained model

Copy your existing `music_model.h5` (or retrain, see below) and
`notes.pkl`/`note_mappings.pkl` into `models/`. If you only have the
original `notes.pkl`, run:

```bash
python -m src.data_preprocessing   # rebuilds notes.pkl + note_mappings.pkl
```

### Or train from scratch

Drop `.mid` files into `data/midi_samples/`, then:

```bash
python -m src.train --epochs 100 --batch-size 64
```

### Run the app

```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow / Keras (stacked LSTM)
- **Music parsing/rendering:** music21, pretty_midi
- **UI:** Streamlit
- **Visualization:** Plotly (interactive piano roll)

---

## 📊 How it works

1. MIDI files are parsed into note/chord sequences (`data_preprocessing.py`).
2. A stacked LSTM learns to predict the next note given the previous
   `SEQUENCE_LENGTH` notes (`model.py`, `train.py`).
3. At generation time, the **Emotion Engine** converts a mood (picked
   directly or inferred from free text) into concrete musical parameters,
   and the **generator** applies them: re-keying notes into the right
   scale, scaling tempo, adjusting note density, and biasing the
   temperature of the softmax sampling.
4. Optional inputs — an uploaded melody (**Finish My Melody**) or two
   uploaded reference tracks (**Music DNA Mixer**) — are converted into
   integer seed patterns that steer generation instead of a random seed.
5. Output is rendered to MIDI, WAV (for in-browser playback), an
   interactive piano-roll chart, and optionally MusicXML sheet music.

---

## 🎯 Applications

- AI-assisted composition and songwriting sketches
- Mood-based background music for games, videos, or apps
- Music education / algorithmic composition demos
- Rapid style exploration by blending reference tracks

---

## 🔮 Future Enhancements

- Multi-instrument / full orchestration (auto-generate bass, strings, drums layers)
- Real-time streaming generation (jam-along mode)
- User accounts with a generation history and community gallery
- Fine-tuning per-user on their own uploaded MIDI library

---

## 👨‍💻 Author

**Kaweri Harinkhede** — Computer Engineering Student, passionate about
AI, Machine Learning, and Web Development.

---

## 📜 License

This project is developed for educational and learning purposes. See
[LICENSE](LICENSE) (MIT).
