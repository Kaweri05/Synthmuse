"""
SynthMuse — Streamlit application.

A mood-aware AI music composition studio built on an LSTM core,
extended with:
  * Emotion Engine        — mood picker or free-text -> musical control knobs
  * Finish My Melody      — upload a MIDI idea, AI continues it
  * Music DNA Mixer       — blend two MIDI styles into one hybrid piece
  * My Library            — persistent history of every track you generate
  * Live Piano-Roll       — see the generated melody, not just hear it
  * In-browser playback   — MIDI rendered to WAV, no extra app required
  * Sheet music export    — MusicXML download for musicians
"""

import os
import tempfile

import streamlit as st

from src import config
from src import history
from src.generator import MusicGenerator
from src.emotion_engine import infer_mood_from_text, available_moods, get_profile
from src.continuation import continue_midi
from src.style_fusion import fuse_and_generate
from src.visualizer import build_piano_roll
from src.audio_render import render_midi_to_wav
from src.sheet_music import export_musicxml

st.set_page_config(
    page_title="SynthMuse",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Style — purple / card-based theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background: #F5F3FF; }
    h1, h2, h3 { color: #241C3D; }

    /* top pill tag */
    .synthmuse-tag {
        display: inline-block; padding: 5px 14px; border-radius: 999px;
        background: linear-gradient(90deg,#8B5CF6,#6D28D9); color:#ffffff;
        font-weight:600; font-size:0.78rem; margin-bottom: 10px;
        letter-spacing: 0.03em;
    }

    /* hero banner, echoes the reference app's "Enjoy All Benefits" card */
    .synthmuse-hero {
        background: linear-gradient(120deg,#8B5CF6 0%, #6D28D9 100%);
        border-radius: 22px; padding: 22px 26px; color: white;
        margin-bottom: 22px; box-shadow: 0 10px 30px rgba(109,40,217,0.25);
    }
    .synthmuse-hero h2 { color: white; margin: 0 0 6px 0; }
    .synthmuse-hero p { color: #EDE9FE; margin: 0; }

    /* generic white rounded card, matches reference list rows */
    .lib-card {
        background: #FFFFFF; border-radius: 18px; padding: 16px 18px;
        margin-bottom: 12px; box-shadow: 0 4px 16px rgba(109,40,217,0.08);
        display: flex; align-items: center; gap: 14px;
    }
    .lib-thumb {
        width: 46px; height: 46px; border-radius: 12px; flex-shrink: 0;
        display: flex; align-items: center; justify-content: center;
        font-size: 20px; color: white;
    }
    .lib-title { font-weight: 600; color: #241C3D; font-size: 0.95rem; margin: 0; }
    .lib-sub { color: #7A7288; font-size: 0.8rem; margin: 2px 0 0 0; }
    .lib-badge {
        display:inline-block; padding: 2px 10px; border-radius: 999px;
        background:#F3E8FF; color:#6D28D9; font-size:0.72rem; font-weight:600;
        margin-left: 8px;
    }

    /* buttons: rounded pill, purple filled, matches reference CTA buttons */
    .stButton > button, .stDownloadButton > button {
        border-radius: 999px !important; border: none !important;
        background: linear-gradient(90deg,#8B5CF6,#6D28D9) !important;
        color: white !important; font-weight: 600 !important;
        padding: 0.5rem 1.4rem !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        opacity: 0.92;
    }

    /* tabs styled as pill nav, echoes the bottom nav bar look */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background: #FFFFFF; border-radius: 999px; padding: 8px 18px;
        color: #6D28D9; font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg,#8B5CF6,#6D28D9) !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<span class="synthmuse-tag">LSTM • MOOD-AWARE • MULTI-MODAL</span>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="synthmuse-hero">
        <h2>🎵 SynthMuse</h2>
        <p>Turn an emotion, a melody fragment, or two different musical styles into
        an original, playable composition — then keep every track in your library.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the trained model...")
def load_generator():
    if not os.path.exists(config.MODEL_PATH) or not os.path.exists(config.MAPPINGS_PATH):
        return None
    return MusicGenerator.load()


generator = load_generator()

if generator is None:
    st.warning(
        "No trained model found in `models/`. Add `music_model.h5` and "
        "`note_mappings.pkl` (produced by `python -m src.train`), or drop "
        "MIDI training files into `data/midi_samples/` and train first."
    )
    st.stop()


def _mood_color(mood: str) -> str:
    return get_profile(mood).get("color", "#8B5CF6")


def _save_to_library(feature, mood, midi_path, wav_path, note_count, extra=None):
    try:
        history.add_entry(
            feature=feature, mood=mood, midi_path=midi_path, wav_path=wav_path,
            note_count=note_count, extra=extra or {},
        )
    except Exception as exc:
        st.warning(f"Track generated, but could not be saved to your library: {exc}")


tab_compose, tab_continue, tab_fusion, tab_library, tab_about = st.tabs(
    ["🎼 Compose by Mood", "✍️ Finish My Melody", "🧬 Music DNA Mixer", "📚 My Library", "ℹ️ About"]
)

# ---------------------------------------------------------------------------
# Tab 1: Compose by mood
# ---------------------------------------------------------------------------
with tab_compose:
    st.subheader("Generate music from an emotion")
    col1, col2 = st.columns([2, 1])

    with col1:
        input_mode = st.radio("How do you want to set the mood?", ["Pick a mood", "Describe it in words"], horizontal=True)

        if input_mode == "Pick a mood":
            mood = st.selectbox("Mood", available_moods(), index=available_moods().index(config.DEFAULT_EMOTION))
        else:
            text = st.text_area("Describe how the music should feel", placeholder="e.g. 'a quiet rainy evening, a little lonely but peaceful'")
            mood = infer_mood_from_text(text) if text.strip() else config.DEFAULT_EMOTION
            if text.strip():
                st.info(f"Detected mood: **{mood}**")

        length = st.slider("Length (notes)", 50, 500, config.DEFAULT_GENERATION_LENGTH, step=25)
        base_tempo = st.slider("Base tempo (BPM)", 60, 200, 120, step=5)

    with col2:
        profile = get_profile(mood)
        st.markdown(f"**Scale:** {profile['scale']}")
        st.markdown(f"**Tempo x{profile['tempo_multiplier']}**")
        st.markdown(f"**Density x{profile['note_density']}**")
        st.markdown(f"**Temperature:** {profile['temperature']}")

    if st.button("🎹 Generate", type="primary", key="gen_mood"):
        with st.spinner("Composing..."):
            out_dir = tempfile.mkdtemp()
            midi_path = os.path.join(out_dir, "synthmuse_output.mid")
            _, notes = generator.generate_midi_file(
                midi_path, length=length, mood=mood, base_tempo=base_tempo
            )
            wav_path = os.path.join(out_dir, "synthmuse_output.wav")
            render_midi_to_wav(midi_path, wav_path)
            _save_to_library("mood", mood, midi_path, wav_path, len(notes), {"base_tempo": base_tempo})

        st.success("Done! Saved to your Library too.")
        st.plotly_chart(build_piano_roll(notes, mood=mood), use_container_width=True)
        st.audio(wav_path)

        c1, c2, c3 = st.columns(3)
        with open(midi_path, "rb") as f:
            c1.download_button("⬇️ Download MIDI", f, file_name="synthmuse_output.mid")
        with open(wav_path, "rb") as f:
            c2.download_button("⬇️ Download WAV", f, file_name="synthmuse_output.wav")

        if c3.button("🎼 Export Sheet Music (MusicXML)"):
            midi_stream = generator.notes_to_midi_stream(notes, mood=mood, base_tempo=base_tempo)
            xml_path = os.path.join(out_dir, "synthmuse_output.musicxml")
            export_musicxml(midi_stream, xml_path)
            with open(xml_path, "rb") as f:
                st.download_button("⬇️ Download MusicXML", f, file_name="synthmuse_output.musicxml")

# ---------------------------------------------------------------------------
# Tab 2: Finish My Melody
# ---------------------------------------------------------------------------
with tab_continue:
    st.subheader("Upload a melody idea — AI finishes composing it")
    uploaded = st.file_uploader("Upload a short MIDI file", type=["mid", "midi"])
    mood2 = st.selectbox("Continuation mood", available_moods(), key="mood2")
    extra_length = st.slider("How many extra notes to compose", 50, 500, 150, step=25, key="len2")

    if uploaded and st.button("✍️ Continue Composing", type="primary"):
        with st.spinner("Analyzing your melody and composing a continuation..."):
            tmp_dir = tempfile.mkdtemp()
            upload_path = os.path.join(tmp_dir, uploaded.name)
            with open(upload_path, "wb") as f:
                f.write(uploaded.getbuffer())

            full_sequence, original_notes, generated_notes = continue_midi(
                generator, upload_path, extra_length=extra_length, mood=mood2
            )

            midi_path = os.path.join(tmp_dir, "continued.mid")
            midi_stream = generator.notes_to_midi_stream(full_sequence, mood=mood2)
            midi_stream.write("midi", fp=midi_path)

            wav_path = os.path.join(tmp_dir, "continued.wav")
            render_midi_to_wav(midi_path, wav_path)
            _save_to_library("continuation", mood2, midi_path, wav_path, len(full_sequence),
                              {"original_notes": len(original_notes), "generated_notes": len(generated_notes)})

        st.success(f"Original {len(original_notes)} notes extended with {len(generated_notes)} new notes. Saved to your Library.")
        st.plotly_chart(build_piano_roll(full_sequence, mood=mood2), use_container_width=True)
        st.audio(wav_path)
        with open(midi_path, "rb") as f:
            st.download_button("⬇️ Download MIDI", f, file_name="synthmuse_continued.mid")

# ---------------------------------------------------------------------------
# Tab 3: Music DNA Mixer
# ---------------------------------------------------------------------------
with tab_fusion:
    st.subheader("Blend two musical styles into a hybrid composition")
    col1, col2 = st.columns(2)
    with col1:
        file_a = st.file_uploader("Style A (MIDI)", type=["mid", "midi"], key="style_a")
    with col2:
        file_b = st.file_uploader("Style B (MIDI)", type=["mid", "midi"], key="style_b")

    ratio = st.slider("Blend ratio (0 = fully Style A, 1 = fully Style B)", 0.0, 1.0, 0.5, step=0.05)
    mood3 = st.selectbox("Mood for the hybrid piece", available_moods(), key="mood3")
    length3 = st.slider("Length (notes)", 50, 500, 200, step=25, key="len3")

    if file_a and file_b and st.button("🧬 Fuse & Generate", type="primary"):
        with st.spinner("Extracting DNA from both styles and fusing..."):
            tmp_dir = tempfile.mkdtemp()
            path_a = os.path.join(tmp_dir, file_a.name)
            path_b = os.path.join(tmp_dir, file_b.name)
            with open(path_a, "wb") as f:
                f.write(file_a.getbuffer())
            with open(path_b, "wb") as f:
                f.write(file_b.getbuffer())

            generated_notes, notes_a, notes_b = fuse_and_generate(
                generator, path_a, path_b, ratio=ratio, length=length3, mood=mood3
            )

            midi_path = os.path.join(tmp_dir, "fusion.mid")
            midi_stream = generator.notes_to_midi_stream(generated_notes, mood=mood3)
            midi_stream.write("midi", fp=midi_path)

            wav_path = os.path.join(tmp_dir, "fusion.wav")
            render_midi_to_wav(midi_path, wav_path)
            _save_to_library("fusion", mood3, midi_path, wav_path, len(generated_notes),
                              {"ratio": ratio, "style_a": file_a.name, "style_b": file_b.name})

        st.success("Hybrid track composed and saved to your Library!")
        st.plotly_chart(build_piano_roll(generated_notes, mood=mood3), use_container_width=True)
        st.audio(wav_path)
        with open(midi_path, "rb") as f:
            st.download_button("⬇️ Download MIDI", f, file_name="synthmuse_fusion.mid")

# ---------------------------------------------------------------------------
# Tab 4: My Library
# ---------------------------------------------------------------------------
FEATURE_LABELS = {
    "mood": ("🎼", "Compose by Mood"),
    "continuation": ("✍️", "Finish My Melody"),
    "fusion": ("🧬", "Music DNA Mixer"),
}

with tab_library:
    st.subheader("Your generated tracks")

    top_col1, top_col2 = st.columns([3, 1])
    with top_col1:
        filter_choice = st.selectbox(
            "Filter by feature",
            ["All", "Compose by Mood", "Finish My Melody", "Music DNA Mixer"],
            key="lib_filter",
        )
    with top_col2:
        st.write("")
        if st.button("🗑️ Clear Library"):
            history.clear_all()
            st.rerun()

    filter_map = {
        "All": "all",
        "Compose by Mood": "mood",
        "Finish My Melody": "continuation",
        "Music DNA Mixer": "fusion",
    }
    entries = history.list_entries(feature_filter=filter_map[filter_choice])

    if not entries:
        st.info("No tracks yet — generate something in the tabs above and it'll show up here.")
    else:
        for entry in entries:
            icon, feature_name = FEATURE_LABELS.get(entry["feature"], ("🎵", entry["feature"]))
            color = _mood_color(entry.get("mood", config.DEFAULT_EMOTION))
            mood_label = entry.get("mood", "—").capitalize()

            card_col, action_col = st.columns([4, 2])
            with card_col:
                st.markdown(
                    f"""
                    <div class="lib-card">
                        <div class="lib-thumb" style="background:{color};">{icon}</div>
                        <div>
                            <p class="lib-title">{feature_name}
                                <span class="lib-badge">{mood_label}</span>
                            </p>
                            <p class="lib-sub">{entry['created_at']} • {entry.get('note_count', 0)} notes</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with action_col:
                sub1, sub2, sub3 = st.columns(3)
                if entry.get("wav_path") and os.path.exists(entry["wav_path"]):
                    with open(entry["wav_path"], "rb") as f:
                        sub1.download_button("🔊", f, file_name=f"synthmuse_{entry['id']}.wav", key=f"wav_{entry['id']}")
                if entry.get("midi_path") and os.path.exists(entry["midi_path"]):
                    with open(entry["midi_path"], "rb") as f:
                        sub2.download_button("⬇️", f, file_name=f"synthmuse_{entry['id']}.mid", key=f"mid_{entry['id']}")
                if sub3.button("🗑️", key=f"del_{entry['id']}"):
                    history.delete_entry(entry["id"])
                    st.rerun()

            if entry.get("wav_path") and os.path.exists(entry["wav_path"]):
                st.audio(entry["wav_path"])

# ---------------------------------------------------------------------------
# Tab 5: About
# ---------------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
        ### What makes SynthMuse different

        Most open-source LSTM music generators do one thing: sample a
        trained model from a random seed and export a MIDI file.
        SynthMuse keeps that solid generative core, but wraps it with
        product features aimed at real use:

        - **Emotion Engine** — mood or free-text input is translated into
          concrete musical parameters (scale, tempo, density, temperature)
          instead of leaving the model to just guess.
        - **Finish My Melody** — continue a melody you hummed or wrote
          yourself, rather than only generating from scratch.
        - **Music DNA Mixer** — blend two different reference tracks into
          one hybrid seed for a genuinely new hybrid piece.
        - **My Library** — every generated track is saved with its mood,
          length, and timestamp, and survives app restarts.
        - **Live piano-roll visualization** and **in-browser playback**,
          so you don't need an external MIDI player just to preview a take.
        - **Sheet-music export (MusicXML)** for musicians who want to read
          and edit the notation, not just the audio.

        Built with TensorFlow/Keras, music21, pretty_midi, Plotly and Streamlit.
        """
    )
