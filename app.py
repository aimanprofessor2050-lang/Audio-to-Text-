import streamlit as st
from faster_whisper import WhisperModel
import os
import tempfile

# Set page config
st.set_page_config(page_title="Free Audio-to-Text App", page_icon="🎙️")

st.title("🎙️ Free Audio-to-Text App")
st.markdown("Powered by **faster-whisper** (Open Source). No paid APIs used.")

# Sidebar for settings
st.sidebar.header("Settings")
model_size = st.sidebar.selectbox(
    "Select Model Size",
    ["tiny", "base", "small", "medium", "large-v3"],
    index=2,
    help="Larger models are more accurate but slower. 'small' or 'medium' is usually a good balance."
)

# Detect GPU
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

st.sidebar.info(f"Running on: **{device.upper()}**")

@st.cache_resource
def load_model(model_size, device, compute_type):
    return WhisperModel(model_size, device=device, compute_type=compute_type)

# Main app
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a", "ogg", "flac"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("Transcribe"):
        with st.spinner("Loading model and transcribing..."):
            try:
                # Save uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                # Load model
                model = load_model(model_size, device, compute_type)
                
                # Transcribe
                # beam_size=5 is standard, language=None for auto-detection
                segments, info = model.transcribe(tmp_path, beam_size=5)
                
                st.success(f"Detected language: **{info.language}** (probability: {info.language_probability:.2f})")
                
                full_text = ""
                transcription_placeholder = st.empty()
                
                for segment in segments:
                    full_text += segment.text + " "
                    transcription_placeholder.markdown(full_text)
                
                # Clean up temp file
                os.remove(tmp_path)
                
                # Download button
                st.download_button(
                    label="Download Text",
                    data=full_text,
                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcription.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.info("Please upload an audio file to start.")

st.markdown("---")
st.caption("Note: Burmese (my) is supported. For better results with Burmese, consider using 'medium' or 'large-v3' models.")
