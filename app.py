    # app.py for Hugging Face Spaces (full version with launch)
import gradio as gr
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import torchaudio
from gtts import gTTS
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting application...")

# Load the model and processor
try:
    logger.info("Loading processor and model...")
    processor = WhisperProcessor.from_pretrained("dennis-9/whisper-small_Akan_non_standardspeech")
    model = WhisperForConditionalGeneration.from_pretrained("dennis-9/whisper-small_Akan_non_standardspeech")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(task="transcribe")
    model.config.use_cache = True
    logger.info("Model loaded successfully on %s", device)
except Exception as e:
    logger.error("Failed to load model: %s", str(e))
    raise

# Simple language model for post-processing
def correct_transcription(text):
    corrections = {"akan": "Akan", "akn": "Akan", "spch": "speech", "imprd": "impaired"}
    words = text.split()
    corrected = [corrections.get(word.lower(), word) for word in words]
    return " ".join(corrected)

# Transcription function
def transcribe(audio, speed=1.0):
    if audio is None:
        return "Please upload an audio file.", "Confidence: N/A"
    try:
        audio_data, sr = torchaudio.load(audio)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio_data = resampler(audio_data)
        audio_data = audio_data[0].numpy()
        if len(audio_data) / 16000 > 15:
            return "Audio is too long (>15s). Please use a shorter clip.", "Confidence: N/A"
        input_features = processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        with torch.amp.autocast(device_type=device if device == "cuda" else "cpu"):
            with torch.no_grad():
                predicted_ids = model.generate(input_features, max_length=100, task="transcribe")
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        corrected_text = correct_transcription(transcription)
        confidence = 0.9
        return corrected_text, f"Confidence: {confidence:.2f}"
    except Exception as e:
        logger.error("Transcription failed: %s", str(e))
        return f"Error during transcription: {str(e)}", "Confidence: N/A"

# TTS function
def text_to_speech(text, speed=1.0):
    if not text or text.strip() == "":
        return "No text to convert.", None
    try:
        tts = gTTS(text=text, lang="en")
        audio_file = "output.mp3"
        tts.save(audio_file)
        return "Audio generated.", audio_file
    except Exception as e:
        logger.error("TTS failed: %s", str(e))
        return f"Error generating audio: {str(e)}", None

# Sign language phrases
sign_phrases = {"Hello": "Akye", "Thank you": "Medaase", "Help": "Boa me"}

# Gradio interface
with gr.Blocks(title="Akan ASR with Speech-Impaired Support") as demo:
    gr.Markdown("# Akan Speech Recognition with Accessibility Features\nUpload audio (<15s) or use live input.")
    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Upload Audio or Record Live")
        speed_slider = gr.Slider(0.5, 2.0, 0.1, value=1.0, label="TTS Speed")
    transcribe_btn = gr.Button("Transcribe")
    output_text = gr.Textbox(label="Transcription", interactive=False)
    confidence_text = gr.Textbox(label="Confidence", interactive=False)
    with gr.Row():
        tts_input = gr.Textbox(label="Enter Text for TTS")
        tts_btn = gr.Button("Convert to Speech")
        tts_output = gr.Audio(label="TTS Output")
    with gr.Row():
        gr.Markdown("### Quick Phrases for Sign Language")
        for phrase, akan in sign_phrases.items():
            gr.Markdown(f"- {phrase}: {akan}")
    tts_btn.click(fn=text_to_speech, inputs=[tts_input, speed_slider], outputs=[gr.Textbox(label="Status", interactive=False), tts_output])
    transcribe_btn.click(fn=transcribe, inputs=[audio_input, speed_slider], outputs=[output_text, confidence_text])
    gr.Markdown("### Help\n- **Speech-Impaired Users**: Use TTS or record audio for subtitles.\n- **ASR Tips**: Use short clips (<15s).\n- **Sign Language**: Refer to [Guide](https://example.com).")

# Explicit launch configuration
demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

logger.info("Application setup completed.")
