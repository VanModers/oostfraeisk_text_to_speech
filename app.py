import gradio as gr
import torch
import spaces
from TTS.utils.synthesizer import Synthesizer

# Initialize Synthesizer (loads model + config + audio processor)
synth = Synthesizer(
    tts_checkpoint="model.pth",
    tts_config_path="config.json",
    use_cuda=False  # Force CPU locally, ZeroGPU will provide GPU
)

# This function will be executed on GPU via ZeroGPU
@spaces.GPU
def generate_audio(text: str):
    wav = synth.tts(text)
    return wav

def tts_fn(text):
    # Call the GPU-decorated function
    wav = generate_audio(text)
    out_path = "out.wav"
    synth.save_wav(wav, out_path)
    return out_path

# Gradio interface
demo = gr.Interface(
    fn=tts_fn,
    inputs=gr.Textbox(label="Enter East Frisian text"),
    outputs=gr.Audio(label="Generated Speech"),
    title="East Frisian Low Saxon TTS",
    description="Type some text and listen to it spoken in East Frisian Low Saxon!"
)

demo.launch()