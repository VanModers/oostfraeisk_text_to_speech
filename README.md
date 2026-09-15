# East Frisian text-to-speech

Piper/VITS speech synthesis for East Frisian Low Saxon (Oostfräisk). The model
learns from two speakers and uses their equally averaged speaker embeddings
for the exported voice.

- [Model on Hugging Face](https://huggingface.co/VanModers114/East_Frisian_TTS)

## Training

Use Linux/WSL with a CUDA GPU and a Python environment with
[Piper's training dependencies and compiled alignment extension](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/TRAINING.md)
installed. This project was trained with Piper 1.4.2. Install `huggingface_hub`
in the same environment.

The dataset in `data/oostfraeisk/` contains 1,500 recordings (2.94 hours), mono
22,050 Hz WAVs, and paired transcripts. `metadata_multispeaker.csv` uses
`utterance_id|speaker_label|text`, with labels `speaker_1` and `speaker_2`.

From the repository root:

```bash
python train_piper.py --mode multi --max-epochs 3000
```

Training uses character input and a German Thorsten acoustic warm start.
The script downloads the pretrained checkpoint, trains both speakers, and
exports the final checkpoint with averaged embeddings to
`model_piper_multi/oostfraeisk.onnx` and its `.onnx.json` configuration.
It also generates listening samples. Use `--skip-checkpoint-samples` to skip
extra per-checkpoint exports, or `--checkpoint-selection best` to export
the lowest-validation-loss checkpoint. Run `--help` for all options.

## Use the model

Install Piper and the Hugging Face client:

```bash
pip install piper-tts huggingface_hub
```

Download the model from Hugging Face and generate speech locally:

```python
import wave

from huggingface_hub import hf_hub_download
from piper import PiperVoice

repo_id = "VanModers114/East_Frisian_TTS"
model_path = hf_hub_download(repo_id, "oostfraeisk.onnx")
config_path = hf_hub_download(repo_id, "oostfraeisk.onnx.json")
voice = PiperVoice.load(model_path, config_path=config_path)

with wave.open("oostfraeisk.wav", "wb") as wav_file:
    voice.synthesize_wav("Moin, woo gaajt 't dii?", wav_file)
```

The model runs on CPU. Alternatively, try the TTS integration of the [Ooversetter](https://oostfraeisk.org/translator)
without installing anything.
