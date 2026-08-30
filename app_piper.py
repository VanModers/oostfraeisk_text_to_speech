import gradio as gr
import os
import re
import tempfile
import wave

from piper import PiperVoice
from huggingface_hub import hf_hub_download
from audio_provenance import (
    append_wav_provenance,
    apply_audioseal_watermark,
    detect_audioseal_watermark,
    read_wav_provenance,
)

# =============================================================================
# East Frisian Text Normalizer
# Converts numbers, dates, and special formats to spoken words
# =============================================================================

# East Frisian number words
ONES = {
    0: "nul", 1: "äien", 2: "twäj", 3: "dräj", 4: "fäär",
    5: "fîiv", 6: "säes", 7: "sööem", 8: "âacht", 9: "neegen",
    10: "tâajn", 11: "elm", 12: "twalm", 13: "daartain", 14: "fäärtain",
    15: "fiiftain", 16: "sestain", 17: "söömtain", 18: "achttain", 19: "neegentain"
}

PRE_ONES = {
    0: "nul", 1: "äin", 2: "twei", 3: "drei", 4: "fäär",
    5: "fiif", 6: "ses", 7: "sööm", 8: "acht", 9: "neegen",
    10: "tain"
}

TENS = {
    2: "twintiğ", 3: "daartiğ", 4: "fäärtiğ", 5: "fiiftiğ",
    6: "tsestiğ", 7: "tsöömtiğ", 8: "tachentiğ", 9: "neegentiğ"
}

HUNDREDS = "hunnert"
THOUSANDS = "duusend"
MILLIONS = "miljoonen"

def number_to_east_frisian(n: int) -> str:
    """Convert an integer to East Frisian words."""
    if n < 0:
        return "minus " + number_to_east_frisian(-n)
    
    if n < 20:
        return ONES[n]
    
    if n == 88:
        return "tachuntachentiğ"  # Special case for 88
    
    if n < 100:
        tens, ones = divmod(n, 10)
        if ones == 0:
            return TENS[tens]
        # East Frisian uses "one-and-twenty" order like German/Dutch
        return f"{PRE_ONES[ones]}un{TENS[tens]}"
    
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        if rest == 0:
            return f"{PRE_ONES[hundreds]}{HUNDREDS}" if hundreds > 1 else HUNDREDS
        prefix = f"{PRE_ONES[hundreds]}{HUNDREDS}" if hundreds > 1 else HUNDREDS
        return f"{prefix}{number_to_east_frisian(rest)}"
    
    if n < 1000000:
        thousands, rest = divmod(n, 1000)
        if rest == 0:
            return f"{number_to_east_frisian(thousands)} {THOUSANDS}" if thousands > 1 else THOUSANDS
        prefix = f"{number_to_east_frisian(thousands)} {THOUSANDS} " if thousands > 1 else f"{THOUSANDS} "
        return prefix + number_to_east_frisian(rest)
    
    if n < 1000000000:
        millions, rest = divmod(n, 1000000)
        prefix = f"{number_to_east_frisian(millions)} {MILLIONS}"
        if rest == 0:
            return prefix
        return prefix + " " + number_to_east_frisian(rest)
    
    # For very large numbers, just spell out digits
    return " ".join(ONES[int(d)] for d in str(n))


# Letter-by-letter East Frisian pronunciation of capital letters
LETTER_NAMES = {
    'A': 'aa',  'B': 'bäi', 'C': 'tsäi', 'D': 'däi', 'E': 'äi',
    'F': 'ef',  'G': 'gäi', 'H': 'haa',  'I': 'ii',  'J': 'jot',
    'K': 'kaa', 'L': 'el',  'M': 'em',   'N': 'en',  'O': 'oo',
    'P': 'päi', 'Q': 'kuu', 'R': 'eer',  'S': 'es',  'T': 'täi',
    'U': 'uu',  'V': 'fau', 'W': 'wäi',  'X': 'iks', 'Y': 'üpsilon',
    'Z': 'tset',
}

# Ordinal number forms in East Frisian
ORDINALS = {
    1: 'êerst',       2: 'twäied',      3: 'dâard',       4: 'fäärd',
    5: 'fîift',       6: 'säest',       7: 'sööemt',      8: 'âacht',
    9: 'neegent',     10: 'tâajnt',     11: 'elft',       12: 'twalf',
    13: 'daartâajnt', 14: 'fäärtâajnt',
}

def number_to_ordinal_east_frisian(n: int) -> str:
    if n in ORDINALS:
        return ORDINALS[n]
    return number_to_east_frisian(n) + 'st'


def normalize_text(text: str) -> str:
    """
    Normalize text for TTS: convert numbers, dates, abbreviations to words.
    """
    # Handle ordinal numbers first (e.g. "1." → "êerst", "14." → "fäärtâajnt")
    def replace_ordinal(match):
        return number_to_ordinal_east_frisian(int(match.group(1)))

    text = re.sub(r'\b(\d+)\.(?=\s+\w)', replace_ordinal, text)

    # Expand capital-letter abbreviations letter by letter (e.g. "USA" → "uu es aa")
    def expand_abbreviation(match):
        return ' '.join(LETTER_NAMES.get(c, c) for c in match.group(0))

    text = re.sub(r'\b[A-Z]{2,}\b', expand_abbreviation, text)

    # Replace numbers with words (longest matches first to handle e.g. "2024" before "20")
    def replace_number(match):
        num_str = match.group(0)
        # Handle decimal numbers
        if ',' in num_str or '.' in num_str:
            parts = re.split(r'[,.]', num_str)
            if len(parts) == 2:
                whole = int(parts[0]) if parts[0] else 0
                decimal = parts[1]
                decimal_words = " ".join(ONES[int(d)] for d in decimal)
                return f"{number_to_east_frisian(whole)} kummó {decimal_words}"
        return number_to_east_frisian(int(num_str))
    
    # Match integers and decimal numbers
    text = re.sub(r'\d+([,.]\d+)?', replace_number, text)
    
    # Abbreviations with periods
    abbreviations_with_period = {
        "Dr.": "Dokter",
        "Hr.": "Heer",
        "Fr.": "Frâau",
        "usw.": "un so wiider",
        "t.B.": "tau 'n biispil",
    }
    
    for abbr, expansion in abbreviations_with_period.items():
        text = text.replace(abbr, expansion)
    
    # Units - only replace when preceded by a space or digit and followed by word boundary
    units = {
        "km": "kilomeeter",
        "cm": "tsentimeeter",
        "mm": "millimeeter",
        "kg": "kilogramm",
        "mg": "milligram",
    }
    
    for unit, expansion in units.items():
        text = re.sub(rf'(?<=\d)\s*{unit}(?=\s|$|[.,;:!?])', f' {expansion}', text)
    
    # Single-letter units - ONLY after digits with optional space
    single_units = {
        "m": "meeter",
        "g": "gram",
    }
    
    for unit, expansion in single_units.items():
        text = re.sub(rf'(?<=\d)\s*{unit}(?=\s|$|[.,;:!?])', f' {expansion}', text)
    
    # Symbols
    symbols = {
        "%": " prosent",
        "€": " oiro",
        "$": " duller",
        "§": "parógróóf ",
    }
    
    for symbol, expansion in symbols.items():
        text = text.replace(symbol, expansion)
    
    # Clean up multiple spaces
    text = re.sub(r' +', ' ', text)
    
    return text


# =============================================================================
# Load Piper Voice Model
# Piper uses ONNX for inference — fast on CPU, no GPU needed!
# =============================================================================

MODEL_REPO_ID = os.environ.get("MODEL_REPO_ID", "VanModers114/East_Frisian_TTS")
MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "oostfraeisk.onnx")

# Download model assets from the Hub model repo at startup.
MODEL_PATH = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME, repo_type="model")
hf_hub_download(repo_id=MODEL_REPO_ID, filename=f"{MODEL_FILENAME}.json", repo_type="model")

voice = PiperVoice.load(MODEL_PATH)

def tts_fn(text):
    if not text or not text.strip():
        return None
    
    # 1. Normalize text (numbers, abbreviations, units → words)
    text = normalize_text(text)

    # 2. Synthesize with Piper
    output_file = tempfile.NamedTemporaryFile(
        prefix="oostfraeisk_ai_", suffix=".wav", delete=False
    )
    out_path = output_file.name
    output_file.close()
    with wave.open(out_path, "w") as wav_file:
        voice.synthesize_wav(text, wav_file)

    # Fail closed: never return synthetic speech if either marking layer fails.
    apply_audioseal_watermark(out_path)
    append_wav_provenance(out_path, watermark_applied=True)

    return out_path


def verify_audio(audio_path):
    if not audio_path:
        return "Upload a WAV file to check it."

    metadata = read_wav_provenance(audio_path)
    probability, message = detect_audioseal_watermark(audio_path)
    metadata_status = "present" if metadata and metadata.get("ai_generated") else "absent"
    message_status = message or "not decoded"
    return (
        f"AudioSeal probability: {probability:.3f}\n\n"
        f"Embedded AIGC metadata: {metadata_status}\n\n"
        f"Watermark message: `{message_status}`"
    )


# Keep api_name="predict" for the existing website integration.
with gr.Blocks(title="East Frisian Low Saxon TTS") as demo:
    gr.Markdown(
        "# East Frisian Low Saxon TTS\n\n"
        "Type some text and listen to it spoken in East Frisian Low Saxon.\n\n"
        "**AI notice:** Every output is synthetic speech generated by a Piper AI model. "
        "Generated WAV files contain an AudioSeal watermark and machine-readable AIGC metadata.\n\n"
        "Powered by [Piper](https://github.com/rhasspy/piper) — fast ONNX inference on CPU."
    )
    with gr.Tab("Generate speech"):
        text_input = gr.Textbox(
            label="Enter East Frisian text",
            placeholder="Moin, woo gaajt 't dii?",
            lines=3,
        )
        generate_button = gr.Button("Generate", variant="primary")
        audio_output = gr.Audio(label="AI-generated speech")
        generate_button.click(
            tts_fn,
            inputs=text_input,
            outputs=audio_output,
            api_name="predict",
        )
        gr.Examples(
            examples=[
                ["Moin, woo gaajt 't dii?"],
                ["Denkent jii, dat ik disser sats gaud uutprooten dau?"],
                ["Hest duu däi süen fandóóeğ al säin?"],
                ["Wii prootent Oostfräisk."],
            ],
            inputs=text_input,
        )
    with gr.Tab("Verify watermark"):
        gr.Markdown("Check an audio file for this service's AudioSeal watermark and AIGC metadata.")
        verify_input = gr.Audio(type="filepath", label="WAV file")
        verify_button = gr.Button("Verify")
        verify_output = gr.Markdown()
        verify_button.click(
            verify_audio,
            inputs=verify_input,
            outputs=verify_output,
            api_name="verify_watermark",
        )

demo.launch()
