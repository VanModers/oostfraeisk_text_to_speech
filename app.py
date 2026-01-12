import gradio as gr
import torch
import spaces
import re
from TTS.utils.synthesizer import Synthesizer

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

def normalize_text(text: str) -> str:
    """
    Normalize text for TTS: convert numbers, dates, abbreviations to words.
    """
    # Replace numbers with words (longest matches first to handle e.g. "2024" before "20")
    def replace_number(match):
        num_str = match.group(0)
        # Handle decimal numbers
        if ',' in num_str or '.' in num_str:
            # Split on comma or period (European format uses comma)
            parts = re.split(r'[,.]', num_str)
            if len(parts) == 2:
                whole = int(parts[0]) if parts[0] else 0
                decimal = parts[1]
                # Read decimal part digit by digit
                decimal_words = " ".join(ONES[int(d)] for d in decimal)
                return f"{number_to_east_frisian(whole)} kummó {decimal_words}"
        return number_to_east_frisian(int(num_str))
    
    # Match integers and decimal numbers
    text = re.sub(r'\d+([,.]\d+)?', replace_number, text)
    
    # Abbreviations with periods (safe to replace directly)
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
    # This prevents "kwam" becoming "kwameeter"
    units = {
        "km": "kilomeeter",
        "cm": "tsentimeeter",
        "mm": "millimeeter",
        "kg": "kilogramm",
        "mg": "milligram",
    }
    
    for unit, expansion in units.items():
        # Match unit only after a digit or space, and before space/punctuation/end
        text = re.sub(rf'(?<=\d)\s*{unit}(?=\s|$|[.,;:!?])', f' {expansion}', text)
    
    # Single-letter units - ONLY after digits with optional space
    single_units = {
        "m": "meeter",
        "g": "gram",
    }
    
    for unit, expansion in single_units.items():
        # Very strict: only "5 m" or "5m" patterns, not inside words
        text = re.sub(rf'(?<=\d)\s*{unit}(?=\s|$|[.,;:!?])', f' {expansion}', text)
    
    # Symbols (safe to replace, they don't appear in words)
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
    # Normalize text (numbers → words)
    normalized_text = normalize_text(text)
    
    # Call the GPU-decorated function
    wav = generate_audio(normalized_text)
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