import os

METADATA_PATH = "data/oostfraeisk/metadata.csv"

def fix_metadata(metadata_path):
    fixed_lines = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            # Only fix if third column is "speaker"
            if len(parts) == 3 and parts[2] == "speaker":
                parts[2] = parts[1]
            fixed_lines.append("|".join(parts))
    # Write back the fixed lines
    with open(metadata_path, "w", encoding="utf-8") as f:
        for line in fixed_lines:
            f.write(line + "\n")

if __name__ == "__main__":
    fix_metadata(METADATA_PATH)