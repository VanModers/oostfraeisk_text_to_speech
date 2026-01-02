import os
import difflib
import string

# -------- CONFIG --------
METADATA_PATH = "data/oostfraeisk/metadata.csv"
DICT_PATH = "frs.dic"
# -------------------------


def load_dictionary(dict_path):
    """Load dictionary from file, return empty set if file doesn't exist."""
    if not os.path.exists(dict_path):
        return set()
    with open(dict_path, "r", encoding="utf-8") as f:
        return set(w.strip() for w in f if w.strip())


def save_dictionary(dict_path, dictionary):
    """Save dictionary to file, sorted alphabetically."""
    with open(dict_path, "w", encoding="utf-8") as f:
        for w in sorted(dictionary):
            f.write(w + "\n")


def clean_word(word):
    """Remove punctuation around words, e.g. pad, → pad"""
    return word.strip(string.punctuation)


def word_in_dictionary(word, dictionary):
    """Check dictionary including lowercase fallback, ignoring punctuation."""
    w = clean_word(word)
    if w in dictionary:
        return True
    if w.lower() in dictionary:
        return True
    return False


def suggest_replacement(word, dictionary):
    """Suggest closest match based on lowercase comparison, ignoring punctuation."""
    base_word = clean_word(word).lower()
    matches = difflib.get_close_matches(base_word, dictionary, n=1, cutoff=0.6)
    return matches[0] if matches else None


def load_metadata(metadata_path):
    """Load metadata.csv and return list of (sentence_id, text, normalized_text) tuples."""
    entries = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) >= 2:
                sentence_id = parts[0]
                text = parts[1]
                # normalized_text is same as text if not present or use third column
                normalized_text = parts[2] if len(parts) >= 3 else text
                entries.append([sentence_id, text, normalized_text])
    return entries


def save_metadata(metadata_path, entries):
    """Save metadata entries back to metadata.csv."""
    with open(metadata_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write("|".join(entry) + "\n")


def main():
    # Load dictionary
    dictionary = load_dictionary(DICT_PATH)
    print(f"Loaded dictionary with {len(dictionary)} words from {DICT_PATH}")

    # Load metadata
    entries = load_metadata(METADATA_PATH)
    total = len(entries)
    print(f"Loaded {total} entries from {METADATA_PATH}")

    modified_count = 0

    for idx, entry in enumerate(entries, start=1):
        sentence_id, text, normalized_text = entry

        words = normalized_text.split()

        # Detect unknown words
        unknown_words = [w for w in words if not word_in_dictionary(w, dictionary)]

        if not unknown_words:
            continue

        print("\n" + "=" * 50)
        print(f"Entry {idx}/{total} ({sentence_id})")
        print(f"Text: {normalized_text}")
        print(f"Unknown words: {unknown_words}")

        do_correct = input("\nCorrect this entry? (y/n/q to quit): ").strip().lower()
        if do_correct == "q":
            print("Quitting...")
            break
        if do_correct != "y":
            continue

        new_words = words[:]

        for uw in unknown_words:
            cleaned = clean_word(uw)

            print(f"\nUnknown word: '{uw}' (cleaned: '{cleaned}')")

            suggestion = suggest_replacement(uw, dictionary)
            if suggestion:
                print(f"Suggested replacement: '{suggestion}'")
            else:
                print("No suggestion available.")

            action = input("Action? (y=use suggestion, n=enter manually, s=skip and add to dictionary): ").strip().lower()

            # If using suggestion
            if action == "y" and suggestion:
                # Preserve original punctuation
                prefix = ""
                suffix = ""
                for c in uw:
                    if c in string.punctuation:
                        prefix += c
                    else:
                        break
                for c in reversed(uw):
                    if c in string.punctuation:
                        suffix = c + suffix
                    else:
                        break
                replacement = prefix + suggestion + suffix
                dictionary.add(suggestion)
                new_words = [replacement if w == uw else w for w in new_words]

            # Manual correction
            elif action == "n":
                replacement = input("Enter correct replacement: ").strip()
                if replacement:
                    dictionary.add(clean_word(replacement))
                    new_words = [replacement if w == uw else w for w in new_words]

            # Skip → but add cleaned word to dictionary
            elif action == "s":
                dictionary.add(cleaned)
                continue

            else:
                continue

        new_text = " ".join(new_words)
        
        # Update both text and normalized_text columns
        entry[1] = new_text
        entry[2] = new_text
        
        print(f"Updated: {new_text}")
        modified_count += 1

        # Save after each modification
        save_metadata(METADATA_PATH, entries)
        save_dictionary(DICT_PATH, dictionary)

    print("\n" + "=" * 50)
    print(f"Done! Modified {modified_count} entries.")
    print(f"Metadata saved to: {METADATA_PATH}")
    print(f"Dictionary saved to: {DICT_PATH}")


if __name__ == "__main__":
    main()