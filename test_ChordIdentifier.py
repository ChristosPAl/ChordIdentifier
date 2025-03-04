import os
import chord_identifier as CI

EQUIVALENT_CHORDS = {
    'A#': 'Bb', 'Bb': 'A#',
    'B#': 'Cb', 'Cb': 'B#',
    'C#': 'Db', 'Db': 'C#',
    'D#': 'Eb', 'Eb': 'D#',
    'F#': 'Gb', 'Gb': 'F#',
    'G#': 'Ab', 'Ab': 'G#'
}

def extract_label_from_filename(filename):
    # Extract the chord label from the filename (e.g., "Em" from "Em_synth.wav")
    return os.path.basename(filename).split('_')[0]

def test_chord_identifier(directory):
    success_count = 0
    total_count = 0

    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory, filename)
            expected_label = extract_label_from_filename(filename)
            note_list = CI.process_audio(file_path, 0.24, 0.03, False)
            if note_list is None:
                print(f"Failed to process audio for file: {filename}")
                continue

            matching_chords = CI.label_chord(note_list)
            matching_chords = [''.join([c for c in chord if c != ' ']) for chord in matching_chords]

            if expected_label[1] == '#' or expected_label[1] == 'b':
                equiv_expected_label = str(EQUIVALENT_CHORDS.get(expected_label[0:2]) + expected_label[2:])
                print(f"Equivalent Expected Label: {equiv_expected_label}")

            if expected_label in matching_chords:
                success_count += 1
                print("SUCCESS")
            elif equiv_expected_label in matching_chords:
                expected_label = equiv_expected_label
                success_count += 1
                print("SUCCESS")
            else:
                print("FAILIURE")
            total_count += 1

            print(f"File: {filename}, Expected: {expected_label}, Identified: {matching_chords}")
            print("____________________________________________")

    if total_count > 0:
        success_rate = (success_count / total_count) * 100
        print(f"Success rate: {success_rate:.2f}%")
    else:
        print("No valid audio files processed.")

if __name__ == "__main__":
    instrument = "synth"
    directory = f"C:/School/2025_Winter/Python for Engineers/Project/ChordIdentifier/ChordIdentifier/GeneratedChords/{instrument}"
    test_chord_identifier(directory)