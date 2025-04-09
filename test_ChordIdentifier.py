import os
import chord_identifier as CI

# Mapping of chord labels with accidentals to their equivalent representations.
EQUIVALENT_CHORDS = {
    'A#': 'Bb', 'Bb': 'A#',
    'B#': 'Cb', 'Cb': 'B#',
    'C#': 'Db', 'Db': 'C#',
    'D#': 'Eb', 'Eb': 'D#',
    'F#': 'Gb', 'Gb': 'F#',
    'G#': 'Ab', 'Ab': 'G#'
}

def extract_label_from_filename(filename):
    """
    Extract the chord label from the filename.
    
    This function assumes that the chord label is the first part of the filename,
    separated by an underscore. For example, "Em_synth.wav" returns "Em".
    
    Args:
        filename (str): The name of the file.
        
    Returns:
        str: Extracted chord label.
    """
    return os.path.basename(filename).split('_')[0]

def test_chord_identifier(directory):
    """
    Process all .wav files in the given directory using the chord identifier,
    compare the expected chord label extracted from the filename to the result,
    and report the success rate.
    
    Args:
        directory (str): Directory containing .wav files to be processed.
    """
    success_count = 0
    total_count = 0

    # Iterate through all files in the provided directory.
    for filename in os.listdir(directory):
        # Only process files with a .wav extension.
        if filename.endswith(".wav"):
            file_path = os.path.join(directory, filename)
            # Extract the expected chord label from the filename.
            expected_label = extract_label_from_filename(filename)
            
            # Process the audio file and retrieve a list of notes.
            note_list = CI.process_audio(file_path, 0.24, 0.03, 2, False)
            if note_list is None:
                # Print an error message if audio processing failed.
                print(f"Failed to process audio for file: {filename}")
                continue

            # Get matching chords using the chord identifier.
            matching_chords = CI.label_chord(note_list)
            # Remove spaces in chord labels in the matching list.
            matching_chords = [''.join([c for c in chord if c != ' ']) for chord in matching_chords]

            # Check if the expected label contains an accidental (sharp '#' or flat 'b')
            # and create the equivalent label if applicable.
            if expected_label[1] == '#' or expected_label[1] == 'b':
                equiv_expected_label = str(EQUIVALENT_CHORDS.get(expected_label[0:2]) + expected_label[2:])
                print(f"Equivalent Expected Label: {equiv_expected_label}")
            else:
                equiv_expected_label = None

            # Compare the expected label to the list of identified chords.
            if expected_label in matching_chords:
                success_count += 1
                print("SUCCESS")
            elif equiv_expected_label is not None and equiv_expected_label in matching_chords:
                # Use the equivalent label if it matches.
                expected_label = equiv_expected_label
                success_count += 1
                print("SUCCESS")
            else:
                print("FAILIURE")
            
            total_count += 1

            # Print detailed information for each processed file.
            print(f"File: {filename}, Expected: {expected_label}, Identified: {matching_chords}")
            print("____________________________________________")

    # Calculate and print the overall success rate.
    if total_count > 0:
        success_rate = (success_count / total_count) * 100
        print(f"Success rate: {success_rate:.2f}%")
    else:
        print("No valid audio files processed.")

if __name__ == "__main__":
    # Specify the instrument type and construct the directory path.
    instrument = "synth"
    directory = f"C:/School/2025_Winter/Python for Engineers/Project/ChordIdentifier/ChordIdentifier/GeneratedChords/{instrument}"
    # Run the test function on the specified directory.
    test_chord_identifier(directory)