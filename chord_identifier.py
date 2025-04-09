import numpy as np
import os
import csv
import scipy.io.wavfile
import scipy.signal
import itertools
import matplotlib.pyplot as plt

# Names of the basic musical notes
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Dictionary mapping notes to their enharmonic equivalents (e.g., A# is equivalent to Bb)
EQUIVALENT_NOTES = {
    'A#': 'Bb', 'Bb': 'A#',
    'B#': 'Cb', 'Cb': 'B#',
    'C#': 'Db', 'Db': 'C#',
    'D#': 'Eb', 'Eb': 'D#',
    'F#': 'Gb', 'Gb': 'F#',
    'G#': 'Ab', 'Ab': 'G#'
}

def load_chord_db(file_path):
    """
    Generator that reads a CSV file containing chord definitions.
    Each row must have columns 'NOTE_NAMES', 'CHORD_ROOT', and 'CHORD_TYPE'.
    It yields a tuple of (frozenset of note names, chord name) for each row.
    """
    with open(file_path, 'r') as csvfile:
        # Use ';' as the delimiter for the CSV file
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            # Convert the comma-separated notes into a frozenset for easy comparison
            note_set = frozenset(row['NOTE_NAMES'].split(','))
            # Construct the chord name from the root and type
            chord_name = f"{row['CHORD_ROOT']} {row['CHORD_TYPE']}"
            yield note_set, chord_name

# Determine the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build path to the chord database CSV file located in the same directory as the script
chord_db_path = os.path.join(script_dir, 'ChordDB.csv')

# Build a dictionary mapping note sets (frozensets) to a list of chord names
chord_dict = {}
for note_set, chord_name in load_chord_db(chord_db_path):
    if note_set not in chord_dict:
        chord_dict[note_set] = []
    chord_dict[note_set].append(chord_name)

def frequency_to_note(freq):
    """
    Convert a frequency in Hz to its corresponding musical note.
    Uses A4 (440 Hz) as the reference.
    Returns None if the frequency is non-positive.
    """
    if freq <= 0:
        return None
    # Calculate how many semitones the frequency is away from 440 Hz
    semitones = 12 * np.log2(freq / 440.0)
    # Adjust the index so that A (440 Hz) lands at index 9 in NOTE_NAMES
    note_index = int(round(semitones)) + 9
    # Make sure the note index wraps around within an octave (12 notes)
    note_index %= 12
    return NOTE_NAMES[note_index]

def get_peaks(freqs, fft_magnitude, threshold=0.2):
    """
    Identify peaks in the FFT magnitude spectrum.
    A peak is significant if its amplitude is at least `threshold` times the maximum amplitude.
    Returns peak frequencies and their magnitudes, sorted by magnitude in descending order.
    """
    # Find peaks in the FFT magnitude above a certain height threshold
    peaks, properties = scipy.signal.find_peaks(fft_magnitude, height=threshold * np.max(fft_magnitude))
    peak_freqs = freqs[peaks]
    peak_mags = properties['peak_heights']
    # Sort the peaks by magnitude (highest first)
    sort_idx = np.argsort(peak_mags)[::-1]
    return peak_freqs[sort_idx], peak_mags[sort_idx]

def filter_harmonics(peak_freqs, filter_tolerance):
    """
    Remove frequencies that are likely harmonics of lower frequencies.
    If a frequency is near an integer multiple of any lower fundamental frequency,
    it is filtered out.
    """
    filtered = []
    for f in peak_freqs:
        is_harmonic = False
        # Compare with frequencies that have already been classified as fundamentals
        for base in filtered:
            ratio = f / base
            # Check if the ratio is nearly integer, within a given tolerance
            if abs(ratio - round(ratio)) < filter_tolerance:
                is_harmonic = True
                break
        if not is_harmonic:
            filtered.append(f)
    return np.array(filtered)

def generate_note_variations(note_list):
    """
    Generate all possible variations of the note list by replacing notes with their enharmonic equivalents.
    Returns a list of tuples, where each tuple is one combination of note names.
    """
    variations = []
    for note in note_list:
        if note in EQUIVALENT_NOTES:
            variations.append([note, EQUIVALENT_NOTES[note]])
        else:
            variations.append([note])
    return list(itertools.product(*variations))

def label_chord(note_list):
    """
    Given a list of detected note names, attempt to label the chord by searching the chord dictionary.
    It generates all enharmonic variations and checks for matching chords in several inversions.
    Prints out matching chords and returns them as a list.
    """
    if len(note_list) == 0:
        print("No chord being played")
        return

    matching_chords = set()
    # The lowest note is often used as the chord's root (or determines inversion)
    lowest_note = note_list[0]

    # Generate all possible variations (considering enharmonic equivalents) of the note list
    note_variations = generate_note_variations(note_list)

    for notes in note_variations:
        notes = list(notes)  # Convert tuple to list for manipulation
        # First pass: look for chords where the lowest note is the assumed root
        for i in range(len(notes)):
            note_set_frozen = frozenset(notes)
            chords = chord_dict.get(note_set_frozen, [])
            for chord in chords:
                matching_chords.add(chord)
            # Rotate the list to check for inversions (move the last note to the front)
            notes = [notes[-1]] + notes[:-1]

        # If no chord is found with this approach, re-check using alternative inversions
        if not matching_chords:
            for i in range(len(notes)):
                note_set_frozen = frozenset(notes)
                chords = chord_dict.get(note_set_frozen, [])
                for chord in chords:
                    matching_chords.add(chord)
                # Rotate for next inversion
                notes = [notes[-1]] + notes[:-1]
                print(f'Note List {i}: {notes}')

    if matching_chords:
        print(f"All matching chords: {list(matching_chords)}")
        return list(matching_chords)

    # Return "Unknown Chord" if no match was found
    return ["Unknown Chord"]

def process_audio(file_path, base_thresh=0.2, filter_tolerance=0.03, dynamic_factor=5.0, plot_data=True):
    """
    Process an audio file to detect chord notes.
    
    Steps:
    1. Read and normalize the WAV audio file.
    2. Apply a Hanning window to reduce spectral leakage.
    3. Compute the FFT with zero-padding to increase frequency resolution.
    4. Limit frequency analysis to a range of interest (20 Hz to 5000 Hz).
    5. Smooth the FFT magnitude using a Gaussian filter to reduce noise.
    6. Determine a dynamic threshold for peak detection.
    7. Detect peaks in the smoothed FFT spectrum.
    8. Filter out harmonic frequencies.
    9. Convert the fundamental frequencies to their corresponding musical note names.
    10. Plot the data for debugging (if plot_data is True).

    Returns a sorted list of detected note names.
    """
    # Load the WAV audio data; if stereo, only the first channel is used
    fs, data = scipy.io.wavfile.read(file_path)
    if data.ndim > 1:
        data = data[:, 0]
    
    # Normalize the signal to range [-1, 1]
    data = data / np.max(np.abs(data))
    
    N = len(data)
    # Generate a Hanning window to reduce spectral leakage during FFT
    window = np.hanning(N)
    data_windowed = data * window

    # Zero-pad the signal for improved frequency resolution; 
    # padded length is twice the next power of 2 from N
    padded_length = 2 ** int(np.ceil(np.log2(N))) * 2
    fft_complex = np.fft.fft(data_windowed, n=padded_length)
    # Only need the first half of the FFT (real signal symmetry)
    fft_magnitude = np.abs(fft_complex)[:padded_length // 2]
    # Map each FFT bin to its corresponding frequency
    freqs = np.fft.fftfreq(padded_length, d=1/fs)[:padded_length // 2]

    # Restrict analysis to frequencies between 20 Hz and 5000 Hz
    min_freq = 20
    max_freq = 5000
    valid_indices = (freqs >= min_freq) & (freqs <= max_freq)
    freqs = freqs[valid_indices]
    fft_magnitude = fft_magnitude[valid_indices]

    # Smooth the FFT magnitude spectrum using a Gaussian filter
    from scipy.ndimage import gaussian_filter1d
    fft_magnitude_smoothed = gaussian_filter1d(fft_magnitude, sigma=2)

    # Calculate average power for establishing a dynamic threshold
    avg_power = np.mean(fft_magnitude_smoothed)
    dynamic_thresh = dynamic_factor * avg_power  # Multiplier can be adjusted as needed
    
    # Use the maximum of a base threshold or the dynamic threshold
    used_thresh = max(base_thresh * np.max(fft_magnitude_smoothed), dynamic_thresh)

    # Detect peaks using the used threshold as the required prominence
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(fft_magnitude_smoothed, prominence=used_thresh)
    peak_freqs = freqs[peaks]
    peak_prominences = properties['prominences']

    # Filter the detected peaks to remove harmonics, retaining only fundamentals
    fundamental_freqs = filter_harmonics(peak_freqs, filter_tolerance)

    # Convert each fundamental frequency to its corresponding musical note
    detected_notes = []
    note_labels = []
    for f in fundamental_freqs:
        note = frequency_to_note(f)
        if note is not None:
            detected_notes.append((f, note))
            note_labels.append(note)

    # Sort detected notes by their frequency (lowest to highest)
    detected_notes.sort(key=lambda x: x[0])
    sorted_notes = [note for freq, note in detected_notes]

    print("Detected fundamental frequencies:", fundamental_freqs)
    print("Detected notes:", sorted_notes)

    if plot_data:
        # Plot three subplots: windowed audio, FFT magnitude spectrum with peaks, and note labels on fundamentals.
        plt.figure(figsize=(12, 8))

        # Plot the windowed audio signal
        plt.subplot(3, 1, 1)
        plt.plot(data_windowed)
        plt.title("Windowed Audio Data")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")

        # Plot the smoothed FFT magnitude spectrum and mark the detected peaks
        plt.subplot(3, 1, 2)
        plt.plot(freqs, fft_magnitude_smoothed, label="Smoothed FFT Magnitude")
        plt.scatter(peak_freqs, peak_prominences, color='red', label="Detected Peaks")
        plt.title("FFT Magnitude Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.legend()

        # Plot the detected fundamental frequencies and annotate them with the corresponding note names
        plt.subplot(3, 1, 3)
        plt.stem(fundamental_freqs, np.ones_like(fundamental_freqs), basefmt=" ")
        for i, freq in enumerate(fundamental_freqs):
            plt.text(freq, 1.05, note_labels[i], ha='center')
        plt.title("Detected Fundamental Frequencies")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Presence")

        plt.tight_layout()
        plt.show()

    return sorted_notes

if __name__ == '__main__':
    # Define the script's directory and build the path to an audio file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    audio_file_path = os.path.join(script_dir, 'GeneratedChords', 'synth', 'Fsus4_synth.wav')
    # Alternatively, you can select a different file:
    # audio_file_path = os.path.join(script_dir, 'ChordSamples', 'Emaj.wav')

    # Process the audio file to detect notes and then attempt to label the chord
    note_list = process_audio(audio_file_path)
    chords = label_chord(note_list)
    print("Identified chords:", chords)