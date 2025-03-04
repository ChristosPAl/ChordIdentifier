import numpy as np
import math
import sys
import os
import csv
import scipy.io.wavfile
import scipy.signal
import matplotlib.pyplot as plt

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def load_chord_db(file_path):
    chord_dict = {}
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            note_set = frozenset(row['NOTE_NAMES'].split(','))
            chord_name = f"{row['CHORD_ROOT']} {row['CHORD_TYPE']}"
            chord_dict[note_set] = chord_name
    return chord_dict

script_dir = os.path.dirname(os.path.abspath(__file__))
chord_db_path = os.path.join(script_dir, 'ChordDB.csv')
chord_dict = load_chord_db(chord_db_path)

def frequency_to_note(freq):
    if freq <= 0:
        return None
    # A4 (440 Hz) is our reference
    semitones = 12 * np.log2(freq / 440.0)
    # Adjust so that A (440 Hz) is index 9 in our NOTE_NAMES list
    note_index = int(round(semitones)) + 9
    note_index %= 12
    return NOTE_NAMES[note_index]

def get_peaks(freqs, fft_magnitude, threshold=0.2):
    """
    Find peaks in the FFT magnitude spectrum.
    A peak is considered significant if its amplitude is at least `threshold` times the maximum.
    """
    peaks, properties = scipy.signal.find_peaks(fft_magnitude, height=threshold * np.max(fft_magnitude))
    peak_freqs = freqs[peaks]
    peak_mags = properties['peak_heights']
    # Return peaks sorted by magnitude (highest first)
    sort_idx = np.argsort(peak_mags)[::-1]
    return peak_freqs[sort_idx], peak_mags[sort_idx]

def filter_harmonics(peak_freqs, tolerance=0.02):
    """
    Remove frequencies that are likely harmonics.
    If a frequency is near an integer multiple of any lower frequency, it is filtered out.
    """
    filtered = []
    for f in peak_freqs:
        is_harmonic = False
        for base in filtered:
            ratio = f / base
            if abs(ratio - round(ratio)) < tolerance:
                is_harmonic = True
                break
        if not is_harmonic:
            filtered.append(f)
    return np.array(filtered)

def label_chord(note_list):
    matching_chords = []
    lowest_note = note_list[0]

    # First, try to find a chord where the root is the same as the lowest note in the set
    for i in range(len(note_list)):
        note_set_frozen = frozenset(note_list)
        chord = chord_dict.get(note_set_frozen, None)
        if chord:
            matching_chords.append(chord)
            if chord.startswith(lowest_note):
                print(f"Matching chord with root as lowest note: {chord}")
        # Move the last element to the first position (check for inversions)
        note_list = [note_list[-1]] + note_list[:-1]

    # If no chord is found with the lowest note as the root, proceed with the original logic
    if not matching_chords:
        for i in range(len(note_list)):
            note_set_frozen = frozenset(note_list)
            chord = chord_dict.get(note_set_frozen, None)
            if chord:
                matching_chords.append(chord)
            # Move the last element to the first position (check for inversions)
            note_list = [note_list[-1]] + note_list[:-1]
            print(f'Note List {i}: {note_list}')

    if matching_chords:
        print(f"All matching chords: {matching_chords}")
        return matching_chords[0]  # Return the first matching chord

    return "Unknown Chord"

def process_audio(file_path):
    # Load audio data (assumes a WAV file; if stereo, only the first channel is used)
    fs, data = scipy.io.wavfile.read(file_path)
    if data.ndim > 1:
        data = data[:, 0]
    
    N = len(data)
    window = np.hanning(N)
    data_windowed = data * window

    # Compute FFT and only consider the positive spectrum
    fft_complex = np.fft.fft(data_windowed)
    fft_magnitude = np.abs(fft_complex)[:N // 2]
    freqs = np.fft.fftfreq(N, d=1/fs)[:N // 2]

    # Limit the frequency range to 5000 Hz
    max_freq = 5000
    valid_indices = freqs <= max_freq
    freqs = freqs[valid_indices]
    fft_magnitude = fft_magnitude[valid_indices]

    # Detect peaks in the magnitude spectrum
    peak_freqs, peak_mags = get_peaks(freqs, fft_magnitude, threshold=0.2)
    # Remove harmonic frequencies
    fundamental_freqs = filter_harmonics(peak_freqs)

    # Convert fundamental frequencies to note names
    detected_notes = []
    note_labels = []
    for f in fundamental_freqs:
        note = frequency_to_note(f)
        if note is not None:
            detected_notes.append((f, note))
            note_labels.append(note)

    # Sort detected notes by frequency
    detected_notes.sort(key=lambda x: x[0])
    sorted_notes = [note for freq, note in detected_notes]

    chord = label_chord(sorted_notes)
    print("Detected fundamental frequencies:", fundamental_freqs)
    print("Detected notes:", sorted_notes)
    print("Identified chord:", chord)

    # Plot the results
    plt.figure(figsize=(12, 8))

    # Plot the windowed audio data
    plt.subplot(3, 1, 1)
    plt.plot(data_windowed)
    plt.title("Windowed Audio Data")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    # Plot the FFT magnitude spectrum
    plt.subplot(3, 1, 2)
    plt.plot(freqs, fft_magnitude)
    plt.scatter(peak_freqs, peak_mags, color='red')
    plt.title("FFT Magnitude Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

    # Plot the detected fundamental frequencies with note labels
    plt.subplot(3, 1, 3)
    plt.stem(fundamental_freqs, np.ones_like(fundamental_freqs), basefmt=" ")
    for i, freq in enumerate(fundamental_freqs):
        plt.text(freq, 1.05, note_labels[i], ha='center')
    plt.title("Detected Fundamental Frequencies")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Presence")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    audio_file_path = os.path.join(script_dir, 'GeneratedChords', 'guitar', 'Dm_guitar.wav')
    process_audio(audio_file_path)