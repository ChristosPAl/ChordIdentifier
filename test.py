import numpy as np
import math
import sys

import scipy.io.wavfile
import scipy.signal

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

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
    return peak_freqs[sort_idx]

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

def label_chord(note_set):
    """
    Label a chord by checking for simple major or minor triad patterns.
    The algorithm assumes note_set (a set of note names, e.g., {'C', 'E', 'G'}).
    """
    def note_to_num(note):
        return NOTE_NAMES.index(note)

    note_nums = sorted([note_to_num(n) for n in note_set])
    for root in note_nums:
        # Compute intervals relative to the root note
        intervals = sorted(((n - root) % 12) for n in note_nums)
        if set(intervals) == set([0, 4, 7]):
            return f"{NOTE_NAMES[root]} Major"
        if set(intervals) == set([0, 3, 7]):
            return f"{NOTE_NAMES[root]} Minor"
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

    # Detect peaks in the magnitude spectrum
    peak_freqs = get_peaks(freqs, fft_magnitude, threshold=0.2)
    # Remove harmonic frequencies
    fundamental_freqs = filter_harmonics(peak_freqs)

    # Convert fundamental frequencies to note names
    detected_notes = set()
    for f in fundamental_freqs:
        note = frequency_to_note(f)
        if note is not None:
            detected_notes.add(note)

    chord = label_chord(detected_notes)
    print("Detected fundamental frequencies:", fundamental_freqs)
    print("Detected notes:", detected_notes)
    print("Identified chord:", chord)

if __name__ == '__main__':
    process_audio('ChordSamples/CMaj')
    # if len(sys.argv) < 2:
    #     print("Usage: python test.py <path_to_audio_file.wav>")
    # else:
    #     process_audio(sys.argv[1])