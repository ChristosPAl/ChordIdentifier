Chord Identifier System
The Chord Identifier is a Python-based application designed to detect and label musical chords from audio input. It integrates digital signal processing techniques with Python programming to analyze and identify chords played on various instruments. The system supports real-time chord detection and includes tools for generating test data, processing audio files, and evaluating system accuracy.

Features
Audio Processing: Detects fundamental frequencies from audio files or live microphone input.
Chord Matching: Matches detected notes to predefined chords in a database, accounting for enharmonic equivalents and chord inversions.
Test Data Generation: Generates MIDI and WAV files for various chords, instruments, and voicings.
Real-Time Detection: Processes live audio input and displays detected chords in a graphical interface.
Repository Structure
Setup Instructions
Prerequisites
Python 3.x
Required Python libraries:
numpy
scipy
matplotlib
pyaudio
midiutil
itertools
tkinter
SoundFont file: FluidR3_GM.sf2 (included in the repository)
Installation
Clone the repository:

Install the required Python libraries:

Ensure the FluidR3_GM.sf2 SoundFont file is in the repository root.

Usage
1. Generate Test Data
Run the chord_generator.py script to generate MIDI and WAV files for various chords:

Generated files will be saved in the GeneratedChords directory.

2. Identify Chords from Audio Files
Use the chord_identifier.py script to process audio files and identify chords:

This script processes WAV files, extracts fundamental frequencies, and matches them to chords in ChordDB.csv.

3. Test the System
Run test_chord_identifier.py to evaluate the accuracy of the chord detection system:

The script compares detected chords against expected labels and prints the success rate.

4. Real-Time Chord Detection
Use mic_input_chord_identifier.py for real-time chord detection from microphone input:

The script displays detected chords and their frequency spectrum in a GUI.

Results
Accuracy:
Synth: 63.87%
Piano: 21.22%
Guitar: 8.61%
Real-time detection works with moderate success, displaying chords momentarily as they are played.
Challenges
Signal Processing: Handling complex spectral signatures of acoustic instruments.
Chord Matching: Accounting for enharmonic equivalents and chord inversions.
Real-Time Detection: Managing buffer updates and volume decay in live audio.
Future Improvements
Refine the signal processing pipeline for better frequency isolation.
Expand the chord database to include advanced chord types.
Integrate machine learning techniques to improve accuracy.
License
This project is licensed under the MIT License. See the LICENSE file for details.