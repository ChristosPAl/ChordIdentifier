import os
import csv
import random
from midiutil import MIDIFile
from midi2audio import FluidSynth

# List of standard note names in Western music.
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def normalize_note_name(note_name):
    """
    Normalize note names to their equivalent standard notes.
    For example, convert 'C##' to 'D', 'E#' to 'F', 'Fb' to 'E', etc.
    
    Parameters:
      note_name (str): The note name string possibly with accidentals.
      
    Returns:
      str: The normalized note name.
    """
    # Extract the base note (first character) and the accidental (everything after)
    note = note_name[0]
    accidental = note_name[1:] if len(note_name) > 1 else ''
    
    # Adjust the note based on the accidental, using modulo arithmetic to cycle notes
    if accidental == '##':
        note = NOTE_NAMES[(NOTE_NAMES.index(note) + 2) % 12]
    elif accidental == '#':
        note = NOTE_NAMES[(NOTE_NAMES.index(note) + 1) % 12]
    elif accidental == 'b':
        note = NOTE_NAMES[(NOTE_NAMES.index(note) - 1) % 12]
    elif accidental == 'bb':
        note = NOTE_NAMES[(NOTE_NAMES.index(note) - 2) % 12]
    
    return note

def note_name_to_midi(note_name, octave_range=(2, 5)):
    """
    Convert a note name (e.g., 'C#') to a MIDI note number.
    The octave is chosen randomly within the provided octave_range.
    
    Parameters:
      note_name (str): The note name (with possible accidentals).
      octave_range (tuple): Two-tuple which sets the lower and upper bound for octave.
      
    Returns:
      int: The derived MIDI note number.
    """
    # Normalize the note name first (handles accidentals)
    note = normalize_note_name(note_name)
    # Choose a random octave between the specified range values
    octave = random.randint(*octave_range)

    # Convert note name to its index and compute the MIDI note value.
    note_index = NOTE_NAMES.index(note)
    midi_note = note_index + (octave + 1) * 12
    return midi_note

def generate_chord_midi(chord_name, note_values, instrument, output_dir):
    """
    Generate a MIDI file for the given chord.
    
    Parameters:
      chord_name (str): The name of the chord (used for track naming).
      note_values (list): A list of MIDI note numbers composing the chord.
      instrument (str): Instrument type (e.g., 'piano', 'guitar', or 'synth').
      output_dir (str): Directory to save the generated MIDI file.
      
    Returns:
      str: Path to the saved MIDI file.
    """
    # Initialize a new MIDI file with one track
    midi = MIDIFile(1)
    track = 0
    time = 0     # Start time at beginning
    channel = 0
    volume = 100 # Maximum volume
    duration = 4 # Playback duration (4 seconds) per chord

    # Add track name and tempo to the MIDI track
    midi.addTrackName(track, time, chord_name)
    midi.addTempo(track, time, 120)

    # Define an instrument mapping to decide the MIDI program number based on the instrument name
    instrument_map = {
        'piano': 0,   # Acoustic Grand Piano
        'guitar': 25, # Acoustic Guitar (steel)
        'synth': 81   # Lead 2 (sawtooth)
    }
    program = instrument_map.get(instrument, 0)  # If instrument is not found, default to piano

    # Set the instrument for the specified channel
    midi.addProgramChange(track, channel, time, program)

    # Add each note into the MIDI file for the chord
    for note in note_values:
        midi.addNote(track, channel, note, time, duration, volume)

    # Sanitize the chord name to create a safe filename (replacing problematic characters)
    safe_chord_name = chord_name.replace('/', '_').replace('\\', '_')
    output_path = os.path.join(output_dir, f"{safe_chord_name}_{instrument}.mid")
    
    # Write the MIDI data to a file
    with open(output_path, 'wb') as output_file:
        midi.writeFile(output_file)

    return output_path

def render_midi_to_wav(midi_path, soundfont_path, output_path, instrument):
    """
    Convert a MIDI file to a WAV file using a given soundfont. 
    After conversion, the MIDI file is removed.
    
    Parameters:
      midi_path (str): File path to the source MIDI file.
      soundfont_path (str): Path to the soundfont file for conversion.
      output_path (str): Path where to save the generated WAV file.
      instrument (str): Instrument name (passed for potential future use).
    """
    # Get the directory where the current script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Build the relative path to the fluidsynth executable
    fluidsynth_dir = os.path.join(script_dir, "FluidSynth", "bin")
    os.environ["PATH"] += os.pathsep + fluidsynth_dir
    
    # Create a FluidSynth object using the provided soundfont
    fs = FluidSynth(soundfont_path)
    # Convert the MIDI file to a WAV file
    fs.midi_to_audio(midi_path, output_path)
    # Clean up by removing the temporary MIDI file
    os.remove(midi_path)

def load_chord_db(file_path):
    """
    Load the chord database from a CSV file.
    
    The CSV is expected to have columns: 'CHORD_ROOT', 'CHORD_TYPE', and 'NOTE_NAMES'.
    NOTE_NAMES should be a comma-separated list of note names.
    
    Parameters:
      file_path (str): Path to the chord database CSV file.
      
    Returns:
      list: A list of tuples, each containing the chord name and a list of MIDI note numbers.
    """
    chords = []
    with open(file_path, 'r') as csvfile:
        # Use ';' as the CSV delimiter
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            # Concatenate chord root and chord type to form the full chord name
            chord_name = f"{row['CHORD_ROOT']}{row['CHORD_TYPE']}"
            # Convert comma-separated note names to MIDI note numbers
            note_values = [note_name_to_midi(note) for note in row['NOTE_NAMES'].split(',')]
            chords.append((chord_name, note_values))
    return chords

def main():
    """
    Main function performing the three key tasks:
      1. Load a chord database.
      2. Generate MIDI files for each chord with different instruments.
      3. Convert the MIDI files to WAV files.
    
    The outputs are organized into separate directories based on instruments.
    """
    # Get the directory where the current script is located.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Build paths for the chord database and the soundfont file.
    chord_db_path = os.path.join(script_dir, 'ChordDB.csv')
    soundfont_path = os.path.join(script_dir, 'FluidR3_GM.sf2')  # Update this path to the new soundfont
    # Load chord data from the CSV database.
    chords = load_chord_db(chord_db_path)

    # List of instruments to be used for MIDI generation.
    instruments = ['guitar', 'piano', 'synth']
    # Base directory for storing generated chord audio files.
    output_base_dir = os.path.join(script_dir, 'GeneratedChords')

    # Process each instrument type separately.
    for instrument in instruments:
        output_dir = os.path.join(output_base_dir, instrument)
        # Create output directory if it does not exist.
        os.makedirs(output_dir, exist_ok=True)

        # Generate and render each chord for the current instrument.
        for chord_name, note_values in chords:
            # Generate a MIDI file for the chord.
            midi_path = generate_chord_midi(chord_name, note_values, instrument, output_dir)
            # Build the target WAV file path.
            wav_path = os.path.join(output_dir, f"{chord_name}_{instrument}.wav")
            # Convert the MIDI file to a WAV file.
            render_midi_to_wav(midi_path, soundfont_path, wav_path, instrument)

# Ensure that the main function is only executed when the script is run directly.
if __name__ == '__main__':
    main()