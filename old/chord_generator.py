import os
import csv
import random
from midiutil import MIDIFile
import fluidsynth

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def normalize_note_name(note_name):
    """
    Normalize note names to their equivalent standard notes.
    For example, 'C##' becomes 'D', 'E#' becomes 'F', 'Fb' becomes 'E', etc.
    """
    note = note_name[0]
    accidental = note_name[1:] if len(note_name) > 1 else ''
    
    if accidental == '##':
        note = NOTE_NAMES[(NOTE_NAMES.index(note) + 2) % 12]
    elif accidental == '#':
        note = NOTE_NAMES[(NOTE_NAMES.index(note) + 1) % 12]
    elif accidental == 'b':
        note = NOTE_NAMES[(NOTE_NAMES.index(note) - 1) % 12]
    elif accidental == 'bb':
        note = NOTE_NAMES[(NOTE_NAMES.index(note) - 2) % 12]
    
    return note

def note_name_to_midi(note_name, octave_range=(3, 5)):
    """
    Convert a note name (e.g., 'C#') to a MIDI note number.
    Assign a random octave within the specified range if the octave is not provided.
    """
    note = normalize_note_name(note_name)
    octave = random.randint(*octave_range)

    note_index = NOTE_NAMES.index(note)
    midi_note = note_index + (octave + 1) * 12
    return midi_note

def generate_chord_midi(chord_name, note_values, instrument, output_dir):
    midi = MIDIFile(1)
    track = 0
    time = 0
    channel = 0
    volume = 100
    duration = 4  # 4 seconds

    midi.addTrackName(track, time, chord_name)
    midi.addTempo(track, time, 120)

    for note in note_values:
        midi.addNote(track, channel, note, time, duration, volume)

    # Replace problematic characters in the chord name
    safe_chord_name = chord_name.replace('/', '_').replace('\\', '_')

    output_path = os.path.join(output_dir, f"{safe_chord_name}_{instrument}.mid")
    with open(output_path, 'wb') as output_file:
        midi.writeFile(output_file)

    return output_path

def render_midi_to_wav(midi_path, soundfont_path, output_path, instrument):
    fs = fluidsynth.Synth()
    fs.start('dsound')  # Let FluidSynth choose the best driver
    sfid = fs.sfload(soundfont_path)
    
    instrument_map = {
        'piano': 0,  # Acoustic Grand Piano
        'guitar': 24,  # Acoustic Guitar (nylon)
        'synth': 80  # Lead 1 (square)
    }
    
    program = instrument_map.get(instrument, 0)  # Default to piano if unknown
    fs.program_select(0, sfid, 0, program)

    # Convert MIDI to WAV using an external tool
    os.system(f'timidity "{midi_path}" -Ow -o "{output_path}"')

    fs.delete()


def load_chord_db(file_path):
    chords = []
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            chord_name = f"{row['CHORD_ROOT']}{row['CHORD_TYPE']}"
            note_values = [note_name_to_midi(note) for note in row['NOTE_NAMES'].split(',')]
            chords.append((chord_name, note_values))
    return chords

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chord_db_path = os.path.join(script_dir, 'ChordDB.csv')
    soundfont_path = os.path.join(script_dir, 'GeneralUser-GS.sf2')
    chords = load_chord_db(chord_db_path)

    instruments = ['guitar', 'piano', 'synth']
    output_base_dir = os.path.join(script_dir, 'GeneratedChords')

    for instrument in instruments:
        output_dir = os.path.join(output_base_dir, instrument)
        os.makedirs(output_dir, exist_ok=True)

        for chord_name, note_values in chords:
            midi_path = generate_chord_midi(chord_name, note_values, instrument, output_dir)
            wav_path = os.path.join(output_dir, f"{chord_name}_{instrument}.wav")
            render_midi_to_wav(midi_path, soundfont_path, wav_path, instrument)

if __name__ == '__main__':
    main()