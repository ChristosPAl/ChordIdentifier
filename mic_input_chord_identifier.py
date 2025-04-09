import threading
import time
import numpy as np
import pyaudio
import tempfile
import os
import scipy.io.wavfile as wavfile
import chord_identifier as CI   # Assumes chord_identifier.py is in the same folder
import queue
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.figure as mplfig

# Global variables
audio_segments = []
recording = True            # Flag to control recording
spectrum_queue = queue.Queue()  # to pass spectrum data from processing thread to main thread
chord_queue = queue.Queue()     # to pass detected chord to GUI

def record_audio(stream, chunk, segment_seconds, sample_rate):
    global audio_segments, recording
    num_chunks = int(sample_rate / chunk * segment_seconds)
    print("Record thread started", flush=True)
    while recording:
        frames = []
        for _ in range(num_chunks):
            try:
                data = stream.read(chunk, exception_on_overflow=False)
                frames.append(np.frombuffer(data, dtype=np.int16))
            except Exception as e:
                print(f"Error during stream.read: {e}", flush=True)
        if frames:
            segment = np.concatenate(frames)
            audio_segments.append(segment)
            rms = np.sqrt(np.mean(segment.astype(np.float64)**2))
            print(f"Recorded segment RMS: {rms:.2f}, length: {len(segment)}", flush=True)
        else:
            print("No frames captured in this segment", flush=True)
        time.sleep(0.1)

def process_audio_thread(sample_rate):
    global audio_segments, recording, spectrum_queue, chord_queue
    print("Process thread started", flush=True)
    gain_factor = 10.0  # Adjust this value as needed
    while recording:
        if audio_segments:
            segment = audio_segments.pop(0)
            
            # Boost the signal and avoid clipping for 16-bit audio:
            max_val = 32767
            min_val = -32768
            amplified_segment = np.clip(segment * gain_factor, min_val, max_val).astype(np.int16)
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_filename = tmp.name
            wavfile.write(tmp_filename, sample_rate, amplified_segment)
            
            # Process the amplified audio segment
            note_list = CI.process_audio(tmp_filename, plot_data=False)
            detected_chord = CI.label_chord(note_list)
            print("Detected chord:", detected_chord, flush=True)
            chord_queue.put(detected_chord)
            os.remove(tmp_filename)
            
            # Compute FFT for spectrum visualization using the amplified signal
            N_seg = len(amplified_segment)
            fft_complex = np.fft.fft(amplified_segment)
            fft_magnitude = np.abs(fft_complex)[:N_seg // 2]
            freqs = np.fft.fftfreq(N_seg, d=1/sample_rate)[:N_seg // 2]
            spectrum_queue.put((freqs, fft_magnitude))
        else:
            time.sleep(0.1)

############################
# Tkinter GUI Application  #
############################
class ChordGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Chord Identifier")
        self.geometry("800x600")
        
        # Label for detected chord
        self.chord_var = tk.StringVar(value="Detected Chord: None")
        self.label = ttk.Label(self, textvariable=self.chord_var, font=("Helvetica", 28))
        self.label.pack(pady=20)
        
        # Set up a matplotlib figure embedded in the Tkinter window
        self.fig = mplfig.Figure(figsize=(8, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Spectrum of Recorded Segment")
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Magnitude")
        self.ax.set_xlim(0, 5000)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Start periodic update of chord and spectrum display
        self.update_gui()
    
    def update_gui(self):
        # Update chord if available
        try:
            chord = chord_queue.get_nowait()
            self.chord_var.set("Detected Chord: " + str(chord))
        except queue.Empty:
            pass
        
        # Update spectrum if available
        try:
            freqs, fft_magnitude = spectrum_queue.get_nowait()
            self.ax.cla()  # Clear previous plot
            self.ax.plot(freqs, fft_magnitude)
            self.ax.set_title("Spectrum of Recorded Segment")
            self.ax.set_xlabel("Frequency (Hz)")
            self.ax.set_ylabel("Magnitude")
            self.ax.set_xlim(0, 5000)
            self.canvas.draw()
        except queue.Empty:
            pass
        
        # Call this method again after 100 ms
        self.after(100, self.update_gui)

def start_audio_processing():
    global p, stream, record_thread, process_thread
    sample_rate = 44100    # Adjust as needed
    chunk = 1024           # Number of samples per read
    segment_seconds = 0.5  # Length of each recorded segment (sec)
    
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate,
                    input=True, frames_per_buffer=chunk)
    
    record_thread = threading.Thread(target=record_audio, args=(stream, chunk, segment_seconds, sample_rate))
    process_thread = threading.Thread(target=process_audio_thread, args=(sample_rate,))
    record_thread.start()
    process_thread.start()
    return record_thread, process_thread, stream, p

def stop_audio_processing(record_thread, process_thread, stream, p):
    global recording
    recording = False
    record_thread.join()
    process_thread.join()
    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == "__main__":
    # Start audio processing threads
    record_thread, process_thread, stream, p = start_audio_processing()
    
    # Create and run the Tkinter GUI
    gui = ChordGUI()
    try:
        gui.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        stop_audio_processing(record_thread, process_thread, stream, p)