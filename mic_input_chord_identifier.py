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
audio_segments = []           # List to hold recorded audio segments
recording = True              # Flag to control recording (used to stop threads gracefully)
spectrum_queue = queue.Queue()  # Queue to pass FFT spectrum data from processing thread to GUI
chord_queue = queue.Queue()     # Queue to pass detected chord information to GUI

def record_audio(stream, chunk, segment_seconds, sample_rate):
    """
    Function to record audio continuously from the microphone.
    Reads audio chunks from the PyAudio stream and assembles them into segments of fixed duration.
    Each segment's RMS is computed and printed for debugging.
    """
    global audio_segments, recording
    # Calculate the number of chunks needed per segment based on provided duration
    num_chunks = int(sample_rate / chunk * segment_seconds)
    print("Record thread started", flush=True)
    while recording:
        frames = []
        # Read the required number of chunks to form a segment
        for _ in range(num_chunks):
            try:
                data = stream.read(chunk, exception_on_overflow=False)
                # Convert byte data to a numpy array of int16
                frames.append(np.frombuffer(data, dtype=np.int16))
            except Exception as e:
                print(f"Error during stream.read: {e}", flush=True)
        if frames:
            # Concatenate all frames into a single segment
            segment = np.concatenate(frames)
            audio_segments.append(segment)
            # Calculate the RMS to assess the signal level
            rms = np.sqrt(np.mean(segment.astype(np.float64)**2))
            print(f"Recorded segment RMS: {rms:.2f}, length: {len(segment)}", flush=True)
        else:
            print("No frames captured in this segment", flush=True)
        # Short pause to prevent overloading CPU
        time.sleep(0.1)

def process_audio_thread(sample_rate):
    """
    Function to process recorded audio segments.
    Amplifies the signal to improve the identification process,
    saves it temporarily as a WAV file, processes it to identify the chord,
    and computes its FFT for spectrum visualization.
    The detected chord and spectrum data are sent to the GUI via queues.
    """
    global audio_segments, recording, spectrum_queue, chord_queue
    print("Process thread started", flush=True)
    gain_factor = 10.0  # Gain factor to boost signal (adjust as needed)
    while recording:
        if audio_segments:
            # Retrieve the first recorded segment in the list
            segment = audio_segments.pop(0)
            
            # Amplify the signal and avoid clipping for 16-bit audio
            max_val = 32767
            min_val = -32768
            amplified_segment = np.clip(segment * gain_factor, min_val, max_val).astype(np.int16)
            
            # Create a temporary WAV file to store the amplified audio segment
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_filename = tmp.name
            # Write the audio segment to the temporary file
            wavfile.write(tmp_filename, sample_rate, amplified_segment)
            
            # Process the audio using functions from chord_identifier module
            note_list = CI.process_audio(tmp_filename, plot_data=False)
            detected_chord = CI.label_chord(note_list)
            print("Detected chord:", detected_chord, flush=True)
            # Put the detected chord into the chord_queue for usage in the GUI
            chord_queue.put(detected_chord)
            
            # Remove the temporary file after processing
            os.remove(tmp_filename)
            
            # Compute FFT for the amplified audio segment for visualization
            N_seg = len(amplified_segment)
            fft_complex = np.fft.fft(amplified_segment)
            fft_magnitude = np.abs(fft_complex)[:N_seg // 2]  # Single-sided spectrum
            freqs = np.fft.fftfreq(N_seg, d=1/sample_rate)[:N_seg // 2]
            # Put FFT result (frequencies and their magnitudes) in the spectrum_queue
            spectrum_queue.put((freqs, fft_magnitude))
        else:
            # No audio segment available; pause briefly
            time.sleep(0.1)

############################
# Tkinter GUI Application  #
############################
# class ChordGUI(tk.Tk):
#     """
#     Tkinter-based GUI for displaying the detected chord and audio spectrum.
#     Updates periodically by polling the chord_queue and spectrum_queue.
#     """
#     def __init__(self):
#         super().__init__()
#         self.title("Chord Identifier")
#         self.geometry("800x600")
        
#         # Label to display the latest detected chord
#         self.chord_var = tk.StringVar(value="Detected Chord: None")
#         self.label = ttk.Label(self, textvariable=self.chord_var, font=("Helvetica", 28))
#         self.label.pack(pady=20)
        
#         # Set up a matplotlib figure embedded in the Tkinter window for spectrum display
#         self.fig = mplfig.Figure(figsize=(8, 4), dpi=100)
#         self.ax = self.fig.add_subplot(111)
#         self.ax.set_title("Spectrum of Recorded Segment")
#         self.ax.set_xlabel("Frequency (Hz)")
#         self.ax.set_ylabel("Magnitude")
#         self.ax.set_xlim(0, 5000)  # Limit frequency axis to 5000 Hz
#         self.canvas = FigureCanvasTkAgg(self.fig, master=self)
#         self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
#         # Start the GUI periodic updates
#         self.update_gui()
    
#     def update_gui(self):
#         """
#         Periodically updates the GUI by polling for:
#         - A new detected chord in chord_queue, updating the label if found.
#         - New FFT spectrum data in spectrum_queue, updating the plot if available.
#         Reschedules itself every 100 ms.
#         """
#         # Update chord if available
#         try:
#             chord = chord_queue.get_nowait()
#             self.chord_var.set("Detected Chord: " + str(chord))
#         except queue.Empty:
#             pass
        
#         # Update spectrum if available
#         try:
#             freqs, fft_magnitude = spectrum_queue.get_nowait()
#             self.ax.cla()  # Clear previous plot
#             self.ax.plot(freqs, fft_magnitude)
#             self.ax.set_title("Spectrum of Recorded Segment")
#             self.ax.set_xlabel("Frequency (Hz)")
#             self.ax.set_ylabel("Magnitude")
#             self.ax.set_xlim(0, 5000)
#             self.canvas.draw()
#         except queue.Empty:
#             pass
        
#         # Schedule the next update in 100 milliseconds
#         self.after(100, self.update_gui)

class ChordGUI(tk.Tk):
    """
    Tkinter-based GUI for displaying the detected chord and a real-looking equalizer.
    Updates periodically by polling the chord_queue and spectrum_queue.
    """
    def __init__(self):
        super().__init__()
        self.title("Chord Identifier")
        self.geometry("800x600")
        self.configure(bg="#2d2d2d")  # Dark background for a modern look
        
        # Label to display the latest detected chord with styling
        self.chord_var = tk.StringVar(value="Detected Chord: None")
        self.label = ttk.Label(
            self, textvariable=self.chord_var, 
            font=("Helvetica", 32, "bold"),
            foreground="#ffffff",
            background="#2d2d2d"
        )
        self.label.pack(pady=20)
        
        # Set up a matplotlib figure embedded in the Tkinter window for the equalizer
        self.fig = mplfig.Figure(figsize=(8, 4), dpi=100, facecolor="#2d2d2d")
        self.ax = self.fig.add_subplot(111, facecolor="#3e3e3e")
        self.ax.set_title("Spectrum (Equalizer)", color="#ffffff", fontsize=14)
        self.ax.set_xlabel("Frequency (Hz)", color="#ffffff")
        self.ax.set_ylabel("Magnitude", color="#ffffff")
        self.ax.tick_params(axis='x', colors='#ffffff')
        self.ax.tick_params(axis='y', colors='#ffffff')
        self.ax.set_xlim(0, 5000)  # Limit frequency axis to 5000 Hz
        # Fixed maximum value for the equalizer’s vertical scale (adjust as needed)
        self.EQ_MAX = 1000  
        self.ax.set_ylim(0, self.EQ_MAX)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Start the GUI periodic updates
        self.update_gui()
    
    def update_gui(self):
        """
        Periodically updates the GUI by polling for:
        - A new detected chord in chord_queue, updating the label if found.
        - New FFT spectrum data in spectrum_queue, updating the equalizer-style plot.
        Reschedules itself every 100 ms.
        """
        # Update chord if available
        try:
            chord = chord_queue.get_nowait()
            self.chord_var.set("Detected Chord: " + str(chord))
        except queue.Empty:
            pass
        
        # Update equalizer display if spectrum data is available
        try:
            freqs, fft_magnitude = spectrum_queue.get_nowait()
            
            # Process FFT data into bins for equalizer visualization
            num_bins = 50  # Number of equalizer bars
            max_freq = 5000
            bins = np.linspace(0, max_freq, num_bins+1)
            bin_centers = []
            binned_values = []
            for i in range(num_bins):
                indices = np.where((freqs >= bins[i]) & (freqs < bins[i+1]))[0]
                # Compute average magnitude within each bin if data exists
                value = np.mean(fft_magnitude[indices]) if indices.size > 0 else 0
                binned_values.append(value)
                bin_centers.append((bins[i] + bins[i+1]) / 2)
            
            # Normalize binned values so the highest value reaches EQ_MAX
            max_bin = max(binned_values) if binned_values and max(binned_values) > 0 else 1
            normalized_values = [ (v / max_bin) * self.EQ_MAX for v in binned_values ]
            
            # Clear the axis and plot equalizer bars with constant y-scale
            self.ax.cla()
            self.ax.bar(bin_centers, normalized_values, 
                        width=(max_freq/num_bins)*0.8, color="#1f77b4")
            self.ax.set_title("Spectrum (Equalizer)", color="#ffffff", fontsize=14)
            self.ax.set_xlabel("Frequency (Hz)", color="#ffffff")
            self.ax.set_ylabel("Magnitude", color="#ffffff")
            self.ax.tick_params(axis='x', colors='#ffffff')
            self.ax.tick_params(axis='y', colors='#ffffff')
            self.ax.set_xlim(0, max_freq)
            self.ax.set_ylim(0, self.EQ_MAX)  # Keep vertical scale fixed
            self.canvas.draw()
        except queue.Empty:
            pass
        
        # Schedule the next update in 100 milliseconds
        self.after(100, self.update_gui)

def start_audio_processing():
    """
    Initialize audio input and start both recording and processing threads.
    Returns the thread objects and audio stream handles for later cleanup.
    """
    global p, stream, record_thread, process_thread
    sample_rate = 44100    # Audio sample rate
    chunk = 1024           # Number of samples to read per chunk
    segment_seconds = 0.5  # Duration of each audio segment in seconds
    
    p = pyaudio.PyAudio()
    # Open an audio stream using PyAudio with 16-bit mono input
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate,
                    input=True, frames_per_buffer=chunk)
    
    # Create and start the audio recording thread
    record_thread = threading.Thread(target=record_audio, args=(stream, chunk, segment_seconds, sample_rate))
    # Create and start the audio processing thread
    process_thread = threading.Thread(target=process_audio_thread, args=(sample_rate,))
    record_thread.start()
    process_thread.start()
    return record_thread, process_thread, stream, p

def stop_audio_processing(record_thread, process_thread, stream, p):
    """
    Gracefully stops the audio processing by signaling the threads to stop,
    joining the threads, and then closing the audio stream.
    """
    global recording
    recording = False  # Signal threads to stop
    record_thread.join()
    process_thread.join()
    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == "__main__":
    # Start audio processing threads
    record_thread, process_thread, stream, p = start_audio_processing()
    
    # Create and run the Tkinter GUI application
    gui = ChordGUI()
    try:
        gui.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        # Ensure that the audio processing is properly stopped on exit
        stop_audio_processing(record_thread, process_thread, stream, p)