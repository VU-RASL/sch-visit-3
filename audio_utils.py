import pyaudio
import numpy as np
import threading
import queue
import time
from scipy.signal import butter, filtfilt
import librosa
import wave
import os
from datetime import datetime

class AudioProcessor:
    def __init__(self, sample_rate=16000, chunk_size=1024, buffer_size=5):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size  # Buffer size in seconds
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.audio_thread = None
        self.p = None
        self.stream = None
        self.wav_file = None
        self.recording_start_time = None
        
        # Initialize feature extraction parameters
        self.n_mels = 40
        self.n_fft = 2048
        self.hop_length = 512
        
        # Initialize circular buffer for audio data
        self.buffer_samples = int(self.buffer_size * self.sample_rate)
        self.audio_buffer = np.zeros(self.buffer_samples)
        self.buffer_index = 0
        self.buffer_lock = threading.Lock()  # Add lock for thread safety
        
        # Initialize PyAudio
        try:
            self.p = pyaudio.PyAudio()
            
            # List available input devices
            info = self.p.get_host_api_info_by_index(0)
            num_devices = info.get('deviceCount')
            input_devices = []
            
            for i in range(num_devices):
                device_info = self.p.get_device_info_by_index(i)
                if device_info.get('maxInputChannels') > 0:
                    input_devices.append((i, device_info.get('name')))
            
            if not input_devices:
                raise RuntimeError("No audio input devices found")
            
            # Use the first available input device
            device_index = input_devices[0][0]
            print(f"Using audio device: {input_devices[0][1]}")
            
            # Initialize audio stream
            self.stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
        except Exception as e:
            self.cleanup()  # Clean up any partially initialized resources
            raise RuntimeError(f"Failed to initialize audio: {str(e)}")
    
    def start_recording(self, output_dir=None):
        """Start recording audio"""
        if not self.stream:
            raise RuntimeError("Audio stream not initialized")
            
        try:
            self.is_recording = True
            self.stream.start_stream()
            
            # Initialize WAV file if output directory is provided
            if output_dir:
                timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                wav_path = os.path.join(output_dir, f"audio_{timestamp}.wav")
                self.wav_file = wave.open(wav_path, 'wb')
                self.wav_file.setnchannels(1)
                self.wav_file.setsampwidth(2)  # 16-bit audio
                self.wav_file.setframerate(self.sample_rate)
                self.recording_start_time = time.time()
                print(f"Started recording audio to {wav_path}")
            
        except Exception as e:
            self.is_recording = False
            raise RuntimeError(f"Failed to start recording: {str(e)}")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        if self.is_recording:
            try:
                # Convert bytes to numpy array
                audio_data = np.frombuffer(in_data, dtype=np.float32)
                
                # Add to circular buffer with thread safety
                with self.buffer_lock:
                    # Check if we need to wrap around
                    if self.buffer_index + len(audio_data) > self.buffer_samples:
                        # Split the data into two parts
                        first_part = self.buffer_samples - self.buffer_index
                        second_part = len(audio_data) - first_part
                        
                        # Copy first part
                        self.audio_buffer[self.buffer_index:] = audio_data[:first_part]
                        # Copy second part
                        self.audio_buffer[:second_part] = audio_data[first_part:]
                    else:
                        # Copy all data
                        self.audio_buffer[self.buffer_index:self.buffer_index + len(audio_data)] = audio_data
                    
                    # Update buffer index
                    self.buffer_index = (self.buffer_index + len(audio_data)) % self.buffer_samples
                
                # Write to WAV file if recording
                if self.wav_file:
                    # Convert float32 to int16 for WAV file
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                    self.wav_file.writeframes(audio_int16.tobytes())
                
                # Put in queue for processing
                self.audio_queue.put((time.time(), audio_data))
                
            except Exception as e:
                print(f"Error in audio callback: {str(e)}")
            
        return (in_data, pyaudio.paContinue)
    
    def stop_recording(self):
        """Stop recording audio"""
        self.is_recording = False
        if self.stream:
            try:
                self.stream.stop_stream()
            except Exception as e:
                print(f"Error stopping stream: {str(e)}")
        
        # Close WAV file if recording
        if self.wav_file:
            try:
                self.wav_file.close()
                self.wav_file = None
                print("Stopped recording audio")
            except Exception as e:
                print(f"Error closing WAV file: {str(e)}")
    
    def get_audio_features(self):
        """Extract audio features from the buffer"""
        if not self.is_recording:
            return None
            
        try:
            # Get the most recent buffer data with thread safety
            with self.buffer_lock:
                if self.buffer_index > 0:
                    recent_audio = np.concatenate([
                        self.audio_buffer[self.buffer_index:],
                        self.audio_buffer[:self.buffer_index]
                    ])
                else:
                    recent_audio = self.audio_buffer.copy()
            
            # Extract features
            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=recent_audio,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Compute RMS energy
            rms = librosa.feature.rms(y=recent_audio)[0]
            
            # Compute zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(recent_audio)[0]
            
            # Combine features
            features = {
                'mel_spectrogram': mel_spec_db,
                'rms_energy': np.mean(rms),
                'zero_crossing_rate': np.mean(zcr),
                'timestamp': time.time()
            }
            
            return features
            
        except Exception as e:
            print(f"Error extracting audio features: {str(e)}")
            return None
    
    def cleanup(self):
        """Clean up audio resources"""
        self.stop_recording()
        
        if self.stream:
            try:
                self.stream.close()
            except Exception as e:
                print(f"Error closing stream: {str(e)}")
            self.stream = None
            
        if self.p:
            try:
                self.p.terminate()
            except Exception as e:
                print(f"Error terminating PyAudio: {str(e)}")
            self.p = None

def process_audio_data(audio_features):
    """Process audio features for ML prediction"""
    if audio_features is None:
        return None
        
    try:
        # Extract relevant features
        rms_energy = audio_features['rms_energy']
        zcr = audio_features['zero_crossing_rate']
        
        # Normalize features
        rms_norm = np.clip(rms_energy / 0.5, 0, 1)  # Assuming max RMS of 0.5
        zcr_norm = np.clip(zcr / 0.1, 0, 1)  # Assuming max ZCR of 0.1
        
        # Combine features (simple weighted average)
        prediction = 0.7 * rms_norm + 0.3 * zcr_norm
        
        return prediction
        
    except Exception as e:
        print(f"Error processing audio data: {str(e)}")
        return None 