import pyaudio
import numpy as np
import threading
import queue
import time
from scipy.signal import butter, filtfilt
import librosa

class AudioProcessor:
    def __init__(self, sample_rate=16000, chunk_size=1024, buffer_size=5):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size  # Buffer size in seconds
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.audio_thread = None
        self.p = pyaudio.PyAudio()
        
        # Initialize audio stream
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        
        # Initialize circular buffer for audio data
        self.buffer_samples = int(self.buffer_size * self.sample_rate)
        self.audio_buffer = np.zeros(self.buffer_samples)
        self.buffer_index = 0
        
        # Initialize feature extraction parameters
        self.n_mels = 40
        self.n_fft = 2048
        self.hop_length = 512
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        if self.is_recording:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            # Add to circular buffer
            self.audio_buffer[self.buffer_index:self.buffer_index + len(audio_data)] = audio_data
            self.buffer_index = (self.buffer_index + len(audio_data)) % self.buffer_samples
            
            # Put in queue for processing
            self.audio_queue.put((time.time(), audio_data))
            
        return (in_data, pyaudio.paContinue)
    
    def start_recording(self):
        """Start recording audio"""
        self.is_recording = True
        self.stream.start_stream()
        
    def stop_recording(self):
        """Stop recording audio"""
        self.is_recording = False
        self.stream.stop_stream()
        
    def get_audio_features(self):
        """Extract audio features from the buffer"""
        if not self.is_recording:
            return None
            
        # Get the most recent buffer data
        if self.buffer_index > 0:
            recent_audio = np.concatenate([
                self.audio_buffer[self.buffer_index:],
                self.audio_buffer[:self.buffer_index]
            ])
        else:
            recent_audio = self.audio_buffer.copy()
            
        # Extract features
        try:
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
        self.stream.close()
        self.p.terminate()

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