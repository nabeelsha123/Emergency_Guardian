
import numpy as np
import soundfile as sf
import os
from datetime import datetime

class AudioRecorder:
    """Utility for recording audio samples"""
    
    def __init__(self, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.recordings_dir = "recordings"
        os.makedirs(self.recordings_dir, exist_ok=True)
    
    def record_sample(self, duration=5, label=""):
        """Record an audio sample"""
        print(f"Recording for {duration} seconds...")
        recording = sd.rec(int(duration * self.sample_rate),
                          samplerate=self.sample_rate,
                          channels=self.channels)
        sd.wait()
        
        # Save recording
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.recordings_dir}/{label}_{timestamp}.wav"
        sf.write(filename, recording, self.sample_rate)
        
        print(f"Saved to {filename}")
        return filename, recording

class AlertSystem:
    """Emergency alert system"""
    
    def __init__(self):
        self.emergency_contacts = []  # Add contact numbers/emails
        self.alert_history = []
    
    def send_alert(self, message, confidence, location=None):
        """Send emergency alert"""
        alert = {
            'timestamp': datetime.now(),
            'message': message,
            'confidence': confidence,
            'location': location,
            'sent': False
        }
        
        # TODO: Implement actual alert sending
        # SMS, Email, Push notifications, etc.
        
        print(f"🚨 SENDING EMERGENCY ALERT: {message}")
        print(f"   Confidence: {confidence:.2%}")
        print(f"   Location: {location}")
        
        alert['sent'] = True
        self.alert_history.append(alert)
        
    def play_voice_response(self):
        """Play automated voice response"""
        try:
            # Generate calming message
            message = "Help is on the way. Please stay calm. Emergency services have been notified."
            
            # TODO: Implement text-to-speech
            print(f"🎧 Playing voice response: {message}")
            
        except Exception as e:
            print(f"Error playing voice response: {e}")