"""
Guardian Net Integration Module
This adds alert-sending capability to your existing detection scripts
"""

import requests
import json
import time
from datetime import datetime

class GuardianAlertSender:
    def __init__(self, server_url="http://localhost:3000", patient_id=1):
        """
        Initialize the alert sender
        1
        Args:
            server_url: URL of the Guardian Net backend
            patient_id: ID of the patient being monitored (from database)
        """
        self.server_url = server_url
        self.patient_id = patient_id
        self.alert_endpoint = f"{server_url}/api/detector/alert"
        self.last_alert_time = 0
        self.alert_cooldown = 30  # seconds between alerts
        self.alert_count = 0
        
    def send_alert(self, alert_type, message=None, confidence=None):
        """
        Send an emergency alert to the backend
        
        Args:
            alert_type (str): Type of alert ('fall', 'gesture', 'voice')
            message (str, optional): Custom message
            confidence (float, optional): Detection confidence score
            
        Returns:
            bool: True if successful, False otherwise
        """
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_alert_time < self.alert_cooldown:
            remaining = self.alert_cooldown - (current_time - self.last_alert_time)
            print(f"⏱️  Alert cooldown: {remaining:.1f}s remaining")
            return False
        
        try:
            # Convert numpy float32 to Python float if needed
            if confidence is not None:
                # Check if it's a numpy type and convert
                if hasattr(confidence, 'item'):
                    confidence = confidence.item()
                # Ensure it's a regular Python float
                confidence = float(confidence)
            
            # Prepare the payload
            payload = {
                "patient_id": self.patient_id,
                "alert_type": alert_type,
                "message": message,
                "confidence": confidence
            }
            
            # Remove None values
            payload = {k: v for k, v in payload.items() if v is not None}
            
            # Send to backend
            response = requests.post(
                self.alert_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=3
            )
            
            if response.status_code == 200:
                self.last_alert_time = current_time
                self.alert_count += 1
                result = response.json()
                
                print("\n" + "="*60)
                print(f"✅ EMERGENCY ALERT SENT!")
                print(f"   Type: {alert_type.upper()}")
                print(f"   Patient ID: {self.patient_id}")
                if confidence:
                    print(f"   Confidence: {confidence:.2f}")
                print(f"   Alert #{self.alert_count}")
                print("="*60 + "\n")
                return True
            else:
                print(f"❌ Server error: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.ConnectionError:
            print("\n⚠️  Cannot connect to Guardian Net server")
            print("   Make sure the backend is running:")
            print("   cd backend && node server.js")
            print("")
            return False
        except Exception as e:
            print(f"❌ Error sending alert: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_connection(self):
        """Test connection to the server"""
        try:
            response = requests.get(f"{self.server_url}", timeout=2)
            return response.status_code == 200
        except:
            return False


# Example usage if run directly
if __name__ == "__main__":
    # Test the alert sender
    sender = GuardianAlertSender(patient_id=1)
    
    if sender.test_connection():
        print("✅ Connected to Guardian Net server")
        sender.send_alert("test", "This is a test alert", 0.95)
    else:
        print("❌ Cannot connect to server")