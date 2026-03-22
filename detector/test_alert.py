#!/usr/bin/env python
"""
Test script to verify Guardian Net integration
Run this to test if alerts are being sent correctly
"""

from guardian_integration import GuardianAlertSender
import time

def main():
    print("\n" + "="*60)
    print("🧪 GUARDIAN NET - ALERT TESTER")
    print("="*60)
    
    # Get patient ID from user
    try:
        patient_id = int(input("Enter patient ID (default: 1): ") or "1")
    except:
        patient_id = 1
    
    # Initialize sender
    sender = GuardianAlertSender(patient_id=patient_id)
    
    # Test connection
    print("\n📡 Testing connection to server...")
    if sender.test_connection():
        print("✅ Server is reachable!")
        
        # Send test alert
        print("\n📱 Sending test alert...")
        sender.send_alert(
            alert_type="test",
            message="This is a test alert from the detector",
            confidence=0.99
        )
        
        print("\n✅ Test complete! Check the web dashboard.")
    else:
        print("\n❌ Cannot connect to server!")
        print("\n   Make sure the backend is running:")
        print("   cd backend")
        print("   node server.js")
        print()
    
    print("="*60)

if __name__ == "__main__":
    main()