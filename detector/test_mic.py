#!/usr/bin/env python
"""
Test microphone and speech recognition
"""

import speech_recognition as sr
import time

def test_microphone():
    print("\n" + "="*50)
    print("🎤 MICROPHONE TEST")
    print("="*50)
    
    recognizer = sr.Recognizer()
    
    try:
        microphone = sr.Microphone()
        print("✅ Microphone found")
    except Exception as e:
        print(f"❌ No microphone found: {e}")
        return
    
    print("\n🔊 Calibrating for ambient noise...")
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)
        print("✅ Calibration complete")
    
    print("\n🎤 Speak now (you have 5 seconds)...")
    print("   Try saying: 'help', 'emergency', 'സഹായം', 'मदद'")
    
    try:
        with microphone as source:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
            print("✅ Audio captured!")
            
            print("\n🔍 Recognizing speech...")
            
            # Try English
            try:
                text = recognizer.recognize_google(audio, language='en-IN')
                print(f"📝 English: '{text}'")
            except:
                print("⚠️ Could not recognize English")
            
            # Try Malayalam
            try:
                text = recognizer.recognize_google(audio, language='ml-IN')
                print(f"📝 Malayalam: '{text}'")
            except:
                pass
            
            # Try Hindi
            try:
                text = recognizer.recognize_google(audio, language='hi-IN')
                print(f"📝 Hindi: '{text}'")
            except:
                pass
            
            print("\n✅ Microphone test successful!")
            
    except sr.WaitTimeoutError:
        print("❌ No speech detected - timeout")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    test_microphone()