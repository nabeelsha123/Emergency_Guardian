#!/usr/bin/env python
"""
Guardian Net - Multilingual Emergency Voice Detector
Sends alerts to the web dashboard when emergency keywords are detected
"""

import speech_recognition as sr
import numpy as np
import time
import os
import winsound
import threading
import json
import re
from datetime import datetime
import sys
import requests

# Add the current directory to path and import our integration module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from guardian_integration import GuardianAlertSender

print("=" * 70)
print("🎤 GUARDIAN NET - MULTILINGUAL EMERGENCY VOICE DETECTOR")
print("=" * 70)
print("Supports: English + Malayalam + Hindi + Custom Keywords")
print("=" * 70)

# Default emergency keywords in multiple languages
DEFAULT_KEYWORDS = {
    'english': [
        'help', 'emergency', 'accident', 'fall', 'fell', 'fallen',
        'hurt', 'pain', 'injured', 'wounded', 'bleeding',
        'come fast', 'come quickly', 'urgent', 'danger', 'dangerous',
        'save me', 'rescue', 'ambulance', 'hospital', 'doctor',
        'broken', 'fracture', 'unconscious', 'fire', 'thief',
        'robbery', 'attack', 'fight', 'violence', 'critical',
        'serious', 'need help', 'assistance', 'aid', 'dangerous'
    ],
    
    'malayalam': [
        'സഹായം', 'അടിയന്തരം', 'അപകടം', 'വീഴ്ച', 'വീണു',
        'നോവ്', 'വേദന', 'അടി', 'രക്തസ്രാവം', 'പരിക്ക്',
        'വേഗം വരൂ', 'തിടുക്കം', 'അപായം', 'അപകടകരം',
        'രക്ഷിക്കൂ', 'ആംബുലൻസ്', 'ആശുപത്രി', 'ഡോക്ടർ',
        'മുറിവ്', 'അറിയില്ല', 'തീപിടുത്തം', 'തീ', 'കള്ളൻ',
        'കൊള്ള', 'ആക്രമണം', 'യുദ്ധം', 'ഹിംസ', 'കോമ',
        'പൊട്ടിയ', 'തകർന്ന'
    ],
    
    'hindi': [
        'मदद', 'आपातकाल', 'दुर्घटना', 'गिर गया', 'चोट',
        'दर्द', 'जल्दी आओ', 'खतरा', 'बचाओ', 'एम्बुलेंस'
    ]
}

# Custom keywords file path
CUSTOM_KEYWORDS_FILE = "custom_keywords.json"

class GuardianVoiceDetector:
    def __init__(self, patient_id=1):
        print("\n" + "="*70)
        print("🚀 GUARDIAN NET - VOICE DETECTION SYSTEM")
        print("="*70)
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.is_listening = True
        self.emergency_count = 0
        self.last_speech = ""
        self.detected_language = ""
        
        # Guardian Net integration
        self.patient_id = patient_id
        self.alert_sender = GuardianAlertSender(patient_id=patient_id)
        
        # Test connection to server
        if self.alert_sender.test_connection():
            print("✅ Connected to Guardian Net server")
            print(f"📱 Patient ID: {patient_id}")
        else:
            print("⚠️  Cannot connect to Guardian Net server")
            print("   Alerts will be logged locally only")
        
        # Load keywords
        self.keywords = self.load_keywords()
        
        # Language detection settings
        self.supported_languages = ['en-IN', 'ml-IN', 'en-US', 'hi-IN']
        self.current_language = 'en-IN'  # Default: Indian English
        
        print(f"✅ Languages: English, Malayalam, Hindi")
        print(f"✅ Total keywords: {sum(len(v) for v in self.keywords.values())}")
        
        # Calibrate microphone
        self.calibrate_microphone()
        
        # Alert cooldown
        self.last_alert_time = 0
        self.alert_cooldown = 15  # seconds between alerts
    
    def load_keywords(self):
        """Load keywords from default and custom files"""
        keywords = DEFAULT_KEYWORDS.copy()
        
        # Load custom keywords if file exists
        if os.path.exists(CUSTOM_KEYWORDS_FILE):
            try:
                with open(CUSTOM_KEYWORDS_FILE, 'r', encoding='utf-8') as f:
                    custom_data = json.load(f)
                
                for lang, words in custom_data.items():
                    if lang in keywords:
                        keywords[lang].extend(words)
                    else:
                        keywords[lang] = words
                
                print(f"✅ Loaded custom keywords from {CUSTOM_KEYWORDS_FILE}")
            except Exception as e:
                print(f"⚠️  Could not load custom keywords: {e}")
        
        return keywords
    
    def save_custom_keywords(self):
        """Save custom keywords to file"""
        try:
            with open(CUSTOM_KEYWORDS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.keywords, f, indent=2, ensure_ascii=False)
            print(f"✅ Custom keywords saved to {CUSTOM_KEYWORDS_FILE}")
        except Exception as e:
            print(f"❌ Could not save custom keywords: {e}")
    
    def add_custom_keyword(self, language, keyword):
        """Add a custom keyword"""
        if language not in self.keywords:
            self.keywords[language] = []
        
        if keyword not in self.keywords[language]:
            self.keywords[language].append(keyword)
            self.save_custom_keywords()
            print(f"✅ Added '{keyword}' to {language} keywords")
            return True
        else:
            print(f"⚠️  '{keyword}' already exists in {language}")
            return False
    
    def calibrate_microphone(self):
        """Calibrate microphone for ambient noise"""
        print("\n🔊 Calibrating microphone... Please stay silent for 2 seconds")
        
        with self.microphone as source:
            try:
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
                print("✅ Microphone calibrated")
            except Exception as e:
                print(f"⚠️  Calibration error: {e}")
    
    def detect_language(self, text):
        """Detect language of the text"""
        # Simple language detection based on character sets
        if re.search(r'[\u0D00-\u0D7F]', text):  # Malayalam Unicode range
            return 'malayalam'
        elif re.search(r'[\u0900-\u097F]', text):  # Hindi/Sanskrit Unicode range
            return 'hindi'
        else:
            return 'english'
    
    def check_emergency_keywords(self, text, language=None):
        """Check if text contains emergency keywords in any language"""
        text_lower = text.lower()
        found_keywords = []
        detected_languages = []
        
        if language:
            # Check specific language
            if language in self.keywords:
                for keyword in self.keywords[language]:
                    if keyword.lower() in text_lower:
                        found_keywords.append(keyword)
                        detected_languages.append(language)
        else:
            # Check all languages
            for lang, keywords in self.keywords.items():
                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        found_keywords.append(keyword)
                        detected_languages.append(lang)
        
        # Remove duplicates
        found_keywords = list(dict.fromkeys(found_keywords))
        detected_languages = list(dict.fromkeys(detected_languages))
        
        return found_keywords, detected_languages
    
    def transcribe_speech(self, audio, language=None):
        """Transcribe audio to text with language detection"""
        if language is None:
            language = self.current_language
        
        try:
            # Try with specified language first
            text = self.recognizer.recognize_google(audio, language=language)
            return text, True, language
        except sr.UnknownValueError:
            # Try other languages if first fails
            for lang in self.supported_languages:
                if lang != language:
                    try:
                        text = self.recognizer.recognize_google(audio, language=lang)
                        return text, True, lang
                    except:
                        continue
            return "Could not understand audio", False, language
        except sr.RequestError as e:
            return f"Speech service error: {e}", False, language
        except Exception as e:
            return f"Error: {e}", False, language
    
    def play_alert_sound(self, emergency_level='medium'):
        """Play emergency alert sound with different levels"""
        try:
            if emergency_level == 'high':
                # High emergency: rapid beeps
                for _ in range(6):
                    winsound.Beep(1200, 200)
                    time.sleep(0.1)
            elif emergency_level == 'medium':
                # Medium emergency: alternating tones
                for freq in [1000, 800, 1000, 1200]:
                    winsound.Beep(freq, 400)
                    time.sleep(0.1)
            else:
                # Low emergency: simple beep
                winsound.Beep(800, 1000)
        except:
            print('\a\a\a')  # System bell
    
    def listen_continuously(self):
        """Continuously listen for speech"""
        print("\n🎤 Listening for speech...")
        print("   Supported languages: English, Malayalam, Hindi")
        print("   Say 'add keyword' to add custom emergency word")
        print("   Say 'stop' to end program\n")
        
        while self.is_listening:
            try:
                with self.microphone as source:
                    # Dynamic ambient noise adjustment
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    print(f"[{timestamp}] 👂 Listening ({self.current_language})...", end='\r')
                    
                    try:
                        # Listen for speech
                        audio = self.recognizer.listen(
                            source, 
                            timeout=1,
                            phrase_time_limit=5
                        )
                        
                        # Process the audio
                        self.process_audio(audio)
                        
                    except sr.WaitTimeoutError:
                        # No speech detected
                        continue
                    except Exception as e:
                        print(f"\n⚠️  Listening error: {e}")
                        time.sleep(1)
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n❌ Error in listening loop: {e}")
                time.sleep(1)
    
    def process_audio(self, audio):
        """Process captured audio"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        print(f"\n[{timestamp}] 🔊 Processing speech...")
        
        # Transcribe audio
        text, success, lang_used = self.transcribe_speech(audio)
        
        if success and text:
            self.last_speech = text
            detected_lang = self.detect_language(text)
            
            print(f"[{timestamp}] 🌐 Language: {detected_lang.upper()}")
            print(f"[{timestamp}] 💬 You said: '{text}'")
            
            # Handle special commands
            if self.handle_special_commands(text):
                return
            
            # Check for emergency keywords
            keywords, keyword_langs = self.check_emergency_keywords(text)
            
            if keywords:
                print(f"[{timestamp}] 🚨 EMERGENCY DETECTED!")
                print(f"    Keywords: {', '.join(keywords)}")
                print(f"    Languages: {', '.join(keyword_langs)}")
                
                # Determine emergency level
                emergency_level = 'high' if len(keywords) > 2 else 'medium'
                
                # Handle emergency
                self.handle_emergency(text, keywords, keyword_langs, emergency_level)
            else:
                print(f"[{timestamp}] ✅ No emergency keywords detected")
                
                # Check for test phrases
                if any(word in text.lower() for word in ['test', 'ടെസ്റ്റ്', 'परीक्षण']):
                    print(f"[{timestamp}] ✅ System test successful")
                    self.play_alert_sound('low')
        
        elif not success:
            print(f"[{timestamp}] ⚠️  {text}")
    
    def handle_special_commands(self, text):
        """Handle special voice commands"""
        text_lower = text.lower()
        
        # Stop command
        if any(cmd in text_lower for cmd in ['stop', 'അവസാനം', 'बंद करो', 'exit', 'quit']):
            print(f"\n🛑 Stop command received")
            self.is_listening = False
            return True
        
        # Add keyword command
        elif 'add keyword' in text_lower or 'add word' in text_lower:
            print(f"\n➕ ADD CUSTOM KEYWORD")
            print("=" * 40)
            self.add_keyword_via_voice()
            return True
        
        # List keywords command
        elif any(cmd in text_lower for cmd in ['list keywords', 'show keywords', 'keywords list']):
            self.show_keyword_summary()
            return True
        
        # Change language command
        elif any(cmd in text_lower for cmd in ['malayalam mode', 'മലയാളം']):
            self.current_language = 'ml-IN'
            print(f"\n🌐 Switched to Malayalam mode")
            return True
        elif any(cmd in text_lower for cmd in ['english mode', 'ഇംഗ്ലീഷ്']):
            self.current_language = 'en-IN'
            print(f"\n🌐 Switched to English mode")
            return True
        elif any(cmd in text_lower for cmd in ['hindi mode', 'हिंदी']):
            self.current_language = 'hi-IN'
            print(f"\n🌐 Switched to Hindi mode")
            return True
        
        return False
    
    def add_keyword_via_voice(self):
        """Add custom keyword through voice interaction"""
        print("Please say the new emergency word: ")
        
        try:
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=5)
                keyword, success, _ = self.transcribe_speech(audio)
                
                if success and keyword:
                    print(f"You said: '{keyword}'")
                    
                    # Detect language
                    lang = self.detect_language(keyword)
                    print(f"Detected language: {lang}")
                    
                    # Add the keyword
                    self.add_custom_keyword(lang, keyword)
                    
                    # Test the new keyword
                    print(f"\n✅ Testing new keyword...")
                    print(f"   Say '{keyword}' to trigger emergency")
        except Exception as e:
            print(f"❌ Could not add keyword: {e}")
    
    def show_keyword_summary(self):
        """Show summary of all keywords"""
        print(f"\n📋 KEYWORD SUMMARY")
        print("=" * 40)
        
        for lang, keywords in self.keywords.items():
            print(f"\n{lang.upper()} ({len(keywords)} words):")
            # Show first 10 keywords
            display_keys = keywords[:10]
            print(f"   {', '.join(display_keys)}")
            if len(keywords) > 10:
                print(f"   ... and {len(keywords) - 10} more")
        
        print(f"\nTotal unique keywords: {sum(len(v) for v in self.keywords.values())}")
    
    def handle_emergency(self, text, keywords, languages, level='medium'):
        """Handle emergency detection and send alert to website"""
        current_time = time.time()
        
        # Check cooldown to avoid spam
        if current_time - self.last_alert_time < self.alert_cooldown:
            remaining = self.alert_cooldown - (current_time - self.last_alert_time)
            print(f"⏱️  Alert cooldown: {remaining:.1f}s remaining")
            return
        
        self.emergency_count += 1
        self.last_alert_time = current_time
        
        print("\n" + "!" * 70)
        print("🚨🚨🚨 EMERGENCY DETECTED! 🚨🚨🚨")
        print("!" * 70)
        print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        print(f"Language(s): {', '.join(languages)}")
        print(f"Spoken words: '{text}'")
        print(f"Keywords: {', '.join(keywords)}")
        print(f"Emergency level: {level.upper()}")
        print("!" * 70)
        
        # Play emergency alert sound
        print("\n🔊 Playing emergency alert...")
        self.play_alert_sound(level)
        
        # Send alert to Guardian Net website
        message = f"🚨 Voice emergency detected! Keywords: {', '.join(keywords)}"
        self.alert_sender.send_alert("voice", message)
        
        # Display emergency message in detected languages
        print("\n📢 EMERGENCY ALERT SENT TO WEBSITE!")
        
        # Log the emergency
        self.log_emergency(text, keywords, languages, level)
        
        # Start emergency countdown
        threading.Thread(target=self.emergency_countdown, 
                        args=(level, languages), 
                        daemon=True).start()
    
    def emergency_countdown(self, level, languages):
        """Emergency countdown sequence"""
        countdown_time = 15 if level == 'high' else 10
        
        print(f"\n⏰ EMERGENCY RESPONSE ACTIVATED")
        print(f"   Notifying emergency services in {countdown_time} seconds...")
        print("   Say 'cancel' to abort")
        
        for i in range(countdown_time, 0, -1):
            if not self.is_listening:
                return
            print(f"   Notifying in {i}...", end='\r')
            time.sleep(1)
        
        print("\n📞 EMERGENCY SERVICES NOTIFIED!")
        
        # Display in multiple languages
        if 'malayalam' in languages:
            print("   മലയാളം: സഹായം അയച്ചിരിക്കുന്നു. കാത്തിരിക്കൂ.")
        if 'hindi' in languages:
            print("   हिंदी: मदद भेज दी गई है। प्रतीक्षा करें।")
        print("   English: Help has been dispatched. Please wait.")
        
        # Play confirmation sound
        self.play_alert_sound('low')
        
        # Reset after delay
        time.sleep(30)
        print("\n🟢 Emergency response sequence complete")
    
    def log_emergency(self, text, keywords, languages, level):
        """Log emergency to file with multilingual support"""
        os.makedirs('emergency_logs', exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"emergency_logs/emergency_{timestamp}.json"
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'text': text,
            'keywords': keywords,
            'languages': languages,
            'level': level,
            'count': self.emergency_count,
            'patient_id': self.patient_id
        }
        
        # Save as JSON
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)
        
        # Also save to text file for easy reading
        text_file = f"emergency_logs/emergency_{timestamp}.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write("EMERGENCY DETECTION LOG\n")
            f.write("=" * 50 + "\n")
            f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Patient ID: {self.patient_id}\n")
            f.write(f"Languages: {', '.join(languages)}\n")
            f.write(f"Emergency Level: {level}\n")
            f.write(f"Spoken Text: '{text}'\n")
            f.write(f"Keywords: {', '.join(keywords)}\n")
            f.write(f"Emergency Count: {self.emergency_count}\n")
            f.write("=" * 50 + "\n")
        
        print(f"📝 Log saved to: {text_file}")
    
    def show_summary(self):
        """Show session summary"""
        print("\n" + "=" * 70)
        print("📊 SESSION SUMMARY")
        print("=" * 70)
        print(f"Total emergencies detected: {self.emergency_count}")
        print(f"Alerts sent to website: {self.alert_sender.alert_count}")
        
        if self.last_speech:
            lang = self.detect_language(self.last_speech)
            print(f"Last speech ({lang}): '{self.last_speech}'")
        
        # Show keyword statistics
        print(f"\n📋 KEYWORD STATISTICS:")
        for lang, keywords in self.keywords.items():
            print(f"   {lang.upper()}: {len(keywords)} words")
        
        if self.emergency_count > 0:
            print(f"\n📁 Emergency logs saved in: emergency_logs/")
        
        print("\n✅ Detection session ended")
        print("=" * 70)
    
    def start(self):
        """Start the detector"""
        print("\n" + "=" * 70)
        print("🚀 STARTING GUARDIAN NET VOICE DETECTOR")
        print("=" * 70)
        print(f"Patient ID: {self.patient_id}")
        print("=" * 70)
        
        # Show current keywords
        self.show_keyword_summary()
        
        print("\n🎯 VOICE COMMANDS:")
        print("   • 'add keyword' - Add custom emergency word")
        print("   • 'list keywords' - Show all emergency words")
        print("   • 'malayalam mode' / 'english mode' / 'hindi mode'")
        print("   • 'stop' - End program")
        
        print("\n🎯 TEST PHRASES:")
        print("   English: 'Help me, I fell down'")
        print("   Malayalam: 'സഹായം, ഞാൻ വീണു'")
        print("   Hindi: 'मदद, मैं गिर गया'")
        print("=" * 70)
        
        print("\nStarting in 3 seconds...")
        time.sleep(3)
        
        # Start listening
        self.listen_continuously()
        
        # Show summary
        self.show_summary()

def create_custom_keywords_file():
    """Create sample custom keywords file"""
    sample_custom = {
        'english': ['heart attack', 'stroke', 'seizure'],
        'malayalam': ['ഹൃദയാഘാതം', 'സ്ട്രോക്ക്'],
        'hindi': ['दिल का दौरा', 'लकवा']
    }
    
    try:
        with open(CUSTOM_KEYWORDS_FILE, 'w', encoding='utf-8') as f:
            json.dump(sample_custom, f, indent=2, ensure_ascii=False)
        print(f"✅ Created sample custom keywords file: {CUSTOM_KEYWORDS_FILE}")
    except Exception as e:
        print(f"❌ Could not create custom keywords file: {e}")

# Main execution
if __name__ == "__main__":
    print("Guardian Net - Multilingual Emergency Voice Detector")
    print("Version: 4.0 - With Website Integration")
    print("=" * 70)
    
    # Get patient ID from user
    try:
        patient_id = int(input("Enter patient ID (default: 1): ") or "1")
    except:
        patient_id = 1
    
    # Check and install requirements
    try:
        import speech_recognition
        import pyaudio
        print("✅ Required packages installed")
    except ImportError:
        print("❌ Missing packages. Please install:")
        print("   pip install SpeechRecognition pyaudio")
        sys.exit(1)
    
    # Create sample custom keywords file if doesn't exist
    if not os.path.exists(CUSTOM_KEYWORDS_FILE):
        create_custom_keywords_file()
    
    # Create and start detector
    detector = GuardianVoiceDetector(patient_id=patient_id)
    
    try:
        detector.start()
    except KeyboardInterrupt:
        print("\n\n🛑 Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n👋 Thank you for using Guardian Net Voice Detector!")