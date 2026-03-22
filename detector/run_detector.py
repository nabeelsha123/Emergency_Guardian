#!/usr/bin/env python
"""
Guardian Net - Unified Detector Launcher
Run this to start all detection systems
"""

import subprocess
import sys
import os
import time
import threading

def run_fall_detector():
    """Run the fall detection script"""
    print("\n🔴 Starting Fall Detection...")
    subprocess.run([sys.executable, "guardian_fall.py"])

def run_gesture_detector():
    """Run the gesture detection script"""
    print("\n🟢 Starting Gesture Detection...")
    subprocess.run([sys.executable, "guardian_gesture.py"])

def main():
    print("\n" + "="*70)
    print("🚀 GUARDIAN NET - DETECTION SYSTEMS LAUNCHER")
    print("="*70)
    print("\nAvailable options:")
    print("  1. Run Fall Detection only")
    print("  2. Run Gesture Detection only")
    print("  3. Run Both (in separate windows)")
    print("  4. Exit")
    print("-"*50)
    
    choice = input("Enter your choice (1-4): ").strip()
    
    if choice == "1":
        run_fall_detector()
    elif choice == "2":
        run_gesture_detector()
    elif choice == "3":
        print("\n⚠️  Running both detectors in same window may cause conflicts")
        print("   It's better to run them in separate terminals")
        print("\n   Option A: Open new terminals and run:")
        print("     python guardian_fall.py")
        print("     python guardian_gesture.py")
        print("\n   Option B: Run them sequentially (press Ctrl+C to switch)")
        print("-"*50)
        
        subchoice = input("Run fall first? (y/n): ").strip().lower()
        if subchoice == 'y':
            run_fall_detector()
            print("\n" + "="*50)
            run_gesture_detector()
        else:
            run_gesture_detector()
            print("\n" + "="*50)
            run_fall_detector()
    else:
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()