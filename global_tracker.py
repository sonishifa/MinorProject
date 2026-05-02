import time
import requests
import threading
from pynput import keyboard
import tkinter as tk

# ── Global State ──
tracking_enabled = False
keystroke_buffer = []
keydown_times = {}

def send_to_backend():
    global keystroke_buffer
    if not keystroke_buffer: return
    
    # Safely duplicate and clear buffer so we don't drop keys while sending
    payload = keystroke_buffer[:]
    keystroke_buffer = []
    
    try:
        requests.post("http://localhost:8000/api/telemetry/keystrokes", json=payload)
        print(f"Sent {len(payload)} keystrokes to sync baseline.")
    except Exception as e:
        print(f"Backend offline, could not sync: {e}")

# ── Pynput Keyboard Listeners ──
def on_press(key):
    if not tracking_enabled: return
    try:
        k = key.char
    except AttributeError:
        k = str(key)
    
    if k not in keydown_times:
        keydown_times[k] = time.time() * 1000

def on_release(key):
    if not tracking_enabled: return
    try:
        k = key.char
    except AttributeError:
        k = str(key)
        
    press_time = keydown_times.pop(k, None)
    if press_time:
        release_time = time.time() * 1000
        keystroke_buffer.append({
            "key": k,
            "code": k,
            "press_time": press_time,
            "release_time": release_time
        })

# Background loop to send data every 2 seconds if typing stopped
def buffer_sender():
    while True:
        time.sleep(2.0)
        if tracking_enabled and len(keystroke_buffer) > 0:
            send_to_backend()

# ── Tkinter UI ──
def toggle_tracking():
    global tracking_enabled
    tracking_enabled = not tracking_enabled
    if tracking_enabled:
        btn.config(text="Global Tracking: ON", bg="#27ae60", fg="white")
        print("Global tracking enabled. Syncing keystrokes...")
    else:
        btn.config(text="Global Tracking: OFF", bg="#e74c3c", fg="white")
        print("Global tracking disabled. Keyboard ignored.")

# 1. Start background threads
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

sender_thread = threading.Thread(target=buffer_sender, daemon=True)
sender_thread.start()

# 2. Build the Tiny Floating UI
root = tk.Tk()
root.title("BFMT Sync")
root.geometry("250x80")
root.attributes('-topmost', True) # Keep it floating above Notepad/Chrome

btn = tk.Button(
    root, 
    text="Global Tracking: OFF", 
    command=toggle_tracking, 
    bg="#e74c3c", 
    fg="white", 
    font=("Arial", 14, "bold")
)
btn.pack(expand=True, fill="both")

print("Tracker started! Turn the toggle ON to sync keystrokes across all apps.")
root.mainloop()
