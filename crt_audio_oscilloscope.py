import os
import json
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import queue
from matplotlib.widgets import Slider

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
CONFIG_FILE = "crt_audio_oscilloscope_config.json"

# 🔄 Load and Save Configuration
def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
            return {
                "device_index": config.get("device_index", 0),
                "gain": float(config.get("gain", 1.0)),
                "smoothing": float(config.get("smoothing", 0.8))
            }
        except Exception as e:
            print(f"Error loading config: {e}")
            os.remove(CONFIG_FILE)  # Auto-delete corrupt config file
            print("⚠️ Corrupt config file removed. Resetting settings to default.")
    return {}

def save_config(config):
    """Save only serializable settings"""
    try:
        json_safe_config = {
            "device_index": config["device_index"],
            "gain": float(config["gain"]),
            "smoothing": float(config["smoothing"])
        }
        with open(CONFIG_FILE, "w") as f:
            json.dump(json_safe_config, f)
        print("✅ Settings saved.")
    except Exception as e:
        print(f"Error saving config: {e}")

# 🎤 Get Microphone Device
def get_device(saved_device=None):
    valid_devices = [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]
    if not valid_devices:
        raise RuntimeError("No valid microphone devices found.")

    if saved_device is not None and any(saved_device == dev[0] for dev in valid_devices):
        print(f"Using saved microphone: {saved_device}")
        return saved_device

    print("\nAvailable microphones:")
    for idx, name in valid_devices:
        print(f"  {idx}: {name}")

    while True:
        try:
            selected = int(input(f"\nSelect microphone index (default {valid_devices[0][0]}): ") or valid_devices[0][0])
            if any(selected == dev[0] for dev in valid_devices):
                return selected
        except ValueError:
            print("Invalid input. Try again.")

##############################
# 🎛 CRT-Style Oscilloscope with Stream Fix
##############################
class CRT_Oscilloscope:
    def __init__(self, block_size=2048):
        self.block_size = block_size
        self.config = load_config()

        # Load saved microphone device
        self.device = get_device(self.config.get("device_index"))

        try:
            info = sd.query_devices(self.device, "input")
            self.fs = info["default_samplerate"]
        except Exception as e:
            print(f"Error getting device info: {e}")
            self.fs = 44100  # Default sample rate

        # 🔧 Load Previous Settings or Default
        self.gain = self.config.get("gain", 1.0)
        self.smoothing = self.config.get("smoothing", 0.8)
        self.audio_queue = queue.Queue(maxsize=10)
        self.fullscreen = False  # Fullscreen toggle state (not saved)

        # 🔄 Setup UI
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.fig.patch.set_facecolor("black")
        self.ax.set_facecolor("black")
        self.fig.canvas.manager.set_window_title("CRT Audio Oscilloscope")

        plt.subplots_adjust(bottom=0.3)

        # 🟢 Display Waveform
        self.x_data = np.linspace(0, 1, self.block_size)
        self.line, = self.ax.plot(self.x_data, np.zeros(self.block_size), color="lime", lw=2, alpha=0.8)

        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_xlabel("Time (scaled)", color="white")
        self.ax.set_ylabel("Amplitude (dB SPL)", color="white")
        self.ax.set_title("CRT Audio Oscilloscope", color="white")
        self.ax.tick_params(axis="x", colors="white")
        self.ax.tick_params(axis="y", colors="white")

        # 🛠 Add Grid Lines
        self.ax.grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.5)

        # 🎛 UI Controls
        self.fig.text(0.15, 0.18, "Gain", color="white", fontsize=12)
        self.fig.text(0.15, 0.11, "Smoothing", color="white", fontsize=12)

        # Sliders
        self.slider_gain_ax = plt.axes([0.15, 0.15, 0.5, 0.03], facecolor="black")
        self.slider_smooth_ax = plt.axes([0.15, 0.08, 0.5, 0.03], facecolor="black")

        self.slider_gain = Slider(self.slider_gain_ax, "", 0.5, 50.0, valinit=self.gain, color="lime")
        self.slider_smooth = Slider(self.slider_smooth_ax, "", 0.0, 0.99, valinit=self.smoothing, color="lime")

        self.slider_gain.on_changed(self.update_gain)
        self.slider_smooth.on_changed(self.update_smoothing)

        # 🎤 Loudness Display
        self.text_loudness = self.ax.text(1.05, 0.85, "Loudness: 0 dB SPL", color="white", transform=self.fig.transFigure, fontsize=12)

    def update_gain(self, val):
        self.gain = self.slider_gain.val
        self.save_state()

    def update_smoothing(self, val):
        self.smoothing = self.slider_smooth.val
        self.save_state()

    def save_state(self):
        """Save UI settings so they persist on restart."""
        save_config({
            "device_index": self.device,
            "gain": float(self.gain),
            "smoothing": float(self.smoothing)
        })

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print("Status:", status)
        if len(indata) == 0:
            print("⚠️ No audio data received.")
            return

        data = indata[:, 0]
        data = np.clip(data * self.gain, -1, 1)

        try:
            self.audio_queue.put_nowait(data)
        except queue.Full:
            pass

    def run(self):
        print("🎤 Starting CRT Audio Oscilloscope - Speak or make sound to see movement.")

        try:
            stream = sd.InputStream(
                samplerate=self.fs, 
                channels=1, 
                blocksize=self.block_size,
                device=self.device, 
                callback=self.audio_callback
            )
            with stream:
                plt.show(block=False)
                while plt.fignum_exists(self.fig.number):
                    try:
                        new_waveform = self.audio_queue.get(timeout=0.02)
                        self.line.set_ydata(new_waveform)
                        self.fig.canvas.draw_idle()
                        self.fig.canvas.flush_events()
                    except queue.Empty:
                        pass
                    plt.pause(0.01)
        except Exception as e:
            print(f"❌ Error starting audio stream: {e}")
            print("Try selecting a different microphone.")

        save_config({
            "device_index": self.device,
            "gain": float(self.gain),
            "smoothing": float(self.smoothing)
        })

if __name__ == "__main__":
    CRT_Oscilloscope().run()
