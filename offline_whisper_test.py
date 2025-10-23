import sounddevice as sd
import numpy as np
import time
import threading
from faster_whisper import WhisperModel

# =====================================================
# ⚙️ Configuration (simple, robust)
# =====================================================
SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.006    # increase if it never shows dots; decrease if it never shows speakers
SILENCE_MS = 1200
MAX_RECORD_TIME = 30
CHUNK_MS = 150

# =====================================================
# 🧠 Load Whisper model
# =====================================================
print("🎧 Loading Whisper model (base.en int8)…")
model = WhisperModel("base.en", compute_type="int8", num_workers=2, device="auto")
print("✅ Model ready! (Press ENTER anytime to stop)")

# =====================================================
# 🎙️ Record with ENTER to stop (no calibration, default mic)
# =====================================================
stop_flag = False

def wait_for_enter():
    global stop_flag
    try:
        input()  # waits for ENTER
        stop_flag = True
    except EOFError:
        stop_flag = True

def record_until_enter():
    global stop_flag
    stop_flag = False
    threading.Thread(target=wait_for_enter, daemon=True).start()

    print("🎙️ Speak now… (Press ENTER to stop)")
    buf = []
    chunk = int(CHUNK_MS / 1000 * SAMPLE_RATE)
    silence_chunks = int(SILENCE_MS / CHUNK_MS)
    silent = 0
    started_talking = False

    # No device arg -> use system default input
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
        start = time.time()
        while not stop_flag:
            try:
                data, _ = stream.read(chunk)
            except Exception as e:
                print(f"\n❌ Audio read error: {e}")
                break

            buf.append(data.copy())

            # Simple RMS-based speaking detector
            rms = float(np.sqrt((data**2).mean()))
            speaking = rms > SILENCE_THRESHOLD

            if speaking:
                started_talking = True
                silent = 0
                print("🔊", end="", flush=True)
            else:
                if started_talking:
                    silent += 1
                    print(".", end="", flush=True)

            # Auto-stop on silence after you started speaking
            if started_talking and silent >= silence_chunks:
                print("\n🛑 Silence detected, stopping…")
                break

            # Safety stop
            if (time.time() - start) > MAX_RECORD_TIME:
                print("\n⏰ Max recording time reached.")
                break

    if buf:
        audio = np.concatenate(buf, axis=0).squeeze()
        return audio.astype("float32")
    else:
        return np.zeros(0, dtype="float32")

# =====================================================
# 🧠 Transcribe
# =====================================================
def transcribe_audio(audio):
    if audio.size == 0:
        return ""
    print("\n🧠 Transcribing…")
    segments, _ = model.transcribe(
        audio,
        language="en",
        vad_filter=True,
        beam_size=1,
        condition_on_previous_text=False
    )
    return "".join(seg.text for seg in segments).strip()

# =====================================================
# ▶️ Main
# =====================================================
if __name__ == "__main__":
    try:
        audio = record_until_enter()
        text = transcribe_audio(audio)
        print("\n🗣️ You said:", text or "(no speech detected)")
    except KeyboardInterrupt:
        # Graceful: attempt to transcribe whatever we captured
        try:
            if 'buf' in globals() and len(buf) > 0:
                audio = np.concatenate(buf, axis=0).squeeze().astype("float32")
                text = transcribe_audio(audio)
                print("\n🗣️ You said:", text or "(no speech detected)")
            else:
                print("\n🛑 Interrupted by user before any audio was captured.")
        except Exception:
            print("\n🛑 Interrupted by user.")