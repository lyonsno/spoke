import wave
import numpy as np
import os
import io

SAMPLE_RATE = 16000
BLOCKSIZE = 1024  # 64ms at 16kHz

# VAD parameters
MIN_SPEECH_FRAMES = 3      # ~192ms of consecutive speech to trigger
MIN_SILENCE_FRAMES = 12    # ~768ms of consecutive silence to slice
PRE_SPEECH_MARGIN = 6      # ~384ms padding before speech
NOISE_FLOOR_WINDOW = 50    # ~3.2s of history for noise floor
THRESHOLD_MULTIPLIER = 2.5
MIN_THRESHOLD = 0.001

def encode_wav(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    if samples.size == 0:
        return b""
    pcm = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()

def test_vad(wav_file):
    with wave.open(wav_file, 'rb') as wf:
        n_frames = wf.getnframes()
        audio_bytes = wf.readframes(n_frames)
        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    print(f"Loaded {len(samples)/SAMPLE_RATE:.2f}s of audio")

    chunks = []
    for i in range(0, len(samples), BLOCKSIZE):
        chunk = samples[i:i+BLOCKSIZE]
        if len(chunk) < BLOCKSIZE:
            break
        chunks.append(chunk)

    is_speech = False
    speech_trigger_count = 0
    silence_trigger_count = 0
    noise_floor_history = []
    
    current_segment_chunks = []
    segments = []
    ring_buffer = []

    out_dir = "bench/segments"
    os.makedirs(out_dir, exist_ok=True)

    for idx, chunk in enumerate(chunks):
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        
        noise_floor_history.append(rms)
        if len(noise_floor_history) > NOISE_FLOOR_WINDOW:
            noise_floor_history.pop(0)
            
        noise_floor = min(noise_floor_history) if noise_floor_history else 0.0
        threshold = max(noise_floor * THRESHOLD_MULTIPLIER, MIN_THRESHOLD)

        time_sec = idx * BLOCKSIZE / SAMPLE_RATE

        if not is_speech:
            ring_buffer.append(chunk)
            if len(ring_buffer) > PRE_SPEECH_MARGIN:
                ring_buffer.pop(0)

            if rms > threshold:
                speech_trigger_count += 1
                if speech_trigger_count >= MIN_SPEECH_FRAMES:
                    is_speech = True
                    current_segment_chunks.extend(ring_buffer)
                    ring_buffer = []
                    print(f"[{time_sec:.2f}s] -> SPEECH START (RMS: {rms:.4f}, Thr: {threshold:.4f}, NF: {noise_floor:.4f})")
            else:
                speech_trigger_count = 0
        else:
            current_segment_chunks.append(chunk)
            if rms <= threshold:
                silence_trigger_count += 1
                if silence_trigger_count >= MIN_SILENCE_FRAMES:
                    is_speech = False
                    speech_trigger_count = 0
                    silence_trigger_count = 0
                    
                    segment_samples = np.concatenate(current_segment_chunks)
                    seg_duration = len(segment_samples) / SAMPLE_RATE
                    print(f"[{time_sec:.2f}s] -> SILENCE. Slice duration: {seg_duration:.2f}s")
                    
                    seg_idx = len(segments) + 1
                    with open(f"{out_dir}/segment_{seg_idx:03d}.wav", "wb") as f:
                        f.write(encode_wav(segment_samples))
                    
                    segments.append(segment_samples)
                    current_segment_chunks = []
            else:
                silence_trigger_count = 0

    if current_segment_chunks:
        segment_samples = np.concatenate(current_segment_chunks)
        seg_duration = len(segment_samples) / SAMPLE_RATE
        print(f"[{len(chunks) * BLOCKSIZE / SAMPLE_RATE:.2f}s] -> END OF AUDIO. Slice duration: {seg_duration:.2f}s")
        seg_idx = len(segments) + 1
        with open(f"{out_dir}/segment_{seg_idx:03d}.wav", "wb") as f:
            f.write(encode_wav(segment_samples))

if __name__ == "__main__":
    test_vad("bench/speech_16k.wav")
