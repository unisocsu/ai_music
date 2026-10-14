import os
import time

import numpy as np
import torch
from scipy.io.wavfile import write
from transformers import AutoProcessor, MusicgenForConditionalGeneration

prompt = os.environ["PROMPT_EN"].strip()
duration = int(os.environ.get("DURATION", "4"))
if not 1 <= duration <= 10:
    raise SystemExit("Duration must be between 1 and 10 seconds.")

torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))
device = "cpu"
start = time.time()

# facebook/musicgen-small is a public model; it does not require HF_TOKEN.
model_name = "facebook/musicgen-small"
print(f"Loading public MusicGen Small (first run downloads model weights)...")
processor = AutoProcessor.from_pretrained(model_name)
model = MusicgenForConditionalGeneration.from_pretrained(model_name).to(device).eval()
inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)

# MusicGen produces about 50 audio tokens per second. CPU generation can take a while.
max_new_tokens = duration * 50
print(f"Model loaded in {time.time() - start:.1f}s; generating {duration}s on CPU.")
with torch.inference_mode():
    audio_values = model.generate(
        **inputs,
        do_sample=True,
        guidance_scale=3.0,
        max_new_tokens=max_new_tokens,
    )

sample_rate = int(model.config.audio_encoder.sampling_rate)
audio = audio_values[0].detach().float().cpu().numpy()
if audio.ndim == 2:
    audio = audio.T
audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
os.makedirs("output", exist_ok=True)
write("output/music.wav", sample_rate, audio)
print(f"Saved output/music.wav ({time.time() - start:.1f}s total).")
